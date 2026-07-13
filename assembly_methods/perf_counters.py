"""Process-local performance counters for the SIEVE pipeline.

Each worker process (or the main process in single-threaded mode) owns one
``PerfCounters`` instance accessed via ``get_counters()``.  Callers increment
the counters inline; at the end of a row the pipeline snapshot a delta and
attaches it to the selection result so the parent process can aggregate without
shared-memory coordination.

Usage pattern
-------------
    from assembly_methods.perf_counters import get_counters, snapshot_delta

    # In an instrumented function:
    get_counters().spacy_parses_fresh += 1

    # At the boundary of a single pipeline row:
    before = get_counters().snapshot()
    ... run the row ...
    row_stats = get_counters().delta_since(before)

    # In the parent, sum across all row dicts:
    agg = aggregate_row_stats(row_stats_list)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, fields
from typing import Any


# ---------------------------------------------------------------------------
# Counter dataclass
# ---------------------------------------------------------------------------

@dataclass
class _Counters:
    """All mutable counters.  Arithmetic fields only — no locks needed because
    each process has its own instance."""

    # ---- spaCy ---------------------------------------------------------------
    spacy_parses_fresh: int = 0       # actual nlp(text) calls (cache miss)
    spacy_parses_cache_hit: int = 0   # lookups that returned a cached Doc

    # ---- memory objects ------------------------------------------------------
    memory_views_built: int = 0       # build_memory_objects() calls
    memory_objects_created: int = 0   # individual MemoryObject instances
    sentences_processed: int = 0      # sentences passed through _objects_from_sentence

    # ---- evidence units ------------------------------------------------------
    evidence_unit_build_calls: int = 0   # build_evidence_units_for_views() calls
    evidence_units_created: int = 0       # EvidenceUnit instances

    # ---- double-build savings ------------------------------------------------
    objects_reused_from_store: int = 0    # objects skipped due to prebuilt_objects

    # ---- compilation ---------------------------------------------------------
    compile_evidence_calls: int = 0
    expansion_view_parses: int = 0        # views parsed during expansion fallback

    # ---- coarse wall-clock (seconds, float) ----------------------------------
    wall_object_building: float = 0.0
    wall_proposal_retrieval: float = 0.0
    wall_compilation: float = 0.0

    def snapshot(self) -> "_Counters":
        """Return a shallow copy of current values (used to compute deltas)."""
        c = _Counters()
        for f in fields(self):
            setattr(c, f.name, getattr(self, f.name))
        return c

    def delta_since(self, before: "_Counters") -> dict[str, Any]:
        """Return a plain dict of increments since ``before`` was snapshotted."""
        result: dict[str, Any] = {}
        for f in fields(self):
            val = getattr(self, f.name) - getattr(before, f.name)
            if val:   # omit zero deltas to keep row dicts compact
                result[f.name] = round(val, 6) if isinstance(val, float) else val
        return result

    def add_delta(self, delta: dict[str, Any]) -> None:
        """Merge a delta dict back in (used by the parent to aggregate)."""
        for key, val in delta.items():
            current = getattr(self, key, None)
            if current is not None:
                setattr(self, key, current + val)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# One instance per process — never shared across processes.
_process_counters = _Counters()


def get_counters() -> _Counters:
    """Return the process-local counter instance."""
    return _process_counters


def reset_counters() -> None:
    """Zero all counters (call at the start of a fresh profiling run)."""
    global _process_counters
    _process_counters = _Counters()


# ---------------------------------------------------------------------------
# Aggregation helpers (used in the parent/main process)
# ---------------------------------------------------------------------------

def aggregate_row_stats(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum a list of per-row delta dicts returned by ``delta_since``."""
    agg = _Counters()
    for delta in stats_list:
        agg.add_delta(delta)
    return agg.to_dict()
