"""Interrogator v0 conversion: build the four evidence conditions per BEAM row.

Deterministic evidence assembly only (no network here). Reuses the frozen
interrogator modules byte-for-byte. The reader driver lives in
``cli/interrogator_conversion_run.py``; this module is pure and testable.
See research/interrogator-v0-conversion-2026-07.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from analysis.interrogator_beam_store import BEAM_PARQUET, build_true_stores
from analysis.interrogator_bm25 import BM25Index
from analysis.interrogator_probes import generate_probes
from analysis.interrogator_reachability import (
    ADJACENCY_RADIUS,
    BEAM_SLICE,
    RETRIEVAL_BOUND_ABILITIES,
    _turn_index,
)

# Frozen experiment constants (see the pre-registered writeup).
BUDGET_UNITS = 30  # B: candidate acquisition units per condition
N_PROBES = 5  # matched to the B=30 budget spec in interrogator_reachability
TOP_M = 6
EVIDENCE_TOKEN_CAP = 24_000  # E: matched total-evidence-token budget (fits 128K ctx)
CHARS_PER_TOKEN = 4  # char/4 token proxy, consistent across conditions

CONDITION_FIXED = "fixed"
CONDITION_ADAPTIVE_K = "adaptive_k"
CONDITION_INTERROGATOR = "interrogator"
CONDITION_CEILING = "ceiling"
CONDITIONS = (
    CONDITION_FIXED,
    CONDITION_ADAPTIVE_K,
    CONDITION_INTERROGATOR,
    CONDITION_CEILING,
)


@dataclass(frozen=True)
class RowCohort:
    """Cohort membership for one evaluation row."""

    example_id: str
    ability: str
    is_missing_gold: bool


@dataclass(frozen=True)
class ConditionEvidence:
    """Assembled evidence for one (row, condition) pair."""

    example_id: str
    condition: str
    evidence_text: str
    evidence_token_estimate: int
    acquired_ids: tuple[str, ...]
    n_acquired_survived: int
    source_memory_ids: tuple[str, ...]


def load_rb_rows(slice_path: Path = BEAM_SLICE) -> list[dict]:
    """Retrieval-bound rows only, in slice order."""
    with slice_path.open() as fh:
        rows = [json.loads(line) for line in fh]
    return [r for r in rows if r["ability"] in RETRIEVAL_BOUND_ABILITIES]


def _missing_gold(row: dict) -> set[str]:
    gold = set(row["evidence_sufficiency"].get("answer_bearing_memory_ids") or [])
    pool = {c["memory_id"] for c in row["candidate_memories"]}
    return gold - pool


def cohort_of(row: dict) -> RowCohort:
    return RowCohort(
        example_id=row["example_id"],
        ability=row["ability"],
        is_missing_gold=bool(_missing_gold(row)),
    )


def _adjacency_ids(pool_ids: frozenset[str], store: dict[str, str], budget: int) -> list[str]:
    """Turns adjacent to in-pool turns (BEAM order signal), byte-identical to the harness."""
    pool_indices = {i for i in (_turn_index(m) for m in pool_ids) if i is not None}
    wanted: list[str] = []
    for idx in sorted(pool_indices):
        for delta in range(-ADJACENCY_RADIUS, ADJACENCY_RADIUS + 1):
            if delta == 0:
                continue
            cand = f"turn_{idx + delta:04d}"
            if cand in store and cand not in pool_ids and cand not in wanted:
                wanted.append(cand)
    return wanted[:budget]


def _k_acquired(index: BM25Index, query: str, pool_ids: frozenset[str], budget: int) -> list[str]:
    ranked = index.rank(query, exclude=pool_ids)
    return [doc_id for doc_id, score in ranked[:budget] if score > 0]


def acquire(row: dict, store: dict[str, str], condition: str) -> list[str]:
    """Ordered acquisition unit ids for a condition (before the token cap)."""
    pool_ids = frozenset(c["memory_id"] for c in row["candidate_memories"])
    if condition == CONDITION_FIXED:
        return []
    if condition == CONDITION_CEILING:
        return sorted(_missing_gold(row))
    index = BM25Index.build(list(store.items()))
    if condition == CONDITION_ADAPTIVE_K:
        return _k_acquired(index, row["query"], pool_ids, BUDGET_UNITS)
    if condition == CONDITION_INTERROGATOR:
        pool_texts = [c["content"] for c in row["candidate_memories"]]
        probes = generate_probes(row["query"], pool_texts, N_PROBES)
        acquired = _probe_only(index, probes, pool_ids, BUDGET_UNITS)
        for doc_id in _adjacency_ids(pool_ids, store, BUDGET_UNITS):
            if doc_id not in acquired and doc_id not in pool_ids and len(acquired) < BUDGET_UNITS:
                acquired.append(doc_id)
        return acquired
    raise ValueError(f"unknown condition: {condition}")


def _probe_only(index: BM25Index, probes: list, pool_ids: frozenset[str], budget: int) -> list[str]:
    acquired: list[str] = []
    exclude = set(pool_ids)
    for probe in probes:
        if len(acquired) >= budget:
            break
        for doc_id, score in index.rank(probe.text, exclude=frozenset(exclude))[:TOP_M]:
            if score <= 0 or doc_id in exclude:
                continue
            acquired.append(doc_id)
            exclude.add(doc_id)
            if len(acquired) >= budget:
                break
    return acquired


def _token_estimate(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def assemble_evidence(row: dict, store: dict[str, str], condition: str) -> ConditionEvidence:
    """Acquired units first, then pool by BM25 rank, whole-unit until E, then truncate."""
    acquired = acquire(row, store, condition)
    pool = sorted(row["candidate_memories"], key=lambda c: c.get("bm25_rank", 1_000_000))
    pool_ids = [c["memory_id"] for c in pool]
    id_to_text = {c["memory_id"]: c["content"] for c in pool}
    for mid in acquired:
        id_to_text.setdefault(mid, store.get(mid, ""))

    ordered_ids = list(acquired) + [pid for pid in pool_ids if pid not in set(acquired)]
    parts: list[str] = []
    used_ids: list[str] = []
    n_acquired_survived = 0
    acquired_set = set(acquired)
    budget = EVIDENCE_TOKEN_CAP
    for mid in ordered_ids:
        text = id_to_text.get(mid, "")
        if not text:
            continue
        cost = _token_estimate(text)
        if budget <= 0:
            break
        if cost > budget:
            keep_chars = budget * CHARS_PER_TOKEN
            text = text[:keep_chars]
            cost = budget
        parts.append(text)
        used_ids.append(mid)
        if mid in acquired_set:
            n_acquired_survived += 1
        budget -= cost
    evidence_text = "\n".join(parts)
    return ConditionEvidence(
        example_id=row["example_id"],
        condition=condition,
        evidence_text=evidence_text,
        evidence_token_estimate=_token_estimate(evidence_text),
        acquired_ids=tuple(acquired),
        n_acquired_survived=n_acquired_survived,
        source_memory_ids=tuple(used_ids),
    )


def load_stores() -> dict[str, dict[str, str]]:
    return build_true_stores(BEAM_PARQUET)


def store_for(row: dict, stores: dict[str, dict[str, str]]) -> dict[str, str]:
    return stores[str(row["chat_id"])]
