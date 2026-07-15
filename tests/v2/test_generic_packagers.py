"""Tests for the v3 generic pilot packagers (structured_generic, guarded_structured).

All offline: no network, no v1 cache, no benchmark vocabulary. Covers three
row shapes required by the pilot:
  (a) a LongMemEval-shaped row (date-prefixed content),
  (b) a HotpotQA-shaped row (no dates, question_date=None),
  (c) a degenerate row that triggers the starvation floor and the guard.
"""

from __future__ import annotations

from typing import Any

import pytest

from v2.budget import count_tokens
from v2.packagers import get_packager
from v2.packagers.base import PackagerContext
from v2.packagers.guarded_structured import COVERAGE_MIN_FRACTION
from v2.packagers.structured_generic import MIN_EVIDENCE_TOKENS
from v2.scorer import StemOverlapScorer
from v2.types import EvidenceVariant


def _ctx() -> PackagerContext:
    return PackagerContext(
        relevance_scorer=StemOverlapScorer(),
        params={},
        encoding="cl100k_base",
    )


def _hotpotqa_row() -> dict[str, Any]:
    """HotpotQA-shaped row: no date prefixes, question_date=None."""
    return {
        "example_id": "hp_1",
        "query": "Which magazine was started first, Arthur's Magazine or First for Women?",
        "question_date": None,
        "candidate_memories": [
            {"memory_id": "m0", "content": "Arthur's Magazine was an American literary periodical first published in 1844 in Philadelphia."},
            {"memory_id": "m1", "content": "First for Women is a woman's magazine that was launched in 1989 by Bauer Media Group."},
            {"memory_id": "m2", "content": "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris."},
            {"memory_id": "m3", "content": "Photosynthesis converts light energy into chemical energy stored in glucose molecules."},
        ],
    }


def _degenerate_row() -> dict[str, Any]:
    """One short candidate, minimal query overlap: forces floor + guard."""
    return {
        "example_id": "deg_1",
        "query": "What is the capital of France?",
        "question_date": None,
        "candidate_memories": [
            {"memory_id": "d0", "content": "It rained."},
            {"memory_id": "d1", "content": "The meeting was moved."},
        ],
    }


# ---------------------------------------------------------------------------
# structured_generic
# ---------------------------------------------------------------------------

class TestStructuredGeneric:
    def test_longmemeval_row_orders_and_quotes(self, mini_slice_rows):
        row = mini_slice_rows[0]
        v = get_packager("structured_generic").package(row, budget=400, ctx=_ctx())
        assert isinstance(v, EvidenceVariant)
        assert v.style == "structured_generic"
        assert v.evidence_text.strip()
        assert v.token_count <= 400 or v.meta["overflow"]
        # At least one labelled section header is emitted.
        assert any(line.startswith("# ") for line in v.evidence_text.splitlines())

    def test_hotpotqa_row_no_dates(self):
        row = _hotpotqa_row()
        v = get_packager("structured_generic").package(row, budget=400, ctx=_ctx())
        assert v.evidence_text.strip()
        assert v.token_count <= 400 or v.meta["overflow"]
        # Exact value quotes for the two publication years must appear.
        assert '"1844"' in v.evidence_text
        assert '"1989"' in v.evidence_text

    def test_never_below_floor_when_content_available(self):
        row = _hotpotqa_row()
        v = get_packager("structured_generic").package(row, budget=0, ctx=_ctx())
        # Uncapped: rich content means the floor is comfortably exceeded.
        assert count_tokens(v.evidence_text) >= MIN_EVIDENCE_TOKENS

    def test_degenerate_row_fires_floor(self):
        row = _degenerate_row()
        v = get_packager("structured_generic").package(row, budget=0, ctx=_ctx())
        # Total available content is far below the floor -> floor fires, and we
        # never emit fewer tokens than the available content allows.
        assert v.meta["floor_fired"] is True
        # Every candidate is included (nothing left to reach the floor).
        assert v.meta["n_kept"] >= len(row["candidate_memories"])

    def test_respects_budget(self, mini_slice_rows):
        row = mini_slice_rows[0]
        v = get_packager("structured_generic").package(row, budget=200, ctx=_ctx())
        assert v.token_count <= 200 or v.meta["overflow"]

    def test_deterministic(self, mini_slice_rows):
        row = mini_slice_rows[0]
        a = get_packager("structured_generic").package(row, budget=400, ctx=_ctx())
        b = get_packager("structured_generic").package(row, budget=400, ctx=_ctx())
        assert a.evidence_text == b.evidence_text
        assert a.source_memory_ids == b.source_memory_ids


# ---------------------------------------------------------------------------
# guarded_structured
# ---------------------------------------------------------------------------

class TestGuardedStructured:
    def test_records_guard_path(self, mini_slice_rows):
        row = mini_slice_rows[0]
        v = get_packager("guarded_structured").package(row, budget=400, ctx=_ctx())
        assert v.style == "guarded_structured"
        assert v.meta["guard_path"] in {"structured", "filtered_raw"}
        assert isinstance(v.meta["guard_fired"], bool)
        assert 0.0 <= v.meta["coverage"] <= 1.0

    def test_high_coverage_keeps_structured(self):
        # HotpotQA row: every relevant passage is quoted in the structured pack,
        # so coverage should be high and the guard should NOT fire.
        row = _hotpotqa_row()
        v = get_packager("guarded_structured").package(row, budget=0, ctx=_ctx())
        assert v.meta["coverage"] >= COVERAGE_MIN_FRACTION
        assert v.meta["guard_path"] == "structured"
        assert v.meta["guard_fired"] is False

    def test_low_coverage_falls_back_to_filtered_raw(self):
        # Tight budget starves the structured rendering of most relevant
        # passages -> coverage drops -> guard fires -> filtered_raw emitted.
        row = _hotpotqa_row()
        v = get_packager("guarded_structured").package(row, budget=30, ctx=_ctx())
        assert v.meta["guard_fired"] is True
        assert v.meta["guard_path"] == "filtered_raw"
        assert v.meta["inner_style"] == "filtered_raw"

    def test_respects_budget(self):
        row = _hotpotqa_row()
        v = get_packager("guarded_structured").package(row, budget=200, ctx=_ctx())
        inner_overflow = v.meta["inner_meta"].get("overflow", False)
        assert v.token_count <= 200 or inner_overflow

    def test_deterministic(self, mini_slice_rows):
        row = mini_slice_rows[0]
        a = get_packager("guarded_structured").package(row, budget=400, ctx=_ctx())
        b = get_packager("guarded_structured").package(row, budget=400, ctx=_ctx())
        assert a.evidence_text == b.evidence_text
        assert a.meta["guard_path"] == b.meta["guard_path"]


# ---------------------------------------------------------------------------
# no benchmark vocabulary / no v1 cache dependency
# ---------------------------------------------------------------------------

class TestGenericIndependence:
    @pytest.mark.parametrize("style", ["structured_generic", "guarded_structured"])
    def test_runs_without_sieve_entry(self, style):
        row = _hotpotqa_row()
        # ctx has sieve_entry=None by default; must not raise.
        v = get_packager(style).package(row, budget=400, ctx=_ctx())
        assert v.evidence_text.strip()

    @pytest.mark.parametrize(
        "mod_name",
        ["v2.packagers.structured_generic", "v2.packagers.guarded_structured"],
    )
    def test_no_v1_cache_import(self, mod_name):
        import ast
        import importlib
        import inspect

        mod = importlib.import_module(mod_name)
        tree = ast.parse(inspect.getsource(mod))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        # Structured-from-cache (v1) modules must not be imported.
        assert "v2.packagers.structured" not in imported
        assert "v2.packagers.structured_quotes" not in imported
