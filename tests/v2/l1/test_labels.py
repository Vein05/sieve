"""Tests for v2.l1.labels: label extraction on synthetic rows."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from v2.l1.labels import (
    LabeledExample,
    _build_pack_variants,
    load_beam,
    load_hotpotqa,
    load_longmemeval,
)

# --- helpers for synthetic data ---

def _make_candidates(memory_ids: list[str]) -> list[dict]:
    return [{"memory_id": mid, "content": f"Text for {mid}."} for mid in memory_ids]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


# --- _build_pack_variants ---

class TestBuildPackVariants:
    def test_full_gold_present_gives_label_1(self):
        candidates = _make_candidates(["a", "b", "c", "d", "e"])
        gold_ids = {"a", "b"}
        variants = _build_pack_variants(candidates, gold_ids, "ex1", "test", "q?")
        # Top-1 pack contains "a" only — not fully contained.
        top1 = next(v for v in variants if v.pack_memory_ids == ("a",))
        assert top1.label == 0.0
        # Top-3 pack contains "a" and "b" — fully contained.
        top3 = next(v for v in variants if v.pack_memory_ids == ("a", "b", "c"))
        assert top3.label == 1.0

    def test_empty_gold_set_label_zero(self):
        candidates = _make_candidates(["x", "y"])
        variants = _build_pack_variants(candidates, set(), "ex2", "test", "q?")
        # With no gold IDs, label is always 0.0 (vacuously nothing to cover).
        # Updated: empty gold_ids => covered=0 == len(gold_ids)=0 => label=1.0 is wrong.
        # The implementation sets label=0.0 when gold_ids is empty (no sufficiency signal).
        assert all(v.label == 0.0 for v in variants)

    def test_degenerate_one_candidate(self):
        candidates = _make_candidates(["only"])
        gold_ids = {"only"}
        variants = _build_pack_variants(candidates, gold_ids, "ex3", "test", "q?")
        # Only one candidate, so only top-1 and full pool (same).
        assert len(variants) > 0
        assert all(v.label == 1.0 for v in variants)

    def test_gold_not_in_candidates_label_zero(self):
        candidates = _make_candidates(["a", "b"])
        gold_ids = {"z"}  # z not in candidates
        variants = _build_pack_variants(candidates, gold_ids, "ex4", "test", "q?")
        assert all(v.label == 0.0 for v in variants)

    def test_variant_memory_ids_are_subsets_of_all_ids(self):
        candidates = _make_candidates(["a", "b", "c", "d", "e"])
        gold_ids = {"b"}
        variants = _build_pack_variants(candidates, gold_ids, "ex5", "test", "q?")
        all_ids = {"a", "b", "c", "d", "e"}
        for v in variants:
            assert set(v.pack_memory_ids).issubset(all_ids)


# --- load_hotpotqa ---

class TestLoadHotpotQA:
    def test_loads_rows_with_gold_decision(self, tmp_path):
        rows = [
            {
                "example_id": "hpqa-0001",
                "query": "Who directed Big Stone Gap?",
                "candidate_memories": _make_candidates(["para_a", "para_b", "para_c"]),
                "gold_decision": {
                    "selected_memory_ids": ["para_a", "para_b"],
                    "suppressed_memory_ids": [],
                    "reason_codes": [],
                },
            }
        ]
        path = tmp_path / "hotpotqa.jsonl"
        _write_jsonl(path, rows)
        examples = load_hotpotqa(path)
        assert len(examples) > 0
        # Check that some examples have label 1.0 (top-3 covers para_a and para_b).
        assert any(e.label == 1.0 for e in examples)
        # All have source = hotpotqa.
        assert all(e.source == "hotpotqa" for e in examples)

    def test_skips_rows_with_empty_gold(self, tmp_path):
        rows = [
            {
                "example_id": "hpqa-0002",
                "query": "Test query?",
                "candidate_memories": _make_candidates(["a", "b"]),
                "gold_decision": {
                    "selected_memory_ids": [],
                    "suppressed_memory_ids": [],
                    "reason_codes": [],
                },
            }
        ]
        path = tmp_path / "hotpotqa_empty.jsonl"
        _write_jsonl(path, rows)
        examples = load_hotpotqa(path)
        assert len(examples) == 0

    def test_handles_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        examples = load_hotpotqa(path)
        assert examples == []


# --- load_beam ---

class TestLoadBEAM:
    def test_loads_rows_with_answer_bearing_ids(self, tmp_path):
        rows = [
            {
                "example_id": "beam-001",
                "query": "What sprint ended on March 29?",
                "candidate_memories": _make_candidates(["turn_001", "turn_002", "turn_003"]),
                "evidence_sufficiency": {
                    "label": "sufficient",
                    "answer_bearing_memory_ids": ["turn_001"],
                    "active_context_support": False,
                },
                "memory_usefulness_labels": {},
            }
        ]
        path = tmp_path / "beam.jsonl"
        _write_jsonl(path, rows)
        examples = load_beam(path)
        assert len(examples) > 0
        assert all(e.source == "beam" for e in examples)
        assert any(e.label == 1.0 for e in examples)

    def test_fallback_to_memory_usefulness_labels(self, tmp_path):
        rows = [
            {
                "example_id": "beam-002",
                "query": "What is X?",
                "candidate_memories": _make_candidates(["t1", "t2"]),
                "evidence_sufficiency": {},
                "memory_usefulness_labels": {
                    "t1": {"labels": ["answer_bearing"], "leave_one_out_effect": "no_change"},
                    "t2": {"labels": ["distractor"], "leave_one_out_effect": "no_change"},
                },
            }
        ]
        path = tmp_path / "beam_fallback.jsonl"
        _write_jsonl(path, rows)
        examples = load_beam(path)
        assert len(examples) > 0
        assert any(e.label == 1.0 for e in examples)

    def test_skips_rows_with_no_gold(self, tmp_path):
        rows = [
            {
                "example_id": "beam-003",
                "query": "Q?",
                "candidate_memories": _make_candidates(["t1"]),
                "evidence_sufficiency": {},
                "memory_usefulness_labels": {},
            }
        ]
        path = tmp_path / "beam_no_gold.jsonl"
        _write_jsonl(path, rows)
        examples = load_beam(path)
        assert len(examples) == 0


# --- load_longmemeval ---

class TestLoadLongMemEval:
    def test_gold_session_identified_by_answer_prefix(self, tmp_path):
        rows = [
            {
                "example_id": "lmf-001",
                "query": "What degree did I graduate with?",
                "candidate_memories": [
                    {"memory_id": "answer_abc123", "content": "I graduated with a degree in Business Administration."},
                    {"memory_id": "random_xyz", "content": "Some irrelevant text."},
                ],
                "gold_decision": {"selected_memory_ids": [], "suppressed_memory_ids": [], "reason_codes": []},
                "evidence_sufficiency": {},
            }
        ]
        path = tmp_path / "lme.jsonl"
        _write_jsonl(path, rows)
        examples = load_longmemeval(path)
        assert len(examples) > 0
        # Pack containing answer_abc123 should have label 1.0.
        full_pack_ex = max(examples, key=lambda e: len(e.pack_memory_ids))
        assert full_pack_ex.label == 1.0
        assert all(e.source == "longmemeval" for e in examples)

    def test_no_gold_session_all_label_zero(self, tmp_path):
        rows = [
            {
                "example_id": "lmf-002",
                "query": "Q?",
                "candidate_memories": [
                    {"memory_id": "normal_001", "content": "Some text."},
                    {"memory_id": "normal_002", "content": "More text."},
                ],
                "gold_decision": {"selected_memory_ids": [], "suppressed_memory_ids": [], "reason_codes": []},
                "evidence_sufficiency": {},
            }
        ]
        path = tmp_path / "lme_no_gold.jsonl"
        _write_jsonl(path, rows)
        examples = load_longmemeval(path)
        # All labels are 0.0 since no answer_ prefix memory is present.
        assert len(examples) > 0
        assert all(e.label == 0.0 for e in examples)

    def test_handles_blank_lines(self, tmp_path):
        path = tmp_path / "lme_blanks.jsonl"
        path.write_text("\n\n\n")
        examples = load_longmemeval(path)
        assert examples == []
