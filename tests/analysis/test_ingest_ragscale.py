"""
Offline unit tests for ingest_ragscale_matrix.py.
All tests use tiny synthetic fixtures — no reads from the real 18 GB run directory.

Label rule (copied from module docstring):
  - judge_score_v2 == 2  -> correct = 1
  - judge_score_v2 == 0  -> correct = 0
  - judge_score_v2 == 1  -> correct = NaN (malformed)
  - QMSum runs           -> correct = NaN (rubric task)
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import sys

# Ensure repo root on path (matches conftest.py setup)
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.ingest_ragscale_matrix import (
    _correct_label,
    _infer_reader_model,
    _infer_slice,
    _map_style,
    _parse_budget,
    dedup_rows,
    ingest,
    normalize_output_row,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_output(
    example_id: str = "ex1",
    dataset_name: str = "longmemeval",
    judge_score_v2: int | None = 2,
    system: str = "naive_top_k",
    compiler_model: str | None = None,
    question_type: str = "single_session_user",
) -> dict:
    return {
        "example_id": example_id,
        "dataset_name": dataset_name,
        "question_type": question_type,
        "judge_score_v2": judge_score_v2,
        "judge_factuality_v2": None,
        "judge_completeness_v2": None,
        "judge_abstention_v2": None,
        "judge_reason_v2": "yes" if judge_score_v2 == 2 else "no",
        "system": system,
        "compiler_model": compiler_model,
        "token_f1": 0.5,
        "prompt_tokens": 1000,
        "completion_tokens": 20,
        "selected_memory_tokens": 800,
        "latency_ms": 500,
    }


def _make_judge_run(
    outputs: list[dict],
    meta: dict | None = None,
    generator_model: str | None = None,
    generated_at: str | None = None,
    systems: list[str] | None = None,
) -> dict:
    data: dict = {
        "outputs": outputs,
        "judge_model_v2": "deepseek/deepseek-chat-v3-0324",
    }
    if meta is not None:
        data["meta"] = meta
    if generator_model is not None:
        data["generator_model"] = generator_model
    if generated_at is not None:
        data["generated_at"] = generated_at
    if systems is not None:
        data["systems"] = systems
    return data


def _write_run(
    base_dir: Path,
    run_name: str,
    judge_data: dict,
) -> Path:
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "judge_run_v2.json").write_text(json.dumps(judge_data), encoding="utf-8")
    return run_dir


# ---------------------------------------------------------------------------
# Label rule tests
# ---------------------------------------------------------------------------

class TestCorrectLabel:
    def test_score_2_is_correct(self):
        assert _correct_label({"judge_score_v2": 2}, is_qsum=False) == 1

    def test_score_0_is_incorrect(self):
        assert _correct_label({"judge_score_v2": 0}, is_qsum=False) == 0

    def test_score_1_is_nan(self):
        # score==1 is malformed for binary tasks
        assert _correct_label({"judge_score_v2": 1}, is_qsum=False) is None

    def test_missing_score_is_nan(self):
        assert _correct_label({}, is_qsum=False) is None

    def test_qmsum_always_nan(self):
        # Even score==2 for qmsum returns None
        assert _correct_label({"judge_score_v2": 2}, is_qsum=True) is None
        assert _correct_label({"judge_score_v2": 0}, is_qsum=True) is None
        assert _correct_label({"judge_score_v2": 1}, is_qsum=True) is None


# ---------------------------------------------------------------------------
# Style mapping tests
# ---------------------------------------------------------------------------

class TestStyleMap:
    def test_naive_top_k(self):
        assert _map_style("naive_top_k") == "raw"

    def test_bm25_top_k(self):
        assert _map_style("bm25_top_k") == "raw"

    def test_bm25_sieve(self):
        assert _map_style("bm25_sieve") == "structured_v1"

    def test_sieve(self):
        assert _map_style("sieve") == "structured_v1"

    def test_llm_summarize(self):
        assert _map_style("llm_summarize") == "summary"

    def test_recomp_abstractive(self):
        assert _map_style("recomp_abstractive") == "summary_trained"

    def test_exit_extractive(self):
        assert _map_style("exit_extractive") == "extractive_trained"

    def test_provence_extractive(self):
        assert _map_style("provence_extractive") == "extractive_trained"

    def test_llmlingua_v2(self):
        assert _map_style("llmlingua_v2") == "token_pruned"

    def test_unknown_passthrough(self):
        # Unknown systems are passed through verbatim
        assert _map_style("future_system") == "future_system"


# ---------------------------------------------------------------------------
# Budget parsing tests
# ---------------------------------------------------------------------------

class TestBudgetParsing:
    def test_at_suffix(self):
        assert _parse_budget("paper_llama8b_naive_at200", {}, {}) == 200

    def test_cap_suffix(self):
        assert _parse_budget("paper_claude_raw_cap40", {}, {}) == 40

    def test_at800(self):
        assert _parse_budget("paper_gemma_sieve_at800", {}, {}) == 800

    def test_cap400(self):
        assert _parse_budget("paper_gpt_raw_cap400", {}, {}) == 400

    def test_no_budget(self):
        assert _parse_budget("paper_llama8b_naive", {}, {}) is None

    def test_external_prompt_cap_in_top_level(self):
        assert _parse_budget("paper_some_run", {}, {"external_prompt_cap": 150}) == 150


# ---------------------------------------------------------------------------
# Slice inference tests
# ---------------------------------------------------------------------------

class TestSliceInference:
    def test_longmemeval_default(self):
        assert _infer_slice("paper_llama8b_naive", {}, {}) == "longmemeval"

    def test_hotpotqa(self):
        assert _infer_slice("paper_hotpotqa_llama8b_naive", {}, {}) == "hotpotqa"

    def test_qmsum(self):
        assert _infer_slice("qmsum_naive_gpt-4.1-mini", {}, {}) == "qmsum"

    def test_musique(self):
        assert _infer_slice("paper_musique_llama8b_naive", {}, {}) == "musique"

    def test_nq(self):
        assert _infer_slice("paper_nq_llama8b_naive", {}, {}) == "nq"

    def test_locomo(self):
        assert _infer_slice("paper_llama8b_locomo_naive", {}, {}) == "locomo"

    def test_convomem(self):
        assert _infer_slice("paper_llama8b_convomem_naive", {}, {}) == "convomem"

    def test_oracle(self):
        assert _infer_slice("paper_llama8b_oracle_v2_naive", {}, {}) == "longmemeval_oracle"

    def test_scifact(self):
        assert _infer_slice("scifact_llama-3.1-8b-instruct_naive", {}, {}) == "scifact"


# ---------------------------------------------------------------------------
# Model inference tests
# ---------------------------------------------------------------------------

class TestModelInference:
    def test_from_meta(self):
        meta = {"reader_model": "openai/gpt-4.1-mini"}
        assert _infer_reader_model("paper_something", meta, {}) == "openai/gpt-4.1-mini"

    def test_from_generator_model(self):
        assert _infer_reader_model(
            "paper_x", {}, {"generator_model": "meta-llama/llama-3.1-8b-instruct"}
        ) == "meta-llama/llama-3.1-8b-instruct"

    def test_from_dir_name_llama8b(self):
        result = _infer_reader_model("paper_llama8b_naive", {}, {})
        assert result == "meta-llama/llama-3.1-8b-instruct"

    def test_from_dir_name_gpt41mini(self):
        result = _infer_reader_model("paper_gpt41mini_sieve", {}, {})
        assert result == "openai/gpt-4.1-mini"

    def test_from_dir_name_gemma12b(self):
        result = _infer_reader_model("paper_gemma12b_naive", {}, {})
        assert result == "google/gemma-3-12b-it"

    def test_from_dir_name_commandr(self):
        result = _infer_reader_model("paper_commandr_naive", {}, {})
        assert result == "cohere/command-r-08-2024"


# ---------------------------------------------------------------------------
# Dedup tests
# ---------------------------------------------------------------------------

class TestDedup:
    def _make_row(
        self,
        reader_model: str = "openai/gpt-4.1-mini",
        style: str = "raw",
        budget: int | None = None,
        slice_name: str = "longmemeval",
        example_id: str = "ex1",
        correct: int | None = 1,
        ts: str = "2026-04-22T10:00:00",
        run_name: str = "run_a",
    ) -> dict:
        return {
            "reader_model": reader_model,
            "style": style,
            "budget": budget,
            "slice_name": slice_name,
            "example_id": example_id,
            "correct": correct,
            "timestamp": ts,
            "run_name": run_name,
            "provider": None,
            "compiler_model": None,
            "dataset_name": "longmemeval",
            "system_raw": "naive_top_k",
            "question_type": "single_session_user",
            "task_type": "qa",
            "token_f1": 0.5,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "selected_memory_tokens": 80,
            "latency_ms": 200,
            "judge_score_v2": 2 if correct == 1 else 0,
            "judge_factuality_v2": None,
            "judge_completeness_v2": None,
            "judge_abstention_v2": None,
            "judge_reason_v2": None,
        }

    def test_single_row_no_dup(self):
        rows = [self._make_row()]
        kept, stats = dedup_rows(rows)
        assert len(kept) == 1
        assert stats["n_duplicates_removed"] == 0

    def test_keeps_most_recent(self):
        older = self._make_row(ts="2026-04-20T10:00:00", run_name="run_old", correct=0)
        newer = self._make_row(ts="2026-04-22T10:00:00", run_name="run_new", correct=1)
        kept, stats = dedup_rows([older, newer])
        assert len(kept) == 1
        assert kept[0]["run_name"] == "run_new"
        assert stats["n_duplicates_removed"] == 1

    def test_no_disagreement_when_same_label(self):
        r1 = self._make_row(ts="2026-04-20T10:00:00", correct=1, run_name="a")
        r2 = self._make_row(ts="2026-04-22T10:00:00", correct=1, run_name="b")
        kept, stats = dedup_rows([r1, r2])
        assert stats["n_disagreement_pairs"] == 0
        assert kept[0]["label_disagreement"] is False

    def test_disagreement_flagged(self):
        r1 = self._make_row(ts="2026-04-20T10:00:00", correct=0, run_name="a")
        r2 = self._make_row(ts="2026-04-22T10:00:00", correct=1, run_name="b")
        kept, stats = dedup_rows([r1, r2])
        assert stats["n_disagreement_pairs"] == 1
        assert kept[0]["label_disagreement"] is True

    def test_different_keys_not_deduped(self):
        r1 = self._make_row(example_id="ex1")
        r2 = self._make_row(example_id="ex2")
        kept, stats = dedup_rows([r1, r2])
        assert len(kept) == 2
        assert stats["n_duplicates_removed"] == 0

    def test_nan_labels_dont_trigger_disagreement(self):
        """Two rows with None correct should not be counted as disagreement."""
        r1 = self._make_row(ts="2026-04-20T10:00:00", correct=None, run_name="a")
        r2 = self._make_row(ts="2026-04-22T10:00:00", correct=None, run_name="b")
        kept, stats = dedup_rows([r1, r2])
        assert stats["n_disagreement_pairs"] == 0


# ---------------------------------------------------------------------------
# Integration / ingest tests with synthetic fixtures
# ---------------------------------------------------------------------------

class TestIngestFixtures:
    """Test full ingest() call with tiny synthetic judge_run_v2.json files."""

    def test_missing_field_tolerance(self, tmp_path):
        """Runs with missing optional fields should ingest without crash."""
        outputs = [
            {
                "example_id": "ex1",
                "dataset_name": "longmemeval",
                "system": "naive_top_k",
                "judge_score_v2": 2,
                # intentionally omit many optional fields
            }
        ]
        judge_data = _make_judge_run(
            outputs=outputs,
            meta={"reader_model": "openai/gpt-4.1-mini", "timestamp": "2026-04-22T10:00:00"},
        )
        _write_run(tmp_path, "paper_gpt41mini_naive", judge_data)

        rows, stats = ingest(tmp_path)
        assert stats["loaded_runs"] == 1
        assert len(rows) == 1
        assert rows[0]["correct"] == 1
        assert rows[0]["token_f1"] is None  # missing field -> None
        assert rows[0]["prompt_tokens"] is None

    def test_label_rule_applied(self, tmp_path):
        """Score 0 -> 0, score 2 -> 1, score 1 -> None."""
        outputs = [
            _make_output("ex1", judge_score_v2=2),
            _make_output("ex2", judge_score_v2=0),
            _make_output("ex3", judge_score_v2=1),  # malformed
        ]
        judge_data = _make_judge_run(
            outputs=outputs,
            meta={"reader_model": "openai/gpt-4.1-mini", "timestamp": "2026-04-22T10:00:00"},
        )
        _write_run(tmp_path, "paper_gpt41mini_naive", judge_data)

        rows, _ = ingest(tmp_path)
        by_id = {r["example_id"]: r for r in rows}
        assert by_id["ex1"]["correct"] == 1
        assert by_id["ex2"]["correct"] == 0
        assert by_id["ex3"]["correct"] is None

    def test_qmsum_label_excluded(self, tmp_path):
        """QMSum rows must have correct=None regardless of judge_score_v2."""
        outputs = [
            {
                "example_id": "q1",
                "dataset_name": "qmsum",
                "system": "qmsum_naive",
                "judge_score_v2": 2,
                "judge_reason_v2": "yes",
            }
        ]
        judge_data = _make_judge_run(
            outputs=outputs,
            meta={
                "reader_model": "openai/gpt-4.1-mini",
                "timestamp": "2026-05-01T10:00:00",
                "system": "qmsum_naive",
            },
            systems=["qmsum_naive"],
        )
        _write_run(tmp_path, "qmsum_naive_gpt-4.1-mini", judge_data)

        rows, stats = ingest(tmp_path)
        assert len(rows) == 1
        assert rows[0]["task_type"] == "qmsum"
        assert rows[0]["correct"] is None

    def test_skip_dirs_without_judge(self, tmp_path):
        """Dirs without judge_run_v2.json should be counted as skipped."""
        # Create two run dirs: one with judge, one without
        outputs = [_make_output("ex1")]
        judge_data = _make_judge_run(
            outputs=outputs,
            meta={"reader_model": "openai/gpt-4.1-mini", "timestamp": "2026-04-22T10:00:00"},
        )
        _write_run(tmp_path, "paper_gpt41mini_naive", judge_data)

        # Dir without judge
        empty_dir = tmp_path / "paper_missing_run"
        empty_dir.mkdir()
        (empty_dir / "trace.json").write_text("{}")

        rows, stats = ingest(tmp_path)
        assert stats["skipped_no_judge"] == 1
        assert stats["loaded_runs"] == 1

    def test_dedup_keep_latest_across_fixture_runs(self, tmp_path):
        """Two runs sharing same (reader, style, budget, slice, example_id) -> keep latest."""
        shared_example = "ex_shared"
        ts_old = "2026-04-20T10:00:00"
        ts_new = "2026-04-25T10:00:00"

        outputs_old = [_make_output(shared_example, judge_score_v2=0)]
        outputs_new = [_make_output(shared_example, judge_score_v2=2)]

        judge_old = _make_judge_run(
            outputs=outputs_old,
            meta={"reader_model": "openai/gpt-4.1-mini", "timestamp": ts_old},
        )
        judge_new = _make_judge_run(
            outputs=outputs_new,
            meta={"reader_model": "openai/gpt-4.1-mini", "timestamp": ts_new},
        )

        _write_run(tmp_path, "paper_gpt41mini_naive_old", judge_old)
        _write_run(tmp_path, "paper_gpt41mini_naive_new", judge_new)

        rows, _ = ingest(tmp_path)
        kept, dedup_stats = dedup_rows(rows)

        assert dedup_stats["n_duplicates_removed"] == 1
        # The newer run (score=2, correct=1) should be kept
        shared_kept = [r for r in kept if r["example_id"] == shared_example]
        assert len(shared_kept) == 1
        assert shared_kept[0]["correct"] == 1  # from the newer run

    def test_budget_parsed_from_dir_name(self, tmp_path):
        outputs = [_make_output("ex1")]
        judge_data = _make_judge_run(
            outputs=outputs,
            meta={"reader_model": "openai/gpt-4.1-mini", "timestamp": "2026-04-22T10:00:00"},
        )
        _write_run(tmp_path, "paper_gpt41mini_naive_at400", judge_data)

        rows, _ = ingest(tmp_path)
        assert rows[0]["budget"] == 400

    def test_no_budget_returns_none(self, tmp_path):
        outputs = [_make_output("ex1")]
        judge_data = _make_judge_run(
            outputs=outputs,
            meta={"reader_model": "openai/gpt-4.1-mini", "timestamp": "2026-04-22T10:00:00"},
        )
        _write_run(tmp_path, "paper_gpt41mini_naive", judge_data)

        rows, _ = ingest(tmp_path)
        assert rows[0]["budget"] is None

    def test_style_mapped_correctly(self, tmp_path):
        outputs = [_make_output("ex1", system="bm25_sieve")]
        judge_data = _make_judge_run(
            outputs=outputs,
            meta={"reader_model": "openai/gpt-4.1-mini", "timestamp": "2026-04-22T10:00:00"},
        )
        _write_run(tmp_path, "paper_gpt41mini_sieve", judge_data)

        rows, _ = ingest(tmp_path)
        assert rows[0]["style"] == "structured_v1"
        assert rows[0]["system_raw"] == "bm25_sieve"

    def test_multiple_runs_two_models(self, tmp_path):
        """Two different models should produce separate rows."""
        for model_alias, model_id in [
            ("llama8b", "meta-llama/llama-3.1-8b-instruct"),
            ("gpt41mini", "openai/gpt-4.1-mini"),
        ]:
            outputs = [_make_output("ex1", judge_score_v2=2)]
            judge_data = _make_judge_run(
                outputs=outputs,
                meta={"reader_model": model_id, "timestamp": "2026-04-22T10:00:00"},
            )
            _write_run(tmp_path, f"paper_{model_alias}_naive", judge_data)

        rows, stats = ingest(tmp_path)
        assert stats["loaded_runs"] == 2
        assert len(rows) == 2
        models = {r["reader_model"] for r in rows}
        assert "meta-llama/llama-3.1-8b-instruct" in models
        assert "openai/gpt-4.1-mini" in models
