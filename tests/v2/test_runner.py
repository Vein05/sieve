"""End-to-end runner tests with a FakeReaderClient (no network).

Covers: judge-compatible output layout, the shared-prompt control, resumability,
and the cost-guard dry-run plan.
"""

from __future__ import annotations

import json
from dataclasses import replace
import pytest

from v2.replay_io import existing_pairs
from v2.runner import (
    GenerationConfig,
    ReplayRequest,
    build_reader_prompt,
    build_replay_plan,
    format_cost_plan,
    replay,
)
from v2.variant_cache import (
    VariantBuildSpec,
    build_and_save,
    load_variant_cache,
    slice_fingerprint,
)

_CONFIG = {"tiktoken_encoding": "cl100k_base", "packager_params": {}, "seed": 42}
_STYLES = ["raw", "extractive"]
_BUDGETS = [200, 400]
_READERS = ["fake/reader-a", "fake/reader-b"]

# Fields cli/judge.py reads from each output (verified against cli/judge.py).
_JUDGE_REQUIRED = {
    "question", "reference_answer", "generated_answer",
    "question_type", "example_id", "system", "dataset_name", "harm_type",
}


def _request(cache, rows_by_id, run_dir):
    return ReplayRequest(
        cache=cache,
        rows_by_id=rows_by_id,
        readers=tuple(_READERS),
        run_dir=run_dir,
        parallelism=4,
    )


@pytest.fixture
def built_cache(tmp_path, mini_slice_rows):
    path = tmp_path / "vc.json"
    spec = VariantBuildSpec(_STYLES, _BUDGETS, _CONFIG)
    build_and_save(path, mini_slice_rows, spec)
    sfp = slice_fingerprint(mini_slice_rows)
    cache = load_variant_cache(path, expected_config=_CONFIG, expected_slice_fp=sfp)
    return cache


class TestSharedPromptControl:
    def test_identical_scaffolding_across_styles(self):
        # Same question+date, different evidence -> the only difference in the
        # prompt is the evidence block (the experimental control).
        p1 = build_reader_prompt(question="Q?", question_date="2023/01/01", evidence_text="AAA")
        p2 = build_reader_prompt(question="Q?", question_date="2023/01/01", evidence_text="BBB")
        assert p1.replace("AAA", "X") == p2.replace("BBB", "X")

    def test_empty_evidence_placeholder(self):
        p = build_reader_prompt(question="Q?", question_date=None, evidence_text="")
        assert "(no evidence)" in p


class TestCostGuard:
    def test_plan_call_count(self, built_cache):
        plan = build_replay_plan(built_cache, _READERS)
        # 3 rows x 2 styles x 2 budgets = 12 variants; x 2 readers = 24 calls.
        assert plan.n_variants == 12
        assert plan.n_readers == 2
        assert plan.n_calls == 24

    def test_format_cost_plan_mentions_counts(self, built_cache):
        plan = build_replay_plan(built_cache, _READERS)
        text = format_cost_plan(plan)
        assert "24" in text
        assert "--yes" in text

    def test_plan_subtracts_deterministic_routes(self, built_cache):
        key = sorted(built_cache.entries)[0]
        built_cache.entries[key] = replace(
            built_cache.entries[key], answer_override="cached"
        )
        plan = build_replay_plan(built_cache, _READERS)
        assert plan.n_deterministic_calls == 2
        assert plan.n_llm_calls == 22


def test_failed_shard_is_not_treated_as_completed(tmp_path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    record = {
        "example_id": "example-1",
        "style": "raw",
        "budget": 400,
        "reader_model": "fake/reader",
        "success": False,
    }
    (shards / "failed.json").write_text(json.dumps(record), encoding="utf-8")

    assert existing_pairs(tmp_path) == set()


class TestReplaySmoke:
    def test_generation_controls_are_forwarded_and_fingerprinted(
        self, built_cache, mini_slice_rows, fake_reader_client, tmp_path
    ):
        rows_by_id = {str(row["example_id"]): row for row in mini_slice_rows}
        generation = GenerationConfig(
            provider_routing={"order": ["deepinfra/bf16"]},
            reasoning={"effort": "none"},
        )
        request = ReplayRequest(
            built_cache,
            rows_by_id,
            ("fake/reader-a",),
            tmp_path / "run",
            generation=generation,
        )
        replay(request, fake_reader_client.generate_answer)

        assert fake_reader_client.calls[0]["provider_routing"] == {
            "order": ["deepinfra/bf16"]
        }
        assert fake_reader_client.calls[0]["reasoning"] == {"effort": "none"}
        identity = json.loads((request.run_dir / "run_identity.json").read_text())
        assert identity["generation"]["provider_routing"] == {
            "order": ["deepinfra/bf16"]
        }

    def test_deterministic_route_skips_reader(
        self, built_cache, mini_slice_rows, fake_reader_client, tmp_path
    ):
        key = sorted(built_cache.entries)[0]
        built_cache.entries[key] = replace(
            built_cache.entries[key], answer_override="cached"
        )
        rows_by_id = {str(row["example_id"]): row for row in mini_slice_rows}
        run_json = replay(
            _request(built_cache, rows_by_id, tmp_path / "run"),
            fake_reader_client.generate_answer,
        )
        deterministic = [
            output for output in run_json["outputs"]
            if output["deterministic_route"]
        ]
        assert len(deterministic) == 2
        assert {output["generated_answer"] for output in deterministic} == {"cached"}
        assert len(fake_reader_client.calls) == 22

    def test_output_layout_matches_judge(self, built_cache, mini_slice_rows, fake_reader_client, tmp_path):
        run_dir = tmp_path / "run"
        rows_by_id = {str(r["example_id"]): r for r in mini_slice_rows}
        run_json = replay(
            _request(built_cache, rows_by_id, run_dir),
            fake_reader_client.generate_answer,
        )
        # run.json exists and has an outputs list.
        assert (run_dir / "run.json").exists()
        outputs = run_json["outputs"]
        assert len(outputs) == 24
        assert len(run_json["systems"]) == 8
        # Every output has the judge-required fields.
        for o in outputs:
            assert _JUDGE_REQUIRED.issubset(o.keys()), _JUDGE_REQUIRED - o.keys()
        # No network: the fake client recorded exactly 24 calls.
        assert len(fake_reader_client.calls) == 24

    def test_condition_system_prevents_judge_score_collisions(
        self, built_cache, mini_slice_rows, fake_reader_client, tmp_path
    ):
        rows_by_id = {str(row["example_id"]): row for row in mini_slice_rows}
        run_json = replay(
            _request(built_cache, rows_by_id, tmp_path / "run"),
            fake_reader_client.generate_answer,
        )
        conditions = {
            (output["reader_model"], output["style"], output["budget"])
            for output in run_json["outputs"]
        }
        assert len(run_json["systems"]) == len(conditions)

    def test_judge_can_load_run_json(self, built_cache, mini_slice_rows, fake_reader_client, tmp_path):
        run_dir = tmp_path / "run"
        rows_by_id = {str(r["example_id"]): r for r in mini_slice_rows}
        replay(
            _request(built_cache, rows_by_id, run_dir),
            fake_reader_client.generate_answer,
        )
        # Structural check: mimic cli/judge.py's load + key access + dedup key.
        run_data = json.loads((run_dir / "run.json").read_text())
        outputs = run_data["outputs"]
        for o in outputs:
            _ = (o["question"], o["reference_answer"], o["generated_answer"], o.get("question_type", ""))
            _ = o["system"]
            _ = o["example_id"]

    def test_resumability(self, built_cache, mini_slice_rows, tmp_path):
        from tests.v2.conftest import FakeReaderClient

        run_dir = tmp_path / "run"
        rows_by_id = {str(r["example_id"]): r for r in mini_slice_rows}

        client1 = FakeReaderClient()
        replay(_request(built_cache, rows_by_id, run_dir), client1.generate_answer)
        assert len(client1.calls) == 24
        shard_files = sorted((run_dir / "shards").glob("*.json"))
        assert len(shard_files) == 24

        # Delete one shard, rerun: ONLY that pair regenerates.
        shard_files[0].unlink()
        client2 = FakeReaderClient()
        replay(_request(built_cache, rows_by_id, run_dir), client2.generate_answer)
        assert len(client2.calls) == 1
        # All 24 shards present again, run.json has 24 outputs.
        assert len(list((run_dir / "shards").glob("*.json"))) == 24
        run_data = json.loads((run_dir / "run.json").read_text())
        assert len(run_data["outputs"]) == 24

    def test_resume_rejects_changed_system(
        self, built_cache, mini_slice_rows, fake_reader_client, tmp_path
    ):
        run_dir = tmp_path / "run"
        rows_by_id = {str(row["example_id"]): row for row in mini_slice_rows}
        request = ReplayRequest(
            built_cache, rows_by_id, ("fake/reader-a",), run_dir
        )
        replay(request, fake_reader_client.generate_answer)
        with pytest.raises(ValueError, match="different cache"):
            changed = ReplayRequest(
                built_cache,
                rows_by_id,
                ("fake/reader-a",),
                run_dir,
                system="changed-system",
            )
            replay(changed, fake_reader_client.generate_answer)

    def test_reader_error_recorded_not_swallowed(self, built_cache, mini_slice_rows, tmp_path):
        run_dir = tmp_path / "run"
        rows_by_id = {str(r["example_id"]): r for r in mini_slice_rows}

        def exploding_reader(**kwargs):
            raise RuntimeError("boom")

        request = ReplayRequest(
            built_cache,
            rows_by_id,
            ("fake/reader-a",),
            run_dir,
            parallelism=2,
        )
        run_json = replay(request, exploding_reader)
        # Every output records the error string; none silently succeeds.
        for o in run_json["outputs"]:
            assert o["success"] is False
            assert "boom" in str(o["error"])
