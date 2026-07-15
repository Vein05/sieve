"""Generation-configuration guards for the v2 replay CLI."""

from __future__ import annotations

from argparse import Namespace

from cli.run_v2 import _generation_config, _select_evaluation_rows


def test_generation_config_preserves_locked_provider_controls() -> None:
    args = Namespace(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        max_tokens=64,
        timeout=60.0,
        api_key_env="OPENROUTER_API_KEY",
    )
    routing = {
        "order": ["alibaba"],
        "only": ["alibaba"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    config = {
        "reader": {
            "provider_routing": routing,
            "reasoning": {"effort": "none"},
        }
    }

    generation = _generation_config(args, config)

    assert dict(generation.provider_routing) == routing
    assert dict(generation.reasoning) == {"effort": "none"}


def test_pilot_split_filters_rows_after_deterministic_annotation() -> None:
    rows = [{"example_id": f"example-{index}"} for index in range(20)]
    config = {
        "evaluation_split": {
            "seed": 42,
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "pilot_split": "validation",
        }
    }

    selected = _select_evaluation_rows(rows, config)

    assert selected
    assert all(row["v2_split"] == "validation" for row in selected)


def test_compile_llm_disables_reasoning(monkeypatch) -> None:
    import cli.run_v2 as run_v2

    captured: dict[str, object] = {}

    def _fake_generate_answer(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"success": True, "clean_answer": "a summary"}

    import reader.client as client

    monkeypatch.setattr(client, "generate_answer", _fake_generate_answer)
    compile_llm = run_v2._make_compile_llm(
        model="qwen/qwen3-8b",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
    )
    result = compile_llm("summarize this", 400)

    assert result["text"] == "a summary"
    assert captured["reasoning"] == run_v2.COMPILE_REASONING
    assert captured["max_tokens"] == 400
