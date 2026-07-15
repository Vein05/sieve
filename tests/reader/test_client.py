"""Request-shape tests for the shared generation client."""

from __future__ import annotations

from typing import Any

from reader import client


def test_openrouter_forwards_routing_and_reasoning(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_request(url, payload, timeout_s, headers=None):
        captured.update(payload)
        return 200, {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            "provider": "Alibaba",
            "model": "qwen/qwen3.5-397b-a17b",
        }

    monkeypatch.setenv("TEST_OPENROUTER_KEY", "secret")
    monkeypatch.setattr(client, "_json_request", fake_request)
    routing = {
        "order": ["deepinfra/bf16"],
        "quantizations": ["bf16"],
        "allow_fallbacks": False,
    }
    reasoning = {"effort": "none"}

    result = client.generate_answer(
        provider="openrouter",
        base_url="https://example.test/v1",
        model="qwen/qwen3.5-9b",
        prompt="question",
        api_key_env="TEST_OPENROUTER_KEY",
        provider_routing=routing,
        reasoning=reasoning,
    )

    assert result["success"] is True
    assert captured["provider"] == routing
    assert captured["reasoning"] == reasoning
    assert result["serving_provider"] == "Alibaba"
    assert result["served_model"] == "qwen/qwen3.5-397b-a17b"
