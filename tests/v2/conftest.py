"""Shared fixtures for the v2 test suite (all offline; no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def mini_slice_rows() -> list[dict[str, Any]]:
    """3 real LongMemEval rows (trimmed content) with 20 candidates each."""
    path = FIXTURES / "mini_slice.jsonl"
    return [json.loads(line) for line in path.read_text().strip().splitlines()]


@pytest.fixture(scope="session")
def cache_fixture() -> dict[str, Any]:
    """Real sampled SIEVE + naive compilation-cache entries."""
    return json.loads((FIXTURES / "cache_entries.json").read_text())


@pytest.fixture(scope="session")
def sieve_entries_by_id(cache_fixture) -> dict[str, dict[str, Any]]:
    return {e["example_id"]: e for e in cache_fixture["sieve"]}


@pytest.fixture(scope="session")
def naive_entries_by_id(cache_fixture) -> dict[str, dict[str, Any]]:
    return {e["example_id"]: e for e in cache_fixture["naive"]}


class FakeReaderClient:
    """Deterministic stand-in for reader.client.generate_answer.

    Returns a canned answer derived from the prompt so tests can assert on
    round-tripped identity without any network call. Records every call.
    """

    def __init__(self, answer_prefix: str = "ANSWER") -> None:
        self.answer_prefix = answer_prefix
        self.calls: list[dict[str, Any]] = []

    def generate_answer(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        model = str(kwargs.get("model", "fake"))
        prompt = str(kwargs.get("prompt", ""))
        # Deterministic canned answer.
        answer = f"{self.answer_prefix}:{model}:{len(prompt)}"
        return {
            "success": True,
            "status_code": 200,
            "raw_answer": answer,
            "clean_answer": answer,
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "latency_ms": 1.5,
            "error": None,
        }


@pytest.fixture
def fake_reader_client() -> FakeReaderClient:
    return FakeReaderClient()


def fake_compile_llm(prompt: str, max_tokens: int) -> str:
    """Deterministic fake compile-LLM for the summary packager tests."""
    return f"Summary ({max_tokens} tok budget) of evidence for the query."
