"""Compile-once, replay-many execution for the v2 response surface.

Factorial representation cells share one reader prompt. ``sieve_v1`` is an
explicit fixed-system control and replays its cached full prompt verbatim.
Reader calls are injected, sharded, resumable, and guarded by a run identity.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from v2.replay_io import (
    build_reader_prompt,
    collect_outputs,
    ensure_run_identity,
    existing_pairs,
    output_record,
    write_shard,
)
from v2.types import EvidenceVariant, ReaderResult, VariantKey
from v2.variant_cache import VariantCache

ReaderCallable = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class GenerationConfig:
    """Reader-generation settings included in the durable run identity."""

    provider: str = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.0
    max_tokens: int = 64
    timeout_s: float = 60.0
    api_key_env: str = "OPENROUTER_API_KEY"
    provider_routing: Mapping[str, Any] = field(default_factory=dict)
    reasoning: Mapping[str, Any] = field(default_factory=dict)

    def identity(self) -> dict[str, Any]:
        """Return non-secret settings that determine generated outputs."""
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_s": self.timeout_s,
            "provider_routing": dict(self.provider_routing),
            "reasoning": dict(self.reasoning),
        }


@dataclass(frozen=True)
class ReplayRequest:
    """All state required for one response-surface replay."""

    cache: VariantCache
    rows_by_id: Mapping[str, Mapping[str, Any]]
    readers: tuple[str, ...]
    run_dir: Path
    system: str = "v2_response_surface"
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    parallelism: int = 16


@dataclass(frozen=True)
class ReplayPlan:
    """Deterministic variant-reader work list and cost summary."""

    pairs: list[tuple[VariantKey, str]]
    n_variants: int
    n_readers: int
    n_deterministic_calls: int

    @property
    def n_calls(self) -> int:
        return len(self.pairs)

    @property
    def n_llm_calls(self) -> int:
        return self.n_calls - self.n_deterministic_calls


def build_replay_plan(cache: VariantCache, readers: Sequence[str]) -> ReplayPlan:
    """Enumerate every variant-reader pair in stable order."""
    variant_keys = sorted(cache.entries)
    unique_readers = sorted(set(readers))
    pairs = [
        (VariantKey.from_str(key), reader)
        for key in variant_keys
        for reader in unique_readers
    ]
    deterministic_variants = sum(
        cache.entries[key].answer_override is not None for key in variant_keys
    )
    return ReplayPlan(
        pairs,
        len(variant_keys),
        len(unique_readers),
        deterministic_variants * len(unique_readers),
    )


def _override_result(variant: EvidenceVariant, reader_model: str) -> ReaderResult:
    return ReaderResult(
        example_id=variant.example_id,
        style=variant.style,
        budget=variant.budget,
        reader_model=reader_model,
        answer=str(variant.answer_override),
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=0.0,
    )


def _error_result(
    variant: EvidenceVariant,
    reader_model: str,
    started: float,
    error: Exception,
) -> ReaderResult:
    latency = (perf_counter() - started) * 1000.0
    return ReaderResult(
        example_id=variant.example_id,
        style=variant.style,
        budget=variant.budget,
        reader_model=reader_model,
        answer="Unknown",
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=round(latency, 3),
        error=f"{type(error).__name__}: {error}",
    )


def _run_one(
    variant: EvidenceVariant,
    row: Mapping[str, Any],
    reader_model: str,
    reader_callable: ReaderCallable,
    config: GenerationConfig,
) -> ReaderResult:
    if variant.answer_override is not None:
        return _override_result(variant, reader_model)
    prompt = variant.prompt_override or build_reader_prompt(
        question=str(row.get("query") or row.get("question") or ""),
        question_date=(
            str(row.get("question_date")) if row.get("question_date") else None
        ),
        evidence_text=variant.evidence_text,
    )
    started = perf_counter()
    try:
        generated = reader_callable(
            provider=config.provider,
            base_url=config.base_url,
            model=reader_model,
            prompt=prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
            api_key_env=config.api_key_env,
            provider_routing=dict(config.provider_routing),
            reasoning=dict(config.reasoning),
        )
    except Exception as error:
        return _error_result(variant, reader_model, started, error)
    error_text = None
    if not generated.get("success"):
        error_text = json.dumps(generated.get("error"), default=str)
    return ReaderResult(
        example_id=variant.example_id,
        style=variant.style,
        budget=variant.budget,
        reader_model=reader_model,
        answer=str(generated.get("clean_answer") or "Unknown"),
        prompt_tokens=int(generated.get("prompt_tokens") or 0),
        completion_tokens=int(generated.get("completion_tokens") or 0),
        latency_ms=float(generated.get("latency_ms") or 0.0),
        error=error_text,
        serving_provider=(
            str(generated["serving_provider"])
            if generated.get("serving_provider")
            else None
        ),
        served_model=(
            str(generated["served_model"])
            if generated.get("served_model")
            else None
        ),
    )


def _resolve_pair(
    request: ReplayRequest, key: VariantKey
) -> tuple[EvidenceVariant, Mapping[str, Any]]:
    variant = request.cache.get(key)
    row = request.rows_by_id.get(key.example_id)
    if variant is None or row is None:
        raise ValueError(f"Missing variant or row for {key.as_str()}")
    return variant, row


def _submit_pairs(
    executor: ThreadPoolExecutor,
    request: ReplayRequest,
    pairs: Sequence[tuple[VariantKey, str]],
    reader_callable: ReaderCallable,
) -> dict[Future[ReaderResult], tuple[VariantKey, str]]:
    futures: dict[Future[ReaderResult], tuple[VariantKey, str]] = {}
    for key, reader in pairs:
        variant, row = _resolve_pair(request, key)
        future = executor.submit(
            _run_one, variant, row, reader, reader_callable, request.generation
        )
        futures[future] = key, reader
    return futures


def _execute_pairs(
    request: ReplayRequest,
    pairs: Sequence[tuple[VariantKey, str]],
    reader_callable: ReaderCallable,
) -> None:
    with ThreadPoolExecutor(max_workers=max(1, request.parallelism)) as executor:
        futures = _submit_pairs(executor, request, pairs, reader_callable)
        for future in as_completed(futures):
            key, reader = futures[future]
            result = future.result()
            variant, row = _resolve_pair(request, key)
            record = output_record(variant, row, reader, result, request.system)
            write_shard(request.run_dir, key, reader, record)


def _run_payload(
    request: ReplayRequest,
    plan: ReplayPlan,
    calls_run: int,
    wall_seconds: float,
) -> dict[str, Any]:
    outputs = collect_outputs(request.run_dir)
    return {
        "meta": {
            "system": request.system,
            "cache_fingerprint": request.cache.fingerprint,
            "readers": sorted(set(request.readers)),
            "n_variants": plan.n_variants,
            "n_readers": plan.n_readers,
            "n_calls_planned": plan.n_calls,
            "n_calls_run": calls_run,
            "wall_seconds": round(wall_seconds, 2),
        },
        "systems": sorted({str(output["system"]) for output in outputs}),
        "outputs": outputs,
    }


def replay(request: ReplayRequest, reader_callable: ReaderCallable) -> dict[str, Any]:
    """Replay all missing pairs and write judge-compatible outputs."""
    (request.run_dir / "shards").mkdir(parents=True, exist_ok=True)
    ensure_run_identity(
        request.run_dir,
        request.cache,
        request.readers,
        request.system,
        request.generation.identity(),
    )
    plan = build_replay_plan(request.cache, request.readers)
    done = existing_pairs(request.run_dir)
    todo = [pair for pair in plan.pairs if (pair[0].as_str(), pair[1]) not in done]
    started = perf_counter()
    _execute_pairs(request, todo, reader_callable)
    payload = _run_payload(request, plan, len(todo), perf_counter() - started)
    (request.run_dir / "run.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def format_cost_plan(plan: ReplayPlan) -> str:
    """Return the cost-guard summary printed before any reader calls."""
    return (
        "v2 replay cost plan:\n"
        f"  variants           : {plan.n_variants}\n"
        f"  readers            : {plan.n_readers}\n"
        f"  total (v x r) calls: {plan.n_calls}\n"
        f"  deterministic routes: {plan.n_deterministic_calls}\n"
        f"  estimated LLM calls: {plan.n_llm_calls}\n"
        "  (re-run with --yes to proceed; default is a dry-run)"
    )


__all__ = [
    "GenerationConfig",
    "ReplayPlan",
    "ReplayRequest",
    "build_reader_prompt",
    "build_replay_plan",
    "format_cost_plan",
    "replay",
]
