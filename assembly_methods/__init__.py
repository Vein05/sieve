"""Shared assembly methods for selector-only and answer-generation experiments."""

from __future__ import annotations

import importlib
from time import perf_counter
from typing import Any


DEFAULT_PAPER_METHODS = [
    "naive_top_k",
    "bm25_sieve",
]


LEARNED_METHOD_REQUIRED_MODELS: dict[str, tuple[str, ...]] = {}


LEGACY_METHOD_ALIASES: dict[str, str] = {}


_MODELING_EXPORTS = {
    "ACTIVE_METHOD_MODEL_CACHE_PATH",
    "CACHED_METHOD_MODEL_KEYS",
    "METHOD_MODEL_CACHE_PATH",
    "METHOD_CACHE_MODEL_KEYS",
    "METHOD_MODEL_CACHE_PROTOCOL_VERSION",
    "_method_model_cache_metadata",
    "_write_active_method_model_cache",
    "build_method_context",
    "load_training_rows",
}

_METHOD_EXPORTS = {
    "method_bm25_top_k": (".bm25_top_k", "method_bm25_top_k"),
    "method_bm25_sieve": (".bm25_sieve", "method_bm25_sieve"),
    "method_naive_top_k": (".top_k", "method_naive_top_k"),
}

_METHOD_CACHE: dict[str, Any] | None = None


def _modeling_module():
    return importlib.import_module(".modeling", __name__)


def _method_module(module_name: str):
    return importlib.import_module(module_name, __name__)


def _modeling_attr(name: str) -> Any:
    module = _modeling_module()
    return getattr(module, name)


def _method_registry() -> dict[str, Any]:
    global _METHOD_CACHE
    if _METHOD_CACHE is None:
        _METHOD_CACHE = {}
    return _METHOD_CACHE


def _get_method(name: str) -> Any:
    registry = _method_registry()
    if name not in registry:
        module_name, attr_name = _METHOD_EXPORTS.get(f"method_{name}", (f".{name}", f"method_{name}"))
        registry[name] = getattr(_method_module(module_name), attr_name)
    return registry[name]



def normalize_method_name(name: str) -> str:
    return LEGACY_METHOD_ALIASES.get(name, name)


def validate_method_context_for_methods(
    method_names: list[str],
    context: dict[str, Any],
) -> None:
    missing_by_method: dict[str, list[str]] = {}
    for name in method_names:
        canonical = normalize_method_name(name)
        required_keys = LEARNED_METHOD_REQUIRED_MODELS.get(canonical, ())
        missing = [key for key in required_keys if context.get(key) is None]
        if missing:
            missing_by_method[canonical] = missing
    if not missing_by_method:
        return
    details = "; ".join(
        f"{method} missing {', '.join(keys)}"
        for method, keys in sorted(missing_by_method.items())
    )
    raise ValueError(
        "Paper-facing learned method context is invalid: "
        f"{details}. "
        "Fix the training slice inputs or the cached learned-model bundle before rerunning."
    )


def run_method(
    name: str,
    row: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    started = perf_counter()
    normalized_name = normalize_method_name(name)
    result = _get_method(normalized_name)(row, context)
    latency_ms = (perf_counter() - started) * 1000.0
    all_ids = [str(candidate["memory_id"]) for candidate in row["candidate_memories"]]
    selected_ids = list(result["selected_memory_ids"])
    selected_set = set(selected_ids)
    suppressed_ids = [memory_id for memory_id in all_ids if memory_id not in selected_set]
    selection_meta = dict(result.get("selection_meta") or {})
    decision_summary = {
        "predicted_action": "commit" if selected_ids else "suppress",
        "predicted_selected_memory_ids": selected_ids,
        "predicted_suppressed_memory_ids": suppressed_ids,
        "selected_count": result["selected_count"],
        "selected_token_count": result["selected_token_count"],
        "target_token_budget": result["target_token_budget"],
        "budget_mode": result["budget_mode"],
        "latency_ms": round(latency_ms, 3),
    }
    if selection_meta:
        decision_summary.update(
            {
                "query_family": selection_meta.get("query_family"),
                "selection_mode": selection_meta.get("selection_mode"),
                "selection_margin": selection_meta.get("selection_margin"),
                "reason_codes": list(selection_meta.get("reason_codes", [])),
                "anchor_count": selection_meta.get("anchor_count"),
            }
        )
    return {
        "strategy": name,
        "example_id": row["example_id"],
        "source": row.get("source"),
        "harm_type": row.get("harm_type"),
        "perturbation_family": row.get("perturbation_family"),
        "evidence_sufficiency": row.get("evidence_sufficiency"),
        "query": row["query"],
        "gold_decision": row.get("gold_decision"),
        "selection_meta": selection_meta,
        "query_family": selection_meta.get("query_family"),
        "selection_mode": selection_meta.get("selection_mode"),
        "selection_margin": selection_meta.get("selection_margin"),
        "reason_codes": list(selection_meta.get("reason_codes", [])),
        "anchor_count": selection_meta.get("anchor_count"),
        "decision_summary": decision_summary,
    }


def __getattr__(name: str) -> Any:
    if name in _MODELING_EXPORTS:
        return _modeling_attr(name)
    if name in _METHOD_EXPORTS:
        module_name, attr_name = _METHOD_EXPORTS[name]
        return getattr(_method_module(module_name), attr_name)
    if name == "METHODS":
        return _method_registry()
    raise AttributeError(name)
