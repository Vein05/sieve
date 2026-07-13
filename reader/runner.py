"""Runner for answer-generation phase 1."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime
from pathlib import Path
import re
from time import perf_counter
import traceback
from typing import Any, Callable

from reader.client import generate_answer
from reader.prompts import _COMPACT_READER_VARIANTS, build_answer_generation_prompt
from reader.scoring import evaluate_run, harmful_use_flags, quality_metrics, quality_metrics_for_row
from compiler.llm_compiler import llm_compile, format_llm_evidence
from reader.selector import (
    SYSTEMS,
    SYSTEM_NOTES,
    _is_answer_bearing_slot_payload,
    build_system_selections,
    load_controller_config,
    load_slice,
    normalize_systems,
    selected_texts_for_system,
)
from controller.logic import token_count


REALISTIC_BASELINES = [
    "naive_top_k",
]

# Thresholds used in routing and confidence decisions.
_MIN_COMPILED_TOKEN_COUNT = 30  # below this, compiler output is too sparse to trust
_MINILM_CONFIDENCE_THRESHOLD = 0.85  # span extraction confidence gate for abstain override

_QUESTION_DATE_PROMPT_RE = re.compile(r"^Question date:\s*(.+)$", re.MULTILINE)
_KV_FRAG_RE = re.compile(r"^(?:user|assistant):\s+.{1,30}\s+(?:name|location|time|state):\s+.+$")
_DATE_PREFIX_RE = re.compile(r"^\s*(\d{4}/\d{2}/\d{2})(?:\s+\([^)]+\))?")
_SOURCE_RUN_QUESTION_DATE_CACHE: dict[str, dict[str, str]] = {}


def _append_progress_log(progress_log_path: Path | None, message: str) -> None:
    if progress_log_path is None:
        return
    stamp = datetime.now().isoformat(timespec="seconds")
    progress_log_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {message}\n")


def _progress_logger(progress_log_path: Path | None) -> Callable[[str], None]:
    def log(message: str) -> None:
        _append_progress_log(progress_log_path, message)

    return log


def _row_split_name(row: dict[str, Any]) -> str:
    return str(row.get("split", "")).strip().lower()


def _filter_eval_rows(rows: list[dict[str, Any]], eval_split: str | None) -> list[dict[str, Any]]:
    if not eval_split:
        return rows
    normalized = eval_split.strip().lower()
    return [row for row in rows if _row_split_name(row) == normalized]


def _is_beam_payload(rows: list[dict[str, Any]]) -> bool:
    return any(str(row.get("dataset_name", "")).strip().lower() == "beam" for row in rows)


def _primary_system_name(summaries: dict[str, dict[str, Any]]) -> str:
    for candidate in ("bm25_sieve", "bm25_top_k"):
        if candidate in summaries:
            return candidate
    for system in summaries:
        if system not in REALISTIC_BASELINES:
            return system
    return next(iter(summaries))


def _headline_pairwise_metric(summary: dict[str, Any], *, is_beam: bool) -> tuple[str, float | None]:
    key = (
        "pairwise_quality_score_vs_naive_top_k"
        if is_beam
        else "constrained_pairwise_score_vs_naive_top_k"
    )
    value = summary.get(key)
    return key, float(value) if value is not None else None


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = line.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _insufficient_answer_mode(row: dict[str, Any]) -> str:
    if str(row.get("harm_type", "")).strip().lower() == "abstention":
        if str(row.get("dataset_name", "")).strip().lower() == "beam":
            return "question_topic_absent_sentence"
        return "chat_absent_sentence"
    return "unknown"


def _extract_question_date_from_prompt(prompt: str | None) -> str | None:
    if not prompt:
        return None
    match = _QUESTION_DATE_PROMPT_RE.search(prompt)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _load_question_dates_from_source_run(source_run: str) -> dict[str, str]:
    cached = _SOURCE_RUN_QUESTION_DATE_CACHE.get(source_run)
    if cached is not None:
        return cached

    path = Path(source_run)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        _SOURCE_RUN_QUESTION_DATE_CACHE[source_run] = {}
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _SOURCE_RUN_QUESTION_DATE_CACHE[source_run] = {}
        return {}

    mapping: dict[str, str] = {}
    traces = payload.get("traces", [])
    if isinstance(traces, list):
        for trace in traces:
            if not isinstance(trace, dict):
                continue
            example_id = str(trace.get("example_id") or "").strip()
            if not example_id or example_id in mapping:
                continue
            prompt = str(trace.get("prompt") or "")
            question_date = _extract_question_date_from_prompt(prompt)
            if question_date:
                mapping[example_id] = question_date

    _SOURCE_RUN_QUESTION_DATE_CACHE[source_run] = mapping
    return mapping


def _infer_date_from_texts(texts: list[str]) -> str | None:
    latest: str | None = None
    for text in texts:
        match = _DATE_PREFIX_RE.match(str(text))
        if not match:
            continue
        value = match.group(1)
        if latest is None or value > latest:
            latest = value
    return latest


def _resolve_question_date(
    row: dict[str, Any],
    selected_memory_texts: list[str],
) -> str | None:
    explicit = str(row.get("question_date") or "").strip()
    if explicit:
        return explicit

    source_run = str(row.get("source_run") or "").strip()
    if source_run:
        by_example = _load_question_dates_from_source_run(source_run)
        recovered = by_example.get(str(row.get("example_id")))
        if recovered:
            return recovered

    active_context = [str(line) for line in row.get("active_context", [])]
    candidate_lines = [
        str(item.get("content", ""))
        for item in row.get("candidate_memories", [])
        if isinstance(item, dict)
    ]
    inferred = _infer_date_from_texts(active_context + selected_memory_texts + candidate_lines)
    return inferred


def _abstention_answer(question: str, insufficient_answer_mode: str) -> str:
    if insufficient_answer_mode == "question_topic_absent_sentence":
        topic = str(question or "").strip().rstrip("?.!") or "the requested information"
        return f"Based on the provided chat, there is no information related to {topic}."
    if insufficient_answer_mode == "chat_absent_sentence":
        return "This information is not available in the chat."
    return "Unknown"


def _deterministic_generation(answer: str) -> dict[str, Any]:
    clean_answer = str(answer or "").strip() or "Unknown"
    return {
        "success": True,
        "status_code": 200,
        "raw_answer": clean_answer,
        "clean_answer": clean_answer,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_ms": 0.0,
        "error": None,
    }


def _trace_compiler_payload(selection_meta: dict[str, Any]) -> dict[str, Any] | None:
    explicit = selection_meta.get("compiler_payload")
    if isinstance(explicit, dict):
        return dict(explicit)

    keys = (
        "compiler_seed_memory_ids",
        "compiled_memory_ids",
        "compiled_unit_count",
        "compiled_token_count",
        "requirements",
        "coverage",
        "requirement_coverage",
        "compiler_sufficient",
        "compiler_expansion_steps",
        "compiler_used_fallback",
        "compiler_compiled_texts",
    )
    payload = {key: selection_meta[key] for key in keys if key in selection_meta}
    return payload or None


def _trace_rendered_evidence_package(selection_meta: dict[str, Any]) -> Any:
    for key in (
        "rendered_evidence_package",
        "evidence_package",
        "structured_evidence_package",
        "rendered_evidence",
    ):
        if key in selection_meta:
            return selection_meta.get(key)
    return None


def _suppressed_evidence_ids(selection_meta: dict[str, Any]) -> tuple[set[str], set[str]]:
    invalid_slots = dict(selection_meta.get("invalid_slots") or {})
    raw_slot_bindings = dict(selection_meta.get("raw_slot_bindings") or {})
    suppressed_unit_ids = {
        str((raw_slot_bindings.get(slot_name) or {}).get("unit_id") or "").strip()
        for slot_name, reason in invalid_slots.items()
        if str(reason) not in {"", "generic_fragment", "missing"}
    }
    suppressed_unit_ids.discard("")
    # Only suppress individual invalid units, not entire memories.
    # Other evidence lines from the same memory may still be useful for the reader.
    return suppressed_unit_ids, set()


def _filtered_rendered_evidence_package(selection_meta: dict[str, Any]) -> dict[str, Any] | None:
    rendered_evidence_package = _trace_rendered_evidence_package(selection_meta)
    if not isinstance(rendered_evidence_package, dict):
        return None
    suppressed_unit_ids, suppressed_memory_ids = _suppressed_evidence_ids(selection_meta)
    filtered_package = dict(rendered_evidence_package)
    evidence_lines = rendered_evidence_package.get("evidence_lines")
    if isinstance(evidence_lines, list):
        deduped_lines: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        for item in evidence_lines:
            if not isinstance(item, dict):
                continue
            unit_id = str(item.get("unit_id") or "").strip()
            memory_id = str(item.get("memory_id") or "").strip()
            text = str(item.get("text") or "").strip()
            if unit_id in suppressed_unit_ids or memory_id in suppressed_memory_ids or not text:
                continue
            dedupe_key = (memory_id, text)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            deduped_lines.append(item)
        filtered_package["evidence_lines"] = deduped_lines
    return filtered_package


_has_answer_bearing_slot = _is_answer_bearing_slot_payload


def _package_has_usable_reader_evidence(
    selection_meta: dict[str, Any],
    rendered_evidence_package: dict[str, Any] | None,
    selected_memory_ids: list[str],
) -> bool:
    compiled_texts = selection_meta.get("compiled_texts")
    if isinstance(compiled_texts, list) and any(str(text).strip() for text in compiled_texts):
        return True
    validated_slot_bindings = selection_meta.get("validated_slot_bindings")
    if isinstance(validated_slot_bindings, dict) and any(
        _has_answer_bearing_slot(payload) for payload in validated_slot_bindings.values()
    ):
        return True
    if not isinstance(rendered_evidence_package, dict):
        return False
    slots = rendered_evidence_package.get("validated_slots") or rendered_evidence_package.get("slots")
    if isinstance(slots, dict) and any(_has_answer_bearing_slot(payload) for payload in slots.values()):
        return True
    evidence_lines = rendered_evidence_package.get("evidence_lines")
    if isinstance(evidence_lines, list):
        return any(isinstance(item, dict) and str(item.get("text") or "").strip() for item in evidence_lines)
    return False



def _unselected_candidate_pool_texts(
    row: dict[str, Any],
    selected_memory_ids: list[str],
) -> list[str]:
    """Return content of candidate pool memories not selected by the compiler, in pool order."""
    candidates = row.get("candidate_memories", [])
    selected_set = {str(mid) for mid in selected_memory_ids}
    unselected: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        mid = str(candidate.get("memory_id", ""))
        content = str(candidate.get("content", "")).strip()
        if mid and mid not in selected_set and content:
            unselected.append(content)
    return unselected


def _reader_variant_for_selection(reader_mode: str, selection_meta: dict[str, Any]) -> str:
    compiler_declared_variant = str(selection_meta.get("reader_prompt_variant") or "").strip()
    if compiler_declared_variant and compiler_declared_variant != "structured_reader_v1":
        return compiler_declared_variant
    return "crisp_compact_v1"


def _compiler_surface_success_from_output(output: dict[str, Any]) -> bool:
    answerability_level = str(output.get("answerability_level") or "").strip().lower()
    return bool(
        int(output.get("compiled_memory_count") or 0) > 0
        or int(output.get("compiled_token_count") or 0) > 0
        or answerability_level in {"reader_only", "deterministic_safe"}
    )


def _package_sufficient_for_reader_from_output(output: dict[str, Any]) -> bool:
    answerability_level = str(output.get("answerability_level") or "").strip().lower()
    if answerability_level == "abstain":
        return False
    if answerability_level == "deterministic_safe":
        return True
    if answerability_level != "reader_only":
        return False

    selection_meta = dict(output.get("selection_meta") or {})
    if bool(selection_meta.get("semantic_sufficiency_applied")) and not bool(selection_meta.get("semantic_sufficiency_passed")):
        return False

    route_reason = str(output.get("reader_route_reason") or "").strip().lower()
    if route_reason in {"no_usable_bindings", "hard_validation_missing"}:
        return False

    rendered_evidence_package = output.get("rendered_evidence_package")
    return _package_has_usable_reader_evidence(
        selection_meta,
        rendered_evidence_package if isinstance(rendered_evidence_package, dict) else None,
        list(output.get("selected_memory_ids") or []),
    )


def _failure_stage_from_output(output: dict[str, Any]) -> str:
    quality_score = int(output.get("quality_score", 0))
    if quality_score >= 1:
        return "success"

    selection_meta = dict(output.get("selection_meta") or {})
    answer_mode = str(
        selection_meta.get("answer_mode")
        or output.get("effective_prompt_variant")
        or ""
    ).strip().lower()

    if answer_mode in {"deterministic_span", "deterministic_numeric"} and bool(output.get("deterministic_allowed")):
        return "deterministic_fast_path"
    if bool(output.get("package_sufficient_for_reader")):
        return "reader"
    if bool(output.get("compiler_surface_success")):
        return "packaging"
    return "retrieval_or_compilation"


def _diagnostic_fields_for_output(output: dict[str, Any]) -> dict[str, Any]:
    unknown_like = _is_unknown_like_answer(str(output.get("generated_answer") or ""))
    package_sufficient = _package_sufficient_for_reader_from_output(output)
    compiler_surface_success = _compiler_surface_success_from_output(output)
    enriched = {
        "compiler_surface_success": int(compiler_surface_success),
        "package_sufficient_for_reader": bool(package_sufficient),
        "unknown_was_correct": bool(unknown_like and int(output.get("quality_score", 0)) >= 1),
    }
    temp_output = dict(output)
    temp_output.update(enriched)
    enriched["failure_stage"] = _failure_stage_from_output(temp_output)
    return enriched


def _budget_fields_for_selection(selection_meta: dict[str, Any]) -> dict[str, Any]:
    budget_window = selection_meta.get("budget_window") or {}
    proposal_budget_window = selection_meta.get("proposal_budget_window") or {}
    return {
        "budget_target_tokens": budget_window.get("target_tokens"),
        "budget_hard_tokens": budget_window.get("hard_tokens"),
        "budget_overshoot_reasons": list(selection_meta.get("budget_overshoot_reasons") or []),
        "proposal_budget_target_tokens": proposal_budget_window.get("target_tokens"),
        "proposal_budget_hard_tokens": proposal_budget_window.get("hard_tokens"),
        "proposal_budget_overshoot_reasons": list(selection_meta.get("proposal_budget_overshoot_reasons") or []),
    }


def _build_capped_generation_prompt(
    *,
    question: str,
    question_date: str | None,
    active_context: list[str],
    retrieved_memories: list[str],
    variant: str,
    harm_type: str,
    insufficient_answer_mode: str,
    query_family: str,
    rendered_evidence_package: dict[str, Any] | None,
    question_type: str,
    external_prompt_cap: int | None,
) -> tuple[str, list[str], dict[str, Any]]:
    deduped_memories = _dedupe_lines(retrieved_memories)
    prompt = build_answer_generation_prompt(
        question=question,
        question_date=question_date,
        active_context=active_context,
        retrieved_memories=deduped_memories,
        variant=variant,
        harm_type=harm_type,
        insufficient_answer_mode=insufficient_answer_mode,
        query_family=query_family,
        rendered_evidence_package=rendered_evidence_package,
        question_type=question_type,
    )
    if external_prompt_cap is None:
        return prompt, deduped_memories, {
            "evidence_budget": None,
            "bare_overhead": None,
            "effective_prompt_cap": None,
            "estimated_prompt_tokens": token_count(prompt),
            "prompt_cap_truncated": False,
            "dropped_memory_count": 0,
        }

    evidence_budget = max(0, int(external_prompt_cap))

    # Cap specifies evidence tokens only. Measure bare prompt overhead per row
    # (system prompt + rules + question, no memories/evidence) and add it, so
    # both systems get exactly `evidence_budget` tokens for their payload.
    # When a rendered_evidence_package is present, it is included in overhead
    # (the compiler's output is "free") and the budget controls only the
    # supplementary retrieved_memories.  For naive/baseline systems where
    # rendered_evidence_package is None, the full budget goes to memories.
    bare_prompt = build_answer_generation_prompt(
        question=question,
        question_date=question_date,
        active_context=active_context,
        retrieved_memories=[],
        variant=variant,
        harm_type=harm_type,
        insufficient_answer_mode=insufficient_answer_mode,
        query_family=query_family,
        rendered_evidence_package=rendered_evidence_package,
        question_type=question_type,
    )
    bare_overhead = token_count(bare_prompt)
    requested_cap = bare_overhead + evidence_budget
    kept_memories = list(deduped_memories)
    dropped_memory_count = 0
    while kept_memories and token_count(prompt) > requested_cap:
        kept_memories.pop()
        dropped_memory_count += 1
        prompt = build_answer_generation_prompt(
            question=question,
            question_date=question_date,
            active_context=active_context,
            retrieved_memories=kept_memories,
            variant=variant,
            harm_type=harm_type,
            insufficient_answer_mode=insufficient_answer_mode,
            query_family=query_family,
            rendered_evidence_package=rendered_evidence_package,
            question_type=question_type,
        )
    return prompt, kept_memories, {
        "evidence_budget": evidence_budget,
        "bare_overhead": bare_overhead,
        "effective_prompt_cap": requested_cap,
        "estimated_prompt_tokens": token_count(prompt),
        "prompt_cap_truncated": dropped_memory_count > 0,
        "dropped_memory_count": dropped_memory_count,
    }


_HEAVY_SELECTION_META_KEYS = frozenset({
    "compiled_texts",
    "compiler_compiled_texts",
    "compiler_projected_candidates",
    "bound_evidence_package",
    "requirements",
    "coverage",
    "requirement_coverage",
    "raw_slot_bindings",
    "compiler_expansion_steps",
    "proposal_debug",
    "_perf",
})


def _slim_selection_meta(selection_meta: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip heavy debug-only keys from selection_meta to reduce memory."""
    if not isinstance(selection_meta, dict):
        return selection_meta
    return {k: v for k, v in selection_meta.items() if k not in _HEAVY_SELECTION_META_KEYS}


def _finalize_output(
    *,
    output: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = _diagnostic_fields_for_output(output)
    output.update(diagnostics)
    trace.update(diagnostics)
    # Slim selection_meta in output — trace keeps the full version for debugging.
    output["selection_meta"] = _slim_selection_meta(output.get("selection_meta"))
    output["trace"] = trace
    return output


def _should_cascade_to_llm(
    compiler_payload: dict[str, Any] | None,
    rendered_evidence_package: dict[str, Any] | None,
    selection_meta: dict[str, Any] | None,
) -> bool:
    """Decide whether the rule-based compiler output is insufficient.

    Uses the compiler's own sufficiency signals — no dataset-specific
    heuristics.  Three tiers:

    1. **Structural failure** — compiler produced nothing usable
       (no bindings, empty multi-session, near-zero tokens).
    2. **Grounding failure** — schema matched but slots couldn't be
       filled from evidence (partial grounding, hard validation missing).
    3. **Semantic insufficiency** — compiler's own sufficiency check
       said the evidence is too thin to answer confidently.
    """
    sm = selection_meta or {}
    cp = compiler_payload or {}
    rep = rendered_evidence_package or {}

    # Tier 1: structural — compiler produced (near-)nothing
    compiled_tok = cp.get("compiled_token_count", 0)
    if compiled_tok < _MIN_COMPILED_TOKEN_COUNT:
        return True

    route_reason = str(sm.get("reader_route_reason") or "").strip().lower()
    if route_reason in {
        "no_usable_bindings",
        "hard_validation_missing",
    }:
        return True

    # Multi-session rows use naive raw memories — compilation loses items
    # the reader needs to count.  Skip cascade entirely for these.
    if route_reason == "multi_session_reader":
        return False
    if str(rep.get("mode", "")) == "multi_session_fact_list":
        return False

    # Tier 2: grounding failure — schema matched but evidence is partial
    if route_reason == "schema_grounding_partial":
        return True

    answerability = str(sm.get("answerability_level") or "").strip().lower()
    if answerability == "abstain":
        return True

    # Tier 3: semantic insufficiency — compiler says evidence is thin
    if bool(sm.get("semantic_sufficiency_applied")) and not bool(sm.get("semantic_sufficiency_passed")):
        return True

    return False


def _run_phase1_step(
    *,
    row: dict[str, Any],
    system: str,
    system_selection: dict[str, Any],
    provider: str,
    base_url: str,
    model: str,
    timeout_s: float,
    max_tokens: int,
    temperature: float,
    api_key_env: str,
    app_url: str,
    app_title: str,
    prompt_variant: str,
    sieve_reader_mode: str = "calibrated_default",
    external_prompt_cap: int | None = None,
    provider_routing: dict[str, Any] | None,
    pool_augmentation: bool = True,
    cascade_compiler_model: str | None = None,
    cascade_compiler_top_k: int = 5,
    cascade_compiler_provider: str = "openrouter",
    cascade_compiler_base_url: str = "https://openrouter.ai/api/v1",
) -> dict[str, Any]:
    active_context = [str(line) for line in row.get("active_context", [])]
    if not system_selection or not isinstance(system_selection, dict):
        raise ValueError(f"system_selection is {type(system_selection)} for {row.get('example_id')}")
    selected_memory_ids = list(system_selection.get("selected_memory_ids") or [])
    selection_meta = dict(system_selection.get("selection_meta") or {})
    compiler_payload = _trace_compiler_payload(selection_meta)
    rendered_evidence_package = _filtered_rendered_evidence_package(selection_meta)
    selected_memory_texts = selected_texts_for_system(row, system_selection)
    question_date = _resolve_question_date(row, selected_memory_texts)
    insufficient_answer_mode = _insufficient_answer_mode(row)
    compiler_authoritative = bool(selection_meta.get("compiler_authoritative")) or (
        str(selection_meta.get("selection_mode") or "").strip().lower() == "evidence_compiler"
    )
    answer_mode = str(selection_meta.get("answer_mode") or "").strip().lower()
    answer_text = str(selection_meta.get("answer_text") or "").strip()
    answerability_level = str(selection_meta.get("answerability_level") or "").strip().lower()
    _det_base = bool(selection_meta.get("deterministic_allowed")) or answerability_level == "deterministic_safe"
    # Disable deterministic fast-path for question types where the compiler
    # systematically picks stale values (temporal state confusion).  The
    # reader with full compiled evidence handles recency far better.
    _qt_str = str(row.get("question_type", "")).strip()
    deterministic_allowed = _det_base and _qt_str not in ("knowledge-update", "temporal-reasoning")
    reader_route_reason = str(selection_meta.get("reader_route_reason") or "").strip()
    effective_prompt_variant = prompt_variant
    prompt = ""
    generation = None
    abstain_rerouted_to_reader = False
    # --- Preference question bypass: skip compiler, use raw memories ---
    is_preference_question = str(row.get("question_type", "")).strip() == "single-session-preference"
    # --- Multi-session bypass: compilation loses items the reader needs to
    #     count.  Use naive-style raw memories instead.  Reader does better
    #     counting from full context than from any compiled/summarized form. ---
    is_multi_session = str(row.get("question_type", "")).strip() == "multi-session"
    if is_preference_question and str(system).strip().lower() in ("sieve", "bm25_sieve") and selected_memory_texts:
        effective_prompt_variant = "preference_reader_v1"
        reader_route_reason = "preference_bypass"
        rendered_evidence_package = {}  # empty dict — preference prompt ignores compiled evidence
        # Augment with full candidate pool so preference reader has all context
        if pool_augmentation:
            unselected_pool = _unselected_candidate_pool_texts(row, selected_memory_ids)
            if unselected_pool:
                selected_memory_texts = selected_memory_texts + unselected_pool
    elif is_multi_session and str(system).strip().lower() in ("sieve", "bm25_sieve"):
        # Multi-session: naive dump.  Every compilation approach tested
        # (nano summarize, NLP filter, rule-based objects) loses countable
        # items and hurts accuracy.  Raw candidate dump is the ceiling.
        effective_prompt_variant = "baseline"
        reader_route_reason = "multi_session_naive_dump"
        rendered_evidence_package = None
        all_pool = _unselected_candidate_pool_texts(row, [])
        if all_pool:
            selected_memory_texts = all_pool
    elif answer_mode == "abstain" and answerability_level == "abstain":
        has_evidence = _package_has_usable_reader_evidence(
            selection_meta,
            rendered_evidence_package if isinstance(rendered_evidence_package, dict) else None,
            selected_memory_ids,
        )
        if has_evidence and str(system).strip().lower() in ("sieve", "bm25_sieve"):
            effective_prompt_variant = "crisp_compact_v1"
            abstain_rerouted_to_reader = True
        else:
            effective_prompt_variant = "compiler_abstain"
            generation = _deterministic_generation(
                _abstention_answer(str(row["query"]), insufficient_answer_mode)
            )
    elif answer_mode in {"deterministic_span", "deterministic_numeric"} and answer_text and deterministic_allowed:
        effective_prompt_variant = answer_mode
        generation = _deterministic_generation(answer_text)
    # --- Deterministic temporal hint: if computed_date_hint has a valid
    #     numeric answer, use it directly instead of risking reader error.
    #     Only for RelativeTime schema where hints are reliable (single event
    #     vs question date). TemporalInterval hints often pick wrong events. ---
    _schema_for_hint = str(selection_meta.get("schema_name") or "").strip()
    if (
        generation is None
        and str(system).strip().lower() in ("sieve", "bm25_sieve")
        and isinstance(rendered_evidence_package, dict)
        and _schema_for_hint == "RelativeTime"
    ):
        _hint_slots = rendered_evidence_package.get("validated_slots") or rendered_evidence_package.get("slots") or {}
        _hint_val = _hint_slots.get("computed_date_hint", {})
        _hint_text = str(_hint_val.get("display_text") or _hint_val.get("text") or "") if isinstance(_hint_val, dict) else ""
        if _hint_text and "Difference:" in _hint_text:
            _diff_match = re.search(r"Difference:\s*(\d+)\s*days?\s*\((\d+(?:\.\d+)?)\s*weeks?,\s*(\d+(?:\.\d+)?)\s*months?\)", _hint_text)
            if _diff_match:
                _days = int(_diff_match.group(1))
                _weeks = round(float(_diff_match.group(2)))
                _months = round(float(_diff_match.group(3)))
                _q_lower = str(row.get("query") or "").strip().lower()
                # Pick the right unit based on what the question asks
                if "month" in _q_lower:
                    _temporal_answer = f"{_months} months ago" if "ago" in _q_lower else str(_months)
                elif "week" in _q_lower:
                    _temporal_answer = f"{_weeks} weeks ago" if "ago" in _q_lower else str(_weeks)
                elif "day" in _q_lower and ("between" in _q_lower or "before" in _q_lower or "after" in _q_lower or "passed" in _q_lower):
                    _temporal_answer = f"{_days} days"
                else:
                    _temporal_answer = ""
                if _temporal_answer:
                    effective_prompt_variant = "deterministic_temporal_hint"
                    reader_route_reason = "deterministic_temporal_hint"
                    generation = _deterministic_generation(_temporal_answer)
    is_preference_bypass = effective_prompt_variant == "preference_reader_v1"
    # Confidence gate: if compiled evidence is only garbled KV fragments,
    # fallback to baseline reader with raw retrieved memories.
    _compiled_token_count = int((compiler_payload or {}).get("compiled_token_count") or 0)
    _confidence_gate_triggered = False
    if (
        not is_preference_bypass
        and str(system).strip().lower() in ("sieve", "bm25_sieve")
        and _compiled_token_count < _MIN_COMPILED_TOKEN_COUNT
        and selected_memory_texts
        and generation is None
    ):
        _all_kv = True
        for _mem in selected_memory_texts:
            _is_kv = any(
                _KV_FRAG_RE.match(ln.strip()) and len(ln.strip()) < 100
                for ln in _mem.split("|")
            )
            if not (_is_kv and len(_mem) < 100):
                _all_kv = False
                break
        if _all_kv:
            _confidence_gate_triggered = True
            effective_prompt_variant = "baseline"
            rendered_evidence_package = None
            reader_route_reason = "confidence_gate_fallback"
            # Expand to full candidate pool so the reader has more context
            if pool_augmentation:
                _extra = _unselected_candidate_pool_texts(row, selected_memory_ids)
                if _extra:
                    selected_memory_texts = selected_memory_texts + _extra
    is_multi_session_bypass = reader_route_reason in {"multi_session_naive_bypass", "multi_session_summarize", "multi_session_naive_dump"}
    if generation is None and not abstain_rerouted_to_reader:
        if (
            not is_preference_bypass
            and not is_multi_session_bypass
            and str(system).strip().lower() in ("sieve", "bm25_sieve")
            and compiler_authoritative
            and answer_mode in {"reader_from_slots", "deterministic_span", "deterministic_numeric", "abstain"}
        ):
            effective_prompt_variant = _reader_variant_for_selection(sieve_reader_mode, selection_meta)
        # Cascading compiler: when rule-based produces insufficient evidence,
        # call LLM compiler on the raw candidates.  Skips preference rows
        # (handled by their own reader path) and only replaces evidence
        # when the LLM call succeeds.
        if (
            cascade_compiler_model
            and str(system).strip().lower() in ("sieve", "bm25_sieve")
            and generation is None
            and not is_preference_bypass
            and _should_cascade_to_llm(compiler_payload, rendered_evidence_package, selection_meta)
        ):
            _query_text = str(row["query"])
            _cascade_kwargs = dict(
                query=_query_text,
                candidates=row.get("candidate_memories", []),
                model=cascade_compiler_model,
                provider=cascade_compiler_provider,
                base_url=cascade_compiler_base_url,
                api_key_env=api_key_env,
                app_url=app_url,
                app_title=app_title,
                question_date=question_date,
            )
            llm_evidence = llm_compile(top_k=cascade_compiler_top_k, **_cascade_kwargs)
            if llm_evidence.get("success"):
                rendered_evidence_package = format_llm_evidence(llm_evidence)
                selection_meta["compiler_type"] = "llm"
                selection_meta["compiler_model"] = cascade_compiler_model
                selection_meta["llm_compiler_input_tokens"] = llm_evidence.get("compiler_input_tokens", 0)
                selection_meta["llm_compiler_output_tokens"] = llm_evidence.get("compiler_output_tokens", 0)
                effective_prompt_variant = _reader_variant_for_selection(sieve_reader_mode, selection_meta)
        prompt, selected_memory_texts, prompt_cap_meta = _build_capped_generation_prompt(
            question=str(row["query"]),
            question_date=question_date,
            active_context=active_context,
            retrieved_memories=selected_memory_texts,
            variant=effective_prompt_variant,
            harm_type=str(row.get("harm_type") or ""),
            insufficient_answer_mode=insufficient_answer_mode,
            query_family=str(selection_meta.get("query_family") or ""),
            rendered_evidence_package=(
                rendered_evidence_package if isinstance(rendered_evidence_package, dict) else None
            ),
            question_type=str(row.get("question_type") or ""),
            external_prompt_cap=external_prompt_cap,
        )
        selection_meta["prompt_cap_meta"] = prompt_cap_meta
        generation = generate_answer(
            provider=provider,
            base_url=base_url,
            model=model,
            prompt=prompt,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key_env=api_key_env,
            app_url=app_url,
            app_title=app_title,
            provider_routing=provider_routing,
        )
    if generation is None and abstain_rerouted_to_reader:
        # Budget fill for abstain-rerouted rows is also handled by the compiler pipeline.
        prompt, selected_memory_texts, prompt_cap_meta = _build_capped_generation_prompt(
            question=str(row["query"]),
            question_date=question_date,
            active_context=active_context,
            retrieved_memories=selected_memory_texts,
            variant=effective_prompt_variant,
            harm_type=str(row.get("harm_type") or ""),
            insufficient_answer_mode=insufficient_answer_mode,
            query_family=str(selection_meta.get("query_family") or ""),
            rendered_evidence_package=(
                rendered_evidence_package if isinstance(rendered_evidence_package, dict) else None
            ),
            question_type=str(row.get("question_type") or ""),
            external_prompt_cap=external_prompt_cap,
        )
        selection_meta["prompt_cap_meta"] = prompt_cap_meta
        generation = generate_answer(
            provider=provider,
            base_url=base_url,
            model=model,
            prompt=prompt,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key_env=api_key_env,
            app_url=app_url,
            app_title=app_title,
            provider_routing=provider_routing,
        )
    if generation is None:
        generation = {"success": False, "raw_answer": "", "clean_answer": "", "error": "generation_returned_none"}
    if generation.get("status_code") == 402:
        raise RuntimeError(f"API credit limit reached (402) for {row.get('example_id')}")
    # MiniLM abstain override: when reader outputs "Unknown" but compiled
    # evidence exists, attempt span extraction as a fallback.
    _clean = (generation.get("clean_answer") or "").strip().lower()
    if _clean in ("", "unknown", "unknown.") and selected_memory_texts and str(system).strip().lower() in ("sieve", "bm25_sieve"):
        try:
            from compiler.adapters.minilm_qa_adapter import extract_span as _minilm_extract
            _evidence_concat = " ".join(str(t) for t in selected_memory_texts)[:800]
            _span = _minilm_extract(str(row.get("query", "")), _evidence_concat)
            if _span.answer and _span.confidence >= _MINILM_CONFIDENCE_THRESHOLD:
                generation = dict(generation)
                generation["clean_answer"] = _span.answer
                generation["raw_answer"] = _span.answer
        except Exception:
            pass  # MiniLM is optional; fall through to original answer

    quality = quality_metrics_for_row(row, str(row["query"]), generation["clean_answer"], str(row["reference_answer"]))
    flags = harmful_use_flags(
        row=row,
        generated_answer=generation["clean_answer"],
        selected_memory_ids=selected_memory_ids,
        quality_score=int(quality["quality_score"]),
    )
    gold_selection_ids = row.get("gold_decision", {}).get("selected_memory_ids")
    gold_selection_available = isinstance(gold_selection_ids, list) and (
        bool(gold_selection_ids)
        or str(row.get("harm_type", "")).strip().lower() == "abstention"
        or str(row.get("evidence_sufficiency", {}).get("label", "")).strip().lower() == "abstention"
    )
    gold_selected = (
        {str(memory_id) for memory_id in gold_selection_ids}
        if gold_selection_available and isinstance(gold_selection_ids, list)
        else None
    )
    output = {
        "system": system,
        "example_id": row["example_id"],
        "harm_type": row.get("harm_type"),
        "dataset_name": row.get("dataset_name"),
        "question_type": row.get("question_type"),
        "locomo_category": row.get("locomo_category"),
        "ability": row.get("ability"),
        "chat_size": row.get("chat_size"),
        "chat_id": row.get("chat_id"),
        "question": row["query"],
        "question_date": question_date,
        "reference_answer": row.get("reference_answer"),
        "active_context": active_context,
        "selected_memory_ids": selected_memory_ids,
        "selected_memory_count": len(selected_memory_ids),
        "selected_memory_tokens": sum(token_count(text) for text in selected_memory_texts),
        "gold_memory_covered": (
            int(gold_selected.issubset(selected_memory_ids))
            if gold_selected is not None
            else None
        ),
        "extra_non_gold_memory_used": (
            int(bool(set(selected_memory_ids) - gold_selected))
            if gold_selected is not None
            else None
        ),
        "selection_meta": selection_meta,
        "compiler_payload": compiler_payload,
        "rendered_evidence_package": rendered_evidence_package,
        "effective_prompt_variant": effective_prompt_variant,
        "query_family": selection_meta.get("query_family"),
        "selection_mode": selection_meta.get("selection_mode"),
        "selection_margin": selection_meta.get("selection_margin"),
        "reason_codes": list(selection_meta.get("reason_codes", [])),
        "anchor_count": selection_meta.get("anchor_count"),
        "answerability_level": answerability_level,
        "deterministic_allowed": deterministic_allowed,
        "reader_route_reason": reader_route_reason,
        "proposal_pool_size": selection_meta.get("proposal_pool_size"),
        "proposal_rank_cutoff": selection_meta.get("proposal_rank_cutoff"),
        "candidate_pool_augmented": selection_meta.get("candidate_pool_augmented"),
        "compiled_memory_count": selection_meta.get("compiled_memory_count"),
        "compiled_token_count": selection_meta.get("compiled_token_count"),
        "schema_name": selection_meta.get("schema_name"),
        "question_subtype": selection_meta.get("question_subtype"),
        "compiler_type": selection_meta.get("compiler_type", "rule_based"),
        "compiler_model": selection_meta.get("compiler_model"),
        "llm_compiler_input_tokens": selection_meta.get("llm_compiler_input_tokens", 0),
        "llm_compiler_output_tokens": selection_meta.get("llm_compiler_output_tokens", 0),
        "generated_answer": generation["clean_answer"],
        "raw_answer": generation["raw_answer"],
        "prompt_tokens": generation["prompt_tokens"],
        "completion_tokens": generation["completion_tokens"],
        "latency_ms": generation["latency_ms"],
        "success": generation["success"],
        **_budget_fields_for_selection(selection_meta),
        **quality,
        **flags,
    }
    trace = {
        "system": system,
        "example_id": row["example_id"],
        "dataset_name": row.get("dataset_name"),
        "harm_type": row.get("harm_type"),
        "ability": row.get("ability"),
        "chat_size": row.get("chat_size"),
        "chat_id": row.get("chat_id"),
        "question": row["query"],
        "question_date": question_date,
        "reference_answer": row.get("reference_answer"),
        "prompt_variant": effective_prompt_variant,
        "insufficient_answer_mode": insufficient_answer_mode,
        "selection_empty": not bool(selected_memory_ids),
        "active_context": active_context,
        "selected_memory_ids": selected_memory_ids,
        "selected_memory_texts": selected_memory_texts,
        "selection_meta": selection_meta,
        "compiler_payload": compiler_payload,
        "rendered_evidence_package": rendered_evidence_package,
        "query_family": selection_meta.get("query_family"),
        "selection_mode": selection_meta.get("selection_mode"),
        "selection_margin": selection_meta.get("selection_margin"),
        "reason_codes": list(selection_meta.get("reason_codes", [])),
        "anchor_count": selection_meta.get("anchor_count"),
        "answerability_level": answerability_level,
        "deterministic_allowed": deterministic_allowed,
        "reader_route_reason": reader_route_reason,
        "proposal_pool_size": selection_meta.get("proposal_pool_size"),
        "proposal_rank_cutoff": selection_meta.get("proposal_rank_cutoff"),
        "candidate_pool_augmented": selection_meta.get("candidate_pool_augmented"),
        "compiled_memory_count": selection_meta.get("compiled_memory_count"),
        "compiled_token_count": selection_meta.get("compiled_token_count"),
        "schema_name": selection_meta.get("schema_name"),
        "question_subtype": selection_meta.get("question_subtype"),
        "compiler_type": selection_meta.get("compiler_type", "rule_based"),
        "compiler_model": selection_meta.get("compiler_model"),
        "llm_compiler_input_tokens": selection_meta.get("llm_compiler_input_tokens", 0),
        "llm_compiler_output_tokens": selection_meta.get("llm_compiler_output_tokens", 0),
        **_budget_fields_for_selection(selection_meta),
        "prompt": prompt,
        "generation": {
            "success": generation["success"],
            "status_code": generation.get("status_code"),
            "prompt_tokens": generation["prompt_tokens"],
            "completion_tokens": generation["completion_tokens"],
            "latency_ms": generation["latency_ms"],
            "raw_answer": generation["raw_answer"],
            "clean_answer": generation["clean_answer"],
            "error": generation.get("error"),
        },
        "quality": quality,
        "flags": flags,
    }
    return _finalize_output(output=output, trace=trace)


def run_phase1(
    *,
    slice_path: Path,
    controller_config_path: Path,
    provider: str,
    base_url: str,
    model: str,
    timeout_s: float,
    max_tokens: int,
    temperature: float,
    parallelism: int = 8,
    api_key_env: str = "OPENROUTER_API_KEY",
    app_url: str = "https://anonymous.4open.science/r/from-reliable-to-random-BB62",
    app_title: str = "anonymous-rag-compression-artifact",
    prompt_variant: str = "baseline",
    sieve_reader_mode: str = "calibrated_default",
    max_examples: int | None = None,
    eval_split: str | None = None,
    systems: list[str] | None = None,
    training_slice_paths: list[str] | None = None,
    training_rows: list[dict[str, Any]] | None = None,
    provider_routing: dict[str, Any] | None = None,
    external_prompt_cap: int | None = None,
    progress_log_path: Path | None = None,
    pool_augmentation: bool = True,
    cascade_compiler_model: str | None = None,
    cascade_compiler_top_k: int = 5,
    cascade_compiler_provider: str = "openrouter",
    cascade_compiler_base_url: str = "https://openrouter.ai/api/v1",
) -> dict[str, Any]:
    log_progress = _progress_logger(progress_log_path)
    run_started = perf_counter()
    log_progress(
        "run_phase1:start "
        f"slice={slice_path} provider={provider} model={model} parallelism={parallelism}"
    )
    rows = load_slice(slice_path)
    log_progress(f"run_phase1:loaded_slice rows={len(rows)}")
    rows = _filter_eval_rows(rows, eval_split)
    log_progress(f"run_phase1:filtered_rows rows={len(rows)} eval_split={eval_split or 'all'}")
    if max_examples is not None:
        rows = rows[:max_examples]
        log_progress(f"run_phase1:max_examples rows={len(rows)}")
    controller_config = load_controller_config(controller_config_path)
    selected_systems = normalize_systems(systems)
    log_progress(f"run_phase1:systems systems={','.join(selected_systems)}")
    selection_started = perf_counter()
    log_progress("run_phase1:build_system_selections:start")
    selections = build_system_selections(
        rows,
        controller_config,
        systems=selected_systems,
        slice_path=slice_path,
        training_slice_paths=training_slice_paths,
        training_rows=training_rows,
        external_prompt_cap=external_prompt_cap,
        progress_callback=log_progress,
    )
    log_progress(
        "run_phase1:build_system_selections:done "
        f"elapsed_s={perf_counter() - selection_started:.3f}"
    )

    total_steps = len(rows) * len(selected_systems)
    tasks: list[tuple[int, dict[str, Any], str, dict[str, Any]]] = []
    current_step = 0
    for row in rows:
        for system in selected_systems:
            current_step += 1
            tasks.append((current_step, row, system, selections[row["example_id"]][system]))
    del selections  # Release ~2-5MB per SIEVE row; tasks holds refs to what's needed

    outputs_by_index: dict[int, dict[str, Any]] = {}
    worker_count = max(1, min(parallelism, total_steps))
    log_progress(
        "run_phase1:generation:start "
        f"total_steps={total_steps} worker_count={worker_count}"
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                _run_phase1_step,
                row=row,
                system=system,
                system_selection=system_selection,
                provider=provider,
                base_url=base_url,
                model=model,
                timeout_s=timeout_s,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key_env=api_key_env,
                app_url=app_url,
                app_title=app_title,
                prompt_variant=prompt_variant,
                sieve_reader_mode=sieve_reader_mode,
                external_prompt_cap=external_prompt_cap,
                provider_routing=provider_routing,
                pool_augmentation=pool_augmentation,
                cascade_compiler_model=cascade_compiler_model,
                cascade_compiler_top_k=cascade_compiler_top_k,
                cascade_compiler_provider=cascade_compiler_provider,
                cascade_compiler_base_url=cascade_compiler_base_url,
            ): (index, system, row["example_id"])
            for index, row, system, system_selection in tasks
        }
        task_indices = [index for index, _, _, _ in tasks]
        del tasks  # Release refs to rows and system_selections
        completed = 0
        for future in as_completed(future_map):
            index, system, example_id = future_map.pop(future)
            try:
                result = future.result()
            except Exception as exc:
                traceback.print_exc()
                print(f"SKIPPING {system} :: {example_id}: {exc!r}", flush=True)
                completed += 1
                continue
            completed += 1
            print(f"[{completed:03d}/{total_steps:03d}] {system} :: {example_id}", flush=True)
            log_progress(f"run_phase1:step [{completed:03d}/{total_steps:03d}] {system} :: {example_id}")
            outputs_by_index[index] = result

    outputs = [outputs_by_index[index] for index in task_indices if index in outputs_by_index]
    traces = [output["trace"] for output in outputs]

    scoring_started = perf_counter()
    log_progress("run_phase1:evaluate:start")
    summaries = evaluate_run(outputs)
    log_progress(f"run_phase1:evaluate:done elapsed_s={perf_counter() - scoring_started:.3f}")
    present_baselines = [name for name in REALISTIC_BASELINES if name in summaries]
    has_realistic_baseline = bool(present_baselines)
    best_baseline_quality = max(
        (summaries[name]["avg_quality_score"] for name in present_baselines),
        default=0.0,
    )
    primary_system = _primary_system_name(summaries)
    primary_summary = summaries[primary_system]
    naive_top_k = summaries.get("naive_top_k", primary_summary)
    is_beam = _is_beam_payload(rows)
    headline_pairwise_key, headline_pairwise_value = _headline_pairwise_metric(
        primary_summary,
        is_beam=is_beam,
    )
    if not has_realistic_baseline:
        success = False
        success_note = (
            "No realistic baseline was run, so this result is diagnostic only and should not be treated as a success call."
        )
    elif "naive_top_k" in summaries and headline_pairwise_value is not None:
        success = headline_pairwise_value >= 0.5
        if is_beam:
            success_note = (
                "With naive_top_k present, the default success call uses pairwise quality score vs naive top-k. "
                "A value of at least 0.5 means the method is at least tie-level matchwise while the prompt budget can still be inspected separately."
            )
        else:
            success_note = (
                "With naive_top_k present, the default success call uses constrained pairwise score vs naive top-k. "
                "A value of at least 0.5 means the method is at least tie-level matchwise without safety regressions counting as wins."
            )
    elif is_beam:
        success = (
            primary_summary["avg_quality_score"] >= best_baseline_quality
            and primary_summary["avg_prompt_tokens"] <= max(
                (summaries[name]["avg_prompt_tokens"] for name in present_baselines),
                default=primary_summary["avg_prompt_tokens"],
            )
        )
        success_note = (
            "Success here means answer quality stays at least as strong as the best realistic baseline while using no more prompt tokens. "
            "BEAM transfer rows do not use the harm-flag safety buckets from the curated controller slices."
        )
    else:
        success = (
            primary_summary["avg_quality_score"] >= best_baseline_quality
            and primary_summary["stale_memory_use_rate"] <= naive_top_k["stale_memory_use_rate"]
            and primary_summary["contradictory_memory_use_rate"]
            <= naive_top_k["contradictory_memory_use_rate"]
            and primary_summary["unnecessary_personalization_rate"]
            <= naive_top_k["unnecessary_personalization_rate"]
            and primary_summary["avg_prompt_tokens"] <= max(
                (summaries[name]["avg_prompt_tokens"] for name in present_baselines),
                default=primary_summary["avg_prompt_tokens"],
            )
        )
        success_note = (
            "Success here means answer quality stays at least as strong as the best realistic baseline while harmful memory use remains lower."
        )

    payload = {
        "generated_at": datetime.now().isoformat(),
        "slice_path": str(slice_path),
        "controller_config_path": str(controller_config_path),
        "systems": selected_systems,
        "dataset_names": sorted(
            {
                str(row.get("dataset_name")).strip()
                for row in rows
                if str(row.get("dataset_name", "")).strip()
            }
        ),
        "generator_provider": provider,
        "generator_model": model,
        "generator_base_url": base_url,
        "provider_routing": provider_routing,
        "external_prompt_cap": external_prompt_cap,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout_s": timeout_s,
        "prompt_variant": prompt_variant,
        "sieve_reader_mode": sieve_reader_mode,
        "pool_augmentation": pool_augmentation,
        "eval_split": eval_split,
        "row_count": len(rows),
        "parallelism": worker_count,
        "outputs": outputs,
        "traces": traces,
        "summaries": summaries,
        "success_call": {
            "answer_generation_phase_success": success,
            "has_realistic_baseline": has_realistic_baseline,
            "best_realistic_baseline_quality": (
                best_baseline_quality if has_realistic_baseline else None
            ),
            "primary_system": primary_system,
            "primary_system_quality": primary_summary["avg_quality_score"],
            "primary_system_pairwise_metric": headline_pairwise_key if "naive_top_k" in summaries else None,
            "primary_system_pairwise_score": headline_pairwise_value if "naive_top_k" in summaries else None,
            "note": success_note,
        },
    }
    log_progress(f"run_phase1:done elapsed_s={perf_counter() - run_started:.3f}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_trace_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trace_payload = {
        "generated_at": payload.get("generated_at"),
        "slice_path": payload.get("slice_path"),
        "controller_config_path": payload.get("controller_config_path"),
        "generator_provider": payload.get("generator_provider"),
        "generator_model": payload.get("generator_model"),
        "prompt_variant": payload.get("prompt_variant"),
        "sieve_reader_mode": payload.get("sieve_reader_mode"),
        "systems": payload.get("systems"),
        "traces": payload.get("traces", []),
    }
    path.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")


def _is_unknown_like_answer(answer: str) -> bool:
    normalized = str(answer or "").strip().lower()
    return normalized in {
        "",
        "unknown",
        "i don't know",
        "i do not know",
        "not enough information",
        "this information is not available in the chat.",
    } or normalized.startswith("based on the provided chat, there is no information related to ")


def _system_rows(payload: dict[str, Any], system: str) -> list[dict[str, Any]]:
    return [row for row in payload.get("outputs", []) if str(row.get("system")) == system]


def _failure_accounting(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "rows": len(items),
        "failed_rows": 0,
        "unknown_answers": 0,
        "correct_unknown": 0,
        "false_unknown": 0,
        "gold_covered_but_wrong": 0,
        "reader_failures": 0,
        "deterministic_failures": 0,
        "abstain_failures": 0,
        "compiler_surface_success": 0,
        "package_sufficient_rows": 0,
        "package_insufficient_failures": 0,
        "package_sufficient_reader_failures": 0,
    }
    for item in items:
        quality_score = int(item.get("quality_score", 0))
        answer = str(item.get("generated_answer") or "")
        answerability_level = str(item.get("answerability_level") or "").strip().lower()
        selection_meta = item.get("selection_meta") or {}
        answer_mode = str(selection_meta.get("answer_mode") or item.get("effective_prompt_variant") or "").strip().lower()
        gold_covered = item.get("gold_memory_covered")
        unknown_like = _is_unknown_like_answer(answer)
        compiler_surface_success = bool(item.get("compiler_surface_success"))
        package_sufficient = bool(item.get("package_sufficient_for_reader"))
        if compiler_surface_success:
            counts["compiler_surface_success"] += 1
        if package_sufficient:
            counts["package_sufficient_rows"] += 1
        if bool(item.get("unknown_was_correct")):
            counts["correct_unknown"] += 1
        if quality_score >= 1:
            continue
        counts["failed_rows"] += 1
        if unknown_like:
            counts["unknown_answers"] += 1
        if unknown_like and gold_covered == 1:
            counts["false_unknown"] += 1
        if gold_covered == 1:
            counts["gold_covered_but_wrong"] += 1
        if not package_sufficient:
            counts["package_insufficient_failures"] += 1
        elif answerability_level == "reader_only":
            counts["package_sufficient_reader_failures"] += 1
        if answerability_level == "reader_only":
            counts["reader_failures"] += 1
        elif answerability_level == "abstain":
            counts["abstain_failures"] += 1
        if answer_mode in {"deterministic_span", "deterministic_numeric"}:
            counts["deterministic_failures"] += 1
    return counts


def _comparative_failure_accounting(
    primary_items: list[dict[str, Any]],
    baseline_items: list[dict[str, Any]],
) -> dict[str, int]:
    baseline_by_id = {str(item.get("example_id")): item for item in baseline_items}
    counts = {
        "baseline_failed_rows": 0,
        "baseline_failed_sieve_gold_covered": 0,
        "baseline_failed_sieve_partial_or_better": 0,
        "baseline_failed_sieve_correct": 0,
        "baseline_failed_sieve_package_sufficient": 0,
        "baseline_package_insufficient_sieve_package_sufficient": 0,
        "baseline_failed_sieve_used_fewer_prompt_tokens": 0,
        "both_package_sufficient_sieve_used_fewer_prompt_tokens": 0,
        "both_failed_sieve_failed_later_stage": 0,
        "sieve_failed_baseline_succeeded": 0,
    }
    stage_rank = {
        "retrieval_or_compilation": 0,
        "packaging": 1,
        "reader": 2,
        "deterministic_fast_path": 3,
        "rubric_or_eval_gap": 4,
        "success": 5,
    }
    for item in primary_items:
        example_id = str(item.get("example_id"))
        baseline = baseline_by_id.get(example_id)
        if not baseline:
            continue
        baseline_failed = int(baseline.get("quality_score", 0)) == 0
        sieve_failed = int(item.get("quality_score", 0)) == 0
        baseline_succeeded = int(baseline.get("quality_score", 0)) >= 1
        if baseline_failed:
            counts["baseline_failed_rows"] += 1
            if item.get("gold_memory_covered") == 1:
                counts["baseline_failed_sieve_gold_covered"] += 1
            if bool(item.get("package_sufficient_for_reader")):
                counts["baseline_failed_sieve_package_sufficient"] += 1
            if (
                not bool(baseline.get("package_sufficient_for_reader"))
                and bool(item.get("package_sufficient_for_reader"))
            ):
                counts["baseline_package_insufficient_sieve_package_sufficient"] += 1
            if int(item.get("quality_score", 0)) >= 1:
                counts["baseline_failed_sieve_partial_or_better"] += 1
                if int(item.get("quality_score", 0)) == 2:
                    counts["baseline_failed_sieve_correct"] += 1
                if int(item.get("prompt_tokens", 0)) < int(baseline.get("prompt_tokens", 0)):
                    counts["baseline_failed_sieve_used_fewer_prompt_tokens"] += 1
        if (
            bool(item.get("package_sufficient_for_reader"))
            and bool(baseline.get("package_sufficient_for_reader"))
            and int(item.get("prompt_tokens", 0)) < int(baseline.get("prompt_tokens", 0))
        ):
            counts["both_package_sufficient_sieve_used_fewer_prompt_tokens"] += 1
        if sieve_failed and baseline_failed:
            sieve_stage = stage_rank.get(str(item.get("failure_stage") or ""), -1)
            baseline_stage = stage_rank.get(str(baseline.get("failure_stage") or ""), -1)
            if sieve_stage > baseline_stage:
                counts["both_failed_sieve_failed_later_stage"] += 1
        if sieve_failed and baseline_succeeded:
            counts["sieve_failed_baseline_succeeded"] += 1
    return counts


def _top_counter_rows(counter: Counter[str], *, limit: int = 8) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]


def write_summary_markdown(path: Path, payload: dict[str, Any]) -> None:
    summaries = payload["summaries"]
    is_beam = "BEAM" in payload.get("dataset_names", []) or "beam" in payload.get("dataset_names", [])
    lines = [
        "# Answer Generation Phase 1",
        "",
        f"- date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- slice: `{payload['slice_path']}`",
        f"- controller config: `{payload['controller_config_path']}`",
        f"- generator: `{payload['generator_model']}` via `{payload.get('generator_provider', 'ollama')}` at `{payload['generator_base_url']}`",
        f"- prompt variant: `{payload.get('prompt_variant', 'baseline')}`",
        f"- sieve reader mode: `{payload.get('sieve_reader_mode', 'calibrated_default')}`",
        f"- pool augmentation: `{payload.get('pool_augmentation', True)}`",
        f"- systems: {', '.join(payload['systems'])}",
    ]
    if payload.get("eval_split"):
        lines.append(f"- eval split: `{payload['eval_split']}`")
    if payload.get("row_count") is not None:
        lines.append(f"- rows: `{payload['row_count']}`")
    if payload.get("dataset_names"):
        lines.append(f"- datasets: {', '.join(f'`{name}`' for name in payload['dataset_names'])}")
    if payload.get("parallelism") is not None:
        lines.append(f"- parallelism: `{payload['parallelism']}`")
    if payload.get("rubric_version"):
        lines.append(f"- rubric: `{payload['rubric_version']}`")
    if payload.get("rescored_from"):
        lines.append(f"- rescored from: `{payload['rescored_from']}`")
    # Extract naive benchmark if exists for reduction comparison
    naive_summary = summaries.get("naive_top_k")
    naive_mem = float(naive_summary.get("avg_selected_memory_tokens", 0.0)) if naive_summary else 0.0

    primary_system = payload["success_call"].get("primary_system", payload["systems"][0])
    primary_items = _system_rows(payload, primary_system)
    primary_failure_counts = _failure_accounting(primary_items)
    primary_failure_stage_counts = Counter(
        str(item.get("failure_stage") or "unknown")
        for item in primary_items
    )
    primary_route_reason_counts = Counter(
        str(item.get("reader_route_reason") or "unknown")
        for item in primary_items
    )
    primary_query_family_counts = Counter(
        str(item.get("query_family") or "unknown")
        for item in primary_items
    )
    naive_items = _system_rows(payload, "naive_top_k")
    comparative_counts = (
        _comparative_failure_accounting(primary_items, naive_items)
        if naive_items and primary_system != "naive_top_k"
        else None
    )

    lines.extend(
        [
            "",
            "## Failure Accounting",
            "",
            f"- primary system: `{primary_system}`",
            "- quality scale: row `quality_score` uses the raw `0..2` rubric; `avg_quality_norm` is the same score normalized to `0..1` by dividing by 2.",
            f"- failed rows: `{primary_failure_counts['failed_rows']}/{primary_failure_counts['rows']}`",
            f"- unknown answers: `{primary_failure_counts['unknown_answers']}`",
            f"- correct `Unknown`: `{primary_failure_counts['correct_unknown']}`",
            f"- false `Unknown` (gold covered but abstained anyway): `{primary_failure_counts['false_unknown']}`",
            f"- gold covered but still wrong: `{primary_failure_counts['gold_covered_but_wrong']}`",
            f"- reader-routed failures: `{primary_failure_counts['reader_failures']}`",
            f"- deterministic-route failures: `{primary_failure_counts['deterministic_failures']}`",
            f"- abstain-route failures: `{primary_failure_counts['abstain_failures']}`",
            f"- rows where the compiler surfaced some usable evidence: `{primary_failure_counts['compiler_surface_success']}`",
            f"- rows where the package was sufficient for a bounded reader: `{primary_failure_counts['package_sufficient_rows']}`",
            f"- failed rows with package insufficiency: `{primary_failure_counts['package_insufficient_failures']}`",
            f"- failed rows that still had a reader-sufficient package: `{primary_failure_counts['package_sufficient_reader_failures']}`",
            "",
        ]
    )
    lines.extend(
        [
            "## Failure Stage Breakdown",
            "",
            "| failure_stage | count | % |",
            "| --- | --- | --- |",
        ]
    )
    total_primary_rows = max(1, primary_failure_counts["rows"])
    for stage, count in _top_counter_rows(primary_failure_stage_counts, limit=12):
        lines.append(f"| `{stage}` | {count} | {(count / total_primary_rows) * 100:.1f}% |")
    lines.extend(["", "## Route Reasons", "", "| reader_route_reason | count | % |", "| --- | --- | --- |"])
    for route_reason, count in _top_counter_rows(primary_route_reason_counts, limit=12):
        lines.append(f"| `{route_reason}` | {count} | {(count / total_primary_rows) * 100:.1f}% |")
    lines.extend(["", "## Query Families", "", "| query_family | count | % |", "| --- | --- | --- |"])
    for family, count in _top_counter_rows(primary_query_family_counts, limit=12):
        lines.append(f"| `{family}` | {count} | {(count / total_primary_rows) * 100:.1f}% |")
    lines.append("")
    if comparative_counts is not None:
        lines.extend(
            [
                "## Comparative Failure Accounting Vs `naive_top_k`",
                "",
                f"- baseline-failed rows: `{comparative_counts['baseline_failed_rows']}`",
                f"- baseline failed, SIEVE covered the gold memory: `{comparative_counts['baseline_failed_sieve_gold_covered']}`",
                f"- baseline failed, SIEVE reached partial-or-better: `{comparative_counts['baseline_failed_sieve_partial_or_better']}`",
                f"- baseline failed, SIEVE fully corrected the row: `{comparative_counts['baseline_failed_sieve_correct']}`",
                f"- baseline failed, SIEVE still surfaced a reader-sufficient package: `{comparative_counts['baseline_failed_sieve_package_sufficient']}`",
                f"- baseline package insufficient, SIEVE package sufficient: `{comparative_counts['baseline_package_insufficient_sieve_package_sufficient']}`",
                f"- baseline failed, SIEVE succeeded with fewer prompt tokens: `{comparative_counts['baseline_failed_sieve_used_fewer_prompt_tokens']}`",
                f"- both packages sufficient, SIEVE used fewer prompt tokens: `{comparative_counts['both_package_sufficient_sieve_used_fewer_prompt_tokens']}`",
                f"- both failed, but SIEVE failed later in the pipeline: `{comparative_counts['both_failed_sieve_failed_later_stage']}`",
                f"- SIEVE failed while baseline succeeded: `{comparative_counts['sieve_failed_baseline_succeeded']}`",
                "",
            ]
        )

    primary_summary = summaries.get(primary_system, {})
    if primary_summary:
        lines.extend(
            [
                "",
                "## Budget Windows",
                "",
                f"- avg compiler budget target tokens: `{primary_summary.get('avg_budget_target_tokens')}`",
                f"- avg compiler budget hard tokens: `{primary_summary.get('avg_budget_hard_tokens')}`",
                f"- avg proposal budget target tokens: `{primary_summary.get('avg_proposal_budget_target_tokens')}`",
                f"- avg proposal budget hard tokens: `{primary_summary.get('avg_proposal_budget_hard_tokens')}`",
                "",
                "| compiler overshoot reason | count |",
                "| --- | --- |",
            ]
        )
        compiler_overshoot = primary_summary.get("budget_overshoot_reason_counts") or {}
        if compiler_overshoot:
            for reason, count in sorted(compiler_overshoot.items()):
                lines.append(f"| `{reason}` | {count} |")
        else:
            lines.append("| `none` | 0 |")
        lines.extend(["", "| proposal overshoot reason | count |", "| --- | --- |"])
        proposal_overshoot = primary_summary.get("proposal_budget_overshoot_reason_counts") or {}
        if proposal_overshoot:
            for reason, count in sorted(proposal_overshoot.items()):
                lines.append(f"| `{reason}` | {count} |")
        else:
            lines.append("| `none` | 0 |")
    lines.append("")

    lines.extend(
        [
            "",
            "## System Efficiency Summary",
            "",
            "| system | Instruction Tokens (fixed) | Memory Tokens (variable) | Total Prompt Tokens | Memory Reduction (vs Naive) |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for system in payload["systems"]:
        summary = summaries[system]
        mem = float(summary.get("avg_selected_memory_tokens", 1e-9))
        total_p = float(summary.get("avg_prompt_tokens", 0.0))
        fixed = max(0.0, total_p - mem)
        reduction = 0.0
        if naive_mem > 0:
            reduction = (naive_mem - mem) / naive_mem
        
        reduction_str = f"{reduction * 100:.1f}%" if system != "naive_top_k" else "baseline"
        
        lines.append(
            f"| `{system}` | ~{fixed:.1f} | {mem:.1f} | **~{total_p:.1f}** | {reduction_str} |"
        )

    lines.extend(
        [
            "",
            "## Quality & Safety Summary",
            "",
            (
                "| system | avg_quality (norm) | exact_match | partial_or_better | agreement_vs_naive | regret_vs_naive | "
                "pairwise_vs_naive | avg_latency_ms |"
                if is_beam
                else "| system | avg_quality (norm) | exact_match | partial_or_better | avg_memory_tokens | pairwise_vs_naive | "
                "stale_use | contradiction_use | "
                "redundant_use | unnecessary_pers | avg_latency_ms |"
            ),
            (
                "| --- | --- | --- | --- | --- | --- | --- | --- |"
                if is_beam
                else "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
            ),
        ]
    )
    for system in payload["systems"]:
        summary = summaries[system]
        if is_beam:
            lines.append(
                f"| `{system}` | {summary['avg_quality_norm']} | {summary['exact_correct_rate']} | "
                f"{summary['partial_or_better_rate']} | {summary.get('full_context_agreement_rate', 0.0)} | "
                f"{summary.get('full_context_regret_rate', 0.0)} | "
                f"{summary.get('pairwise_quality_score_vs_naive_top_k', 0.0)} | "
                f"{summary['avg_latency_ms']} |"
            )
        else:
            lines.append(
                f"| `{system}` | {summary['avg_quality_norm']} | {summary['exact_correct_rate']} | "
                f"{summary['partial_or_better_rate']} | {summary['avg_selected_memory_tokens']} | "
                f"{summary.get('pairwise_quality_score_vs_naive_top_k', 0.0)} | "
                f"{summary['stale_memory_use_rate']} | "
                f"{summary['contradictory_memory_use_rate']} | {summary['redundant_memory_use_rate']} | "
                f"{summary['unnecessary_personalization_rate']} | "
                f"{summary['avg_latency_ms']} |"
            )

    # --- Paper-aligned metric sections ---
    for system in payload["systems"]:
        summary = summaries[system]

        lme_metrics = summary.get("longmemeval_paper_metrics")
        if lme_metrics:
            lines.extend([
                "",
                f"## LongMemEval Paper Metrics (Binary Accuracy): `{system}`",
                "",
                f"- Overall accuracy: **{lme_metrics['accuracy']}** ({lme_metrics['rows']} rows)",
                f"- Task-averaged accuracy: **{lme_metrics['task_averaged_accuracy']}**",
                "",
                "| question_type | rows | accuracy |",
                "| --- | ---: | ---: |",
            ])
            for qt, stats in lme_metrics.get("by_question_type", {}).items():
                lines.append(f"| `{qt}` | {stats['rows']} | {stats['accuracy']} |")

        loc_metrics = summary.get("locomo_paper_metrics")
        if loc_metrics:
            cat_names = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
            lines.extend([
                "",
                f"## LoCoMo Paper Metrics (Stemmed F1): `{system}`",
                "",
                f"- Mean F1 (paper-aligned): **{loc_metrics['mean_f1_paper']}** ({loc_metrics['rows']} rows)",
                "",
                "| category | rows | mean_f1 |",
                "| --- | ---: | ---: |",
            ])
            for cat, stats in loc_metrics.get("by_locomo_category", {}).items():
                name = cat_names.get(int(cat), f"cat-{cat}")
                lines.append(f"| {cat} ({name}) | {stats['rows']} | {stats['mean_f1']} |")

    if is_beam:
        for system in payload["systems"]:
            summary = summaries[system]
            ability_summary = summary.get("by_ability", {})
            if ability_summary:
                lines.extend(["", f"## BEAM By Ability: `{system}`", ""])
                lines.extend(
                    [
                        "| ability | rows | avg_quality | exact_correct | partial_or_better | agreement_vs_naive | regret_vs_naive |",
                        "| --- | --- | --- | --- | --- | --- | --- |",
                    ]
                )
                for ability, stats in ability_summary.items():
                    lines.append(
                        f"| `{ability}` | {stats['rows']} | {stats['avg_quality_norm']} | {stats['exact_correct_rate']} | "
                        f"{stats['partial_or_better_rate']} | {stats.get('full_context_agreement_rate', 0.0)} | "
                        f"{stats.get('full_context_regret_rate', 0.0)} |"
                    )

            chat_size_summary = summary.get("by_chat_size", {})
            if chat_size_summary:
                lines.extend(["", f"## BEAM By Chat Size: `{system}`", ""])
                lines.extend(
                    [
                        "| chat_size | rows | avg_quality | exact_correct | partial_or_better | agreement_vs_naive | regret_vs_naive |",
                        "| --- | --- | --- | --- | --- | --- | --- |",
                    ]
                )
                for chat_size, stats in chat_size_summary.items():
                    lines.append(
                        f"| `{chat_size}` | {stats['rows']} | {stats['avg_quality_norm']} | {stats['exact_correct_rate']} | "
                        f"{stats['partial_or_better_rate']} | {stats.get('full_context_agreement_rate', 0.0)} | "
                        f"{stats.get('full_context_regret_rate', 0.0)} |"
                    )

            dataset_summary = summary.get("by_dataset", {})
            if len(dataset_summary) > 1:
                lines.extend(["", f"## By Dataset: `{system}`", ""])
                lines.extend(
                    [
                        "| dataset | rows | avg_quality | exact_correct | partial_or_better | agreement_vs_naive | regret_vs_naive |",
                        "| --- | --- | --- | --- | --- | --- | --- |",
                    ]
                )
                for dataset_name, stats in dataset_summary.items():
                    lines.append(
                        f"| `{dataset_name}` | {stats['rows']} | {stats['avg_quality_norm']} | {stats['exact_correct_rate']} | "
                        f"{stats['partial_or_better_rate']} | {stats.get('full_context_agreement_rate', 0.0)} | "
                        f"{stats.get('full_context_regret_rate', 0.0)} |"
                    )

    lines.extend(
        [
            "",
            "## Call",
            "",
            f"- answer-generation phase success: `{payload['success_call']['answer_generation_phase_success']}`",
            f"- primary method: `{primary_system}`",
            f"- primary method avg quality (norm): {summaries[primary_system].get('avg_quality_norm', 0.0)}",
            (
                f"- primary method constrained pairwise score vs naive top-k: "
                f"{summaries[primary_system].get('constrained_pairwise_score_vs_naive_top_k', 'n/a')}"
            ),
            (
                f"- primary method pairwise headline metric: "
                f"`{payload['success_call'].get('primary_system_pairwise_metric')}` = "
                f"{payload['success_call'].get('primary_system_pairwise_score')}"
                if payload["success_call"].get("primary_system_pairwise_metric")
                else "- primary method pairwise headline metric: n/a"
            ),
            (
                f"- best realistic baseline avg quality score: {payload['success_call']['best_realistic_baseline_quality']}"
                if payload["success_call"].get("has_realistic_baseline", False)
                else "- best realistic baseline avg quality score: n/a (no realistic baseline run)"
            ),
            f"- interpretation: {payload['success_call']['note']}",
            "",
            "## Notes",
            "",
        ]
    )
    for system in payload["systems"]:
        summary = summaries[system]
        modes = summary.get("selection_mode_counts")
        if system == primary_system:
            failure_stage_counts = summary.get("failure_stage_counts") or {}
            if failure_stage_counts:
                lines.extend([f"### Failure Stages: `{system}`", ""])
                lines.extend(["| stage | count | % |", "| --- | --- | --- |"])
                for stage, count in failure_stage_counts.items():
                    pct = (count / summary["rows"]) * 100 if summary["rows"] else 0.0
                    lines.append(f"| `{stage}` | {count} | {pct:.1f}% |")
                lines.append("")
            route_counts = summary.get("reader_route_reason_counts") or {}
            if route_counts:
                lines.extend([f"### Route Reasons: `{system}`", ""])
                lines.extend(["| route | count | % |", "| --- | --- | --- |"])
                for route, count in route_counts.items():
                    pct = (count / summary["rows"]) * 100 if summary["rows"] else 0.0
                    lines.append(f"| `{route}` | {count} | {pct:.1f}% |")
                lines.append("")
        if modes:
            lines.extend([f"### Selection Modes: `{system}`", ""])
            lines.extend(["| mode | count | % |", "| --- | --- | --- |"])
            total_rows = summary["rows"]
            for mode, count in modes.items():
                pct = (count / total_rows) * 100
                lines.append(f"| `{mode}` | {count} | {pct:.1f}% |")
            lines.append("")
            if "surgical_coverage_rate" in summary:
                lines.extend(
                    [
                        f"- surgical_coverage_rate: {summary['surgical_coverage_rate'] * 100:.1f}%   # % of rows using anchor or bundle strategies",
                        f"- fallback_rate: {summary['fallback_rate'] * 100:.1f}%            # % of rows using fallback_pack",
                        f"- avg_tokens_surgical: {summary.get('avg_tokens_surgical', 'n/a')}",
                        f"- avg_tokens_fallback: {summary.get('avg_tokens_fallback', 'n/a')}",
                        "",
                    ]
                )
            
    for system in payload["systems"]:
        note = SYSTEM_NOTES.get(system)
        if note:
            lines.append(f"- `{system}` {note}")
    for system in payload["systems"]:
        failures = summaries[system]["notable_failures"]
        if not failures:
            continue
        lines.extend(["", f"## Notable Failures: `{system}`", ""])
        for failure in failures[:4]:
            if is_beam:
                lines.append(
                    f"- `{failure['example_id']}` ({failure.get('ability', failure['harm_type'])}, "
                    f"{failure.get('chat_size', 'unknown')}): answer=`{failure['answer']}` vs "
                    f"ref=`{failure['reference_answer']}`"
                )
            else:
                lines.append(
                    f"- `{failure['example_id']}` ({failure['harm_type']}): "
                    f"answer=`{failure['answer']}` vs ref=`{failure['reference_answer']}`"
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
