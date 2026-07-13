"""Materialization helpers for compiler execution selection."""

from __future__ import annotations

from typing import Any, Callable

from ..runtime.bindings import (
    EvidencePlan,
    EvidenceUnit,
    _semantic_sufficiency_analysis,
    _slot_bindings,
    _slot_bindings_with_display_text,
    _unit_payload,
)
from .deterministic import build_answer_contract


def _materialize_compiler_outputs(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets: Any,
    selected_units: list[EvidenceUnit],
    score_weights: dict[str, float],
    memory_labels_by_id: dict[str, dict[str, Any]],
    validate_slot_binding: Callable[..., tuple[dict[str, Any] | None, str | None]],
    infer_missing_slot_invalid_reason: Callable[..., str | None],
    promote_support_grounding: Callable[..., None],
    apply_reference_time_fallback: Callable[..., None],
    rescue_current_state_where_binding: Callable[..., None],
    rescue_information_extraction_where_binding: Callable[..., None],
    rescue_temporal_when_event_binding: Callable[..., None],
    binding_source_text: Callable[[dict[str, Any]], str],
    extract_numeric_mentions: Callable[[str], list[tuple[float, str, str, str]]],
    count_numeric_spans_excluding_dates: Callable[[str], int],
    schema_grounding_analysis: Callable[..., dict[str, Any]],
    execute_distinct_count_from_compiled_units: Callable[..., tuple[str | None, list[str]]],
    execute_percentage_aggregate_from_bindings: Callable[..., tuple[str | None, list[str]]],
    execute_aggregate_from_compiled_units: Callable[..., tuple[str | None, list[str]]],
    execute_direct_duration_from_compiled_units: Callable[..., tuple[str | None, list[str]]],
    execute_relative_time: Callable[..., tuple[str | None, list[str]]],
    execute_ordered_choice: Callable[..., tuple[str | None, list[str]]],
    execute_comparison: Callable[..., tuple[str | None, list[str]]],
    execute_temporal_interval: Callable[..., tuple[str | None, list[str]]],
    deterministic_span_allowed: Callable[..., bool],
    parse_numeric_text: Callable[[str], tuple[float, str, str] | None],
    format_numeric_answer: Callable[[float, str, str], str],
    soft_invalid_reasons: set[str],
    query_focus_tokens: Callable[[dict[str, Any]], set[str]],
) -> tuple[list[dict[str, Any]], list[str], list[str], dict[str, dict[str, Any] | None], dict[str, Any], dict[str, Any]]:
    compiled_units = [
        _unit_payload(
            row=row,
            plan=plan,
            query_targets=query_targets,
            unit=unit,
            memory_labels_by_id=memory_labels_by_id,
            score_weights=score_weights,
        )
        for unit in selected_units
    ]
    compiled_texts = [unit["render_text"] for unit in compiled_units if str(unit["render_text"]).strip()]
    compiled_memory_ids: list[str] = []
    for unit in compiled_units:
        memory_id = str(unit["memory_id"])
        if memory_id not in compiled_memory_ids:
            compiled_memory_ids.append(memory_id)
    raw_slot_bindings = _slot_bindings(plan, compiled_units, query_targets=query_targets)
    answer_contract = build_answer_contract(
        row=row,
        plan=plan,
        query_targets=query_targets,
        compiled_units=compiled_units,
        raw_slot_bindings=raw_slot_bindings,
        slot_bindings_with_display_text=_slot_bindings_with_display_text,
        query_focus_tokens=query_focus_tokens,
        validate_slot_binding=validate_slot_binding,
        infer_missing_slot_invalid_reason=infer_missing_slot_invalid_reason,
        promote_support_grounding=promote_support_grounding,
        apply_reference_time_fallback=apply_reference_time_fallback,
        rescue_current_state_where_binding=rescue_current_state_where_binding,
        rescue_information_extraction_where_binding=rescue_information_extraction_where_binding,
        rescue_temporal_when_event_binding=rescue_temporal_when_event_binding,
        binding_source_text=binding_source_text,
        extract_numeric_mentions=extract_numeric_mentions,
        count_numeric_spans_excluding_dates=count_numeric_spans_excluding_dates,
        schema_grounding_analysis=schema_grounding_analysis,
        execute_distinct_count_from_compiled_units=execute_distinct_count_from_compiled_units,
        execute_percentage_aggregate_from_bindings=execute_percentage_aggregate_from_bindings,
        execute_aggregate_from_compiled_units=execute_aggregate_from_compiled_units,
        execute_direct_duration_from_compiled_units=execute_direct_duration_from_compiled_units,
        execute_relative_time=execute_relative_time,
        execute_ordered_choice=execute_ordered_choice,
        execute_comparison=execute_comparison,
        execute_temporal_interval=execute_temporal_interval,
        deterministic_span_allowed=deterministic_span_allowed,
        parse_numeric_text=parse_numeric_text,
        format_numeric_answer=format_numeric_answer,
        soft_invalid_reasons=soft_invalid_reasons,
    )
    semantic_analysis = _semantic_sufficiency_analysis(
        row=row,
        plan=plan,
        query_targets=query_targets,
        validated_slot_bindings=dict(answer_contract["validated_slot_bindings"]),
        slot_grounding_scores=dict(answer_contract.get("slot_grounding_scores") or {}),
        invalid_slots=dict(answer_contract.get("invalid_slots") or {}),
    )
    return compiled_units, compiled_texts, compiled_memory_ids, raw_slot_bindings, answer_contract, semantic_analysis
