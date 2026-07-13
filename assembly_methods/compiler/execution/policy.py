"""Selection policy helpers for compiler execution."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..runtime import bindings as rb
from . import requirements as req

_GENERIC_COUNT_OBJECT_TOKENS = {"different", "distinct", "type", "types", "kind", "kinds", "item", "items"}


def _plan_metadata_text(plan: rb.EvidencePlan, key: str) -> str:
    return str(getattr(plan, "plan_metadata", {}).get(key) or "").strip().lower()


def _reader_first_schema(plan: rb.EvidencePlan) -> bool:
    return bool(
        req.is_numeric_aggregate_schema(plan)
        or req.is_comparison_schema(plan)
        or req.is_count_aggregate_schema(plan)
        or plan.schema_name in {"TemporalInterval", "RelativeTime"}
    )


def _infer_missing_slot_invalid_reason(
    *,
    row: dict[str, Any],
    plan: rb.EvidencePlan,
    query_targets: rb.QueryTargets,
    slot: Any,
    compiled_units: list[dict[str, Any]],
) -> str | None:
    del query_targets
    if not compiled_units:
        return "missing"
    if req.is_numeric_aggregate_schema(plan) or req.is_comparison_schema(plan) or plan.schema_name in {"TemporalInterval", "OrderedChoice"}:
        return "missing_counterpart"
    if req.is_count_aggregate_schema(plan):
        object_tokens = rb._count_lookup_object_tokens(row) - _GENERIC_COUNT_OBJECT_TOKENS
        predicate_tokens = rb._count_lookup_predicate_tokens(row)
        for unit in compiled_units:
            if not isinstance(unit, dict):
                continue
            source_text = rb._compiled_unit_body_text(unit)
            normalized_source = rb._normalize_text(source_text)
            unit_tokens = set(normalized_source.split())
            if (object_tokens and unit_tokens & object_tokens) or (predicate_tokens and unit_tokens & predicate_tokens):
                return "missing"
        return "wrong_attribute"
    if slot.slot_type in {"numeric_operand", "comparison_operand"}:
        return "numeric_pollution"
    return None


def _deterministic_span_allowed(
    *,
    row: dict[str, Any],
    plan: rb.EvidencePlan,
    slot_name: str,
    binding: dict[str, Any],
) -> bool:
    answer_shape = _plan_metadata_text(plan, "answer_shape")
    display_text = str(binding.get("display_text") or binding.get("text") or "").strip()
    if not display_text:
        return False
    # Query-echo hard gate: if the display text appears verbatim in the query,
    # the compiler grabbed the question entity, not the answer.
    normalized_display = rb._normalize_text(display_text)
    normalized_query = rb._normalize_text(str(row.get("query", "")))
    if normalized_display and f" {normalized_display} " in f" {normalized_query} ":
        return False
    if answer_shape in {"entity_span", "attribute_span", "resolved_state"}:
        return slot_name != "numeric_value"
    if plan.family in {"current_state", "knowledge_update", "conflict_update"} and slot_name in {"current_value", "direct_value", "new_value", "event"}:
        return True
    if plan.schema_name == "DirectValue" and slot_name in {"direct_value", "current_value", "new_value"}:
        return not plan.requires_numeric_reasoning
    return rb._query_expects_time_literal(row)


def _single_value_seed_slot(plan: rb.EvidencePlan) -> Any | None:
    if not (req.is_direct_lookup_schema(plan) or req.is_current_state_schema(plan) or req.is_state_update_schema(plan)):
        return None
    candidates = [
        slot
        for slot in plan.slots
        if slot.required and str(getattr(slot, "slot_type", "") or "") in {"direct_value", "current_resolution", "new_state", "old_state"}
    ]
    if len(candidates) != 1:
        return None
    return candidates[0]


def _validated_single_value_seed_unit(
    *,
    row: dict[str, Any],
    plan: rb.EvidencePlan,
    query_targets: rb.QueryTargets,
    available_units: list[rb.EvidenceUnit],
    memory_labels_by_id: dict[str, dict[str, Any]],
    score_weights: dict[str, float],
    validate_slot_binding: Any,
) -> rb.EvidenceUnit | None:
    seed_slot = _single_value_seed_slot(plan)
    if seed_slot is None:
        return None

    focus_tokens = rb._query_focus_tokens(row)
    needs_personal_gate = rb._query_needs_personal_semantic_gate(
        row=row,
        plan=plan,
        query_targets=query_targets,
    )
    best_key: tuple[float, ...] | None = None
    best_unit: rb.EvidenceUnit | None = None

    for unit in available_units:
        semantic_support = rb._unit_semantic_support(
            row=row,
            query_family=plan.query_family,
            unit=unit,
        )
        if needs_personal_gate:
            speaker = str(unit.speaker or "").strip().lower()
            if semantic_support < 0.5:
                continue
            if speaker != "user" and not str(unit.memory_id).startswith("answer_"):
                continue

        compiled_unit = rb._unit_payload(
            row=row,
            plan=plan,
            query_targets=query_targets,
            unit=unit,
            memory_labels_by_id=memory_labels_by_id,
            score_weights=score_weights,
        )
        slot_candidates = compiled_unit.get("slot_candidates")
        if not isinstance(slot_candidates, list) or seed_slot.slot_name not in slot_candidates:
            continue

        single_slot_plan = replace(plan, slots=(seed_slot,))
        raw_bindings = rb._slot_bindings(
            single_slot_plan,
            [compiled_unit],
            query_targets=query_targets,
        )
        displayed_binding = rb._slot_bindings_with_display_text(
            plan=single_slot_plan,
            compiled_units=[compiled_unit],
            slot_bindings=raw_bindings,
        ).get(seed_slot.slot_name)
        if not isinstance(displayed_binding, dict):
            continue

        validated_binding, _ = validate_slot_binding(
            row=row,
            plan=single_slot_plan,
            query_targets=query_targets,
            slot=seed_slot,
            binding=displayed_binding,
            focus_tokens=focus_tokens,
        )
        if not isinstance(validated_binding, dict):
            continue

        is_state = bool(getattr(unit, "is_state_assertion", False))
        is_current_state = bool(getattr(unit, "is_current_state_candidate", False) or getattr(unit, "has_current_state_language", False))
        is_completion = bool(getattr(unit, "has_consumption_or_completion_marker", False))
        is_question = bool(getattr(unit, "is_question_or_request", False))
        labels = set(memory_labels_by_id.get(unit.memory_id, {}).get("labels", []))
        date_sort_key = str(unit.date_key or "")

        key = (
            semantic_support,
            1 if is_completion else 0,
            1 if is_current_state else 0,
            0 if is_question else 1,
            1 if "answer_bearing" in labels else 0,
            rb._binding_quality_score(validated_binding),
            # Recency/State tiebreaker: for direct lookups and state queries,
            # prefer the latest state assertion when semantic support is comparable.
            1 if is_state else 0,
            date_sort_key,
            rb.score_evidence_unit(
                row=row,
                query_family=plan.query_family,
                unit=unit,
                memory_labels_by_id=memory_labels_by_id,
                score_weights=score_weights,
                plan=plan,
            ),
            -unit.token_count,
            -unit.parent_rank,
            unit.unit_id,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_unit = unit

    return best_unit


def reader_prompt_variant(
    *,
    plan: rb.EvidencePlan,
    answer_contract: dict[str, Any],
    semantic_analysis: dict[str, Any],
    compiled_unit_count: int,
    compiled_token_count: int,
) -> str:
    """Return the compiler-owned prompt route for answer generation."""
    answer_mode = str(answer_contract.get("answer_mode") or "").strip().lower()
    answerability_level = str(answer_contract.get("answerability_level") or "").strip().lower()
    invalid_slots = dict(answer_contract.get("invalid_slots") or {})
    fail_reasons = dict(answer_contract.get("slot_grounding_fail_reasons") or {})
    validated_slots = dict(answer_contract.get("validated_slot_bindings") or {})
    validated_slot_count = sum(1 for binding in validated_slots.values() if isinstance(binding, dict))
    flat_fail_reasons = {
        str(reason).strip().lower()
        for reasons in fail_reasons.values()
        if isinstance(reasons, list)
        for reason in reasons
        if str(reason).strip()
    }
    semantic_sufficiency_passed = bool(semantic_analysis.get("semantic_sufficiency_passed", True))
    semantic_sufficiency_applied = bool(semantic_analysis.get("semantic_sufficiency_applied", False))
    semantic_score_raw = semantic_analysis.get("semantic_sufficiency_score")
    semantic_sufficiency_score = float(semantic_score_raw) if semantic_score_raw is not None else None

    if answer_mode == "abstain" and answerability_level == "abstain":
        return "compiler_abstain"
    if answer_mode in {"deterministic_span", "deterministic_numeric"} and bool(answer_contract.get("answer_text")):
        return answer_mode

    # Compact reader outperforms structured reader on every question type
    # (verified Apr 20: +0.127 multi-session, +0.458 single-session-assistant,
    # +0.571 knowledge-update, +0.364 single-session-user, +0.009 temporal).
    # Route all reader rows to compact format regardless of evidence size.
    if answer_mode == "reader_from_slots" and answerability_level in {"reader_only", "deterministic_safe"}:
        return "crisp_compact_v1"

    return "structured_reader_v1"
