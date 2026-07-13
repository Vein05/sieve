"""Selection and orchestration entry point for compiler execution."""

from __future__ import annotations

import re
from typing import Any

from ..runtime import bindings as rb
from ..runtime import entities as re_entities
from ..runtime import text as rtext
from ..validation.answer_type import validate_slot_binding as _validate_slot_binding_impl
from ..validation.invariants import schema_grounding_analysis as _schema_grounding_analysis_impl
from ..validation import rescue as rescue_validation
from ..evidence_planner import plan_evidence
from evidence.projection import project_evidence_candidates
from evidence.units import build_evidence_units_for_views
from retrieval.query_targets import extract_query_targets
from . import evaluation as evaluation_mod
from . import coverage as coverage_mod
from . import materialization as materialization_mod
from . import requirements as req
from . import sufficiency as sufficiency_mod
from .extractors import intent as intent_mod
from .expansion import _augment_with_anchor_fallback, _expansion_candidates
from .greedy import _expand_trusted_memory_units, _select_greedy_units, _unit_sort_key
from .router import apply_semantic_reader_downgrade
from .adaptive_budget import build_compiler_budget_window
from .policy import (
    _deterministic_span_allowed,
    _infer_missing_slot_invalid_reason,
    reader_prompt_variant,
    _single_value_seed_slot,
    _validated_single_value_seed_unit,
)
from .sufficiency import (
    _role_specific_missing,
    _role_specific_sufficient,
    _should_expand_for_sufficiency,
)

_FIRST_PERSON_SOURCE_RE = re.compile(r"\b(i|i'm|i've|i’d|i'll|me|my|mine|we|we're|our|ours|us)\b", re.IGNORECASE)


def _preferred_user_evidence_units(
    *,
    row: dict[str, Any],
    plan: rb.EvidencePlan,
    available_units: list[rb.EvidenceUnit],
    memory_labels_by_id: dict[str, dict[str, Any]],
    score_weights: dict[str, float],
    max_compiled_tokens: int,
) -> list[rb.EvidenceUnit]:
    if plan.family not in {"aggregation", "temporal", "ordering"}:
        return []

    ranked: list[tuple[tuple[float, ...], rb.EvidenceUnit]] = []
    for unit in available_units:
        source_text = str(unit.provenance.get("source_text") or unit.render_text or "").strip()
        if str(unit.speaker or "").strip().lower() != "user":
            continue
        if not _FIRST_PERSON_SOURCE_RE.search(source_text):
            continue
        if not (
            unit.numeric_values
            or unit.time_markers
            or str(unit.provenance.get("duration_value") or "").strip()
        ):
            continue
        labels = set(memory_labels_by_id.get(unit.memory_id, {}).get("labels", []))
        key = (
            1.0 if "answer_bearing" in labels else 0.0,
            rb.score_evidence_unit(
                row=row,
                query_family=plan.query_family,
                unit=unit,
                memory_labels_by_id=memory_labels_by_id,
                score_weights=score_weights,
                plan=plan,
            ),
            1.0 if unit.numeric_values else 0.0,
            1.0 if str(unit.provenance.get("duration_value") or "").strip() else 0.0,
            1.0 if unit.time_markers else 0.0,
            -float(unit.token_count),
            -float(unit.parent_rank),
        )
        ranked.append((key, unit))

    if not ranked:
        return []

    ranked.sort(key=lambda item: item[0], reverse=True)
    chosen: list[rb.EvidenceUnit] = []
    seen_memory_ids: set[str] = set()
    used_tokens = 0
    for _, unit in ranked:
        projected = used_tokens + int(unit.token_count)
        if max_compiled_tokens > 0 and chosen and projected > max_compiled_tokens:
            continue
        if unit.memory_id in seen_memory_ids and len(seen_memory_ids) < 2:
            continue
        chosen.append(unit)
        seen_memory_ids.add(unit.memory_id)
        used_tokens = projected
        if len(chosen) >= 2:
            break
    return sorted(chosen, key=_unit_sort_key)


def _validate_slot_binding(
    *,
    row: dict[str, Any],
    plan: rb.EvidencePlan,
    query_targets: rb.QueryTargets,
    slot: Any,
    binding: dict[str, Any] | None,
    focus_tokens: set[str],
) -> tuple[dict[str, Any] | None, str | None]:
    return _validate_slot_binding_impl(
        row=row,
        plan=plan,
        query_targets=query_targets,
        slot=slot,
        binding=binding,
        focus_tokens=focus_tokens,
        binding_source_text=rb._binding_source_text,
        normalize_text=rb._normalize_text,
        is_generic_fragment=rb._is_generic_fragment,
        query_expects_time_literal=rb._query_expects_time_literal,
        binding_value_from_session_header=rb._binding_value_from_session_header,
        best_matching_entity=rb._best_matching_entity,
        direct_value_requires_entity_match=rb._direct_value_requires_entity_match,
        binding_focus_overlap_tokens=rb._binding_focus_overlap_tokens,
        binding_matches_query_focus=rb._binding_matches_query_focus,
        slot_expected_entities=rb._slot_expected_entities,
        slot_entity_match=rb._slot_entity_match,
        slot_allows_descriptive_fragment=rb._slot_allows_descriptive_fragment,
        copy_binding_with_text=rb._copy_binding_with_text,
        parse_numeric_text=rb._parse_numeric_text,
        numeric_query_requires_money=intent_mod._numeric_query_requires_money,
        numeric_query_requires_distance=intent_mod._numeric_query_requires_distance,
        numeric_query_requires_speed=intent_mod._numeric_query_requires_speed,
        contains_math_expression=rb._contains_math_expression,
    )


def _schema_grounding_analysis(
    *,
    row: dict[str, Any],
    plan: rb.EvidencePlan,
    query_targets: rb.QueryTargets,
    validated_bindings: dict[str, dict[str, Any] | None],
    displayed_bindings: dict[str, dict[str, Any] | None],
    invalid_slots: dict[str, str],
) -> dict[str, Any]:
    return _schema_grounding_analysis_impl(
        row=row,
        plan=plan,
        query_targets=query_targets,
        validated_bindings=validated_bindings,
        displayed_bindings=displayed_bindings,
        invalid_slots=invalid_slots,
        normalize_text=rb._normalize_text,
        binding_source_text=rb._binding_source_text,
        binding_quality_score=rb._binding_quality_score,
        is_generic_fragment=rb._is_generic_fragment,
        slot_allows_descriptive_fragment=rb._slot_allows_descriptive_fragment,
        slot_expected_entities=rb._slot_expected_entities,
        slot_entity_match=rb._slot_entity_match,
        parse_numeric_text=rb._parse_numeric_text,
        contains_math_expression=rb._contains_math_expression,
        binding_value_from_session_header=rb._binding_value_from_session_header,
        query_expects_time_literal=rb._query_expects_time_literal,
        event_matches_expected=rb._event_matches_expected,
    )


def _promote_support_grounding(
    *,
    plan: rb.EvidencePlan,
    validated_bindings: dict[str, dict[str, Any] | None],
    displayed_bindings: dict[str, dict[str, Any] | None],
    invalid_slots: dict[str, str],
) -> None:
    rescue_validation.promote_support_grounding(
        plan=plan,
        validated_bindings=validated_bindings,
        displayed_bindings=displayed_bindings,
        invalid_slots=invalid_slots,
        binding_source_text=rb._binding_source_text,
        is_generic_fragment=rb._is_generic_fragment,
        binding_quality_score=rb._binding_quality_score,
    )


def _apply_reference_time_fallback(
    *,
    row: dict[str, Any],
    plan: rb.EvidencePlan,
    validated_bindings: dict[str, dict[str, Any] | None],
    invalid_slots: dict[str, str],
) -> None:
    rescue_validation.apply_reference_time_fallback(
        row=row,
        plan=plan,
        validated_bindings=validated_bindings,
        invalid_slots=invalid_slots,
        query_has_relative_reference_clause=rtext._query_has_relative_reference_clause,
        question_date_binding=re_entities._question_date_binding,
    )


def _rescue_current_state_where_binding(
    *,
    row: dict[str, Any],
    plan: rb.EvidencePlan,
    displayed_bindings: dict[str, dict[str, Any] | None],
    validated_bindings: dict[str, dict[str, Any] | None],
    invalid_slots: dict[str, str],
) -> None:
    rescue_validation.rescue_current_state_where_binding(
        row=row,
        plan=plan,
        displayed_bindings=displayed_bindings,
        validated_bindings=validated_bindings,
        invalid_slots=invalid_slots,
        normalize_text=rb._normalize_text,
        binding_source_text=rb._binding_source_text,
        extract_location_from_source=rescue_validation._extract_location_from_source,
        extract_state_entity_from_source=rescue_validation._extract_state_entity_from_source,
        copy_binding_with_text=rb._copy_binding_with_text,
        is_generic_fragment=rb._is_generic_fragment,
    )


def _rescue_information_extraction_where_binding(
    *,
    row: dict[str, Any],
    plan: rb.EvidencePlan,
    displayed_bindings: dict[str, dict[str, Any] | None],
    validated_bindings: dict[str, dict[str, Any] | None],
    invalid_slots: dict[str, str],
) -> None:
    rescue_validation.rescue_information_extraction_where_binding(
        row=row,
        plan=plan,
        displayed_bindings=displayed_bindings,
        validated_bindings=validated_bindings,
        invalid_slots=invalid_slots,
        normalize_text=rb._normalize_text,
        binding_source_text=rb._binding_source_text,
        copy_binding_with_text=rb._copy_binding_with_text,
        extract_location_from_source=rescue_validation._extract_location_from_source,
    )


def _rescue_temporal_when_event_binding(
    *,
    row: dict[str, Any],
    plan: rb.EvidencePlan,
    compiled_units: list[dict[str, Any]],
    displayed_bindings: dict[str, dict[str, Any] | None],
    validated_bindings: dict[str, dict[str, Any] | None],
    invalid_slots: dict[str, str],
) -> None:
    rescue_validation.rescue_temporal_when_event_binding(
        row=row,
        plan=plan,
        compiled_units=compiled_units,
        displayed_bindings=displayed_bindings,
        validated_bindings=validated_bindings,
        invalid_slots=invalid_slots,
        normalize_text=rb._normalize_text,
        binding_source_text=rb._binding_source_text,
        copy_binding_with_text=rb._copy_binding_with_text,
        extract_calendar_phrase_from_source=rescue_validation._extract_calendar_phrase_from_source,
    )


def compile_evidence(
    *,
    row: dict[str, Any],
    context: dict[str, Any],
    query_family: str,
    ranked_views: list[rb.CandidateView],
    selected_views: list[rb.CandidateView],
    target_budget: int,
    memory_labels_by_id: dict[str, dict[str, Any]],
    prebuilt_objects: list | None = None,
) -> dict[str, Any]:
    plan = plan_evidence(row, query_family)
    query_targets = extract_query_targets(row, query_family)
    score_weights = rb._resolve_compiler_score_weights(context)
    requirements = req.requirements_dict(plan)
    selection_requirements = req.selection_requirements_dict(plan)
    compile_budget_window = build_compiler_budget_window(
        row=row,
        context=context,
        plan=plan,
        query_targets=query_targets,
        ranked_views=ranked_views,
        selected_views=selected_views,
        proposal_budget_hint=target_budget,
    )
    effective_target_budget = int(compile_budget_window.target_tokens)
    budget_hard_cap = int(compile_budget_window.hard_tokens)
    budget_throttle_reason = None if budget_hard_cap <= effective_target_budget else "adaptive_window"
    budget_throttle_ratio = (
        round(float(effective_target_budget) / float(max(1, budget_hard_cap)), 6)
        if budget_hard_cap > 0
        else 1.0
    )
    raw_selected_token_count = sum(view.token_count for view in selected_views)
    max_compiled_tokens = req.max_compiled_tokens(
        raw_selected_token_count=raw_selected_token_count,
        target_budget=effective_target_budget,
        hard_budget=budget_hard_cap,
    )

    from shared.perf import get_counters
    get_counters().compile_evidence_calls += 1

    initial_units = build_evidence_units_for_views(selected_views, prebuilt_objects=prebuilt_objects)
    projected_candidates = project_evidence_candidates(
        row=row,
        plan=plan,
        units=initial_units,
        memory_labels_by_id=memory_labels_by_id,
    )
    selected_memory_ids = [view.memory_id for view in selected_views]
    available_units = list(initial_units)
    selected_units: list[rb.EvidenceUnit] = []
    coverage = req.coverage_dict(selection_requirements)
    expansion_steps = [f"initial_selected_memories={len(selected_views)}"]
    used_fallback = False
    validated_seed_unit = _validated_single_value_seed_unit(
        row=row,
        plan=plan,
        query_targets=query_targets,
        available_units=available_units,
        memory_labels_by_id=memory_labels_by_id,
        score_weights=score_weights,
        validate_slot_binding=_validate_slot_binding,
    )

    if not available_units:
        used_fallback = True
    else:
        if validated_seed_unit is not None:
            selected_units.append(validated_seed_unit)
            expansion_steps.append(f"validated_seed_unit={validated_seed_unit.unit_id}")
        else:
            seed_units, coverage, seed_steps = _select_greedy_units(
                row=row,
                plan=plan,
                query_targets=query_targets,
                available_units=available_units,
                budget_window=compile_budget_window,
                memory_labels_by_id=memory_labels_by_id,
                initial_selected_memory_ids=selected_memory_ids,
                score_weights=score_weights,
            )
            selected_units.extend(seed_units)
            expansion_steps.extend(seed_steps)
        if not selected_units:
            answer_bearing_units = [
                unit
                for unit in available_units
                if "answer_bearing" in set(memory_labels_by_id.get(unit.memory_id, {}).get("labels", []))
            ]
            if answer_bearing_units:
                best_unit = max(
                    answer_bearing_units,
                    key=lambda unit: rb.score_evidence_unit(
                        row=row,
                        query_family=plan.query_family,
                        unit=unit,
                        memory_labels_by_id=memory_labels_by_id,
                        score_weights=score_weights,
                        plan=plan,
                    ),
                )
                selected_units.append(best_unit)
                expansion_steps.append(f"reader_seed_memory={best_unit.memory_id}")
        selected_units = _expand_trusted_memory_units(
            selected_units=selected_units,
            available_units=available_units,
            max_compiled_tokens=max_compiled_tokens,
            memory_labels_by_id=memory_labels_by_id,
        )
        coverage = coverage_mod._coverage_for_selected_units(
            requirements=selection_requirements,
            selected_units=selected_units,
            row=row,
            plan=plan,
            query_targets=query_targets,
            memory_labels_by_id=memory_labels_by_id,
        )

    selected_memory_id_set = {unit.memory_id for unit in selected_units}
    expansions = 0
    max_expansions = 6
    while expansions < max_expansions:
        base_satisfied = req.plan_satisfied(selection_requirements, coverage)
        role_satisfied = _role_specific_sufficient(
            plan=plan,
            requirements=requirements,
            selected_units=selected_units,
            row=row,
            query_targets=query_targets,
            memory_labels_by_id=memory_labels_by_id,
        ) if base_satisfied else False
        if base_satisfied and role_satisfied:
            # Competing evidence: if plan is satisfied but there are unselected
            # memories in the pool and we still have expansion budget, inject a
            # synthetic role so expansion keeps looking for better evidence.
            has_unselected = len(selected_memory_id_set) < len(ranked_views)
            if has_unselected and expansions < max_expansions:
                missing = {"competing_anchor": 1}
            else:
                break
            # skip normal missing computation — we already have the synthetic one
        else:
            missing = req.missing_requirements(selection_requirements, coverage)
        if not missing:
            missing = _role_specific_missing(
                plan=plan,
                requirements=requirements,
                selected_units=selected_units,
                row=row,
                query_targets=query_targets,
                memory_labels_by_id=memory_labels_by_id,
            )
        if not missing:
            break
        candidates = _expansion_candidates(
            row=row,
            plan=plan,
            query_targets=query_targets,
            ranked_views=ranked_views,
            selected_memory_ids=selected_memory_id_set,
            missing=missing,
            memory_labels_by_id=memory_labels_by_id,
            score_weights=score_weights,
        )
        if not candidates:
            break
        _, view, units = candidates[0]
        expansion_steps.append(f"expanded_memory={view.memory_id}")
        expansions += 1
        selected_memory_id_set.add(view.memory_id)
        available_units.extend(units)
        seed_units, coverage, seed_steps = _select_greedy_units(
            row=row,
            plan=plan,
            query_targets=query_targets,
            available_units=[unit for unit in available_units if unit.memory_id in selected_memory_id_set],
            budget_window=compile_budget_window,
            memory_labels_by_id=memory_labels_by_id,
            initial_selected_memory_ids=list(selected_memory_id_set),
            score_weights=score_weights,
        )
        selected_units = seed_units
        selected_units = _expand_trusted_memory_units(
            selected_units=selected_units,
            available_units=[unit for unit in available_units if unit.memory_id in selected_memory_id_set],
            max_compiled_tokens=max_compiled_tokens,
            memory_labels_by_id=memory_labels_by_id,
        )
        coverage = coverage_mod._coverage_for_selected_units(
            requirements=selection_requirements,
            selected_units=selected_units,
            row=row,
            plan=plan,
            query_targets=query_targets,
            memory_labels_by_id=memory_labels_by_id,
        )
        expansion_steps.extend(seed_steps[1:])

    base_satisfied = req.plan_satisfied(selection_requirements, coverage)
    role_satisfied = _role_specific_sufficient(
        plan=plan,
        requirements=requirements,
        selected_units=selected_units,
        row=row,
        query_targets=query_targets,
        memory_labels_by_id=memory_labels_by_id,
    ) if base_satisfied else False
    if not (base_satisfied and role_satisfied):
        selected_units, fallback_steps = _augment_with_anchor_fallback(
            row=row,
            plan=plan,
            query_targets=query_targets,
            selected_units=selected_units,
            available_units=available_units,
            ranked_views=ranked_views,
            max_compiled_tokens=max_compiled_tokens,
            memory_labels_by_id=memory_labels_by_id,
            score_weights=score_weights,
        )
        if fallback_steps:
            expansion_steps.extend(fallback_steps)
        elif plan.schema_name == "TemporalInterval":
            newly_selected_memory_ids = [
                unit.memory_id
                for unit in selected_units
                if unit.memory_id not in selected_memory_ids
                and "answer_bearing" in set(memory_labels_by_id.get(unit.memory_id, {}).get("labels", []))
            ]
            if newly_selected_memory_ids:
                expansion_steps.append(f"fallback_anchor_memory={newly_selected_memory_ids[0]}")

    selected_units = sorted(selected_units, key=_unit_sort_key)
    compiled_token_count = sum(unit.token_count for unit in selected_units)
    sufficient = req.plan_satisfied(requirements, coverage) and _role_specific_sufficient(
        plan=plan,
        requirements=requirements,
        selected_units=selected_units,
        row=row,
        query_targets=query_targets,
        memory_labels_by_id=memory_labels_by_id,
    )
    if compiled_token_count > max_compiled_tokens:
        sufficient = False

    if not sufficient:
        used_fallback = True

    missing_slots = req.missing_requirements(requirements, coverage)
    required_total = max(1, sum(int(required) for required in requirements.values()))
    covered_total = sum(
        min(int(required), int(coverage.get(slot_name, 0)))
        for slot_name, required in requirements.items()
    )
    slot_completion_ratio = float(covered_total) / float(required_total)
    budget_base = max(1, int(effective_target_budget) if int(effective_target_budget or 0) > 0 else int(compiled_token_count))
    token_budget_ratio = float(compiled_token_count) / float(budget_base)
    minimality_ratio = max(0.0, 1.0 - min(1.0, token_budget_ratio))

    compiled_units, compiled_texts, compiled_memory_ids, raw_slot_bindings, answer_contract, semantic_analysis = materialization_mod._materialize_compiler_outputs(
        row=row,
        plan=plan,
        query_targets=query_targets,
        selected_units=selected_units,
        score_weights=score_weights,
        memory_labels_by_id=memory_labels_by_id,
        validate_slot_binding=_validate_slot_binding,
        infer_missing_slot_invalid_reason=_infer_missing_slot_invalid_reason,
        promote_support_grounding=_promote_support_grounding,
        apply_reference_time_fallback=_apply_reference_time_fallback,
        rescue_current_state_where_binding=_rescue_current_state_where_binding,
        rescue_information_extraction_where_binding=_rescue_information_extraction_where_binding,
        rescue_temporal_when_event_binding=_rescue_temporal_when_event_binding,
        binding_source_text=rb._binding_source_text,
        extract_numeric_mentions=rb._extract_numeric_mentions,
        count_numeric_spans_excluding_dates=rb._count_numeric_spans_excluding_dates,
        schema_grounding_analysis=_schema_grounding_analysis,
        execute_distinct_count_from_compiled_units=evaluation_mod._execute_distinct_count_from_compiled_units,
        execute_percentage_aggregate_from_bindings=evaluation_mod._execute_percentage_aggregate_from_bindings,
        execute_aggregate_from_compiled_units=evaluation_mod._execute_aggregate_from_compiled_units,
        execute_direct_duration_from_compiled_units=evaluation_mod._execute_direct_duration_from_compiled_units,
        execute_relative_time=evaluation_mod._execute_relative_time,
        execute_ordered_choice=evaluation_mod._execute_ordered_choice,
        execute_comparison=evaluation_mod._execute_comparison,
        execute_temporal_interval=evaluation_mod._execute_temporal_interval,
        deterministic_span_allowed=_deterministic_span_allowed,
        parse_numeric_text=rb._parse_numeric_text,
        format_numeric_answer=rb._format_numeric_answer,
        soft_invalid_reasons=rb._SOFT_INVALID_REASONS,
        query_focus_tokens=rb._query_focus_tokens,
    )
    seed_slot = _single_value_seed_slot(plan)
    if (
        seed_slot is not None
        and not isinstance((answer_contract.get("validated_slot_bindings") or {}).get(seed_slot.slot_name), dict)
        and available_units
    ):
        proposal_order = {
            view.memory_id: index
            for index, view in enumerate(selected_views)
        }
        current_order = min(
            (proposal_order.get(memory_id, len(selected_views)) for memory_id in compiled_memory_ids),
            default=len(selected_views),
        )
        rescue_pool = [
            unit
            for unit in available_units
            if proposal_order.get(unit.memory_id, len(selected_views)) < current_order
        ]
        rescue_unit = _validated_single_value_seed_unit(
            row=row,
            plan=plan,
            query_targets=query_targets,
            available_units=rescue_pool,
            memory_labels_by_id=memory_labels_by_id,
            score_weights=score_weights,
            validate_slot_binding=_validate_slot_binding,
        )
        if rescue_unit is not None:
            selected_units = [rescue_unit]
            selected_units = _expand_trusted_memory_units(
                selected_units=selected_units,
                available_units=available_units,
                max_compiled_tokens=max_compiled_tokens,
                memory_labels_by_id=memory_labels_by_id,
            )
            coverage = coverage_mod._coverage_for_selected_units(
                requirements=selection_requirements,
                selected_units=selected_units,
                row=row,
                plan=plan,
                query_targets=query_targets,
                memory_labels_by_id=memory_labels_by_id,
            )
            compiled_token_count = sum(unit.token_count for unit in selected_units)
            expansion_steps.append(f"validated_rescue_unit={rescue_unit.unit_id}")
            compiled_units, compiled_texts, compiled_memory_ids, raw_slot_bindings, answer_contract, semantic_analysis = materialization_mod._materialize_compiler_outputs(
                row=row,
                plan=plan,
                query_targets=query_targets,
                selected_units=selected_units,
                score_weights=score_weights,
                memory_labels_by_id=memory_labels_by_id,
                validate_slot_binding=_validate_slot_binding,
                infer_missing_slot_invalid_reason=_infer_missing_slot_invalid_reason,
                promote_support_grounding=_promote_support_grounding,
                apply_reference_time_fallback=_apply_reference_time_fallback,
                rescue_current_state_where_binding=_rescue_current_state_where_binding,
                rescue_information_extraction_where_binding=_rescue_information_extraction_where_binding,
                rescue_temporal_when_event_binding=_rescue_temporal_when_event_binding,
                binding_source_text=rb._binding_source_text,
                extract_numeric_mentions=rb._extract_numeric_mentions,
                count_numeric_spans_excluding_dates=rb._count_numeric_spans_excluding_dates,
                schema_grounding_analysis=_schema_grounding_analysis,
                execute_distinct_count_from_compiled_units=evaluation_mod._execute_distinct_count_from_compiled_units,
                execute_percentage_aggregate_from_bindings=evaluation_mod._execute_percentage_aggregate_from_bindings,
                execute_aggregate_from_compiled_units=evaluation_mod._execute_aggregate_from_compiled_units,
                execute_direct_duration_from_compiled_units=evaluation_mod._execute_direct_duration_from_compiled_units,
                execute_relative_time=evaluation_mod._execute_relative_time,
                execute_ordered_choice=evaluation_mod._execute_ordered_choice,
                execute_comparison=evaluation_mod._execute_comparison,
                execute_temporal_interval=evaluation_mod._execute_temporal_interval,
                deterministic_span_allowed=_deterministic_span_allowed,
                parse_numeric_text=rb._parse_numeric_text,
                format_numeric_answer=rb._format_numeric_answer,
                soft_invalid_reasons=rb._SOFT_INVALID_REASONS,
                query_focus_tokens=rb._query_focus_tokens,
            )
    if not any(isinstance(binding, dict) for binding in dict(answer_contract.get("validated_slot_bindings") or {}).values()):
        preferred_units = _preferred_user_evidence_units(
            row=row,
            plan=plan,
            available_units=available_units,
            memory_labels_by_id=memory_labels_by_id,
            score_weights=score_weights,
            max_compiled_tokens=max_compiled_tokens,
        )
        if preferred_units:
            selected_units = _expand_trusted_memory_units(
                selected_units=preferred_units,
                available_units=available_units,
                max_compiled_tokens=max_compiled_tokens,
                memory_labels_by_id=memory_labels_by_id,
            )
            coverage = coverage_mod._coverage_for_selected_units(
                requirements=selection_requirements,
                selected_units=selected_units,
                row=row,
                plan=plan,
                query_targets=query_targets,
                memory_labels_by_id=memory_labels_by_id,
            )
            compiled_token_count = sum(unit.token_count for unit in selected_units)
            expansion_steps.append("preferred_user_evidence_rescue=applied")
            compiled_units, compiled_texts, compiled_memory_ids, raw_slot_bindings, answer_contract, semantic_analysis = materialization_mod._materialize_compiler_outputs(
                row=row,
                plan=plan,
                query_targets=query_targets,
                selected_units=selected_units,
                score_weights=score_weights,
                memory_labels_by_id=memory_labels_by_id,
                validate_slot_binding=_validate_slot_binding,
                infer_missing_slot_invalid_reason=_infer_missing_slot_invalid_reason,
                promote_support_grounding=_promote_support_grounding,
                apply_reference_time_fallback=_apply_reference_time_fallback,
                rescue_current_state_where_binding=_rescue_current_state_where_binding,
                rescue_information_extraction_where_binding=_rescue_information_extraction_where_binding,
                rescue_temporal_when_event_binding=_rescue_temporal_when_event_binding,
                binding_source_text=rb._binding_source_text,
                extract_numeric_mentions=rb._extract_numeric_mentions,
                count_numeric_spans_excluding_dates=rb._count_numeric_spans_excluding_dates,
                schema_grounding_analysis=_schema_grounding_analysis,
                execute_distinct_count_from_compiled_units=evaluation_mod._execute_distinct_count_from_compiled_units,
                execute_percentage_aggregate_from_bindings=evaluation_mod._execute_percentage_aggregate_from_bindings,
                execute_aggregate_from_compiled_units=evaluation_mod._execute_aggregate_from_compiled_units,
                execute_direct_duration_from_compiled_units=evaluation_mod._execute_direct_duration_from_compiled_units,
                execute_relative_time=evaluation_mod._execute_relative_time,
                execute_ordered_choice=evaluation_mod._execute_ordered_choice,
                execute_comparison=evaluation_mod._execute_comparison,
                execute_temporal_interval=evaluation_mod._execute_temporal_interval,
                deterministic_span_allowed=_deterministic_span_allowed,
                parse_numeric_text=rb._parse_numeric_text,
                format_numeric_answer=rb._format_numeric_answer,
                soft_invalid_reasons=rb._SOFT_INVALID_REASONS,
                query_focus_tokens=rb._query_focus_tokens,
            )
    rescue_used = False
    if _should_expand_for_sufficiency(
        row=row,
        plan=plan,
        query_targets=query_targets,
        compiled_memory_ids=compiled_memory_ids,
        answer_contract=answer_contract,
        semantic_analysis=semantic_analysis,
    ):
        rescue_missing = req.validated_rescue_missing(
            plan,
            dict(answer_contract.get("validated_slot_bindings") or {}),
            dict(answer_contract.get("invalid_slots") or {}),
        )
        rescued_units, rescued_coverage, rescue_steps = sufficiency_mod._rescue_sufficiency_with_more_evidence(
            row=row,
            plan=plan,
            query_targets=query_targets,
            ranked_views=ranked_views,
            available_units=available_units,
            selected_units=selected_units,
            budget_window=compile_budget_window,
            max_compiled_tokens=max_compiled_tokens,
            memory_labels_by_id=memory_labels_by_id,
            score_weights=score_weights,
            missing_override=rescue_missing,
        )
        if rescue_steps:
            selected_units = sorted(rescued_units, key=_unit_sort_key)
            coverage = rescued_coverage
            compiled_token_count = sum(unit.token_count for unit in selected_units)
            expansion_steps.extend(rescue_steps)
            compiled_units, compiled_texts, compiled_memory_ids, raw_slot_bindings, answer_contract, semantic_analysis = materialization_mod._materialize_compiler_outputs(
                row=row,
                plan=plan,
                query_targets=query_targets,
                selected_units=selected_units,
                score_weights=score_weights,
                memory_labels_by_id=memory_labels_by_id,
                validate_slot_binding=_validate_slot_binding,
                infer_missing_slot_invalid_reason=_infer_missing_slot_invalid_reason,
                promote_support_grounding=_promote_support_grounding,
                apply_reference_time_fallback=_apply_reference_time_fallback,
                rescue_current_state_where_binding=_rescue_current_state_where_binding,
                rescue_information_extraction_where_binding=_rescue_information_extraction_where_binding,
                rescue_temporal_when_event_binding=_rescue_temporal_when_event_binding,
                binding_source_text=rb._binding_source_text,
                extract_numeric_mentions=rb._extract_numeric_mentions,
                count_numeric_spans_excluding_dates=rb._count_numeric_spans_excluding_dates,
                schema_grounding_analysis=_schema_grounding_analysis,
                execute_distinct_count_from_compiled_units=evaluation_mod._execute_distinct_count_from_compiled_units,
                execute_percentage_aggregate_from_bindings=evaluation_mod._execute_percentage_aggregate_from_bindings,
                execute_aggregate_from_compiled_units=evaluation_mod._execute_aggregate_from_compiled_units,
                execute_direct_duration_from_compiled_units=evaluation_mod._execute_direct_duration_from_compiled_units,
                execute_relative_time=evaluation_mod._execute_relative_time,
                execute_ordered_choice=evaluation_mod._execute_ordered_choice,
                execute_comparison=evaluation_mod._execute_comparison,
                execute_temporal_interval=evaluation_mod._execute_temporal_interval,
                deterministic_span_allowed=_deterministic_span_allowed,
                parse_numeric_text=rb._parse_numeric_text,
                format_numeric_answer=rb._format_numeric_answer,
                soft_invalid_reasons=rb._SOFT_INVALID_REASONS,
                query_focus_tokens=rb._query_focus_tokens,
            )
            rescue_used = True
    pre_downgrade_answer_mode = str(answer_contract.get("answer_mode") or "")
    pre_downgrade_answerability = str(answer_contract.get("answerability_level") or "")
    answer_contract = apply_semantic_reader_downgrade(answer_contract, semantic_analysis)
    semantic_downgraded = bool(
        str(answer_contract.get("answer_mode") or "") != pre_downgrade_answer_mode
        or str(answer_contract.get("answerability_level") or "") != pre_downgrade_answerability
    )
    compiler_prompt_variant = reader_prompt_variant(
        plan=plan,
        answer_contract=answer_contract,
        semantic_analysis=semantic_analysis,
        compiled_unit_count=len(compiled_units),
        compiled_token_count=compiled_token_count,
    )
    slot_bindings = dict(answer_contract["validated_slot_bindings"])
    missing_slots = req.validated_missing_requirements(plan, slot_bindings)
    sufficient = bool(
        sufficient
        and answer_contract["compiler_answerable"]
        and semantic_analysis["semantic_sufficiency_passed"]
    )
    if rescue_used or not sufficient:
        used_fallback = True
    budget_overshoot_reasons = sorted(
        {
            str(step).split("=", 1)[1].split(":", 1)[0]
            for step in expansion_steps
            if str(step).startswith("soft_budget_overshoot=")
        }
    )

    return {
        "compiled_memory_ids": compiled_memory_ids,
        "compiled_units": compiled_units,
        "compiled_texts": compiled_texts,
        "slot_bindings": slot_bindings,
        "raw_slot_bindings": raw_slot_bindings,
        "compiled_token_count": compiled_token_count,
        "effective_target_budget": effective_target_budget,
        "budget_hard_cap": budget_hard_cap,
        "budget_throttle_reason": budget_throttle_reason,
        "budget_throttle_ratio": budget_throttle_ratio,
        "budget_window": {
            "target_tokens": compile_budget_window.target_tokens,
            "hard_tokens": compile_budget_window.hard_tokens,
            "max_selected": compile_budget_window.max_selected,
            "min_distinct_memories": compile_budget_window.min_distinct_memories,
            "min_date_contexts": compile_budget_window.min_date_contexts,
            "rationale": list(compile_budget_window.rationale),
        },
        "budget_overshoot_reasons": budget_overshoot_reasons,
        "requirements": requirements,
        "coverage": coverage,
        "sufficient": sufficient,
        "expansion_steps": expansion_steps,
        "used_fallback": used_fallback,
        "rescue_used": rescue_used,
        "semantic_downgraded": semantic_downgraded,
        "missing_slots": missing_slots,
        "answer_mode": answer_contract["answer_mode"],
        "compiler_answerable": bool(answer_contract["compiler_answerable"]),
        "validated_slot_bindings": dict(answer_contract["validated_slot_bindings"]),
        "invalid_slots": dict(answer_contract["invalid_slots"]),
        "answer_text": answer_contract["answer_text"],
        "supporting_unit_ids": list(answer_contract["supporting_unit_ids"]),
        "answerability_level": answer_contract.get("answerability_level"),
        "deterministic_allowed": answer_contract.get("deterministic_allowed"),
        "reader_route_reason": answer_contract.get("reader_route_reason"),
        "reader_prompt_variant": compiler_prompt_variant,
        "slot_grounding_scores": dict(answer_contract.get("slot_grounding_scores") or {}),
        "slot_grounding_fail_reasons": dict(answer_contract.get("slot_grounding_fail_reasons") or {}),
        "schema_name": answer_contract.get("schema_name"),
        "question_subtype": answer_contract.get("question_subtype"),
        "semantic_support_scores": dict(semantic_analysis.get("semantic_support_scores") or {}),
        "semantic_support_mean": semantic_analysis.get("semantic_support_mean"),
        "slot_grounding_mean": semantic_analysis.get("slot_grounding_mean"),
        "semantic_sufficiency_score": semantic_analysis.get("semantic_sufficiency_score"),
        "semantic_sufficiency_passed": semantic_analysis.get("semantic_sufficiency_passed"),
        "semantic_sufficiency_applied": semantic_analysis.get("semantic_sufficiency_applied"),
        "semantic_sufficiency_reasons": list(semantic_analysis.get("semantic_sufficiency_reasons") or []),
        "projected_candidates": [
            {
                "unit_id": candidate.unit_id,
                "memory_id": candidate.memory_id,
                "slot_names": list(candidate.slot_names),
                "token_count": candidate.token_count,
                "slot_utility": round(candidate.slot_utility, 6),
            }
            for candidate in projected_candidates
        ],
        "slot_completion_ratio": round(slot_completion_ratio, 6),
        "token_budget_ratio": round(token_budget_ratio, 6),
        "minimality_ratio": round(minimality_ratio, 6),
    }
