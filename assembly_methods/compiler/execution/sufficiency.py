"""Sufficiency and rescue helpers for compiler execution selection."""

from __future__ import annotations

from typing import Any

from .adaptive_budget import BudgetWindow
from ..runtime.text import _build_compiled_unit_analysis
from ..runtime.bindings import (
    CandidateView,
    EvidencePlan,
    EvidenceUnit,
    QueryTargets,
    _best_matching_entity,
    _query_focus_tokens,
    _same_memory_pair_sufficient,
    _unit_has_strong_focus_alignment,
    _selected_unit_roles,
    _WEAK_QUERY_FOCUS_TOKENS,
    score_evidence_unit,
)
from . import requirements as req
from .requirements import is_aggregate_schema, is_count_aggregate_schema, is_numeric_aggregate_schema
from ...evidence_units import build_evidence_units_for_views
from .coverage import _coverage_for_selected_units
from .expansion import _augment_with_anchor_fallback, _expansion_candidates
from .greedy import _expand_trusted_memory_units, _select_greedy_units, _unit_sort_key
from .requirements import (
    missing_requirements as _missing_requirements,
    plan_satisfied as _plan_satisfied,
    requirements_dict as _requirements_dict,
    selection_requirements_dict as _selection_requirements_dict,
)


def _role_specific_sufficient(
    *,
    plan: EvidencePlan,
    requirements: dict[str, int],
    selected_units: list[EvidenceUnit],
    row: dict[str, Any],
    query_targets=None,
    memory_labels_by_id: dict[str, dict[str, Any]],
) -> bool:
    if not selected_units:
        return False

    _, role_units, memory_roles = _selected_unit_roles(
        selected_units=selected_units,
        row=row,
        plan=plan,
        query_targets=query_targets,
        memory_labels_by_id=memory_labels_by_id,
    )

    def _pair_has_coverage(role_a: str, role_b: str) -> bool:
        pair_units = role_units.get(role_a, []) + role_units.get(role_b, [])
        if not pair_units:
            return False
        memory_ids = {unit.memory_id for unit in pair_units}
        if len(memory_ids) >= 2:
            return True
        memory_id = next(iter(memory_ids))
        if plan.query_family == "multi_session":
            return False
        if not {role_a, role_b}.issubset(memory_roles.get(memory_id, set())):
            return False
        return _same_memory_pair_sufficient(
            plan=plan,
            query_targets=query_targets,
            role_a=role_a,
            role_b=role_b,
            pair_units=[unit for unit in pair_units if unit.memory_id == memory_id],
        )

    def _shared_recall_unit_is_grounded(unit: EvidenceUnit) -> bool:
        source_text = str(unit.provenance.get("source_text") or unit.render_text or "").strip()
        source_analysis = _build_compiled_unit_analysis(source_text)
        primary_entities = tuple(entity for entity in getattr(query_targets, "subject_entities", ()) if entity) or tuple(
            entity for entity in getattr(query_targets, "candidate_entities", ())[:2] if entity
        )
        if primary_entities:
            if _best_matching_entity(source_text, primary_entities):
                return True
        focus_overlap = (
            set(source_analysis.content_tokens)
            & (_query_focus_tokens(row) - _WEAK_QUERY_FOCUS_TOKENS)
        )
        return len(focus_overlap) >= 2

    if requirements.get("support_anchor", 0) > 0:
        direct_units = role_units.get("direct_anchor", [])
        support_units = role_units.get("support_anchor", [])
        if not direct_units or not support_units:
            return False
        direct_unit_ids = {unit.unit_id for unit in direct_units}
        support_unit_ids = {unit.unit_id for unit in support_units}
        if direct_unit_ids & support_unit_ids:
            if bool(getattr(query_targets, "asks_for_recall_support", False)):
                shared_units = [
                    unit
                    for unit in direct_units + support_units
                    if unit.unit_id in direct_unit_ids & support_unit_ids
                ]
                if not any(_shared_recall_unit_is_grounded(unit) for unit in shared_units):
                    return False
        elif len({unit.unit_id for unit in direct_units + support_units}) < 2:
            return False

    if plan.query_family in {"aggregation", "ordering"} and requirements.get("ordering_event", 0) >= 2:
        ordering_units = role_units.get("ordering_event", [])
        if len(ordering_units) < 2:
            return False
        primary_entities = tuple(entity for entity in getattr(query_targets, "subject_entities", ()) if entity) or tuple(
            entity for entity in getattr(query_targets, "candidate_entities", ())[:2] if entity
        )
        if primary_entities:
            matched_entities = {
                _best_matching_entity(
                    str(unit.provenance.get("source_text") or unit.render_text or "").strip(),
                    primary_entities,
                )
                for unit in ordering_units
            }
            matched_entities.discard(None)
            if len(matched_entities) < 2:
                return False

    if plan.query_family == "aggregation" and {"comparison_left", "comparison_right"} <= requirements.keys():
        if not _pair_has_coverage("comparison_left", "comparison_right"):
            return False

    if plan.query_family == "temporal":
        if {"temporal_event_a", "temporal_event_b"} <= requirements.keys():
            if not _pair_has_coverage("temporal_event_a", "temporal_event_b"):
                return False
        if {"temporal_time_a", "temporal_time_b"} <= requirements.keys():
            if not _pair_has_coverage("temporal_time_a", "temporal_time_b"):
                return False

    if plan.query_family == "current_state":
        if not role_units.get("state_anchor") or not role_units.get("current_resolution"):
            return False

    if plan.query_family in {"knowledge_update", "conflict_update"}:
        if not role_units.get("old_state") or not role_units.get("new_state"):
            return False

    return True


def _role_specific_missing(
    *,
    plan: EvidencePlan,
    requirements: dict[str, int],
    selected_units: list[EvidenceUnit],
    row: dict[str, Any],
    query_targets=None,
    memory_labels_by_id: dict[str, dict[str, Any]],
) -> dict[str, int]:
    missing: dict[str, int] = {}
    _, role_units, memory_roles = _selected_unit_roles(
        selected_units=selected_units,
        row=row,
        plan=plan,
        query_targets=query_targets,
        memory_labels_by_id=memory_labels_by_id,
    )

    for role, count in requirements.items():
        if count > 0 and not role_units.get(role):
            missing[role] = 1

    paired_roles = (
        ("comparison_left", "comparison_right"),
        ("temporal_event_a", "temporal_event_b"),
        ("temporal_time_a", "temporal_time_b"),
        ("old_state", "new_state"),
    )
    for left_role, right_role in paired_roles:
        if requirements.get(left_role, 0) <= 0 or requirements.get(right_role, 0) <= 0:
            continue
        pair_units = role_units.get(left_role, []) + role_units.get(right_role, [])
        if not pair_units:
            continue
        memory_ids = {unit.memory_id for unit in pair_units}
        if len(memory_ids) >= 2:
            continue
        memory_id = next(iter(memory_ids))
        if plan.query_family != "multi_session" and {left_role, right_role}.issubset(memory_roles.get(memory_id, set())) and _same_memory_pair_sufficient(
            plan=plan,
            query_targets=query_targets,
            role_a=left_role,
            role_b=right_role,
            pair_units=[unit for unit in pair_units if unit.memory_id == memory_id],
        ):
            continue
        missing[left_role] = max(missing.get(left_role, 0), 1)
        missing[right_role] = max(missing.get(right_role, 0), 1)

    if requirements.get("support_anchor", 0) > 0:
        direct_units = role_units.get("direct_anchor", [])
        support_units = role_units.get("support_anchor", [])
        direct_unit_ids = {unit.unit_id for unit in direct_units}
        support_unit_ids = {unit.unit_id for unit in support_units}
        if bool(getattr(query_targets, "asks_for_recall_support", False)) and direct_unit_ids & support_unit_ids:
            shared_units = [
                unit
                for unit in direct_units + support_units
                if unit.unit_id in direct_unit_ids & support_unit_ids
            ]
            grounded_shared = False
            for unit in shared_units:
                source_text = str(unit.provenance.get("source_text") or unit.render_text or "").strip()
                source_analysis = _build_compiled_unit_analysis(source_text)
                primary_entities = tuple(entity for entity in getattr(query_targets, "subject_entities", ()) if entity) or tuple(
                    entity for entity in getattr(query_targets, "candidate_entities", ())[:2] if entity
                )
                if primary_entities and _best_matching_entity(source_text, primary_entities):
                    grounded_shared = True
                    break
                overlap = set(source_analysis.content_tokens) & (_query_focus_tokens(row) - _WEAK_QUERY_FOCUS_TOKENS)
                if len(overlap) >= 2:
                    grounded_shared = True
                    break
            if not grounded_shared:
                missing["support_anchor"] = max(missing.get("support_anchor", 0), 1)
        if not (direct_units and support_units and (direct_unit_ids & support_unit_ids)) and len(
            {unit.unit_id for unit in direct_units + support_units}
        ) < 2:
            missing["support_anchor"] = max(missing.get("support_anchor", 0), 1)

    if plan.query_family in {"aggregation", "ordering"} and requirements.get("ordering_event", 0) >= 2:
        ordering_units = role_units.get("ordering_event", [])
        primary_entities = tuple(entity for entity in getattr(query_targets, "alternative_entities", ()) if entity) or tuple(
            entity for entity in getattr(query_targets, "subject_entities", ()) if entity
        ) or tuple(entity for entity in getattr(query_targets, "candidate_entities", ())[:2] if entity)
        if not ordering_units:
            missing["ordering_event"] = max(missing.get("ordering_event", 0), 1)
        elif primary_entities:
            matched_entities = {
                _best_matching_entity(
                    str(unit.provenance.get("source_text") or unit.render_text or "").strip(),
                    primary_entities,
                )
                for unit in ordering_units
            }
            matched_entities.discard(None)
            if len(matched_entities) < 2:
                missing["ordering_event"] = max(missing.get("ordering_event", 0), 1)

    return missing


def _schema_rescue_missing(plan: EvidencePlan) -> dict[str, int]:
    if is_numeric_aggregate_schema(plan):
        return {"numeric_operand": 1}
    if is_count_aggregate_schema(plan):
        return {"count_item" if str(plan.schema_name or "") == "CountDistinctItems" else "count_evidence": 1}
    if req.is_comparison_schema(plan):
        return {"comparison_left": 1, "comparison_right": 1}
    if plan.schema_name == "OrderedChoice":
        return {"ordering_event": 1}
    if plan.schema_name == "TemporalInterval":
        return {
            "temporal_event_a": 1,
            "temporal_event_b": 1,
            "temporal_time_a": 1,
            "temporal_time_b": 1,
        }
    if plan.schema_name == "RelativeTime":
        return {"temporal_event_a": 1, "reference_time": 1}
    if req.is_current_state_schema(plan):
        return {"state_anchor": 1, "current_resolution": 1}
    if req.is_state_update_schema(plan):
        return {"old_state": 1, "new_state": 1}
    if req.is_support_lookup_schema(plan):
        return {"support_anchor": 1}
    return {}


def _desired_schema_memory_breadth(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets: QueryTargets,
) -> int:
    if is_numeric_aggregate_schema(plan) or req.is_comparison_schema(plan) or plan.schema_name in {"OrderedChoice", "TemporalInterval"} or req.is_state_update_schema(plan):
        return 2
    if is_count_aggregate_schema(plan):
        return 1
    if plan.schema_name == "RelativeTime":
        from ..runtime.text import _query_has_relative_reference_clause

        return 2 if _query_has_relative_reference_clause(row) else 1
    if req.is_current_state_schema(plan):
        if plan.requires_numeric_reasoning:
            return 2
        normalized_query = str(row.get("query", "")).strip().lower()
        if normalized_query.startswith("where ") or normalized_query.startswith("what is the current"):
            return 2
        return 1
    if req.is_support_lookup_schema(plan):
        return 2 if query_targets.asks_for_recall_support else 1
    return 1


def _should_expand_for_sufficiency(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets: QueryTargets,
    compiled_memory_ids: list[str],
    answer_contract: dict[str, Any],
    semantic_analysis: dict[str, Any],
) -> bool:
    desired_breadth = _desired_schema_memory_breadth(
        row=row,
        plan=plan,
        query_targets=query_targets,
    )
    if desired_breadth <= 1:
        return False
    current_breadth = len({memory_id for memory_id in compiled_memory_ids if str(memory_id).strip()})
    if current_breadth >= desired_breadth and semantic_analysis.get("semantic_sufficiency_passed"):
        return False
    answer_mode = str(answer_contract.get("answer_mode") or "")
    answerability = str(answer_contract.get("answerability_level") or "")
    route_reason = str(answer_contract.get("reader_route_reason") or "")
    return bool(
        current_breadth < desired_breadth
        and (
            answerability != "deterministic_safe"
            or answer_mode == "reader_from_slots"
            or not semantic_analysis.get("semantic_sufficiency_passed")
            or "partial" in route_reason
        )
    )


def _rebuild_selected_units_for_memory_ids(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets: QueryTargets,
    available_units: list[EvidenceUnit],
    selected_memory_ids: set[str],
    budget_window: BudgetWindow | None,
    max_compiled_tokens: int,
    memory_labels_by_id: dict[str, dict[str, Any]],
    score_weights: dict[str, float],
) -> tuple[list[EvidenceUnit], dict[str, int]]:
    scoped_units = [unit for unit in available_units if unit.memory_id in selected_memory_ids]
    selected_units, coverage, _ = _select_greedy_units(
        row=row,
        plan=plan,
        query_targets=query_targets,
        available_units=scoped_units,
        budget_window=budget_window,
        memory_labels_by_id=memory_labels_by_id,
        initial_selected_memory_ids=sorted(selected_memory_ids),
        score_weights=score_weights,
    )
    selected_units = _expand_trusted_memory_units(
        selected_units=selected_units,
        available_units=scoped_units,
        max_compiled_tokens=max_compiled_tokens,
        memory_labels_by_id=memory_labels_by_id,
    )
    coverage = _coverage_for_selected_units(
        requirements=_selection_requirements_dict(plan),
        selected_units=selected_units,
        row=row,
        plan=plan,
        query_targets=query_targets,
        memory_labels_by_id=memory_labels_by_id,
    )
    return selected_units, coverage


def _rescue_sufficiency_with_more_evidence(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets: QueryTargets,
    ranked_views: list[CandidateView],
    available_units: list[EvidenceUnit],
    selected_units: list[EvidenceUnit],
    budget_window: BudgetWindow | None,
    max_compiled_tokens: int,
    memory_labels_by_id: dict[str, dict[str, Any]],
    score_weights: dict[str, float],
    missing_override: dict[str, int] | None = None,
) -> tuple[list[EvidenceUnit], dict[str, int], list[str]]:
    selected_memory_ids = {unit.memory_id for unit in selected_units}
    if len(selected_memory_ids) >= len({view.memory_id for view in ranked_views}):
        return selected_units, _coverage_for_selected_units(
            requirements=_selection_requirements_dict(plan),
            selected_units=selected_units,
            row=row,
            plan=plan,
            query_targets=query_targets,
            memory_labels_by_id=memory_labels_by_id,
        ), []

    missing = dict(missing_override or {})
    if not missing:
        missing = _role_specific_missing(
            plan=plan,
            requirements=_requirements_dict(plan),
            selected_units=selected_units,
            row=row,
            query_targets=query_targets,
            memory_labels_by_id=memory_labels_by_id,
        )
    if not missing:
        missing = _schema_rescue_missing(plan)
    if not missing:
        return selected_units, _coverage_for_selected_units(
            requirements=_selection_requirements_dict(plan),
            selected_units=selected_units,
            row=row,
            plan=plan,
            query_targets=query_targets,
            memory_labels_by_id=memory_labels_by_id,
        ), []

    candidates = _expansion_candidates(
        row=row,
        plan=plan,
        query_targets=query_targets,
        ranked_views=ranked_views,
        selected_memory_ids=selected_memory_ids,
        missing=missing,
        memory_labels_by_id=memory_labels_by_id,
        score_weights=score_weights,
    )
    if not candidates:
        fallback_ranked: list[tuple[float, CandidateView, list[EvidenceUnit]]] = []
        for view in ranked_views:
            if view.memory_id in selected_memory_ids:
                continue
            units = build_evidence_units_for_views([view])
            if not units:
                continue
            best_score: float | None = None
            for unit in units:
                source_text = str(unit.provenance.get("source_text") or unit.render_text or "").strip()
                primary_entities = tuple(entity for entity in getattr(query_targets, "subject_entities", ()) if entity) or tuple(
                    entity for entity in getattr(query_targets, "candidate_entities", ())[:2] if entity
                )
                entity_match = bool(primary_entities and _best_matching_entity(source_text, primary_entities))
                focus_match = _unit_has_strong_focus_alignment(row=row, unit=unit)
                if not entity_match and not focus_match:
                    continue
                score = score_evidence_unit(
                    row=row,
                    query_family=plan.query_family,
                    unit=unit,
                    memory_labels_by_id=memory_labels_by_id,
                    score_weights=score_weights,
                    plan=plan,
                )
                if entity_match:
                    score += 0.5
                if focus_match:
                    score += 0.25
                if plan.query_family == "temporal" and unit.time_markers:
                    score += 0.5
                best_score = score if best_score is None else max(best_score, score)
            if best_score is None:
                continue
            fallback_ranked.append((best_score, view, units))
        fallback_ranked.sort(key=lambda item: item[0], reverse=True)
        candidates = fallback_ranked

    desired_breadth = _desired_schema_memory_breadth(row=row, plan=plan, query_targets=query_targets)
    rescue_steps: list[str] = []
    remaining_needed = max(1, desired_breadth - len(selected_memory_ids))
    for _, view, units in candidates:
        selected_memory_ids.add(view.memory_id)
        available_units.extend(units)
        rescue_steps.append(f"sufficiency_rescue_memory={view.memory_id}")
        remaining_needed -= 1
        if remaining_needed <= 0:
            break

    rebuilt_units, rebuilt_coverage = _rebuild_selected_units_for_memory_ids(
        row=row,
        plan=plan,
        query_targets=query_targets,
        available_units=available_units,
        selected_memory_ids=selected_memory_ids,
        budget_window=budget_window,
        max_compiled_tokens=max_compiled_tokens,
        memory_labels_by_id=memory_labels_by_id,
        score_weights=score_weights,
    )
    return rebuilt_units, rebuilt_coverage, rescue_steps
