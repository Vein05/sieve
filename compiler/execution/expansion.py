"""Expansion helpers for compiler execution selection."""

from __future__ import annotations

from typing import Any

from ..runtime.bindings import (
    CandidateView,
    EvidencePlan,
    EvidenceUnit,
    QueryTargets,
    _best_matching_entity,
    _missing_role_gain,
    _query_focus_tokens,
    _selected_unit_roles,
    _unit_has_strong_focus_alignment,
    _unit_requirement_tags_cached,
    _unit_semantic_gate_allows_anchor,
    cached_build_profile,
    score_evidence_unit,
)
from evidence.units import build_evidence_units_for_views
from .coverage import _coverage_for_selected_units
from .greedy import _select_greedy_units
from .requirements import (
    is_current_state_schema as _is_current_state_schema,
    is_direct_lookup_schema as _is_direct_lookup_schema,
    is_state_update_schema as _is_state_update_schema,
    is_support_lookup_schema as _is_support_lookup_schema,
    selection_requirements_dict as _selection_requirements_dict,
)


def _expansion_candidates(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets=None,
    ranked_views: list[CandidateView],
    selected_memory_ids: set[str],
    selected_date_keys: set[tuple[int, int, int]] | None = None,
    missing: dict[str, int],
    memory_labels_by_id: dict[str, dict[str, Any]],
    score_weights: dict[str, float],
) -> list[tuple[float, CandidateView, list[EvidenceUnit]]]:
    ranked_candidates: list[tuple[tuple[float, float, float, float, int, str], CandidateView, list[EvidenceUnit]]] = []
    role_cache: dict[str, set[str]] = {}
    from shared.perf import get_counters as _get_pc
    for view in ranked_views:
        if view.memory_id in selected_memory_ids:
            continue
        _get_pc().expansion_view_parses += 1
        units = build_evidence_units_for_views([view])
        if not units:
            continue
        tags = set()
        best_score = 0.0
        best_token_count: int | None = None
        for unit in units:
            unit_tags = _unit_requirement_tags_cached(
                row=row,
                plan=plan,
                query_targets=query_targets,
                unit=unit,
                memory_labels_by_id=memory_labels_by_id,
                role_cache=role_cache,
            )
            if not _unit_semantic_gate_allows_anchor(
                row=row,
                plan=plan,
                query_targets=query_targets,
                unit=unit,
                tags=unit_tags,
                expansion=True,
            ):
                continue
            tags |= unit_tags
            best_score = max(
                best_score,
                score_evidence_unit(
                    row=row,
                    query_family=plan.query_family,
                    unit=unit,
                    memory_labels_by_id=memory_labels_by_id,
                    score_weights=score_weights,
                    plan=plan,
                ),
            )
            best_token_count = unit.token_count if best_token_count is None else min(best_token_count, unit.token_count)
        covered_count, weighted_gain = _missing_role_gain(tags, missing)
        # Accept any memory that passed the semantic gate when the compiler
        # is looking for competing evidence (plan satisfied but single-source).
        if weighted_gain <= 0 and "competing_anchor" in missing and tags:
            weighted_gain = 1
            covered_count = 1
        if weighted_gain <= 0:
            continue
        role_spread = float(len(tags.intersection(missing)))
        diversity_bonus = 0.0
        paired_plan = str(getattr(plan, "schema_name", "") or "") in {
            "SumOperands",
            "TemporalInterval",
            "DifferenceAggregate",
            "PercentageAggregate",
            "OrderedChoice",
        }
        paired_roles = (
            {"comparison_left", "comparison_right"},
            {"temporal_event_a", "temporal_event_b"},
            {"temporal_time_a", "temporal_time_b"},
            {"old_state", "new_state"},
        )
        for pair in paired_roles:
            if pair <= tags and pair.intersection(missing):
                diversity_bonus += 0.5
        if paired_plan and missing and any(role in missing for role in ("operand_1", "operand_2", "event_a", "event_b", "time_a", "time_b", "choice_a", "choice_b", "comparison_left", "comparison_right")):
            date_key = tuple(getattr(view, "date_key", ()) or ())
            if selected_date_keys and date_key and date_key not in selected_date_keys:
                diversity_bonus += 0.4
            elif selected_date_keys and date_key and date_key in selected_date_keys:
                diversity_bonus -= 0.15
        if {"direct_anchor", "support_anchor"} <= tags and tags.intersection(missing):
            diversity_bonus += 0.25
        rank_prior = 1.0 / max(1, int(view.rank))
        ranked_candidates.append(
            (
                (
                    float(weighted_gain),
                    float(covered_count),
                    best_score,
                    rank_prior + diversity_bonus + (0.25 * role_spread),
                    -int(best_token_count or 0),
                    view.memory_id,
                ),
                view,
                units,
            )
        )
    ranked_candidates.sort(key=lambda item: item[0], reverse=True)
    return [(float(rank[0]), view, units) for rank, view, units in ranked_candidates]


def _augment_with_anchor_fallback(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets: QueryTargets,
    selected_units: list[EvidenceUnit],
    available_units: list[EvidenceUnit],
    ranked_views: list[CandidateView],
    max_compiled_tokens: int,
    memory_labels_by_id: dict[str, dict[str, Any]],
    score_weights: dict[str, float],
) -> tuple[list[EvidenceUnit], list[str]]:
    _, role_units, _ = _selected_unit_roles(
        selected_units=selected_units,
        row=row,
        plan=plan,
        query_targets=query_targets,
        memory_labels_by_id=memory_labels_by_id,
    )
    needs_direct = not (_is_direct_lookup_schema(plan) or _is_current_state_schema(plan) or _is_state_update_schema(plan)) or not role_units.get("direct_anchor")
    needs_support = (
        (_is_support_lookup_schema(plan) or query_targets.asks_for_recall_support)
        and not role_units.get("support_anchor")
    )
    if not needs_direct and not needs_support:
        return selected_units, []

    selected_unit_ids = {unit.unit_id for unit in selected_units}
    selected_memory_ids = {unit.memory_id for unit in selected_units}
    current_tokens = sum(unit.token_count for unit in selected_units)
    role_cache: dict[str, set[str]] = {}
    pool: list[EvidenceUnit] = list(available_units)
    seen_memory_ids = {unit.memory_id for unit in available_units}
    for view in ranked_views:
        if view.memory_id in seen_memory_ids:
            continue
        pool.extend(build_evidence_units_for_views([view]))
        seen_memory_ids.add(view.memory_id)

    chosen_units: list[EvidenceUnit] = []
    steps: list[str] = []
    target_roles = ["direct_anchor"]
    if needs_support:
        target_roles.append("support_anchor")

    for target_role in target_roles:
        best_unit: EvidenceUnit | None = None
        best_key: tuple[float, ...] | None = None
        for unit in pool:
            if unit.unit_id in selected_unit_ids:
                continue
            projected_total = current_tokens + sum(item.token_count for item in chosen_units) + unit.token_count
            if max_compiled_tokens > 0 and projected_total > max_compiled_tokens:
                continue
            tags = _unit_requirement_tags_cached(
                row=row,
                plan=plan,
                query_targets=query_targets,
                unit=unit,
                memory_labels_by_id=memory_labels_by_id,
                role_cache=role_cache,
            )
            if not _unit_semantic_gate_allows_anchor(
                row=row,
                plan=plan,
                query_targets=query_targets,
                unit=unit,
                tags=tags,
            ):
                continue
            covers_target = target_role in tags or (
                target_role == "direct_anchor"
                and _is_support_lookup_schema(plan)
                and "support_anchor" in tags
            )
            if not covers_target:
                continue
            key = (
                1.0 if unit.memory_id not in selected_memory_ids else 0.0,
                1.0 if "direct_anchor" in tags else 0.0,
                1.0 if "support_anchor" in tags else 0.0,
                score_evidence_unit(
                    row=row,
                    query_family=plan.query_family,
                    unit=unit,
                    memory_labels_by_id=memory_labels_by_id,
                    score_weights=score_weights,
                    plan=plan,
                ),
                1.0 / max(1, unit.parent_rank),
                -float(unit.token_count),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_unit = unit
        if best_unit is None:
            continue
        chosen_units.append(best_unit)
        selected_unit_ids.add(best_unit.unit_id)
        steps.append(f"fallback_anchor_memory={best_unit.memory_id}")

    if not chosen_units:
        return selected_units, []
    return selected_units + chosen_units, steps
