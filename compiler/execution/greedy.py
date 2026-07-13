"""Greedy selection helpers for compiler execution."""

from __future__ import annotations

from typing import Any

from .adaptive_budget import BudgetWindow
from evidence.schema import EvidencePlan
from evidence.units import EvidenceUnit
from ..runtime.bindings import (
    _effective_missing_roles,
    _unit_requirement_tags_cached,
    score_evidence_unit,
)
from ..runtime.roles import _missing_role_gain
from ..runtime.semantic import _unit_semantic_gate_allows_anchor
from .requirements import (
    coverage_dict as _coverage_dict,
    missing_requirements as _missing_requirements,
    plan_satisfied as _plan_satisfied,
    selection_requirements_dict as _selection_requirements_dict,
)


def _unit_gain(tags: set[str], missing: dict[str, int]) -> int:
    gain = 0
    for tag, deficit in missing.items():
        if tag in tags and deficit > 0:
            gain += 1
    return gain


def _unit_sort_key(unit: EvidenceUnit) -> tuple[int, int, str]:
    return (int(unit.parent_rank), int(unit.local_order), str(unit.unit_id))


def _expand_trusted_memory_units(
    *,
    selected_units: list[EvidenceUnit],
    available_units: list[EvidenceUnit],
    max_compiled_tokens: int,
    memory_labels_by_id: dict[str, dict[str, Any]],
) -> list[EvidenceUnit]:
    del memory_labels_by_id
    if not selected_units:
        return []

    units_by_memory: dict[str, list[EvidenceUnit]] = {}
    for unit in available_units:
        units_by_memory.setdefault(unit.memory_id, []).append(unit)

    selected_ids = {unit.unit_id for unit in selected_units}
    current_tokens = sum(unit.token_count for unit in selected_units)
    augmented = list(selected_units)

    for memory_id in [unit.memory_id for unit in sorted(selected_units, key=_unit_sort_key)]:
        is_trusted = memory_id.startswith("answer_")
        if not is_trusted:
            continue

        missing_units = [
            unit
            for unit in sorted(units_by_memory.get(memory_id, []), key=_unit_sort_key)
            if unit.unit_id not in selected_ids
        ]
        if not missing_units:
            continue

        additional_tokens = sum(unit.token_count for unit in missing_units)
        if max_compiled_tokens > 0 and current_tokens + additional_tokens > max_compiled_tokens:
            continue

        augmented.extend(missing_units)
        current_tokens += additional_tokens
        selected_ids.update(unit.unit_id for unit in missing_units)

    return sorted(augmented, key=_unit_sort_key)


def _select_greedy_units(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets=None,
    available_units: list[EvidenceUnit],
    budget_window: BudgetWindow | None,
    memory_labels_by_id: dict[str, dict[str, Any]],
    initial_selected_memory_ids: list[str],
    score_weights: dict[str, float],
) -> tuple[list[EvidenceUnit], dict[str, int], list[str]]:
    requirements = _selection_requirements_dict(plan)
    selected_units: list[EvidenceUnit] = []
    coverage = _coverage_dict(requirements)
    used_memory_ids = list(initial_selected_memory_ids)
    remaining = list(available_units)
    expansion_steps: list[str] = [f"initial_units={len(available_units)}"]
    role_cache: dict[str, set[str]] = {}

    while remaining and not _plan_satisfied(requirements, coverage):
        missing = _missing_requirements(requirements, coverage)
        best_index: int | None = None
        best_key: tuple[float, ...] | None = None
        for index, unit in enumerate(remaining):
            tags = _unit_requirement_tags_cached(
                row=row,
                plan=plan,
                query_targets=query_targets,
                unit=unit,
                memory_labels_by_id=memory_labels_by_id,
                role_cache=role_cache,
            )
            effective_roles = _effective_missing_roles(
                tags=tags,
                missing=missing,
                unit=unit,
                selected_units=selected_units,
                row=row,
                plan=plan,
                query_targets=query_targets,
                memory_labels_by_id=memory_labels_by_id,
                role_cache=role_cache,
            )
            if not _unit_semantic_gate_allows_anchor(
                row=row,
                plan=plan,
                query_targets=query_targets,
                unit=unit,
                tags=effective_roles,
            ):
                continue
            covered_count, weighted_gain = _missing_role_gain(effective_roles, missing)
            if weighted_gain <= 0:
                continue
            projected_total = sum(item.token_count for item in selected_units) + unit.token_count
            overshoot_reason: str | None = None
            if budget_window is not None:
                hard_budget = max(0, int(budget_window.hard_tokens))
                if hard_budget > 0 and projected_total > hard_budget:
                    continue
                if projected_total > max(0, int(budget_window.target_tokens)):
                    selected_memory_ids = {selected.memory_id for selected in selected_units}
                    selected_dates = {
                        tuple(selected.date_key)
                        for selected in selected_units
                        if getattr(selected, "date_key", None)
                    }
                    adds_new_memory = unit.memory_id not in selected_memory_ids
                    adds_new_date = bool(getattr(unit, "date_key", None)) and tuple(unit.date_key) not in selected_dates
                    needs_memory_diversity = len(selected_memory_ids) < int(budget_window.min_distinct_memories or 0)
                    needs_date_diversity = len(selected_dates) < int(budget_window.min_date_contexts or 0)
                    role_spread = len(effective_roles.intersection(missing))
                    if role_spread >= 2:
                        overshoot_reason = "unmet_need"
                    elif needs_memory_diversity and adds_new_memory:
                        overshoot_reason = "memory_diversity"
                    elif needs_date_diversity and adds_new_date:
                        overshoot_reason = "date_diversity"
                    if overshoot_reason is None:
                        continue
            reuse_bonus = 0 if unit.memory_id in used_memory_ids else 1
            role_spread = len(effective_roles.intersection(missing))
            key = (
                weighted_gain,
                covered_count,
                reuse_bonus,
                role_spread,
                score_evidence_unit(
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
                best_index = index
        if best_index is None:
            break
        unit = remaining.pop(best_index)
        selected_units.append(unit)
        if best_key is not None and budget_window is not None:
            projected_total = sum(item.token_count for item in selected_units)
            if projected_total > max(0, int(budget_window.target_tokens)):
                selected_memory_ids = {selected.memory_id for selected in selected_units[:-1]}
                selected_dates = {
                    tuple(selected.date_key)
                    for selected in selected_units[:-1]
                    if getattr(selected, "date_key", None)
                }
                overshoot_reason = None
                if unit.memory_id not in selected_memory_ids and len(selected_memory_ids) < int(budget_window.min_distinct_memories or 0):
                    overshoot_reason = "memory_diversity"
                elif bool(getattr(unit, "date_key", None)) and tuple(unit.date_key) not in selected_dates and len(selected_dates) < int(budget_window.min_date_contexts or 0):
                    overshoot_reason = "date_diversity"
                else:
                    overshoot_reason = "unmet_need"
                expansion_steps.append(f"soft_budget_overshoot={overshoot_reason}:{unit.unit_id}")
        unit_tags = _unit_requirement_tags_cached(
            row=row,
            plan=plan,
            query_targets=query_targets,
            unit=unit,
            memory_labels_by_id=memory_labels_by_id,
            role_cache=role_cache,
        )
        effective_roles = _effective_missing_roles(
            tags=unit_tags,
            missing=_missing_requirements(requirements, coverage),
            unit=unit,
            selected_units=selected_units[:-1],
            row=row,
            plan=plan,
            query_targets=query_targets,
            memory_labels_by_id=memory_labels_by_id,
            role_cache=role_cache,
        )
        for tag in coverage:
            if tag in effective_roles:
                coverage[tag] += 1
        if unit.memory_id not in used_memory_ids:
            used_memory_ids.append(unit.memory_id)

    return selected_units, coverage, expansion_steps
