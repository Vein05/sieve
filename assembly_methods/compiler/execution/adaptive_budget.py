"""Adaptive token-window policy for proposal retrieval and compiler selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from ...common import CandidateView
from ...evidence_schema import EvidencePlan
from ...query_targets import QueryTargets


@dataclass(frozen=True)
class BudgetWindow:
    target_tokens: int
    hard_tokens: int
    max_selected: int
    min_distinct_memories: int
    min_date_contexts: int
    rationale: tuple[str, ...] = ()


def _row_budget_override(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    overrides = context.get("v2_budget_overrides", {})
    return dict(overrides.get(str(row.get("example_id") or ""), {}))


def _quantile(sorted_values: list[int], numerator: int, denominator: int) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return int(sorted_values[0])
    index = int(math.ceil(((len(sorted_values) - 1) * numerator) / float(denominator)))
    index = max(0, min(len(sorted_values) - 1, index))
    return int(sorted_values[index])


def _view_token_stats(views: list[CandidateView]) -> dict[str, int]:
    tokens = sorted(max(0, int(view.token_count)) for view in views)
    if not tokens:
        return {"total": 0, "mean": 0, "p50": 0, "p75": 0, "max": 0}
    total = sum(tokens)
    return {
        "total": total,
        "mean": int(round(total / float(len(tokens)))),
        "p50": _quantile(tokens, 1, 2),
        "p75": _quantile(tokens, 3, 4),
        "max": int(tokens[-1]),
    }


def _selection_depth(plan: EvidencePlan) -> int:
    metadata = getattr(plan, "plan_metadata", {}) or {}
    raw_depth = metadata.get("selection_depth")
    if raw_depth is not None:
        try:
            return max(1, int(raw_depth))
        except (TypeError, ValueError):
            pass
    return max(
        1,
        max(
            (
                max(1, int(getattr(slot, "required_count", 1)))
                for slot in getattr(plan, "slots", ())
                if getattr(slot, "required", False)
            ),
            default=1,
        ),
    )


def _required_slot_total(plan: EvidencePlan) -> int:
    return max(
        1,
        sum(
            max(1, int(getattr(slot, "required_count", 1)))
            for slot in getattr(plan, "slots", ())
            if getattr(slot, "required", False)
        ),
    )


def _support_required(plan: EvidencePlan, query_targets: QueryTargets) -> bool:
    if bool(getattr(query_targets, "asks_for_recall_support", False)):
        return True
    return any(
        getattr(slot, "required", False)
        and (
            "support" in str(getattr(slot, "slot_name", "") or "").lower()
            or "support" in str(getattr(slot, "slot_type", "") or "").lower()
        )
        for slot in getattr(plan, "slots", ())
    )


def _pair_like_requirement(plan: EvidencePlan, query_targets: QueryTargets) -> bool:
    schema = str(getattr(plan, "schema_name", "") or "")
    family = str(getattr(plan, "family", "") or "")
    if bool(getattr(plan, "requires_ordering", False)):
        return True
    if bool(getattr(query_targets, "asks_for_comparison", False)):
        return True
    if bool(getattr(query_targets, "asks_for_temporal_difference", False)):
        return True
    return family in {"ordering", "temporal"} or schema in {
        "TemporalInterval",
        "RelativeTime",
        "OrderedChoice",
        "Comparison",
        "DifferenceAggregate",
        "SumOperands",
        "PercentageAggregate",
        "AverageAggregate",
        "DeltaAggregate",
        "ExtremumSelection",
    }


def _state_chain_required(plan: EvidencePlan) -> bool:
    family = str(getattr(plan, "family", "") or "")
    schema = str(getattr(plan, "schema_name", "") or "")
    return family in {"current_state", "knowledge_update", "conflict_update"} or schema in {
        "CurrentState",
        "CurrentStateResolution",
        "StateUpdate",
        "StateUpdateResolution",
    }


def _date_context_requirement(plan: EvidencePlan, query_targets: QueryTargets) -> int:
    if (
        bool(getattr(plan, "requires_ordering", False))
        or bool(getattr(query_targets, "asks_for_temporal_difference", False))
        or bool(getattr(query_targets, "asks_for_relative_time", False))
        or str(getattr(plan, "schema_name", "") or "") in {"TemporalInterval", "RelativeTime", "OrderedChoice", "Comparison"}
    ):
        return 2
    return 1


def _distinct_memory_requirement(
    plan: EvidencePlan,
    query_targets: QueryTargets,
    *,
    pool_size: int,
) -> int:
    selection_depth = _selection_depth(plan)
    requirement = 1
    if _pair_like_requirement(plan, query_targets):
        requirement += 1
    if _support_required(plan, query_targets):
        requirement += 1
    if _state_chain_required(plan):
        requirement += 1
    if selection_depth > 2:
        requirement += min(2, selection_depth - 2)
    alternative_entities = tuple(getattr(query_targets, "alternative_entities", ()) or ())
    candidate_entities = tuple(getattr(query_targets, "candidate_entities", ()) or ())
    if len(alternative_entities or candidate_entities) >= 2:
        requirement += 1
    return max(1, min(max(1, pool_size), requirement))


def build_proposal_budget_window(
    *,
    row: dict[str, Any],
    context: dict[str, Any],
    plan: EvidencePlan,
    query_targets: QueryTargets,
    proposal_pool_views: list[CandidateView],
) -> BudgetWindow:
    override = _row_budget_override(row, context)
    if "token_budget" in override or "max_selected" in override:
        hard = max(0, int(override.get("token_budget", 0)))
        max_selected = max(0, int(override.get("max_selected", len(proposal_pool_views))))
        target = hard if hard > 0 else sum(max(0, int(view.token_count)) for view in proposal_pool_views[:max_selected])
        return BudgetWindow(
            target_tokens=max(0, int(target)),
            hard_tokens=max(0, int(hard or target)),
            max_selected=max(1, min(len(proposal_pool_views), max_selected)) if proposal_pool_views else max_selected,
            min_distinct_memories=min(max_selected, max(1, _distinct_memory_requirement(plan, query_targets, pool_size=max_selected or 1))),
            min_date_contexts=min(max_selected or 1, _date_context_requirement(plan, query_targets)),
            rationale=("row_override",),
        )

    pool_size = len(proposal_pool_views)
    if pool_size <= 0:
        return BudgetWindow(0, 0, 0, 0, 0, ("empty_pool",))

    stats = _view_token_stats(proposal_pool_views)
    if pool_size <= 12:
        return BudgetWindow(
            target_tokens=stats["total"],
            hard_tokens=stats["total"],
            max_selected=pool_size,
            min_distinct_memories=_distinct_memory_requirement(plan, query_targets, pool_size=pool_size),
            min_date_contexts=min(pool_size, _date_context_requirement(plan, query_targets)),
            rationale=("full_pool_small_candidate_set",),
        )

    slot_pressure = min(6, _required_slot_total(plan))
    selection_depth = min(6, _selection_depth(plan))
    entity_pressure = min(
        3,
        max(
            len(tuple(getattr(query_targets, "subject_entities", ()) or ())),
            len(tuple(getattr(query_targets, "candidate_entities", ()) or ())),
            len(tuple(getattr(query_targets, "alternative_entities", ()) or ())),
        ),
    )
    attribute_pressure = min(2, len(tuple(getattr(query_targets, "preferred_attributes", ()) or ())))
    temporal_pressure = int(
        bool(getattr(query_targets, "asks_for_temporal_difference", False))
        or bool(getattr(query_targets, "asks_for_relative_time", False))
    )
    support_pressure = int(_support_required(plan, query_targets))
    pair_pressure = int(_pair_like_requirement(plan, query_targets))
    min_distinct_memories = _distinct_memory_requirement(plan, query_targets, pool_size=pool_size)
    min_date_contexts = min(pool_size, _date_context_requirement(plan, query_targets))

    typical_tokens = max(48, stats["p75"] or stats["mean"] or stats["p50"] or 48)
    coverage_bonus = 48 * (slot_pressure + entity_pressure + attribute_pressure + temporal_pressure + support_pressure)
    soft = (typical_tokens * min_distinct_memories) + coverage_bonus + (32 * max(0, selection_depth - min_distinct_memories))
    hard = soft + max(typical_tokens, 96 * (1 + pair_pressure + support_pressure))

    if stats["total"] > 0:
        soft = min(stats["total"], soft)
        hard = min(stats["total"], max(soft, hard))

    max_selected = max(
        min_distinct_memories + 2 + max(0, selection_depth - 2),
        int(math.ceil(float(max(soft, typical_tokens)) / float(max(1, stats["p50"] or typical_tokens)))),
    )
    max_selected = max(1, min(pool_size, max_selected))

    return BudgetWindow(
        target_tokens=max(0, int(soft)),
        hard_tokens=max(0, int(max(soft, hard))),
        max_selected=max_selected,
        min_distinct_memories=min(max_selected, min_distinct_memories),
        min_date_contexts=min(max_selected, min_date_contexts),
        rationale=("adaptive_proposal_window_v1",),
    )


def build_compiler_budget_window(
    *,
    row: dict[str, Any],
    context: dict[str, Any],
    plan: EvidencePlan,
    query_targets: QueryTargets,
    ranked_views: list[CandidateView],
    selected_views: list[CandidateView],
    proposal_budget_hint: int = 0,
) -> BudgetWindow:
    override = _row_budget_override(row, context)
    if "token_budget" in override:
        hard = max(0, int(override.get("token_budget", 0)))
        return BudgetWindow(
            target_tokens=hard,
            hard_tokens=hard,
            max_selected=max(1, len(selected_views) or len(ranked_views) or 1),
            min_distinct_memories=_distinct_memory_requirement(
                plan,
                query_targets,
                pool_size=max(1, len(selected_views) or len(ranked_views)),
            ),
            min_date_contexts=_date_context_requirement(plan, query_targets),
            rationale=("row_override",),
        )

    stats_source = selected_views or ranked_views[:4]
    stats = _view_token_stats(stats_source)
    unseen_views = [
        view
        for view in ranked_views
        if view.memory_id not in {selected.memory_id for selected in selected_views}
    ]
    unseen_stats = _view_token_stats(unseen_views[:4])
    slot_pressure = min(6, _required_slot_total(plan))
    selection_depth = min(6, _selection_depth(plan))
    pair_pressure = int(_pair_like_requirement(plan, query_targets))
    support_pressure = int(_support_required(plan, query_targets))
    temporal_pressure = int(
        bool(getattr(query_targets, "asks_for_temporal_difference", False))
        or bool(getattr(query_targets, "asks_for_relative_time", False))
        or bool(getattr(plan, "requires_ordering", False))
    )
    min_distinct_memories = _distinct_memory_requirement(
        plan,
        query_targets,
        pool_size=max(1, len(selected_views) or len(ranked_views)),
    )
    min_date_contexts = _date_context_requirement(plan, query_targets)

    selected_total = sum(max(0, int(view.token_count)) for view in selected_views)
    typical_selected = max(48, stats["p50"] or stats["mean"] or stats["p75"] or 48)
    expansion_typical = max(48, unseen_stats["p50"] or unseen_stats["mean"] or stats["p75"] or typical_selected)
    memory_tokens = sorted(max(0, int(view.token_count)) for view in selected_views)
    base_memory_count = min(len(memory_tokens), max(1, min_distinct_memories))
    seed_tokens = sum(memory_tokens[:base_memory_count]) if memory_tokens else typical_selected * max(1, min_distinct_memories)
    soft = max(
        seed_tokens + (48 * (slot_pressure + support_pressure + temporal_pressure)),
        int(round(selected_total * 0.7)) if selected_total > 0 else 0,
    )
    reserve_memories = max(1, pair_pressure + support_pressure + max(0, min_distinct_memories - base_memory_count))
    hard = soft + (reserve_memories * expansion_typical) + (32 * max(0, selection_depth - 2))

    if selected_total > 0:
        soft = min(max(selected_total, soft), max(selected_total, soft))
        hard = max(selected_total, hard)

    if proposal_budget_hint > 0:
        hard = max(hard, int(proposal_budget_hint))

    return BudgetWindow(
        target_tokens=max(0, int(soft)),
        hard_tokens=max(max(0, int(soft)), int(hard)),
        max_selected=max(1, len(selected_views) or len(ranked_views) or 1),
        min_distinct_memories=min(max(1, len(selected_views) or len(ranked_views) or 1), min_distinct_memories),
        min_date_contexts=min(max(1, len(selected_views) or len(ranked_views) or 1), min_date_contexts),
        rationale=("adaptive_compile_window_v1",),
    )


def append_with_budget_window(
    selected: list[CandidateView],
    candidate: CandidateView,
    *,
    budget_window: BudgetWindow,
    used_tokens: int,
    covers_unmet_need: bool,
) -> tuple[bool, int, str | None]:
    if any(existing.memory_id == candidate.memory_id for existing in selected):
        return False, used_tokens, None
    if budget_window.max_selected > 0 and len(selected) >= budget_window.max_selected:
        return False, used_tokens, None

    projected = used_tokens + max(0, int(candidate.token_count))
    if budget_window.hard_tokens > 0 and projected > budget_window.hard_tokens and selected:
        return False, used_tokens, None
    if projected <= budget_window.target_tokens or not selected:
        selected.append(candidate)
        return True, projected, None

    selected_memory_ids = {view.memory_id for view in selected}
    selected_dates = {tuple(view.date_key) for view in selected if getattr(view, "date_key", None)}
    adds_new_memory = candidate.memory_id not in selected_memory_ids
    adds_new_date = bool(getattr(candidate, "date_key", None)) and tuple(candidate.date_key) not in selected_dates
    needs_memory_diversity = len(selected_memory_ids) < budget_window.min_distinct_memories
    needs_date_diversity = len(selected_dates) < budget_window.min_date_contexts

    overshoot_reason: str | None = None
    if covers_unmet_need:
        overshoot_reason = "unmet_need"
    elif needs_memory_diversity and adds_new_memory:
        overshoot_reason = "memory_diversity"
    elif needs_date_diversity and adds_new_date:
        overshoot_reason = "date_diversity"

    if projected <= budget_window.hard_tokens and overshoot_reason is not None:
        selected.append(candidate)
        return True, projected, overshoot_reason
    return False, used_tokens, None


def max_compiled_tokens(
    *,
    raw_selected_token_count: int,
    target_budget: int,
    hard_budget: int | None = None,
) -> int:
    """Absolute ceiling on compiled token count after greedy unit selection."""
    effective_hard_budget = max(0, int(hard_budget if hard_budget is not None else target_budget))
    if effective_hard_budget <= 0:
        effective_hard_budget = max(384, int(raw_selected_token_count * 1.5))
    base_floor = 768
    growth_cap = max(raw_selected_token_count, int(raw_selected_token_count * 1.75))
    growth_cap = max(growth_cap, base_floor)
    return min(effective_hard_budget, growth_cap) if effective_hard_budget > 0 else growth_cap
