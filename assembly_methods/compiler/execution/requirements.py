"""Slot coverage, plan satisfaction, and requirement helpers for compiler execution."""

from __future__ import annotations

# max_compiled_tokens lives in adaptive_budget.py; re-exported here for any callers that
# import it from this module.
from .adaptive_budget import max_compiled_tokens as max_compiled_tokens  # noqa: F401

from typing import Any

from ...evidence_schema import EvidencePlan, plan_requirements, slot_requirement_name

_VARIADIC_SELECTION_CAP = 10

_AGGREGATE_SCHEMA_NAMES = {"Aggregate", "CountLookup", "CountDistinctItems", "SumOperands", "PercentageAggregate", "AverageAggregate", "DeltaAggregate", "ExtremumSelection"}
_AGGREGATE_SCHEMA_NAMES.add("CountEvents")
_NUMERIC_AGGREGATE_SCHEMA_NAMES = {"Aggregate", "SumOperands", "PercentageAggregate", "AverageAggregate", "DeltaAggregate", "ExtremumSelection"}
_COUNT_AGGREGATE_SCHEMA_NAMES = {"CountLookup", "CountDistinctItems", "CountEvents"}
_DIRECT_LOOKUP_SCHEMA_NAMES = {
    "DirectValue",
    "DirectValue+Support",
    "AttributeLookup",
    "AttributeLookup+Support",
    "EntityLookup",
    "EntityLookup+Support",
}
_SUPPORT_LOOKUP_SCHEMA_NAMES = {"DirectValue+Support", "AttributeLookup+Support", "EntityLookup+Support"}
_CURRENT_STATE_SCHEMA_NAMES = {"CurrentState", "CurrentStateResolution"}
_STATE_UPDATE_SCHEMA_NAMES = {"StateUpdate", "StateUpdateResolution"}
_COMPARISON_SCHEMA_NAMES = {"Comparison", "DifferenceAggregate"}


def _selection_depth(plan: EvidencePlan, slot: Any) -> int:
    metadata = getattr(plan, "plan_metadata", {}) or {}
    raw_depth = metadata.get("selection_depth")
    if raw_depth is not None:
        try:
            return max(1, int(raw_depth))
        except (TypeError, ValueError):
            pass
    return max(1, int(getattr(slot, "required_count", 1)))


def requirements_dict(plan: EvidencePlan) -> dict[str, int]:
    return plan_requirements(plan)


def coverage_dict(requirements: dict[str, int]) -> dict[str, int]:
    return {name: 0 for name in requirements}


def selection_requirements_dict(plan: EvidencePlan) -> dict[str, int]:
    requirements = requirements_dict(plan)
    for slot in plan.slots:
        if not slot.required or not slot.is_variadic:
            continue
        requirement_name = slot_requirement_name(slot)
        minimum = _selection_depth(plan, slot)
        if minimum <= 1:
            requirements[requirement_name] = max(int(requirements.get(requirement_name, 0)), _VARIADIC_SELECTION_CAP)
        else:
            requirements[requirement_name] = max(int(requirements.get(requirement_name, 0)), minimum)
    return requirements


def aggregate_slot(plan: EvidencePlan) -> Any | None:
    for slot in plan.slots:
        if slot.required and slot.slot_name == "aggregate_items":
            return slot
    return None


def is_aggregate_schema(plan: EvidencePlan) -> bool:
    return str(plan.schema_name or "") in _AGGREGATE_SCHEMA_NAMES


def is_numeric_aggregate_schema(plan: EvidencePlan) -> bool:
    return str(plan.schema_name or "") in _NUMERIC_AGGREGATE_SCHEMA_NAMES


def is_count_aggregate_schema(plan: EvidencePlan) -> bool:
    return str(plan.schema_name or "") in _COUNT_AGGREGATE_SCHEMA_NAMES


def is_direct_lookup_schema(plan: EvidencePlan) -> bool:
    return str(plan.schema_name or "") in _DIRECT_LOOKUP_SCHEMA_NAMES


def is_support_lookup_schema(plan: EvidencePlan) -> bool:
    return str(plan.schema_name or "") in _SUPPORT_LOOKUP_SCHEMA_NAMES


def is_current_state_schema(plan: EvidencePlan) -> bool:
    return str(plan.schema_name or "") in _CURRENT_STATE_SCHEMA_NAMES


def is_state_update_schema(plan: EvidencePlan) -> bool:
    return str(plan.schema_name or "") in _STATE_UPDATE_SCHEMA_NAMES


def is_comparison_schema(plan: EvidencePlan) -> bool:
    return str(plan.schema_name or "") in _COMPARISON_SCHEMA_NAMES


def aggregate_requires_two_operands(plan: EvidencePlan) -> bool:
    slot = aggregate_slot(plan)
    return bool(
        is_numeric_aggregate_schema(plan)
        and slot is not None
        and str(slot.slot_type) == "numeric_operand"
        and int(getattr(slot, "required_count", 1)) >= 2
    )


def required_binding_names(plan: EvidencePlan) -> list[str]:
    if aggregate_requires_two_operands(plan):
        return ["operand_1", "operand_2"]
    return [slot.slot_name for slot in plan.slots if slot.required]


def validated_missing_requirements(
    plan: EvidencePlan,
    validated_slot_bindings: dict[str, dict[str, Any] | None],
) -> dict[str, int]:
    missing: dict[str, int] = {}
    if aggregate_requires_two_operands(plan):
        for slot_name in ("operand_1", "operand_2"):
            if not isinstance(validated_slot_bindings.get(slot_name), dict):
                missing["numeric_operand"] = missing.get("numeric_operand", 0) + 1
        return missing
    for slot in plan.slots:
        if not slot.required:
            continue
        if isinstance(validated_slot_bindings.get(slot.slot_name), dict):
            continue
        requirement_name = slot_requirement_name(slot)
        missing[requirement_name] = missing.get(requirement_name, 0) + 1
    return missing


def validated_rescue_missing(
    plan: EvidencePlan,
    validated_slot_bindings: dict[str, dict[str, Any] | None],
    invalid_slots: dict[str, str],
) -> dict[str, int]:
    missing = validated_missing_requirements(plan, validated_slot_bindings)
    slot_by_name = {slot.slot_name: slot for slot in plan.slots if slot.required}
    for slot_name, reason in invalid_slots.items():
        if not reason or reason == "missing":
            continue
        if aggregate_requires_two_operands(plan) and slot_name in {"operand_1", "operand_2"}:
            missing["numeric_operand"] = max(missing.get("numeric_operand", 0), 1)
            continue
        slot = slot_by_name.get(slot_name)
        if slot is None:
            continue
        requirement_name = slot_requirement_name(slot)
        missing[requirement_name] = max(missing.get(requirement_name, 0), 1)
    return missing


def plan_satisfied(requirements: dict[str, int], coverage: dict[str, int]) -> bool:
    return all(coverage.get(requirement, 0) >= count for requirement, count in requirements.items())


def missing_requirements(requirements: dict[str, int], coverage: dict[str, int]) -> dict[str, int]:
    return {
        requirement: count - coverage.get(requirement, 0)
        for requirement, count in requirements.items()
        if coverage.get(requirement, 0) < count
    }


# max_compiled_tokens is defined in adaptive_budget.py and re-exported above.
