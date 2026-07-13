"""Typed schema contracts for the compiler planner."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


def slot_requirement_name(slot: "EvidenceSlot") -> str:
    if slot.slot_name == "left_operand":
        return "comparison_left"
    if slot.slot_name == "right_operand":
        return "comparison_right"
    if slot.slot_name == "event_a":
        return "temporal_event_a"
    if slot.slot_name == "event_b":
        return "temporal_event_b"
    if slot.slot_name == "time_a":
        return "temporal_time_a"
    if slot.slot_name == "time_b":
        return "temporal_time_b"
    if slot.slot_name == "event" and slot.slot_type == "temporal_event":
        return "temporal_event_a"
    return {
        "direct_value": "direct_anchor",
        "support": "support_anchor",
        "count_evidence": "count_evidence",
        "count_item": "count_item",
        "comparison_operand": "comparison_operand",
        "ordering_event": "ordering_event",
        "temporal_event": "temporal_event",
        "temporal_time": "temporal_time",
        "reference_time": "reference_time",
        "state_anchor": "state_anchor",
        "current_resolution": "current_resolution",
        "old_state": "old_state",
        "new_state": "new_state",
        "numeric_operand": "numeric_operand",
    }.get(slot.slot_type, slot.slot_name)


def plan_requirements(plan: "EvidencePlan") -> dict[str, int]:
    counts = Counter()
    for slot in plan.slots:
        if slot.required:
            name = slot_requirement_name(slot)
            counts[name] += max(1, int(getattr(slot, "required_count", 1)))
    return dict(counts)


@dataclass(frozen=True)
class EvidenceSlot:
    slot_name: str
    slot_type: str
    entity_key: str | None
    attribute_key: str | None
    expected_unit: str | None
    required: bool
    is_variadic: bool = False
    required_count: int = 1
    group_key: str | None = None


@dataclass(frozen=True)
class EvidencePlan:
    family: str
    schema_name: str
    slots: tuple[EvidenceSlot, ...]
    requires_latest_resolution: bool
    requires_ordering: bool
    requires_numeric_reasoning: bool
    plan_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def query_family(self) -> str:
        return self.family
