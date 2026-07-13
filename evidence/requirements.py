"""Compatibility wrapper from canonical planner schema to compiler requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from compiler.evidence_planner import plan_evidence
from .schema import EvidenceSlot, plan_requirements, slot_requirement_name


@dataclass(frozen=True)
class EvidenceRequirement:
    name: str
    min_count: int
    description: str


@dataclass(frozen=True)
class QueryEvidencePlan:
    query_family: str
    requirements: tuple[EvidenceRequirement, ...]
    requires_latest_resolution: bool
    requires_temporal_pair: bool
    requires_numeric_completion: bool
    requires_entity_disambiguation: bool


def _requirement_name(slot_name: str, slot_type: str) -> str:
    return slot_requirement_name(
        EvidenceSlot(
            slot_name=slot_name,
            slot_type=slot_type,
            entity_key=None,
            attribute_key=None,
            expected_unit=None,
            required=True,
        )
    )


def build_query_evidence_plan(row: dict[str, Any], query_family: str) -> QueryEvidencePlan:
    plan = plan_evidence(row, query_family)

    counts = plan_requirements(plan)
    descriptions: dict[str, str] = {}
    for slot in plan.slots:
        if not slot.required:
            continue
        req_name = _requirement_name(slot.slot_name, slot.slot_type)
        descriptions[req_name] = f"Required slot: {slot.slot_name} ({slot.slot_type})."

    requirements = tuple(
        EvidenceRequirement(name=name, min_count=count, description=descriptions.get(name, ""))
        for name, count in counts.items()
    )

    requires_temporal_pair = any(
        name in counts
        for name in (
            "temporal_event_a",
            "temporal_event_b",
            "temporal_time_a",
            "temporal_time_b",
            "ordering_event",
        )
    )
    requires_entity_disambiguation = plan.family == "current_state" or any(
        name in counts
        for name in (
            "comparison_left",
            "comparison_right",
            "ordering_event",
            "old_state",
            "new_state",
        )
    )

    return QueryEvidencePlan(
        query_family=plan.family,
        requirements=requirements,
        requires_latest_resolution=plan.requires_latest_resolution,
        requires_temporal_pair=requires_temporal_pair,
        requires_numeric_completion=plan.requires_numeric_reasoning,
        requires_entity_disambiguation=requires_entity_disambiguation,
    )
