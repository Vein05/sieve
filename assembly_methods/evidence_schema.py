"""Compatibility wrapper for canonical evidence planning schema definitions."""

from __future__ import annotations

from .compiler.planner.schema import EvidencePlan, EvidenceSlot, plan_requirements, slot_requirement_name

__all__ = [
    "EvidencePlan",
    "EvidenceSlot",
    "plan_requirements",
    "slot_requirement_name",
]
