"""Base schema handler contracts."""

from __future__ import annotations

from dataclasses import dataclass

from ..intent.answer_intent import AnswerIntent
from ..planner.schema import EvidencePlan, EvidenceSlot


def slot(
    name: str,
    slot_type: str,
    *,
    entity_key: str | None = None,
    attribute_key: str | None = None,
    expected_unit: str | None = None,
    required: bool = True,
    is_variadic: bool = False,
    required_count: int = 1,
    group_key: str | None = None,
) -> EvidenceSlot:
    return EvidenceSlot(
        slot_name=name,
        slot_type=slot_type,
        entity_key=entity_key,
        attribute_key=attribute_key,
        expected_unit=expected_unit,
        required=required,
        is_variadic=is_variadic,
        required_count=required_count,
        group_key=group_key,
    )


def entity_hint(intent: AnswerIntent, index: int = 0) -> str | None:
    if len(intent.candidate_entities) <= index:
        return None
    value = str(intent.candidate_entities[index]).strip().lower()
    return value or None


@dataclass(frozen=True)
class SchemaContext:
    query: str
    numeric_query: bool
    duration_query: bool
