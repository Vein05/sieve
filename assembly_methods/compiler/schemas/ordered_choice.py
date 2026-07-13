"""Ordered-choice schema planning."""

from __future__ import annotations

from ..intent.answer_intent import AnswerIntent
from ..planner.schema import EvidencePlan
from .base import SchemaContext, entity_hint, slot


class OrderedChoiceSchema:
    schema_name = "OrderedChoice"

    def build_plan(self, intent: AnswerIntent, context: SchemaContext) -> EvidencePlan:
        return EvidencePlan(
            family=intent.query_family,
            schema_name=self.schema_name,
            slots=(
                slot("choice_a", "ordering_event", entity_key=entity_hint(intent, 0), expected_unit="event"),
                slot("choice_b", "ordering_event", entity_key=entity_hint(intent, 1), expected_unit="event"),
            ),
            requires_latest_resolution=False,
            requires_ordering=True,
            requires_numeric_reasoning=False,
        )
