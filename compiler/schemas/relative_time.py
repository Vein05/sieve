"""Relative-time schema planning."""

from __future__ import annotations

from ..intent.answer_intent import AnswerIntent
from ..planner.schema import EvidencePlan
from .base import SchemaContext, entity_hint, slot


class RelativeTimeSchema:
    schema_name = "RelativeTime"

    def build_plan(self, intent: AnswerIntent, context: SchemaContext) -> EvidencePlan:
        return EvidencePlan(
            family=intent.query_family,
            schema_name=self.schema_name,
            slots=(
                slot("event", "temporal_event", entity_key=entity_hint(intent, 0), expected_unit="event"),
                slot("reference_time", "reference_time", expected_unit="time"),
            ),
            requires_latest_resolution=True,
            requires_ordering=False,
            requires_numeric_reasoning=context.numeric_query,
        )
