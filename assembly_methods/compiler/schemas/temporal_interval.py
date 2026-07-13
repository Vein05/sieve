"""Temporal-interval schema planning."""

from __future__ import annotations

from ..intent.answer_intent import AnswerIntent
from ..planner.schema import EvidencePlan
from .base import SchemaContext, entity_hint, slot


class TemporalIntervalSchema:
    schema_name = "TemporalInterval"

    def build_plan(self, intent: AnswerIntent, context: SchemaContext) -> EvidencePlan:
        return EvidencePlan(
            family=intent.query_family,
            schema_name=self.schema_name,
            slots=(
                slot("event_a", "temporal_event", entity_key=entity_hint(intent, 0), expected_unit="event"),
                slot("event_b", "temporal_event", entity_key=entity_hint(intent, 1), expected_unit="event"),
                slot("time_a", "temporal_time", entity_key=entity_hint(intent, 0), expected_unit="time"),
                slot("time_b", "temporal_time", entity_key=entity_hint(intent, 1), expected_unit="time"),
            ),
            requires_latest_resolution=False,
            requires_ordering=False,
            requires_numeric_reasoning=context.numeric_query,
        )
