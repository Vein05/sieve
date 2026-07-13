"""Single-event duration schema planning."""

from __future__ import annotations

from ..intent.answer_intent import AnswerIntent
from ..planner.schema import EvidencePlan
from .base import SchemaContext, entity_hint, slot


class EventDurationSchema:
    schema_name = "EventDuration"

    def build_plan(self, intent: AnswerIntent, context: SchemaContext) -> EvidencePlan:
        event_entity = entity_hint(intent, 0)
        return EvidencePlan(
            family=intent.query_family,
            schema_name=self.schema_name,
            slots=(
                slot("event", "temporal_event", entity_key=event_entity, expected_unit="event"),
                slot("time_a", "temporal_time", entity_key=event_entity, attribute_key="start_time", expected_unit="time", required=False),
                slot("time_b", "temporal_time", entity_key=event_entity, attribute_key="end_time", expected_unit="time", required=False),
                slot("direct_value", "direct_value", entity_key=event_entity, attribute_key="duration", expected_unit="text", required=False),
                slot("numeric_value", "numeric_operand", entity_key=event_entity, attribute_key="duration", expected_unit="number", required=False),
            ),
            requires_latest_resolution=False,
            requires_ordering=False,
            requires_numeric_reasoning=context.numeric_query,
            plan_metadata={
                "plan_kind": "event_duration",
                "preferred_attribute": "duration",
                "answer_shape": "numeric_span" if context.numeric_query else "attribute_span",
                "interrogative": str(intent.interrogative or "").strip(),
                "duration_query": bool(context.duration_query),
                "hard_invariants": ["requires_event_anchor"],
            },
        )
