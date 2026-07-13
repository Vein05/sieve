"""Current-state schema planning."""

from __future__ import annotations

import re

from ..intent.answer_intent import AnswerIntent
from ..planner.schema import EvidencePlan
from .base import SchemaContext, entity_hint, slot


class CurrentStateSchema:
    schema_name = "CurrentStateResolution"

    def build_plan(self, intent: AnswerIntent, context: SchemaContext) -> EvidencePlan:
        primary_entity = entity_hint(intent, 0)
        preferred_attribute = str(intent.preferred_attributes[0]) if intent.preferred_attributes else None
        if context.duration_query and preferred_attribute in {None, "", "occupation", "date", "time"}:
            preferred_attribute = "duration"
        if primary_entity and re.match(r"^(?:brand|type|kind|name)\s+of\s+.+", primary_entity):
            fallback = entity_hint(intent, 1)
            if fallback:
                primary_entity = fallback
        slots = [
            slot("entity", "state_anchor", entity_key=primary_entity, attribute_key=None, expected_unit="entity"),
            slot("current_value", "current_resolution", entity_key=primary_entity, attribute_key=preferred_attribute, expected_unit="value"),
            slot("direct_value", "direct_value", entity_key=primary_entity, attribute_key=preferred_attribute, expected_unit="text"),
        ]
        if context.numeric_query:
            slots.append(slot("numeric_value", "numeric_operand", entity_key=primary_entity, attribute_key=preferred_attribute, expected_unit="number"))
        return EvidencePlan(
            family=intent.query_family,
            schema_name=self.schema_name,
            slots=tuple(slots),
            requires_latest_resolution=True,
            requires_ordering=False,
            requires_numeric_reasoning=context.numeric_query,
            plan_metadata={
                "plan_kind": "current_attribute_lookup",
                "preferred_attribute": preferred_attribute or "",
                "answer_shape": "resolved_state",
                "hard_invariants": ["requires_latest_resolution", "requires_state_anchor"],
            },
        )
