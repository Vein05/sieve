"""State-update schema planning."""

from __future__ import annotations

from ..intent.answer_intent import AnswerIntent
from ..planner.schema import EvidencePlan
from .base import SchemaContext, entity_hint, slot


class StateUpdateSchema:
    schema_name = "StateUpdateResolution"

    def build_plan(self, intent: AnswerIntent, context: SchemaContext) -> EvidencePlan:
        return EvidencePlan(
            family=intent.query_family,
            schema_name=self.schema_name,
            slots=(
                slot("old_value", "old_state", entity_key=entity_hint(intent, 0), expected_unit="value"),
                slot("new_value", "new_state", entity_key=entity_hint(intent, 0), expected_unit="value"),
                slot("direct_value", "direct_value", entity_key=entity_hint(intent, 0), expected_unit="text"),
            ),
            requires_latest_resolution=True,
            requires_ordering=False,
            requires_numeric_reasoning=context.numeric_query,
            plan_metadata={
                "plan_kind": "state_update_resolution",
                "answer_shape": "state_transition",
                "hard_invariants": ["requires_old_and_new_state"],
            },
        )
