"""Comparison schema planning."""

from __future__ import annotations

from ..intent.answer_intent import AnswerIntent
from ..planner.schema import EvidencePlan
from .base import SchemaContext, entity_hint, slot


class ComparisonSchema:
    schema_name = "Comparison"

    def build_plan(self, intent: AnswerIntent, context: SchemaContext) -> EvidencePlan:
        normalized = f" {context.query} "
        difference_query = context.numeric_query and any(
            marker in normalized
            for marker in (" difference ", " how much more ", " how much less ", " by how much ")
        )
        return EvidencePlan(
            family=intent.query_family,
            schema_name="DifferenceAggregate" if difference_query else self.schema_name,
            slots=(
                slot("left_operand", "comparison_operand", entity_key=entity_hint(intent, 0), expected_unit="text_or_number"),
                slot("right_operand", "comparison_operand", entity_key=entity_hint(intent, 1), expected_unit="text_or_number"),
            ),
            requires_latest_resolution=False,
            requires_ordering=False,
            requires_numeric_reasoning=context.numeric_query,
            plan_metadata={
                "plan_kind": "difference" if difference_query else "comparison",
                "answer_shape": "numeric_difference" if difference_query else "contrastive_choice",
                "hard_invariants": ["requires_paired_operands"],
            },
        )
