"""Aggregate schema planning."""

from __future__ import annotations
import re

from ..intent.answer_intent import AnswerIntent
from ..planner.schema import EvidencePlan, EvidenceSlot
from .base import SchemaContext, entity_hint, slot


class AggregateBase:
    """Base class for aggregation schemas."""

    def _build_plan(
        self,
        intent: AnswerIntent,
        schema_name: str,
        slots: list[EvidenceSlot],
        plan_kind: str,
        *,
        selection_depth: int | None = None,
        answer_shape: str = "numeric",
        hard_invariants: list[str] | None = None,
        extra_metadata: dict[str, object] | None = None,
    ) -> EvidencePlan:
        plan_metadata = {
            "plan_kind": plan_kind,
            "preferred_attribute": str(intent.preferred_attributes[0]) if intent.preferred_attributes else "",
            "answer_shape": answer_shape,
            "hard_invariants": list(hard_invariants or []),
        }
        if selection_depth is not None:
            plan_metadata["selection_depth"] = int(selection_depth)
        if extra_metadata:
            plan_metadata.update(extra_metadata)
        return EvidencePlan(
            family=intent.query_family,
            schema_name=schema_name,
            slots=tuple(slots),
            requires_latest_resolution=False,
            requires_ordering=False,
            requires_numeric_reasoning=True,
            plan_metadata=plan_metadata,
        )


_EVENT_COUNT_RE = re.compile(
    r"\b(?:attend(?:ed)?|visit(?:ed)?|go(?:ne|ing|went)?|watch(?:ed)?|read|finish(?:ed)?|complete(?:d)?|join(?:ed)?|host(?:ed)?|buy|bought|purchase(?:d)?)\b",
    re.IGNORECASE,
)


class CountLookupSchema(AggregateBase):
    """Schema for counting mentions of a specific item/event."""

    def build_plan(self, intent: AnswerIntent, context: SchemaContext) -> EvidencePlan:
        slots = [
            slot(
                "aggregate_items",
                "count_evidence",
                entity_key=entity_hint(intent, 0),
                expected_unit="event_or_item",
                is_variadic=True,
            )
        ]
        return self._build_plan(intent, "CountLookup", slots, "count_items", selection_depth=8)


class CountDistinctItemsSchema(AggregateBase):
    """Schema for counting unique items/categories."""

    def build_plan(self, intent: AnswerIntent, context: SchemaContext) -> EvidencePlan:
        slots = [
            slot(
                "aggregate_items",
                "count_item",
                entity_key=entity_hint(intent, 0),
                expected_unit="event_or_item",
                is_variadic=True,
            )
        ]
        return self._build_plan(intent, "CountDistinctItems", slots, "count_distinct_items", selection_depth=8)


class CountEventsSchema(AggregateBase):
    """Schema for counting repeated event instances."""

    def build_plan(self, intent: AnswerIntent, context: SchemaContext) -> EvidencePlan:
        slots = [
            slot(
                "aggregate_items",
                "count_item",
                entity_key=entity_hint(intent, 0),
                expected_unit="event",
                is_variadic=True,
            )
        ]
        return self._build_plan(intent, "CountEvents", slots, "count_events", selection_depth=8)


class SumOperandsSchema(AggregateBase):
    """Schema for summing multiple numeric operands."""

    def build_plan(self, intent: AnswerIntent, context: SchemaContext) -> EvidencePlan:
        slots = [
            slot(
                "aggregate_items",
                "numeric_operand",
                entity_key=entity_hint(intent, 0),
                expected_unit="number",
                is_variadic=True,
                required_count=2,
                group_key="aggregate_operand",
            )
        ]
        return self._build_plan(intent, "SumOperands", slots, "sum_operands", selection_depth=8)


class AverageAggregateSchema(AggregateBase):
    """Schema for averaging multiple numeric operands."""

    def build_plan(self, intent: AnswerIntent, context: SchemaContext) -> EvidencePlan:
        slots = [
            slot(
                "aggregate_items",
                "numeric_operand",
                entity_key=entity_hint(intent, 0),
                expected_unit="number",
                is_variadic=True,
                required_count=2,
                group_key="aggregate_operand",
            )
        ]
        return self._build_plan(
            intent,
            "AverageAggregate",
            slots,
            "average",
            selection_depth=8,
            answer_shape="numeric_average",
        )


class DeltaAggregateSchema(AggregateBase):
    """Schema for delta / increase / decrease queries over numeric operands."""

    def build_plan(self, intent: AnswerIntent, context: SchemaContext) -> EvidencePlan:
        slots = [
            slot(
                "aggregate_items",
                "numeric_operand",
                entity_key=entity_hint(intent, 0),
                expected_unit="number",
                is_variadic=True,
                required_count=2,
                group_key="aggregate_operand",
            )
        ]
        return self._build_plan(
            intent,
            "DeltaAggregate",
            slots,
            "delta",
            selection_depth=8,
            answer_shape="numeric_difference",
        )


class ExtremumSelectionSchema(AggregateBase):
    """Schema for selecting the entity associated with the highest/lowest numeric value."""

    def build_plan(self, intent: AnswerIntent, context: SchemaContext) -> EvidencePlan:
        normalized = f" {context.query} "
        direction = "min" if any(marker in normalized for marker in (" least ", " lowest ", " smallest ", " shortest ")) else "max"
        slots = [
            slot(
                "aggregate_items",
                "numeric_operand",
                entity_key=entity_hint(intent, 0),
                expected_unit="number",
                is_variadic=True,
                required_count=2,
                group_key="aggregate_operand",
            )
        ]
        return self._build_plan(
            intent,
            "ExtremumSelection",
            slots,
            "extremum_selection",
            selection_depth=10,
            answer_shape="entity_span",
            extra_metadata={"extremum_direction": direction},
        )


class PercentageAggregateSchema(AggregateBase):
    """Schema for calculating percentages/ratios."""

    def build_plan(self, intent: AnswerIntent, context: SchemaContext) -> EvidencePlan:
        slots = [
            slot(
                "aggregate_items",
                "numeric_operand",
                entity_key=entity_hint(intent, 0),
                expected_unit="number",
                is_variadic=True,
                required_count=2,
                group_key="aggregate_operand",
            )
        ]
        return self._build_plan(intent, "PercentageAggregate", slots, "percentage", selection_depth=4)


class AggregateSchema:
    """Legacy factory class for aggregation schemas."""
    
    def __init__(self):
        self._count_lookup = CountLookupSchema()
        self._count_distinct = CountDistinctItemsSchema()
        self._count_events = CountEventsSchema()
        self._sum_operands = SumOperandsSchema()
        self._average = AverageAggregateSchema()
        self._delta = DeltaAggregateSchema()
        self._extremum = ExtremumSelectionSchema()
        self._percentage = PercentageAggregateSchema()

    def build_plan(self, intent: AnswerIntent, context: SchemaContext) -> EvidencePlan:
        normalized = f" {context.query} "
        if intent.operator == "average":
            return self._average.build_plan(intent, context)
        if intent.operator == "delta":
            return self._delta.build_plan(intent, context)
        if intent.operator == "select_extreme":
            return self._extremum.build_plan(intent, context)
        if intent.operator == "sum":
            return self._sum_operands.build_plan(intent, context)
        if any(marker in normalized for marker in (" percentage ", " percent ", " ratio ")):
            return self._percentage.build_plan(intent, context)

        if any(marker in normalized for marker in (" total ", " in total ", " altogether ", " combined ", " sum of ")):
            return self._sum_operands.build_plan(intent, context)
        if context.duration_query and " and " in normalized:
            return self._sum_operands.build_plan(intent, context)
        
        if " different " in normalized or " types of " in normalized or " varieties of " in normalized:
            return self._count_distinct.build_plan(intent, context)
        if _EVENT_COUNT_RE.search(context.query):
            return self._count_events.build_plan(intent, context)
        return self._count_lookup.build_plan(intent, context)
