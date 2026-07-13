"""Planner that maps AnswerIntent into schema-specific EvidencePlan objects."""

from __future__ import annotations

from typing import Any

from ..intent.answer_intent import AnswerIntent, parse_answer_intent
from ..intent.query_features import is_duration_query, is_numeric_query, normalized_query
from ..schemas.aggregate import AggregateSchema
from ..schemas.comparison import ComparisonSchema
from ..schemas.current_state import CurrentStateSchema
from ..schemas.direct_value import DirectValueSchema
from ..schemas.ordered_choice import OrderedChoiceSchema
from ..schemas.relative_time import RelativeTimeSchema
from ..schemas.state_update import StateUpdateSchema
from ..schemas.temporal_interval import TemporalIntervalSchema
from ..schemas.base import SchemaContext, entity_hint, slot
from .schema import EvidencePlan

_DIRECT_VALUE = DirectValueSchema()
_AGGREGATE = AggregateSchema()
_COMPARISON = ComparisonSchema()
_ORDERED_CHOICE = OrderedChoiceSchema()
_TEMPORAL_INTERVAL = TemporalIntervalSchema()
_RELATIVE_TIME = RelativeTimeSchema()
_CURRENT_STATE = CurrentStateSchema()
_STATE_UPDATE = StateUpdateSchema()


def _build_context(row: dict[str, Any]) -> SchemaContext:
    query = normalized_query(row)
    return SchemaContext(
        query=query,
        numeric_query=is_numeric_query(query),
        duration_query=is_duration_query(query),
    )


def build_evidence_plan(intent: AnswerIntent, row: dict[str, Any]) -> EvidencePlan:
    context = _build_context(row)
    family = intent.query_family

    if family in {"information_extraction", "single_anchor"}:
        return _DIRECT_VALUE.build_plan(intent, context)
    if family == "aggregation":
        normalized = f" {context.query} "
        # Self-correct aggregation family drift for single-answer recall queries.
        # These questions often inherit the aggregate family upstream because of
        # modifiers like "multiple" or "combined", but the answer contract is
        # still a single direct lookup with support.
        if intent.operator == "lookup" and intent.cardinality == "single" and intent.requires_support:
            return _DIRECT_VALUE.build_plan(intent, context)
        if context.duration_query and any(
            marker in normalized
            for marker in (" and ", " combined ", " total ", " in total ", " altogether ")
        ):
            return _AGGREGATE.build_plan(intent, context)
        if context.duration_query and intent.operator == "lookup":
            return _DIRECT_VALUE.build_plan(intent, context)
        if intent.operator == "compare":
            return _COMPARISON.build_plan(intent, context)
        if intent.operator == "order":
            return _ORDERED_CHOICE.build_plan(intent, context)
        return _AGGREGATE.build_plan(intent, context)
    if family == "ordering":
        return _ORDERED_CHOICE.build_plan(intent, context)
    if family == "temporal":
        if intent.operator == "diff":
            return _TEMPORAL_INTERVAL.build_plan(intent, context)
        if intent.operator == "relative_time":
            return _RELATIVE_TIME.build_plan(intent, context)
        if context.duration_query:
            return _DIRECT_VALUE.build_plan(intent, context)
        return EvidencePlan(
            family=family,
            schema_name="DirectValue",
            slots=(slot("event", "temporal_event", entity_key=entity_hint(intent, 0), expected_unit="event"),),
            requires_latest_resolution=False,
            requires_ordering=False,
            requires_numeric_reasoning=False,
            plan_metadata={"plan_kind": "temporal_event_lookup"},
        )
    if family == "current_state":
        return _CURRENT_STATE.build_plan(intent, context)
    if family in {"knowledge_update", "conflict_update"}:
        return _STATE_UPDATE.build_plan(intent, context)
    return _DIRECT_VALUE.build_plan(intent, context)


def plan_evidence_for_row(row: dict[str, Any], query_family: str) -> tuple[AnswerIntent, EvidencePlan]:
    intent = parse_answer_intent(row, query_family)
    return intent, build_evidence_plan(intent, row)
