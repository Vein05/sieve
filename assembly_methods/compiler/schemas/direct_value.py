"""Direct-value schema planning."""

from __future__ import annotations

from ..intent.answer_intent import AnswerIntent
from ..planner.schema import EvidencePlan
from .base import SchemaContext, entity_hint, slot


class DirectValueSchema:
    schema_name = "DirectValue"

    def build_plan(self, intent: AnswerIntent, context: SchemaContext) -> EvidencePlan:
        preferred_attribute = str(intent.preferred_attributes[0]) if intent.preferred_attributes else None
        if context.duration_query and preferred_attribute in {None, "", "occupation", "date", "time"}:
            preferred_attribute = "duration"
        entity_lookup = (
            not preferred_attribute
            and not context.numeric_query
            and intent.interrogative in {"who", "which"}
        )
        slots = [slot("direct_value", "direct_value", entity_key=entity_hint(intent, 0), attribute_key=preferred_attribute, expected_unit="text")]
        schema_name = "EntityLookup" if entity_lookup else "AttributeLookup"
        requires_numeric_reasoning = context.numeric_query
        if context.numeric_query:
            if context.duration_query:
                slots = [
                    slot("direct_value", "direct_value", entity_key=entity_hint(intent, 0), attribute_key=preferred_attribute, expected_unit="text"),
                    slot("numeric_value", "numeric_operand", entity_key=entity_hint(intent, 0), attribute_key=preferred_attribute, expected_unit="number", required=False),
                ]
            else:
                slots = [
                    slot("direct_value", "direct_value", entity_key=entity_hint(intent, 0), attribute_key=preferred_attribute, expected_unit="text", required=False),
                    slot("numeric_value", "numeric_operand", entity_key=entity_hint(intent, 0), attribute_key=preferred_attribute, expected_unit="number"),
                ]
        if intent.requires_support:
            schema_name = f"{schema_name}+Support"
            slots.append(slot("support", "support", entity_key=entity_hint(intent, 0), attribute_key=preferred_attribute, expected_unit="text"))
        plan_kind = "entity_lookup" if entity_lookup else "attribute_lookup"
        return EvidencePlan(
            family=intent.query_family,
            schema_name=schema_name,
            slots=tuple(slots),
            requires_latest_resolution=intent.requires_latest_resolution,
            requires_ordering=False,
            requires_numeric_reasoning=requires_numeric_reasoning,
            plan_metadata={
                "plan_kind": plan_kind,
                "preferred_attribute": preferred_attribute or "",
                "answer_shape": "entity_span" if entity_lookup else ("numeric_span" if context.numeric_query else "attribute_span"),
                "interrogative": str(intent.interrogative or "").strip(),
                "duration_query": bool(context.duration_query),
                "hard_invariants": (
                    ["requires_entity_like_answer"] if entity_lookup else ["requires_attribute_compatible_answer"]
                ) + (["requires_supporting_evidence"] if intent.requires_support else []),
            },
        )
