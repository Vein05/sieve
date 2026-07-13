"""Tests for evidence/schema.py (re-exports from compiler/planner/schema.py)."""

from __future__ import annotations

import pytest

from compiler.planner.schema import (
    EvidencePlan,
    EvidenceSlot,
    plan_requirements,
    slot_requirement_name,
)


# ---------------------------------------------------------------------------
# EvidenceSlot
# ---------------------------------------------------------------------------

class TestEvidenceSlot:
    def test_creation(self):
        slot = EvidenceSlot(
            slot_name="direct_value",
            slot_type="direct_value",
            entity_key="dog",
            attribute_key="breed",
            expected_unit=None,
            required=True,
        )
        assert slot.slot_name == "direct_value"
        assert slot.entity_key == "dog"
        assert slot.required is True

    def test_frozen(self):
        slot = EvidenceSlot(
            slot_name="x", slot_type="y", entity_key=None,
            attribute_key=None, expected_unit=None, required=False,
        )
        with pytest.raises(AttributeError):
            slot.slot_name = "z"  # type: ignore[misc]

    def test_defaults(self):
        slot = EvidenceSlot(
            slot_name="a", slot_type="b", entity_key=None,
            attribute_key=None, expected_unit=None, required=True,
        )
        assert slot.is_variadic is False
        assert slot.required_count == 1
        assert slot.group_key is None

    def test_variadic_slot(self):
        slot = EvidenceSlot(
            slot_name="items", slot_type="count_item", entity_key=None,
            attribute_key=None, expected_unit=None, required=True,
            is_variadic=True, required_count=3,
        )
        assert slot.is_variadic is True
        assert slot.required_count == 3


# ---------------------------------------------------------------------------
# slot_requirement_name
# ---------------------------------------------------------------------------

class TestSlotRequirementName:
    def test_left_operand(self):
        slot = EvidenceSlot(
            slot_name="left_operand", slot_type="comparison_operand",
            entity_key=None, attribute_key=None, expected_unit=None, required=True,
        )
        assert slot_requirement_name(slot) == "comparison_left"

    def test_right_operand(self):
        slot = EvidenceSlot(
            slot_name="right_operand", slot_type="comparison_operand",
            entity_key=None, attribute_key=None, expected_unit=None, required=True,
        )
        assert slot_requirement_name(slot) == "comparison_right"

    def test_event_a(self):
        slot = EvidenceSlot(
            slot_name="event_a", slot_type="temporal_event",
            entity_key=None, attribute_key=None, expected_unit=None, required=True,
        )
        assert slot_requirement_name(slot) == "temporal_event_a"

    def test_event_b(self):
        slot = EvidenceSlot(
            slot_name="event_b", slot_type="temporal_event",
            entity_key=None, attribute_key=None, expected_unit=None, required=True,
        )
        assert slot_requirement_name(slot) == "temporal_event_b"

    def test_time_a(self):
        slot = EvidenceSlot(
            slot_name="time_a", slot_type="temporal_time",
            entity_key=None, attribute_key=None, expected_unit=None, required=True,
        )
        assert slot_requirement_name(slot) == "temporal_time_a"

    def test_time_b(self):
        slot = EvidenceSlot(
            slot_name="time_b", slot_type="temporal_time",
            entity_key=None, attribute_key=None, expected_unit=None, required=True,
        )
        assert slot_requirement_name(slot) == "temporal_time_b"

    def test_event_with_temporal_event_type(self):
        slot = EvidenceSlot(
            slot_name="event", slot_type="temporal_event",
            entity_key=None, attribute_key=None, expected_unit=None, required=True,
        )
        assert slot_requirement_name(slot) == "temporal_event_a"

    def test_direct_value_type(self):
        slot = EvidenceSlot(
            slot_name="value", slot_type="direct_value",
            entity_key=None, attribute_key=None, expected_unit=None, required=True,
        )
        assert slot_requirement_name(slot) == "direct_anchor"

    def test_count_item_type(self):
        slot = EvidenceSlot(
            slot_name="item", slot_type="count_item",
            entity_key=None, attribute_key=None, expected_unit=None, required=True,
        )
        assert slot_requirement_name(slot) == "count_item"

    def test_fallback_to_slot_name(self):
        slot = EvidenceSlot(
            slot_name="custom_slot", slot_type="unknown_type",
            entity_key=None, attribute_key=None, expected_unit=None, required=True,
        )
        assert slot_requirement_name(slot) == "custom_slot"


# ---------------------------------------------------------------------------
# EvidencePlan
# ---------------------------------------------------------------------------

class TestEvidencePlan:
    def _make_slot(self, name: str = "s", stype: str = "direct_value", required: bool = True) -> EvidenceSlot:
        return EvidenceSlot(
            slot_name=name, slot_type=stype, entity_key=None,
            attribute_key=None, expected_unit=None, required=required,
        )

    def test_creation(self):
        plan = EvidencePlan(
            family="single_anchor",
            schema_name="kv_lookup",
            slots=(self._make_slot(),),
            requires_latest_resolution=False,
            requires_ordering=False,
            requires_numeric_reasoning=False,
        )
        assert plan.family == "single_anchor"
        assert plan.schema_name == "kv_lookup"
        assert len(plan.slots) == 1

    def test_query_family_property(self):
        plan = EvidencePlan(
            family="temporal",
            schema_name="event_order",
            slots=(),
            requires_latest_resolution=False,
            requires_ordering=True,
            requires_numeric_reasoning=False,
        )
        assert plan.query_family == "temporal"

    def test_frozen(self):
        plan = EvidencePlan(
            family="x", schema_name="y", slots=(),
            requires_latest_resolution=False,
            requires_ordering=False,
            requires_numeric_reasoning=False,
        )
        with pytest.raises(AttributeError):
            plan.family = "z"  # type: ignore[misc]

    def test_default_metadata(self):
        plan = EvidencePlan(
            family="x", schema_name="y", slots=(),
            requires_latest_resolution=False,
            requires_ordering=False,
            requires_numeric_reasoning=False,
        )
        assert plan.plan_metadata == {}

    def test_with_metadata(self):
        plan = EvidencePlan(
            family="x", schema_name="y", slots=(),
            requires_latest_resolution=False,
            requires_ordering=False,
            requires_numeric_reasoning=False,
            plan_metadata={"key": "value"},
        )
        assert plan.plan_metadata["key"] == "value"


# ---------------------------------------------------------------------------
# plan_requirements
# ---------------------------------------------------------------------------

class TestPlanRequirements:
    def _make_slot(
        self, name: str, stype: str, required: bool = True, required_count: int = 1,
    ) -> EvidenceSlot:
        return EvidenceSlot(
            slot_name=name, slot_type=stype, entity_key=None,
            attribute_key=None, expected_unit=None, required=required,
            required_count=required_count,
        )

    def test_single_required_slot(self):
        plan = EvidencePlan(
            family="single_anchor",
            schema_name="kv",
            slots=(self._make_slot("v", "direct_value"),),
            requires_latest_resolution=False,
            requires_ordering=False,
            requires_numeric_reasoning=False,
        )
        reqs = plan_requirements(plan)
        assert reqs == {"direct_anchor": 1}

    def test_optional_slots_excluded(self):
        plan = EvidencePlan(
            family="x",
            schema_name="y",
            slots=(
                self._make_slot("v", "direct_value", required=True),
                self._make_slot("s", "support", required=False),
            ),
            requires_latest_resolution=False,
            requires_ordering=False,
            requires_numeric_reasoning=False,
        )
        reqs = plan_requirements(plan)
        assert "support_anchor" not in reqs
        assert "direct_anchor" in reqs

    def test_variadic_count(self):
        plan = EvidencePlan(
            family="counting",
            schema_name="count_items",
            slots=(self._make_slot("items", "count_item", required_count=5),),
            requires_latest_resolution=False,
            requires_ordering=False,
            requires_numeric_reasoning=False,
        )
        reqs = plan_requirements(plan)
        assert reqs["count_item"] == 5

    def test_empty_plan(self):
        plan = EvidencePlan(
            family="x", schema_name="y", slots=(),
            requires_latest_resolution=False,
            requires_ordering=False,
            requires_numeric_reasoning=False,
        )
        assert plan_requirements(plan) == {}

    def test_multiple_same_type_accumulate(self):
        plan = EvidencePlan(
            family="comparison",
            schema_name="compare",
            slots=(
                self._make_slot("left_operand", "comparison_operand"),
                self._make_slot("right_operand", "comparison_operand"),
            ),
            requires_latest_resolution=False,
            requires_ordering=False,
            requires_numeric_reasoning=False,
        )
        reqs = plan_requirements(plan)
        assert reqs["comparison_left"] == 1
        assert reqs["comparison_right"] == 1
