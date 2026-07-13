"""Tests for evidence/memory_objects.py: ProvenanceRecord and MemoryObject."""

from __future__ import annotations

import pytest

from evidence.memory_objects import ProvenanceRecord, MemoryObject


# ---------------------------------------------------------------------------
# ProvenanceRecord
# ---------------------------------------------------------------------------

class TestProvenanceRecordCreation:
    def test_minimal_creation(self):
        rec = ProvenanceRecord(memory_id="m1")
        assert rec.memory_id == "m1"
        assert rec.speaker == "unknown"
        assert rec.date_key == (0, 0, 0)

    def test_full_creation(self):
        rec = ProvenanceRecord(
            memory_id="m1",
            object_id="o1",
            unit_id="u1",
            conversation_id="c1",
            speaker="user",
            date_key=(2025, 6, 15),
            source_span_text="hello world",
            turn_index=3,
        )
        assert rec.object_id == "o1"
        assert rec.speaker == "user"
        assert rec.date_key == (2025, 6, 15)
        assert rec.turn_index == 3

    def test_frozen(self):
        rec = ProvenanceRecord(memory_id="m1")
        with pytest.raises(AttributeError):
            rec.memory_id = "m2"  # type: ignore[misc]


class TestProvenanceRecordPostInit:
    def test_source_text_fallback_from_source_span_text(self):
        rec = ProvenanceRecord(memory_id="m1", source_span_text="span text")
        assert rec.source_text == "span text"

    def test_source_span_text_fallback_from_source_text(self):
        rec = ProvenanceRecord(memory_id="m1", source_text="src text")
        assert rec.source_span_text == "src text"

    def test_render_text_fallback(self):
        rec = ProvenanceRecord(memory_id="m1", source_span_text="span")
        assert rec.render_text == "span"

    def test_explicit_render_text_not_overridden(self):
        rec = ProvenanceRecord(
            memory_id="m1", source_span_text="span", render_text="custom",
        )
        assert rec.render_text == "custom"


class TestProvenanceRecordFromMapping:
    def test_basic_mapping(self):
        data = {"memory_id": "m1", "speaker": "assistant", "date_key": [2025, 1, 1]}
        rec = ProvenanceRecord.from_mapping(data)
        assert rec.memory_id == "m1"
        assert rec.speaker == "assistant"
        assert rec.date_key == (2025, 1, 1)

    def test_date_key_list_to_tuple(self):
        data = {"memory_id": "m1", "date_key": [2024, 12, 31]}
        rec = ProvenanceRecord.from_mapping(data)
        assert isinstance(rec.date_key, tuple)
        assert rec.date_key == (2024, 12, 31)

    def test_extra_keys_go_to_extras(self):
        data = {"memory_id": "m1", "custom_field": "val123"}
        rec = ProvenanceRecord.from_mapping(data)
        assert rec.extras["custom_field"] == "val123"

    def test_missing_memory_id(self):
        rec = ProvenanceRecord.from_mapping({})
        assert rec.memory_id == ""

    def test_none_speaker_becomes_unknown(self):
        rec = ProvenanceRecord.from_mapping({"memory_id": "m1", "speaker": None})
        assert rec.speaker == "unknown"

    def test_empty_mapping(self):
        rec = ProvenanceRecord.from_mapping({})
        assert rec.memory_id == ""
        assert rec.speaker == "unknown"


class TestProvenanceRecordAsDict:
    def test_roundtrip(self):
        rec = ProvenanceRecord(
            memory_id="m1", speaker="user", date_key=(2025, 1, 1),
            source_span_text="hello",
        )
        d = rec.as_dict()
        assert d["memory_id"] == "m1"
        assert d["speaker"] == "user"
        assert d["source_span_text"] == "hello"

    def test_extras_merged(self):
        rec = ProvenanceRecord(
            memory_id="m1", extras={"custom": "value"},
        )
        d = rec.as_dict()
        assert d["custom"] == "value"


class TestProvenanceRecordWithUpdates:
    def test_update_speaker(self):
        rec = ProvenanceRecord(memory_id="m1", speaker="user")
        updated = rec.with_updates(speaker="assistant")
        assert updated.speaker == "assistant"
        assert rec.speaker == "user"  # original unchanged

    def test_update_date_key(self):
        rec = ProvenanceRecord(memory_id="m1")
        updated = rec.with_updates(date_key=[2026, 7, 1])
        assert updated.date_key == (2026, 7, 1)


class TestProvenanceRecordMappingProtocol:
    def test_getitem(self):
        rec = ProvenanceRecord(memory_id="m1", speaker="user")
        assert rec["memory_id"] == "m1"
        assert rec["speaker"] == "user"

    def test_len(self):
        rec = ProvenanceRecord(memory_id="m1")
        length = len(rec)
        assert length > 0

    def test_iter(self):
        rec = ProvenanceRecord(memory_id="m1")
        keys = list(rec)
        assert "memory_id" in keys
        assert "speaker" in keys

    def test_contains(self):
        rec = ProvenanceRecord(memory_id="m1")
        assert "memory_id" in rec


# ---------------------------------------------------------------------------
# MemoryObject
# ---------------------------------------------------------------------------

class TestMemoryObject:
    def _make_provenance(self) -> ProvenanceRecord:
        return ProvenanceRecord(memory_id="m1", source_span_text="test")

    def test_creation(self):
        prov = self._make_provenance()
        mo = MemoryObject(
            object_id="obj1",
            memory_id="m1",
            conversation_id="c1",
            speaker="user",
            date_key=(2025, 6, 15),
            object_type="attribute",
            entity_key="dog",
            attribute_key="breed",
            value_text="golden retriever",
            value_number=None,
            value_unit=None,
            event_key=None,
            update_relation=None,
            canonical_entity_id=None,
            canonical_attribute_id=None,
            canonical_state_id=None,
            canonical_event_id=None,
            update_group_id=None,
            previous_object_id=None,
            source_text="My dog is a golden retriever",
            render_text="dog breed: golden retriever",
            provenance=prov,
        )
        assert mo.object_id == "obj1"
        assert mo.entity_key == "dog"
        assert mo.value_text == "golden retriever"
        assert mo.provenance.memory_id == "m1"

    def test_frozen(self):
        prov = self._make_provenance()
        mo = MemoryObject(
            object_id="obj1", memory_id="m1", conversation_id=None,
            speaker="user", date_key=(0, 0, 0), object_type="event",
            entity_key=None, attribute_key=None, value_text=None,
            value_number=None, value_unit=None, event_key="graduated",
            update_relation=None, canonical_entity_id=None,
            canonical_attribute_id=None, canonical_state_id=None,
            canonical_event_id=None, update_group_id=None,
            previous_object_id=None, source_text="I graduated",
            render_text="graduated", provenance=prov,
        )
        with pytest.raises(AttributeError):
            mo.object_id = "obj2"  # type: ignore[misc]

    def test_equality(self):
        prov = self._make_provenance()
        kwargs = dict(
            object_id="obj1", memory_id="m1", conversation_id=None,
            speaker="user", date_key=(0, 0, 0), object_type="attr",
            entity_key=None, attribute_key=None, value_text="v",
            value_number=None, value_unit=None, event_key=None,
            update_relation=None, canonical_entity_id=None,
            canonical_attribute_id=None, canonical_state_id=None,
            canonical_event_id=None, update_group_id=None,
            previous_object_id=None, source_text="s",
            render_text="r", provenance=prov,
        )
        a = MemoryObject(**kwargs)
        b = MemoryObject(**kwargs)
        assert a == b

    def test_optional_fields_none(self):
        prov = self._make_provenance()
        mo = MemoryObject(
            object_id="obj1", memory_id="m1", conversation_id=None,
            speaker="user", date_key=(0, 0, 0), object_type="attr",
            entity_key=None, attribute_key=None, value_text=None,
            value_number=None, value_unit=None, event_key=None,
            update_relation=None, canonical_entity_id=None,
            canonical_attribute_id=None, canonical_state_id=None,
            canonical_event_id=None, update_group_id=None,
            previous_object_id=None, source_text="s",
            render_text="r", provenance=prov,
        )
        assert mo.entity_key is None
        assert mo.value_number is None
        assert mo.canonical_entity_id is None

    def test_numeric_value(self):
        prov = self._make_provenance()
        mo = MemoryObject(
            object_id="obj1", memory_id="m1", conversation_id=None,
            speaker="user", date_key=(0, 0, 0), object_type="measurement",
            entity_key="internet", attribute_key="speed",
            value_text="100", value_number=100.0, value_unit="mbps",
            event_key=None, update_relation=None,
            canonical_entity_id=None, canonical_attribute_id=None,
            canonical_state_id=None, canonical_event_id=None,
            update_group_id=None, previous_object_id=None,
            source_text="100 mbps", render_text="speed: 100 mbps",
            provenance=prov,
        )
        assert mo.value_number == 100.0
        assert mo.value_unit == "mbps"
