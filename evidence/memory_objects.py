"""Structured conversational memory object schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProvenanceRecord(Mapping[str, Any]):
    """Explicit provenance contract shared by memory objects and evidence units."""

    memory_id: str
    object_id: str | None = None
    unit_id: str | None = None
    conversation_id: str | None = None
    speaker: str = "unknown"
    date_key: tuple[int, int, int] = (0, 0, 0)
    source_span_text: str = ""
    render_text: str = ""
    source_text: str = ""
    turn_index: int | None = None
    sentence_index: int | None = None
    local_order: int | None = None
    parent_rank: int | None = None
    is_sentence_split: bool | None = None
    object_type: str | None = None
    memory_text: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_text:
            object.__setattr__(self, "source_text", self.source_span_text)
        if not self.source_span_text:
            object.__setattr__(self, "source_span_text", self.source_text)
        if not self.render_text:
            object.__setattr__(self, "render_text", self.source_span_text or self.source_text)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | dict[str, Any]) -> "ProvenanceRecord":
        data = dict(mapping or {})
        source_span_text = str(data.pop("source_span_text", data.pop("source_text", "")) or "")
        extras_keys = {
            "memory_id",
            "object_id",
            "unit_id",
            "conversation_id",
            "speaker",
            "date_key",
            "source_span_text",
            "source_text",
            "render_text",
            "turn_index",
            "sentence_index",
            "local_order",
            "parent_rank",
            "is_sentence_split",
            "object_type",
            "memory_text",
            "extras",
        }
        extras = dict(data.pop("extras", {}) or {})
        for key in list(data.keys()):
            if key not in extras_keys:
                extras[key] = data.pop(key)
        date_key = data.pop("date_key", (0, 0, 0))
        if isinstance(date_key, list):
            date_key = tuple(int(value) for value in date_key[:3])
        if not isinstance(date_key, tuple):
            date_key = tuple(date_key) if date_key else (0, 0, 0)
        return cls(
            memory_id=str(data.pop("memory_id", "")),
            object_id=data.pop("object_id", None),
            unit_id=data.pop("unit_id", None),
            conversation_id=data.pop("conversation_id", None),
            speaker=str(data.pop("speaker", "unknown") or "unknown"),
            date_key=tuple(int(value) for value in date_key[:3]) if date_key else (0, 0, 0),
            source_span_text=source_span_text,
            render_text=str(data.pop("render_text", "") or ""),
            source_text=str(data.pop("source_text", "") or ""),
            turn_index=data.pop("turn_index", None),
            sentence_index=data.pop("sentence_index", None),
            local_order=data.pop("local_order", None),
            parent_rank=data.pop("parent_rank", None),
            is_sentence_split=data.pop("is_sentence_split", None),
            object_type=data.pop("object_type", None),
            memory_text=data.pop("memory_text", None),
            extras=extras,
        )

    def as_dict(self) -> dict[str, Any]:
        data = {
            "memory_id": self.memory_id,
            "object_id": self.object_id,
            "unit_id": self.unit_id,
            "conversation_id": self.conversation_id,
            "speaker": self.speaker,
            "date_key": self.date_key,
            "source_span_text": self.source_span_text,
            "source_text": self.source_span_text,
            "render_text": self.render_text,
            "turn_index": self.turn_index,
            "sentence_index": self.sentence_index,
            "local_order": self.local_order,
            "parent_rank": self.parent_rank,
            "is_sentence_split": self.is_sentence_split,
            "object_type": self.object_type,
            "memory_text": self.memory_text,
        }
        data.update(self.extras)
        return data

    def with_updates(self, **updates: Any) -> "ProvenanceRecord":
        data = self.as_dict()
        data.update(updates)
        return ProvenanceRecord.from_mapping(data)

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def __iter__(self):
        return iter(self.as_dict())

    def __len__(self) -> int:
        return len(self.as_dict())


@dataclass(frozen=True)
class MemoryObject:
    object_id: str
    memory_id: str
    conversation_id: str | None
    speaker: str
    date_key: tuple[int, int, int]
    object_type: str
    entity_key: str | None
    attribute_key: str | None
    value_text: str | None
    value_number: float | None
    value_unit: str | None
    event_key: str | None
    update_relation: str | None
    canonical_entity_id: str | None
    canonical_attribute_id: str | None
    canonical_state_id: str | None
    canonical_event_id: str | None
    update_group_id: str | None
    previous_object_id: str | None
    source_text: str
    render_text: str
    provenance: ProvenanceRecord
