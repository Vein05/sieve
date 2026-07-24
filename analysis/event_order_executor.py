"""Pure helpers for the BEAM event-order execution pilot."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

ABILITY_EVENT_ORDERING = "event_ordering"
ASSISTANT_DELIMITER = "->->"
JSON_FENCE_PREFIX = "```json"
TURN_PATTERN = re.compile(r"turn_(\d+)")
UNKNOWN_TURN = 10**9


class EvidenceCondition(StrEnum):
    """Evidence presentations in the content-matched pilot."""

    GOLD_CHRONO = "gold_chrono"
    EXTRACTED_UNSORTED = "extracted_unsorted"
    EXECUTED_SORTED = "executed_sorted"
    EXECUTED_CORRUPTED = "executed_corrupted"


@dataclass(frozen=True)
class EventRecord:
    """One query-relevant event linked to its source turn."""

    source_id: str
    event: str


def turn_number(memory_id: str) -> int:
    """Return the numeric BEAM turn index."""
    match = TURN_PATTERN.fullmatch(memory_id)
    return int(match.group(1)) if match else UNKNOWN_TURN


def gold_ids(row: dict[str, Any]) -> tuple[str, ...]:
    """Return answer-bearing ids in chronological order."""
    sufficiency = row["evidence_sufficiency"]
    ids = sufficiency["answer_bearing_memory_ids"]
    return tuple(sorted(set(ids), key=turn_number))


def user_segment(turn_text: str) -> str:
    """Keep the user half of a BEAM turn pair."""
    return turn_text.split(ASSISTANT_DELIMITER, maxsplit=1)[0].strip()


def stable_scramble(items: tuple[str, ...], example_id: str) -> tuple[str, ...]:
    """Return a deterministic non-chronological ordering."""
    keyed = sorted(
        items,
        key=lambda item: hashlib.sha256(f"{example_id}:{item}".encode()).hexdigest(),
    )
    ordered = tuple(keyed)
    chronological = tuple(sorted(items, key=turn_number))
    if len(ordered) > 1 and ordered == chronological:
        return ordered[1:] + ordered[:1]
    return ordered


def render_gold_evidence(row: dict[str, Any], store: dict[str, str]) -> str:
    """Render gold-complete user turns chronologically."""
    parts = ["Gold-complete conversation turns in chronological order:"]
    for memory_id in gold_ids(row):
        parts.append(f"[{memory_id}] {user_segment(store[memory_id])}")
    return "\n\n".join(parts)


def render_extraction_input(row: dict[str, Any], store: dict[str, str]) -> str:
    """Render the same turns in a stable scrambled order for extraction."""
    parts = ["Relevant conversation turns in deliberately shuffled order:"]
    for memory_id in stable_scramble(gold_ids(row), row["example_id"]):
        parts.append(f"[{memory_id}] {user_segment(store[memory_id])}")
    return "\n\n".join(parts)


def extraction_prompt(row: dict[str, Any], store: dict[str, str]) -> str:
    """Build a reference-free event extraction prompt."""
    return f"""Extract one concise event or aspect from EACH provided source turn that is relevant to the question.
Do not answer the question and do not reorder the events. Preserve each source_id exactly.
Return JSON only using this schema:
{{"events":[{{"source_id":"turn_0001","event":"concise event description"}}]}}

Question: {row['query']}

{render_extraction_input(row, store)}
"""


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith(JSON_FENCE_PREFIX):
        cleaned = cleaned[len(JSON_FENCE_PREFIX) :]
        cleaned = cleaned.rsplit("```", maxsplit=1)[0]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("compiler output has no JSON object")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("compiler output must be a JSON object")
    return payload


def parse_event_records(text: str, expected_ids: tuple[str, ...]) -> tuple[EventRecord, ...]:
    """Parse and validate a complete provenance-linked event set."""
    payload = _json_object(text)
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        raise ValueError("compiler output must contain an events list")
    records = tuple(_parse_record(item) for item in raw_events)
    source_ids = tuple(record.source_id for record in records)
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("compiler output contains duplicate source ids")
    if set(source_ids) != set(expected_ids):
        raise ValueError("compiler output does not cover the expected source ids")
    return records


def _parse_record(item: Any) -> EventRecord:
    if not isinstance(item, dict):
        raise ValueError("each event must be an object")
    source_id = str(item.get("source_id", "")).strip()
    event = str(item.get("event", "")).strip()
    if not source_id or not event:
        raise ValueError("each event needs source_id and event")
    return EventRecord(source_id=source_id, event=event)


def order_records(records: tuple[EventRecord, ...]) -> tuple[EventRecord, ...]:
    """Execute ORDER over source-turn provenance."""
    return tuple(sorted(records, key=lambda record: turn_number(record.source_id)))


def corrupt_order(records: tuple[EventRecord, ...]) -> tuple[EventRecord, ...]:
    """Rotate a correct trace to create a content-matched causal control."""
    ordered = order_records(records)
    return ordered[1:] + ordered[:1] if len(ordered) > 1 else ordered


def render_records(records: tuple[EventRecord, ...], condition: EvidenceCondition) -> str:
    """Render extracted records or an executed trace."""
    if condition == EvidenceCondition.EXTRACTED_UNSORTED:
        shown = records
        heading = "Extracted records; source turns show time, but records are not ordered:"
    elif condition == EvidenceCondition.EXECUTED_SORTED:
        shown = order_records(records)
        heading = "Executed ORDER result, earliest to latest:"
    elif condition == EvidenceCondition.EXECUTED_CORRUPTED:
        shown = corrupt_order(records)
        heading = "Candidate ordered result:"
    else:
        raise ValueError(f"records cannot render condition: {condition}")
    lines = [heading]
    for index, record in enumerate(shown, start=1):
        lines.append(f"{index}. [{record.source_id}] {record.event}")
    return "\n".join(lines)
