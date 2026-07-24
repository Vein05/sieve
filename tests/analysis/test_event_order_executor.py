"""Tests for the event-order execution pilot."""

from __future__ import annotations

import pytest

from analysis.event_order_executor import (
    EvidenceCondition,
    EventRecord,
    corrupt_order,
    order_records,
    parse_event_records,
    render_records,
    stable_scramble,
    user_segment,
)


def test_parse_event_records_accepts_fenced_json() -> None:
    raw = """```json
{"events":[{"source_id":"turn_0002","event":"first"},{"source_id":"turn_0007","event":"second"}]}
```"""
    records = parse_event_records(raw, ("turn_0002", "turn_0007"))
    assert records[1].event == "second"


def test_parse_event_records_requires_complete_provenance() -> None:
    raw = '{"events":[{"source_id":"turn_0002","event":"first"}]}'
    with pytest.raises(ValueError, match="expected source ids"):
        parse_event_records(raw, ("turn_0002", "turn_0007"))


def test_order_and_corruption_use_source_turns() -> None:
    records = (
        EventRecord("turn_0007", "second"),
        EventRecord("turn_0002", "first"),
        EventRecord("turn_0011", "third"),
    )
    assert [record.event for record in order_records(records)] == ["first", "second", "third"]
    assert [record.event for record in corrupt_order(records)] == ["second", "third", "first"]


def test_render_sorted_trace_numbers_executed_order() -> None:
    records = (EventRecord("turn_0007", "second"), EventRecord("turn_0002", "first"))
    rendered = render_records(records, EvidenceCondition.EXECUTED_SORTED)
    assert "1. [turn_0002] first" in rendered
    assert "2. [turn_0007] second" in rendered


def test_stable_scramble_is_deterministic_and_not_chronological() -> None:
    items = ("turn_0001", "turn_0002", "turn_0003")
    assert stable_scramble(items, "example") == stable_scramble(items, "example")
    assert stable_scramble(items, "example") != items


def test_user_segment_drops_assistant_response() -> None:
    assert user_segment("user text ->-> 1,2 assistant text") == "user text"
