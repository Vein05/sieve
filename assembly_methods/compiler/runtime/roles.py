"""Shared role-salience policy for compiler runtime."""

from __future__ import annotations

_ROLE_PRIORITY: dict[str, int] = {
    "reference_time": 12,
    "current_resolution": 11,
    "new_state": 11,
    "comparison_left": 10,
    "comparison_right": 10,
    "temporal_event_a": 9,
    "temporal_event_b": 9,
    "temporal_time_a": 8,
    "temporal_time_b": 8,
    "ordering_event": 7,
    "state_anchor": 6,
    "old_state": 6,
    "numeric_operand": 5,
    "count_evidence": 4,
    "count_item": 15,
    "direct_anchor": 4,
    "support_anchor": 1,
    "competing_anchor": 2,
}

_PAIRED_ROLE_NAMES: dict[str, str] = {
    "comparison_left": "comparison_right",
    "comparison_right": "comparison_left",
    "temporal_event_a": "temporal_event_b",
    "temporal_event_b": "temporal_event_a",
    "temporal_time_a": "temporal_time_b",
    "temporal_time_b": "temporal_time_a",
    "old_state": "new_state",
    "new_state": "old_state",
}


def _role_priority(role: str) -> int:
    return int(_ROLE_PRIORITY.get(str(role), 1))


def _missing_role_gain(roles: set[str], missing: dict[str, int]) -> tuple[int, int]:
    covered_count = 0
    weighted_gain = 0
    for role, deficit in missing.items():
        if role in roles and deficit > 0:
            covered_count += 1
            weighted_gain += _role_priority(role)
    return covered_count, weighted_gain
