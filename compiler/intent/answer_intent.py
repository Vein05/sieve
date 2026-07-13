"""Canonical query intent contract for compiler planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .query_features import (
    abstention_sensitivity,
    answer_type_for_interrogative,
    cardinality_for_targets,
    extract_features,
    has_event_duration_anchor,
    has_current_state_marker,
    has_explicit_temporal_reference,
    has_latest_resolution_marker,
    has_update_resolution_marker,
    interrogative_for_query,
    is_average_query,
    is_delta_query,
    is_duration_query,
    is_duration_sum_query,
    is_extremum_selection_query,
    is_multi_count_query,
    is_numeric_query,
    is_strict_count_query,
    is_temporally_scoped_numeric_aggregate_query,
    needs_support_slot,
    operator_for_targets,
    target_unit,
)


@dataclass(frozen=True)
class AnswerIntent:
    query_family: str
    interrogative: str
    answer_type: str
    operator: str
    cardinality: str
    requires_support: bool
    requires_reference_time: bool
    requires_latest_resolution: bool
    abstention_sensitivity: str
    candidate_entities: tuple[str, ...]
    preferred_attributes: tuple[str, ...]
    target_unit: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_query_family(row: dict[str, Any], decision_summary: dict[str, Any] | None = None) -> str:
    family_hint = "single_anchor"
    query, targets = extract_features(row, family_hint)
    padded = f" {query} "
    normalized_decision = decision_summary or {}
    query_type_name = str(normalized_decision.get("query_type") or "").strip().lower()
    current_state_query = bool(normalized_decision.get("current_state_query"))
    aggregate_total_query = bool(
        any(marker in padded for marker in (" total ", " in total ", " altogether ", " combined ", " sum of ", " across all ", " across "))
        and (is_duration_query(query) or is_multi_count_query(query) or is_strict_count_query(query) or " how many times " in padded or " how much " in padded)
    )
    if query_type_name in {"knowledge_update", "stale_update"}:
        return "knowledge_update"
    if targets.asks_for_ordering:
        return "ordering"
    # Alternative + recency: "which of X or Y did I use most recently?" needs
    # OrderedChoice (recency sort), not temporal/DirectValue (no ordering logic).
    if (
        len(targets.alternative_entities) >= 2
        and has_latest_resolution_marker(query)
        and not targets.asks_for_comparison
    ):
        return "ordering"
    if targets.asks_for_comparison:
        return "aggregation"
    if aggregate_total_query:
        return "aggregation"
    if is_average_query(query) or is_extremum_selection_query(query) or is_delta_query(query) or is_duration_sum_query(query):
        return "aggregation"
    if (
        targets.asks_for_temporal_difference
        or targets.asks_for_relative_time
        or query_type_name == "temporal"
    ):
        return "temporal"
    if is_duration_query(query) and (" when i " in padded or has_event_duration_anchor(query)):
        return "temporal"
    if is_temporally_scoped_numeric_aggregate_query(query):
        return "aggregation"
    if has_latest_resolution_marker(query) and interrogative_for_query(query) in {"who", "what", "which", "where"}:
        return "current_state"
    if has_explicit_temporal_reference(query) and interrogative_for_query(query) in {"who", "what", "which", "where"}:
        return "temporal"
    if has_update_resolution_marker(query) and has_current_state_marker(query):
        return "knowledge_update"
    if is_duration_query(query) and any(marker in padded for marker in (" and ", " combined ", " total ", " altogether ")):
        return "aggregation"
    if targets.asks_for_comparison or is_multi_count_query(query) or is_strict_count_query(query):
        return "aggregation"
    if any(marker in padded for marker in (" total ", " in total ", " altogether ", " combined ", " compared ", " versus ", " vs ", " difference between ")):
        return "aggregation"
    if any(marker in padded for marker in (" percent ", " percentage ", " ratio ")):
        if any(marker in padded for marker in (" compared ", " versus ", " vs ", " difference between ", " total ", " combined ")):
            return "aggregation"
        if padded.startswith(" how much ") or padded.startswith(" what percentage "):
            return "aggregation"
    if targets.asks_for_current_state or current_state_query or has_current_state_marker(query):
        return "current_state"
    if interrogative_for_query(query) in {"who", "where", "when", "which", "what", "how_many", "how_much", "how_long"}:
        return "information_extraction"
    return "single_anchor"


def parse_answer_intent(row: dict[str, Any], query_family: str) -> AnswerIntent:
    family = str(query_family or "").strip().lower() or "single_anchor"
    query, targets = extract_features(row, family)
    numeric_query = is_numeric_query(query)
    duration_query = is_duration_query(query)
    interrogative = interrogative_for_query(query)
    requires_latest_resolution = family in {"current_state", "knowledge_update", "conflict_update"} or targets.asks_for_current_state or targets.asks_for_relative_time
    candidate_entities = tuple(targets.candidate_entities)
    if family in {"ordering", "aggregation"} and len(targets.alternative_entities) >= 2:
        candidate_entities = tuple(targets.alternative_entities) + tuple(
            entity for entity in candidate_entities if entity not in targets.alternative_entities
        )
    return AnswerIntent(
        query_family=family,
        interrogative=interrogative,
        answer_type=answer_type_for_interrogative(
            interrogative,
            numeric_query=numeric_query,
            duration_query=duration_query,
            current_state=family == "current_state" or targets.asks_for_current_state,
        ),
        operator=operator_for_targets(
            targets,
            family=family,
            query=query,
            numeric_query=numeric_query,
            duration_query=duration_query,
        ),
        cardinality=cardinality_for_targets(targets, family=family, query=query),
        requires_support=bool(targets.asks_for_recall_support or needs_support_slot(query)),
        requires_reference_time=bool(targets.asks_for_relative_time),
        requires_latest_resolution=requires_latest_resolution,
        abstention_sensitivity=abstention_sensitivity(targets),
        candidate_entities=candidate_entities,
        preferred_attributes=tuple(targets.preferred_attributes),
        target_unit=target_unit(query),
    )
