"""Entity and temporal matching helpers for the compiler runtime."""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any

from ...common import cached_build_profile, get_stems_for_text
from ...query_targets import QueryTargets
from ..execution.requirements import is_direct_lookup_schema
from ..execution.requirements import aggregate_requires_two_operands as _aggregate_requires_two_operands, aggregate_slot as _aggregate_slot
from .focus import _WEAK_QUERY_FOCUS_TOKENS, _query_focus_tokens
from .temporal import _binding_anchor_date, _extract_event_date, _number_word_value, _shift_months
from .text import _binding_source_text, _normalize_text, _query_expects_time_literal, _requested_duration_unit, _time_unit_from_query
_ENTITY_MATCH_SKIP_TOKENS = {"a", "an", "and", "at", "for", "from", "i", "in", "local", "my", "of", "on", "or", "the", "to"}
_GENERIC_ENTITY_SUFFIX_TOKENS = {"cost", "expense", "expenses", "fare", "price", "prices", "ride", "ticket", "tickets", "trip", "trips"}


def _match_entity_surface(source_text: str, entity: str) -> str | None:
    if not entity or not source_text:
        return None
    e_stems = get_stems_for_text(entity)
    s_stems = set(get_stems_for_text(source_text))
    if not e_stems:
        return None
    
    from ...common import CONVERSATIONAL_SYNONYMS
    
    matched_stems = 0
    for e_stem in e_stems:
        # Direct match or synonym match
        if e_stem in s_stems:
            matched_stems += 1
            continue
            
        syns = CONVERSATIONAL_SYNONYMS.get(e_stem)
        if syns and (syns & s_stems):
            matched_stems += 1
            
    if matched_stems == 0:
        return None
    return entity


def _best_matching_entity(source_text: str, candidate_entities: tuple[str, ...]) -> str | None:
    best: tuple[int, int, str] | None = None
    normalized_source = _normalize_text(source_text)
    for entity in candidate_entities:
        if not _match_entity_surface(source_text, entity):
            continue
        normalized_entity = _normalize_text(entity)
        exact = 1 if re.search(rf"(?<!\w){re.escape(normalized_entity)}(?!\w)", normalized_source) else 0
        key = (exact, len(normalized_entity.split()), normalized_entity)
        if best is None or key > best:
            best = key
    return None if best is None else best[2]


def _slot_expected_entities(slot, query_targets: QueryTargets) -> tuple[str, ...]:
    entity_key = str(getattr(slot, "entity_key", "") or "").strip()
    candidate_entities = tuple(entity for entity in query_targets.candidate_entities if entity)
    if entity_key and candidate_entities:
        normalized_hint = _normalize_text(entity_key)
        hint_tokens = set(normalized_hint.split())
        related = [
            entity
            for entity in candidate_entities
            if normalized_hint in _normalize_text(entity)
            or _normalize_text(entity) in normalized_hint
            or bool(set(_normalize_text(entity).split()) & hint_tokens)
        ]
        if related:
            return tuple(dict.fromkeys(related))
    return query_targets.subject_entities or candidate_entities


def _slot_entity_match(slot, source_text: str, query_targets: QueryTargets) -> str | None:
    expected_entities = _slot_expected_entities(slot, query_targets)
    return _best_matching_entity(source_text, expected_entities)


def _question_date_binding(row: dict[str, Any]) -> dict[str, Any] | None:
    question_date = str(row.get("question_date") or "").strip()
    match = re.search(r"(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})", question_date)
    if not match:
        return None
    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    return {"unit_id": "question_date", "memory_id": "question_date", "text": match.group(0), "date": [year, month, day], "speaker": "system", "source_text": question_date}


def _event_matches_expected(binding: dict[str, Any] | None, expected_entity: str | None) -> bool:
    if not expected_entity or not isinstance(binding, dict):
        return True
    source_text = _binding_source_text(binding)
    return bool(_match_entity_surface(source_text, expected_entity))


def _direct_value_requires_entity_match(*, row: dict[str, Any], plan: Any, query_targets: QueryTargets) -> bool:
    if not is_direct_lookup_schema(plan):
        return False
    if plan.family != "aggregation":
        return False
    if not query_targets.subject_entities:
        return False
    normalized_query = _normalize_text(str(row.get("query", "")))
    if normalized_query.startswith("where ") or normalized_query.startswith("when "):
        return False
    return bool(re.search(r"\bhow\s+long\b", normalized_query))
