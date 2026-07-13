"""Heuristic role inference for evidence units."""

from __future__ import annotations

import re
from typing import Any

from .units import EvidenceUnit
from retrieval.query_targets import QueryTargets

from shared.nlp import cached_build_profile, stem_token, cached_word_tokenize, cached_pos_tag
from shared.constants import (
    COMPARISON_FALLBACK_MARKERS as _COMPARISON_FALLBACK_MARKERS,
    ORDERING_ROLE_MARKERS as _ORDERING_MARKERS,
    CURRENT_STATE_WORDS as _CURRENT_STATE_WORDS,
)

_SOURCE_NUMERIC_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_ANSWERISH_SOURCE_RE = re.compile(
    r'\b(?:called|named|degree in|breed is|breed was|got from|bought from|'
    r'got|bought|purchased|ordered|booked|attended|beat|used to be|'
    r'use|uses|used|visit|visits|visited|tried|try|'
    r'service|serviced|fix|fixed|sell|sold|assemble|assembled|acquired|acquire|'
    r'finish|finished|complete|completed|return|returned|exchange|exchanged|'
    r'pick up|picked up|work on|worked on|download|downloaded|'
    r'worked as|work as|role as|occupation|job as|employed as|'
    r'working at|currently at|settled on|case of|'
    r'received from|received|bought from|'
    r'presented at|presented|presenting|'
    r'cooking|cooked|baked|prepared|'
    r'earned|cashback|saved|'
    r'at [A-Z]|from [A-Z]|to [A-Z])\b',
    re.IGNORECASE,
)
_DURATION_PHRASE_RE = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|few|several)\s+"
    r"(?:days?|weeks?|months?|years?|hours?|minutes?)\b",
    re.IGNORECASE,
)
_CERTIFICATION_SOURCE_RE = re.compile(r"\bcertification in\s+[A-Z]", re.IGNORECASE)
_FAVORITE_VALUE_RE = re.compile(r"\bmy favorite\s+[^,.!?;]{1,80}", re.IGNORECASE)
_WHERE_IT_WAS_RE = re.compile(r"\bit was\s+(?:a|an|the|at|in)\s+[^,.!?;]{1,80}", re.IGNORECASE)
_WHERE_LOCATION_RE = re.compile(
    r"\b(?:at|in|from|to)\s+(?:(?:the|a|an)\s+)?(?:[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4}|[a-z][^,.!?;]{1,60})\b"
)
_FIRST_PERSON_RE = re.compile(r"\b(?:i|i'm|i’ve|i've|my|me|mine|we|our|us)\b", re.IGNORECASE)
_GENERIC_TYPE_LIST_RE = re.compile(
    r"\b(?:there (?:are|were)|different types of|different kinds of|types of|kinds of|varieties of|here are some)\b",
    re.IGNORECASE,
)
_COUNT_SCAFFOLD_TOKENS = frozenset({
    "different",
    "distinct",
    "kind",
    "kinds",
    "type",
    "types",
    "varieties",
    "variety",
})

_ENTITY_MATCH_STOPWORDS = frozenset({
    "a",
    "an",
    "and",
    "at",
    "before",
    "between",
    "by",
    "day",
    "days",
    "did",
    "do",
    "for",
    "from",
    "i",
    "in",
    "it",
    "my",
    "of",
    "on",
    "the",
    "to",
    "was",
    "were",
    "with",
})
_GENERIC_ORDERING_ENTITY_TOKENS = frozenset({
    "arrival",
    "event",
    "participation",
    "post",
    "task",
})

_COUNT_LOOKUP_OBJECT_SKIP_TOKENS = frozenset({
    "many",
    "much",
    "total",
    "number",
    "count",
    "amount",
    "often",
    "time",
    "times",
    "item",
    "items",
})

_COUNT_TARGET_NOISE_TOKENS = frozenset({
    "all",
    "amount",
    "both",
    "combined",
    "count",
    "different",
    "distinct",
    "either",
    "kind",
    "kinds",
    "many",
    "much",
    "number",
    "overall",
    "pair",
    "pairs",
    "piece",
    "pieces",
    "separate",
    "sort",
    "sorts",
    "species",
    "total",
    "type",
    "types",
    "varieties",
    "variety",
})


def _typed_roles(
    *,
    query_family: str,
    query_targets: QueryTargets,
    unit: EvidenceUnit,
) -> set[str]:
    roles: set[str] = set()
    preferred_attributes = {str(value).strip().lower() for value in query_targets.preferred_attributes if value}
    unit_attribute = str((unit.provenance.get("schema_hints", {}) or {}).get("attribute_key") or "").strip().lower()

    if unit.is_temporal_anchor and query_family in {"temporal", "ordering"}:
        roles.update({"event_anchor", "time_anchor"})
    if unit.is_update_assertion:
        roles.add("update_anchor")
        if query_family in {"current_state", "knowledge_update", "conflict_update"}:
            roles.add("state_anchor")
    if unit.is_current_state_candidate and query_family == "current_state":
        roles.update({"state_anchor", "current_resolution"})
    if unit.is_attribute_value:
        if not preferred_attributes or (unit_attribute and unit_attribute in preferred_attributes):
            roles.add("direct_anchor")
            if query_family == "current_state":
                roles.add("current_resolution")
    if unit.is_countable_item and query_family == "aggregation" and not (
        unit.is_question_or_request or unit.is_recommendation_or_advice
    ):
        roles.update({"support_anchor", "count_item", "count_evidence"})
    return roles


def _typed_current_state_signal(unit: EvidenceUnit) -> bool:
    return bool(unit.is_current_state_candidate or unit.is_state_assertion or unit.is_update_assertion)


def _stem_token(token: str) -> str:
    return stem_token(token)


def _entity_phrase_matches_unit(entity: str, unit: EvidenceUnit, unit_profile) -> bool:
    phrase = str(entity or "").strip().lower()
    if not phrase:
        return False
    if phrase in unit_profile.normalized_text:
        return True
    phrase_tokens = [token for token in phrase.split() if token]
    entity_tokens = {
        token
        for token in phrase_tokens
        if token and token not in _ENTITY_MATCH_STOPWORDS
    }
    if not entity_tokens:
        return False
    unit_tokens = (
        set(unit.entity_tokens)
        | set(unit.subject_tokens)
        | set(unit.value_tokens)
        | set(unit_profile.content_tokens)
    )
    entity_stems = {_stem_token(token) for token in entity_tokens}
    unit_stems = {_stem_token(token) for token in unit_tokens}
    overlap = entity_tokens & unit_tokens
    stem_overlap = entity_stems & unit_stems

    # Synonym-aware expansion
    if not (overlap or stem_overlap):
        from shared.nlp import CONVERSATIONAL_SYNONYMS
        for e_stem in entity_stems:
            syns = CONVERSATIONAL_SYNONYMS.get(e_stem, set())
            if syns & unit_stems:
                stem_overlap.add(e_stem) # Treat synonym hit as a stem hit
                break

    entity_tags = cached_pos_tag(tuple(phrase_tokens)) if phrase_tokens else []
    noun_tokens = {
        token
        for token, tag in entity_tags
        if tag.startswith("NN") and token not in _ENTITY_MATCH_STOPWORDS
    }
    adjective_tokens = {
        token
        for token, tag in entity_tags
        if tag.startswith("JJ") and token not in _ENTITY_MATCH_STOPWORDS
    }
    noun_stems = {_stem_token(token) for token in noun_tokens}
    adjective_stems = {_stem_token(token) for token in adjective_tokens}
    noun_hits = (noun_tokens & unit_tokens) | (noun_stems & unit_stems)
    adjective_hits = (adjective_tokens & unit_tokens) | (adjective_stems & unit_stems)
    weak_modifiers = {"vintage", "daily", "local", "new", "old", "current", "previous", "favorite", "final", "total"}
    meaningful_adjective_tokens = {
        token
        for token in adjective_tokens
        if token not in weak_modifiers
    }
    meaningful_adjective_stems = {_stem_token(token) for token in meaningful_adjective_tokens}
    meaningful_adjective_hits = (
        (meaningful_adjective_tokens & unit_tokens)
        | (meaningful_adjective_stems & unit_stems)
    )
    if meaningful_adjective_tokens and not meaningful_adjective_hits:
        return False
    if len(entity_tokens) == 1:
        return bool(overlap or stem_overlap)
    action_like = {
        token
        for token in (entity_tokens | entity_stems)
        if token.endswith("ing")
        or token in {
            "arrive",
            "attend",
            "attended",
            "bake",
            "baked",
            "begin",
            "buy",
            "dedicate",
            "end",
            "finish",
            "harvest",
            "harvested",
            "leave",
            "make",
            "made",
            "move",
            "moved",
            "practice",
            "practicing",
            "sell",
            "start",
            "visit",
            "visited",
        }
    }
    uncommon = {
        token
        for token in (entity_tokens | entity_stems)
        if token not in {"fare", "first", "last", "order", "ride", "price", "cost", "discount"}
    }
    phrase_markers = (" with ", " at ", " in ", " on ", " after ", " before ", " during ", " while ")
    structured_event_phrase = bool(action_like) or any(marker in f" {phrase} " for marker in phrase_markers)
    if structured_event_phrase and noun_tokens:
        relation_heavy_phrase = any(marker in f" {phrase} " for marker in (" with ", " at ", " after ", " before ", " during ", " while "))
        if len(noun_tokens) >= 2:
            if relation_heavy_phrase and len(noun_hits) < 2:
                return False
            if len(noun_hits) < 2 and not (noun_hits and action_like & (overlap | stem_overlap)):
                return False
        elif not noun_hits:
            return False
    if len(overlap) >= 2 or len(stem_overlap) >= 2:
        return bool(uncommon & (overlap | stem_overlap))
    if len(overlap | stem_overlap) >= 1:
        if action_like and not (action_like & (overlap | stem_overlap)):
            return False
        if meaningful_adjective_tokens and not adjective_hits:
            return False
        if len(overlap | stem_overlap) == 1 and (overlap | stem_overlap) <= weak_modifiers:
            return False
        return bool(uncommon & (overlap | stem_overlap))
    return False


def _ordering_entity_fallback_match(entity: str, unit_profile) -> bool:
    phrase = str(entity or "").strip().lower()
    if not phrase:
        return False
    if phrase in unit_profile.normalized_text:
        return True
    ordered_tokens = [
        token
        for token in phrase.split()
        if token and token not in _ENTITY_MATCH_STOPWORDS and token not in _GENERIC_ORDERING_ENTITY_TOKENS
    ]
    if not ordered_tokens:
        ordered_tokens = [token for token in phrase.split() if token and token not in _ENTITY_MATCH_STOPWORDS]
    if not ordered_tokens:
        return False
    phrase_stems = [_stem_token(token) for token in ordered_tokens]
    unit_stems = {_stem_token(token) for token in unit_profile.content_tokens | unit_profile.named_tokens}
    overlap = [stem for stem in phrase_stems if stem in unit_stems]
    if not overlap:
        return False
    if len(phrase_stems) == 1:
        return True
    return phrase_stems[-1] in overlap or len(overlap) >= 2


def _comparison_roles(
    *,
    query_targets: QueryTargets,
    unit: EvidenceUnit,
    unit_profile,
    strong_match: bool,
    lexical_match: bool,
) -> set[str]:
    roles: set[str] = set()
    if not query_targets.asks_for_comparison:
        return roles

    first_entity = query_targets.candidate_entities[0] if len(query_targets.candidate_entities) >= 1 else ""
    second_entity = query_targets.candidate_entities[1] if len(query_targets.candidate_entities) >= 2 else ""
    has_pair_targets = bool(first_entity and second_entity)
    first_match = _entity_phrase_matches_unit(first_entity, unit, unit_profile)
    second_match = _entity_phrase_matches_unit(second_entity, unit, unit_profile)

    if first_match and not second_match:
        roles.add("comparison_left")
    elif second_match and not first_match:
        roles.add("comparison_right")
    elif first_match and second_match:
        roles.update({"comparison_left", "comparison_right"})
    elif not has_pair_targets and (unit.has_comparison_language or strong_match or lexical_match):
        roles.add("comparison_left" if unit.parent_rank % 2 == 1 else "comparison_right")

    return roles


def _temporal_roles(
    *,
    query_family: str,
    query_targets: QueryTargets,
    unit: EvidenceUnit,
    unit_profile,
    strong_match: bool,
    query_overlap: int,
) -> set[str]:
    roles: set[str] = set()
    if query_family != "temporal" and query_family != "ordering":
        return roles

    has_time_signal = bool(unit.time_markers) or unit.unit_type == "dated_event" or unit.date_key != (0, 0, 0)
    temporal_pair_language = any(
        marker in f" {unit_profile.normalized_text} "
        for marker in (" between ", " from ", " to ", " after ", " before ", " until ", " since ")
    )
    if query_family == "ordering":
        pair_entities = tuple(entity for entity in (query_targets.alternative_entities or query_targets.candidate_entities) if entity)
    else:
        pair_entities = tuple(entity for entity in query_targets.candidate_entities if entity)
    first_entity = pair_entities[0] if len(pair_entities) >= 1 else ""
    second_entity = pair_entities[1] if len(pair_entities) >= 2 else ""
    first_match = _entity_phrase_matches_unit(first_entity, unit, unit_profile) if first_entity else False
    second_match = _entity_phrase_matches_unit(second_entity, unit, unit_profile) if second_entity else False
    if query_family == "ordering":
        if not first_match and first_entity:
            first_match = _ordering_entity_fallback_match(first_entity, unit_profile)
        if not second_match and second_entity:
            second_match = _ordering_entity_fallback_match(second_entity, unit_profile)

    if query_family == "ordering":
        if has_time_signal and (
            strong_match
            or unit.has_comparison_language
            or first_match
            or second_match
            or unit.parent_rank <= 4
        ):
            roles.add("ordering_event")
        return roles

    if query_targets.asks_for_temporal_difference and has_time_signal:
        if first_match and not second_match:
            roles.update({"temporal_event_a", "temporal_time_a"})
        elif second_match and not first_match:
            roles.update({"temporal_event_b", "temporal_time_b"})
        elif first_match and second_match:
            roles.update({"temporal_event_a", "temporal_time_a", "temporal_event_b", "temporal_time_b"})
        elif len(query_targets.candidate_entities) < 2 and query_overlap >= 2 and (strong_match or temporal_pair_language):
            if unit.parent_rank % 2 == 1:
                roles.update({"temporal_event_a", "temporal_time_a"})
            else:
                roles.update({"temporal_event_b", "temporal_time_b"})
        return roles

    if query_targets.asks_for_relative_time:
        if first_match and (has_time_signal or strong_match):
            roles.add("temporal_event_a")
            if unit.time_markers:
                roles.add("temporal_time_a")
        if second_entity:
            if second_match and (has_time_signal or strong_match or unit.has_reference_language):
                roles.add("reference_time")
        elif _typed_current_state_signal(unit) and has_time_signal and first_match:
            roles.add("reference_time")
        elif unit.has_reference_language and has_time_signal and first_match:
            roles.add("reference_time")
        return roles

    if has_time_signal or strong_match:
        if first_match and not second_match:
            roles.add("temporal_event_a")
            if unit.time_markers:
                roles.add("temporal_time_a")
        elif second_match and not first_match:
            roles.add("temporal_event_b")
            if unit.time_markers:
                roles.add("temporal_time_b")
        elif unit.parent_rank % 2 == 1:
            roles.add("temporal_event_a")
            if unit.time_markers:
                roles.add("temporal_time_a")
        else:
            roles.add("temporal_event_b")
            if unit.time_markers:
                roles.add("temporal_time_b")
    return roles


def _state_roles(
    *,
    query_family: str,
    unit: EvidenceUnit,
    strong_match: bool,
    lexical_match: bool,
) -> set[str]:
    """Typed fields are the authority; lexical signals only supplement when the typed field is absent."""
    roles: set[str] = set()
    if query_family == "current_state":
        # Typed path: explicit flags set during unit construction
        if unit.is_current_state_candidate or unit.is_state_assertion or unit.is_update_assertion:
            roles.update({"current_resolution", "state_anchor"})
        # Lexical fallback only when typed fields didn't fire
        elif unit.has_current_state_language or unit.update_markers:
            roles.add("current_resolution")
            if unit.has_reference_language or strong_match or lexical_match:
                roles.add("state_anchor")
        elif unit.has_reference_language or strong_match or lexical_match:
            roles.add("state_anchor")
    elif query_family in {"knowledge_update", "conflict_update"}:
        if unit.is_update_assertion:
            # Typed update: map to update roles directly
            roles.update({"old_state", "new_state"})
        else:
            old_markers = {"old", "previous", "prior", "former", "before", "used"}
            new_markers = {"current", "currently", "new", "now", "switch", "switched", "change", "changed", "updated", "update", "instead"}
            has_old_language = unit.has_reference_language or bool(
                old_markers & (set(unit.value_tokens) | set(unit.update_markers))
            )
            has_new_language = unit.has_current_state_language or bool(
                new_markers & (set(unit.value_tokens) | set(unit.update_markers))
            )
            if has_old_language:
                roles.add("old_state")
            if has_new_language:
                roles.add("new_state")
            if not roles and (strong_match or lexical_match):
                roles.add("old_state")
    return roles


def _duration_like_query(query_text: str, query_family: str, query_targets: QueryTargets) -> bool:
    if query_family != "aggregation" or query_targets.asks_for_comparison or query_targets.asks_for_ordering:
        return False
    padded = f" {query_text} "
    if " how long " in padded:
        return True
    return bool(re.search(r"\bhow\s+many\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\b", query_text))


def _count_like_query(query_text: str) -> bool:
    padded = f" {query_text} "
    return any(
        marker in padded
        for marker in (
            " how many ",
            " number of ",
            " count of ",
            " how often ",
            " amount of ",
        )
    )


def _extract_count_object_stems(query: str) -> set[str]:
    """Extract lemmatized object tokens that are being counted."""
    query = query.lower()
    tokens = cached_word_tokenize(query)
    tags = cached_pos_tag(tuple(tokens))
    
    found_many_trigger = False
    object_stems = set()
    
    for word, tag in tags:
        if word in {"many", "number", "count", "often", "amount"}:
            found_many_trigger = True
            continue
        if found_many_trigger:
            if tag.startswith(("NN", "JJ")):
                if (
                    word not in _COUNT_LOOKUP_OBJECT_SKIP_TOKENS
                    and word not in _ENTITY_MATCH_STOPWORDS
                    and word not in _COUNT_SCAFFOLD_TOKENS
                ):
                    stem = stem_token(word)
                    if len(stem) > 2:
                        object_stems.add(stem)
            elif tag.startswith("VB"):
                # Stop at the next verb usually
                break
                
    return object_stems


def _count_target_stem_sets(query_targets: QueryTargets) -> list[set[str]]:
    ordered_targets = list(query_targets.subject_entities) + [
        entity for entity in query_targets.candidate_entities if entity not in query_targets.subject_entities
    ]
    target_stem_sets: list[set[str]] = []
    for entity in ordered_targets[:6]:
        tokens = [
            token
            for token in cached_word_tokenize(str(entity or "").lower())
            if token
            and token not in _ENTITY_MATCH_STOPWORDS
            and token not in _COUNT_TARGET_NOISE_TOKENS
        ]
        stems = {_stem_token(token) for token in tokens if len(_stem_token(token)) > 2}
        if stems:
            target_stem_sets.append(stems)
    return target_stem_sets


def _duration_like_unit(normalized_text: str) -> bool:
    return bool(_DURATION_PHRASE_RE.search(normalized_text))


_NUMERIC_CONTEXT_MARKERS = frozenset({
    " amount ", " assist ", " assists ", " bill ", " bills ",
    " charge ", " charges ", " cost ", " costs ", " day ", " days ",
    " discount ", " fare ", " fee ", " fees ", " goal ", " goals ",
    " hour ", " hours ", " minute ", " minutes ", " month ", " months ",
    " paid ", " payment ", " payments ", " price ", " receipt ",
    " save ", " saved ", " score ", " scored ", " spend ", " spent ",
    " taxes ", " value ", " week ", " weeks ", " year ", " years ",
})

_EXPLICIT_REFERENCE_PHRASES = frozenset({
    "as of", "current as of", "reference date", "reference time",
    "reference point", "today is", "now is", "at this time",
})

_RECALL_SUPPORT_UNIT_MARKERS = frozenset({
    "recommend", "recommended", "mention", "mentioned",
    "told you", "told me", "the one", "last time", "yesterday",
})

_SMALL_NUMBER_WORDS = frozenset({
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
})


def _compute_role_signals(
    *,
    row: dict[str, Any],
    query_family: str,
    query_targets: QueryTargets,
    unit: EvidenceUnit,
) -> dict[str, Any]:
    """Compute all matching signals used by role assignment."""
    query_profile = cached_build_profile(str(row.get("query", "")))
    source_text = str(unit.provenance.get("source_text") or unit.render_text)
    unit_profile = cached_build_profile(source_text)
    normalized = unit_profile.normalized_text
    query_text = query_targets.normalized_query
    source_numeric_values = [m.group(0) for m in _SOURCE_NUMERIC_RE.finditer(source_text)]
    query_content_tokens = set(query_profile.content_tokens)
    unit_content_tokens = set(unit_profile.content_tokens)
    padded_normalized = f" {normalized} "
    query_stems = {_stem_token(t) for t in query_content_tokens}
    unit_stems = {_stem_token(t) for t in unit_content_tokens}
    query_overlap = max(
        len(query_content_tokens & unit_content_tokens),
        len(query_stems & unit_stems),
    )
    named_overlap = bool(query_profile.named_tokens & set(unit.entity_tokens))
    explicit_numeric = any(
        v.startswith("$") or "%" in v or "." in v or "," in v or v.isdigit()
        for v in source_numeric_values
    ) or any(f" {w} " in padded_normalized for w in _SMALL_NUMBER_WORDS)
    has_time = unit.is_temporal_anchor or bool(unit.time_markers) or unit.unit_type == "dated_event"
    has_update = unit.is_update_assertion or bool(unit.update_markers)
    has_answerish_source = bool(_ANSWERISH_SOURCE_RE.search(source_text))
    has_explicit_reference = any(p in normalized for p in _EXPLICIT_REFERENCE_PHRASES)
    has_current_state_language = any(p in normalized for p in _CURRENT_STATE_WORDS)
    moderate_lexical_match = query_overlap >= 2
    duration_query = _duration_like_query(query_text, query_family, query_targets)
    duration_unit = _duration_like_unit(normalized)
    count_query = _count_like_query(query_text) and not duration_query
    count_target_stem_sets = _count_target_stem_sets(query_targets) if count_query else []
    primary_entities = tuple(e for e in query_targets.subject_entities[:2] if e) or tuple(
        e for e in query_targets.candidate_entities if e
    )
    entity_match = any(_entity_phrase_matches_unit(e, unit, unit_profile) for e in primary_entities)
    count_friendly_numeric = explicit_numeric and not _DURATION_PHRASE_RE.search(source_text) and not any(
        v.startswith("$") or "%" in v for v in source_numeric_values
    )
    typed_count_evidence = (
        unit.is_answer_like_quantity_statement
        or unit.is_instructional_quantity
        or unit.has_progress_marker
        or (unit.is_countable_item and (unit.has_acquisition_marker or unit.has_consumption_or_completion_marker))
    )
    count_target_match = any(ss & unit_stems for ss in count_target_stem_sets)
    return {
        "unit_profile": unit_profile,
        "normalized": normalized,
        "query_text": query_text,
        "source_text": source_text,
        "padded_normalized": padded_normalized,
        "unit_stems": unit_stems,
        "query_overlap": query_overlap,
        "named_overlap": named_overlap,
        "explicit_numeric": explicit_numeric,
        "has_time": has_time,
        "has_update": has_update,
        "has_answerish_source": has_answerish_source,
        "has_explicit_reference": has_explicit_reference,
        "has_current_state_language": has_current_state_language,
        "moderate_lexical_match": moderate_lexical_match,
        "strong_named_match": named_overlap,
        "strong_lexical_match": query_overlap >= 3,
        "recommend_lookup": "recommend" in query_text and "recommend" in normalized,
        "designation_lookup": (
            any(m in query_text for m in ("designation", "name", "called"))
            and any(m in source_text for m in ('"', "'", ":"))
        ),
        "duration_query": duration_query,
        "duration_unit": duration_unit,
        "count_query": count_query,
        "count_target_stem_sets": count_target_stem_sets,
        "comparison_like_query": query_family == "aggregation" and (
            query_targets.asks_for_comparison
            or any(m in f" {query_text} " for m in _COMPARISON_FALLBACK_MARKERS)
        ),
        "choice_like_query": " or " in f" {query_text} ",
        "ordering_like_query": query_family == "ordering" or any(
            m in f" {query_text} " for m in _ORDERING_MARKERS
        ),
        "entity_match": entity_match,
        "numeric_context_match": any(m in padded_normalized for m in _NUMERIC_CONTEXT_MARKERS),
        "count_friendly_numeric": count_friendly_numeric,
        "typed_count_item": unit.is_countable_item,
        "typed_count_evidence": typed_count_evidence,
        "event_count_activity": bool(
            unit.has_acquisition_marker or unit.has_consumption_or_completion_marker or unit.is_countable_item
        ),
        "count_target_match": count_target_match,
        "matched_count_target_count": sum(1 for ss in count_target_stem_sets if ss & unit_stems),
        "count_activity_like": (
            unit.is_countable_item or explicit_numeric
            or (has_answerish_source and (unit.speaker == "user" or unit.has_pronoun_language))
        ),
        "event_like": (
            named_overlap or query_overlap >= 2
            or (unit.unit_type == "dated_event" and not has_explicit_reference and query_overlap >= 1)
        ),
    }


def _assign_direct_anchor_tags(
    *,
    tags: set[str],
    query_family: str,
    query_targets: QueryTargets,
    unit: EvidenceUnit,
    sig: dict[str, Any],
) -> None:
    """Add direct_anchor and support_anchor tags based on matching signals."""
    if sig["comparison_like_query"] or sig["choice_like_query"]:
        if sig["strong_named_match"] or sig["strong_lexical_match"]:
            tags.add("direct_anchor")
    elif sig["strong_named_match"] or sig["strong_lexical_match"]:
        tags.add("direct_anchor")
    if query_family in {"current_state", "knowledge_update", "conflict_update"} and (
        unit.is_current_state_candidate or unit.is_state_assertion or unit.is_update_assertion
        or (sig["has_current_state_language"] and (sig["named_overlap"] or sig["query_overlap"] >= 1))
        or (sig["has_update"] and (sig["named_overlap"] or sig["query_overlap"] >= 1))
    ):
        tags.add("direct_anchor")
    is_sa_ie = query_family in {"single_anchor", "information_extraction"}
    if is_sa_ie and unit.is_direct_answer_candidate:
        tags.add("direct_anchor")
    if is_sa_ie and (sig["recommend_lookup"] or sig["designation_lookup"]):
        if (
            unit.is_direct_answer_candidate or sig["designation_lookup"]
            or sig["entity_match"] or len(sig["normalized"].split()) <= 8
            or not sig["recommend_lookup"]
        ):
            tags.add("direct_anchor")
    if is_sa_ie and _CERTIFICATION_SOURCE_RE.search(sig["source_text"]):
        tags.add("direct_anchor")
    if is_sa_ie and _FAVORITE_VALUE_RE.search(sig["source_text"]):
        tags.add("direct_anchor")
    if is_sa_ie and "ethnicity" in sig["query_text"] and "ethnicity" in sig["normalized"]:
        tags.add("direct_anchor")
    if is_sa_ie and "stance" in sig["query_text"] and "used to be" in sig["normalized"]:
        tags.add("direct_anchor")
    if sig["duration_query"] and sig["duration_unit"]:
        tags.add("direct_anchor")
    if query_family in {"single_anchor", "information_extraction", "aggregation"} and sig["explicit_numeric"] and sig["moderate_lexical_match"]:
        tags.add("direct_anchor")
    if query_targets.asks_for_recall_support and sig["entity_match"]:
        tags.add("direct_anchor")
    if is_sa_ie and sig["query_overlap"] >= 1 and sig["has_answerish_source"]:
        tags.add("direct_anchor")
    if is_sa_ie and sig["query_text"].startswith("where "):
        if _WHERE_LOCATION_RE.search(sig["source_text"]):
            tags.add("direct_anchor")
        if _WHERE_IT_WAS_RE.search(sig["source_text"]):
            tags.add("direct_anchor")
    if sig["moderate_lexical_match"] or sig["strong_named_match"] or unit.is_direct_answer_candidate:
        tags.add("support_anchor")


def _assign_count_query_tags(
    *,
    tags: set[str],
    query_family: str,
    query_targets: QueryTargets,
    unit: EvidenceUnit,
    sig: dict[str, Any],
) -> bool:
    """Apply count-query role tags. Returns True if caller should early-return tags."""
    if query_family != "aggregation" or not sig["count_query"]:
        return False
    if unit.is_question_or_request or unit.is_recommendation_or_advice:
        answer_bearing = bool(
            sig["typed_count_evidence"]
            and (sig["count_target_match"] or sig["query_overlap"] >= 1 or sig["entity_match"])
            and sig["explicit_numeric"]
        )
        if not answer_bearing:
            tags.discard("count_item")
            tags.discard("count_evidence")
            return True
    if _GENERIC_TYPE_LIST_RE.search(sig["source_text"]):
        subject_entity_match = bool(
            query_targets.subject_entities
            and any(ss & sig["unit_stems"] for ss in _count_target_stem_sets(query_targets))
        )
        needs_multi = len(tuple(e for e in query_targets.subject_entities if e)) >= 2
        if (
            unit.speaker == "assistant" and not _FIRST_PERSON_RE.search(sig["source_text"])
            and (not subject_entity_match or (needs_multi and sig["matched_count_target_count"] < 2))
        ):
            tags.discard("count_item")
            tags.discard("count_evidence")
    required_subject_matches = len(tuple(e for e in query_targets.subject_entities if e))
    if (
        unit.speaker == "assistant" and required_subject_matches >= 2
        and sig["matched_count_target_count"] < 2
        and not _FIRST_PERSON_RE.search(sig["source_text"])
    ):
        tags.discard("count_item")
        tags.discard("count_evidence")
    if sig["typed_count_evidence"] or (
        sig["count_friendly_numeric"] and sig["count_target_match"]
        and (sig["moderate_lexical_match"] or sig["entity_match"] or sig["query_overlap"] >= 1)
    ):
        tags.add("count_evidence")
    count_object_stems = _extract_count_object_stems(sig["query_text"])
    fallback_count_match = bool(count_object_stems and (count_object_stems & sig["unit_stems"]))
    prefer_fact_statement = bool(
        (unit.is_answer_like_quantity_statement or unit.is_instructional_quantity or unit.has_progress_marker)
        and not sig["typed_count_item"] and not unit.has_consumption_or_completion_marker
    )
    if (
        (sig["count_target_match"] or fallback_count_match)
        and (sig["count_activity_like"] or sig["event_count_activity"])
        and not unit.is_recommendation_or_advice and not prefer_fact_statement
    ):
        tags.add("count_item")
        tags.add("support_anchor")
        if sig["typed_count_evidence"] or sig["explicit_numeric"]:
            tags.add("count_evidence")
    if sig["typed_count_item"] and (
        sig["event_like"] or sig["entity_match"] or sig["query_overlap"] >= 1
    ):
        tags.add("count_item")
        tags.add("support_anchor")
    return False


def _assign_family_specific_tags(
    *,
    tags: set[str],
    query_family: str,
    query_targets: QueryTargets,
    unit: EvidenceUnit,
    sig: dict[str, Any],
) -> None:
    """Apply family-specific support, state, and temporal tags."""
    is_sa_ie = query_family in {"single_anchor", "information_extraction"}
    if is_sa_ie and sig["query_text"].startswith("where "):
        if _WHERE_LOCATION_RE.search(sig["source_text"]):
            tags.add("support_anchor")
        if _WHERE_IT_WAS_RE.search(sig["source_text"]):
            tags.add("support_anchor")
    if query_family == "current_state":
        if sig["has_current_state_language"] and (sig["named_overlap"] or sig["query_overlap"] >= 2):
            tags.add("current_resolution")
            tags.add("state_anchor")
        if sig["strong_named_match"] or sig["moderate_lexical_match"]:
            tags.add("state_anchor")
            tags.add("direct_anchor")
    if query_family in {"knowledge_update", "conflict_update"} and (sig["has_update"] or sig["has_explicit_reference"]):
        tags.add("update_anchor")
        tags.add("state_anchor")
    if query_family == "temporal":
        if sig["has_time"] and (
            sig["event_like"]
            or (unit.unit_type in {"dated_event", "update_fact"} and not sig["has_explicit_reference"])
        ):
            tags.add("event_anchor")
        if "ago" in sig["query_text"] and sig["has_explicit_reference"]:
            tags.add("reference_time")
    if query_targets.asks_for_recall_support and any(
        m in sig["normalized"] for m in _RECALL_SUPPORT_UNIT_MARKERS
    ) and (
        sig["entity_match"] or sig["query_overlap"] >= 2
        or len(sig["normalized"].split()) >= 8
        or any(m in sig["normalized"] for m in ("the one", "last time", "yesterday", "told you", "told me"))
    ):
        tags.add("support_anchor")


def infer_unit_roles(
    *,
    row: dict[str, Any],
    query_family: str,
    query_targets: QueryTargets,
    unit: EvidenceUnit,
    memory_labels_by_id: dict[str, dict[str, Any]],
) -> set[str]:
    sig = _compute_role_signals(
        row=row, query_family=query_family, query_targets=query_targets, unit=unit,
    )
    tags: set[str] = _typed_roles(
        query_family=query_family, query_targets=query_targets, unit=unit,
    )
    # Numeric operand
    if (sig["explicit_numeric"] or unit.is_numeric_operand or sig["duration_query"] and sig["duration_unit"]) and (
        sig["moderate_lexical_match"] or sig["strong_named_match"] or sig["entity_match"]
        or sig["duration_query"] and sig["duration_unit"]
        or sig["comparison_like_query"] or sig["numeric_context_match"]
        or (sig["count_query"] and sig["count_friendly_numeric"] and sig["query_overlap"] >= 1)
    ):
        tags.add("numeric_operand")
    # Event/time anchors
    if sig["has_time"] and (unit.unit_type == "dated_event" or (sig["event_like"] and sig["query_overlap"] >= 3)):
        tags.add("event_anchor")
        tags.add("time_anchor")
    if sig["has_update"] or unit.is_update_assertion:
        tags.add("update_anchor")
        tags.add("state_anchor")
    if unit.is_current_state_candidate or unit.is_state_assertion:
        tags.add("current_resolution")
    elif sig["has_current_state_language"]:
        tags.add("current_resolution")
    if query_family == "temporal" and sig["has_time"] and sig["has_explicit_reference"]:
        tags.add("reference_time")
    if query_family == "ordering" and sig["has_time"] and (
        unit.unit_type == "dated_event" or (sig["ordering_like_query"] and sig["query_overlap"] >= 3)
    ):
        tags.add("event_anchor")
    _assign_direct_anchor_tags(
        tags=tags, query_family=query_family, query_targets=query_targets, unit=unit, sig=sig,
    )
    early_return = _assign_count_query_tags(
        tags=tags, query_family=query_family, query_targets=query_targets, unit=unit, sig=sig,
    )
    if early_return:
        return tags
    _assign_family_specific_tags(
        tags=tags, query_family=query_family, query_targets=query_targets, unit=unit, sig=sig,
    )
    strong_match = sig["strong_named_match"] or sig["strong_lexical_match"]
    tags |= _comparison_roles(
        query_targets=query_targets, unit=unit, unit_profile=sig["unit_profile"],
        strong_match=strong_match, lexical_match=sig["moderate_lexical_match"],
    )
    tags |= _temporal_roles(
        query_family=query_family, query_targets=query_targets, unit=unit,
        unit_profile=sig["unit_profile"], strong_match=strong_match, query_overlap=sig["query_overlap"],
    )
    tags |= _state_roles(
        query_family=query_family, unit=unit,
        strong_match=strong_match, lexical_match=sig["moderate_lexical_match"],
    )
    return tags
