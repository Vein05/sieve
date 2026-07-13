"""Heuristic role inference for evidence units."""

from __future__ import annotations

import re
from typing import Any

from shared.nlp import cached_build_profile
from .units import EvidenceUnit
from retrieval.query_targets import QueryTargets

from shared.nlp import cached_build_profile, stem_token, cached_word_tokenize, cached_pos_tag



_COMPARISON_FALLBACK_MARKERS = (
    " compared ",
    " more ",
    " less ",
    " higher ",
    " lower ",
    " bigger ",
    " smaller ",
    " versus ",
    " vs ",
)

_ORDERING_MARKERS = (
    " first ",
    " second ",
    " third ",
    " before ",
    " after ",
    " order ",
)

_CURRENT_STATE_WORDS = (
    "currently",
    "current ",
    "right now",
    "now use",
    "now uses",
    "now using",
    "still ",
    "as of",
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
_COUNT_SCAFFOLD_TOKENS = {
    "different",
    "distinct",
    "kind",
    "kinds",
    "type",
    "types",
    "varieties",
    "variety",
}

_ENTITY_MATCH_STOPWORDS = {
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
}
_GENERIC_ORDERING_ENTITY_TOKENS = {
    "arrival",
    "event",
    "participation",
    "post",
    "task",
}

_COUNT_LOOKUP_OBJECT_SKIP_TOKENS = {
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
}

_COUNT_TARGET_NOISE_TOKENS = {
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
}


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


def infer_unit_roles(
    *,
    row: dict[str, Any],
    query_family: str,
    query_targets: QueryTargets,
    unit: EvidenceUnit,
    memory_labels_by_id: dict[str, dict[str, Any]],
) -> set[str]:
    query_profile = cached_build_profile(str(row.get("query", "")))
    source_text = str(unit.provenance.get("source_text") or unit.render_text)
    unit_profile = cached_build_profile(source_text)
    normalized = unit_profile.normalized_text
    query_text = query_targets.normalized_query
    source_numeric_values = [match.group(0) for match in _SOURCE_NUMERIC_RE.finditer(source_text)]
    query_content_tokens = set(query_profile.content_tokens)
    unit_content_tokens = set(unit_profile.content_tokens)
    padded_normalized = f" {normalized} "
    query_stems = {_stem_token(token) for token in query_content_tokens}
    unit_stems = {_stem_token(token) for token in unit_content_tokens}
    query_overlap = max(
        len(query_content_tokens & unit_content_tokens),
        len(query_stems & unit_stems),
    )
    named_overlap = bool(query_profile.named_tokens & set(unit.entity_tokens))
    explicit_numeric = any(
        value.startswith("$")
        or "%" in value
        or "." in value
        or "," in value
        or value.isdigit()
        for value in source_numeric_values
    ) or any(f" {word} " in padded_normalized for word in ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"))
    typed_numeric_operand = unit.is_numeric_operand
    has_time = unit.is_temporal_anchor or bool(unit.time_markers) or unit.unit_type == "dated_event"
    has_update = unit.is_update_assertion or bool(unit.update_markers)
    has_answerish_source = bool(_ANSWERISH_SOURCE_RE.search(source_text))
    has_explicit_reference = any(
        phrase in normalized
        for phrase in (
            "as of",
            "current as of",
            "reference date",
            "reference time",
            "reference point",
            "today is",
            "now is",
            "at this time",
        )
    )
    has_current_state_language = any(phrase in normalized for phrase in _CURRENT_STATE_WORDS)
    moderate_lexical_match = query_overlap >= 2
    recommend_lookup = "recommend" in query_text and "recommend" in normalized
    designation_lookup = any(marker in query_text for marker in ("designation", "name", "called")) and any(
        marker in source_text for marker in ('"', "'", ":")
    )
    duration_query = _duration_like_query(query_text, query_family, query_targets)
    duration_unit = _duration_like_unit(normalized)
    count_query = _count_like_query(query_text) and not duration_query
    count_target_stem_sets = _count_target_stem_sets(query_targets) if count_query else []
    comparison_like_query = query_family == "aggregation" and (
        query_targets.asks_for_comparison
        or any(marker in f" {query_text} " for marker in _COMPARISON_FALLBACK_MARKERS)
    )
    choice_like_query = " or " in f" {query_text} "
    ordering_like_query = query_family == "ordering" or any(
        marker in f" {query_text} " for marker in _ORDERING_MARKERS
    )
    primary_entities = tuple(entity for entity in query_targets.subject_entities[:2] if entity) or tuple(
        entity for entity in query_targets.candidate_entities if entity
    )
    entity_match = any(
        _entity_phrase_matches_unit(entity, unit, unit_profile)
        for entity in primary_entities
    )
    numeric_context_match = any(
        marker in padded_normalized
        for marker in (
            " amount ",
            " assist ",
            " assists ",
            " bill ",
            " bills ",
            " charge ",
            " charges ",
            " cost ",
            " costs ",
            " day ",
            " days ",
            " discount ",
            " fare ",
            " fee ",
            " fees ",
            " goal ",
            " goals ",
            " hour ",
            " hours ",
            " minute ",
            " minutes ",
            " month ",
            " months ",
            " paid ",
            " payment ",
            " payments ",
            " price ",
            " receipt ",
            " save ",
            " saved ",
            " score ",
            " scored ",
            " spend ",
            " spent ",
            " taxes ",
            " value ",
            " week ",
            " weeks ",
            " year ",
            " years ",
        )
    )
    strong_named_match = named_overlap
    strong_lexical_match = query_overlap >= 3
    count_friendly_numeric = explicit_numeric and not _DURATION_PHRASE_RE.search(source_text) and not any(
        value.startswith("$") or "%" in value
        for value in source_numeric_values
    )
    typed_count_item = unit.is_countable_item
    typed_count_evidence = (
        unit.is_answer_like_quantity_statement
        or unit.is_instructional_quantity
        or unit.has_progress_marker
        or (typed_count_item and (unit.has_acquisition_marker or unit.has_consumption_or_completion_marker))
    )
    event_count_activity = bool(
        unit.has_acquisition_marker
        or unit.has_consumption_or_completion_marker
        or typed_count_item
    )
    count_target_match = any(stem_set & unit_stems for stem_set in count_target_stem_sets)
    matched_count_target_count = sum(1 for stem_set in count_target_stem_sets if stem_set & unit_stems)
    count_activity_like = typed_count_item or explicit_numeric or (
        has_answerish_source and (unit.speaker == "user" or unit.has_pronoun_language)
    )
    event_like = (
        named_overlap
        or query_overlap >= 2
        or (unit.unit_type == "dated_event" and not has_explicit_reference and query_overlap >= 1)
    )

    tags: set[str] = _typed_roles(
        query_family=query_family,
        query_targets=query_targets,
        unit=unit,
    )
    if (explicit_numeric or typed_numeric_operand or duration_query and duration_unit) and (
        moderate_lexical_match
        or strong_named_match
        or entity_match
        or duration_query and duration_unit
        or comparison_like_query
        or numeric_context_match
        or (count_query and count_friendly_numeric and query_overlap >= 1)
    ):
        tags.add("numeric_operand")
    if has_time and (
        unit.unit_type == "dated_event"
        or (event_like and query_overlap >= 3)
    ):
        tags.add("event_anchor")
        tags.add("time_anchor")
    if has_update or unit.is_update_assertion:
        tags.add("update_anchor")
        tags.add("state_anchor")
    # Typed field is authoritative; lexical is a supplement when typed didn't fire
    if unit.is_current_state_candidate or unit.is_state_assertion:
        tags.add("current_resolution")
    elif has_current_state_language:
        tags.add("current_resolution")
    if query_family == "temporal" and has_time and has_explicit_reference:
        tags.add("reference_time")
    if query_family == "ordering" and has_time and (
        unit.unit_type == "dated_event"
        or (ordering_like_query and query_overlap >= 3)
    ):
        tags.add("event_anchor")
    if comparison_like_query or choice_like_query:
        if strong_named_match or strong_lexical_match:
            tags.add("direct_anchor")
    elif strong_named_match or strong_lexical_match:
        tags.add("direct_anchor")
    # Typed fields are sufficient authority for direct_anchor in state/update families
    if query_family in {"current_state", "knowledge_update", "conflict_update"} and (
        unit.is_current_state_candidate
        or unit.is_state_assertion
        or unit.is_update_assertion
        or (has_current_state_language and (named_overlap or query_overlap >= 1))
        or (has_update and (named_overlap or query_overlap >= 1))
    ):
        tags.add("direct_anchor")
    if query_family in {"single_anchor", "information_extraction"} and unit.is_direct_answer_candidate:
        tags.add("direct_anchor")
    if query_family in {"single_anchor", "information_extraction"} and (recommend_lookup or designation_lookup):
        if (
            unit.is_direct_answer_candidate
            or designation_lookup
            or entity_match
            or len(normalized.split()) <= 8
            or not recommend_lookup
        ):
            tags.add("direct_anchor")
    if query_family in {"single_anchor", "information_extraction"} and _CERTIFICATION_SOURCE_RE.search(source_text):
        tags.add("direct_anchor")
    if query_family in {"single_anchor", "information_extraction"} and _FAVORITE_VALUE_RE.search(source_text):
        tags.add("direct_anchor")
    if query_family in {"single_anchor", "information_extraction"} and "ethnicity" in query_text and "ethnicity" in normalized:
        tags.add("direct_anchor")
    if query_family in {"single_anchor", "information_extraction"} and "stance" in query_text and "used to be" in normalized:
        tags.add("direct_anchor")
    if duration_query and duration_unit:
        tags.add("direct_anchor")
    if query_family in {"single_anchor", "information_extraction", "aggregation"} and explicit_numeric and moderate_lexical_match:
        tags.add("direct_anchor")
    if query_targets.asks_for_recall_support and entity_match:
        tags.add("direct_anchor")
    if query_family in {"single_anchor", "information_extraction"} and query_overlap >= 1 and has_answerish_source:
        tags.add("direct_anchor")
    if query_family in {"single_anchor", "information_extraction"} and query_text.startswith("where ") and _WHERE_LOCATION_RE.search(source_text):
        tags.add("direct_anchor")
    if query_family in {"single_anchor", "information_extraction"} and query_text.startswith("where ") and _WHERE_IT_WAS_RE.search(source_text):
        tags.add("direct_anchor")
    if moderate_lexical_match or strong_named_match or unit.is_direct_answer_candidate:
        tags.add("support_anchor")
    if query_family == "aggregation" and count_query and (
        unit.is_question_or_request
        or unit.is_recommendation_or_advice
    ):
        answer_bearing_count_fact = bool(
            typed_count_evidence
            and (count_target_match or query_overlap >= 1 or entity_match)
            and explicit_numeric
        )
        if not answer_bearing_count_fact:
            tags.discard("count_item")
            tags.discard("count_evidence")
            return tags
    if query_family == "aggregation" and count_query and _GENERIC_TYPE_LIST_RE.search(source_text):
        subject_entity_match = bool(
            query_targets.subject_entities
            and any(
                stem_set & unit_stems
                for stem_set in _count_target_stem_sets(query_targets)
            )
        )
        needs_multi_anchor = len(tuple(entity for entity in query_targets.subject_entities if entity)) >= 2
        if (
            unit.speaker == "assistant"
            and not _FIRST_PERSON_RE.search(source_text)
            and (
                not subject_entity_match
                or (needs_multi_anchor and matched_count_target_count < 2)
            )
        ):
            tags.discard("count_item")
            tags.discard("count_evidence")
    if query_family == "aggregation" and count_query:
        required_subject_matches = len(tuple(entity for entity in query_targets.subject_entities if entity))
        if (
            unit.speaker == "assistant"
            and required_subject_matches >= 2
            and matched_count_target_count < 2
            and not _FIRST_PERSON_RE.search(source_text)
        ):
            tags.discard("count_item")
            tags.discard("count_evidence")
    if query_targets.asks_for_recall_support and any(
        marker in normalized
        for marker in (
            "recommend",
            "recommended",
            "mention",
            "mentioned",
            "told you",
            "told me",
            "the one",
            "last time",
            "yesterday",
        )
    ) and (
        entity_match
        or query_overlap >= 2
        or len(normalized.split()) >= 8
        or any(marker in normalized for marker in ("the one", "last time", "yesterday", "told you", "told me"))
    ):
        tags.add("support_anchor")
    if query_family == "aggregation" and count_query and (
        typed_count_evidence
        or (
            count_friendly_numeric
            and count_target_match
            and (moderate_lexical_match or entity_match or query_overlap >= 1)
        )
    ):
        tags.add("count_evidence")

    if query_family == "aggregation" and count_query:
        count_object_stems = _extract_count_object_stems(query_text)
        fallback_count_match = bool(count_object_stems and (count_object_stems & unit_stems))
        prefer_fact_statement = bool(
            (unit.is_answer_like_quantity_statement or unit.is_instructional_quantity or unit.has_progress_marker)
            and not typed_count_item
            and not unit.has_consumption_or_completion_marker
        )
        if (
            (count_target_match or fallback_count_match)
            and (count_activity_like or event_count_activity)
            and not unit.is_recommendation_or_advice
            and not prefer_fact_statement
        ):
            tags.add("count_item")
            tags.add("support_anchor")
            if typed_count_evidence or explicit_numeric:
                tags.add("count_evidence")

    if query_family == "aggregation" and count_query and typed_count_item and (
        event_like
        or entity_match
        or query_overlap >= 1
    ):
        tags.add("count_item")
        tags.add("support_anchor")
    if query_family in {"single_anchor", "information_extraction"} and query_text.startswith("where ") and _WHERE_LOCATION_RE.search(source_text):
        tags.add("support_anchor")
    if query_family in {"single_anchor", "information_extraction"} and query_text.startswith("where ") and _WHERE_IT_WAS_RE.search(source_text):
        tags.add("support_anchor")
    if query_family == "current_state" and (
        has_current_state_language
        and (named_overlap or query_overlap >= 2)
    ):
        tags.add("current_resolution")
        tags.add("state_anchor")
    if query_family == "current_state" and (strong_named_match or moderate_lexical_match):
        tags.add("state_anchor")
        tags.add("direct_anchor")
    if query_family in {"knowledge_update", "conflict_update"} and (has_update or has_explicit_reference):
        tags.add("update_anchor")
        tags.add("state_anchor")
    if query_family == "temporal":
        if has_time and (
            event_like
            or (unit.unit_type in {"dated_event", "update_fact"} and not has_explicit_reference)
        ):
            tags.add("event_anchor")
        if "ago" in query_text and has_explicit_reference:
            tags.add("reference_time")

    tags |= _comparison_roles(
        query_targets=query_targets,
        unit=unit,
        unit_profile=unit_profile,
        strong_match=strong_named_match or strong_lexical_match,
        lexical_match=moderate_lexical_match,
    )
    tags |= _temporal_roles(
        query_family=query_family,
        query_targets=query_targets,
        unit=unit,
        unit_profile=unit_profile,
        strong_match=strong_named_match or strong_lexical_match,
        query_overlap=query_overlap,
    )
    tags |= _state_roles(
        query_family=query_family,
        unit=unit,
        strong_match=strong_named_match or strong_lexical_match,
        lexical_match=moderate_lexical_match,
    )
    return tags
