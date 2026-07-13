"""Lightweight query-target extraction for evidence planning."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any

from .common import (
    cached_build_profile,
    get_stems_for_text,
    cached_word_tokenize,
    cached_pos_tag,
)



_NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
}

_CHOICE_MARKERS = (
    "between",
    "compared against",
    "compared to",
    "compared with",
    "difference between",
    "difference in",
    "either",
    "first",
    "greater",
    "higher than",
    "less",
    "less than",
    "lower than",
    "more",
    "more than",
    "smaller",
    "or",
    "second",
    "smaller than",
    "than",
    "third",
    "vs",
    "versus",
    "which one",
)

_COMPARISON_MARKERS = {
    "compared against",
    "compared to",
    "compared with",
    "difference between",
    "difference in",
    "greater",
    "higher than",
    "less",
    "less than",
    "lower than",
    "more",
    "more than",
    "smaller",
    "smaller than",
    "than",
    "vs",
    "versus",
}

_ORDERING_MARKERS = {
    "order of",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "before",
    "after",
    "earliest",
    "latest",
    "earliest to latest",
    "latest to earliest",
    "from earliest to latest",
    "from latest to earliest",
    "first to last",
    "last to first",
    "most recent",
    "ordered",
    "sequence",
}

_CURRENT_STATE_MARKERS = (
    "as of now",
    "current ",
    "current state",
    "current status",
    "currently",
    "final state",
    "final status",
    "latest",
    "now ",
    "right now",
    "still ",
    "what is the current",
    "what is the latest",
    "what is the final",
)

_TEMPORAL_DIFFERENCE_MARKERS = (
    "between",
    "difference between",
    "difference in",
)
_GENERIC_TEMPORAL_CONTEXT_ENTITIES = {
    "answer",
    "approval",
    "decision",
    "event",
    "response",
    "result",
    "time",
}
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

_ORDERING_QUERY_RE = re.compile(
    r"\b(order of|ordered|sequence|earliest to latest|latest to earliest|from earliest to latest|from latest to earliest|first to last|last to first|starting from the earliest)\b"
)
_MORE_ABOUT_RE = re.compile(
    r"\b(?:learn|know|read|hear|find out|get|discover)\s+more\s+about\b|\bmore about\b"
)
_EXPLICIT_COMPARISON_RE = re.compile(
    r"\b(compared to|compared with|compared against|versus|\bvs\b|difference between|difference in (?:price|cost|amount|number|rate|salary|fare|age|time|duration)|which cost more|which costs more|which was more expensive|more expensive than|less expensive than|higher than|lower than|greater than|smaller than|how (?:much|many) more|how (?:much|many) less)\b"
)
_TEMPORAL_UNIT_REQUEST_RE = re.compile(
    r"\bhow\s+(?:long|many\s+(?:days?|weeks?|months?|years?|hours?|minutes?))\b"
)
_TEMPORAL_CONNECTOR_RE = re.compile(
    r"\b(after|before|since|until|between|did it take|took|passed|wait|waited)\b"
)
_CALENDAR_CUE_RE = re.compile(
    r"\b(today|yesterday|tomorrow|january|february|march|april|may|june|july|august|september|october|november|december|\d{1,4}/\d{1,2}/\d{1,2}|\d{1,2}/\d{1,2})\b"
)

_RECALL_SUPPORT_MARKERS = (
    "follow up",
    "last time",
    "mentioned",
    "previous conversation",
    "previously",
    "recall",
    "trying to recall",
    "remember",
    "remind",
    "reminder",
    "say i would",
    "said i would",
    "the one",
    "what did i",
    "which one",
)

_QUOTED_PATTERNS = (
    re.compile(r'"([^"]+)"'),
    re.compile(r"'([^']+)'"),
    re.compile(r"“([^”]+)”"),
    re.compile(r"`([^`]+)`"),
)
_ATTRIBUTE_OBJECT_PATTERNS = (
    re.compile(
        r"\bwhat\s+(?:type|kind)\s+of\s+(?P<entity>.+?)\s+(?:did|do|does|have|has|had|am|is|are|was|were|can|could|should|would|will)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+(?:breed|color|model|name|speed|size|brand|route|occupation|job|role)\s+(?:is|was|are|were|do|does|did)\s+(?P<entity>.+?)(?:\?|$)",
        re.IGNORECASE,
    ),
)

_FUNCTIONAL_NOUNS = {
    "type", "name", "thing", "buy", "find", "get", "way", "one", "bit", "part",
    "amount", "number", "total", "kind", "sort", "detail", "info", "information",
    "score", "level", "value", "state", "status", "version", "result"
}

_TEMPORAL_ENTITY_RE = re.compile(
    r"\b("
    r"today|tomorrow|yesterday|tonight|morning|afternoon|evening|night|"
    r"ago|later|earlier|before|after|since|until|throughout|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"\d{1,2}:\d{2}(?:\s*(?:am|pm))?|"
    r"\d{4}(?:[/-]\d{1,2}){1,2}"
    r")\b",
    re.IGNORECASE,
)

# Matches a trailing temporal anchor phrase at the end of an entity string.
# Example: "jewelry last saturday" → strips " last saturday"
# Example: "lunch last tuesday"  → strips " last tuesday"
# Example: "pair of shoes last month" → strips " last month"
# Only strips when *something* meaningful precedes the anchor.
_TRAILING_TEMPORAL_ANCHOR_RE = re.compile(
    r"(?:\s+(?:last|this|next)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month|year|night|morning|afternoon|evening))$"
    r"|(?:\s+(?:on\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))$"
    r"|(?:\s+(?:yesterday|today|tonight|tomorrow))$"
    r"|(?:\s+\d{4}/\d{1,2}/\d{1,2})$",
    re.IGNORECASE,
)


def _strip_trailing_temporal_anchor(entity: str) -> str:
    """Remove a trailing temporal phrase from an entity string.

    Only strips when the remainder is a non-empty noun-like string so
    we never silently reduce an entity to nothing or to a bare stop word.
    """
    stripped = _TRAILING_TEMPORAL_ANCHOR_RE.sub("", entity).strip()
    if not stripped or stripped == entity:
        return entity
    if len(stripped) < 3:
        return entity
    # Guard: a single-token remainder that is a function/stop word is not useful.
    tokens = stripped.split()
    _STOP_WORDS = {
        "a", "an", "the", "my", "our", "your", "their", "its",
        "last", "this", "next", "some", "any", "all", "both",
        "i", "me", "we", "us", "he", "she", "it", "they",
        "is", "was", "are", "were", "be", "been",
    }
    if len(tokens) == 1 and tokens[0].lower() in _STOP_WORDS:
        return entity
    return stripped


def _identify_focus_noun(normalized_query: str) -> str | None:
    """Identifies the 'what' of the question (e.g. 'degree' in 'What degree...')"""
    match = re.search(r"^(?:what|which|whose|how)\s+([a-z]+)\b", normalized_query)
    if match:
        return match.group(1)
    return None

def _is_temporal_entity(normalized: str) -> bool:
    if not normalized:
        return False
    if _TEMPORAL_ENTITY_RE.search(normalized):
        return True
    tokens = set(normalized.split())
    return bool(tokens & _TEMPORAL_UNIT_TOKENS)


def _calculate_specificity_score(
    normalized: str,
    metadata: dict[str, Any],
    focus_noun: str | None = None,
    query_family: str | None = None,
) -> float:
    score = 1.0
    
    # Focus noun penalty (don't ground on the thing we are looking for!)
    if focus_noun and normalized == focus_noun:
        score -= 1.0
    
    # Word count boost (multi-word phrases are more specific)
    word_count = metadata.get("word_count", 1)
    if word_count > 1:
        score += 0.5 * (word_count - 1)
    
    # Proper noun boost
    if metadata.get("is_named"):
        score += 1.0

    # Temporal anchors matter more for temporal/ordering queries.
    if _is_temporal_entity(normalized):
        score += 0.6
        if query_family in {"temporal", "ordering"}:
            score += 0.9
        elif query_family in {"current_state", "knowledge_update", "conflict_update"}:
            score += 0.3
        if metadata.get("source") in {"temporal", "pattern"}:
            score += 0.2
        
    # Function word penalty
    tokens = normalized.split()
    for t in tokens:
        if t in _FUNCTIONAL_NOUNS:
            score -= 0.8
            
    # Length penalty for very short single words (already mostly filtered but just in case)
    if len(normalized) <= 3 and word_count == 1:
        score -= 0.5
        
    return score

_BOUNDARY_TOKENS = {
    "a",
    "an",
    "and",
    "did",
    "do",
    "does",
    "for",
    "how",
    "i",
    "is",
    "my",
    "of",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
}

_TEMPORAL_UNIT_TOKENS = {
    "ago",
    "day",
    "days",
    "hour",
    "hours",
    "minute",
    "minutes",
    "month",
    "months",
    "week",
    "weeks",
    "year",
    "years",
    "parentheses",
}

_GROUNDING_NOISE_ENTITIES = {
    "amount",
    "number",
    "times",
    "total",
    "many",
    "much",
    "which",
    "what",
    "how",
    "i",
    "me",
    "my",
    "time",
    "session",
    "fact",
    "detail",
    "details",
    "meals",
    "meal",
    "lunch",
    "dinner",
    "breakfast",
    "snack",
    "app",
    "item",
    "items",
    "coupon",
    "coupons",
    "price",
    "cost",
}

_COUNT_QUERY_PREFIX_RE = re.compile(r"\b(?:how\s+many|number\s+of|count\s+of|amount\s+of)\b", re.IGNORECASE)
_COUNT_ENTITY_STOP_RE = re.compile(
    r"\b(?:do|does|did|have|has|had|am|is|are|was|were|can|could|should|would|will|i|we|they|he|she|you)\b",
    re.IGNORECASE,
)
_COUNT_ENTITY_LEADING_NOISE_RE = re.compile(
    r"^(?:how\s+many|number\s+of|count\s+of|amount\s+of)\s+|"
    r"^(?:all|both|either)\s+(?:of\s+)?(?:my|our|the)\s+|"
    r"^(?:different|distinct|combined|total|overall|altogether|various|separate)\s+|"
    r"^(?:different|distinct)\s+(?:types?|kinds?|sorts?|varieties?|species|pieces?|pairs?|brands?)\s+of\s+|"
    r"^(?:types?|kinds?|sorts?|varieties?|species|pieces?|pairs?|brands?)\s+of\s+",
    re.IGNORECASE,
)
_COUNT_ENTITY_TRAILING_NOISE_RE = re.compile(
    r"\b(?:recently|lately|currently|now|simultaneously|together|combined|altogether)\b",
    re.IGNORECASE,
)
_COUNT_CLAUSE_ENTITY_RE = re.compile(
    r"^(?:how\s+many|number\s+of|count\s+of|amount\s+of)\b.*\b(?:do|does|did|have|has|had|am|is|are|was|were)\b",
    re.IGNORECASE,
)




@dataclass(frozen=True)
class QueryTargets:
    raw_query: str
    normalized_query: str
    query_family: str
    schema_name: str
    candidate_entities: tuple[str, ...]
    subject_entities: tuple[str, ...]
    context_entities: tuple[str, ...]
    alternative_entities: tuple[str, ...]
    candidate_numbers: tuple[str, ...]
    preferred_attributes: tuple[str, ...]
    choice_markers: tuple[str, ...]
    asks_for_comparison: bool
    asks_for_ordering: bool
    asks_for_current_state: bool
    asks_for_temporal_difference: bool
    asks_for_relative_time: bool
    asks_for_recall_support: bool
    required_slots: tuple[str, ...]
    answerability_hints: dict[str, Any]
    planning_payload: dict[str, Any]
    entities_metadata: dict[str, dict[str, Any]] = None


def _dedupe_preserve_order(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _find_markers(text: str, markers: tuple[str, ...]) -> tuple[str, ...]:
    hits: list[tuple[int, str]] = []
    padded = f" {text} "
    for marker in markers:
        pattern = re.escape(marker)
        if " " not in marker:
            pattern = rf"\b{pattern}\b"
        for match in re.finditer(pattern, padded):
            hits.append((match.start(), marker))
    hits.sort(key=lambda item: (item[0], item[1]))
    return _dedupe_preserve_order([marker for _, marker in hits])


def _tag_label(tag: tuple[str, str] | str) -> str:
    if isinstance(tag, tuple):
        return str(tag[1])
    return str(tag)


def _clean_entity_phrase(tokens: list[str], tags: list[tuple[str, str] | str]) -> str:
    """Skip leading/trailing function words while preserving modifier+noun spans."""
    if not tokens:
        return ""

    skip_tags = {"DT", "IN", "PRP$", "CC", "PRP"}

    start = 0
    while start < len(tags) and _tag_label(tags[start]) in skip_tags:
        start += 1

    end = len(tags)
    while end > start and _tag_label(tags[end - 1]) in skip_tags:
        end -= 1

    cleaned = [token for token in tokens[start:end] if token]
    if not cleaned:
        return ""

    cleaned_tags = [_tag_label(tag) for tag in tags[start:end]]
    has_noun = any(tag.startswith("NN") for tag in cleaned_tags)
    if not has_noun:
        return ""
    return " ".join(cleaned).strip()


def _extract_quoted_phrases(raw_query: str) -> list[str]:
    phrases: list[str] = []
    for pattern in _QUOTED_PATTERNS:
        for match in pattern.finditer(raw_query):
            quoted_text = match.group(1).lower()
            q_tokens = cached_word_tokenize(quoted_text)
            q_tags = cached_pos_tag(tuple(q_tokens))
            phrase = _clean_entity_phrase(q_tokens, q_tags)
            if phrase:
                phrases.append(phrase)
    return phrases


def _extract_numeric_tokens(raw_query: str, normalized_query: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", raw_query):
        values.append(match.group(0))
    for token in normalized_query.split():
        if token in _NUMBER_WORDS:
            values.append(token)
    return values


def _normalize_entity_phrase(text: str) -> str:
    normalized = cached_build_profile(str(text or "")).normalized_text.strip(" ,.?!:;\"'")
    normalized = re.sub(r"\b(?:currently|current|right now|now|final|latest)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    tokens = [token for token in cached_word_tokenize(normalized) if token]
    if not tokens:
        return ""
    start = 0
    while start < len(tokens) and tokens[start] in {"a", "an", "the", "my", "our", "your", "their"}:
        start += 1
    end = len(tokens)
    while end > start and tokens[end - 1] in {"a", "an", "the", "my", "our", "your", "their"}:
        end -= 1
    return " ".join(tokens[start:end]).strip()


def _extract_leading_noun_phrase(text: str) -> str:
    normalized = _normalize_entity_phrase(text)
    if not normalized:
        return ""
    tokens = cached_word_tokenize(normalized)
    tags = cached_pos_tag(tuple(tokens))
    phrase_tokens: list[str] = []
    for word, tag in tags:
        if tag.startswith(("NN", "JJ")) or tag in {"CD", "POS"} or word == "of":
            phrase_tokens.append(word)
            continue
        if phrase_tokens:
            break
    return _normalize_entity_phrase(" ".join(phrase_tokens))


def _extract_trailing_noun_phrase(text: str) -> str:
    normalized = _normalize_entity_phrase(text)
    if not normalized:
        return ""
    tokens = cached_word_tokenize(normalized)
    tags = cached_pos_tag(tuple(tokens))
    phrase_tokens: list[str] = []
    for word, tag in reversed(tags):
        if tag.startswith(("NN", "JJ")) or tag in {"CD", "POS"} or word == "of":
            phrase_tokens.append(word)
            continue
        if phrase_tokens:
            break
    phrase_tokens.reverse()
    return _normalize_entity_phrase(" ".join(phrase_tokens))


def _extract_event_phrase(text: str) -> str:
    normalized = _normalize_entity_phrase(text)
    if not normalized:
        return ""
    tokens = [
        token
        for token in cached_word_tokenize(normalized)
        if token not in {"did", "it", "take", "for", "i", "have", "has", "had", "been", "do", "does", "was", "were", "the"}
    ]
    cleaned = [token for token in tokens if token != "to"]
    return _normalize_entity_phrase(" ".join(cleaned))


def _extract_choice_side_phrase(text: str, focus_noun: str | None = None) -> str:
    raw = str(text or "").strip(" ,?:;")
    if not raw:
        return ""
    for delimiter in (",", ";", ":"):
        if delimiter in raw:
            raw = raw.rsplit(delimiter, 1)[-1].strip()
    if focus_noun:
        raw = re.sub(r"^(?:the\s+)?one\b", focus_noun, raw, count=1)
    normalized = _normalize_entity_phrase(raw)
    if not normalized:
        return ""
    tokens = [token for token in cached_word_tokenize(normalized) if token]
    if not tokens:
        return ""
    if tokens[0] in {"which", "what", "who", "when", "where", "why", "did", "do", "does", "is", "are", "was", "were"}:
        return ""
    return normalized


def _extract_relative_time_entities(normalized_query: str) -> list[str]:
    entities: list[str] = []
    patterns = (
        r"\bhow\s+many\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\s+ago\s+did\s+i\s+(?P<event>.+?)\s+(?:when|after|before|while|during)\s+i\s+(?P<reference>.+?)(?:\?|$)",
        r"\bhow\s+many\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\s+ago\s+did\s+i\s+(?P<event>.+?)(?:\?|$)",
        r"\bhow\s+many\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\s+have\s+passed\s+since\s+i\s+(?P<event>.+?)(?:\?|$)",
        r"\bhow\s+long\s+has\s+it\s+been\s+since\s+i\s+(?P<event>.+?)(?:\?|$)",
        r"\bhow\s+many\s+(?:days?|weeks?|months?|years?)\s+(?:older|younger)\s+am\s+i\s+than\s+when\s+i\s+(?P<event>.+?)(?:\?|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized_query)
        if not match:
            continue
        event = _extract_event_phrase(str(match.groupdict().get("event") or ""))
        if event:
            entities.append(event)
        reference = _extract_event_phrase(str(match.groupdict().get("reference") or ""))
        if reference:
            entities.append(reference)
        if entities:
            return entities
    return entities


def _filter_temporal_entities(candidate_entities: tuple[str, ...]) -> tuple[str, ...]:
    cleaned: list[str] = []
    for entity in candidate_entities:
        tokens = [
            token
            for token in cached_word_tokenize(str(entity or ""))
            if token and token not in _TEMPORAL_UNIT_TOKENS
        ]
        value = _normalize_entity_phrase(" ".join(tokens))
        if not value:
            continue
        cleaned.append(value)
    return _dedupe_preserve_order(cleaned)


def _prune_relative_time_entities(
    candidate_entities: tuple[str, ...],
    relative_entities: tuple[str, ...],
) -> tuple[str, ...]:
    if not relative_entities:
        return candidate_entities
    canonical = set(relative_entities)
    relative_stem_sets = [set(get_stems_for_text(entity)) for entity in relative_entities if entity]
    cleaned: list[str] = []
    for entity in candidate_entities:
        if entity in canonical:
            cleaned.append(entity)
            continue
        normalized = str(entity or "").strip()
        padded = f" {normalized} "
        if any(marker in padded for marker in (" when ", " after ", " before ", " while ", " during ")):
            entity_stems = set(get_stems_for_text(normalized))
            if relative_stem_sets and all(stems <= entity_stems for stems in relative_stem_sets if stems):
                continue
        cleaned.append(entity)
    return _dedupe_preserve_order(cleaned)


def _extract_choice_entities(normalized_query: str) -> list[str]:
    entities: list[str] = []
    normalized = str(normalized_query or "").strip()
    focus_noun = _identify_focus_noun(normalized)

    or_match = re.search(r"(?:,\s*)?(?P<left>.+?)\s+or\s+(?P<right>.+?)(?:\?|$)", normalized)
    if or_match:
        left_segment = _extract_choice_side_phrase(or_match.group("left"), focus_noun)
        right_segment = _extract_choice_side_phrase(or_match.group("right"), focus_noun)
        if left_segment:
            entities.append(left_segment)
        if right_segment:
            entities.append(right_segment)
        left = _extract_trailing_noun_phrase(or_match.group("left"))
        right = _extract_leading_noun_phrase(or_match.group("right"))
        if left:
            entities.append(left)
        if right:
            entities.append(right)

    compared_match = re.search(r"(?P<left>.+?)\s+compared to\s+(?P<right>.+?)(?:\?|$)", normalized)
    if compared_match:
        left = _extract_trailing_noun_phrase(compared_match.group("left"))
        right = _extract_leading_noun_phrase(compared_match.group("right"))
        if left:
            entities.append(left)
        if right:
            entities.append(right)

    versus_match = re.search(r"(?P<left>.+?)\s+(?:versus|vs)\s+(?P<right>.+?)(?:\?|$)", normalized)
    if versus_match:
        left = _extract_trailing_noun_phrase(versus_match.group("left"))
        right = _extract_leading_noun_phrase(versus_match.group("right"))
        if left:
            entities.append(left)
        if right:
            entities.append(right)

    between_match = re.search(r"\bbetween\s+(?P<left>.+?)\s+and\s+(?P<right>.+?)(?:\?|$)", normalized)
    if between_match:
        left = _normalize_entity_phrase(between_match.group("left"))
        right = _normalize_entity_phrase(between_match.group("right"))
        if left:
            entities.append(left)
        if right:
            entities.append(right)

    than_match = re.search(
        r"(?P<left>.+?)\s+(?:more|less|higher|lower|bigger|smaller|cheaper|pricier)\s+.+?\s+than\s+(?P<right>.+?)(?:\?|$)",
        normalized,
    )
    if than_match:
        left = _extract_trailing_noun_phrase(than_match.group("left"))
        right = _extract_leading_noun_phrase(than_match.group("right"))
        if left:
            entities.append(left)
        if right:
            entities.append(right)

    return entities


def _shared_possessive_prefix(phrase: str) -> str:
    tokens = [token for token in cached_word_tokenize(str(phrase or "")) if token]
    if len(tokens) >= 2 and tokens[1] == "s":
        return " ".join(tokens[:2])
    return ""


def _apply_shared_prefix(prefix: str, phrase: str) -> str:
    normalized_phrase = _normalize_entity_phrase(phrase)
    if not prefix or not normalized_phrase:
        return normalized_phrase
    if normalized_phrase.startswith(prefix + " "):
        return normalized_phrase
    return _normalize_entity_phrase(f"{prefix} {normalized_phrase}")


def _extract_coordinated_entities(normalized_query: str) -> list[str]:
    entities: list[str] = []
    normalized = str(normalized_query or "").strip()
    patterns = (
        r"\b(?:of|on|for)\s+(?P<left>.+?)\s+and\s+(?P<right>.+?)(?:\?|$)",
        r"\b(?:between|compare|compared to|compared with)\s+(?P<left>.+?)\s+and\s+(?P<right>.+?)(?:\?|$)",
        r"\b(?P<left>.+?)\s+and\s+(?P<right>.+?)(?:\?|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        left_raw = str(match.group("left") or "").strip()
        right_raw = str(match.group("right") or "").strip()
        if not left_raw or not right_raw:
            continue
        left = _extract_trailing_noun_phrase(left_raw) or _normalize_entity_phrase(left_raw)
        right = _extract_leading_noun_phrase(right_raw) or _normalize_entity_phrase(right_raw)
        if not left or not right:
            continue
        shared_prefix = _shared_possessive_prefix(left)
        if shared_prefix:
            right = _apply_shared_prefix(shared_prefix, right)
        entities.append(left)
        entities.append(right)
        return entities
    return entities


def _extract_pattern_entities(normalized_query: str) -> list[str]:
    entities: list[str] = []
    normalized = str(normalized_query or "").strip()

    state_match = re.search(r"\b(?:current state|current status|final status|latest status) of (?P<entity>.+?)(?:\?|$)", normalized)
    if state_match:
        entity = _extract_leading_noun_phrase(state_match.group("entity"))
        if entity:
            entities.append(entity)

    recent_match = re.search(r"\bmost recent (?P<entity>.+?)(?:\?|$)", normalized)
    if recent_match:
        entity = _extract_trailing_noun_phrase(recent_match.group("entity"))
        if entity:
            entities.append(entity)

    current_use_match = re.search(r"\bwhat (?P<entity>.+?) do i (?:currently |now )?use\b", normalized)
    if current_use_match:
        entity = _normalize_entity_phrase(current_use_match.group("entity"))
        if entity:
            entities.append(entity)
            if " of " in entity:
                tail = _extract_leading_noun_phrase(entity.split(" of ", 1)[1])
                if tail:
                    entities.append(tail)

    current_have_match = re.search(r"\bhow many (?P<entity>.+?) do i (?:currently |now )?have\b", normalized)
    if current_have_match:
        entity = _extract_leading_noun_phrase(current_have_match.group("entity"))
        if entity:
            entities.append(entity)

    collection_match = re.search(r"\bcollecting (?P<entity>.+?)(?:\?|$)", normalized)
    if collection_match:
        entity = _normalize_entity_phrase(collection_match.group("entity"))
        if entity:
            entities.append(entity)

    relative_event_match = re.search(r"\bago did i (?P<event>.+?)(?:\?|$)", normalized)
    if relative_event_match:
        event = _extract_event_phrase(relative_event_match.group("event"))
        if event:
            entities.append(event)

    duration_after_match = re.search(r"\bdid it take for (?P<event_a>.+?) after (?P<event_b>.+?)(?:\?|$)", normalized)
    if duration_after_match:
        event_a = _extract_event_phrase(duration_after_match.group("event_a"))
        event_b = _extract_event_phrase(duration_after_match.group("event_b"))
        if event_a:
            entities.append(event_a)
        if event_b:
            entities.append(event_b)

    single_duration_match = re.search(r"\bdid it take(?: me)? to (?P<event>.+?)(?:\?|$)", normalized)
    if single_duration_match:
        event = _extract_event_phrase(single_duration_match.group("event"))
        if event:
            entities.append(event)

    wait_match = re.search(r"\bwait for (?P<entity>.+?)(?:\?|$)", normalized)
    if wait_match:
        entity = _extract_leading_noun_phrase(wait_match.group("entity"))
        if entity:
            entities.append(entity)

    return entities


def _extract_attribute_object_entities(normalized_query: str) -> list[str]:
    entities: list[str] = []
    normalized = str(normalized_query or "").strip()
    for pattern in _ATTRIBUTE_OBJECT_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        entity = _normalize_entity_phrase(str(match.group("entity") or ""))
        if entity:
            entities.append(entity)
            if " of " in entity:
                tail = _extract_leading_noun_phrase(entity.split(" of ", 1)[1])
                if tail:
                    entities.append(tail)
    return list(_dedupe_preserve_order(entities))


def _count_like_query(normalized_query: str) -> bool:
    return bool(_COUNT_QUERY_PREFIX_RE.search(str(normalized_query or "")))


def _distinct_count_like_query(normalized_query: str) -> bool:
    normalized = f" {str(normalized_query or '')} "
    if not _count_like_query(normalized):
        return False
    return bool(
        any(marker in normalized for marker in (" different ", " distinct ", " types of ", " kinds of ", " varieties of ", " species of "))
        or " or " in normalized
    )


def _clean_count_entity_phrase(text: str) -> str:
    normalized = _normalize_entity_phrase(text)
    if not normalized:
        return ""
    previous = None
    while normalized and normalized != previous:
        previous = normalized
        normalized = _COUNT_ENTITY_LEADING_NOISE_RE.sub("", normalized).strip()
        normalized = _COUNT_ENTITY_TRAILING_NOISE_RE.sub(" ", normalized)
        normalized = _normalize_entity_phrase(normalized)
    return normalized


def _extract_count_query_entities(normalized_query: str) -> list[str]:
    normalized = str(normalized_query or "").strip()
    if not _count_like_query(normalized):
        return []

    spans: list[str] = []
    how_many_match = re.search(
        r"\bhow\s+many\s+(?P<entity>.+?)\s+(?:do|does|did|have|has|had|am|is|are|was|were|can|could|should|would|will)\b",
        normalized,
    )
    if how_many_match:
        spans.append(str(how_many_match.group("entity") or ""))

    for pattern in (
        r"\b(?:number|count|amount)\s+of\s+(?P<entity>.+?)\s+(?:i|we|they|he|she|you|do|does|did|have|has|had)\b",
        r"\b(?:number|count|amount)\s+of\s+(?P<entity>.+?)(?:\?|$)",
    ):
        match = re.search(pattern, normalized)
        if match:
            spans.append(str(match.group("entity") or ""))

    entities: list[str] = []
    for span in spans:
        split_parts: list[str] = []
        for part in re.split(r"\s+(?:or|and)\s+", span):
            part = _clean_count_entity_phrase(part)
            if part:
                split_parts.append(part)
        entities.extend(split_parts)
        cleaned = _clean_count_entity_phrase(span)
        if cleaned and " or " not in cleaned and " and " not in cleaned:
            entities.append(cleaned)
    return list(_dedupe_preserve_order(entities))


def _is_count_scaffold_entity(entity: str, count_entities: tuple[str, ...]) -> bool:
    normalized = _normalize_entity_phrase(entity)
    if not normalized or normalized in count_entities:
        return False
    if re.match(r"^(?:different|distinct|many|number|count|amount)\b", normalized):
        return True
    if re.search(r"\b(?:did|do|does|have|has|had|am|is|are|was|were)\s+i\b", normalized):
        return True
    cleaned = _clean_count_entity_phrase(normalized)
    if cleaned in count_entities and cleaned != normalized:
        return True
    if " or " in normalized or " and " in normalized:
        parts = [
            _clean_count_entity_phrase(part)
            for part in re.split(r"\s+(?:or|and)\s+", normalized)
        ]
        parts = [part for part in parts if part]
        if parts and all(part in count_entities for part in parts):
            return True
    entity_stems = {
        stem
        for stem in get_stems_for_text(normalized)
        if stem and stem not in _GROUNDING_NOISE_ENTITIES and stem not in _TEMPORAL_UNIT_TOKENS
    }
    count_stems = {
        stem
        for target in count_entities
        for stem in get_stems_for_text(target)
        if stem and stem not in _GROUNDING_NOISE_ENTITIES and stem not in _TEMPORAL_UNIT_TOKENS
    }
    if entity_stems and count_stems and entity_stems <= count_stems:
        return True
    return False


def _prune_count_clause_entities(
    candidate_entities: tuple[str, ...],
    *,
    normalized_query: str,
) -> tuple[str, ...]:
    if not _count_like_query(normalized_query):
        return candidate_entities
    cleaned: list[str] = []
    stem_sets = {
        entity: {stem for stem in get_stems_for_text(entity) if stem}
        for entity in candidate_entities
    }
    for entity in candidate_entities:
        normalized = _normalize_entity_phrase(entity)
        if not normalized:
            continue
        if _COUNT_CLAUSE_ENTITY_RE.search(normalized):
            entity_stems = stem_sets.get(entity, set())
            has_shorter_object = any(
                other != entity
                and len(other.split()) < len(normalized.split())
                and stem_sets.get(other, set())
                and stem_sets[other] < entity_stems
                for other in candidate_entities
            )
            if has_shorter_object:
                continue
        cleaned.append(entity)
    return _dedupe_preserve_order(cleaned)


def _prune_count_scaffold_alternatives(
    entities: tuple[str, ...],
    *,
    count_entities: tuple[str, ...],
) -> tuple[str, ...]:
    if not entities or not count_entities:
        return entities
    cleaned: list[str] = []
    for entity in entities:
        normalized = _normalize_entity_phrase(entity)
        if not normalized:
            continue
        if _is_count_scaffold_entity(normalized, count_entities):
            continue
        tokens = [token for token in normalized.split() if token and token not in _COUNT_SCAFFOLD_TOKENS]
        if not tokens:
            continue
        cleaned.append(" ".join(tokens))
    return _dedupe_preserve_order(cleaned)


def _prune_generic_temporal_context_entities(
    candidate_entities: tuple[str, ...],
    *,
    subject_entities: tuple[str, ...],
    context_entities: tuple[str, ...],
    normalized_query: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not subject_entities or not context_entities:
        return candidate_entities, context_entities
    if not any(marker in normalized_query for marker in ("wait for", "waiting for", "decision on", "decision for")):
        return candidate_entities, context_entities
    filtered_context = tuple(
        entity
        for entity in context_entities
        if _normalize_entity_phrase(entity) not in _GENERIC_TEMPORAL_CONTEXT_ENTITIES
    )
    if filtered_context == context_entities:
        return candidate_entities, context_entities
    filtered_candidates = tuple(
        entity
        for entity in candidate_entities
        if entity in subject_entities or entity in filtered_context
    )
    return _dedupe_preserve_order(filtered_candidates), _dedupe_preserve_order(filtered_context)


def _extract_marker_entity_phrases(tags: list[tuple[str, str]] | str) -> list[str]:
    if isinstance(tags, str):
        tokens = cached_word_tokenize(tags.lower())
        tags = list(cached_pos_tag(tuple(tokens)))
    phrases: list[str] = []

    for i, (word, tag) in enumerate(tags):
        if word in ("what", "which") or (word == "many" and i > 0 and tags[i-1][0] == "how"):
            j = i + 1
            temp_tokens: list[str] = []
            temp_tags: list[tuple[str, str]] = []
            while j < len(tags):
                w, t = tags[j]
                if w in ("do", "does", "did", "am", "is", "are", "have", "has", "had", "was", "were", "should", "can"):
                    break
                if t.startswith(("NN", "JJ")) or t == "CD" or t == "POS" or w == "of":
                    temp_tokens.append(w)
                    temp_tags.append((w, t))
                j += 1
            if temp_tokens:
                phrases.append(_clean_entity_phrase(temp_tokens, temp_tags))

    all_phrases = [p for p in phrases if p and p not in {"amount", "number", "times"}]
    return list(_dedupe_preserve_order(all_phrases))


def _extract_noun_phrases_from_tags(query_tags: list[tuple[str, str]]) -> list[str]:
    """Scans the entire query for contiguous Adjective/Noun phrases as grounding anchors."""
    entities: list[str] = []
    current_tokens: list[str] = []
    current_tags: list[tuple[str, str] | str] = []

    for word, tag in query_tags:
        # Only accept nouns that aren't functional or temporal unit tokens
        if tag.startswith("NN") or tag == "CD" or (tag.startswith("JJ") and len(word) > 3):
            lowered = word.lower()
            if lowered in _GROUNDING_NOISE_ENTITIES or lowered in _TEMPORAL_UNIT_TOKENS:
                # Flush existing phrase if we hit noise
                if current_tokens:
                    phrase = _clean_entity_phrase(current_tokens, current_tags)
                    if phrase: entities.append(phrase)
                    current_tokens, current_tags = [], []
                continue
            current_tokens.append(word)
            current_tags.append(tag)
        else:
            if current_tokens:
                phrase = _clean_entity_phrase(current_tokens, current_tags)
                if phrase:
                    entities.append(phrase)
                current_tokens = []
                current_tags = []

    if current_tokens:
        phrase = _clean_entity_phrase(current_tokens, current_tags)
        if phrase and phrase.lower() not in _GROUNDING_NOISE_ENTITIES:
            entities.append(phrase)

    return entities


def _extract_candidate_entities(
    raw_query: str,
    normalized_query: str,
    query_tags: list[tuple[str, str]],
    named_tokens: set[str],
    *,
    query_family: str,
) -> tuple[tuple[str, ...], dict[str, dict[str, Any]]]:
    entities_with_meta: list[tuple[str, str]] = []
    
    # 1. Quoted Phrases
    for q in _extract_quoted_phrases(raw_query):
        entities_with_meta.append((q, "quoted"))
    
    # 2. Relative Time
    for t in _extract_relative_time_entities(normalized_query):
        entities_with_meta.append((t, "temporal"))
        
    # 3. Choice Entities
    for c in _extract_choice_entities(normalized_query):
        entities_with_meta.append((c, "choice"))

    # 3b. Coordinated object entities for aggregate / multi-item questions
    for c in _extract_coordinated_entities(normalized_query):
        entities_with_meta.append((c, "coordinated"))
        
    # 4. Pattern Entities
    for p in _extract_pattern_entities(normalized_query):
        entities_with_meta.append((p, "pattern"))

    # 4b. Attribute-object queries ("what breed is my dog", "what type of cocktail...")
    for p in _extract_attribute_object_entities(normalized_query):
        entities_with_meta.append((p, "pattern"))
        
    # 5. Marker Phrases
    for m in _extract_marker_entity_phrases(query_tags):
        entities_with_meta.append((m, "marker"))
        
    # 6. Noun Phrases (The core extractor)
    for n in _extract_noun_phrases_from_tags(query_tags):
        entities_with_meta.append((n, "noun_phrase"))
        
    # 7. Named Tokens (Proper Nouns)
    for nt in sorted(named_tokens):
        entities_with_meta.append((nt, "named_token"))

    # Cleanup and Metadata Assembly
    metadata: dict[str, dict[str, Any]] = {}
    unique_entities = []
    seen = set()
    
    for i, (entity, source) in enumerate(entities_with_meta):
        # Normalization for matching
        value = _normalize_entity_phrase(str(entity).strip())
        if not value: continue
        profile = cached_build_profile(value)
        normalized = profile.normalized_text.strip()
        
        if not normalized or normalized.startswith("s ") or len(normalized) <= 2 or normalized in _BOUNDARY_TOKENS:
            continue
            
        if normalized not in seen:
            seen.add(normalized)
            unique_entities.append(normalized)
            
            # Build metadata for this unique entity
            is_proper = any(t[1] == "NNP" for t in profile.pos_tags) if hasattr(profile, "pos_tags") else False
            word_count = len(normalized.split())
            
            metadata[normalized] = {
                "source": source,
                "length": len(normalized),
                "word_count": word_count,
                "is_named": normalized in named_tokens or is_proper,
                "initial_rank": len(unique_entities) - 1,
            }

    focus_noun = _identify_focus_noun(normalized_query)
    entities_with_scores: list[tuple[str, float]] = []
    
    for ent in unique_entities:
        m = metadata[ent]
        # Specificity ranking
        specificity = _calculate_specificity_score(ent, m, focus_noun, query_family)
        
        # Source-based priority
        source_priority = {
            "quoted": 2.0,
            "marker": 1.5,
            "pattern": 1.2,
            "noun_phrase": 1.0,
            "choice": 1.0,
            "coordinated": 1.15,
            "temporal": 1.0,
            "named_token": 1.1,
        }.get(m["source"], 1.0)
        
        final_score = specificity * source_priority
        metadata[ent]["specificity_score"] = final_score
        entities_with_scores.append((ent, final_score))

    # Re-rank based on final score
    ranked_entities = sorted(entities_with_scores, key=lambda x: x[1], reverse=True)
    final_ordered = [e[0] for e in ranked_entities]
    
    # Update metadata with final ranks
    for i, ent in enumerate(final_ordered):
        metadata[ent]["final_rank"] = i

    # Split signal into subjects vs context based on specificity score
    subject_ents = [e[0] for e in ranked_entities if e[1] >= 1.5]
    context_ents = [e[0] for e in ranked_entities if e[1] < 1.5]

    return tuple(final_ordered), tuple(subject_ents), tuple(context_ents), metadata


def _has_phrase(normalized_query: str, phrases: tuple[str, ...]) -> bool:
    padded = f" {normalized_query} "
    for phrase in phrases:
        if not phrase:
            continue
        if " " in phrase:
            if phrase in padded:
                return True
        elif re.search(rf"\b{re.escape(phrase)}\b", normalized_query):
            return True
    return False


def _is_explicit_ordering_query(normalized_query: str) -> bool:
    return bool(_ORDERING_QUERY_RE.search(normalized_query))


def _is_explicit_comparison_query(normalized_query: str) -> bool:
    if _MORE_ABOUT_RE.search(normalized_query):
        return False
    if _EXPLICIT_COMPARISON_RE.search(normalized_query):
        return True
    if re.search(r"\b(?:more|less|higher|lower)\b.*\bthan\b", normalized_query):
        return True
    if re.search(r"\b(?:cost|costs|price|priced|spent|spend|paid|pay|earned|earn|save|saved|fare|salary|amount|rate)\b.*\b(?:more|less|higher|lower)\b", normalized_query):
        return True
    return False


def _is_explicit_temporal_difference_query(
    normalized_query: str,
    *,
    has_from_to: bool,
    has_two_anchors: bool,
) -> bool:
    has_unit_request = bool(_TEMPORAL_UNIT_REQUEST_RE.search(normalized_query))
    has_temporal_connector = bool(_TEMPORAL_CONNECTOR_RE.search(normalized_query))
    has_calendar_cue = bool(_CALENDAR_CUE_RE.search(normalized_query))
    has_pair_marker = bool(re.search(r"\b(after|before|between|since|until)\b", normalized_query))
    has_wait_like_marker = bool(re.search(r"\b(wait|waited|took|passed)\b", normalized_query))
    if has_unit_request and has_pair_marker and has_temporal_connector:
        return True
    if has_unit_request and has_wait_like_marker:
        return has_two_anchors or has_from_to or has_pair_marker
    if has_unit_request and has_temporal_connector:
        if has_wait_like_marker and not (has_two_anchors or has_from_to or has_pair_marker):
            return False
        return True
    if has_from_to and (has_unit_request or has_calendar_cue or has_two_anchors):
        return True
    if has_calendar_cue and re.search(r"\b(before|after|since|until|between)\b", normalized_query):
        return True
    if "difference between" in normalized_query and has_unit_request:
        return True
    return False


def _infer_schema_name(
    *,
    family: str,
    asks_for_comparison: bool,
    asks_for_ordering: bool,
    asks_for_temporal_difference: bool,
    asks_for_relative_time: bool,
    asks_for_current_state: bool,
    asks_for_recall_support: bool,
) -> str:
    if family in {"information_extraction", "single_anchor"}:
        if asks_for_recall_support:
            return "AttributeLookup+Support"
        return "AttributeLookup"
    if family == "aggregation":
        if asks_for_comparison:
            return "DifferenceAggregate"
        if asks_for_ordering:
            return "OrderedChoice"
        return "CountLookup"
    if family == "ordering":
        return "OrderedChoice"
    if family == "temporal":
        if asks_for_temporal_difference:
            return "TemporalInterval"
        if asks_for_relative_time:
            return "RelativeTime"
        return "DirectValue"
    if family == "current_state":
        return "CurrentStateResolution"
    if family in {"knowledge_update", "conflict_update"}:
        return "StateUpdateResolution"
    return "AttributeLookup"


def _required_slots_for_schema(schema_name: str) -> tuple[str, ...]:
    schema = str(schema_name or "").strip()
    return {
        "DirectValue": ("direct_value",),
        "DirectValue+Support": ("direct_value", "support"),
        "AttributeLookup": ("direct_value",),
        "AttributeLookup+Support": ("direct_value", "support"),
        "EntityLookup": ("direct_value",),
        "EntityLookup+Support": ("direct_value", "support"),
        "Comparison": ("left_operand", "right_operand"),
        "DifferenceAggregate": ("left_operand", "right_operand"),
        "Aggregate": ("operand_1", "operand_2"),
        "CountLookup": ("aggregate_items",),
        "CountDistinctItems": ("aggregate_items",),
        "CountEvents": ("aggregate_items",),
        "SumOperands": ("operand_1", "operand_2"),
        "PercentageAggregate": ("operand_1", "operand_2"),
        "OrderedChoice": ("choice_a", "choice_b"),
        "TemporalInterval": ("event_a", "event_b", "time_a", "time_b"),
        "RelativeTime": ("event", "reference_time"),
        "CurrentState": ("entity", "current_value", "direct_value"),
        "CurrentStateResolution": ("entity", "current_value", "direct_value"),
        "StateUpdate": ("old_value", "new_value", "direct_value"),
        "StateUpdateResolution": ("old_value", "new_value", "direct_value"),
    }.get(schema, ("direct_value",))


def _preferred_attributes_for_query(normalized_query: str) -> tuple[str, ...]:
    preferences: list[str] = []
    is_duration_query = bool(
        re.search(r"\bhow\s+long\b", normalized_query)
        or re.search(r"\bduration\b", normalized_query)
    )
    if " where " in f" {normalized_query} ":
        preferences.append("location")
    if " when " in f" {normalized_query} " and not is_duration_query:
        preferences.extend(["date", "time"])
    if re.search(r"\bwhat\s+.+\s+am\s+i\s+(?:currently\s+)?reading\b", normalized_query):
        preferences.append("reading")
    if re.search(r"\bwhat\s+.+\s+am\s+i\s+(?:currently\s+)?watching\b", normalized_query):
        preferences.append("watching")
    if re.search(r"\bwhat\s+.+\s+am\s+i\s+(?:currently\s+)?studying\b", normalized_query):
        preferences.append("studying")
    if re.search(r"\bwhat\s+.+\s+do\s+i\s+(?:currently\s+)?use\b", normalized_query):
        preferences.append("using")
    if not is_duration_query and any(token in normalized_query.split() for token in {"occupation", "job", "role", "profession", "work"}):
        preferences.append("occupation")
    if " breed " in f" {normalized_query} ":
        preferences.append("breed")
    if any(token in normalized_query.split() for token in {"name", "called", "named", "title"}):
        preferences.append("name")
    if any(token in normalized_query.split() for token in {"speed", "rate", "bandwidth", "mbps", "gbps"}):
        preferences.append("speed")
    if any(token in normalized_query.split() for token in {"price", "cost", "spent", "pay", "paid", "fare", "cashback", "saved", "earn", "earned"}):
        preferences.extend(["cost", "amount"])
    return _dedupe_preserve_order(preferences)


def _extract_query_targets_impl(row: dict[str, Any], query_family: str) -> QueryTargets:
    raw_query = str(row.get("query", "") or "").strip()
    profile = cached_build_profile(raw_query)
    normalized_query = profile.normalized_text
    named_tokens = {token.lower() for token in profile.named_tokens}
    query_tokens = cached_word_tokenize(normalized_query)
    query_tags = cached_pos_tag(tuple(query_tokens))
    family = str(query_family or "").strip().lower()

    choice_markers = _find_markers(normalized_query, _CHOICE_MARKERS)
    candidate_numbers = _dedupe_preserve_order(_extract_numeric_tokens(raw_query, normalized_query))
    candidate_entities, subject_entities, context_entities, entities_metadata = _extract_candidate_entities(
        raw_query,
        normalized_query,
        query_tags,
        named_tokens,
        query_family=family,
    )
    asks_for_comparison = _is_explicit_comparison_query(normalized_query)
    explicit_ordering_markers = {
        marker
        for marker in choice_markers
        if marker in _ORDERING_MARKERS and marker not in {"first", "second", "third", "fourth", "fifth"}
    }
    asks_for_ordering = family == "ordering" or _is_explicit_ordering_query(normalized_query) or bool(
        choice_markers
        and not re.search(r"\bfirst\s+order\b", normalized_query)
        and explicit_ordering_markers
    )
    asks_for_current_state = family == "current_state" or (
        _has_phrase(normalized_query, _CURRENT_STATE_MARKERS) and not _is_explicit_ordering_query(normalized_query)
    )
    has_from_to = bool(re.search(r"\bfrom\s+.+?\s+to\s+.+$", normalized_query))
    
    # Use stem sets to find truly distinct anchors
    all_stem_sets = [set(get_stems_for_text(e)) for e in candidate_entities if e]
    distinct_stem_sets = []
    for s_set in all_stem_sets:
        if not any(s_set.issubset(other) or other.issubset(s_set) for other in distinct_stem_sets):
            distinct_stem_sets.append(s_set)
    
    has_two_anchors = len(distinct_stem_sets) >= 2 or (len(distinct_stem_sets) == 1 and any(token in normalized_query for token in {"between", "after", "before", "since", "until"}))

    relative_since_query = bool(
        re.search(
            r"\bhow\s+(?:long|many)\b.*\b(?:ago|have\s+passed\s+since|has\s+it\s+been\s+since)\b",
            normalized_query,
        )
    )
    asks_for_relative_time = bool(
        _extract_relative_time_entities(normalized_query)
        or re.search(r"\bhow\s+(?:long|many)\b.*\bago\b", normalized_query)
        or relative_since_query
        or re.search(r"\bhow\s+many\s+(?:days?|weeks?|months?|years?)\s+(?:older|younger)\b.*\bthan\s+when\s+i\b", normalized_query)
    )
    asks_for_temporal_difference = not asks_for_relative_time and _is_explicit_temporal_difference_query(
        normalized_query,
        has_from_to=has_from_to,
        has_two_anchors=has_two_anchors,
    )
    if family in {"temporal", "information_extraction"}:
        # Strip trailing temporal anchors from entity keys so that
        # "jewelry last saturday" → "jewelry" and "music event last saturday"
        # → "music event".  This prevents slot binding from needing the full
        # temporally-anchored phrase verbatim in the source memory.
        stripped_candidates = tuple(
            _strip_trailing_temporal_anchor(e) for e in candidate_entities
        )
        # Re-dedupe in case stripping caused collisions.
        seen_stripped: set[str] = set()
        deduped_stripped: list[str] = []
        for ent in stripped_candidates:
            if ent and ent not in seen_stripped:
                seen_stripped.add(ent)
                deduped_stripped.append(ent)
        candidate_entities = tuple(deduped_stripped)
        subject_entities = tuple(e for e in (
            _strip_trailing_temporal_anchor(e) for e in subject_entities
        ) if e)
    if family == "temporal":
        candidate_entities = _filter_temporal_entities(candidate_entities)
        if asks_for_relative_time:
            relative_entities = _filter_temporal_entities(
                tuple(_extract_relative_time_entities(normalized_query))
            )
            candidate_entities = _prune_relative_time_entities(candidate_entities, relative_entities)
        candidate_entities, context_entities = _prune_generic_temporal_context_entities(
            candidate_entities,
            subject_entities=subject_entities,
            context_entities=context_entities,
            normalized_query=normalized_query,
        )
        if asks_for_temporal_difference:
            pruned_stem_sets: list[set[str]] = []
            for entity in candidate_entities:
                stems = {stem for stem in get_stems_for_text(entity) if stem}
                if not stems:
                    continue
                if any(stems <= other or other <= stems for other in pruned_stem_sets):
                    continue
                pruned_stem_sets.append(stems)
            pruned_anchor_count = len(pruned_stem_sets)
            if (
                pruned_anchor_count < 2
                and not has_from_to
                and not re.search(r"\b(after|before|between|since|until)\b", normalized_query)
            ):
                asks_for_temporal_difference = False
    if family == "aggregation":
        candidate_entities = _prune_count_clause_entities(
            candidate_entities,
            normalized_query=normalized_query,
        )
        subject_entities = tuple(entity for entity in subject_entities if entity in candidate_entities)
        context_entities = tuple(entity for entity in context_entities if entity in candidate_entities)
    asks_for_recall_support = bool(
        _has_phrase(normalized_query, _RECALL_SUPPORT_MARKERS)
        or (
            family in {"single_anchor", "information_extraction", "aggregation"}
            and "recommend" in normalized_query
            and normalized_query.startswith(("which ", "what ", "who "))
        )
        or (family in {"single_anchor", "information_extraction"} and _has_phrase(normalized_query, ("what did i", "which one", "the one")))
    )
    ordered_choice_entities = tuple(_dedupe_preserve_order(_extract_choice_entities(raw_query.lower())))
    count_entities = tuple()
    if family in {"aggregation", "current_state"} and _distinct_count_like_query(normalized_query):
        count_entities = tuple(_extract_count_query_entities(normalized_query))
    schema_name = _infer_schema_name(
        family=family,
        asks_for_comparison=asks_for_comparison,
        asks_for_ordering=asks_for_ordering,
        asks_for_temporal_difference=asks_for_temporal_difference,
        asks_for_relative_time=asks_for_relative_time,
        asks_for_current_state=asks_for_current_state,
        asks_for_recall_support=asks_for_recall_support,
    )
    alternative_entities = (
        ordered_choice_entities[:2]
        if len(ordered_choice_entities) >= 2 and (asks_for_ordering or asks_for_comparison)
        else tuple((subject_entities or candidate_entities)[:2])
    )
    if len(alternative_entities) >= 2 and (asks_for_ordering or asks_for_comparison):
        candidate_entities = alternative_entities + tuple(
            entity for entity in candidate_entities if entity not in alternative_entities
        )
    if count_entities:
        candidate_entities = count_entities + tuple(
            entity
            for entity in candidate_entities
            if entity not in count_entities and not _is_count_scaffold_entity(entity, count_entities)
        )
        # For count queries, subject_entities should represent the counted object(s).
        # Surrounding usage/activity/location context remains available, but as context.
        subject_entities = count_entities
        alternative_entities = _prune_count_scaffold_alternatives(
            alternative_entities,
            count_entities=count_entities,
        )
        context_entities = tuple(
            entity for entity in candidate_entities if entity not in subject_entities
        )
    preferred_attributes = _preferred_attributes_for_query(normalized_query)
    required_slots = _required_slots_for_schema(schema_name)
    answerability_hints = {
        "requires_reference_time": asks_for_relative_time,
        "requires_ordering": asks_for_ordering,
        "requires_numeric_reasoning": asks_for_comparison
        or asks_for_temporal_difference
        or (" how much " in f" {normalized_query} "),
        "requires_support": asks_for_recall_support,
    }
    planning_payload = {
        "query_family": family,
        "schema_name": schema_name,
        "required_slots": required_slots,
        "alternative_entities": alternative_entities,
        "preferred_attributes": preferred_attributes,
        "answerability_hints": answerability_hints,
    }

    return QueryTargets(
        raw_query=raw_query,
        normalized_query=normalized_query,
        query_family=family,
        schema_name=schema_name,
        candidate_entities=candidate_entities,
        subject_entities=subject_entities,
        context_entities=context_entities,
        alternative_entities=alternative_entities,
        candidate_numbers=candidate_numbers,
        preferred_attributes=preferred_attributes,
        choice_markers=choice_markers,
        asks_for_comparison=asks_for_comparison,
        asks_for_ordering=asks_for_ordering,
        asks_for_current_state=asks_for_current_state,
        asks_for_temporal_difference=asks_for_temporal_difference,
        asks_for_relative_time=asks_for_relative_time,
        asks_for_recall_support=asks_for_recall_support,
        required_slots=required_slots,
        answerability_hints=answerability_hints,
        planning_payload=planning_payload,
        entities_metadata=entities_metadata,
    )


@lru_cache(maxsize=4_000)
def _extract_query_targets_cached(raw_query: str, query_family: str) -> QueryTargets:
    """Cache wrapper — key is the raw query string + family (both hashable)."""
    # We need a minimal row-like dict; only 'query' is read inside the impl.
    return _extract_query_targets_impl({"query": raw_query}, query_family)


def extract_query_targets(row: dict[str, Any], query_family: str) -> QueryTargets:
    """Public API — cached by (query, family) to avoid redundant NLP work."""
    raw_query = str(row.get("query", "") or "").strip()
    return _extract_query_targets_cached(raw_query, str(query_family or "").strip().lower())
