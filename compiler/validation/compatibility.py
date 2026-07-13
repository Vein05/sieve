"""Compatibility and validation helpers shared by compiler layers."""

from __future__ import annotations

import re
from typing import Any

from shared.nlp import cached_build_profile, check_stem_equivalence, get_clean_tokens, get_stems_for_text, stem_token
from evidence.schema import EvidencePlan
from retrieval.query_targets import QueryTargets
from ..adapters.spacy_adapter import extract_head_nouns
from ..execution.requirements import is_current_state_schema, is_direct_lookup_schema, is_state_update_schema
from ..runtime.text import _binding_source_text, _binding_token_sets, _normalize_text, _split_session_header

_GENERIC_FRAGMENT_PHRASES = (
    "advice on",
    "can you",
    "could you",
    "for my",
    "help me",
    "i want",
    "i need",
    "recommend",
    "some good",
    "what should",
    "which one",
    "what kind",
)
_GENERIC_VALUE_TEXTS = {
    "a",
    "an",
    "it",
    "me",
    "my",
    "the",
    "this",
    "that",
    "thing",
    "something",
    "someone",
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
}
_LOW_QUALITY_SPAN_TEXTS = {"a", "an", "can", "it", "my", "one", "that", "the", "this"}
_SOFT_INVALID_REASONS = {"generic_fragment"}
_QUERY_FOCUS_SKIP_TOKENS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "which",
    "who",
    "with",
}
_DURATION_OBJECT_QUERY_RE = re.compile(
    r"\bhow\s+(?:long|many\s+(?:days?|weeks?|months?|years?|hours?|minutes?))\b.*?\bhave\s+i\s+been\s+[a-z]+ing\s+.+",
    re.IGNORECASE,
)
_GENERIC_ENTITY_HEADS = {
    "issue",
    "problem",
    "thing",
    "name",
    "kind",
    "type",
    "person",
    "place",
    "location",
    "disease",
    "condition",
    "symptom",
}
_GENERIC_MATCH_HEADS = {
    "class",
    "classes",
    "device",
    "devices",
    "event",
    "events",
    "issue",
    "issues",
    "plan",
    "plans",
    "project",
    "projects",
    "service",
    "services",
    "show",
    "shows",
    "trip",
    "trips",
}


def _generic_query_entity(entity: str) -> bool:
    normalized = _normalize_text(entity)
    if not normalized:
        return False
    tokens = set(normalized.split())
    if not tokens:
        return False
    if tokens <= _GENERIC_ENTITY_HEADS | {"health"}:
        return True
    heads = extract_head_nouns(entity)
    if not heads:
        return False
    return set(heads) <= _GENERIC_ENTITY_HEADS


def _binding_value_from_session_header(display_text: str, source_text: str) -> bool:
    value = str(display_text or "").strip()
    if not value:
        return False
    header_text, body_text = _split_session_header(source_text)
    if not header_text:
        return False
    normalized_value = _normalize_text(value)
    normalized_header = _normalize_text(header_text)
    normalized_body = _normalize_text(body_text)
    return bool(
        normalized_value
        and normalized_value in normalized_header
        and normalized_value not in normalized_body
    )


def _is_generic_fragment(text: str, source_text: str) -> bool:
    normalized = _normalize_text(text)
    normalized_source = _normalize_text(source_text)
    if not normalized:
        return True
    if normalized in _GENERIC_VALUE_TEXTS:
        return True
    tokens = normalized.split()
    if tokens:
        functional_heads = {"of", "by", "for", "with", "from", "to", "at", "on", "in", "or", "and"}
        if tokens[0].lower() in functional_heads:
            return True

        answerish_head = tokens[0] not in {"i", "ive", "im", "my", "that", "this", "it", "we", "you"}
        answerish_body = not any(
            token in {"am", "are", "be", "been", "being", "bought", "got", "have", "is", "ordered", "recommend", "thinking", "trying", "use", "using", "was", "were", "wondering"}
            for token in tokens
        )
        if len(tokens) <= 3:
            has_substantive = any(not (token in functional_heads or token in {"the", "a", "an", "this", "that", "it"}) for token in tokens)
            if not has_substantive:
                return True

        if answerish_head and answerish_body and 2 <= len(tokens) <= 12:
            return False

    has_numeric = bool(re.search(r"\d+", text))
    has_time = bool(re.search(r"\d{1,2}:\d{2}", text) or re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|january|february|march|april|may|june|july|august|september|october|november|december|today|yesterday|ago)\b", text.lower()))
    if has_time and len(normalized.split()) >= 1:
        return False
    if has_numeric and len(normalized.split()) >= 2:
        return False
    if len(normalized.split()) > 5 and any(phrase in normalized_source for phrase in _GENERIC_FRAGMENT_PHRASES):
        return True
    if len(normalized.split()) > 8 and any(token in normalized_source.split() for token in ("i", "ive", "im", "you", "we")):
        return True
    return False


def _slot_allows_descriptive_fragment(slot_type: str) -> bool:
    return slot_type in {"temporal_event", "ordering_event"}


def _binding_matches_query_focus(binding: dict[str, Any], focus_tokens: set[str]) -> bool:
    if not focus_tokens:
        return True
    content_tokens, named_tokens = _binding_token_sets(binding)
    binding_tokens = content_tokens | named_tokens
    return bool(binding_tokens & focus_tokens)


def _binding_focus_overlap_tokens(binding: dict[str, Any], focus_tokens: set[str]) -> set[str]:
    if not focus_tokens:
        return set()
    content_tokens, named_tokens = _binding_token_sets(binding)
    binding_tokens = content_tokens | named_tokens
    return set(binding_tokens & focus_tokens)


def _direct_value_requires_entity_match(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets: QueryTargets,
) -> bool:
    if not is_direct_lookup_schema(plan):
        return False
    subject_entities = tuple(entity for entity in query_targets.subject_entities if entity)
    candidate_entities = subject_entities or tuple(entity for entity in query_targets.candidate_entities if entity)
    if not candidate_entities:
        return False
    if all(_generic_query_entity(entity) for entity in candidate_entities):
        return False
    normalized_query = _normalize_text(str(row.get("query", "")))
    if normalized_query.startswith("where ") or normalized_query.startswith("when "):
        return False
    if plan.family == "aggregation":
        return bool(_DURATION_OBJECT_QUERY_RE.search(normalized_query))
    if plan.family not in {"single_anchor", "information_extraction"}:
        return False
    if not any(marker in normalized_query for marker in (" speed ", " rate ", " age ", " price ", " cost ", " amount ", " total ")):
        return False
    return bool(re.search(r"\bwhat\s+(?:speed|rate|age|price|cost|amount)\b", normalized_query))


def _match_entity_surface(source_text: str, entity: str) -> str | None:
    if not entity or not source_text:
        return None
    entity_tokens = get_clean_tokens(entity)
    source_tokens = get_clean_tokens(source_text)
    e_stems = get_stems_for_text(entity)
    s_stems = get_stems_for_text(source_text)
    if not e_stems:
        return None
    meaningful_tokens = [
        token
        for token in entity_tokens
        if stem_token(token) not in {"vintage", "daily", "local", "new", "old", "current", "previous", "favorite", "final", "total", "mention", "order", "list"}
    ]
    if meaningful_tokens and all(
        any(check_stem_equivalence(entity_token, source_token) for source_token in source_tokens)
        for entity_token in meaningful_tokens
    ):
        return entity
    matched_stems = 0
    for e_stem in e_stems:
        found = False
        if e_stem in s_stems:
            found = True
        else:
            from shared.nlp import CONVERSATIONAL_SYNONYMS

            syns = CONVERSATIONAL_SYNONYMS.get(e_stem, set())
            if syns and any(syn in s_stems for syn in syns):
                found = True
        if found:
            matched_stems += 1
    if matched_stems == len(e_stems):
        return entity
    normalized_entity = _normalize_text(entity)
    normalized_source = _normalize_text(source_text)
    if normalized_entity and re.search(rf"(?<!\w){re.escape(normalized_entity)}(?!\w)", normalized_source):
        return entity
    entity_profile = cached_build_profile(entity)
    source_profile = cached_build_profile(source_text)
    entity_tokens = {
        token
        for token in (entity_profile.content_tokens | entity_profile.named_tokens)
        if token and token not in _QUERY_FOCUS_SKIP_TOKENS
    }
    if not entity_tokens:
        return None
    source_tokens = source_profile.content_tokens | source_profile.named_tokens
    overlap = entity_tokens & source_tokens
    action_like = {
        token
        for token in entity_tokens
        if token.endswith("ing") or token in {"start", "finish", "begin", "end", "move", "buy", "sell", "leave", "arrive"}
    }
    generic = {"fare", "first", "last", "order", "ride", "price", "cost", "discount"} | _GENERIC_MATCH_HEADS
    weak_modifiers = {"vintage", "daily", "local", "new", "old", "current", "previous", "favorite", "final", "total"}
    informative_modifiers = entity_tokens - generic - weak_modifiers
    if overlap and (len(overlap) / float(len(entity_tokens))) >= 0.5:
        if action_like and not (action_like & overlap):
            return None
        if len(overlap) == 1 and len(entity_tokens) >= 2 and overlap <= weak_modifiers:
            return None
        if informative_modifiers and not (informative_modifiers & source_tokens):
            return None
        if overlap - generic:
            return entity
        if not informative_modifiers:
            return entity
    return None


def _best_matching_entity(source_text: str, candidate_entities: tuple[str, ...]) -> str | None:
    best: tuple[int, int, int, str] | None = None
    normalized_source = _normalize_text(source_text)
    for entity in candidate_entities:
        surface = _match_entity_surface(source_text, entity)
        if not surface:
            continue
        exact = 1 if re.search(rf"(?<!\w){re.escape(_normalize_text(entity))}(?!\w)", normalized_source) else 0
        key = (exact, len(_normalize_text(entity).split()), -len(surface), surface)
        if best is None or key > best:
            best = key
    return None if best is None else best[3]


def _slot_expected_entities(slot, query_targets: QueryTargets) -> tuple[str, ...]:
    hint = str(getattr(slot, "entity_key", "") or "").strip()
    slot_type = str(getattr(slot, "slot_type", "") or "").strip()
    if hint:
        related: list[str] = [hint]
        if slot_type in {"temporal_event", "temporal_time"}:
            related.extend(entity for entity in query_targets.candidate_entities if entity)
            related.extend(entity for entity in query_targets.alternative_entities if entity)
        if slot_type in {"ordering_event", "comparison_operand", "temporal_event", "temporal_time"}:
            normalized_hint = _normalize_text(hint)
            hint_tokens = set(normalized_hint.split())
            for entity in query_targets.candidate_entities:
                candidate = str(entity or "").strip()
                normalized_candidate = _normalize_text(candidate)
                if not candidate or normalized_candidate == normalized_hint:
                    continue
                candidate_tokens = set(normalized_candidate.split())
                if not candidate_tokens:
                    continue
                if normalized_candidate in normalized_hint or normalized_hint in normalized_candidate or bool(candidate_tokens & hint_tokens):
                    related.append(candidate)
        return tuple(dict.fromkeys(entity for entity in related if entity))
    return tuple(entity for entity in query_targets.candidate_entities if entity)


def _slot_entity_match(slot, source_text: str, query_targets: QueryTargets) -> str | None:
    expected_entities = _slot_expected_entities(slot, query_targets)
    if not expected_entities:
        return None
    return _best_matching_entity(source_text, expected_entities)


def _event_matches_expected(binding: dict[str, Any] | None, expected_entity: str | None) -> bool:
    if not expected_entity:
        return True
    if not isinstance(binding, dict):
        return False
    source_text = _binding_source_text(binding)
    return _match_entity_surface(source_text, expected_entity) is not None


def _binding_quality_score(binding: dict[str, Any] | None) -> float:
    if not isinstance(binding, dict):
        return -1.0
    text = str(binding.get("text") or "").strip()
    if not text:
        return -1.0
    normalized = _normalize_text(text)
    score = 0.0
    token_count = len(normalized.split())
    if normalized in _LOW_QUALITY_SPAN_TEXTS:
        score -= 5.0
    if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", text):
        score += 3.0
    if any(char.isdigit() for char in text):
        score += 2.0
    if ":" in text or "$" in text or "%" in text:
        score += 1.5
    if token_count > 8:
        score -= 1.0
    if _is_generic_fragment(text, _binding_source_text(binding)):
        score -= 3.0
    schema_hints = binding.get("schema_hints")
    labels = set(schema_hints.get("labels", [])) if isinstance(schema_hints, dict) else set()
    is_answer_memory = str(binding.get("memory_id") or "").startswith("answer_")
    if str(binding.get("speaker") or "").strip().lower() == "user":
        score += 0.5
    if is_answer_memory:
        score += 0.75
    if is_answer_memory and binding.get("object_id"):
        score += 0.5
    if is_answer_memory and binding.get("object_id") and (
        "direct_answer_candidate" in labels
        or "attribute_value" in labels
        or str(binding.get("object_type") or "") in {"state", "attribute_fact", "preference", "update"}
    ):
        score += 0.75
    extracted_value_text = str(binding.get("extracted_value_text") or "").strip()
    if is_answer_memory and extracted_value_text and _normalize_text(extracted_value_text) == normalized:
        score += 0.5
        if len(_normalize_text(extracted_value_text).split()) >= 2:
            score += 1.5
    elif is_answer_memory and extracted_value_text:
        normalized_extracted = _normalize_text(extracted_value_text)
        if normalized_extracted and normalized and normalized != normalized_extracted:
            extracted_tokens = normalized_extracted.split()
            if len(extracted_tokens) >= 2 and (normalized in normalized_extracted or normalized_extracted.startswith(normalized)):
                score -= 2.0
    return score


def _entity_head_noun_compatible(
    *,
    plan: EvidencePlan,
    query_targets: QueryTargets,
    source_text: str,
) -> bool:
    """Check whether evidence source text shares a head noun with query entities.

    Returns True (compatible) when:
    - The plan family is aggregation/temporal/ordering (operands may differ)
    - The schema is not a personal-fact lookup
    - No query entities are available
    - Head nouns overlap between query entities and evidence source
    - Lexical entity matching already passes (via _best_matching_entity)

    The gate is conservative: if lexical entity matching finds the source
    relevant, the head-noun check is skipped.  This avoids false rejections
    when the source uses synonyms or related terms (e.g. "vermouth ratios"
    for a "gin martini" query).
    """
    if plan.family in {"aggregation", "temporal", "ordering"} and not is_direct_lookup_schema(plan):
        return True
    if not (is_direct_lookup_schema(plan) or is_current_state_schema(plan) or is_state_update_schema(plan)):
        return True
    primary_entities = tuple(e for e in query_targets.subject_entities if e) or tuple(
        e for e in query_targets.candidate_entities if e
    )
    if not primary_entities:
        return True
    if all(_generic_query_entity(entity) for entity in primary_entities):
        return True
    if _best_matching_entity(source_text, primary_entities):
        return True
    query_heads: set[str] = set()
    for entity in primary_entities:
        query_heads |= extract_head_nouns(entity)
    if not query_heads:
        return True
    source_heads = extract_head_nouns(source_text)
    if not source_heads:
        return True
    overlapping_heads = query_heads & source_heads
    if not overlapping_heads:
        return False
    matched_entity = _best_matching_entity(source_text, primary_entities)
    if not (overlapping_heads & _GENERIC_MATCH_HEADS):
        return True

    source_profile = cached_build_profile(source_text)
    source_tokens = {
        token
        for token in (source_profile.content_tokens | source_profile.named_tokens)
        if token and token not in _QUERY_FOCUS_SKIP_TOKENS
    }
    if not source_tokens:
        return False

    def _entity_modifier_overlap(entity: str) -> bool:
        entity_heads = {head for head in extract_head_nouns(entity) if head}
        entity_tokens = {
            token
            for token in get_clean_tokens(entity)
            if token and token not in _QUERY_FOCUS_SKIP_TOKENS
        }
        modifier_tokens = {
            token
            for token in entity_tokens
            if token not in entity_heads and token not in _GENERIC_MATCH_HEADS
        }
        if not modifier_tokens:
            return bool(_best_matching_entity(source_text, (entity,)))
        return bool(modifier_tokens & source_tokens)

    if matched_entity and _entity_modifier_overlap(matched_entity):
        return True

    for entity in primary_entities:
        if _entity_modifier_overlap(entity):
            return True
    return False


def _slot_attribute_compatible(slot_attribute: str, unit_attribute: str) -> bool:
    slot_attribute = str(slot_attribute or "").strip().lower()
    unit_attribute = str(unit_attribute or "").strip().lower()
    if not slot_attribute or not unit_attribute:
        return False
    if slot_attribute == unit_attribute:
        return True
    if slot_attribute == "name" and unit_attribute in {"reading", "watching", "studying", "using", "employer"}:
        return True
    if slot_attribute == "brand" and unit_attribute == "using":
        return True
    if slot_attribute in {"duration", "time"} and unit_attribute in {"duration", "time"}:
        return True
    return False
