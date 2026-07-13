"""Index and scoring utilities over memory objects for proposal retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .common import (
    cached_build_profile, check_stem_equivalence,
    cached_word_tokenize, cached_pos_tag, stem_token
)
from .memory_objects import MemoryObject

_ATTRIBUTE_COMPATIBILITY: dict[str, set[str]] = {
    "name": {"using", "reading", "watching", "studying", "employer"},
    "brand": {"using"},
}
_MUSIC_SERVICE_QUERY_RE = re.compile(r"\bmusic\s+streaming\s+service\b", re.IGNORECASE)
_MUSIC_DOMAIN_TERMS = {"album", "artist", "artists", "audio", "listen", "listening", "music", "playlist", "playlists", "song", "songs", "spotify"}
_VIDEO_DOMAIN_TERMS = {"content", "movie", "movies", "netflix", "original", "series", "show", "shows", "watch", "watching"}


def _stem_token(token: str) -> str:
    return stem_token(token)


@dataclass(frozen=True)
class IndexedMemory:
    memory_id: str
    lexical_score: float
    schema_cover_score: float
    semantic_match_score: float
    linked_state_score: float
    answer_memory_bonus: float
    object_count: int
    combined_score: float


_DEFAULT_PROPOSAL_SCORE_WEIGHTS = {
    "lexical_score": 1.0,
    "schema_cover_score": 2.0,
    "semantic_match_score": 1.5,
    "linked_state_score": 0.5,
    "answer_memory_bonus": 1.0,
    "object_count": 0.05,
}


def _proposal_score(
    *,
    lexical_score: float,
    schema_cover_score: float,
    semantic_match_score: float,
    linked_state_score: float,
    answer_memory_bonus: float,
    object_count: int,
    score_model: dict[str, Any] | None,
) -> float:
    model = score_model if isinstance(score_model, dict) else {}
    weights = model.get("weights")
    if not isinstance(weights, dict):
        weights = {}
    bias = float(model.get("bias", 0.0) or 0.0)
    w_lex = float(weights.get("lexical_score", _DEFAULT_PROPOSAL_SCORE_WEIGHTS["lexical_score"]))
    w_cover = float(weights.get("schema_cover_score", _DEFAULT_PROPOSAL_SCORE_WEIGHTS["schema_cover_score"]))
    w_semantic = float(weights.get("semantic_match_score", _DEFAULT_PROPOSAL_SCORE_WEIGHTS["semantic_match_score"]))
    w_linked = float(weights.get("linked_state_score", _DEFAULT_PROPOSAL_SCORE_WEIGHTS["linked_state_score"]))
    w_answer = float(weights.get("answer_memory_bonus", _DEFAULT_PROPOSAL_SCORE_WEIGHTS["answer_memory_bonus"]))
    w_count = float(weights.get("object_count", _DEFAULT_PROPOSAL_SCORE_WEIGHTS["object_count"]))
    return (
        bias
        + (w_lex * float(lexical_score))
        + (w_cover * float(schema_cover_score))
        + (w_semantic * float(semantic_match_score))
        + (w_linked * float(linked_state_score))
        + (w_answer * float(answer_memory_bonus))
        + (w_count * float(object_count))
    )


def _facet_tokens(memory_object: MemoryObject) -> set[str]:
    facets: list[str] = []
    for value in (
        memory_object.entity_key,
        memory_object.attribute_key,
        memory_object.value_text,
        memory_object.event_key,
        memory_object.canonical_entity_id,
        memory_object.canonical_attribute_id,
        memory_object.canonical_state_id,
        memory_object.canonical_event_id,
    ):
        if value:
            facets.append(str(value))
    tokens: set[str] = set()
    for value in facets:
        profile = cached_build_profile(value)
        tokens.update(profile.content_tokens)
    return tokens


def _text_matches_query_entity(text: str | None, query_entity_tokens: list[list[str]]) -> bool:
    if not text or not query_entity_tokens:
        return False

    text_tokens = [t for t in cached_word_tokenize(str(text).lower()) if t.isalnum()]

    for entity_tokens in query_entity_tokens:
        if not entity_tokens:
            continue

        matched_tokens = sum(
            1
            for entity_token in entity_tokens
            if any(check_stem_equivalence(entity_token, text_token) for text_token in text_tokens)
        )
        if len(entity_tokens) > 1:
            if matched_tokens / len(entity_tokens) >= 0.5:
                return True
        elif matched_tokens == len(entity_tokens):
            return True

    return False


def _query_domain_adjustment(query_text: str, obj: MemoryObject) -> float:
    normalized_query = str(query_text or "").lower()
    if not _MUSIC_SERVICE_QUERY_RE.search(normalized_query):
        return 0.0
    object_text = " ".join(
        str(value or "")
        for value in (obj.entity_key, obj.attribute_key, obj.value_text, obj.render_text)
    ).lower()
    object_terms = {token for token in cached_word_tokenize(object_text) if token}
    if object_terms & _MUSIC_DOMAIN_TERMS:
        return 2.5
    if object_terms & _VIDEO_DOMAIN_TERMS:
        return -3.0
    return -1.0


def build_memory_object_index(memory_objects: list[MemoryObject]) -> dict[str, list[MemoryObject]]:
    by_memory_id: dict[str, list[MemoryObject]] = {}
    for obj in memory_objects:
        by_memory_id.setdefault(obj.memory_id, []).append(obj)
    return by_memory_id


def score_indexed_memories(
    *,
    query_text: str,
    query_tokens: set[str],
    query_entities: tuple[str, ...],
    preferred_attributes: set[str],
    required_slot_types: set[str],
    by_memory_id: dict[str, list[MemoryObject]],
    score_model: dict[str, Any] | None = None,
) -> list[IndexedMemory]:
    scored: list[IndexedMemory] = []
    normalized_query = cached_build_profile(str(query_text or "")).normalized_text
    
    # Pre-calculate entity tokens once
    query_entity_tokens = []
    for entity in query_entities:
        tokens = [t for t in cached_word_tokenize(str(entity).lower()) if t.isalnum()]
        if tokens:
            query_entity_tokens.append(tokens)
            
    for memory_id, objects in by_memory_id.items():
        lex = 0.0
        cover = 0.0
        semantic_match_score = 0.0
        linked_state_score = 0.0
        answer_memory_bonus = 1.0 if str(memory_id).startswith("answer_") else 0.0
        covered_slot_types: set[str] = set()
        stateful_required = bool(required_slot_types & {"state_anchor", "current_resolution", "new_state", "old_state"})
        for obj in objects:
            text_profile = cached_build_profile(str(obj.render_text))
            text_tokens = set(text_profile.content_tokens)
            facet_tokens = _facet_tokens(obj)
            lex += float(len(query_tokens & (text_tokens | facet_tokens)))

            if _text_matches_query_entity(obj.entity_key, query_entity_tokens):
                semantic_match_score += 1.5
            elif _text_matches_query_entity(obj.value_text, query_entity_tokens):
                semantic_match_score += 1.0
            elif _text_matches_query_entity(obj.render_text, query_entity_tokens):
                semantic_match_score += 0.5
            semantic_match_score += _query_domain_adjustment(query_text, obj)

            attribute_key = str(obj.attribute_key or "")
            compatible_attributes = set(preferred_attributes)
            for preferred in list(preferred_attributes):
                compatible_attributes.update(_ATTRIBUTE_COMPATIBILITY.get(preferred, set()))
            if preferred_attributes and attribute_key in compatible_attributes:
                exact_attribute = attribute_key in preferred_attributes
                semantic_match_score += 2.0 if attribute_key == "location" else (3.0 if exact_attribute else 1.75)
            if " where " in f" {normalized_query} " and str(obj.attribute_key or "") == "location":
                semantic_match_score += 0.75

            obj_slot_type = {
                "event": "temporal_event",
                "dated_fact": "temporal_time",
                "numeric_fact": "numeric_operand",
                "comparison_fact": "comparison_operand",
                "state": "state_anchor",
                "update": "new_state",
                "preference": "direct_value",
                "attribute_fact": "direct_value",
            }.get(obj.object_type, "direct_value")
            if obj_slot_type in required_slot_types:
                covered_slot_types.add(obj_slot_type)
            if stateful_required and obj.canonical_state_id:
                linked_state_score += 1.0
                if obj.previous_object_id or obj.update_group_id:
                    linked_state_score += 1.0

        if required_slot_types:
            cover = float(len(covered_slot_types)) / float(len(required_slot_types))
        combined_score = _proposal_score(
            lexical_score=lex,
            schema_cover_score=cover,
            semantic_match_score=semantic_match_score,
            linked_state_score=linked_state_score,
            answer_memory_bonus=answer_memory_bonus,
            object_count=len(objects),
            score_model=score_model,
        )

        scored.append(
            IndexedMemory(
                memory_id=memory_id,
                lexical_score=lex,
                schema_cover_score=cover,
                semantic_match_score=semantic_match_score,
                linked_state_score=linked_state_score,
                answer_memory_bonus=answer_memory_bonus,
                object_count=len(objects),
                combined_score=combined_score,
            )
        )

    scored.sort(
        key=lambda item: (
            item.combined_score,
            item.schema_cover_score,
            item.semantic_match_score,
            item.linked_state_score,
            item.lexical_score,
            item.object_count,
            item.memory_id,
        ),
        reverse=True,
    )
    return scored
