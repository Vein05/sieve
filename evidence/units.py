"""Typed evidence-unit decomposition for CRISP evidence compilation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from controller.constants import DATE_RE, NEGATION_PATTERNS, TEMPORAL_HINTS, TIME_RE, UPDATE_MARKERS
from controller.logic import token_count
from controller.text_utils import normalize_text

from shared.nlp import CandidateView, cached_build_profile, cached_sent_tokenize, cached_word_tokenize, cached_pos_tag
from shared.constants import (
    COMPARISON_LANGUAGE_MARKERS as _COMPARISON_LANGUAGE_MARKERS,
    CURRENT_STATE_LANGUAGE_MARKERS as _CURRENT_STATE_LANGUAGE_MARKERS,
    REFERENCE_LANGUAGE_MARKERS as _REFERENCE_LANGUAGE_MARKERS,
)
from retrieval.memory_builder import (
    build_memory_objects,
    build_memory_objects_for_views,
)
from .memory_objects import MemoryObject, ProvenanceRecord


_SPEAKER_RE = re.compile(r"(?:^|[\n|])\s*(?P<speaker>[A-Za-z][\w-]{0,31})\s*:\s*")


_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:'[A-Za-z]+)?|[A-Z]{2,}|[A-Z][A-Za-z0-9_-]{1,})\b")
_NUMERIC_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_NUMBER_WORD_RE = re.compile(r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b", re.IGNORECASE)
_VALUE_TAIL_RE = re.compile(
    r"\b(?:is|was|are|were|becomes?|became|costs?|cost|worth|uses?|using|"
    r"switched to|changed to|now uses?|now use|currently|current|as of|"
    r"compared to|compared with|compared against|more than|less than|higher than|"
    r"lower than|bigger than|smaller than)\s+(?P<tail>[^.!?;|]+)"
)

_SUBJECT_SKIP_TOKENS = frozenset({
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
})

_NUMBER_WORD_VALUES = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}

_QUESTION_PREFIXES = frozenset({
    "can",
    "could",
    "did",
    "do",
    "does",
    "how",
    "is",
    "should",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "would",
})
_REQUEST_MARKERS = frozenset({
    "advice",
    "help",
    "ideas",
    "recommend",
    "recommended",
    "suggest",
    "suggestion",
    "suggestions",
    "tips",
})
_COUNT_STATEMENT_PREDICATES = frozenset({
    "bought",
    "collected",
    "completed",
    "finished",
    "got",
    "had",
    "have",
    "keep",
    "kept",
    "own",
    "owned",
    "read",
    "tracked",
    "use",
    "used",
    "watched",
})
_ADVICE_MARKERS = frozenset({
    "recommend",
    "recommended",
    "should",
    "suggest",
    "suggested",
    "tips",
    "try",
})
_INSTRUCTION_MARKERS = frozenset({
    "add",
    "bake",
    "beat",
    "blend",
    "boil",
    "cook",
    "fold",
    "fry",
    "mix",
    "pour",
    "preheat",
    "saute",
    "use",
    "whisk",
})
_FIRST_PERSON_TOKENS = frozenset({"i", "i'd", "i'll", "i'm", "i've", "im", "ive", "me", "my", "we", "our", "us"})
_PROGRESS_MARKERS = frozenset({
    "completed",
    "episodes",
    "finished",
    "pages",
    "progress",
    "read",
    "so",
    "videos",
    "watched",
})
_ACQUISITION_MARKERS = frozenset({
    "acquire",
    "acquired",
    "bought",
    "collected",
    "got",
    "ordered",
    "picked",
    "picked up",
    "purchased",
})
_COMPLETION_MARKERS = frozenset({
    "completed",
    "done",
    "finished",
    "read",
    "rewatched",
    "saw",
    "tried",
    "visited",
    "watched",
})


@dataclass(frozen=True)
class EvidenceUnit:
    unit_id: str
    memory_id: str
    parent_rank: int
    local_order: int
    source_text: str
    render_text: str
    speaker: str
    date_key: tuple[int, int, int]
    token_count: int
    unit_type: str
    entity_tokens: tuple[str, ...]
    numeric_values: tuple[str, ...]
    time_markers: tuple[str, ...]
    update_markers: tuple[str, ...]
    subject_tokens: tuple[str, ...]
    value_tokens: tuple[str, ...]
    has_comparison_language: bool
    has_current_state_language: bool
    has_reference_language: bool
    has_pronoun_language: bool
    eligibility_labels: tuple[str, ...]
    is_direct_answer_candidate: bool
    is_numeric_operand: bool
    is_temporal_anchor: bool
    is_state_assertion: bool
    is_update_assertion: bool
    is_current_state_candidate: bool
    is_attribute_value: bool
    is_countable_item: bool
    is_generic_state_text: bool
    is_low_authority_text: bool
    is_question_or_request: bool
    is_recommendation_or_advice: bool
    is_instructional_quantity: bool
    is_answer_like_quantity_statement: bool
    has_progress_marker: bool
    has_acquisition_marker: bool
    has_consumption_or_completion_marker: bool
    canonical_entity_id: str | None
    canonical_attribute_id: str | None
    object_type: str | None
    provenance: ProvenanceRecord


@dataclass(frozen=True)
class EvidenceUnitBundle:
    memory_id: str
    units: tuple[EvidenceUnit, ...]


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


def _split_speaker_turns(text: str) -> list[tuple[str, str, str]]:
    stripped = str(text or "").strip()
    if not stripped:
        return []

    matches = list(_SPEAKER_RE.finditer(stripped))
    if not matches:
        return [("", "unknown", stripped)]

    turns: list[tuple[str, str, str]] = []
    segment_start = 0
    for index, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(stripped)
        prefix = stripped[segment_start:match.start()].strip()
        speaker = match.group("speaker").strip().lower()
        body = stripped[body_start:body_end].strip()
        if body:
            turns.append((prefix, speaker, body))
        segment_start = body_end

    if not turns:
        return [("", "unknown", stripped)]
    return turns


def _split_sentences(text: str) -> list[str]:
    return cached_sent_tokenize(str(text or ""))


def _has_pronoun_language(tokens: list[str]) -> bool:
    # Use POS tags to identify pronouns reliably
    tags = cached_pos_tag(tuple(tokens))
    return any(tag in {"PRP", "PRP$"} for _, tag in tags)


def _extract_entity_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for match in _ENTITY_RE.finditer(text):
        token = match.group(0).strip()
        lowered = token.lower()
        if lowered in {"assistant", "human", "system", "user"}:
            continue
        tokens.append(lowered)
    return _dedupe_preserve_order(tokens)


def _extract_subject_tokens(
    *,
    profile_tokens: list[str],
    entity_tokens: tuple[str, ...],
    normalized_text: str,
) -> tuple[str, ...]:
    tokens: list[str] = []
    # Tag tokens for pronoun/entity skipping
    all_tokens = list(entity_tokens) + profile_tokens
    tags = cached_pos_tag(tuple(all_tokens))
    
    for (word, tag) in tags:
        lowered = word.strip().lower()
        if not lowered or lowered in _SUBJECT_SKIP_TOKENS:
            continue
        if tag in {"PRP", "PRP$"}:
            continue
        if lowered.isdigit():
            continue
        if lowered in TEMPORAL_HINTS or lowered in UPDATE_MARKERS:
            continue
        if any(marker.strip() == lowered for marker in ("more", "less", "higher", "lower", "current", "currently")):
            continue
        tokens.append(lowered)
        if len(tokens) >= 4:
            break
    if tokens:
        return _dedupe_preserve_order(tokens)

    fallback = [token for token in re.split(r"\s+", normalize_text(normalized_text)) if token and token not in _SUBJECT_SKIP_TOKENS]
    return _dedupe_preserve_order(fallback[:4])


def _extract_numeric_values(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for match in _NUMERIC_RE.finditer(text):
        token = match.group(0)
        start, end = match.span()
        left_char = text[start - 1] if start > 0 else ""
        right_char = text[end] if end < len(text) else ""
        if left_char in "/:" or right_char in "/:":
            continue
        if left_char.isalpha() or right_char.isalpha() or left_char == "_" or right_char == "_":
            continue
        values.append(token)
    lowered = str(text or "").lower()
    for match in _NUMBER_WORD_RE.finditer(lowered):
        start, end = match.span()
        left_char = lowered[start - 1] if start > 0 else ""
        right_char = lowered[end] if end < len(lowered) else ""
        if left_char.isalpha() or right_char.isalpha():
            continue
        normalized = _NUMBER_WORD_VALUES.get(match.group(1).lower())
        if normalized:
            values.append(normalized)
    return _dedupe_preserve_order(values)


def _extract_value_tokens(
    *,
    text: str,
    normalized_text: str,
    numeric_values: tuple[str, ...],
    time_markers: tuple[str, ...],
    update_markers: tuple[str, ...],
) -> tuple[str, ...]:
    values: list[str] = []
    values.extend(numeric_values)
    values.extend(time_markers)
    values.extend(update_markers)

    lowered = f" {normalized_text} "
    for match in _VALUE_TAIL_RE.finditer(lowered):
        tail = match.group("tail").strip()
        if not tail:
            continue
        tail_tokens = [token for token in tail.split() if token and token not in _SUBJECT_SKIP_TOKENS]
        values.extend(tail_tokens[:4])

    if not values:
        text_profile = cached_build_profile(text)
        values.extend(list(text_profile.content_tokens)[:4])

    return _dedupe_preserve_order(values)


def _has_comparison_language(normalized_text: str) -> bool:
    padded = f" {normalized_text} "
    return any(marker in padded for marker in _COMPARISON_LANGUAGE_MARKERS)


def _has_current_state_language(normalized_text: str) -> bool:
    padded = f" {normalized_text} "
    return any(marker in padded for marker in _CURRENT_STATE_LANGUAGE_MARKERS)


def _has_reference_language(normalized_text: str) -> bool:
    padded = f" {normalized_text} "
    return any(marker in padded for marker in _REFERENCE_LANGUAGE_MARKERS)


def _count_semantic_labels(
    *,
    text: str,
    normalized_text: str,
    profile_tokens: list[str],
    numeric_values: tuple[str, ...],
    speaker: str,
) -> tuple[str, ...]:
    labels: list[str] = []
    padded = f" {normalized_text} "
    token_set = {str(token).strip().lower() for token in profile_tokens if str(token).strip()}
    prefix_text = re.split(r"\?|\b(?:can you|could you|do you have any tips|any tips|please)\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    prefix_normalized = normalize_text(prefix_text)
    prefix_token_set = {str(token).strip().lower() for token in cached_word_tokenize(prefix_text) if str(token).strip()}
    starts_with_question = bool(profile_tokens and profile_tokens[0] in _QUESTION_PREFIXES)
    has_first_person = bool(token_set & _FIRST_PERSON_TOKENS)
    prefix_has_first_person = bool(prefix_token_set & _FIRST_PERSON_TOKENS)
    has_request_marker = bool(token_set & _REQUEST_MARKERS)
    has_advice_marker = bool(token_set & _ADVICE_MARKERS)
    has_instruction_marker = bool(token_set & _INSTRUCTION_MARKERS)
    has_progress_marker = (
        (" so far " in padded or " progress " in padded)
        and bool(token_set & {"completed", "finished", "read", "watched", "videos", "episodes", "pages"})
    ) or bool(token_set & {"completed", "episodes", "pages", "videos"} and token_set & {"finished", "read", "watched"})
    has_acquisition_marker = bool(token_set & _ACQUISITION_MARKERS) or " picked up " in padded
    has_completion_marker = bool(token_set & _COMPLETION_MARKERS)
    question_or_request = "?" in text or starts_with_question or has_request_marker or " please " in padded
    recommendation_or_advice = (
        has_advice_marker
        and not has_first_person
        and (speaker == "assistant" or has_instruction_marker or question_or_request)
    )
    instructional_quantity = bool(numeric_values) and has_instruction_marker and not has_first_person and not question_or_request
    answer_like_quantity_statement = (
        bool(numeric_values)
        and has_first_person
        and not question_or_request
        and not recommendation_or_advice
    )
    if (
        not answer_like_quantity_statement
        and bool(numeric_values)
        and prefix_has_first_person
        and bool(prefix_token_set & _COUNT_STATEMENT_PREDICATES)
        and not recommendation_or_advice
        and len(prefix_normalized.split()) >= 4
    ):
        answer_like_quantity_statement = True
    if question_or_request:
        labels.append("question_or_request")
    if recommendation_or_advice:
        labels.append("recommendation_or_advice")
    if instructional_quantity:
        labels.append("instructional_quantity")
    if answer_like_quantity_statement:
        labels.append("answer_like_quantity_statement")
    if has_progress_marker:
        labels.append("progress_marker")
    if has_acquisition_marker:
        labels.append("acquisition_marker")
    if has_completion_marker:
        labels.append("consumption_or_completion_marker")
    return _dedupe_preserve_order(labels)





def _extract_time_markers(text: str, *, profile_tokens: set[str], normalized_text: str) -> tuple[str, ...]:
    markers: list[str] = []
    for match in DATE_RE.finditer(text):
        markers.append(match.group(0))
    for match in TIME_RE.finditer(normalized_text):
        markers.append(match.group(0))
    for token in profile_tokens:
        if token in TEMPORAL_HINTS:
            markers.append(token)
    return _dedupe_preserve_order(markers)


def _extract_update_markers(text: str, *, profile_tokens: set[str], normalized_text: str) -> tuple[str, ...]:
    markers: list[str] = []
    for pattern in NEGATION_PATTERNS:
        if " " in pattern:
            matched = pattern in normalized_text
        else:
            matched = pattern in profile_tokens
        if matched:
            markers.append(pattern)
    for token in profile_tokens:
        if token in UPDATE_MARKERS:
            markers.append(token)
    return _dedupe_preserve_order(markers)


def _infer_unit_type(
    *,
    sentence_count: int,
    entity_tokens: tuple[str, ...],
    numeric_values: tuple[str, ...],
    time_markers: tuple[str, ...],
    update_markers: tuple[str, ...],
) -> str:
    if update_markers:
        return "update_fact"
    if time_markers:
        return "dated_event"
    if numeric_values:
        return "numeric_fact"
    if entity_tokens:
        return "entity_fact"
    if sentence_count > 1:
        return "sentence"
    return "utterance"


def _eligibility_labels_for_sentence(
    *,
    unit_type: str,
    numeric_values: tuple[str, ...],
    time_markers: tuple[str, ...],
    update_markers: tuple[str, ...],
    has_current_state_language: bool,
) -> tuple[str, ...]:
    labels: list[str] = []
    if unit_type in {"entity_fact", "utterance", "sentence"}:
        labels.append("direct_answer_candidate")
    if numeric_values:
        labels.append("numeric_operand")
    if time_markers:
        labels.append("temporal_anchor")
    if update_markers:
        labels.extend(["state_transition", "current_state_candidate"])
    if has_current_state_language:
        labels.append("current_state_candidate")
    return _dedupe_preserve_order(labels)


def _typed_eligibility_fields(labels: tuple[str, ...]) -> dict[str, bool]:
    label_set = {str(label).strip().lower() for label in labels if str(label).strip()}
    return {
        "is_direct_answer_candidate": "direct_answer_candidate" in label_set,
        "is_numeric_operand": "numeric_operand" in label_set,
        "is_temporal_anchor": "temporal_anchor" in label_set,
        "is_state_assertion": bool({"current_state_candidate", "state_assertion", "state_transition"} & label_set),
        "is_update_assertion": "state_transition" in label_set,
        "is_current_state_candidate": "current_state_candidate" in label_set,
        "is_attribute_value": "attribute_value" in label_set,
        "is_countable_item": "countable_item" in label_set,
        "is_generic_state_text": "generic_state_text" in label_set,
        "is_low_authority_text": "low_authority_text" in label_set,
        "is_question_or_request": "question_or_request" in label_set,
        "is_recommendation_or_advice": "recommendation_or_advice" in label_set,
        "is_instructional_quantity": "instructional_quantity" in label_set,
        "is_answer_like_quantity_statement": "answer_like_quantity_statement" in label_set,
        "has_progress_marker": "progress_marker" in label_set,
        "has_acquisition_marker": "acquisition_marker" in label_set,
        "has_consumption_or_completion_marker": "consumption_or_completion_marker" in label_set,
    }


def _eligibility_labels_from_object(memory_object: MemoryObject) -> tuple[str, ...]:
    schema_hints = memory_object.provenance.get("schema_hints", {})
    if isinstance(schema_hints, dict):
        labels = schema_hints.get("labels")
        if isinstance(labels, list):
            return _dedupe_preserve_order([str(label) for label in labels if label])
    labels: list[str] = []
    if memory_object.object_type in {"state", "attribute_fact", "preference"}:
        labels.extend(["attribute_value", "direct_answer_candidate"])
    if memory_object.object_type in {"update"}:
        labels.extend(["state_transition", "current_state_candidate"])
    if memory_object.object_type in {"dated_fact", "event"}:
        labels.append("temporal_anchor")
    if memory_object.object_type in {"numeric_fact", "comparison_fact"}:
        labels.append("numeric_operand")
    predicate_metadata = memory_object.provenance.get("predicate_metadata", {})
    if isinstance(predicate_metadata, dict):
        predicate_lemma = str(predicate_metadata.get("predicate_lemma") or "").strip().lower()
        if predicate_lemma in {"acquire", "buy", "collect", "get", "order", "pick", "purchase"}:
            labels.append("acquisition_marker")
        if predicate_lemma in {"complete", "finish", "read", "rewatch", "see", "try", "visit", "watch"}:
            labels.append("consumption_or_completion_marker")
        if predicate_lemma in {"complete", "finish", "read", "watch"}:
            labels.append("progress_marker")
    return _dedupe_preserve_order(labels)


def _build_unit(
    *,
    view: CandidateView,
    span_text: str,
    speaker: str,
    local_order: int,
    turn_index: int,
    sentence_index: int,
    sentence_count: int,
    is_sentence_split: bool,
) -> EvidenceUnit:
    span_profile = cached_build_profile(span_text)
    profile_tokens = list(span_profile.tokens)
    entity_tokens = _extract_entity_tokens(span_text)
    numeric_values = _extract_numeric_values(span_text)
    time_markers = _extract_time_markers(
        span_text,
        profile_tokens=set(span_profile.token_set),
        normalized_text=span_profile.normalized_text,
    )
    update_markers = _extract_update_markers(
        span_text,
        profile_tokens=set(span_profile.token_set),
        normalized_text=span_profile.normalized_text,
    )
    subject_tokens = _extract_subject_tokens(
        profile_tokens=profile_tokens,
        entity_tokens=entity_tokens,
        normalized_text=span_profile.normalized_text,
    )
    value_tokens = _extract_value_tokens(
        text=span_text,
        normalized_text=span_profile.normalized_text,
        numeric_values=numeric_values,
        time_markers=time_markers,
        update_markers=update_markers,
    )
    has_comparison_language = _has_comparison_language(span_profile.normalized_text)
    has_current_state_language = _has_current_state_language(span_profile.normalized_text)
    has_reference_language = _has_reference_language(span_profile.normalized_text)
    has_pronoun_language = _has_pronoun_language(profile_tokens)
    unit_type = _infer_unit_type(
        sentence_count=sentence_count,
        entity_tokens=entity_tokens,
        numeric_values=numeric_values,
        time_markers=time_markers,
        update_markers=update_markers,
    )
    rendered = span_text.strip()
    provenance = {
        "memory_id": view.memory_id,
        "parent_rank": view.rank,
        "turn_index": turn_index,
        "sentence_index": sentence_index,
        "speaker": speaker,
        "source_text": rendered,
        "memory_text": view.text,
        "render_text": rendered,
        "is_sentence_split": is_sentence_split,
    }
    eligibility_labels = _eligibility_labels_for_sentence(
        unit_type=unit_type,
        numeric_values=numeric_values,
        time_markers=time_markers,
        update_markers=update_markers,
        has_current_state_language=has_current_state_language,
    )
    eligibility_labels = _dedupe_preserve_order(
        list(eligibility_labels)
        + list(
            _count_semantic_labels(
                text=rendered,
                normalized_text=span_profile.normalized_text,
                profile_tokens=profile_tokens,
                numeric_values=numeric_values,
                speaker=speaker,
            )
        )
    )
    typed_fields = _typed_eligibility_fields(eligibility_labels)
    return EvidenceUnit(
        unit_id=f"{view.memory_id}:t{turn_index:02d}:u{local_order:02d}",
        memory_id=view.memory_id,
        parent_rank=view.rank,
        local_order=local_order,
        source_text=rendered,
        render_text=rendered,
        speaker=speaker,
        date_key=view.date_key,
        token_count=token_count(rendered),
        unit_type=unit_type,
        entity_tokens=entity_tokens,
        numeric_values=numeric_values,
        time_markers=time_markers,
        update_markers=update_markers,
        subject_tokens=subject_tokens,
        value_tokens=value_tokens,
        has_comparison_language=has_comparison_language,
        has_current_state_language=has_current_state_language,
        has_reference_language=has_reference_language,
        has_pronoun_language=has_pronoun_language,
        eligibility_labels=eligibility_labels,
        is_direct_answer_candidate=typed_fields["is_direct_answer_candidate"],
        is_numeric_operand=typed_fields["is_numeric_operand"],
        is_temporal_anchor=typed_fields["is_temporal_anchor"],
        is_state_assertion=typed_fields["is_state_assertion"],
        is_update_assertion=typed_fields["is_update_assertion"],
        is_current_state_candidate=typed_fields["is_current_state_candidate"],
        is_attribute_value=typed_fields["is_attribute_value"],
        is_countable_item=typed_fields["is_countable_item"],
        is_generic_state_text=typed_fields["is_generic_state_text"],
        is_low_authority_text=typed_fields["is_low_authority_text"],
        is_question_or_request=typed_fields["is_question_or_request"],
        is_recommendation_or_advice=typed_fields["is_recommendation_or_advice"],
        is_instructional_quantity=typed_fields["is_instructional_quantity"],
        is_answer_like_quantity_statement=typed_fields["is_answer_like_quantity_statement"],
        has_progress_marker=typed_fields["has_progress_marker"],
        has_acquisition_marker=typed_fields["has_acquisition_marker"],
        has_consumption_or_completion_marker=typed_fields["has_consumption_or_completion_marker"],
        canonical_entity_id=None,
        canonical_attribute_id=None,
        object_type=None,
        provenance=provenance,
    )


def _render_unit_text(prefix: str, speaker: str, sentence: str) -> str:
    parts: list[str] = []
    cleaned_prefix = prefix.strip()
    if cleaned_prefix:
        parts.append(cleaned_prefix)
    if speaker and speaker != "unknown":
        parts.append(f"{speaker}:")
    cleaned_sentence = sentence.strip()
    if cleaned_sentence:
        parts.append(cleaned_sentence)
    return " ".join(parts).strip()


def _unit_type_from_object_type(memory_object: MemoryObject) -> str:
    mapping = {
        "update": "update_fact",
        "event": "dated_event" if memory_object.date_key != (0, 0, 0) else "entity_fact",
        "dated_fact": "dated_event",
        "numeric_fact": "numeric_fact",
        "comparison_fact": "numeric_fact",
        "preference": "entity_fact",
        "state": "entity_fact",
        "attribute_fact": "entity_fact",
    }
    return mapping.get(memory_object.object_type, "entity_fact")


def _time_markers_from_object(memory_object: MemoryObject) -> tuple[str, ...]:
    markers: list[str] = []
    if memory_object.date_key != (0, 0, 0):
        year, month, day = memory_object.date_key
        markers.append(f"{year:04d}/{month:02d}/{day:02d}")
    normalized = normalize_text(memory_object.render_text)
    for match in DATE_RE.finditer(memory_object.render_text):
        markers.append(match.group(0))
    for match in TIME_RE.finditer(normalized):
        markers.append(match.group(0))
    for token in normalized.split():
        if token in TEMPORAL_HINTS:
            markers.append(token)
    return _dedupe_preserve_order(markers)


def _update_markers_from_object(memory_object: MemoryObject) -> tuple[str, ...]:
    markers: list[str] = []
    if memory_object.update_relation:
        markers.append(memory_object.update_relation)
    provenance_hints = memory_object.provenance.get("update_hints", [])
    if isinstance(provenance_hints, list):
        markers.extend(str(item) for item in provenance_hints)
    normalized = normalize_text(memory_object.render_text)
    token_set = set(normalized.split())
    for pattern in NEGATION_PATTERNS:
        if (" " in pattern and pattern in f" {normalized} ") or (" " not in pattern and pattern in token_set):
            markers.append(pattern)
    for token in token_set:
        if token in UPDATE_MARKERS:
            markers.append(token)
    return _dedupe_preserve_order(markers)


def _numeric_values_from_object(memory_object: MemoryObject) -> tuple[str, ...]:
    values: list[str] = []
    if memory_object.value_number is not None:
        if memory_object.value_unit == "usd":
            values.append(f"${int(memory_object.value_number) if memory_object.value_number.is_integer() else memory_object.value_number}")
        elif memory_object.value_unit == "percent":
            values.append(f"{memory_object.value_number}%")
        else:
            values.append(str(int(memory_object.value_number) if memory_object.value_number.is_integer() else memory_object.value_number))
    values.extend(list(_extract_numeric_values(memory_object.render_text)))
    return _dedupe_preserve_order(values)


def _build_unit_from_object(
    *,
    view: CandidateView,
    memory_object: MemoryObject,
    local_order: int,
) -> EvidenceUnit:
    analysis_text = memory_object.source_text or memory_object.render_text
    span_profile = cached_build_profile(analysis_text)
    profile_tokens = list(span_profile.tokens)
    entity_tokens = _dedupe_preserve_order(
        [memory_object.entity_key] if memory_object.entity_key else list(_extract_entity_tokens(analysis_text))
    )
    numeric_values = _numeric_values_from_object(memory_object)
    time_markers = _time_markers_from_object(memory_object)
    update_markers = _update_markers_from_object(memory_object)
    subject_tokens = _extract_subject_tokens(
        profile_tokens=profile_tokens,
        entity_tokens=entity_tokens,
        normalized_text=span_profile.normalized_text,
    )
    value_tokens = _extract_value_tokens(
        text=memory_object.value_text or analysis_text,
        normalized_text=span_profile.normalized_text,
        numeric_values=numeric_values,
        time_markers=time_markers,
        update_markers=update_markers,
    )
    has_comparison_language = (
        memory_object.object_type == "comparison_fact"
        or _has_comparison_language(span_profile.normalized_text)
    )
    has_current_state_language = _has_current_state_language(span_profile.normalized_text)
    has_reference_language = _has_reference_language(span_profile.normalized_text)
    has_pronoun_language = _has_pronoun_language(profile_tokens)
    unit_type = _unit_type_from_object_type(memory_object)
    eligibility_labels = _eligibility_labels_from_object(memory_object)
    eligibility_labels = _dedupe_preserve_order(
        list(eligibility_labels)
        + list(
            _count_semantic_labels(
                text=analysis_text,
                normalized_text=span_profile.normalized_text,
                profile_tokens=profile_tokens,
                numeric_values=numeric_values,
                speaker=memory_object.speaker,
            )
        )
    )
    typed_fields = _typed_eligibility_fields(eligibility_labels)

    provenance = memory_object.provenance.with_updates(
        unit_id=memory_object.object_id,
        object_id=memory_object.object_id,
        object_type=memory_object.object_type,
        memory_id=view.memory_id,
        parent_rank=view.rank,
        render_text=memory_object.render_text,
        source_text=memory_object.source_text,
        source_span_text=memory_object.source_text,
    )
    sentence_index = int(provenance.get("sentence_index", local_order + 1))
    turn_index = int(provenance.get("turn_index", 1))
    is_sentence_split = bool(provenance.get("is_sentence_split", False))

    return EvidenceUnit(
        unit_id=memory_object.object_id,
        memory_id=view.memory_id,
        parent_rank=view.rank,
        local_order=local_order,
        source_text=memory_object.source_text,
        render_text=memory_object.render_text,
        speaker=memory_object.speaker,
        date_key=memory_object.date_key,
        token_count=token_count(memory_object.render_text),
        unit_type=unit_type,
        entity_tokens=entity_tokens,
        numeric_values=numeric_values,
        time_markers=time_markers,
        update_markers=update_markers,
        subject_tokens=subject_tokens,
        value_tokens=value_tokens,
        has_comparison_language=has_comparison_language,
        has_current_state_language=has_current_state_language,
        has_reference_language=has_reference_language,
        has_pronoun_language=has_pronoun_language,
        eligibility_labels=eligibility_labels,
        is_direct_answer_candidate=typed_fields["is_direct_answer_candidate"],
        is_numeric_operand=typed_fields["is_numeric_operand"],
        is_temporal_anchor=typed_fields["is_temporal_anchor"],
        is_state_assertion=typed_fields["is_state_assertion"],
        is_update_assertion=typed_fields["is_update_assertion"],
        is_current_state_candidate=typed_fields["is_current_state_candidate"],
        is_attribute_value=typed_fields["is_attribute_value"],
        is_countable_item=typed_fields["is_countable_item"],
        is_generic_state_text=typed_fields["is_generic_state_text"],
        is_low_authority_text=typed_fields["is_low_authority_text"],
        is_question_or_request=typed_fields["is_question_or_request"],
        is_recommendation_or_advice=typed_fields["is_recommendation_or_advice"],
        is_instructional_quantity=typed_fields["is_instructional_quantity"],
        is_answer_like_quantity_statement=typed_fields["is_answer_like_quantity_statement"],
        has_progress_marker=typed_fields["has_progress_marker"],
        has_acquisition_marker=typed_fields["has_acquisition_marker"],
        has_consumption_or_completion_marker=typed_fields["has_consumption_or_completion_marker"],
        canonical_entity_id=memory_object.canonical_entity_id,
        canonical_attribute_id=memory_object.canonical_attribute_id,
        object_type=memory_object.object_type,
        provenance=provenance.with_updates(
            turn_index=turn_index,
            sentence_index=sentence_index,
            is_sentence_split=is_sentence_split,
        ),
    )


def build_evidence_units(view: CandidateView) -> list[EvidenceUnit]:
    if not str(view.text or "").strip():
        return []

    objects = build_memory_objects(view)
    return [
        _build_unit_from_object(view=view, memory_object=memory_object, local_order=local_order)
        for local_order, memory_object in enumerate(objects)
    ]


def build_evidence_units_for_views(
    views: list[CandidateView],
    prebuilt_objects: list[MemoryObject] | None = None,
) -> list[EvidenceUnit]:
    """Build evidence units from views, optionally reusing already-built memory objects.

    Pass ``prebuilt_objects`` (e.g. from the proposal object_store) to skip the
    redundant rebuild of memory objects for selected views.
    """
    from shared.perf import get_counters
    _pc = get_counters()
    _pc.evidence_unit_build_calls += 1

    ordered_views = sorted(views, key=lambda view: (view.rank, view.memory_id))
    view_ids = {view.memory_id for view in ordered_views}

    if prebuilt_objects is not None:
        by_memory_id: dict[str, list[MemoryObject]] = {}
        for obj in prebuilt_objects:
            if obj.memory_id in view_ids:
                by_memory_id.setdefault(obj.memory_id, []).append(obj)
        reused = sum(len(v) for v in by_memory_id.values())
        _pc.objects_reused_from_store += reused
        missing_ids = view_ids - set(by_memory_id)
        if missing_ids:
            missing_views = [v for v in ordered_views if v.memory_id in missing_ids]
            for obj in build_memory_objects_for_views(missing_views):
                by_memory_id.setdefault(obj.memory_id, []).append(obj)
    else:
        objects = build_memory_objects_for_views(ordered_views)
        by_memory_id = {}
        for memory_object in objects:
            by_memory_id.setdefault(memory_object.memory_id, []).append(memory_object)

    units: list[EvidenceUnit] = []
    for view in ordered_views:
        memory_objects = by_memory_id.get(view.memory_id, [])
        units.extend(
            _build_unit_from_object(view=view, memory_object=memory_object, local_order=local_order)
            for local_order, memory_object in enumerate(memory_objects)
        )
    _pc.evidence_units_created += len(units)
    return units
