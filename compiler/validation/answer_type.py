"""Answer-type validation for compiler slot bindings."""

from __future__ import annotations

import re
from typing import Any

from evidence.schema import EvidencePlan
from retrieval.query_targets import QueryTargets
from ..execution.requirements import is_comparison_schema, is_count_aggregate_schema, is_direct_lookup_schema, is_numeric_aggregate_schema
from .compatibility import _entity_head_noun_compatible

_MONTH_NAME_TO_NUMBER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_TIME_LIKE_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
_DATE_LIKE_RE = re.compile(r"\b\d{4}/\d{2}/\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_WEAK_QUERY_FOCUS_TOKENS = {
    "about",
    "advice",
    "been",
    "did",
    "does",
    "had",
    "have",
    "help",
    "how",
    "initially",
    "just",
    "know",
    "many",
    "much",
    "new",
    "current",
    "currently",
    "now",
    "previous",
    "prior",
    "old",
    "still",
    "think",
    "thought",
    "thinking",
    "want",
    "wanted",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "wondering",
}
_GENERIC_TEMPORAL_ENTITY_TOKENS = {
    "decision",
    "event",
    "meeting",
    "plan",
    "trip",
    "visit",
    "workshop",
    "webinar",
}
_STRICT_COUNT_QUERY_RE = re.compile(r"^(?:how many|number of|count of)\b", re.IGNORECASE)
_COUNT_DURATION_RE = re.compile(r"\bhow\s+many\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\b", re.IGNORECASE)
_DURATION_VALUE_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|few|several|a\s+few|a\s+couple(?:\s+of)?|couple)\s+"
    r"(?:minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_COUNT_QUERY_RE = re.compile(r"\b(?:did|have|do|am|was|were)\s+i\b", re.IGNORECASE)
_FIRST_PERSON_SOURCE_RE = re.compile(r"\b(i|i'm|i’ve|i've|my|me|mine|we|our|us)\b", re.IGNORECASE)
_ADVICEY_SOURCE_RE = re.compile(
    r"\b(?:declare|remember to|look for|you may|you might|consider|recommended|traditionally|tips?|advice|helpful|should)\b",
    re.IGNORECASE,
)
_MUSIC_SERVICE_QUERY_RE = re.compile(r"\bmusic\s+streaming\s+service\b", re.IGNORECASE)
_MUSIC_DOMAIN_RE = re.compile(r"\b(?:music|song|songs|playlist|playlists|album|albums|artist|artists|listen|listening|audio|spotify)\b", re.IGNORECASE)
_VIDEO_DOMAIN_RE = re.compile(r"\b(?:watch|watching|shows?|movies?|series|content|original content|netflix)\b", re.IGNORECASE)

# Interrogative-type compatibility patterns.
# "where" answers should contain location-like content.
_LOCATION_LIKE_RE = re.compile(
    r"(?:\b(?:at|in|from|to|near|on)\s+(?:the\s+)?[A-Z])"
    r"|(?:\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b)"
    r"|(?:\b(?:park|city|town|village|store|restaurant|museum|hotel|airport|station|beach|lake|mountain"
    r"|street|avenue|road|drive|plaza|center|centre|theatre|theater|church|temple|library|gym|hospital"
    r"|clinic|school|university|college|office|building|market|mall|square|garden|zoo|stadium|arena"
    r"|trail|island|coast|valley|harbor|harbour|bridge|campus|resort|cafe|bar|pub|bakery|pharmacy"
    r"|studio|gallery|warehouse|factory|downtown|uptown|midtown)\b)"
    # Common abbreviated city/location names (2-4 uppercase letters)
    r"|(?:\b(?:SF|NYC|LA|DC|UK|US|USA)\b)",
    re.IGNORECASE,
)
# "who" answers should contain person-like content.
_PERSON_LIKE_RE = re.compile(
    r"(?:\b[A-Z][a-z]{1,15}(?:\s+[A-Z][a-z]{1,15}){0,2}\b)"
    r"|(?:\b(?:friend|sister|brother|mom|mother|dad|father|aunt|auntie|uncle|cousin|wife|husband"
    r"|partner|colleague|boss|neighbor|neighbour|roommate|classmate|trainer|instructor|doctor|therapist"
    r"|mentor|coach|professor|teacher|tutor|manager|supervisor|coworker|grandma|grandmother|grandpa"
    r"|grandfather|nephew|niece|son|daughter|fiancé|fiancee|boyfriend|girlfriend|ex)\b)"
    # Titled names: Dr. Smith, Professor Jones
    r"|(?:\b(?:Dr|Mr|Mrs|Ms|Prof|Professor|Captain|Coach)\\.?\s+[A-Z][a-z]+\b)",
    re.IGNORECASE,
)
# "when" answers should contain temporal content.
_TEMPORAL_LIKE_RE = re.compile(
    r"(?:\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b)"
    r"|(?:\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b)"
    r"|(?:\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b)"
    r"|(?:\b(?:morning|afternoon|evening|night|noon|midnight|dawn|dusk|yesterday|tomorrow|tonight"
    r"|last\s+(?:week|month|year|night)|next\s+(?:week|month|year)"
    r"|ago|recently|earlier|later|before|after)\b)"
    r"|(?:\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b)"
    r"|(?:\b\d{4}/\d{2}/\d{2}\b)"
    # Holiday names
    r"|(?:\b(?:Christmas|Thanksgiving|Easter|Halloween|New\s+Year|Valentine)\b)"
    # Ordinal dates: "the 15th", "on the 3rd"
    r"|(?:\b\d{1,2}(?:st|nd|rd|th)\b)",
    re.IGNORECASE,
)


def _attribute_compatible(expected_attribute: str, candidate_attribute: str) -> bool:
    expected = str(expected_attribute or "").strip().lower()
    candidate = str(candidate_attribute or "").strip().lower()
    if not expected or not candidate:
        return False
    if expected == candidate:
        return True
    if expected == "occupation" and candidate in {"job", "profession", "employer", "work"}:
        return True
    if expected == "name" and candidate in {"reading", "watching", "studying", "using", "employer"}:
        return True
    if expected == "brand" and candidate == "using":
        return True
    if expected in {"duration", "time"} and candidate in {"duration", "time", "amount"}:
        return True
    return False


def validate_slot_binding(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets: QueryTargets,
    slot: Any,
    binding: dict[str, Any] | None,
    focus_tokens: set[str],
    binding_source_text: Any,
    normalize_text: Any,
    is_generic_fragment: Any,
    query_expects_time_literal: Any,
    binding_value_from_session_header: Any,
    best_matching_entity: Any,
    direct_value_requires_entity_match: Any,
    binding_focus_overlap_tokens: Any,
    binding_matches_query_focus: Any,
    slot_expected_entities: Any,
    slot_entity_match: Any,
    slot_allows_descriptive_fragment: Any,
    copy_binding_with_text: Any,
    parse_numeric_text: Any,
    numeric_query_requires_money: Any,
    numeric_query_requires_distance: Any,
    numeric_query_requires_speed: Any,
    contains_math_expression: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(binding, dict):
        return None, "missing"

    display_text = str(binding.get("display_text") or binding.get("text") or "").strip()
    source_text = binding_source_text(binding)
    normalized_query = normalize_text(str(row.get("query", "")))
    normalized_text = normalize_text(display_text)
    normalized_source = normalize_text(source_text)

    if not display_text and not source_text:
        return None, "missing"

    primary_entities = tuple(entity for entity in query_targets.subject_entities if entity) or tuple(
        entity for entity in query_targets.candidate_entities if entity
    )

    if slot.slot_type == "state_anchor":
        if primary_entities:
            for entity in primary_entities:
                if entity and entity in normalized_source:
                    return binding, None
            return None, "wrong_entity"
        return binding, None

    if slot.slot_type in {"direct_value", "current_resolution", "new_state", "old_state"}:
        if _MUSIC_SERVICE_QUERY_RE.search(normalized_query):
            music_like = bool(_MUSIC_DOMAIN_RE.search(display_text) or _MUSIC_DOMAIN_RE.search(source_text))
            video_like = bool(_VIDEO_DOMAIN_RE.search(display_text) or _VIDEO_DOMAIN_RE.search(source_text))
            if not music_like or video_like:
                return None, "wrong_attribute"
        if not _entity_head_noun_compatible(plan=plan, query_targets=query_targets, source_text=source_text):
            return None, "wrong_entity"
        preferred_attribute = str(getattr(slot, "attribute_key", "") or "").strip().lower()
        schema_hints = binding.get("schema_hints") if isinstance(binding.get("schema_hints"), dict) else {}
        candidate_attribute = str(schema_hints.get("attribute_key") or "").strip().lower()
        if preferred_attribute and candidate_attribute and not _attribute_compatible(preferred_attribute, candidate_attribute):
            return None, "wrong_attribute"
        if is_generic_fragment(display_text, source_text):
            # Single-anchor and IE lookups often recover the right answer span even when the
            # extracted surface is a bit generic. Let semantic sufficiency decide those rows.
            if plan.family not in {"single_anchor", "information_extraction"}:
                return None, "generic_fragment"
        if not query_expects_time_literal(row) and binding_value_from_session_header(display_text, source_text):
            return None, "header_hijack"
        if not query_expects_time_literal(row) and (_TIME_LIKE_RE.fullmatch(display_text) or _DATE_LIKE_RE.fullmatch(display_text)):
            return None, "time_literal"
        if re.fullmatch(r"(can|what|which|who|when|where|why|how)\b.*", normalized_text):
            return None, "generic_fragment"
        if normalized_query.startswith("how long"):
            duration_value = str(binding.get("duration_value") or "").strip()
            duration_like = bool(
                duration_value
                or _DURATION_VALUE_RE.search(display_text)
                or _DURATION_VALUE_RE.search(source_text)
            )
            if not duration_like:
                return None, "numeric_pollution"
        # Interrogative-type compatibility: reject spans that clearly don't
        # match the question type.  Only applied to direct_value slots in
        # extraction/single-anchor families where the plan carries an
        # interrogative, and only when the span has *no* signal for the
        # expected type.  This prevents the compiler from deterministically
        # returning a location for a duration question or vice versa.
        if slot.slot_type == "direct_value" and plan.family in {"single_anchor", "information_extraction"}:
            interrogative = str((plan.plan_metadata or {}).get("interrogative") or "").strip().lower()
            if interrogative == "where":
                if not _LOCATION_LIKE_RE.search(display_text) and not _LOCATION_LIKE_RE.search(source_text):
                    return None, "answer_type_mismatch"
            elif interrogative == "who":
                if not _PERSON_LIKE_RE.search(display_text) and not _PERSON_LIKE_RE.search(source_text):
                    return None, "answer_type_mismatch"
            elif interrogative == "when":
                if not _TEMPORAL_LIKE_RE.search(display_text) and not _TEMPORAL_LIKE_RE.search(source_text):
                    return None, "answer_type_mismatch"
        if (
            any(marker in normalized_source for marker in ("recommend", "suggest", "advice", "top recommendation"))
            and len(normalized_source.split()) > 14
        ):
            candidate_match = bool(
                primary_entities
                and best_matching_entity(source_text, primary_entities)
            )
            if not candidate_match:
                return None, "preference_summary"
        has_update_language = any(
            marker in normalized_source
            for marker in (" now ", " current ", " switched ", " changed ", " instead ", " currently ")
        )
        overlap_tokens = binding_focus_overlap_tokens(binding, focus_tokens)
        strong_overlap = overlap_tokens - _WEAK_QUERY_FOCUS_TOKENS
        if not has_update_language:
            if not overlap_tokens:
                return None, "wrong_attribute"
        if direct_value_requires_entity_match(row=row, plan=plan, query_targets=query_targets):
            if primary_entities and not best_matching_entity(source_text, primary_entities):
                return None, "wrong_entity"
        if query_targets.asks_for_recall_support and len(normalized_source.split()) > 18:
            candidate_match = bool(
                primary_entities
                and best_matching_entity(source_text, primary_entities)
            )
            if not candidate_match:
                return None, "preference_summary"
        if slot.slot_type == "old_state" and any(
            marker in normalized_source for marker in (" switched ", " currently ", " current ", " now ", " new ")
        ):
            return None, "wrong_attribute"
        if slot.slot_type == "new_state" and any(
            marker in normalized_source for marker in (" used to ", " previous ", " prior ", " former ", " old ")
        ):
            return None, "wrong_attribute"

    if slot.slot_type == "temporal_event":
        expected_entities = slot_expected_entities(slot, query_targets)
        overlap_tokens = binding_focus_overlap_tokens(binding, focus_tokens)
        strong_overlap = overlap_tokens - _WEAK_QUERY_FOCUS_TOKENS
        if expected_entities:
            matched = slot_entity_match(slot, source_text, query_targets)
            if not matched:
                # Stricter gate (Fix 3): reject when the expected entity is
                # NOT present in the source — focus overlap alone is not
                # enough because an adjacent fragment may provide spurious overlap.
                expected_tokens = {
                    token
                    for token in normalize_text(expected_entities[0]).split()
                    if token and token not in _WEAK_QUERY_FOCUS_TOKENS
                }
                if len(expected_tokens) >= 2 or not (strong_overlap - expected_tokens):
                    return None, "wrong_entity"
            else:
                # Guard against generic temporal tokens used as entity keys.
                expected_tokens = {
                    token
                    for token in normalize_text(expected_entities[0]).split()
                    if token and token not in _WEAK_QUERY_FOCUS_TOKENS
                }
                if len(expected_tokens) == 1 and expected_tokens <= _GENERIC_TEMPORAL_ENTITY_TOKENS:
                    if not (strong_overlap - expected_tokens):
                        return None, "wrong_entity"
        has_calendar = bool(
            re.search(
                r"(day|january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}(?::\d{2})?\s*(am|pm))",
                normalized_text,
            )
        )
        if slot.slot_name == "event" and has_calendar:
            return binding, None
        if not slot_allows_descriptive_fragment(str(slot.slot_type)) and is_generic_fragment(display_text, source_text):
            return None, "generic_fragment"
        if not binding_matches_query_focus(binding, focus_tokens) and not has_calendar:
            return None, "wrong_attribute"
        if any(month in normalized_text for month in _MONTH_NAME_TO_NUMBER):
            if " when " not in f" {normalized_query} " and " date " not in f" {normalized_query} ":
                return None, "wrong_attribute"

    if slot.slot_type == "temporal_time":
        expected_entities = slot_expected_entities(slot, query_targets)
        overlap_tokens = binding_focus_overlap_tokens(binding, focus_tokens)
        strong_overlap = overlap_tokens - _WEAK_QUERY_FOCUS_TOKENS
        if expected_entities:
            matched = slot_entity_match(slot, source_text, query_targets)
            if not matched:
                # Stricter gate (Fix 3): entity mismatch blocks binding except
                # when the source clearly has a calendar date.
                has_date = bool(re.search(r"\b\d{4}/\d{2}/\d{2}\b", source_text))
                if not has_date:
                    return None, "wrong_entity"
            else:
                expected_tokens = {
                    token
                    for token in normalize_text(expected_entities[0]).split()
                    if token and token not in _WEAK_QUERY_FOCUS_TOKENS
                }
                if len(expected_tokens) == 1 and expected_tokens <= _GENERIC_TEMPORAL_ENTITY_TOKENS:
                    if not (strong_overlap - expected_tokens):
                        return None, "wrong_entity"
        if is_generic_fragment(display_text, source_text):
            return None, "generic_fragment"

    if slot.slot_type == "ordering_event":
        matched_entity = slot_entity_match(slot, source_text, query_targets)
        if slot_expected_entities(slot, query_targets) and not matched_entity:
            if not binding_focus_overlap_tokens(binding, focus_tokens):
                return None, "wrong_entity"
        if matched_entity:
            return copy_binding_with_text(binding, matched_entity), None
        if not slot_allows_descriptive_fragment(str(slot.slot_type)) and is_generic_fragment(display_text, source_text):
            return None, "generic_fragment"
        if not binding_matches_query_focus(binding, focus_tokens):
            return None, "wrong_attribute"
        if any(month in normalized_text for month in _MONTH_NAME_TO_NUMBER) and " date " not in f" {normalized_query} ":
            return None, "wrong_attribute"

    if slot.slot_type == "numeric_operand":
        if normalized_query.startswith("how long"):
            duration_value = str(binding.get("duration_value") or "").strip()
            if duration_value:
                return binding, None
        parsed = parse_numeric_text(display_text) or parse_numeric_text(source_text)
        if parsed is None:
            if not slot.is_variadic and not (is_numeric_aggregate_schema(plan) or is_comparison_schema(plan)):
                return None, "numeric_pollution"
        if numeric_query_requires_money(row):
            if "$" not in source_text and not any(token in normalized_source for token in (" cost ", " price ", " spent ", " save ", " worth ", " dollar ")):
                if primary_entities:
                    return None, "numeric_pollution"
        if numeric_query_requires_distance(row):
            if not any(token in normalized_source for token in (" mile", " miles", " km", " kilometer")):
                if primary_entities:
                    return None, "numeric_pollution"
        if numeric_query_requires_speed(row):
            if not any(
                token in normalized_source
                for token in (" mbps", " gbps", " kbps", " mph", " km/h", " speed ", " download ", " upload ", " internet plan ")
            ):
                return None, "numeric_pollution"
        if contains_math_expression(source_text) and query_targets.candidate_entities:
            if not best_matching_entity(source_text, primary_entities):
                return None, "numeric_pollution"
    if slot.slot_type == "comparison_operand":
        parsed = parse_numeric_text(display_text) or parse_numeric_text(source_text)
        has_comparison_language = bool(
            re.search(r"\b(more|less|higher|lower|cheaper|pricier)\b", normalized_source)
        )
        if parsed is None and not has_comparison_language:
            return None, "numeric_pollution"
        if slot_expected_entities(slot, query_targets):
            if not slot_entity_match(slot, source_text, query_targets):
                return None, "wrong_entity"
        if contains_math_expression(source_text) and not has_comparison_language:
            if primary_entities:
                return None, "numeric_pollution"

    if slot.slot_type in {"count_item", "count_evidence"}:
        overlap_tokens = binding_focus_overlap_tokens(binding, focus_tokens)
        if (
            is_count_aggregate_schema(plan)
            and _STRICT_COUNT_QUERY_RE.search(normalized_query)
            and _FIRST_PERSON_COUNT_QUERY_RE.search(normalized_query)
            and not _FIRST_PERSON_SOURCE_RE.search(source_text)
            and _ADVICEY_SOURCE_RE.search(normalized_source)
        ):
            return None, "wrong_attribute"
        if (
            is_count_aggregate_schema(plan)
            and len(tuple(entity for entity in query_targets.subject_entities if entity)) >= 2
            and len(overlap_tokens - _WEAK_QUERY_FOCUS_TOKENS) < 2
            and not _FIRST_PERSON_SOURCE_RE.search(source_text)
        ):
            return None, "wrong_attribute"
        if primary_entities and not best_matching_entity(source_text, primary_entities):
            return None, "wrong_entity"

    if slot.slot_type == "support":
        if (
            is_count_aggregate_schema(plan)
            and _STRICT_COUNT_QUERY_RE.search(normalized_query)
            and _FIRST_PERSON_COUNT_QUERY_RE.search(normalized_query)
            and not _FIRST_PERSON_SOURCE_RE.search(source_text)
            and _ADVICEY_SOURCE_RE.search(normalized_source)
        ):
            return None, "wrong_attribute"
        if (
            is_count_aggregate_schema(plan)
            and _STRICT_COUNT_QUERY_RE.search(normalized_query)
            and not _COUNT_DURATION_RE.search(normalized_query)
            and primary_entities
            and not best_matching_entity(source_text, primary_entities)
        ):
            return None, "wrong_entity"
        if not binding_matches_query_focus(binding, focus_tokens):
            return None, "wrong_attribute"

    if slot.slot_type == "reference_time":
        return binding, None

    return binding, None
