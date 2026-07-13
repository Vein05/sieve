"""Family-specific rescue and fallback rules for compiler validation.

These helpers are intentionally separate from the runtime binding hub so the
binding layer can stay focused on assembly and display text, while schema and
family policy live in validation.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from evidence.schema import EvidencePlan
from retrieval.query_targets import QueryTargets

_CALENDAR_PHRASE_RE = re.compile(
    r"\b(?:on|back on)\s+(?P<value>(?:[A-Z][a-z]+(?:'\w+)?\s+Day|[A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?))\b",
    re.IGNORECASE,
)


def _extract_location_from_source(source_text: str) -> str | None:
    text = str(source_text or "")
    patterns = (
        r"\bit was (?P<span>(?:a|an|the)\s+[^,.!?;]{1,60})\b",
        r"\btrip to (?P<span>[A-Z][A-Za-z'&.-]*(?:\s+[A-Z][A-Za-z'&.-]*){0,3})\b",
        r"\bwent to (?P<span>[A-Z][A-Za-z'&.-]*(?:\s+[A-Z][A-Za-z'&.-]*){0,3})\b",
        r"\btravel(?:ed|ling)? to (?P<span>[A-Z][A-Za-z'&.-]*(?:\s+[A-Z][A-Za-z'&.-]*){0,3})\b",
        r"\bto (?P<span>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4})\b",
        r"\bvacation in (?P<span>[A-Z][A-Za-z'&.-]*(?:\s+[A-Z][A-Za-z'&.-]*){0,3})\b",
        r"\bat (?P<span>(?:the\s+)?[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,5})\b",
        r"\bin (?P<span>[A-Z][A-Za-z0-9'&.-]*(?:\s+[A-Z][A-Za-z0-9'&.-]*){0,4})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            span = str(match.group("span")).strip().strip(" .,:;!?")
            if span.lower() in {"june", "july", "august", "september", "october", "november", "december"}:
                continue
            if span.lower().startswith("the "):
                span = span[4:].strip()
            return span
    return None


def _extract_calendar_phrase_from_source(source_text: str) -> str | None:
    match = _CALENDAR_PHRASE_RE.search(str(source_text or ""))
    if not match:
        return None
    return str(match.group("value") or "").strip().strip(" .,:;!?")


def _extract_state_entity_from_source(source_text: str) -> str | None:
    normalized = str(source_text or "").lower()
    if "family trip" in normalized:
        return "family trip"
    if "trip" in normalized:
        return "trip"
    return None


def promote_support_grounding(
    *,
    plan: EvidencePlan,
    validated_bindings: dict[str, dict[str, Any] | None],
    displayed_bindings: dict[str, dict[str, Any] | None],
    invalid_slots: dict[str, str],
    binding_source_text: Any,
    is_generic_fragment: Any,
    binding_quality_score: Any,
) -> None:
    del plan, validated_bindings, displayed_bindings, invalid_slots, binding_source_text, is_generic_fragment, binding_quality_score
    # Disabled: this is a late-stage overwrite that lets support text replace the
    # main answer slot after selection has already made a weak choice. Fix the
    # upstream slot admission/ranking instead.
    return


def apply_reference_time_fallback(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    validated_bindings: dict[str, dict[str, Any] | None],
    invalid_slots: dict[str, str],
    query_has_relative_reference_clause: Any,
    question_date_binding: Any,
) -> None:
    if plan.schema_name != "RelativeTime":
        return
    if not query_has_relative_reference_clause(row):
        question_binding = question_date_binding(row)
        if question_binding is not None:
            validated_bindings["reference_time"] = question_binding
            invalid_slots.pop("reference_time", None)
        return
    if isinstance(validated_bindings.get("reference_time"), dict):
        return
    question_binding = question_date_binding(row)
    if question_binding is None:
        return
    validated_bindings["reference_time"] = question_binding
    invalid_slots.pop("reference_time", None)


def rescue_current_state_where_binding(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    displayed_bindings: dict[str, dict[str, Any] | None],
    validated_bindings: dict[str, dict[str, Any] | None],
    invalid_slots: dict[str, str],
    normalize_text: Any,
    binding_source_text: Any,
    extract_location_from_source: Any,
    extract_state_entity_from_source: Any,
    copy_binding_with_text: Any,
    is_generic_fragment: Any,
) -> None:
    del row, plan, displayed_bindings, validated_bindings, invalid_slots, normalize_text, binding_source_text, extract_location_from_source, extract_state_entity_from_source, copy_binding_with_text, is_generic_fragment
    return


def rescue_information_extraction_where_binding(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    displayed_bindings: dict[str, dict[str, Any] | None],
    validated_bindings: dict[str, dict[str, Any] | None],
    invalid_slots: dict[str, str],
    normalize_text: Any,
    binding_source_text: Any,
    copy_binding_with_text: Any,
    extract_location_from_source: Any,
) -> None:
    del row, plan, displayed_bindings, validated_bindings, invalid_slots, normalize_text, binding_source_text, copy_binding_with_text, extract_location_from_source
    return


def rescue_temporal_when_event_binding(
    *,
    row: dict[str, Any],
    plan: EvidencePlan,
    compiled_units: list[dict[str, Any]],
    displayed_bindings: dict[str, dict[str, Any] | None],
    validated_bindings: dict[str, dict[str, Any] | None],
    invalid_slots: dict[str, str],
    normalize_text: Any,
    binding_source_text: Any,
    copy_binding_with_text: Any,
    extract_calendar_phrase_from_source: Any,
) -> None:
    del row, plan, compiled_units, displayed_bindings, validated_bindings, invalid_slots, normalize_text, binding_source_text, copy_binding_with_text, extract_calendar_phrase_from_source
    return
