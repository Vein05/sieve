"""Render compiled evidence into compact and structured packages."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
import re
from typing import Any

from shared.constants import QUERY_STOP_WORDS


def _parse_date_flexible(text: str) -> date | None:
    """Parse a date string accepting both YYYY/MM/DD and YYYY/M/D formats."""
    m = re.match(r"(\d{4})/(\d{1,2})/(\d{1,2})", str(text or "").strip())
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def _calendar_month_diff(d1: date, d2: date) -> float:
    """Compute the calendar month difference between two dates.

    Returns a float — e.g. Jan 15 → Mar 15 = 2.0 months, Jan 15 → Mar 1 ≈ 1.5.
    """
    earlier, later = (d1, d2) if d1 <= d2 else (d2, d1)
    whole_months = (later.year - earlier.year) * 12 + (later.month - earlier.month)
    # Fractional part from remaining days
    if later.day < earlier.day:
        whole_months -= 1
        # days into the partial month
        from calendar import monthrange
        _, prev_month_days = monthrange(later.year, later.month - 1 if later.month > 1 else 12)
        frac = (prev_month_days - earlier.day + later.day) / prev_month_days
    else:
        from calendar import monthrange
        _, month_days = monthrange(earlier.year, earlier.month)
        frac = (later.day - earlier.day) / month_days if month_days else 0
    return whole_months + frac

from .schema import EvidencePlan

_FIRST_PERSON_RE = re.compile(r"\b(?:i|i'm|i've|i'd|i'll|me|my|mine|we|we're|we've|our|ours)\b", re.IGNORECASE)
_GENERIC_ASSISTANT_RE_RENDER = re.compile(
    r"\b(?:can you|could you|do you have|here are|recommend|suggest|tips?|consider|for example|try to|make sure|remember to)\b",
    re.IGNORECASE,
)

_AGGREGATE_SCHEMA_NAMES_RENDER = frozenset({
    "CountLookup", "CountDistinctItems", "CountEvents",
    "SumOperands", "AverageAggregate", "DeltaAggregate",
    "ExtremumSelection", "PercentageAggregate", "Comparison",
    "Aggregate",
})
_COUNT_SCHEMA_NAMES_RENDER = frozenset({"CountLookup", "CountDistinctItems", "CountEvents"})


def _is_user_first_person(text: str) -> bool:
    return bool(_FIRST_PERSON_RE.search(text))


def _is_generic_assistant(text: str) -> bool:
    return bool(_GENERIC_ASSISTANT_RE_RENDER.search(text))


def _dedupe_evidence_lines(compiled_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    evidence_lines: list[dict[str, Any]] = []
    for unit in compiled_units:
        if not isinstance(unit, dict):
            continue
        memory_id = str(unit.get("memory_id") or "").strip()
        text = str(unit.get("render_text") or "").strip()
        key = (memory_id, text)
        if not text or key in seen:
            continue
        if _is_object_rendered_text(text):
            continue
        seen.add(key)
        evidence_lines.append(
            {
                "memory_id": unit.get("memory_id"),
                "speaker": unit.get("speaker"),
                "text": unit.get("render_text"),
                "unit_id": unit.get("unit_id"),
            }
        )
    return evidence_lines

_NUMERIC_SLOT_NAMES = frozenset({
    "numeric_value",
    "operand_1",
    "operand_2",
    "left_operand",
    "right_operand",
})
_TIME_SLOT_NAMES = frozenset({"time_a", "time_b", "reference_time"})
_TEMPORAL_DIRECT_SLOT_NAMES = frozenset({"event"})
_QUOTED_VALUE_RE = re.compile(r'["“](?P<value>[^"\n]{1,80})["”]')
_CALLED_VALUE_RE = re.compile(r"\b(?:called|named)\s+(?P<value>[A-Z0-9][^,.!?;]{1,80})", re.IGNORECASE)
_CALENDAR_VALUE_RE = re.compile(
    r"\b(?:on|back on)\s+(?P<value>(?:[A-Z][a-z]+(?:'\w+)?\s+Day|[A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?))\b"
)
_CAPITALIZED_PHRASE_RE = re.compile(
    r"\b([A-Z0-9][A-Za-z0-9&/-]*(?:\s+(?:[A-Z0-9][A-Za-z0-9&/-]*|of|the|and|for|to|in|at|from|with|by)){0,8})\b"
)
_TAIL_VALUE_RE = re.compile(
    r"\b(?:is|was|are|were|uses|use|using|contains|holds|cost|costs|spent|currently|current|now uses|now use)\s+"
    r"(?P<value>[^,.!?;]{1,100})",
    re.IGNORECASE,
)
_NUMERIC_VALUE_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_PERCENT_VALUE_RE = re.compile(r"\b\d+(?:\.\d+)?%")
_RATIO_VALUE_RE = re.compile(r"\b(?P<value>\d+(?::\d+)+(?:\s+ratio)?(?:\s+with\s+[^,.!?;]{1,50})?)", re.IGNORECASE)
_TIME_VALUE_RE = re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", re.IGNORECASE)
_SIZE_VALUE_RE = re.compile(r"\b\d+(?:\.\d+)?(?:-|\s)?inch\b(?:\s+\w+)?", re.IGNORECASE)
_PLATFORM_VALUE_RE = re.compile(r"\b(?:like|on)\s+(?P<value>TikTok|Instagram|Facebook|YouTube|LinkedIn|Twitter|X|Reddit|Pinterest|Snapchat)\b", re.IGNORECASE)
_ORDINAL_VALUE_RE = re.compile(r"\b(\d{1,3})(?:st|nd|rd|th)\b", re.IGNORECASE)
_LOCATION_VALUE_RE = re.compile(
    r"\b(?:at|in|from|on)\s+(?:the\s+)?(?P<value>[A-Z0-9][A-Za-z0-9&/-]*(?:\s+(?:[A-Z0-9][A-Za-z0-9&/-]*|of|the|and|for|to|in|at)){0,6}|[^,.!?;]{1,60})",
    re.IGNORECASE,
)
_DEGREE_VALUE_RE = re.compile(r"\bdegree\s+in\s+(?P<value>[A-Z][^,.!?;]{1,60})", re.IGNORECASE)
_CERTIFICATION_VALUE_RE = re.compile(
    r"\bcertification in\s+(?P<value>[A-Z][^,.!?;]{1,60}?)(?:,\s+which\b|,|$)",
    re.IGNORECASE,
)
_FAVORITE_VALUE_RE = re.compile(
    r"\bmy favorite\s+(?P<value>[^,.!?;]{1,80}?)(?:\.|,|$)",
    re.IGNORECASE,
)
_USED_TO_BE_RE = re.compile(r"\bused to be\s+(?P<value>[^,.!?;]{1,80})", re.IGNORECASE)
_ROLE_AS_RE = re.compile(
    r"\b(?:previous|prior)\s+role\s+as\s+(?P<value>[^,.!?;]{1,80}?)(?:\s+and\s+i['’]m\b|$)",
    re.IGNORECASE,
)
_WORKED_AS_RE = re.compile(
    r"\bworked\s+as\s+(?P<value>[^,.!?;]{1,80}?)(?:\s+and\s+i['’]m\b|$)",
    re.IGNORECASE,
)
_MIX_VALUE_RE = re.compile(
    r"\b(?:mix(?:ed)?\s+ethnicity\s*[-:]\s*|mix\s+of\s+)(?P<value>[^,.!?;]{1,80}?)(?:\s+-\s+|$)",
    re.IGNORECASE,
)
_CASE_OF_RE = re.compile(r"\bcase of\s+(?P<value>[^,.!?;]{1,60}?)(?:\s+that\b|$)", re.IGNORECASE)
_DURATION_VALUE_RE = re.compile(
    r"\b(?:for|spent|takes?|took|lasted|around|about|approximately)?\s*(?P<value>(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|few|several)\s+"
    r"(?:days?|weeks?|months?|years?|hours?|minutes?))\b",
    re.IGNORECASE,
)
_PURCHASE_VALUE_RE = re.compile(
    r"\b(?:got|bought|purchased|ordered|received|gotten)\b(?:\s+(?:her|him|them|my\s+\w+))?\s+(?P<value>[^,!?;]{1,80}?)(?:\s+from\b|,|\.|$)",
    re.IGNORECASE,
)
_GIFT_FROM_PERSON_RE = re.compile(
    r"\b(?:gift|present|surprise)\s+from\s+(?P<value>[^,.!?;]{1,40}?)(?=\s+(?:last|this|next|for|and)\b|[,.!?;]|$)",
    re.IGNORECASE,
)
_GIVEN_BY_PERSON_RE = re.compile(
    r"\b(?:gave|given|gifted|bought|purchased)\s+(?:me|us)\b.*?\bby\s+(?P<value>[^,.!?;]{1,40}?)(?=\s+(?:last|this|next|for|and)\b|[,.!?;]|$)",
    re.IGNORECASE,
)
_NUMERIC_WITH_UNIT_RE = re.compile(
    r"\$?\d[\d,]*(?:\.\d+)?(?:\s?(?:gb|mb|kb|tb|gbps|mbps|kbps|mph|km/h|hours?|minutes?|days?|weeks?|months?|years?|%))?",
    re.IGNORECASE,
)
_CURRENTLY_AT_RE = re.compile(r"\b(?:currently|current(?:ly)?)\s+(?:at|with)\s+(?P<value>[A-Z][^,.!?;]{1,60})", re.IGNORECASE)
_WORKING_AT_RE = re.compile(r"\bworking\s+at\s+(?P<value>[A-Z][^,.!?;]{1,60})", re.IGNORECASE)
_PERSONAL_BEST_RE = re.compile(
    r"\bpersonal best time of\s+(?P<value>\d{1,2}\s+minutes?(?:\s+and\s+\d{1,2}\s+seconds?)?)",
    re.IGNORECASE,
)

_LEADING_VALUE_WORDS = frozenset({"a", "an", "the"})
_PREFIX_RE = re.compile(r"^\s*\d{4}/\d{2}/\d{2}(?:\s+session)?\s*(?:\|\s*)?", re.IGNORECASE)
_SESSION_HEADER_RE = re.compile(
    r"^\s*(?P<header>\d{4}/\d{2}/\d{2}\s+\([A-Za-z]{3}\)\s+\d{1,2}:\d{2}\s+session\s+\S+\s*\|?\s*(?:user|assistant|system)\s*:)\s*(?P<body>.*)$",
    re.IGNORECASE | re.DOTALL,
)


def _source_text(unit: dict[str, Any] | None, payload: dict[str, Any]) -> str:
    if isinstance(unit, dict):
        provenance = unit.get("provenance")
        if isinstance(provenance, Mapping):
            text = str(provenance.get("source_text") or "").strip()
            if text:
                return text
            memory_text = str(provenance.get("memory_text") or "").strip()
            if memory_text:
                return memory_text
    return str(payload.get("text") or "").strip()


def _clean_value_text(value: str) -> str:
    cleaned = str(value or "").strip().strip(" .,:;")
    while True:
        parts = cleaned.split(None, 1)
        if len(parts) <= 1 or parts[0].lower() not in _LEADING_VALUE_WORDS:
            break
        cleaned = parts[1].strip()
    return cleaned.strip(" .,:;")


def _split_session_header(text: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    if not raw:
        return "", ""
    match = _SESSION_HEADER_RE.match(raw)
    if not match:
        return "", raw
    return str(match.group("header") or "").strip(), str(match.group("body") or "").strip()


def _strip_prefix(text: str) -> str:
    stripped = _PREFIX_RE.sub("", str(text or "").strip())
    if ":" in stripped:
        head, tail = stripped.split(":", 1)
        if len(head.split()) <= 3:
            stripped = tail.strip()
    return stripped.strip(" .,:;")


def _interrogative_preferred_numeric(source_text: str, interrogative: str) -> str | None:
    """Pick the right number from a multi-number source using the interrogative.

    "how much" / price questions → prefer currency ($) spans.
    "how many" / count questions → prefer plain counts (no $ prefix).
    "how long" → prefer duration spans (handled upstream, but catch here too).
    """
    if not source_text or not interrogative:
        return None
    lowered = source_text.lower()
    if interrogative in {"how_much", "what"} and "$" in source_text:
        # Price question with currency in source — prefer $ amount
        currency_match = re.search(r"\$\s*\d[\d,]*(?:\.\d+)?", source_text)
        if currency_match:
            return currency_match.group(0).strip()
    if interrogative in {"how_much", "what"} and any(
        marker in lowered for marker in ("price", "cost", "paid", "spent", "each", "per ")
    ):
        # Price context without $ — prefer number near price words
        price_match = re.search(
            r"(?:price|cost|paid|spent|each|per\s+\w+)\s+(?:is\s+|was\s+|of\s+)?(?P<value>\$?\d[\d,]*(?:\.\d+)?)",
            source_text,
            re.IGNORECASE,
        )
        if price_match:
            return price_match.group("value").strip()
        # Also try number immediately before "each"/"per"
        before_each = re.search(r"(?P<value>\$?\d[\d,]*(?:\.\d+)?)\s+(?:each|per\b)", source_text, re.IGNORECASE)
        if before_each:
            return before_each.group("value").strip()
    return None


def _extract_entity_adjacent_numeric_value(source_text: str, entity_key: str) -> str | None:
    lowered = source_text.lower()
    num_pattern = _NUMERIC_WITH_UNIT_RE.pattern
    patterns = []
    if entity_key:
        escaped = re.escape(entity_key)
        patterns.extend(
            [
                rf"\b{escaped}\b(?:\s+\w+){{0,6}}\s+to\s+(?P<value>{num_pattern})",
                rf"\b{escaped}\b(?:\s+\w+){{0,6}}\s+(?:is|was|at|of)\s+(?P<value>{num_pattern})",
            ]
        )
    if "upgrade" in lowered:
        patterns.append(rf"\bupgrade(?:d)?(?:\s+\w+){{0,6}}\s+to\s+(?P<value>{num_pattern})")
    if "internet plan" in lowered or "speed" in lowered:
        patterns.extend(
            [
                rf"\binternet\s+plan(?:\s+\w+){{0,4}}\s+(?:is|was|at)\s+(?P<value>{num_pattern})",
                rf"\bspeed(?:\s+\w+){{0,4}}\s+(?:is|was|at)\s+(?P<value>{num_pattern})",
            ]
        )
    for pattern in patterns:
        match = re.search(pattern, source_text, re.IGNORECASE)
        if not match:
            continue
        value = _clean_value_text(str(match.group("value") or ""))
        if value:
            return value
    return None



def display_slot_text(
    slot,
    payload: dict[str, Any],
    unit: dict[str, Any] | None,
    plan_metadata: dict[str, Any] | None = None,
) -> str:
    raw_source_text = _source_text(unit, payload)
    if not raw_source_text:
        return str(payload.get("text") or "").strip()
    _, source_text = _split_session_header(raw_source_text)
    if not source_text:
        source_text = raw_source_text
    slot_name = str(slot.slot_name)
    entity_key = str(slot.entity_key or "").strip().lower()
    provenance = unit.get("provenance") if isinstance(unit, dict) else None
    value_source = dict(payload)
    if isinstance(provenance, Mapping):
        value_source.update({key: value for key, value in provenance.items() if value is not None and value != ""})
    schema_hints = provenance.get("schema_hints") if isinstance(provenance, Mapping) else {}
    hint_answer_type = str(schema_hints.get("answer_type") or "").strip().lower() if isinstance(schema_hints, Mapping) else ""
    object_type = str((provenance or {}).get("object_type") or payload.get("object_type") or "").strip().lower()

    interrogative = str((plan_metadata or {}).get("interrogative") or "").strip().lower()
    is_direct_slot = str(slot.slot_type) in {"direct_value", "current_resolution", "new_state", "old_state"}

    extracted_value_text = str(value_source.get("extracted_value_text") or "").strip() if isinstance(value_source, Mapping) else ""
    if is_direct_slot:
        # Prefer builder-provided typed values and compact payload text. The
        # renderer should not re-extract the answer from a whole sentence when
        # upstream object construction has already produced a candidate value.
        if interrogative == "where":
            place_val = str(value_source.get("place_value") or "").strip()
            if place_val:
                return place_val
        if interrogative == "when":
            time_val = str(value_source.get("time_value") or "").strip()
            if time_val:
                return time_val
        if interrogative == "how_long" or (
            interrogative == "how_many"
            and str(value_source.get("duration_value") or "").strip()
        ):
            dur_val = str(value_source.get("duration_value") or "").strip()
            if dur_val:
                return dur_val
        if extracted_value_text:
            return extracted_value_text
        payload_text = _clean_value_text(str(payload.get("text") or "").strip())
        if payload_text and payload_text != _clean_value_text(source_text) and len(payload_text.split()) <= 12:
            return payload_text
        if hint_answer_type in {"percent", "number"} or object_type == "numeric_fact":
            ratio_match = _RATIO_VALUE_RE.search(source_text)
            if ratio_match:
                return _clean_value_text(ratio_match.group("value"))
            duration_match = _DURATION_VALUE_RE.search(source_text)
            if duration_match:
                return _clean_value_text(duration_match.group("value"))
            percent_match = _PERCENT_VALUE_RE.search(source_text)
            if percent_match:
                return percent_match.group(0).strip()
            size_match = _SIZE_VALUE_RE.search(source_text)
            if size_match:
                return size_match.group(0).strip()
            numeric_matches = list(_NUMERIC_VALUE_RE.finditer(source_text))
            if numeric_matches:
                return numeric_matches[-1].group(0).strip().rstrip(".,;:")
        # Interrogative-conditioned extraction: before falling back to the full
        # source text, try to extract the answer type implied by the question.
        if interrogative == "where":
            loc_matches = list(_LOCATION_VALUE_RE.finditer(source_text))
            if loc_matches:
                loc_val = _clean_value_text(loc_matches[-1].group("value"))
                if loc_val and loc_val.lower() not in {"the", "a", "an", "i", "my"}:
                    return loc_val
        if interrogative in {"how_much", "how_many"} or (
            interrogative == "what" and any(
                marker in source_text.lower()
                for marker in ("$", "price", "cost", "paid", "spent", "each")
            )
        ):
            currency_match = re.search(r"\$\s*\d[\d,]*(?:\.\d+)?", source_text)
            if currency_match:
                return currency_match.group(0).strip()
        if interrogative == "who":
            # Prefer capitalized person-like phrases
            cap_matches = [
                m.group(1).strip()
                for m in _CAPITALIZED_PHRASE_RE.finditer(source_text)
                if m.group(1).strip().lower() not in {
                    "i", "session", "user", "assistant", "mon", "tue", "wed",
                    "thu", "fri", "sat", "sun",
                }
            ]
            if cap_matches:
                return _clean_value_text(max(cap_matches, key=len))
        stripped = _strip_prefix(source_text)
        return _clean_value_text(stripped or source_text)

    if str(slot.slot_type) in {"count_item", "count_evidence"}:
        stripped = _strip_prefix(source_text)
        return stripped or source_text

    if slot_name in _NUMERIC_SLOT_NAMES or slot_name.startswith("operand_"):
        # Try interrogative-aware disambiguation first (e.g. prefer $ for price questions)
        interrog_value = _interrogative_preferred_numeric(source_text, interrogative)
        if interrog_value:
            return interrog_value
        adjacent_value = _extract_entity_adjacent_numeric_value(source_text, entity_key)
        if adjacent_value:
            return adjacent_value
        payload_text = str(payload.get("text") or "").strip()
        if payload_text and (
            _NUMERIC_VALUE_RE.fullmatch(payload_text.rstrip(".,;:"))
            or _DURATION_VALUE_RE.fullmatch(payload_text.rstrip(".,;:"))
        ):
            return payload_text.rstrip(".,;:")
        duration_match = _DURATION_VALUE_RE.search(source_text)
        if duration_match:
            return duration_match.group("value").strip()
        numeric_matches = list(_NUMERIC_VALUE_RE.finditer(source_text))
        if numeric_matches:
            return numeric_matches[-1].group(0).strip().rstrip(".,;:")
        numeric_values = unit.get("numeric_values") if isinstance(unit, dict) else None
        if isinstance(numeric_values, list) and numeric_values:
            return str(numeric_values[-1]).strip().rstrip(".,;:")

    if slot_name in _TIME_SLOT_NAMES:
        time_markers = unit.get("time_markers") if isinstance(unit, dict) else None
        if isinstance(time_markers, list) and time_markers:
            return str(time_markers[0]).strip()
        date_value = payload.get("date")
        if date_value:
            return str(date_value)

    if slot_name in _TEMPORAL_DIRECT_SLOT_NAMES and str(slot.slot_type) == "temporal_event":
        stripped = _strip_prefix(source_text)
        calendar_match = _CALENDAR_VALUE_RE.search(stripped) or _CALENDAR_VALUE_RE.search(source_text)
        if calendar_match:
            return _clean_value_text(calendar_match.group("value"))
        if stripped and stripped != source_text:
            return stripped

    for pattern in (
        _GIFT_FROM_PERSON_RE,
        _GIVEN_BY_PERSON_RE,
        _QUOTED_VALUE_RE,
        _CALLED_VALUE_RE,
        _PLATFORM_VALUE_RE,
        _CURRENTLY_AT_RE,
        _WORKING_AT_RE,
        _PERSONAL_BEST_RE,
        _CERTIFICATION_VALUE_RE,
        _FAVORITE_VALUE_RE,
        _USED_TO_BE_RE,
        _ROLE_AS_RE,
        _WORKED_AS_RE,
        _MIX_VALUE_RE,
        _CASE_OF_RE,
        _DURATION_VALUE_RE,
        _RATIO_VALUE_RE,
    ):
        match = pattern.search(source_text)
        if match:
            value = _clean_value_text(match.group("value"))
            if value:
                return value

    percent_match = _PERCENT_VALUE_RE.search(source_text)
    if percent_match:
        return percent_match.group(0).strip()

    time_match = _TIME_VALUE_RE.search(source_text)
    if time_match and any(token in source_text.lower() for token in (" by ", " at ", " stop", " stopping ", " time ")):
        return time_match.group(0).strip()

    size_match = _SIZE_VALUE_RE.search(source_text)
    if size_match:
        return size_match.group(0).strip()

    if entity_key == "where":
        matches = list(_LOCATION_VALUE_RE.finditer(source_text))
        if matches:
            value = _clean_value_text(matches[-1].group("value"))
            if value:
                return value

    for pattern in (_DEGREE_VALUE_RE, _PURCHASE_VALUE_RE):
        match = pattern.search(source_text)
        if match:
            value = _clean_value_text(match.group("value"))
            if value:
                return value

    ordinal_match = _ORDINAL_VALUE_RE.search(source_text)
    if ordinal_match and "birthday" in source_text.lower():
        return ordinal_match.group(1).strip()

    match = _TAIL_VALUE_RE.search(source_text)
    if match:
        value = _clean_value_text(match.group("value"))
        if value:
            return value

    if slot.slot_type in {"state_anchor", "temporal_event"} or "where" in str(getattr(slot, "slot_name", "")).lower():
        location_match = re.search(r"\b(?:at|in|to|from|near|located at|inside of|outside|by)\s+((?:[A-Z][a-z0-9\-']+(?:\s+(?:of|the|at|in|for|and|to)\s+)?){1,5}|[a-z0-9\-']+(?:\s+[a-z0-9\-']+){0,3})", source_text)
        if location_match:
            loc = location_match.group(1).strip()
            if loc.lower() not in {"the", "a", "an", "i", "my", "me", "you", "them"}:
                return _clean_value_text(loc)

    cap_matches = [
        match.group(1).strip()
        for match in _CAPITALIZED_PHRASE_RE.finditer(source_text)
        if match.group(1).strip().lower() not in {"i", "session", "user", "assistant", "mon", "tue", "wed", "thu", "fri", "sat", "sun", "for", "on", "at", "in", "with", "from", "to", "how", "what", "which", "who", "why"}
    ]
    if cap_matches:
        return _clean_value_text(max(cap_matches, key=len))

    return source_text

_DATE_RE = re.compile(r"(\d{4}/\d{2}/\d{2})")


_SESSION_DATE_RE = re.compile(
    r"^(\d{4}/\d{2}/\d{2})\s+\([A-Za-z]{3}\)\s+\d{1,2}:\d{2}\s+session\s+\S+",
    re.IGNORECASE,
)


def _resolve_event_date(line_text: str) -> date | None:
    """Resolve the actual event date from an evidence line.

    Uses the compiler's temporal runtime to handle relative expressions
    ("yesterday", "last Monday", "February 5th", "two weeks ago") anchored
    to the line's session date.  This is Stage-3 temporal grounding — the
    compiler resolves expressions so the reader doesn't have to.
    """
    from compiler.runtime.temporal import _extract_event_date

    text = str(line_text or "").strip()
    if not text:
        return None

    # Extract the session date as anchor
    session_match = _SESSION_DATE_RE.match(text)
    if not session_match:
        return None
    try:
        parts = session_match.group(1).split("/")
        anchor = date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, IndexError):
        return None

    # Strip the session header to get the utterance body
    body = re.sub(
        r"^\d{4}/\d{2}/\d{2}(?:\s+\([A-Za-z]{3}\)\s+\d{1,2}:\d{2})?\s+session\s+\S+\s*\|?\s*(?:user|assistant|system)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # _extract_event_date resolves "yesterday", "last Monday", "Feb 5th",
    # "two weeks ago" etc. relative to the anchor.  If no relative expression
    # is found, it returns the anchor (= session date) as fallback.
    return _extract_event_date(body, anchor)


def _compute_temporal_hint(
    evidence_lines: list[dict[str, Any]],
    question_date_str: str,
    schema_name: str,
    slots: dict[str, dict[str, Any] | None] | None = None,
) -> str:
    """Pre-compute date differences from evidence for temporal schemas.

    Uses the compiler's temporal runtime (_extract_event_date) to resolve
    relative temporal expressions ("yesterday", "two months ago") into
    absolute dates before computing differences.  This is the compiler's
    contribution to temporal reasoning — structured date grounding — so
    the reader only needs to read the pre-computed result.
    """
    # Parse question date (flexible: accepts YYYY/M/D and YYYY/MM/DD)
    q_date: date | None = _parse_date_flexible(question_date_str) if question_date_str else None

    # Resolve event dates from all evidence lines using temporal grounding
    resolved: list[tuple[date, str]] = []  # (resolved_date, line_text_snippet)
    for line in evidence_lines:
        text = str(line.get("text") or "")
        ev_date = _resolve_event_date(text)
        if ev_date:
            # Dedupe: skip if we already have this exact date
            if not any(d == ev_date for d, _ in resolved):
                snippet = text[:120]
                resolved.append((ev_date, snippet))

    if not resolved:
        return ""

    resolved.sort(key=lambda x: x[0])

    if schema_name == "RelativeTime":
        if not q_date:
            return ""
        # Pick the event date closest to (but not after) the question date
        best: tuple[date, str] | None = None

        # Priority 1: Use specific bound 'event' slot if available
        if slots and slots.get("event"):
            s_date = slots["event"].get("date")
            if s_date:
                best_date = s_date if isinstance(s_date, date) else _parse_date_flexible(str(s_date))
                if best_date and best_date <= q_date:
                    best = (best_date, slots["event"].get("text") or "")

        # Priority 2: Fallback to scanning resolved evidence lines
        if best is None:
            for ev_date, snippet in resolved:
                if ev_date <= q_date:
                    if best is None or ev_date > best[0]:
                        best = (ev_date, snippet)

        if best is None:
            return ""
        diff_days = (q_date - best[0]).days
        weeks = round(diff_days / 7)
        months = _calendar_month_diff(best[0], q_date)
        months_rounded = round(months)
        return (
            f"Event date: {best[0].isoformat()}, "
            f"Question date: {q_date.isoformat()}, "
            f"Difference: {diff_days} days ({weeks} weeks, {months_rounded} months)"
        )

    if schema_name in {"TemporalInterval", "EventDuration"}:
        d1, d2 = None, None
        # Priority 1: Use specific bound event_a / event_b slots if available
        if slots:
            sa = slots.get("event_a") or slots.get("time_a")
            sb = slots.get("event_b") or slots.get("time_b")
            if schema_name == "EventDuration":
                sa = slots.get("time_a") or slots.get("event")
                sb = slots.get("time_b")
            if sa and sa.get("date"):
                d_val = sa.get("date")
                d1 = d_val if isinstance(d_val, date) else _parse_date_flexible(str(d_val))
            if sb and sb.get("date"):
                d_val = sb.get("date")
                d2 = d_val if isinstance(d_val, date) else _parse_date_flexible(str(d_val))

        if d1 and d2:
            diff_days = abs((d2 - d1).days)
            weeks = round(diff_days / 7)
            months = _calendar_month_diff(d1, d2)
            months_rounded = round(months)
            return (
                f"Date A: {d1.isoformat()}, "
                f"Date B: {d2.isoformat()}, "
                f"Difference: {diff_days} days ({weeks} weeks, {months_rounded} months)"
            )

        # Priority 2: Fallback to scanning resolved evidence lines
        if len(resolved) >= 2:
            d1_res, _ = resolved[0]
            d2_res, _ = resolved[-1]
            diff_days = abs((d2_res - d1_res).days)
            weeks = round(diff_days / 7)
            months = _calendar_month_diff(d1_res, d2_res)
            months_rounded = round(months)
            return (
                f"Date A: {d1_res.isoformat()}, "
                f"Date B: {d2_res.isoformat()}, "
                f"Difference: {diff_days} days ({weeks} weeks, {months_rounded} months)"
            )
        # Only one resolved date — can't compute an interval between two
        # events.  Don't fall back to question_date as it's when the user
        # asked, not when the second event happened.  Emit nothing so the
        # reader doesn't get a misleading hint.

    return ""


def _enumerate_count_operands(
    user_lines: list[dict[str, Any]],
    compiled_units: list[dict[str, Any]],
    query_entity_words: set[str] | None = None,
) -> list[str]:
    """Extract distinct countable items from user-authored evidence lines.

    Returns short descriptive strings, one per distinct item/event, that the
    reader can enumerate for count queries.  When *query_entity_words* is
    provided, only include lines that share at least one content word with
    the query — this filters out irrelevant user lines.

    Authoritative constraint (Fix 2): Only include a line when it is backed
    by at least one compiled unit tagged as a typed countable item
    (is_countable_item flag or count_item tag). This prevents generic user
    advice or un-typed lines from inflating the count.
    """
    # Build set of render_text snippets from qualifying typed-countable units.
    typed_snippets: set[str] = set()
    for unit in compiled_units:
        if not isinstance(unit, dict):
            continue
        typed_flags = unit.get("typed_flags") or {}
        tags = set(unit.get("tags") or [])
        if not (typed_flags.get("is_countable_item") or "count_item" in tags or "countable_item" in tags):
            continue
        render = str(unit.get("render_text") or "").strip()
        if render:
            typed_snippets.add(render[:60].lower())
        # Also include the provenance source_text snippet for matching.
        prov = unit.get("provenance") or {}
        src = str(prov.get("source_text") or "").strip()
        if src:
            typed_snippets.add(src[:60].lower())

    seen: set[str] = set()
    items: list[str] = []
    for line in user_lines:
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        # Strip session header to get the body
        _, body = _split_session_header(text)
        if not body:
            body = text
        # Remove speaker prefix ("user: ...")
        if body.lower().startswith(("user:", "assistant:")):
            body = body.split(":", 1)[1].strip()
        # Skip very short or object-rendered lines (e.g. "user: furniture name: new coffee table")
        if _is_object_rendered_text(body):
            continue
        if len(body) < 10:
            continue
        # Query-relevance filter: skip lines that don't share any content
        # words with the query entities.
        if query_entity_words:
            body_lower = body.lower()
            if not any(w in body_lower for w in query_entity_words):
                continue
        # Typed provenance gate: only include lines backed by a typed countable unit.
        # If we have ANY typed units, enforce this gate; if there are no typed units
        # at all (e.g. object-extraction didn't fire), fall back to the old behavior.
        if typed_snippets:
            body_snippet = body[:60].lower()
            if not any(body_snippet in ts or ts in body_snippet for ts in typed_snippets):
                continue
        # Dedupe by first 40 chars lowered
        key = body[:40].lower()
        if key in seen:
            continue
        seen.add(key)
        # Truncate to a readable summary
        summary = body[:200].rstrip(" .,;:")
        items.append(summary)
    return items


def _is_object_rendered_text(text: str) -> bool:
    """Detect memory-object rendered lines like 'user: furniture name: new coffee table'."""
    stripped = text.strip()
    if not stripped:
        return False
    # Pattern: "speaker: <attribute> <key>: <value>" or "<key> name: <value>"
    if re.match(r"^(?:user|assistant):\s+\S+\s+(?:name|location|time|value):", stripped, re.IGNORECASE):
        return True
    # Also catch standalone attribute lines without speaker prefix
    if re.match(r"^\S+\s+(?:name|location|time|value):\s+", stripped, re.IGNORECASE):
        return True
    return False


def _enumerate_numeric_operands(
    slots: dict[str, dict[str, Any] | None],
    raw_bindings: dict[str, dict[str, Any] | None],
    evidence_lines: list[dict[str, Any]] | None = None,
) -> str:
    """Build a human-readable operand summary for numeric aggregate schemas.

    First tries bound operand slots (operand_1, operand_2, ...).  If fewer
    than 2 are bound, falls back to scanning user-authored evidence lines
    for all numeric values (currency, counts, durations).  This handles the
    common case where the binding only found 2 operands but 3-4 exist in
    the evidence.

    Excludes recommendation/advice lines — those carry instructional numbers
    that are not answer operands (Fix 2 aggregation packaging tightening).
    """
    _ADVICE_RE = re.compile(
        r"\b(?:recommend|suggest|tip|should|consider|aim for|try to|goal is|target|ideally|optimal)\b",
        re.IGNORECASE,
    )
    # 1. Collect from bound operand slots
    operands: list[str] = []
    for key in sorted(raw_bindings.keys()):
        if not key.startswith("operand_"):
            continue
        payload = raw_bindings.get(key)
        if not isinstance(payload, dict):
            continue
        slot_data = slots.get(key)
        text = ""
        if isinstance(slot_data, dict):
            text = str(slot_data.get("display_text") or slot_data.get("text") or "").strip()
        if not text:
            text = str(payload.get("text") or "").strip()
        if text:
            operands.append(text)

    # 2. Also scan user evidence lines for additional numeric mentions
    #    (the binding often finds only 2 operands when 3-4 exist)
    if not evidence_lines:
        return ""
    _NUM_WITH_CONTEXT_RE = re.compile(
        r"(?:(?:\$\s*\d[\d,]*(?:\.\d+)?)|(?:\d[\d,]*(?:\.\d+)?\s*(?:hours?|minutes?|days?|weeks?|months?|years?|miles?|km|dollars?|percent|%|lbs?|pounds?|kg)))",
        re.IGNORECASE,
    )
    seen: set[str] = set()
    extracted: list[str] = []
    for line in evidence_lines:
        if str(line.get("speaker") or "").lower() != "user":
            continue
        text = str(line.get("text") or "")
        _, body = _split_session_header(text)
        if not body:
            body = text
        if _is_object_rendered_text(body):
            continue
        # Skip advice/recommendation lines — they carry instructional numbers.
        if _ADVICE_RE.search(body):
            continue
        for m in _NUM_WITH_CONTEXT_RE.finditer(body):
            val = m.group(0).strip().rstrip(".,;:")
            if val.lower() not in seen:
                seen.add(val.lower())
                extracted.append(val)
    # Merge bound operands with evidence-extracted ones (deduped)
    all_operands = list(operands)
    for val in extracted:
        if val.lower() not in {o.lower() for o in all_operands}:
            all_operands.append(val)
    if len(all_operands) >= 2:
        return " + ".join(all_operands)
    return ""


def _render_budget_fill_context(fill_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render budget fill units grouped by session with date provenance."""
    if not fill_units:
        return []
    from collections import defaultdict
    by_memory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for u in fill_units:
        by_memory[str(u.get("memory_id") or "")].append(u)

    memory_order = []
    for mid, units in by_memory.items():
        best_score = max(float(u.get("fill_score") or 0) for u in units)
        date_key = None
        for u in units:
            dk = u.get("date_key")
            if dk and dk != [0, 0, 0]:
                date_key = dk
                break
        memory_order.append((best_score, mid, date_key, units))
    memory_order.sort(key=lambda x: -x[0])

    sessions: list[dict[str, Any]] = []
    for _, mid, date_key, units in memory_order:
        date_str = f"{date_key[0]}/{date_key[1]:02d}/{date_key[2]:02d}" if date_key else None
        user_units = sorted(
            [u for u in units if str(u.get("speaker") or "").lower() == "user"],
            key=lambda u: -float(u.get("fill_score") or 0),
        )
        other_units = sorted(
            [u for u in units if str(u.get("speaker") or "").lower() != "user"],
            key=lambda u: -float(u.get("fill_score") or 0),
        )
        ordered = user_units + other_units
        lines = []
        for u in ordered:
            text = str(u.get("render_text") or "").strip()
            if not text or len(text) < 10:
                continue
            text = re.sub(
                r"^\d{4}/\d{2}/\d{2}\s+\([A-Za-z]+\)\s+\d{2}:\d{2}\s+session\s+\S+\s+",
                "",
                text,
            )
            # Drop object-rendered garbage lines (e.g. "user: way name: some new socks")
            # These are memory-object metadata that lost the original conversation text.
            if _is_object_rendered_text(text):
                continue
            lines.append({
                "speaker": u.get("speaker"),
                "text": text[:250],
                "memory_id": mid,
                "unit_id": u.get("unit_id"),
            })
        if lines:
            sessions.append({
                "memory_id": mid,
                "date": date_str,
                "lines": lines,
            })
    return sessions


def render_multi_session_fact_list(
    *,
    row: dict[str, Any],
    all_units: list[Any],
    query_targets: Any | None = None,
    target_budget: int = 600,
) -> dict[str, Any]:
    """Render a dated fact list from evidence units across ALL candidate memories.

    Instead of binding to a single schema, this extracts the best user-spoken
    facts from each memory and presents them as a flat, dated list for the
    reader to aggregate (count, sum, compare, etc.).
    """
    query = str(row.get("query") or row.get("question") or "").strip().lower()

    # --- Extract query entity words for relevance filtering ---
    _stop = QUERY_STOP_WORDS
    query_words = {w for w in query.split() if w not in _stop and len(w) > 2}
    if query_targets is not None:
        for ent in getattr(query_targets, "candidate_entities", ()) or ():
            for w in str(ent).lower().split():
                if w not in _stop and len(w) > 2:
                    query_words.add(w)

    # Semantic relevance scoring via MiniLM QA model (optional fallback).
    semantic_scores: dict[str, float] = {}
    try:
        from compiler.adapters.minilm_qa_adapter import extract_span
        for unit in all_units:
            text = str(unit.render_text or "").strip()
            if text and len(text) > 15:
                result = extract_span(query, text[:500])
                semantic_scores[unit.unit_id] = result.confidence
    except Exception:
        pass  # Fall back to word matching if MiniLM unavailable

    # --- Score and filter units ---
    # Prefer: user-spoken, first-person, query-relevant, non-generic
    scored_units: list[tuple[float, Any]] = []
    for unit in all_units:
        text = str(unit.render_text or "").strip()
        if not text or len(text) < 10:
            continue
        # Check object-rendered text on both raw and header-stripped form
        if _is_object_rendered_text(text):
            continue
        _, body = _split_session_header(text)
        if body and _is_object_rendered_text(body):
            continue
        # Also skip lines that look like extracted object fields
        check_body = body or text
        if re.match(r"^(?:user|assistant):\s+\S+(?:\s+\S+){0,2}\s*:\s+", check_body, re.IGNORECASE):
            continue
        # Skip lines with "field_name: value" object metadata patterns
        if re.search(r"\b\w+\s+(?:location|name|time|value|state|type|date|event)\s*:", check_body, re.IGNORECASE):
            continue

        score = 0.0
        text_lower = text.lower()
        speaker = str(unit.speaker or "").lower()

        # Speaker scoring
        if speaker == "user":
            score += 3.0
        elif _is_user_first_person(text):
            score += 2.0
        elif speaker == "assistant":
            # Gate assistant lines: must have a number and not be generic advice
            has_number = bool(re.search(r"\d", text))
            is_generic = _is_generic_assistant(text)
            if not has_number or is_generic:
                continue  # skip generic assistant content entirely
            score += 0.5

        # Semantic relevance (MiniLM QA confidence)
        sem_score = semantic_scores.get(unit.unit_id, 0.0)
        if sem_score > 0.6:
            score += 5.0  # strong answer signal
        elif sem_score > 0.3:
            score += 2.5  # moderate signal

        # Query relevance (lexical, supplements semantic)
        word_hits = sum(1 for w in query_words if w in text_lower)
        score += min(word_hits * 1.5, 6.0)

        # Typed evidence flags (countable items, numeric operands, etc.)
        if unit.is_countable_item:
            score += 3.0
        if unit.is_numeric_operand:
            score += 2.0
        if unit.has_acquisition_marker or unit.has_consumption_or_completion_marker:
            score += 2.0
        if unit.is_direct_answer_candidate:
            score += 1.5
        if unit.is_temporal_anchor:
            score += 1.0

        # Penalties
        if unit.is_generic_state_text or unit.is_low_authority_text:
            score -= 3.0
        if unit.is_question_or_request:
            score -= 2.0
        if unit.is_recommendation_or_advice:
            score -= 2.0

        if score > 0:
            scored_units.append((score, unit))

    scored_units.sort(key=lambda x: -x[0])

    # --- Build fact list within token budget ---
    facts: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    token_total = 0

    for _score, unit in scored_units:
        if token_total >= target_budget:
            break

        text = str(unit.render_text or "").strip()
        # Dedupe by first 50 chars
        dedup_key = text[:50].lower()
        if dedup_key in seen_texts:
            continue
        seen_texts.add(dedup_key)

        # Format date from date_key
        date_str = None
        if unit.date_key and unit.date_key != (0, 0, 0):
            y, m, d = unit.date_key
            date_str = f"{y}/{m:02d}/{d:02d}"

        facts.append({
            "memory_id": unit.memory_id,
            "date": date_str,
            "speaker": unit.speaker,
            "text": text[:250],
            "unit_id": unit.unit_id,
            "is_countable": bool(unit.is_countable_item),
            "is_numeric": bool(unit.is_numeric_operand),
            "score": round(_score, 2),
        })
        token_total += unit.token_count

    # --- Count distinct memories represented ---
    memory_ids = list(dict.fromkeys(f["memory_id"] for f in facts))

    # --- Build evidence lines in dated format ---
    evidence_lines: list[dict[str, Any]] = []
    for f in facts:
        text = f["text"]
        # Only add date prefix if the text doesn't already start with a date
        if f["date"] and not re.match(r"^\d{4}/\d{2}/\d{2}", text):
            text = f"[{f['date']}] {text}"
        evidence_lines.append({
            "memory_id": f["memory_id"],
            "speaker": f["speaker"],
            "text": text,
            "unit_id": f["unit_id"],
        })

    # --- Targeted item extraction for counting/aggregation queries ---
    #
    # Design:
    #   1. Extract the COUNTING TARGET from the query ("plants acquired",
    #      "babies born", "tanks I have").
    #   2. For each evidence line, ask MiniLM: "What <target> is mentioned?"
    #      If it extracts a confident span, that span IS a countable item.
    #   3. Deduplicate extracted items by word overlap (same item from two
    #      sessions = 1, not 2).
    #   4. Count = len(unique_items).  Put into computed_aggregate so the
    #      reader just outputs the number.
    #
    # This replaces the old approach of enumerating entire evidence lines
    # (which was noisy — 10 items for a 3-answer question) with precise
    # span extraction of the actual items being counted.

    is_counting = any(p in query for p in ("how many", "how much", "number of", "count of", "total"))
    enumerated_items = None
    computed_answer = None
    computed_operation = None

    _is_money = "$" in query or any(w in query for w in ("spent", "spend", "cost", "paid", "price", "money", "earn", "save"))
    _is_sum = _is_money or any(p in query for p in ("total", "sum")) and not any(p in query for p in ("how many",))
    _is_avg = "average" in query or "mean" in query
    _is_diff = any(p in query for p in ("difference", "more than", "older than", "how much more", "how much older"))

    if is_counting and not _is_sum and not _is_avg and not _is_diff:

        # Build enumerated items from semantically relevant user lines.
        # Uses MiniLM semantic scores to filter for lines likely containing
        # an answer to the counting query.
        scored_facts: list[tuple[float, dict[str, Any]]] = []
        for f in facts:
            if str(f.get("speaker") or "").lower() != "user":
                continue
            sem = semantic_scores.get(f.get("unit_id", ""), 0.0) if semantic_scores else 0.0
            combined = sem * 5.0 + f.get("score", 0.0)
            scored_facts.append((combined, f))
        scored_facts.sort(key=lambda x: -x[0])

        has_semantic = bool(semantic_scores)
        threshold = 5.0 if has_semantic else 3.0

        items = []
        seen_bodies: set[str] = set()
        for combined_score, f in scored_facts:
            if combined_score < threshold:
                continue
            _, body = _split_session_header(f["text"])
            if not body:
                body = f["text"]
            if body.lower().startswith(("user:", "assistant:")):
                body = body.split(":", 1)[1].strip()
            key = body[:40].lower()
            if key in seen_bodies:
                continue
            seen_bodies.add(key)
            items.append(body[:200])

        if items:
            enumerated_items = {
                "text": "; ".join(f"({i+1}) {item}" for i, item in enumerate(items)),
                "display_text": "; ".join(f"({i+1}) {item}" for i, item in enumerate(items)),
                "count": len(items),
            }

    elif _is_sum or _is_avg or _is_diff:
        # --- NUMERIC AGGREGATION MODE ---
        _extracted_numbers: list[float] = []
        for f in facts:
            _, body = _split_session_header(f["text"])
            if not body:
                body = f["text"]
            if body.lower().startswith(("user:", "assistant:")):
                body = body.split(":", 1)[1].strip()

            if _is_money:
                for m in re.finditer(r'\$\s*([\d,]+(?:\.\d+)?)', body):
                    try:
                        val = float(m.group(1).replace(",", ""))
                        if 0.01 < val < 1_000_000:
                            _extracted_numbers.append(val)
                    except ValueError:
                        pass
            else:
                for m in re.finditer(
                    r'(\d+(?:\.\d+)?)\s*(?:hours?|days?|weeks?|months?|years?|miles?|km|pounds?|lbs?|points?|views?|meals?|items?|fish|runs?|classes|siblings?)',
                    body, re.IGNORECASE,
                ):
                    try:
                        val = float(m.group(1))
                        if 0 < val < 1_000_000:
                            _extracted_numbers.append(val)
                    except ValueError:
                        pass

        if len(_extracted_numbers) >= 2:
            if _is_sum:
                computed_answer = sum(_extracted_numbers)
                computed_operation = "sum"
            elif _is_avg:
                computed_answer = sum(_extracted_numbers) / len(_extracted_numbers)
                computed_operation = "average"
            elif _is_diff:
                computed_answer = abs(_extracted_numbers[0] - _extracted_numbers[1])
                computed_operation = "difference"

    # --- Assemble package ---
    slots: dict[str, Any] = {}
    if enumerated_items:
        slots["enumerated_items"] = enumerated_items
    if computed_answer is not None:
        prefix = "$" if _is_money else ""
        if isinstance(computed_answer, float) and computed_answer == int(computed_answer):
            display = f"{prefix}{int(computed_answer)}"
        elif isinstance(computed_answer, int):
            display = f"{prefix}{computed_answer}"
        else:
            display = f"{prefix}{computed_answer:.1f}"

        # Build display text based on operation type
        if computed_operation == "count" and enumerated_items:
            item_spans = [it["span"] for it in enumerated_items.get("items", [])]
            display_text = f"{display} (counted {len(item_spans)} distinct items: {', '.join(item_spans[:10])})"
        elif computed_operation in ("sum", "average", "difference"):
            _nums = locals().get("_extracted_numbers", [])
            display_text = f"{display} (computed {computed_operation} of {len(_nums)} values: {', '.join(str(n) for n in _nums[:8])})"
        else:
            display_text = display

        slots["computed_aggregate"] = {
            "text": display,
            "display_text": display_text,
            "operation": computed_operation,
        }

    target_entities = [
        str(item).strip()
        for item in getattr(query_targets, "candidate_entities", ()) if str(item).strip()
    ] if query_targets is not None else []

    # Always route to reader for multi-session — the evidence list is for the
    # reader to interpret. Deterministic counting from is_countable_item flags
    # is unreliable in multi-session mode because most user utterances don't
    # get typed object extraction.
    package = {
        "question_family": "multi_session",
        "mode": "multi_session_fact_list",
        "schema_name": "MultiSessionFactList",
        "target_entities": target_entities,
        "subject_entities": [],
        "alternative_entities": [],
        "preferred_attributes": [],
        "requirements": {},
        "coverage": {"fact_list": len(facts)},
        "slots": slots,
        "validated_slots": slots,
        "answer_mode": "reader_from_slots",
        "compiler_answerable": False,
        "invalid_slots": {},
        "answer_text": None,
        "answer_units": [f["unit_id"] for f in facts],
        "evidence_lines": evidence_lines,
        "additional_context": [],
        "compact_text": "\n".join(
            f"- [{f['date'] or '?'}] {f['speaker']}: {f['text'][:120]}"
            for f in facts
        ),
        "multi_session_meta": {
            "total_facts": len(facts),
            "memory_count": len(memory_ids),
            "memory_ids": memory_ids,
            "enumerated_count": enumerated_items["count"] if enumerated_items else 0,
            "token_total": token_total,
        },
    }
    return package


def render_evidence_package(
    *,
    query_family: str,
    evidence_plan: EvidencePlan,
    query_targets: Any | None = None,
    coverage: dict[str, int],
    compiled_units: list[dict[str, Any]],
    slot_bindings: dict[str, dict[str, Any] | None] | None = None,
    answer_mode: str | None = None,
    compiler_answerable: bool | None = None,
    invalid_slots: dict[str, str] | None = None,
    answer_text: str | None = None,
    answer_units: list[str] | None = None,
    budget_fill_units: list[dict[str, Any]] | None = None,
    mode: str = "structured_json",
) -> dict[str, Any]:
    rendered_slots = list(evidence_plan.slots)
    rendered_slot_names = [slot.slot_name for slot in rendered_slots]
    supplemental_slot_names = [
        str(name)
        for name in (
            set(dict(slot_bindings or {}).keys())
            | set(dict(coverage or {}).keys())
            | set(dict(invalid_slots or {}).keys())
        )
        if str(name).strip() and str(name) not in rendered_slot_names
    ]
    rendered_slot_names.extend(sorted(supplemental_slot_names))
    slots: dict[str, dict[str, Any] | None] = {name: None for name in rendered_slot_names}
    raw_bindings = slot_bindings if isinstance(slot_bindings, dict) else {}
    units_by_id = {
        str(unit.get("unit_id")): unit
        for unit in compiled_units
        if isinstance(unit, dict) and unit.get("unit_id") is not None
    }
    slots_by_name = {slot.slot_name: slot for slot in rendered_slots}
    for slot_name in rendered_slot_names:
        payload = raw_bindings.get(slot_name)
        if isinstance(payload, dict):
            unit = units_by_id.get(str(payload.get("unit_id")))
            slot_spec = slots_by_name.get(slot_name)
            if slot_spec is not None:
                display_text = display_slot_text(slot_spec, payload, unit, evidence_plan.plan_metadata if evidence_plan is not None else None)
            else:
                display_text = str(payload.get("display_text") or payload.get("text") or "").strip()
            source_span_text = str(
                payload.get("source_span_text")
                or ((unit or {}).get("provenance") or {}).get("memory_text")
                or payload.get("source_text")
                or (unit or {}).get("render_text")
                or ((unit or {}).get("provenance") or {}).get("source_text")
                or payload.get("text")
                or ""
            ).strip()
            slots[slot_name] = {
                "text": display_text,
                "display_text": display_text,
                "source_span_text": source_span_text,
                "source_text": source_span_text,
                "date": payload.get("date"),
                "memory_id": payload.get("memory_id"),
                "speaker": payload.get("speaker"),
                "unit_id": payload.get("unit_id"),
                "memory_text": ((unit or {}).get("provenance") or {}).get("memory_text"),
            }

    evidence_lines = _dedupe_evidence_lines(compiled_units)

    # --- Aggregation evidence enhancement ---
    schema_name = str(evidence_plan.schema_name or "")
    is_agg = schema_name in _AGGREGATE_SCHEMA_NAMES_RENDER
    is_count = schema_name in _COUNT_SCHEMA_NAMES_RENDER

    if is_agg:
        # 1. Prioritise user-authored first-person lines over assistant boilerplate.
        #    Keep assistant lines only when they carry a number or a specific item name
        #    that matches the query entities.
        user_lines: list[dict[str, Any]] = []
        assistant_useful: list[dict[str, Any]] = []
        assistant_noise: list[dict[str, Any]] = []
        for line in evidence_lines:
            speaker = str(line.get("speaker") or "").lower()
            text = str(line.get("text") or "")
            if speaker == "user":
                user_lines.append(line)
            elif speaker != "assistant" and _is_user_first_person(text):
                user_lines.append(line)
            elif speaker == "assistant":
                # Keep assistant lines that contain numbers or aren't generic advice
                has_number = bool(re.search(r"\d", text))
                is_generic = _is_generic_assistant(text)
                if has_number and not is_generic:
                    assistant_useful.append(line)
                else:
                    assistant_noise.append(line)
            else:
                user_lines.append(line)

        # Rebuild: user lines first, then useful assistant, then noise (truncated)
        evidence_lines = user_lines + assistant_useful
        # Only add noise if we'd otherwise have very few lines
        if len(evidence_lines) < 3:
            evidence_lines.extend(assistant_noise[:2])

        # 2. For count schemas, enumerate distinct user-authored items as operands
        #    so the reader sees an explicit list to count.
        _stop = QUERY_STOP_WORDS
        _qt_words: set[str] = set()
        for ent in (getattr(query_targets, "candidate_entities", ()) or ()) if query_targets is not None else ():
            for w in str(ent).lower().split():
                if w not in _stop and len(w) > 2:
                    _qt_words.add(w)
        if is_count and user_lines:
            operand_items = _enumerate_count_operands(user_lines, compiled_units, _qt_words or None)
            if operand_items:
                slots["enumerated_items"] = {
                    "text": "; ".join(f"({i+1}) {item}" for i, item in enumerate(operand_items)),
                    "display_text": "; ".join(f"({i+1}) {item}" for i, item in enumerate(operand_items)),
                    "count": len(operand_items),
                }

        # 3. For numeric aggregate schemas (SumOperands etc.), build an operand summary
        if not is_count and is_agg:
            operand_summary = _enumerate_numeric_operands(slots, raw_bindings, evidence_lines)
            if operand_summary:
                slots["operand_summary"] = {
                    "text": operand_summary,
                    "display_text": operand_summary,
                }

    # --- Temporal date-diff pre-computation ---
    _TEMPORAL_SCHEMAS = {"TemporalInterval", "RelativeTime", "EventDuration"}
    if schema_name in _TEMPORAL_SCHEMAS:
        question_date_str = str((evidence_plan.plan_metadata or {}).get("question_date") or "").strip()
        computed = _compute_temporal_hint(evidence_lines, question_date_str, schema_name, slots)
        if computed:
            slots["computed_date_hint"] = {
                "text": computed,
                "display_text": computed,
            }

    compact_lines: list[str] = []
    for slot_name, payload in slots.items():
        if payload is None:
            compact_lines.append(f"- {slot_name}: <missing>")
            continue
        compact_lines.append(f"- {slot_name}: {payload.get('display_text') or payload.get('text')}")

    compact_text = "\n".join(compact_lines)

    target_entities = [str(item).strip() for item in getattr(query_targets, "candidate_entities", ()) if str(item).strip()] if query_targets is not None else []
    subject_entities = [str(item).strip() for item in getattr(query_targets, "subject_entities", ()) if str(item).strip()] if query_targets is not None else []
    alternative_entities = [str(item).strip() for item in getattr(query_targets, "alternative_entities", ()) if str(item).strip()] if query_targets is not None else []
    preferred_attributes = [str(item).strip() for item in getattr(query_targets, "preferred_attributes", ()) if str(item).strip()] if query_targets is not None else []

    additional_context = _render_budget_fill_context(budget_fill_units or [])

    package = {
        "question_family": query_family,
        "mode": mode,
        "schema_name": evidence_plan.schema_name,
        "target_entities": target_entities,
        "subject_entities": subject_entities,
        "alternative_entities": alternative_entities,
        "preferred_attributes": preferred_attributes,
        "requirements": {
            slot.slot_name: {
                "slot_type": slot.slot_type,
                "required": bool(slot.required),
                "entity_key": slot.entity_key,
                "attribute_key": slot.attribute_key,
                "expected_unit": slot.expected_unit,
            }
            for slot in evidence_plan.slots
        },
        "coverage": dict(coverage),
        "slots": slots,
        "validated_slots": slots,
        "answer_mode": str(answer_mode or "").strip(),
        "compiler_answerable": bool(compiler_answerable),
        "invalid_slots": dict(invalid_slots or {}),
        "answer_text": str(answer_text).strip() if answer_text is not None else None,
        "answer_units": [str(unit_id) for unit_id in (answer_units or []) if str(unit_id).strip()],
        "evidence_lines": evidence_lines,
        "additional_context": additional_context,
        "compact_text": compact_text,
    }
    return package
