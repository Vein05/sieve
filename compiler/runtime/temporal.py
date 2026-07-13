"""Shared temporal helpers for the compiler runtime."""

from __future__ import annotations

from datetime import date, datetime as _datetime, timedelta
from typing import Any
import re

from .parsing import _NUMBER_WORD_TO_INT, _parse_number_word
from .text import _normalize_text

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
_WEEKDAY_NAME_TO_NUMBER = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_DATE_LIKE_RE = re.compile(r"\b\d{4}/\d{2}/\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
_TIME_LIKE_RE = re.compile(r"\b\d{1,2}:\d{2}\b")


def _binding_anchor_date(binding: dict[str, Any] | None) -> date | None:
    if not isinstance(binding, dict):
        return None
    value = binding.get("date")
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return date(int(value[0]), int(value[1]), int(value[2]))
    except (TypeError, ValueError):
        return None


def _shift_months(anchor: date, months: int) -> date:
    total_months = (anchor.year * 12 + (anchor.month - 1)) + months
    year = total_months // 12
    month = total_months % 12 + 1
    day = min(anchor.day, 28)
    return date(year, month, day)


def _number_word_value(text: str) -> int | None:
    return _parse_number_word(text)


_TEMPORAL_SIGNAL_RE = re.compile(
    r"\b(?:ago|before|after|since|last|next|previous|past|recent|earlier|later"
    r"|week|month|year|day|night|morning|evening"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)


def _dateparser_fallback(text: str, anchor: date) -> date | None:
    """Use dateparser library as fallback for patterns the hand-rolled regex misses."""
    if not _TEMPORAL_SIGNAL_RE.search(text):
        return None
    try:
        from ..adapters.dateparser_adapter import parse_datetime
        result = parse_datetime(text, relative_base=_datetime(anchor.year, anchor.month, anchor.day, 12, 0, 0))
        if result is not None:
            parsed = result.date()
            # Reject if dateparser just returned today (likely a false positive)
            if parsed != anchor:
                return parsed
    except Exception:
        pass
    return None


def _extract_event_date(source_text: str, anchor_date: date | None) -> date | None:
    text = str(source_text or "").strip()
    if not text:
        return anchor_date
    explicit = re.search(r"\b(\d{4})/(\d{1,2})/(\d{1,2})\b", text)
    explicit_is_session_prefix = bool(
        explicit
        and explicit.start() == 0
        and "session" in text[explicit.end(): min(len(text), explicit.end() + 24)].lower()
    )
    if explicit_is_session_prefix:
        text = re.sub(
            r"^\d{4}/\d{1,2}/\d{1,2}(?:\s+\([A-Za-z]{3}\)\s+\d{1,2}:\d{2})?\s+session(?:\s*\|\s*)?\s*(?:user|assistant|system)\s*:\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
    if explicit and not explicit_is_session_prefix:
        try:
            return date(int(explicit.group(1)), int(explicit.group(2)), int(explicit.group(3)))
        except ValueError:
            pass
    anchor = anchor_date
    normalized = _normalize_text(text)
    if anchor is not None and re.search(r"\b(?:just\s+)?started\b", normalized):
        arrived_match = re.search(
            r"\barrived on (" + "|".join(_MONTH_NAME_TO_NUMBER.keys()) + r")\s+\d{1,2}(?:st|nd|rd|th)?\b",
            normalized,
        )
        if arrived_match:
            return anchor
    md_match = re.search(
        r"\b(" + "|".join(_MONTH_NAME_TO_NUMBER.keys()) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b",
        normalized,
    )
    if md_match and anchor is not None:
        try:
            candidate = date(anchor.year, _MONTH_NAME_TO_NUMBER[md_match.group(1)], int(md_match.group(2)))
            # Prefer the past: if resolved date is after anchor, try previous year
            if candidate > anchor:
                candidate = date(anchor.year - 1, candidate.month, candidate.day)
            return candidate
        except ValueError:
            pass
    of_month_match = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+of\s+(" + "|".join(_MONTH_NAME_TO_NUMBER.keys()) + r")\b",
        _normalize_text(text),
    )
    if of_month_match and anchor is not None:
        try:
            candidate = date(anchor.year, _MONTH_NAME_TO_NUMBER[of_month_match.group(2)], int(of_month_match.group(1)))
            if candidate > anchor:
                candidate = date(anchor.year - 1, candidate.month, candidate.day)
            return candidate
        except ValueError:
            pass
    numeric_md = re.search(r"\b(\d{1,2})/(\d{1,2})\b", text)
    if numeric_md and anchor is not None:
        try:
            candidate = date(anchor.year, int(numeric_md.group(1)), int(numeric_md.group(2)))
            if candidate > anchor:
                candidate = date(anchor.year - 1, candidate.month, candidate.day)
            return candidate
        except ValueError:
            pass
    if anchor is not None:
        if " today " in f" {normalized} ":
            return anchor
        # "day before yesterday" must be checked before bare "yesterday"
        if re.search(r"\bday\s+before\s+yesterday\b", normalized):
            return anchor - timedelta(days=2)
        if " yesterday " in f" {normalized} ":
            return anchor - timedelta(days=1)
        if " tomorrow " in f" {normalized} ":
            return anchor + timedelta(days=1)
        weekday_match = re.search(r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", normalized)
        if weekday_match:
            target = _WEEKDAY_NAME_TO_NUMBER[weekday_match.group(1)]
            delta = (anchor.weekday() - target) % 7 or 7
            return anchor - timedelta(days=delta)
        # Bare weekday without "last" — resolve to most recent occurrence
        bare_weekday = re.search(
            r"\b(?:on\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            normalized,
        )
        if bare_weekday:
            target = _WEEKDAY_NAME_TO_NUMBER[bare_weekday.group(1)]
            delta = (anchor.weekday() - target) % 7 or 7
            return anchor - timedelta(days=delta)
        # "a/an week ago", "a month ago" — the word "a" isn't in _NUMBER_WORD_TO_INT
        ago_match = re.search(r"\b(\d+|a|an|[a-z]+(?:-[a-z]+)?)\s+(days?|weeks?|months?|years?)\s+ago\b", normalized)
        if ago_match:
            amount_text = ago_match.group(1)
            amount = 1 if amount_text in ("a", "an") else _number_word_value(amount_text)
            unit = ago_match.group(2)
            if amount is not None:
                if unit.startswith("day"):
                    return anchor - timedelta(days=amount)
                if unit.startswith("week"):
                    return anchor - timedelta(days=7 * amount)
                if unit.startswith("month"):
                    return _shift_months(anchor, -amount)
                if unit.startswith("year"):
                    return _shift_months(anchor, -12 * amount)
        # "N days/weeks/months before" (without "ago")
        before_match = re.search(r"\b(\d+|a|an|[a-z]+(?:-[a-z]+)?)\s+(days?|weeks?|months?|years?)\s+before\b", normalized)
        if before_match:
            amount_text = before_match.group(1)
            amount = 1 if amount_text in ("a", "an") else _number_word_value(amount_text)
            unit = before_match.group(2)
            if amount is not None:
                if unit.startswith("day"):
                    return anchor - timedelta(days=amount)
                if unit.startswith("week"):
                    return anchor - timedelta(days=7 * amount)
                if unit.startswith("month"):
                    return _shift_months(anchor, -amount)
                if unit.startswith("year"):
                    return _shift_months(anchor, -12 * amount)
        # "last week/month/year" without a specific weekday/month name
        last_period = re.search(r"\blast\s+(week|month|year)\b", normalized)
        if last_period:
            period = last_period.group(1)
            if period == "week":
                return anchor - timedelta(days=7)
            if period == "month":
                return _shift_months(anchor, -1)
            if period == "year":
                return _shift_months(anchor, -12)
        since_match = re.search(r"\bsince\s+(" + "|".join(_MONTH_NAME_TO_NUMBER.keys()) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b", normalized)
        if since_match:
            try:
                candidate = date(anchor.year, _MONTH_NAME_TO_NUMBER[since_match.group(1)], int(since_match.group(2)))
                if candidate > anchor:
                    candidate = date(anchor.year - 1, candidate.month, candidate.day)
                return candidate
            except ValueError:
                pass
        past_match = re.search(r"\bfor\s+the\s+past\s+(\d+|[a-z]+(?:-[a-z]+)?)\s+(days?|weeks?|months?|years?)\b", normalized)
        if past_match:
            amount = _number_word_value(past_match.group(1))
            unit = past_match.group(2)
            if amount is not None:
                if unit.startswith("day"):
                    return anchor - timedelta(days=amount)
                if unit.startswith("week"):
                    return anchor - timedelta(days=7 * amount)
                if unit.startswith("month"):
                    return _shift_months(anchor, -amount)
                if unit.startswith("year"):
                    return _shift_months(anchor, -12 * amount)
        # Final fallback: use dateparser for patterns not covered above.
        # Only attempt if the text has plausible temporal content to avoid
        # slow false-positive parses on arbitrary text.
        dp_result = _dateparser_fallback(text, anchor)
        if dp_result is not None:
            return dp_result
    return anchor
