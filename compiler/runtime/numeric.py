"""Shared runtime numeric parsing helpers for compiler binding and execution code."""

from __future__ import annotations

import re

from .parsing import _DURATION_TEXT_RE, _NUMBER_WORD_TO_INT, _NUMERIC_PARSE_RE


def _contains_math_expression(text: str) -> bool:
    cleaned = re.sub(r"\b\d{4}/\d{2}/\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", " ", str(text or ""))
    cleaned = re.sub(r"\b\d{1,2}:\d{2}\b", " ", cleaned)
    return bool(re.search(r"\d\s*[+*=/]\s*\d|\d\s+-\s+\d", cleaned))


def _count_numeric_spans_excluding_dates(text: str) -> int:
    cleaned = re.sub(r"\b\d{4}/\d{2}/\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b", " ", str(text or ""))
    cleaned = re.sub(r"\b\d{1,2}:\d{2}\b", " ", cleaned)
    return len(list(_NUMERIC_PARSE_RE.finditer(cleaned)))


def _parse_numeric_text(text: str) -> tuple[float, str, str] | None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    duration_match = _DURATION_TEXT_RE.search(cleaned)
    if duration_match:
        number_text = duration_match.group("number").strip().lower()
        if number_text in _NUMBER_WORD_TO_INT:
            number_text = str(_NUMBER_WORD_TO_INT[number_text])
        return float(number_text.replace(",", "")), "", duration_match.group("unit").lower()
    match = _NUMERIC_PARSE_RE.search(cleaned)
    if not match:
        return None
    number = float(match.group("number").replace(",", ""))
    prefix = match.group("prefix") or ""
    suffix = match.group("suffix") or ""
    trailing = cleaned[match.end():].strip().split()
    unit = trailing[0].lower() if trailing else suffix
    return number, prefix, unit


def _extract_numeric_mentions(text: str) -> list[tuple[float, str, str, str]]:
    raw = str(text or "").strip()
    if not raw:
        return []
    mentions: list[tuple[float, str, str, str]] = []
    seen_spans: set[tuple[int, int]] = set()
    duration_spans: list[tuple[int, int]] = []

    for match in _DURATION_TEXT_RE.finditer(raw):
        span = match.span()
        duration_spans.append(span)
        number_text = str(match.group("number") or "").strip().lower()
        if number_text in _NUMBER_WORD_TO_INT:
            number_text = str(_NUMBER_WORD_TO_INT[number_text])
        try:
            parsed = (float(number_text.replace(",", "")), "", str(match.group("unit") or "").lower(), match.group(0).strip())
        except ValueError:
            continue
        mentions.append(parsed)
        seen_spans.add(span)

    for match in _NUMERIC_PARSE_RE.finditer(raw):
        start, end = match.span()
        if any(start >= d_start and end <= d_end for d_start, d_end in duration_spans):
            continue
        if (start, end) in seen_spans:
            continue
        left_char = raw[start - 1] if start > 0 else ""
        right_char = raw[end] if end < len(raw) else ""
        if left_char in "/:" or right_char in "/:":
            continue
        number = float(str(match.group("number") or "0").replace(",", ""))
        prefix = str(match.group("prefix") or "")
        suffix = str(match.group("suffix") or "")
        trailing = raw[end:].strip().split()
        unit = trailing[0].lower().strip(".,;:") if trailing else suffix
        mentions.append((number, prefix, unit, match.group(0).strip()))
        seen_spans.add((start, end))

    return mentions


def _format_numeric_answer(total: float, prefix: str = "", unit: str = "") -> str:
    if total.is_integer():
        number_text = f"{int(total):,}"
    else:
        number_text = f"{total:.2f}".rstrip("0").rstrip(".")
    if prefix:
        return f"{prefix}{number_text}"
    if unit and unit not in {"%", ""}:
        return f"{number_text} {unit}"
    if unit == "%":
        return f"{number_text}%"
    return number_text
