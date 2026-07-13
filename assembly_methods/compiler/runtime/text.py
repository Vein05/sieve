"""Shared runtime text helpers for compiler binding and execution code."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from ...common import cached_build_profile
from .numeric import _count_numeric_spans_excluding_dates, _extract_numeric_mentions

_SESSION_HEADER_RE = re.compile(
    r"^\s*(?P<header>\d{4}/\d{2}/\d{2}\s+\([A-Za-z]{3}\)\s+\d{1,2}:\d{2}\s+session\s+\S+\s*\|?\s*(?:user|assistant|system)\s*:)\s*(?P<body>.*)$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class CompiledUnitAnalysis:
    source_text: str
    body_text: str
    normalized_source: str
    content_tokens: tuple[str, ...]
    named_tokens: tuple[str, ...]
    numeric_mentions: tuple[tuple[float, str, str, str], ...]
    numeric_span_count_excluding_dates: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_text": self.source_text,
            "body_text": self.body_text,
            "normalized_source": self.normalized_source,
            "content_tokens": list(self.content_tokens),
            "named_tokens": list(self.named_tokens),
            "numeric_mentions": [list(item) for item in self.numeric_mentions],
            "numeric_span_count_excluding_dates": self.numeric_span_count_excluding_dates,
        }


@dataclass(frozen=True)
class TextAnalysis:
    source_text: str
    normalized_source: str
    content_tokens: tuple[str, ...]
    named_tokens: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_text": self.source_text,
            "normalized_source": self.normalized_source,
            "content_tokens": list(self.content_tokens),
            "named_tokens": list(self.named_tokens),
        }


def _build_text_analysis(source_text: str) -> TextAnalysis:
    raw_source = str(source_text or "").strip()
    profile = cached_build_profile(raw_source)
    return TextAnalysis(
        source_text=raw_source,
        normalized_source=profile.normalized_text,
        content_tokens=tuple(sorted(profile.content_tokens)),
        named_tokens=tuple(sorted(profile.named_tokens)),
    )


def _split_session_header(text: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    if not raw:
        return "", ""
    match = _SESSION_HEADER_RE.match(raw)
    if not match:
        return "", raw
    return str(match.group("header") or "").strip(), str(match.group("body") or "").strip()


def _build_compiled_unit_analysis(source_text: str) -> CompiledUnitAnalysis:
    raw_source = str(source_text or "").strip()
    _, body_text = _split_session_header(raw_source)
    body = body_text or raw_source
    profile = cached_build_profile(body)
    numeric_mentions = tuple(_extract_numeric_mentions(body))
    return CompiledUnitAnalysis(
        source_text=raw_source,
        body_text=body,
        normalized_source=profile.normalized_text,
        content_tokens=tuple(sorted(profile.content_tokens)),
        named_tokens=tuple(sorted(profile.named_tokens)),
        numeric_mentions=numeric_mentions,
        numeric_span_count_excluding_dates=_count_numeric_spans_excluding_dates(body),
    )


def _compiled_unit_analysis_payload(unit: dict[str, Any]) -> dict[str, Any]:
    analysis = unit.get("analysis")
    if isinstance(analysis, dict):
        return analysis
    source_text = str(((unit.get("provenance") or {}).get("source_text") or unit.get("render_text") or "")).strip()
    payload = _build_compiled_unit_analysis(source_text).to_payload()
    if isinstance(unit, dict):
        unit["analysis"] = payload
    return payload


def _compiled_unit_source_text(unit: dict[str, Any]) -> str:
    analysis = _compiled_unit_analysis_payload(unit)
    return str(analysis.get("source_text") or "").strip()


def _compiled_unit_body_text(unit: dict[str, Any]) -> str:
    analysis = _compiled_unit_analysis_payload(unit)
    return str(analysis.get("body_text") or "").strip()


def _compiled_unit_token_sets(unit: dict[str, Any]) -> tuple[set[str], set[str]]:
    analysis = _compiled_unit_analysis_payload(unit)
    return set(str(token) for token in (analysis.get("content_tokens") or [])), set(
        str(token) for token in (analysis.get("named_tokens") or [])
    )


def _compiled_unit_numeric_mentions(unit: dict[str, Any]) -> list[tuple[float, str, str, str]]:
    analysis = _compiled_unit_analysis_payload(unit)
    mentions = analysis.get("numeric_mentions") or []
    return [
        (float(item[0]), str(item[1]), str(item[2]), str(item[3]))
        for item in mentions
        if isinstance(item, (list, tuple)) and len(item) == 4
    ]


def _compiled_unit_numeric_span_count(unit: dict[str, Any]) -> int:
    analysis = _compiled_unit_analysis_payload(unit)
    try:
        return int(analysis.get("numeric_span_count_excluding_dates") or 0)
    except Exception:
        return 0


def _normalize_text(value: str) -> str:
    return cached_build_profile(str(value or "")).normalized_text


def _query_analysis_payload(row: dict[str, Any]) -> dict[str, Any]:
    cached = row.get("_compiler_query_analysis")
    if isinstance(cached, dict):
        return cached
    payload = _build_text_analysis(str(row.get("query", ""))).to_payload()
    row["_compiler_query_analysis"] = payload
    return payload


def _query_token_sets(row: dict[str, Any]) -> tuple[set[str], set[str]]:
    analysis = _query_analysis_payload(row)
    return set(str(token) for token in (analysis.get("content_tokens") or [])), set(
        str(token) for token in (analysis.get("named_tokens") or [])
    )


def _query_normalized_text(row: dict[str, Any]) -> str:
    analysis = _query_analysis_payload(row)
    return str(analysis.get("normalized_source") or "").strip()


def _query_expects_time_literal(row: dict[str, Any]) -> bool:
    normalized = _normalize_text(str(row.get("query", "")))
    padded = f" {normalized} "
    return bool(
        normalized.startswith("when ")
        or " what time " in padded
        or " what date " in padded
        or " what day " in padded
        or " how long " in padded
        or " how many days " in padded
        or " how many weeks " in padded
        or " how many months " in padded
        or " how many years " in padded
    )


def _binding_source_text(binding: dict[str, Any]) -> str:
    return str(
        binding.get("source_span_text")
        or binding.get("source_text")
        or binding.get("display_text")
        or binding.get("text")
        or ""
    ).strip()


def _binding_analysis_payload(binding: dict[str, Any]) -> dict[str, Any]:
    cached = binding.get("_compiler_text_analysis")
    if isinstance(cached, dict):
        return cached
    payload = _build_text_analysis(_binding_source_text(binding)).to_payload()
    binding["_compiler_text_analysis"] = payload
    return payload


def _binding_token_sets(binding: dict[str, Any]) -> tuple[set[str], set[str]]:
    analysis = _binding_analysis_payload(binding)
    return set(str(token) for token in (analysis.get("content_tokens") or [])), set(
        str(token) for token in (analysis.get("named_tokens") or [])
    )


def _requested_duration_unit(row: dict[str, Any]) -> str | None:
    normalized = _normalize_text(str(row.get("query", "")))
    for unit in ("day", "week", "month", "year", "hour", "minute"):
        if re.search(rf"\b{unit}s?\b", normalized):
            return unit
    if " how long " in f" {normalized} ":
        return None
    return None


def _query_has_relative_reference_clause(row: dict[str, Any]) -> bool:
    normalized = _normalize_text(str(row.get("query", "")))
    padded = f" {normalized} "
    return any(marker in padded for marker in (" when ", " after ", " before ", " while ", " during ", " as i "))


def _time_unit_from_query(row: dict[str, Any]) -> str:
    normalized = _normalize_text(str(row.get("query", "")))
    for unit in ("day", "week", "month", "year", "hour", "minute"):
        if re.search(rf"\b{unit}s?\b", normalized):
            return unit
    return "day"
