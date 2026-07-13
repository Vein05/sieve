"""Deterministic evidence compiler for CRISP-Evidence Compiler."""

from __future__ import annotations

from datetime import date, timedelta
from dataclasses import replace
import re
from typing import Any

from .compiler.runtime.parsing import (
    _DURATION_TEXT_RE,
    _NUMERIC_PARSE_RE,
    _NUMBER_WORD_TO_INT,
)

from .common import (
    CandidateView,
    cached_build_profile,
    check_stem_equivalence,
    get_clean_tokens,
    get_stems_for_text,
    stem_token,
)
from .evidence_planner import plan_evidence
from .evidence_requirements import build_query_evidence_plan
from .evidence_renderer import display_slot_text
from .evidence_schema import EvidencePlan, slot_requirement_name
from .compiler.execution.requirements import (
    aggregate_requires_two_operands as _aggregate_requires_two_operands,
    aggregate_slot as _aggregate_slot,
    coverage_dict as _coverage_dict,
    max_compiled_tokens as _max_compiled_tokens,
    missing_requirements as _missing_requirements,
    plan_satisfied as _plan_satisfied,
    required_binding_names as _required_binding_names,
    requirements_dict as _requirements_dict,
    selection_requirements_dict as _selection_requirements_dict,
    validated_missing_requirements as _validated_missing_requirements,
    validated_rescue_missing as _validated_rescue_missing,
)
from .compiler.execution.router import apply_semantic_reader_downgrade
from .compiler.execution.deterministic import build_answer_contract
from .compiler.validation.answer_type import validate_slot_binding as _validate_slot_binding_impl
from .compiler.validation.invariants import schema_grounding_analysis as _schema_grounding_analysis_impl
from .evidence_roles import infer_unit_roles
from .evidence_units import EvidenceUnit, build_evidence_units_for_views
from .evidence_projection import project_evidence_candidates, unresolved_slots
from .modeling_logic.features import _answer_shape_signals
from .query_targets import QueryTargets, extract_query_targets
from .compiler.execution.selection import compile_evidence

def _parse_numeric_text(text: str) -> tuple[float, str, str] | None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    duration_match = _DURATION_TEXT_RE.search(cleaned)
    if duration_match:
        number_text = duration_match.group("number").strip().lower()
        if number_text in {
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
        }:
            number_text = {
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
            }[number_text]
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
        parsed = (number, prefix, unit, match.group(0).strip())
        mentions.append(parsed)
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



