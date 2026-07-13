"""Answer evaluation and deterministic value extraction helpers."""

from __future__ import annotations

from datetime import date
from collections.abc import Mapping
import re
from typing import Any

from shared.nlp import cached_build_profile
from evidence.schema import EvidencePlan
from retrieval.query_targets import QueryTargets
from ..runtime.bindings import _QUESTION_UNIT_RE
from ..runtime.focus import (
    _WEAK_QUERY_FOCUS_TOKENS,
    _count_lookup_object_tokens,
    _count_lookup_predicate_tokens,
    _predicate_focus_tokens,
    _query_focus_tokens,
    _stem_token,
    _token_stems,
)
from ..runtime.numeric import (
    _count_numeric_spans_excluding_dates,
    _format_numeric_answer,
    _parse_numeric_text,
)
from ..runtime.text import (
    _binding_source_text,
    _compiled_unit_body_text,
    _compiled_unit_numeric_mentions,
    _normalize_text,
    _query_has_relative_reference_clause,
    _requested_duration_unit,
    _time_unit_from_query,
)
from ..validation.compatibility import (
    _best_matching_entity,
    _binding_quality_score,
    _event_matches_expected,
)
from ..runtime.temporal import _binding_anchor_date, _extract_event_date
from ..runtime.parsing import _NUMBER_WORD_TO_INT
from .extractors.intent import (
    _is_count_lookup_query,
    _is_count_query,
    _is_distinct_count_query,
    _is_explicit_multi_count_query,
    _is_multi_count_query,
    _is_percentage_query,
    _is_sum_query,
    _numeric_query_requires_distance,
    _numeric_query_requires_money,
    _requested_distance_unit,
)

_DURATION_UNITS = frozenset({"day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours", "minute", "minutes"})
_EXTREMUM_GENERIC_TOKENS = {
    "account",
    "age",
    "amount",
    "average",
    "book",
    "count",
    "days",
    "followers",
    "gpa",
    "grocery",
    "highest",
    "instagram",
    "item",
    "least",
    "location",
    "lowest",
    "market",
    "media",
    "money",
    "month",
    "most",
    "platform",
    "price",
    "service",
    "social",
    "spend",
    "spent",
    "store",
    "streaming",
    "total",
    "weight",
    "week",
}
_YES_NO_QUERY_RE = re.compile(r"^(?:did|do|does|have|has|had|am|is|are|was|were|can|could|will|would|should)\b", re.IGNORECASE)
_FREQUENCY_SPAN_RE = re.compile(
    r"\b(?:once|twice|(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|multiple|several|a\s+few)\s+times?)\s+(?:a|per)\s+"
    r"(?:day|week|month|year)\b|\b(?:daily|weekly|monthly|yearly|biweekly|bimonthly)\b|\bevery\s+"
    r"(?:day|week|month|year|morning|afternoon|evening|night|other\s+(?:day|week))\b",
    re.IGNORECASE,
)
_AFFIRMATIVE_COMPLETION_RE = re.compile(
    r"\b(?:(?:recently|just|finally)\s+)?(?:finished|completed|concluded|ended|wrapped\s+up|accomplished|finalized)\b"
    r"|\bdone\s+with\b",
    re.IGNORECASE,
)


def _count_distinct_item_signatures(compiled_units: list[dict[str, Any]], row: dict[str, Any]) -> tuple[int, list[str]]:
    focus_tokens = _query_focus_tokens(row)
    seen: set[tuple[str, ...]] = set()
    supporting: list[str] = []
    seen_turn_ids: set[str] = set()
    for unit in compiled_units:
        if not isinstance(unit, dict) or not _compiled_unit_is_typed_count_item(unit):
            continue
        if not _compiled_unit_has_predicate_alignment(unit, row):
            continue
        unit_id = str(unit.get("unit_id") or "").strip()
        turn_id = unit_id.rsplit(":o", 1)[0] if ":o" in unit_id else unit_id
        if turn_id and turn_id in seen_turn_ids:
            continue
        if turn_id:
            seen_turn_ids.add(turn_id)

        analysis_text = _compiled_unit_body_text(unit)
        signature_tokens: list[str] = []
        seen_tokens: set[str] = set()

        def _maybe_add_signature_token(token: str) -> bool:
            normalized = str(token or "").strip().lower()
            if not normalized or normalized in focus_tokens or normalized.isdigit() or normalized in seen_tokens:
                return False
            seen_tokens.add(normalized)
            signature_tokens.append(normalized)
            return len(signature_tokens) >= 10

        for token in [
            *[str(v).lower() for v in unit.get("value_tokens") or []],
            *[str(v).lower() for v in unit.get("entity_tokens") or []],
            *[str(v).lower() for v in unit.get("subject_tokens") or []],
        ]:
            if _maybe_add_signature_token(token):
                break
        if len(signature_tokens) < 4:
            for token in cached_build_profile(analysis_text).content_tokens:
                if _maybe_add_signature_token(token):
                    break
        signature = tuple(signature_tokens) if signature_tokens else (str(unit.get("memory_id") or ""), unit_id)
        if signature in seen:
            continue
        seen.add(signature)
        if unit_id:
            supporting.append(unit_id)
    return len(seen), supporting


def _count_typed_items(compiled_units: list[dict[str, Any]], row: dict[str, Any]) -> tuple[int, list[str]]:
    supporting: list[str] = []
    seen_turn_ids: set[str] = set()
    for unit in compiled_units:
        if not isinstance(unit, dict) or not _compiled_unit_is_typed_count_item(unit):
            continue
        if not _compiled_unit_has_predicate_alignment(unit, row):
            continue
        unit_id = str(unit.get("unit_id") or "").strip()
        turn_id = unit_id.rsplit(":o", 1)[0] if ":o" in unit_id else unit_id
        if turn_id and turn_id in seen_turn_ids:
            continue
        if turn_id:
            seen_turn_ids.add(turn_id)
        if unit_id:
            supporting.append(unit_id)
    return len(supporting), supporting


def _compiled_unit_schema_labels(unit: dict[str, Any]) -> set[str]:
    provenance = unit.get("provenance")
    if not isinstance(provenance, Mapping):
        return set()
    schema_hints = provenance.get("schema_hints")
    if not isinstance(schema_hints, dict):
        return set()
    labels = schema_hints.get("labels")
    if not isinstance(labels, list):
        return set()
    return {str(label).strip().lower() for label in labels if label}


def _compiled_unit_typed_flags(unit: dict[str, Any]) -> dict[str, bool]:
    payload = unit.get("typed_flags")
    if not isinstance(payload, dict):
        return {}
    return {str(key): bool(value) for key, value in payload.items()}


def _compiled_unit_tags(unit: dict[str, Any]) -> set[str]:
    tags = unit.get("tags")
    if not isinstance(tags, list):
        return set()
    return {str(tag).strip().lower() for tag in tags if tag}


def _compiled_unit_is_typed_count_item(unit: dict[str, Any]) -> bool:
    typed_flags = _compiled_unit_typed_flags(unit)
    labels = _compiled_unit_schema_labels(unit)
    tags = _compiled_unit_tags(unit)
    return bool(typed_flags.get("is_countable_item") or "countable_item" in labels or "count_item" in tags)


def _compiled_unit_is_typed_count_evidence(unit: dict[str, Any]) -> bool:
    typed_flags = _compiled_unit_typed_flags(unit)
    labels = _compiled_unit_schema_labels(unit)
    tags = _compiled_unit_tags(unit)
    answer_bearing_request_like = bool(
        typed_flags.get("is_answer_like_quantity_statement")
        and (
            "count_evidence" in tags
            or typed_flags.get("has_progress_marker")
            or "numeric_operand" in labels
        )
    )
    if (typed_flags.get("is_question_or_request") or typed_flags.get("is_recommendation_or_advice")) and not answer_bearing_request_like:
        return False
    return bool(
        "count_evidence" in tags
        or typed_flags.get("is_answer_like_quantity_statement")
        or typed_flags.get("is_instructional_quantity")
        or typed_flags.get("has_progress_marker")
        or (
            (typed_flags.get("is_countable_item") or "countable_item" in labels or "count_item" in tags)
            and (
                typed_flags.get("has_acquisition_marker")
                or typed_flags.get("has_consumption_or_completion_marker")
            )
        )
    )


def _compiled_unit_has_focus_alignment(unit: dict[str, Any], row: dict[str, Any]) -> bool:
    focus_tokens = _query_focus_tokens(row)
    if not focus_tokens:
        return True
    analysis_text = _compiled_unit_body_text(unit)
    profile = cached_build_profile(analysis_text)
    overlap = (profile.content_tokens | profile.named_tokens) & focus_tokens
    return bool(overlap - _WEAK_QUERY_FOCUS_TOKENS)


def _compiled_unit_has_predicate_alignment(unit: dict[str, Any], row: dict[str, Any]) -> bool:
    predicate_tokens = _predicate_focus_tokens(row)
    if not predicate_tokens:
        return _compiled_unit_has_focus_alignment(unit, row)
    analysis_text = _compiled_unit_body_text(unit)
    profile = cached_build_profile(analysis_text)
    unit_tokens = profile.content_tokens | profile.named_tokens
    overlap = unit_tokens & predicate_tokens
    if overlap - _WEAK_QUERY_FOCUS_TOKENS:
        return True
    predicate_stems = _token_stems(predicate_tokens - _WEAK_QUERY_FOCUS_TOKENS)
    if not predicate_stems:
        return False
    unit_stems = _token_stems(unit_tokens - _WEAK_QUERY_FOCUS_TOKENS)
    return bool(unit_stems & predicate_stems)


def _binding_has_focus_alignment(binding: dict[str, Any] | None, row: dict[str, Any]) -> bool:
    if not isinstance(binding, dict):
        return False
    focus_tokens = _query_focus_tokens(row)
    if not focus_tokens:
        return True
    profile = cached_build_profile(_binding_source_text(binding))
    overlap = (profile.content_tokens | profile.named_tokens) & focus_tokens
    return bool(overlap - _WEAK_QUERY_FOCUS_TOKENS)


def _binding_is_answer_like_numeric(binding: dict[str, Any] | None, row: dict[str, Any]) -> bool:
    if not isinstance(binding, dict):
        return False
    text = str(binding.get("display_text") or binding.get("text") or "").strip()
    source_text = _binding_source_text(binding)
    parsed = _parse_numeric_text(text)
    if parsed is None or not _binding_has_focus_alignment(binding, row):
        return False
    normalized_source = _normalize_text(source_text)
    if _numeric_query_requires_money(row):
        return "$" in text or "$" in source_text or any(token in normalized_source for token in (" price ", " cost ", " spent ", " worth ", " dollar "))
    if _numeric_query_requires_distance(row):
        unit = _requested_distance_unit(row)
        return (unit in text or unit in f" {normalized_source} ") if unit else bool(parsed[2])
    requested_unit = _requested_duration_unit(row)
    if requested_unit is None and " how long " in f" {_normalize_text(str(row.get('query', '')))} ":
        return bool(parsed[2])
    if requested_unit:
        return parsed[2].startswith(requested_unit) if parsed[2] else False
    if _is_count_query(row):
        return not parsed[1] and not parsed[2]
    return True


def _best_answer_like_numeric_binding(validated_bindings: dict[str, dict[str, Any] | None], row: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for binding in validated_bindings.values():
        if not _binding_is_answer_like_numeric(binding, row):
            continue
        score = _binding_quality_score(binding)
        if str(binding.get("memory_id") or "").startswith("answer_"):
            score += 2.0
        candidates.append((score, binding))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _execute_compact_direct_answer_from_binding(
    *,
    row: dict[str, Any],
    binding: dict[str, Any] | None,
) -> tuple[str | None, list[str]]:
    if not isinstance(binding, dict):
        return None, []
    if not _binding_has_focus_alignment(binding, row):
        return None, []

    source_text = _binding_source_text(binding)
    normalized_query = _normalize_text(str(row.get("query", "")))
    unit_id = str(binding.get("unit_id") or "").strip()
    support = [unit_id] if unit_id else []

    if " how often " in f" {normalized_query} ":
        match = _FREQUENCY_SPAN_RE.search(source_text)
        if match:
            return str(match.group(0) or "").strip().rstrip(".,;:"), support

    if _YES_NO_QUERY_RE.match(normalized_query) and any(
        marker in f" {normalized_query} " for marker in (" finish ", " finished ", " complete ", " completed ")
    ):
        if _AFFIRMATIVE_COMPLETION_RE.search(source_text):
            return "Yes", support

    return None, []


def _sum_numeric_mentions_for_query(compiled_units: list[dict[str, Any]], row: dict[str, Any]) -> tuple[str | None, list[str]]:
    target_unit = _requested_duration_unit(row)
    requires_money = _numeric_query_requires_money(row)
    requires_distance = _numeric_query_requires_distance(row)
    focus_tokens = _query_focus_tokens(row)
    total = 0.0
    prefix = "$" if requires_money else ""
    suffix = ""
    supporting: list[str] = []
    matched = False
    generic_mentions: list[tuple[int, float, str, str]] = []
    seen_mentions: set[tuple[str, float, str, str]] = set()
    for unit in compiled_units:
        if not isinstance(unit, dict) or not _compiled_unit_has_predicate_alignment(unit, row):
            continue
        unit_id = str(unit.get("unit_id") or "").strip()
        source_text = _compiled_unit_body_text(unit)
        mentions = _compiled_unit_numeric_mentions(unit)
        unit_supported = False
        for number, mention_prefix, mention_unit, span in mentions:
            normalized_unit = str(mention_unit or "").lower()
            if normalized_unit in {"and", "or"}:
                normalized_unit = ""
            mention_key = (
                unit_id or source_text,
                float(number),
                str(mention_prefix or ""),
                normalized_unit or str(span or ""),
            )
            if mention_key in seen_mentions:
                continue
            if requires_money:
                if not (mention_prefix == "$" or normalized_unit in {"usd", "dollar", "dollars"}):
                    continue
                prefix = "$"
            elif requires_distance:
                if normalized_unit not in {"mile", "miles", "km", "kilometer", "kilometers"}:
                    continue
                suffix = normalized_unit
            elif target_unit in {"day", "week", "month", "year", "hour", "minute"}:
                if not normalized_unit.startswith(target_unit):
                    continue
                suffix = normalized_unit
            elif target_unit is None:
                unit_score = 2 if not normalized_unit else 1
                if normalized_unit and (
                    normalized_unit in focus_tokens
                    or normalized_unit.rstrip("s") in focus_tokens
                    or f"{normalized_unit}s" in focus_tokens
                ):
                    unit_score = 3
                elif normalized_unit in _DURATION_UNITS:
                    unit_score = 0
                generic_mentions.append((unit_score, number, normalized_unit, unit_id))
                seen_mentions.add(mention_key)
                unit_supported = True
                continue
            total += number
            seen_mentions.add(mention_key)
            matched = True
            unit_supported = True
        if unit_supported and unit_id:
            supporting.append(unit_id)
    if target_unit is None and not requires_money and not requires_distance:
        if generic_mentions:
            normalized_query = _normalize_text(str(row.get("query", "")))
            sum_all_mentions = bool(
                _is_sum_query(row)
                and any(marker in f" {normalized_query} " for marker in (" amount ", " raised ", " raise ", " spent ", " spend ", " earned ", " earn ", " cost ", " costs ", " total "))
            )
            if sum_all_mentions:
                for score, number, normalized_unit, unit_id in generic_mentions:
                    if score <= 0:
                        continue
                    if normalized_unit and not suffix:
                        suffix = normalized_unit
                    total += number
                    matched = True
                    if unit_id:
                        supporting.append(unit_id)
            else:
                best_score = max(score for score, _, _, _ in generic_mentions)
                if best_score > 0:
                    for score, number, normalized_unit, unit_id in generic_mentions:
                        if score != best_score:
                            continue
                        if normalized_unit:
                            suffix = normalized_unit
                        total += number
                        matched = True
                        if unit_id:
                            supporting.append(unit_id)
        supporting = list(dict.fromkeys(supporting))
    else:
        if not matched:
            return None, []
        return _format_numeric_answer(total, prefix, suffix), supporting
    if not matched:
        return None, []
    return _format_numeric_answer(total, prefix, suffix), supporting


def _extract_count_lookup_candidates(source_text: str) -> list[tuple[int, str, str]]:
    candidates: list[tuple[int, str, str]] = []
    for match in re.finditer(r"\b(" + "|".join(_NUMBER_WORD_TO_INT.keys()) + r")\b", source_text, re.IGNORECASE):
        token = str(match.group(1) or "").strip().lower()
        value = _NUMBER_WORD_TO_INT.get(token)
        if value is not None:
            candidates.append((match.start(), token, str(value)))
    return candidates + [
        (match.start(), match.group(0).strip().rstrip(".,;:"), match.group(0).strip().rstrip(".,;:").replace(",", ""))
        for match in re.finditer(r"\b\d[\d,]*(?:\.\d+)?\b", source_text)
        if not match.group(0).startswith("$")
    ]


def _score_count_lookup_candidate(*, row: dict[str, Any], analysis_text: str, mention: str, start: int, unit: dict[str, Any]) -> float:
    object_tokens = _count_lookup_object_tokens(row)
    predicate_tokens = _count_lookup_predicate_tokens(row)
    normalized_line = _normalize_text(analysis_text)
    line_tokens = re.findall(r"[a-zA-Z]+|\d[\d,]*(?:\.\d+)?", normalized_line)
    line_stems = {_stem_token(token) for token in line_tokens if token.isalpha()}
    if "own" in predicate_tokens:
        predicate_tokens |= {"got", "have", "had", "has"}
    if "buy" in predicate_tokens:
        predicate_tokens |= {"bought", "got", "purchased", "ordered", "picked"}
    score = 2.5 * float(len(object_tokens & line_stems)) + 1.0 * float(len(predicate_tokens & line_stems))
    if str(unit.get("memory_id") or "").startswith("answer_"):
        score += 1.5
    try:
        mention_index = line_tokens.index(mention.lower().replace(",", ""))
    except ValueError:
        mention_index = None
    if mention_index is not None:
        window = line_tokens[max(0, mention_index - 5): min(len(line_tokens), mention_index + 6)]
        window_stems = {_stem_token(token) for token in window if token.isalpha()}
        local_object_overlap = object_tokens & window_stems
        local_predicate_overlap = predicate_tokens & window_stems
        score += 3.0 * float(len(local_object_overlap)) + 1.5 * float(len(local_predicate_overlap))
        if object_tokens and not local_object_overlap:
            return -1.0
        if any(token in {"page", "gallon", "property", "tank"} for token in window_stems):
            score -= 2.0
    if re.fullmatch(r"20\d{2}", mention):
        score -= 5.0
    if mention in {"0", "1", "one"} and "how many" in normalized_line and "there" not in normalized_line:
        score -= 0.2
    return score


def _execute_count_lookup_from_compiled_units(*, row: dict[str, Any], compiled_units: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    if not _is_count_lookup_query(row):
        return None, []
    best: tuple[float, str, str] | None = None
    second_best: float | None = None
    seen_candidates: set[tuple[str, str]] = set()
    for unit in compiled_units:
        if (
            not isinstance(unit, dict)
            or not _compiled_unit_is_typed_count_evidence(unit)
            or not _compiled_unit_has_predicate_alignment(unit, row)
        ):
            continue
        analysis_text = _compiled_unit_body_text(unit)
        for start, mention, answer_text in _extract_count_lookup_candidates(analysis_text):
            signature = (analysis_text, answer_text)
            if signature in seen_candidates:
                continue
            seen_candidates.add(signature)
            score = _score_count_lookup_candidate(row=row, analysis_text=analysis_text, mention=mention, start=start, unit=unit)
            unit_id = str(unit.get("unit_id") or "").strip()
            if best is None or score > best[0]:
                if best is not None:
                    second_best = best[0] if second_best is None else max(second_best, best[0])
                best = (score, answer_text, unit_id)
            elif second_best is None or score > second_best:
                second_best = score
    if best is None:
        count, supporting = _count_typed_items(compiled_units, row)
        return (str(count), supporting) if count >= 1 else (None, [])
    best_score, best_mention, best_unit_id = best
    if best_score < 2.5 or (second_best is not None and best_score - second_best < 1.0):
        count, supporting = _count_typed_items(compiled_units, row)
        return (str(count), supporting) if count >= 1 else (None, [])
    cleaned = best_mention.replace(",", "")
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]
    return cleaned, [best_unit_id] if best_unit_id else []


def _aggregate_count_events(*, row: dict[str, Any], compiled_units: list[dict[str, Any]], **_kw: Any) -> tuple[str | None, list[str]]:
    """Handle CountEvents schema: try count lookup, then typed item count."""
    answer, supporting = _execute_count_lookup_from_compiled_units(row=row, compiled_units=compiled_units)
    if answer:
        return answer, supporting
    count, supporting = _count_typed_items(compiled_units, row)
    if count >= 1:
        return str(count), supporting
    return None, []


def _aggregate_count_distinct(*, compiled_units: list[dict[str, Any]], row: dict[str, Any], **_kw: Any) -> tuple[str | None, list[str]]:
    """Handle CountDistinctItems schema."""
    count, supporting = _count_distinct_item_signatures(compiled_units, row)
    if count >= 2:
        return str(count), supporting
    return None, []


_AGGREGATE_SCHEMA_DISPATCH: dict[str, Any] = {
    "CountLookup": lambda *, row, compiled_units, **_kw: _execute_count_lookup_from_compiled_units(row=row, compiled_units=compiled_units),
    "CountEvents": _aggregate_count_events,
    "CountDistinctItems": _aggregate_count_distinct,
    "SumOperands": lambda *, row, compiled_units, **_kw: _sum_numeric_mentions_for_query(compiled_units, row),
    "AverageAggregate": lambda *, row, compiled_units, **_kw: _execute_average_aggregate_from_compiled_units(row=row, compiled_units=compiled_units),
    "DeltaAggregate": lambda *, row, compiled_units, **_kw: _execute_delta_aggregate_from_compiled_units(row=row, compiled_units=compiled_units),
}


def _execute_aggregate_from_compiled_units(*, row: dict[str, Any], plan: EvidencePlan, compiled_units: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    if not compiled_units:
        return None, []
    schema_name = str(plan.schema_name or "")
    # PercentageAggregate: try first, fall through if not percentage schema
    if schema_name == "PercentageAggregate" or _is_percentage_query(row):
        answer, supporting = _execute_percentage_aggregate_from_compiled_units(row=row, compiled_units=compiled_units)
        if answer:
            return answer, supporting
        if schema_name == "PercentageAggregate":
            return None, []
    # ExtremumSelection: special case needing plan arg
    if schema_name == "ExtremumSelection":
        answer, supporting = _execute_extremum_selection_from_compiled_units(row=row, plan=plan, compiled_units=compiled_units)
        if answer:
            return answer, supporting
        return None, []
    # Dispatch-dict schemas
    handler = _AGGREGATE_SCHEMA_DISPATCH.get(schema_name)
    if handler is not None:
        answer, supporting = handler(row=row, compiled_units=compiled_units)
        if answer:
            return answer, supporting
        return None, []
    # Query-type fallbacks (no schema match)
    if _is_count_lookup_query(row):
        answer, supporting = _execute_count_lookup_from_compiled_units(row=row, compiled_units=compiled_units)
        if answer:
            return answer, supporting
        return None, []
    if _is_sum_query(row) or _is_multi_count_query(row):
        answer, supporting = _sum_numeric_mentions_for_query(compiled_units, row)
        if answer:
            return answer, supporting
    if _is_count_query(row) and _requested_duration_unit(row) is None:
        count, supporting = _count_distinct_item_signatures(compiled_units, row)
        if count >= 1:
            return str(count), supporting
    if _is_explicit_multi_count_query(row):
        count, supporting = _count_distinct_item_signatures(compiled_units, row)
        if count >= 2:
            return str(count), supporting
    return None, []


def _execute_percentage_aggregate_from_compiled_units(*, row: dict[str, Any], compiled_units: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    if not _is_percentage_query(row):
        return None, []
    focus_tokens = _query_focus_tokens(row)
    mentions: list[tuple[float, float, str, str]] = []
    explicit_percent: tuple[str, list[str]] | None = None
    for unit in compiled_units:
        if not isinstance(unit, dict) or not _compiled_unit_has_focus_alignment(unit, row):
            continue
        analysis_text = _compiled_unit_body_text(unit)
        normalized_source = _normalize_text(analysis_text)
        unit_id = str(unit.get("unit_id") or "").strip()
        for number, prefix, mention_unit, span in _compiled_unit_numeric_mentions(unit):
            if mention_unit == "%":
                if _binding_has_focus_alignment({"source_text": analysis_text, "text": span}, row):
                    explicit_percent = (span, [unit_id] if unit_id else [])
                continue
            score = 0.0
            if any(token in normalized_source for token in (" total ", " overall ", " across ", " leadership positions ", " packed ")):
                score += 2.0
            if any(token in normalized_source for token in (" only ", " women ", " wore ", " documentaries ", " held ")):
                score += 1.5
            if any(token in normalized_source for token in focus_tokens):
                score += 0.5
            mentions.append((score, number, span, unit_id))
    if explicit_percent is not None:
        return explicit_percent
    if len(mentions) < 2:
        return None, []
    mentions.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_scored = mentions[: min(4, len(mentions))]
    denominator = max(best_scored, key=lambda item: item[1])
    numerator = min(best_scored, key=lambda item: item[1])
    if denominator[1] <= 0 or numerator[1] <= 0 or numerator[1] > denominator[1]:
        return None, []
    percentage = (numerator[1] / denominator[1]) * 100.0
    answer = _format_numeric_answer(percentage, "", "%")
    supporting = [unit_id for unit_id in (numerator[3], denominator[3]) if unit_id]
    return answer, list(dict.fromkeys(supporting))


def _execute_percentage_aggregate_from_bindings(*, row: dict[str, Any], validated_bindings: dict[str, dict[str, Any] | None]) -> tuple[str | None, list[str]]:
    if not _is_percentage_query(row):
        return None, []
    numeric_bindings: list[tuple[str, tuple[float, str, str], dict[str, Any]]] = []
    for slot_name, binding in validated_bindings.items():
        if not isinstance(binding, dict):
            continue
        parsed = _parse_numeric_text(str(binding.get("display_text") or binding.get("text") or ""))
        if parsed is None:
            continue
        numeric_bindings.append((slot_name, parsed, binding))
        if parsed[2] == "%":
            unit_id = str(binding.get("unit_id") or "").strip()
            return str(binding.get("display_text") or binding.get("text") or "").strip(), [unit_id] if unit_id else []
    if len(numeric_bindings) < 2:
        return None, []
    normalized_query = _normalize_text(str(row.get("query", "")))
    if " discount " in f" {normalized_query} ":
        price_bindings = [item for item in numeric_bindings if item[1][1] == "$" or "$" in _binding_source_text(item[2])]
        if len(price_bindings) < 2:
            return None, []
        original = final = None
        for item in price_bindings:
            source = _normalize_text(_binding_source_text(item[2]))
            if any(marker in f" {source} " for marker in (" original ", " originally ", " original price ", " priced at ")):
                if original is None or item[1][0] > original[1][0]:
                    original = item
            if any(marker in f" {source} " for marker in (" after discount ", " after a discount ", " sale ", " got the book for ", " paid ")):
                if final is None or item[1][0] < final[1][0]:
                    final = item
        ordered_prices = sorted(price_bindings, key=lambda item: item[1][0])
        final = final or ordered_prices[0]
        original = original or ordered_prices[-1]
        if original[1][0] <= 0 or final[1][0] > original[1][0]:
            return None, []
        discount_pct = ((original[1][0] - final[1][0]) / original[1][0]) * 100.0
        support = [str(final[2].get("unit_id") or "").strip(), str(original[2].get("unit_id") or "").strip()]
        return _format_numeric_answer(discount_pct, "", "%"), [unit_id for unit_id in support if unit_id]
    positive = [item for item in numeric_bindings if item[1][0] > 0]
    if len(positive) < 2:
        return None, []
    numerator = min(positive, key=lambda item: item[1][0])
    denominator = max(positive, key=lambda item: item[1][0])
    if numerator[1][0] > denominator[1][0]:
        return None, []
    percentage = (numerator[1][0] / denominator[1][0]) * 100.0
    support = [str(numerator[2].get("unit_id") or "").strip(), str(denominator[2].get("unit_id") or "").strip()]
    return _format_numeric_answer(percentage, "", "%"), [unit_id for unit_id in support if unit_id]


def _collect_numeric_mentions_for_query(*, row: dict[str, Any], compiled_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_query = _normalize_text(str(row.get("query", "")))
    focus_tokens = _query_focus_tokens(row)
    requires_money = _numeric_query_requires_money(row)
    requires_distance = _numeric_query_requires_distance(row)
    requested_duration = _requested_duration_unit(row)
    mentions: list[dict[str, Any]] = []

    for unit in compiled_units:
        if not isinstance(unit, dict) or not _compiled_unit_has_predicate_alignment(unit, row):
            continue
        source_text = _compiled_unit_body_text(unit)
        normalized_source = _normalize_text(source_text)
        unit_id = str(unit.get("unit_id") or "").strip()
        for number, prefix, mention_unit, span in _compiled_unit_numeric_mentions(unit):
            normalized_unit = str(mention_unit or "").lower()
            if normalized_unit in {"and", "or"}:
                normalized_unit = ""
            score = 0.0
            if requires_money:
                if prefix != "$" and normalized_unit not in {"usd", "dollar", "dollars"}:
                    continue
                score += 2.0
            elif requires_distance:
                if normalized_unit not in {"mile", "miles", "km", "kilometer", "kilometers"}:
                    continue
                score += 2.0
            elif requested_duration:
                if not normalized_unit.startswith(requested_duration):
                    if normalized_unit in _DURATION_UNITS:
                        continue
                else:
                    score += 1.5
            elif normalized_unit in _DURATION_UNITS and not any(marker in normalized_query for marker in (" how long ", " duration ", " lasted ")):
                score -= 2.0

            if " age " in f" {normalized_query} " and 0 < number < 120:
                score += 1.5
            if " gpa " in f" {normalized_query} " and 0 < number <= 5:
                score += 1.5
            if " followers " in f" {normalized_query} " and not normalized_unit:
                score += 1.0
            if " weight " in f" {normalized_query} " and normalized_unit in {"pound", "pounds", "lb", "lbs", "kg", "kilogram", "kilograms"}:
                score += 1.5
            if any(token in normalized_source for token in focus_tokens):
                score += 0.5
            mentions.append(
                {
                    "score": score,
                    "number": float(number),
                    "prefix": prefix,
                    "unit": normalized_unit,
                    "span": span,
                    "unit_id": unit_id,
                    "source_text": source_text,
                    "compiled_unit": unit,
                }
            )
    return mentions


def _execute_average_aggregate_from_compiled_units(*, row: dict[str, Any], compiled_units: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    mentions = _collect_numeric_mentions_for_query(row=row, compiled_units=compiled_units)
    if not mentions:
        return None, []
    positive = [mention for mention in mentions if mention["score"] > 0]
    candidates = positive if len(positive) >= 2 else mentions
    if len(candidates) < 2:
        return None, []
    prefix = candidates[0]["prefix"]
    non_empty_units = [mention["unit"] for mention in candidates if mention["unit"]]
    unit = non_empty_units[0] if non_empty_units and len(set(non_empty_units)) == 1 else ""
    average = sum(mention["number"] for mention in candidates) / float(len(candidates))
    supporting = list(dict.fromkeys(mention["unit_id"] for mention in candidates if mention["unit_id"]))
    return _format_numeric_answer(average, prefix, unit), supporting


def _execute_delta_aggregate_from_compiled_units(*, row: dict[str, Any], compiled_units: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    mentions = _collect_numeric_mentions_for_query(row=row, compiled_units=compiled_units)
    if not mentions:
        return None, []
    positive = [mention for mention in mentions if mention["score"] > 0]
    candidates = positive if positive else mentions
    for mention in candidates:
        normalized_source = _normalize_text(mention["source_text"])
        if any(marker in normalized_source for marker in (" lost ", " gain ", " gained ", " increase ", " increased ", " decrease ", " decreased ", " dropped ")):
            return _format_numeric_answer(mention["number"], mention["prefix"], mention["unit"]), [mention["unit_id"]] if mention["unit_id"] else []
    if len(candidates) < 2:
        return None, []
    candidates = sorted(candidates, key=lambda mention: (mention["score"], mention["number"]), reverse=True)
    top = candidates[:2]
    delta = abs(top[0]["number"] - top[1]["number"])
    prefix = top[0]["prefix"] or top[1]["prefix"]
    unit = top[0]["unit"] or top[1]["unit"]
    supporting = list(dict.fromkeys(mention["unit_id"] for mention in top if mention["unit_id"]))
    return _format_numeric_answer(delta, prefix, unit), supporting


def _extremum_surface_text(*, unit: dict[str, Any], row: dict[str, Any]) -> str | None:
    source_text = _compiled_unit_body_text(unit)
    for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', source_text):
        quoted = str(match.group(1) or match.group(2) or "").strip()
        if quoted and _normalize_text(quoted) not in _EXTREMUM_GENERIC_TOKENS:
            return quoted
    phrases: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\b(?:[A-Z][A-Za-z0-9&.'-]*)(?:\s+[A-Z][A-Za-z0-9&.'-]*)*\b", source_text):
        phrase = str(match.group(0) or "").strip()
        normalized = _normalize_text(phrase)
        if not normalized or normalized in _EXTREMUM_GENERIC_TOKENS:
            continue
        if normalized in {
            "i",
            "my",
            "by",
            "the",
            "a",
            "an",
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        }:
            continue
        phrases.append((len(normalized.split()), len(phrase), phrase))
    if phrases:
        phrases.sort(reverse=True)
        return phrases[0][2]
    focus_tokens = _query_focus_tokens(row)
    for token_group in ("entity_tokens", "value_tokens", "subject_tokens"):
        for token in unit.get(token_group) or []:
            normalized = _normalize_text(str(token))
            if not normalized or normalized in _EXTREMUM_GENERIC_TOKENS or normalized in focus_tokens:
                continue
            return str(token)
    return None


def _execute_extremum_selection_from_compiled_units(*, row: dict[str, Any], plan: EvidencePlan, compiled_units: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    mentions = _collect_numeric_mentions_for_query(row=row, compiled_units=compiled_units)
    if not mentions:
        return None, []
    grouped: dict[str, list[dict[str, Any]]] = {}
    compiled_unit_by_id: dict[str, dict[str, Any]] = {}
    for mention in mentions:
        unit_id = str(mention["unit_id"] or "")
        if not unit_id:
            continue
        grouped.setdefault(unit_id, []).append(mention)
        compiled_unit_by_id[unit_id] = mention["compiled_unit"]
    if len(grouped) < 2:
        return None, []
    direction = str((plan.plan_metadata or {}).get("extremum_direction") or "max").strip().lower()
    scored: list[tuple[float, str, str]] = []
    for unit_id, unit_mentions in grouped.items():
        positive = [mention for mention in unit_mentions if mention["score"] > 0]
        candidates = positive if positive else unit_mentions
        normalized_source = _normalize_text(str(candidates[0]["source_text"] or ""))
        if len(candidates) >= 2 and any(marker in normalized_source for marker in (" from ", " to ", " increased ", " increase ", " jumped ", " gained ", " gain ")):
            magnitude = max(mention["number"] for mention in candidates) - min(mention["number"] for mention in candidates)
        else:
            magnitude = max(mention["number"] for mention in candidates)
        surface = _extremum_surface_text(unit=compiled_unit_by_id[unit_id], row=row)
        if not surface:
            continue
        scored.append((magnitude, surface, unit_id))
    if len(scored) < 2:
        return None, []
    chosen = min(scored, key=lambda item: item[0]) if direction == "min" else max(scored, key=lambda item: item[0])
    return chosen[1], [chosen[2]]


def _execute_distinct_count_from_compiled_units(*, row: dict[str, Any], plan: EvidencePlan, compiled_units: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    if str(plan.schema_name or "") not in {"CountDistinctItems", "Aggregate", "CountEvents"}:
        return None, []
    if not _is_distinct_count_query(row, plan):
        return None, []
    count, supporting = _count_distinct_item_signatures(compiled_units, row)
    if count < 2 or len(supporting) < 2:
        return None, []
    return str(count), supporting


def _execute_direct_duration_from_compiled_units(*, row: dict[str, Any], compiled_units: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    target_unit = _requested_duration_unit(row)
    if target_unit not in {None, "day", "week", "month", "year", "hour", "minute"}:
        return None, []
    for unit in compiled_units:
        if not isinstance(unit, dict) or not _compiled_unit_has_focus_alignment(unit, row):
            continue
        for number, _, mention_unit, span in _compiled_unit_numeric_mentions(unit):
            normalized_unit = str(mention_unit or "").lower()
            if target_unit is None and normalized_unit in {"day", "days", "week", "weeks", "month", "months", "year", "years", "hour", "hours", "minute", "minutes"}:
                unit_id = str(unit.get("unit_id") or "").strip()
                return (_format_numeric_answer(number, "", normalized_unit), [unit_id] if unit_id else [])
            if target_unit and normalized_unit.startswith(target_unit):
                unit_id = str(unit.get("unit_id") or "").strip()
                return (_format_numeric_answer(number, "", normalized_unit), [unit_id] if unit_id else [])
            if target_unit == "hour" and normalized_unit.startswith("minute"):
                unit_id = str(unit.get("unit_id") or "").strip()
                return (_format_numeric_answer(number / 60.0, "", "hours"), [unit_id] if unit_id else [])
            if span:
                continue
    return None, []


def _calendar_delta(date_a: date, date_b: date, unit: str) -> tuple[float, bool] | None:
    """Return (value, discrete) for calendar-aware delta, or None if unsupported."""
    delta_days = abs((date_b - date_a).days)
    if unit == "week":
        return delta_days / 7.0, True
    if unit == "month":
        earlier, later = (date_a, date_b) if date_a <= date_b else (date_b, date_a)
        whole = (later.year - earlier.year) * 12 + (later.month - earlier.month)
        if later.day < earlier.day:
            whole -= 1
        return (float(whole) if whole > 0 else delta_days / 30.0), True
    if unit == "year":
        earlier, later = (date_a, date_b) if date_a <= date_b else (date_b, date_a)
        whole = later.year - earlier.year
        if (later.month, later.day) < (earlier.month, earlier.day):
            whole -= 1
        return (float(whole) if whole > 0 else delta_days / 365.0), True
    if unit in {"hour", "minute"}:
        return None
    return float(delta_days), False


def _execute_relative_time(*, row: dict[str, Any], validated_bindings: dict[str, dict[str, Any] | None], displayed_bindings: dict[str, dict[str, Any] | None]) -> tuple[str | None, list[str]]:
    event_binding = validated_bindings.get("event") or displayed_bindings.get("event")
    if not isinstance(event_binding, dict):
        return None, []
    event_date = _extract_event_date(_binding_source_text(event_binding), _binding_anchor_date(event_binding))
    if event_date is None:
        return None, []
    reference_date = None
    supporting = [str(event_binding.get("unit_id") or "")] if str(event_binding.get("unit_id") or "").strip() else []
    reference_binding = validated_bindings.get("reference_time") or displayed_bindings.get("reference_time")
    if isinstance(reference_binding, dict):
        if str(reference_binding.get("memory_id") or "") == "question_date":
            reference_date = _binding_anchor_date(reference_binding)
            unit_id = str(reference_binding.get("unit_id") or "").strip()
            if reference_date is None:
                return None, supporting
            if unit_id and unit_id not in supporting:
                supporting.append(unit_id)
        if _query_has_relative_reference_clause(row):
            extracted_reference = _extract_event_date(_binding_source_text(reference_binding), _binding_anchor_date(reference_binding))
            if extracted_reference is not None:
                reference_date = extracted_reference
                unit_id = str(reference_binding.get("unit_id") or "").strip()
                if unit_id and unit_id not in supporting:
                    supporting.append(unit_id)
        else:
            reference_date = _extract_event_date(_binding_source_text(reference_binding), _binding_anchor_date(reference_binding))
            unit_id = str(reference_binding.get("unit_id") or "").strip()
            if unit_id and unit_id not in supporting:
                supporting.append(unit_id)
    if reference_date is None:
        return None, []
    unit = _time_unit_from_query(row)
    result = _calendar_delta(event_date, reference_date, unit)
    if result is None:
        return None, []
    value, discrete_unit = result
    value_text = str(int(round(value))) if discrete_unit else (str(int(value)) if value.is_integer() else f"{value:.1f}".rstrip("0").rstrip("."))
    suffix = unit if value_text == "1" else f"{unit}s"
    return f"{value_text} {suffix}", supporting


def _execute_ordered_choice(*, row: dict[str, Any], query_targets: QueryTargets, validated_bindings: dict[str, dict[str, Any] | None]) -> tuple[str | None, list[str]]:
    def _ordering_surface(binding: dict[str, Any]) -> str:
        source_text = _binding_source_text(binding)
        primary_entities = tuple(entity for entity in query_targets.subject_entities if entity) or tuple(
            entity for entity in query_targets.candidate_entities if entity
        )
        for match in re.finditer(r'"([^"]+)"|\'([^\']+)\'', source_text):
            quoted = str(match.group(1) or match.group(2) or "").strip()
            if quoted and _best_matching_entity(quoted, primary_entities):
                return quoted
        matched = _best_matching_entity(source_text, primary_entities)
        if matched:
            return matched
        return str(binding.get("display_text") or binding.get("text") or "").strip()

    choices: list[tuple[date, str, str]] = []
    seen_labels: set[str] = set()
    for slot_name in ("choice_a", "choice_b"):
        binding = validated_bindings.get(slot_name)
        if not isinstance(binding, dict):
            return None, []
        source_text = _binding_source_text(binding)
        label = _ordering_surface(binding)
        normalized_label = _normalize_text(label)
        if not normalized_label or normalized_label in seen_labels:
            return None, []
        seen_labels.add(normalized_label)
        event_date = _extract_event_date(source_text, _binding_anchor_date(binding))
        if event_date is None:
            return None, []
        choices.append((event_date, label, str(binding.get("unit_id") or "")))
    normalized_query = _normalize_text(str(row.get("query", "")))
    wants_latest = any(marker in normalized_query for marker in (" latest", " most recent", " happened last", " last "))
    chosen = max(choices, key=lambda item: item[0]) if wants_latest else min(choices, key=lambda item: item[0])
    return chosen[1], [chosen[2]] if chosen[2] else []


def _execute_comparison(*, row: dict[str, Any], query_targets: QueryTargets, validated_bindings: dict[str, dict[str, Any] | None]) -> tuple[str | None, list[str]]:
    left = validated_bindings.get("left_operand")
    right = validated_bindings.get("right_operand")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None, []
    primary_entities = tuple(entity for entity in query_targets.subject_entities if entity) or tuple(
        entity for entity in query_targets.candidate_entities if entity
    )
    if len(primary_entities) >= 2:
        left_match = _best_matching_entity(_binding_source_text(left), primary_entities)
        right_match = _best_matching_entity(_binding_source_text(right), primary_entities)
        expected_left = primary_entities[0]
        expected_right = primary_entities[1]
        if left_match and right_match and _normalize_text(left_match) == _normalize_text(expected_right) and _normalize_text(right_match) == _normalize_text(expected_left):
            left, right = right, left
    parsed_left = _parse_numeric_text(str(left.get("text") or ""))
    parsed_right = _parse_numeric_text(str(right.get("text") or ""))
    if parsed_left is None or parsed_right is None:
        return None, []
    if parsed_left[1] != parsed_right[1] and parsed_left[1] and parsed_right[1]:
        return None, []
    normalized_query = _normalize_text(str(row.get("query", "")))
    if " less " in normalized_query or " lower " in normalized_query:
        total = parsed_right[0] - parsed_left[0]
        prefix = parsed_right[1] or parsed_left[1]
        unit = parsed_right[2] or parsed_left[2]
    else:
        total = parsed_left[0] - parsed_right[0]
        prefix = parsed_left[1] or parsed_right[1]
        unit = parsed_left[2] or parsed_right[2]
    return _format_numeric_answer(abs(total), prefix, unit), [str(left.get("unit_id") or ""), str(right.get("unit_id") or "")]


def _execute_temporal_interval(*, row: dict[str, Any], validated_bindings: dict[str, dict[str, Any] | None]) -> tuple[str | None, list[str]]:
    left = validated_bindings.get("time_a")
    right = validated_bindings.get("time_b")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None, []
    left_date = _extract_event_date(_binding_source_text(left), _binding_anchor_date(left))
    right_date = _extract_event_date(_binding_source_text(right), _binding_anchor_date(right))
    if left_date is None or right_date is None:
        return None, []
    unit = _time_unit_from_query(row)
    result = _calendar_delta(left_date, right_date, unit)
    if result is None:
        return None, []
    value, _ = result
    value_text = str(int(value)) if value.is_integer() else f"{value:.1f}".rstrip("0").rstrip(".")
    suffix = unit if value_text == "1" else f"{unit}s"
    return f"{value_text} {suffix}", [str(left.get("unit_id") or ""), str(right.get("unit_id") or "")]
