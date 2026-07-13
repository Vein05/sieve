"""Query intent classification for numeric and aggregate queries."""

from __future__ import annotations

import re
from typing import Any

from evidence.schema import EvidencePlan
from ...runtime.bindings import (
    _normalize_text,
    _aggregate_slot,
    _requested_duration_unit,
)
from ..requirements import is_aggregate_schema, is_count_aggregate_schema, is_direct_lookup_schema

def _numeric_query_requires_money(row: dict[str, Any]) -> bool:
    normalized = _normalize_text(str(row.get("query", "")))
    return any(token in f" {normalized} " for token in (" cost ", " price ", " save ", " spent ", " amount ", " worth ", " money ", " dollar ", " dollars "))

def _numeric_query_requires_distance(row: dict[str, Any]) -> bool:
    normalized = _normalize_text(str(row.get("query", "")))
    return any(token in normalized for token in (" distance ", " mile ", " miles ", " km ", " kilometer "))

def _numeric_query_requires_speed(row: dict[str, Any]) -> bool:
    normalized = _normalize_text(str(row.get("query", "")))
    padded = f" {normalized} "
    return " speed " in padded or " mbps " in padded or " gbps " in padded or " mph " in padded

def _requested_distance_unit(row: dict[str, Any]) -> str | None:
    query = _normalize_text(str(row.get("query", "")))
    for unit in ("mile", "meter", "kilometer", "km", "yard", "feet", "foot"):
        if f"{unit}" in query:
            return unit
    return None

def _is_count_query(row: dict[str, Any]) -> bool:
    normalized = _normalize_text(str(row.get("query", "")))
    return bool(
        " how many " in f" {normalized} "
        or " number of " in f" {normalized} "
        or re.search(r"\bhow\s+often\b", normalized)
    )

def _is_sum_query(row: dict[str, Any]) -> bool:
    normalized = _normalize_text(str(row.get("query", "")))
    padded = f" {normalized} "
    if any(marker in padded for marker in (" total ", " in total ", " combined ", " altogether ", " overall ")):
        return True
    if " how much " in padded and any(
        marker in padded
        for marker in (" spent ", " spend ", " earned ", " earn ", " raised ", " raise ", " saved ", " save ", " cost ", " costs ", " amount ")
    ):
        return True
    return False

def _is_percentage_query(row: dict[str, Any]) -> bool:
    normalized = _normalize_text(str(row.get("query", "")))
    padded = f" {normalized} "
    return any(marker in padded for marker in (" percent ", " percentage ", " ratio "))

def _is_explicit_multi_count_query(row: dict[str, Any]) -> bool:
    normalized = _normalize_text(str(row.get("query", "")))
    padded = f" {normalized} "
    return any(
        marker in padded
        for marker in (
            " different ",
            " types of ",
            " across ",
            " both ",
            " times ",
            " including ",
            " excluding ",
            " simultaneously ",
        )
    )

def _is_simple_numeric_lookup_query(row: dict[str, Any]) -> bool:
    return (_is_count_query(row) or _requested_duration_unit(row) is not None or " how much " in f" {_normalize_text(str(row.get('query', '')))} ") and not _is_sum_query(row) and not _is_explicit_multi_count_query(row)

def _is_distinct_count_query(row: dict[str, Any], plan: EvidencePlan) -> bool:
    if plan.family not in {"aggregation", "information_extraction"}:
        return False
    if not is_direct_lookup_schema(plan) and not is_aggregate_schema(plan):
        return False
    if is_aggregate_schema(plan):
        aggregate_slot = _aggregate_slot(plan)
        if aggregate_slot is None or str(aggregate_slot.slot_type) not in {"count_item", "count_evidence", "support"}:
            return False
        if not is_count_aggregate_schema(plan):
            return False
    normalized = _normalize_text(str(row.get("query", "")))
    padded = f" {normalized} "
    if " how many " not in padded and " number of " not in padded:
        return False
    if _requested_duration_unit(row) is not None:
        return False
    if any(marker in padded for marker in (" ago ", " how long ", " how often ")):
        return False
    if _is_sum_query(row):
        return False
    if _numeric_query_requires_money(row) or _numeric_query_requires_distance(row):
        return False
    return True

def _is_multi_count_query(row: dict[str, Any]) -> bool:
    query = _normalize_text(str(row.get("query", "")))
    return any(token in f" {query} " for token in (" total ", " in total ", " altogether ", " combined ", " across ", " both ", " all ", " times "))

def _is_count_lookup_query(row: dict[str, Any]) -> bool:
    normalized = _normalize_text(str(row.get("query", "")))
    return bool(
        normalized.startswith("how many ")
        and not re.match(r"how many (days?|weeks?|months?|years?|hours?|minutes?)\b", normalized)
        and not _is_sum_query(row)
        and " different " not in f" {normalized} "
        and " types of " not in f" {normalized} "
        and " number of " not in f" {normalized} "
    )
