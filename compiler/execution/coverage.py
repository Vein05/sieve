"""Coverage helpers for compiler execution selection."""

from __future__ import annotations

from typing import Any

from evidence.schema import EvidencePlan
from evidence.units import EvidenceUnit
from ..runtime.bindings import _unit_requirement_tags_cached
from .requirements import coverage_dict as _coverage_dict


def _coverage_for_selected_units(
    *,
    requirements: dict[str, int],
    selected_units: list[EvidenceUnit],
    row: dict[str, Any],
    plan: EvidencePlan,
    query_targets=None,
    memory_labels_by_id: dict[str, dict[str, Any]],
) -> dict[str, int]:
    coverage = _coverage_dict(requirements)
    role_cache: dict[str, set[str]] = {}
    for unit in selected_units:
        unit_tags = _unit_requirement_tags_cached(
            row=row,
            plan=plan,
            query_targets=query_targets,
            unit=unit,
            memory_labels_by_id=memory_labels_by_id,
            role_cache=role_cache,
        )
        for tag in coverage:
            if tag in unit_tags:
                coverage[tag] += 1
    return coverage
