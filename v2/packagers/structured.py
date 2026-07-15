"""Structured packager: re-render the v1 SIEVE package at a requested budget.

This wraps v1 as one representation option (per ``research/idea-v2.md``: the v1
structured compiler is retained as one expert, not discarded). It pulls the
already-computed ``rendered_evidence_package`` from the SIEVE compilation-cache
entry for this example_id and re-renders its slots + evidence lines as text
units, then budget-truncates on whole-unit boundaries.

It does NOT re-run the v1 compiler — the cache is authoritative for the
structured content, and re-rendering is a pure, offline transform.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from v2.budget import count_tokens, truncate_units_to_budget
from v2.packagers.base import PackagerContext, row_example_id
from v2.types import EvidenceVariant

STYLE_NAME = "structured"


def structured_units(rendered_evidence_package: Mapping[str, Any] | None) -> list[str]:
    """Flatten a v1 rendered_evidence_package into prioritized text units.

    Order: validated slots first (they carry the compiler's synthesised answer
    hints), then evidence lines. Each becomes one whole unit so budget
    truncation never splits a slot or a line mid-sentence.
    """
    if not isinstance(rendered_evidence_package, Mapping):
        return []
    units: list[str] = []

    slots = (
        rendered_evidence_package.get("validated_slots")
        or rendered_evidence_package.get("slots")
        or {}
    )
    if isinstance(slots, Mapping):
        for slot_name in sorted(slots):  # sorted iteration -> determinism
            payload = slots[slot_name]
            if not isinstance(payload, Mapping):
                continue
            text = str(payload.get("display_text") or payload.get("text") or "").strip()
            if text:
                units.append(f"- {slot_name}: {text}")

    evidence_lines = rendered_evidence_package.get("evidence_lines")
    if isinstance(evidence_lines, list):
        for item in evidence_lines:
            if not isinstance(item, Mapping):
                continue
            text = str(item.get("text") or "").strip()
            if text:
                units.append(text)
    return units


def effective_units(entry: Mapping[str, Any] | None) -> tuple[list[str], str]:
    """Return v1 structured units or the route's effective evidence fallback."""
    if not isinstance(entry, Mapping):
        return [], "missing_cache"
    rendered = entry.get("rendered_evidence_package")
    units = structured_units(rendered if isinstance(rendered, Mapping) else None)
    if units:
        return units, "structured"
    fallback = [
        str(text).strip()
        for text in entry.get("selected_memory_texts") or []
        if str(text).strip()
    ]
    return fallback, "v1_route_fallback"


class StructuredPackager:
    """Re-render the cached v1 SIEVE package at the requested budget."""

    style_name = STYLE_NAME

    def package(
        self, row: Mapping[str, Any], budget: int, ctx: PackagerContext
    ) -> EvidenceVariant:
        entry = ctx.sieve_entry
        units, representation_mode = effective_units(entry)
        packing = truncate_units_to_budget(units, budget, encoding=ctx.encoding)
        evidence_text = "\n".join(packing.units)
        source_ids = (
            tuple(str(mid) for mid in entry.get("selected_memory_ids") or [])
            if isinstance(entry, Mapping)
            else ()
        )

        return EvidenceVariant(
            example_id=row_example_id(row),
            style=self.style_name,
            budget=budget,
            evidence_text=evidence_text,
            token_count=count_tokens(evidence_text, encoding=ctx.encoding),
            source_memory_ids=source_ids,
            meta={
                "has_sieve_entry": entry is not None,
                "n_units": len(units),
                "n_kept": len(packing.units),
                "overflow": packing.overflow,
                "representation_mode": representation_mode,
                "reader_route_reason": (
                    entry.get("reader_route_reason")
                    if isinstance(entry, Mapping)
                    else None
                ),
                "schema_name": (entry.get("schema_name") if isinstance(entry, Mapping) else None),
            },
        )
