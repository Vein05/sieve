"""SIEVE-v1 effective evidence plus verbatim selected source quotations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from v2.budget import count_tokens, hard_truncate_text, truncate_units_to_budget
from v2.packagers.base import PackagerContext, candidate_memories, row_example_id
from v2.packagers.structured import effective_units
from v2.types import EvidenceVariant

STYLE_NAME = "structured_quotes"
DEFAULT_STRUCTURED_FRACTION = 0.6
STRUCTURED_LABEL = "Structured:"
QUOTES_LABEL = "Exact quotes:"


def _pack_section(
    label: str,
    units: list[str],
    budget: int,
    encoding: str,
) -> tuple[list[str], int, bool]:
    """Pack a labelled section while charging the label to its budget."""
    if not units:
        return [], 0, False
    if budget == 0:
        lines = [label, *units]
        return lines, count_tokens("\n".join(lines), encoding=encoding), False
    label_cost = count_tokens(f"{label}\n", encoding=encoding)
    if budget <= label_cost:
        return [], 0, True
    packing = truncate_units_to_budget(
        units,
        budget - label_cost,
        encoding=encoding,
    )
    lines = [label, *packing.units] if packing.units else []
    text = "\n".join(lines)
    return lines, count_tokens(text, encoding=encoding), packing.overflow


def _selected_quotes(
    row: Mapping[str, Any], entry: Mapping[str, Any] | None
) -> tuple[list[str], list[str]]:
    if not isinstance(entry, Mapping):
        return [], []
    selected_ids = [str(mid) for mid in entry.get("selected_memory_ids") or []]
    by_id = {
        str(candidate.get("memory_id", "")): str(candidate["content"]).strip()
        for candidate in candidate_memories(row)
    }
    kept_ids = [mid for mid in selected_ids if mid in by_id and by_id[mid]]
    return [by_id[mid] for mid in kept_ids], kept_ids


def _pack_sections(
    structured: list[str],
    quotes: list[str],
    budget: int,
    fraction: float,
    encoding: str,
) -> tuple[list[str], int, int, bool]:
    if budget == 0:
        s_lines, _, s_overflow = _pack_section(
            STRUCTURED_LABEL, structured, 0, encoding
        )
        q_lines, _, q_overflow = _pack_section(QUOTES_LABEL, quotes, 0, encoding)
        return [*s_lines, *q_lines], len(s_lines[1:]), len(q_lines[1:]), (
            s_overflow or q_overflow
        )
    structured_target = int(round(budget * fraction))
    s_lines, s_cost, s_overflow = _pack_section(
        STRUCTURED_LABEL, structured, structured_target, encoding
    )
    q_lines, q_cost, q_overflow = _pack_section(
        QUOTES_LABEL, quotes, max(0, budget - s_cost), encoding
    )
    unused = budget - count_tokens("\n".join([*s_lines, *q_lines]), encoding=encoding)
    if unused > 0 and len(s_lines[1:]) < len(structured):
        s_lines, s_cost, s_overflow = _pack_section(
            STRUCTURED_LABEL, structured, structured_target + unused, encoding
        )
        q_lines, q_cost, q_overflow = _pack_section(
            QUOTES_LABEL, quotes, max(0, budget - s_cost), encoding
        )
    return [*s_lines, *q_lines], len(s_lines[1:]), len(q_lines[1:]), (
        s_overflow or q_overflow
    )


class StructuredQuotesPackager:
    """Combine the effective v1 package with exact selected-source quotations."""

    style_name = STYLE_NAME

    def package(
        self, row: Mapping[str, Any], budget: int, ctx: PackagerContext
    ) -> EvidenceVariant:
        entry = ctx.sieve_entry
        structured, mode = effective_units(entry)
        quotes, quote_ids = _selected_quotes(row, entry)
        fraction = float(ctx.params.get("structured_fraction", DEFAULT_STRUCTURED_FRACTION))
        fraction = min(max(fraction, 0.0), 1.0)
        lines, n_structured, n_quotes, overflow = _pack_sections(
            structured, quotes, budget, fraction, ctx.encoding
        )
        evidence_text = "\n".join(lines)
        if budget > 0 and count_tokens(evidence_text, encoding=ctx.encoding) > budget:
            evidence_text = hard_truncate_text(
                evidence_text, budget, encoding=ctx.encoding
            )
            overflow = True
        return EvidenceVariant(
            example_id=row_example_id(row),
            style=self.style_name,
            budget=budget,
            evidence_text=evidence_text,
            token_count=count_tokens(evidence_text, encoding=ctx.encoding),
            source_memory_ids=tuple(quote_ids[:n_quotes]),
            meta={
                "structured_fraction": fraction,
                "representation_mode": mode,
                "n_structured_kept": n_structured,
                "n_quotes_kept": n_quotes,
                "overflow": overflow,
            },
        )
