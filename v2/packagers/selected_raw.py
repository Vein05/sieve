"""Matched-content raw control over the memories selected by SIEVE-v1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from v2.budget import count_tokens, truncate_units_to_budget
from v2.packagers.base import (
    PackagerContext,
    row_example_id,
    selected_candidate_memories,
)
from v2.types import EvidenceVariant

STYLE_NAME = "selected_raw"


class SelectedRawPackager:
    """Render the v1-selected content verbatim, without structured rewriting."""

    style_name = STYLE_NAME

    def package(
        self, row: Mapping[str, Any], budget: int, ctx: PackagerContext
    ) -> EvidenceVariant:
        candidates = selected_candidate_memories(row, ctx.sieve_entry)
        units = [str(candidate["content"]).strip() for candidate in candidates]
        packing = truncate_units_to_budget(units, budget, encoding=ctx.encoding)
        kept_ids = tuple(
            str(candidate.get("memory_id", ""))
            for candidate in candidates[: len(packing.units)]
        )
        evidence_text = "\n".join(packing.units)
        return EvidenceVariant(
            example_id=row_example_id(row),
            style=self.style_name,
            budget=budget,
            evidence_text=evidence_text,
            token_count=count_tokens(evidence_text, encoding=ctx.encoding),
            source_memory_ids=kept_ids,
            meta={
                "n_selected": len(candidates),
                "n_kept": len(packing.units),
                "overflow": packing.overflow,
            },
        )
