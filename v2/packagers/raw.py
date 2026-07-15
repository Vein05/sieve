"""Raw packager: top-20 candidate memories in retrieval order (the control).

This is the control representation. It concatenates the pre-retrieved BM25
top-20 candidate memories in retrieval order and budget-truncates on whole-turn
boundaries. At the largest budget it reproduces the naive baseline evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from v2.budget import count_tokens, truncate_units_to_budget
from v2.packagers.base import (
    PackagerContext,
    candidate_memories,
    row_example_id,
)
from v2.types import EvidenceVariant

STYLE_NAME = "raw"


class RawPackager:
    """Concatenate candidate memories in retrieval order, budget-truncated."""

    style_name = STYLE_NAME

    def package(
        self, row: Mapping[str, Any], budget: int, ctx: PackagerContext
    ) -> EvidenceVariant:
        candidates = candidate_memories(row)
        # Retrieval order == list order; keep memory_ids aligned with units.
        units = [str(c["content"]).strip() for c in candidates]
        packing = truncate_units_to_budget(units, budget, encoding=ctx.encoding)

        kept_ids = [
            str(candidate.get("memory_id", ""))
            for candidate in candidates[: len(packing.units)]
        ]

        evidence_text = "\n".join(packing.units)
        return EvidenceVariant(
            example_id=row_example_id(row),
            style=self.style_name,
            budget=budget,
            evidence_text=evidence_text,
            token_count=count_tokens(evidence_text, encoding=ctx.encoding),
            source_memory_ids=tuple(kept_ids),
            meta={
                "n_candidates": len(candidates),
                "n_kept": len(packing.units),
                "overflow": packing.overflow,
                "uncapped": budget == 0,
            },
        )
