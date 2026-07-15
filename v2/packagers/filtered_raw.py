"""Filtered-raw packager: drop low-relevance turns, keep survivors verbatim.

Distractors are removed by relevance, but every surviving turn keeps its
ORIGINAL WORDING verbatim (no paraphrase, no restructuring). Turns above a
relative relevance threshold are kept, ordered by retrieval rank, then
budget-truncated on whole-turn boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from v2.budget import count_tokens, truncate_units_to_budget
from v2.packagers.base import (
    PackagerContext,
    candidate_memories,
    row_example_id,
    row_query,
)
from v2.scorer import StemOverlapScorer
from v2.types import EvidenceVariant

STYLE_NAME = "filtered_raw"
_DEFAULT_RELATIVE_THRESHOLD = 0.5


def _survivors(
    candidates: list[Mapping[str, Any]], scores: list[float], threshold: float
) -> list[tuple[int, Mapping[str, Any], str]]:
    max_score = max(scores) if scores else 0.0
    ranked = [
        (rank, candidate, str(candidate["content"]).strip())
        for rank, candidate in enumerate(candidates)
    ]
    if max_score <= 0.0:
        return ranked
    cutoff = threshold * max_score
    return [item for item, score in zip(ranked, scores) if score >= cutoff]


class FilteredRawPackager:
    """Keep high-relevance turns verbatim; drop distractors; rank-ordered."""

    style_name = STYLE_NAME

    def package(
        self, row: Mapping[str, Any], budget: int, ctx: PackagerContext
    ) -> EvidenceVariant:
        candidates = candidate_memories(row)
        query = row_query(row)
        contents = [str(c["content"]).strip() for c in candidates]

        scorer = ctx.relevance_scorer or StemOverlapScorer()
        scores = scorer(query, contents) if callable(scorer) else scorer.score(query, contents)

        threshold_frac = float(
            ctx.params.get("relative_threshold", _DEFAULT_RELATIVE_THRESHOLD)
        )
        max_score = max(scores) if scores else 0.0
        survivors = _survivors(candidates, scores, threshold_frac)

        # Order survivors by retrieval rank (input order), then budget-truncate.
        survivors.sort(key=lambda t: t[0])
        units = [text for _, _, text in survivors]
        packing = truncate_units_to_budget(units, budget, encoding=ctx.encoding)

        kept_ids = [
            str(c.get("memory_id", ""))
            for _, c, _ in survivors[: len(packing.units)]
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
                "n_survivors": len(survivors),
                "n_kept": len(packing.units),
                "relative_threshold": threshold_frac,
                "max_relevance": round(float(max_score), 6),
                "overflow": packing.overflow,
            },
        )
