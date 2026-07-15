"""Guarded structured packager: structured_generic with a filtered_raw fallback.

Runs :class:`StructuredGenericPackager`, then measures how many query-relevant
passages the structured rendering actually covers (stem-overlap coverage). If
coverage falls below :data:`COVERAGE_MIN_FRACTION`, the structured pack is
discarded and ``filtered_raw`` output is emitted instead. The path that fired
(``structured`` vs ``filtered_raw``) is recorded in the variant metadata so the
pilot can report guard-fire rates.

This is the simplest form of the do-no-harm fallback (proposal S268): when the
structured representation would starve the reader of relevant evidence, ship
verbatim survivors instead of a sparse structure.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from v2.packagers.base import (
    PackagerContext,
    candidate_memories,
    row_query,
)
from v2.packagers.filtered_raw import FilteredRawPackager
from v2.packagers.structured_generic import StructuredGenericPackager
from v2.scorer import StemOverlapScorer
from v2.types import EvidenceVariant

STYLE_NAME = "guarded_structured"

# A passage is "query-relevant" if its stem-overlap score exceeds this floor.
_RELEVANCE_FLOOR = 0.0

# Guard fires (fall back to filtered_raw) when the structured rendering covers
# fewer than this fraction of query-relevant passages.
COVERAGE_MIN_FRACTION = 0.5

_PATH_STRUCTURED = "structured"
_PATH_FILTERED_RAW = "filtered_raw"


def _relevant_passages(
    candidates: list[dict[str, Any]], query: str, scorer: Any
) -> list[str]:
    contents = [str(c["content"]).strip() for c in candidates]
    scores = scorer(query, contents) if callable(scorer) else scorer.score(query, contents)
    return [
        content
        for content, score in zip(contents, scores)
        if float(score) > _RELEVANCE_FLOOR
    ]


def _coverage(relevant: list[str], evidence_text: str) -> float:
    """Fraction of query-relevant passages present verbatim in the rendering."""
    if not relevant:
        return 1.0
    covered = sum(1 for passage in relevant if passage and passage in evidence_text)
    return covered / len(relevant)


class GuardedStructuredPackager:
    """structured_generic, with a filtered_raw fallback on low coverage."""

    style_name = STYLE_NAME

    def package(
        self, row: Mapping[str, Any], budget: int, ctx: PackagerContext
    ) -> EvidenceVariant:
        structured = StructuredGenericPackager().package(row, budget, ctx)
        scorer = ctx.relevance_scorer or StemOverlapScorer()
        relevant = _relevant_passages(candidate_memories(row), row_query(row), scorer)
        coverage = _coverage(relevant, structured.evidence_text)

        if coverage >= COVERAGE_MIN_FRACTION:
            return self._relabel(structured, _PATH_STRUCTURED, coverage, len(relevant))

        fallback = FilteredRawPackager().package(row, budget, ctx)
        return self._relabel(fallback, _PATH_FILTERED_RAW, coverage, len(relevant))

    def _relabel(
        self,
        source: EvidenceVariant,
        path: str,
        coverage: float,
        n_relevant: int,
    ) -> EvidenceVariant:
        return EvidenceVariant(
            example_id=source.example_id,
            style=self.style_name,
            budget=source.budget,
            evidence_text=source.evidence_text,
            token_count=source.token_count,
            source_memory_ids=source.source_memory_ids,
            meta={
                "guard_path": path,
                "guard_fired": path == _PATH_FILTERED_RAW,
                "coverage": round(coverage, 6),
                "coverage_min_fraction": COVERAGE_MIN_FRACTION,
                "n_relevant_passages": n_relevant,
                "inner_style": source.style,
                "inner_meta": dict(source.meta),
            },
        )
