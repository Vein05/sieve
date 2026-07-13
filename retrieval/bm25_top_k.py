"""BM25 reranker over the fixed candidate memory pool."""

from __future__ import annotations

from typing import Any

from shared.nlp import candidate_views
from .baselines import bm25_scores, ranked_fixed_top_k_result


def method_bm25_top_k(
    row: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    candidates = candidate_views(row)
    scores = bm25_scores(str(row.get("query", "")), candidates)
    return ranked_fixed_top_k_result(
        row=row,
        scores=scores,
        mode_name="bm25_top_k",
        reason_codes=["bm25_rerank"],
    )
