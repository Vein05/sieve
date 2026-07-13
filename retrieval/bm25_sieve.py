"""BM25 retrieval + SIEVE evidence compilation.

Uses BM25 to rerank the candidate pool, then feeds the top-k to the
SIEVE compiler for structured evidence compilation.  This tests the
paper's claim that SIEVE improves any retrieval backend's output.
"""

from __future__ import annotations

from typing import Any

from shared.nlp import candidate_views
from compiler.pipeline import run_sieve_pipeline
from .baselines import bm25_scores


def method_bm25_sieve(
    row: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    candidates = candidate_views(row)
    scores = bm25_scores(str(row.get("query", "")), candidates)

    # Rank by BM25 score and keep top-k
    top_k = int(context.get("retrieval_top_k", 8))
    ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])
    top_ids = {c.memory_id for _, c in ranked[:top_k]}

    # Build a filtered row with only BM25's top-k candidates
    filtered_row = dict(row)
    filtered_row["candidate_memories"] = [
        m for m in row.get("candidate_memories", [])
        if m.get("memory_id") in top_ids
    ]

    # Run SIEVE on BM25-filtered candidates
    result = run_sieve_pipeline(filtered_row, context)

    # Tag the selection meta so we know the upstream retriever
    meta = result.get("selection_meta", {})
    meta["upstream_retriever"] = "bm25"
    meta["upstream_top_k"] = top_k
    result["selection_meta"] = meta

    return result
