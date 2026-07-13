"""Shared helpers for retrieval-style baselines over the fixed candidate pool."""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any

from shared.nlp import (
    CandidateView,
    append_with_budget,
    candidate_views,
    shared_result,
    v2_budget_family_hint,
    v2_max_selected,
    v2_target_token_budget,
)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
DEFAULT_RETRIEVAL_BASELINE_TOP_K = 8


def lexical_tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


def bm25_scores(
    query: str,
    candidates: list[CandidateView],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    delta: float = 0.25,
) -> list[float]:
    if not candidates:
        return []
    query_terms = Counter(lexical_tokens(query))
    if not query_terms:
        return [0.0] * len(candidates)

    tokenized_docs = [lexical_tokens(candidate.text) for candidate in candidates]
    doc_term_freqs = [Counter(tokens) for tokens in tokenized_docs]
    doc_freqs: Counter[str] = Counter()
    for tokens in tokenized_docs:
        doc_freqs.update(set(tokens))

    avg_doc_len = sum(len(tokens) for tokens in tokenized_docs) / max(1, len(tokenized_docs))
    avg_doc_len = max(avg_doc_len, 1.0)
    num_docs = len(candidates)
    scores: list[float] = []
    for tokens, term_freqs in zip(tokenized_docs, doc_term_freqs):
        doc_len = max(1, len(tokens))
        score = 0.0
        for term, query_tf in query_terms.items():
            tf = term_freqs.get(term, 0)
            if tf <= 0:
                continue
            idf = math.log(1.0 + ((num_docs - doc_freqs[term] + 0.5) / (doc_freqs[term] + 0.5)))
            denom = tf + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
            tf_weight = ((tf * (k1 + 1.0)) / denom) + delta
            query_weight = 1.0 + 0.25 * max(0, query_tf - 1)
            score += idf * tf_weight * query_weight
        scores.append(score)
    return scores


def ranked_selection_result(
    *,
    row: dict[str, Any],
    context: dict[str, Any],
    scores: list[float],
    mode_name: str,
    reason_codes: list[str] | None = None,
    query_family: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = candidate_views(row)
    if not candidates:
        return shared_result(
            selected=[],
            budget=None,
            budget_mode=mode_name,
            selection_meta={
                "query_family": query_family or v2_budget_family_hint(row),
                "selection_mode": mode_name,
                "selection_margin": None,
                "reason_codes": list(reason_codes or []),
                "anchor_count": 0,
            },
        )
    if len(scores) != len(candidates):
        raise ValueError(
            f"{mode_name} produced {len(scores)} scores for {len(candidates)} candidates"
        )

    effective_query_family = query_family or v2_budget_family_hint(row)
    token_budget = v2_target_token_budget(row, context, query_family=effective_query_family)
    max_selected = v2_max_selected(row, context, query_family=effective_query_family)
    ranked = sorted(
        zip(scores, candidates),
        key=lambda item: (item[0], -item[1].rank),
        reverse=True,
    )

    selected: list[CandidateView] = []
    used_tokens = 0
    if max_selected > 0:
        for score, candidate in ranked:
            _, used_tokens = append_with_budget(
                selected,
                candidate,
                token_budget=token_budget,
                max_selected=max_selected,
                used_tokens=used_tokens,
            )
        if not selected and ranked:
            selected = [ranked[0][1]]

    score_lookup = {candidate.memory_id: float(score) for score, candidate in ranked}
    top_score = float(ranked[0][0]) if ranked else None
    second_score = float(ranked[1][0]) if len(ranked) > 1 else None
    margin = top_score - second_score if top_score is not None and second_score is not None else None
    selection_meta = {
        "query_family": effective_query_family,
        "selection_mode": mode_name,
        "selection_margin": round(margin, 6) if margin is not None else None,
        "reason_codes": list(reason_codes or []),
        "anchor_count": len(selected),
        "retrieval_top_hits": [
            {"memory_id": candidate.memory_id, "score": round(float(score), 6)}
            for score, candidate in ranked[: min(5, len(ranked))]
        ],
        "retrieval_scores_selected": [
            round(score_lookup[candidate.memory_id], 6) for candidate in selected
        ],
    }
    if extra_meta:
        selection_meta.update(extra_meta)
    return shared_result(
        selected=selected,
        budget=token_budget,
        budget_mode=mode_name,
        selection_meta=selection_meta,
    )


def ranked_fixed_top_k_result(
    *,
    row: dict[str, Any],
    scores: list[float],
    mode_name: str,
    top_k: int = DEFAULT_RETRIEVAL_BASELINE_TOP_K,
    reason_codes: list[str] | None = None,
    query_family: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = candidate_views(row)
    if not candidates:
        return shared_result(
            selected=[],
            budget=None,
            budget_mode=mode_name,
            selection_meta={
                "query_family": query_family or v2_budget_family_hint(row),
                "selection_mode": mode_name,
                "selection_margin": None,
                "reason_codes": list(reason_codes or []),
                "anchor_count": 0,
                "retrieval_top_k": int(top_k),
            },
        )
    if len(scores) != len(candidates):
        raise ValueError(
            f"{mode_name} produced {len(scores)} scores for {len(candidates)} candidates"
        )

    effective_query_family = query_family or v2_budget_family_hint(row)
    ranked = sorted(
        zip(scores, candidates),
        key=lambda item: (item[0], -item[1].rank),
        reverse=True,
    )
    selected = [candidate for _, candidate in ranked[: max(0, int(top_k))]]

    top_score = float(ranked[0][0]) if ranked else None
    second_score = float(ranked[1][0]) if len(ranked) > 1 else None
    margin = top_score - second_score if top_score is not None and second_score is not None else None
    score_lookup = {candidate.memory_id: float(score) for score, candidate in ranked}
    selection_meta = {
        "query_family": effective_query_family,
        "selection_mode": mode_name,
        "selection_margin": round(margin, 6) if margin is not None else None,
        "reason_codes": list(reason_codes or []),
        "anchor_count": len(selected),
        "retrieval_top_k": int(top_k),
        "retrieval_top_hits": [
            {"memory_id": candidate.memory_id, "score": round(float(score), 6)}
            for score, candidate in ranked[: min(5, len(ranked))]
        ],
        "retrieval_scores_selected": [
            round(score_lookup[candidate.memory_id], 6) for candidate in selected
        ],
    }
    if extra_meta:
        selection_meta.update(extra_meta)
    return shared_result(
        selected=selected,
        budget=None,
        budget_mode=mode_name,
        selection_meta=selection_meta,
    )
