"""Top-k baseline assembly method."""

from __future__ import annotations

from typing import Any

from shared.nlp import candidate_views
from .baselines import ranked_fixed_top_k_result


def method_naive_top_k(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    candidates = candidate_views(row)
    scores = [-float(c.rank) for c in candidates]

    return ranked_fixed_top_k_result(
        row=row,
        scores=scores,
        mode_name="naive_top_k",
        reason_codes=["naive_baseline_top_k"],
    )
