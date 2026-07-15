"""Secondary validation: correlate L1 scores with reader success on pilot packs.

Scores each pilot pack variant using a trained L1 model, then computes
Spearman correlation between L1 score and judge_score_v2 per style.
No network access; reads local JSON files only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from v2.l1.features import extract
from v2.l1.train import TrainedModels, predict_proba, ex_to_fv
from v2.l1.labels import LabeledExample

# Unique key for a variant cache entry.
_CACHE_KEY_SEP = "::"


@dataclass
class PilotPackEntry:
    """A single variant cache pack entry with its judge scores."""

    example_id: str
    style: str
    budget: int
    pack_texts: tuple[str, ...]
    judge_scores: list[float]  # one per reader model
    mean_judge_score: float


@dataclass
class StyleCorrelation:
    """Spearman correlation between L1 score and judge score for one style."""

    style: str
    model_name: str
    spearman_rho: float
    n: int


def _load_variant_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load variant cache, return entries dict keyed by example_id::style::budget."""
    data = json.loads(cache_path.read_text())
    return data.get("entries", {})


def _load_judge_outputs(judge_path: Path) -> dict[str, list[float]]:
    """Load judge run, return dict keyed by example_id::style::budget -> scores list."""
    data = json.loads(judge_path.read_text())
    outputs = data.get("outputs", [])
    scores: dict[str, list[float]] = {}
    for out in outputs:
        key = f"{out['example_id']}{_CACHE_KEY_SEP}{out['style']}{_CACHE_KEY_SEP}{out['budget']}"
        score = out.get("judge_score_v2")
        if score is not None:
            scores.setdefault(key, []).append(float(score))
    return scores


def _query_from_outputs(judge_path: Path) -> dict[str, str]:
    """Build example_id -> query text mapping from judge outputs."""
    data = json.loads(judge_path.read_text())
    outputs = data.get("outputs", [])
    return {o["example_id"]: o.get("question", "") for o in outputs}


def _source_memory_ids_to_texts(
    source_ids: list[str],
    entry: dict[str, Any],
) -> tuple[str, ...]:
    """Extract evidence text from an entry; return as single-element tuple."""
    text = entry.get("evidence_text", "")
    return (text,) if text else ()


def load_pilot_packs(
    cache_path: Path,
    judge_path: Path,
) -> tuple[list[PilotPackEntry], dict[str, str]]:
    """Load pilot packs paired with judge scores.

    Returns:
        (list of PilotPackEntry, dict example_id -> query text)
    """
    entries = _load_variant_cache(cache_path)
    judge_scores = _load_judge_outputs(judge_path)
    id_to_query = _query_from_outputs(judge_path)

    pilot_packs: list[PilotPackEntry] = []
    for key, entry in entries.items():
        scores = judge_scores.get(key)
        if not scores:
            continue
        texts = _source_memory_ids_to_texts(entry.get("source_memory_ids", []), entry)
        pilot_packs.append(PilotPackEntry(
            example_id=entry["example_id"],
            style=entry["style"],
            budget=int(entry.get("budget", 0)),
            pack_texts=texts,
            judge_scores=scores,
            mean_judge_score=float(np.mean(scores)),
        ))
    return pilot_packs, id_to_query


def _spearman_rho(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation."""
    if len(x) < 3:
        return float("nan")
    n = len(x)

    def rank(arr: list[float]) -> list[float]:
        sorted_idx = sorted(range(n), key=lambda i: arr[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and arr[sorted_idx[j]] == arr[sorted_idx[i]]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1
            for k in range(i, j):
                ranks[sorted_idx[k]] = avg_rank
            i = j
        return ranks

    rx = rank(x)
    ry = rank(y)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1.0 - (6.0 * d2) / (n * (n * n - 1))
    return rho


def score_pilot_packs(
    models: TrainedModels,
    pilot_packs: list[PilotPackEntry],
    id_to_query: dict[str, str],
    model_names: tuple[str, ...] = ("baseline", "lr", "gbm"),
) -> list[StyleCorrelation]:
    """Compute L1 scores for each pack and correlate with mean judge score per style.

    Uses evidence_text as a single-unit pack (proxy for the full pack).
    """
    # Build pseudo-LabeledExamples (labels not used for scoring).
    pseudo_examples: list[LabeledExample] = []
    for pp in pilot_packs:
        query = id_to_query.get(pp.example_id, "")
        pseudo_examples.append(LabeledExample(
            example_id=pp.example_id,
            source="pilot",
            query=query,
            pack_memory_ids=(),
            pack_texts=pp.pack_texts,
            label=0.0,  # unused
            gold_memory_ids=(),
        ))

    if not pseudo_examples:
        return []

    probas = predict_proba(models, pseudo_examples)

    # Group by style.
    styles = sorted({pp.style for pp in pilot_packs})
    correlations: list[StyleCorrelation] = []

    for style in styles:
        indices = [i for i, pp in enumerate(pilot_packs) if pp.style == style]
        if len(indices) < 3:
            continue
        judge_s = [pilot_packs[i].mean_judge_score for i in indices]
        for model_name in model_names:
            model_s = [float(probas[model_name][i]) for i in indices]
            rho = _spearman_rho(model_s, judge_s)
            correlations.append(StyleCorrelation(
                style=style,
                model_name=model_name,
                spearman_rho=rho,
                n=len(indices),
            ))

    return correlations


def format_correlation_table(correlations: list[StyleCorrelation]) -> str:
    """Format correlation table as markdown."""
    header = "| style | model | spearman_rho | n |"
    sep = "|---|---|---:|---:|"
    rows = [header, sep]
    for c in correlations:
        rho_str = f"{c.spearman_rho:.3f}" if not math.isnan(c.spearman_rho) else "n/a"
        rows.append(f"| {c.style} | {c.model_name} | {rho_str} | {c.n} |")
    return "\n".join(rows)
