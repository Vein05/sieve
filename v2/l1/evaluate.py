"""Evaluation harness for L1 sufficiency estimator.

Implements leave-one-benchmark-out (LOBO) transfer evaluation, calibration
bins, and the decision-change metric (vs incumbent guard). All computation
is local / offline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from v2.l1.labels import LabeledExample
from v2.l1.train import TrainedModels, predict_proba, train

# Number of calibration bins for reliability diagram.
_N_CALIBRATION_BINS = 10

# Model names used throughout evaluation output.
MODEL_NAMES = ("baseline", "lr", "gbm")


@dataclass
class AUCResult:
    """AUC computed from predicted probabilities vs binary labels."""

    auc: float
    n_positive: int
    n_negative: int
    n_total: int


@dataclass
class CalibrationResult:
    """Reliability diagram binned statistics."""

    bin_mean_predicted: list[float]
    bin_mean_actual: list[float]
    bin_counts: list[int]
    ece: float  # expected calibration error


@dataclass
class DecisionChangeResult:
    """Decision change analysis vs baseline at matched fire rate."""

    fire_rate_baseline: float
    fire_rate_model: float
    n_decisions_changed: int
    n_changed_correct: int
    changed_precision: float  # P(decision was correct | decision changed)
    baseline_precision_at_changes: float  # P(baseline was correct) at those positions


@dataclass
class LOBOResult:
    """Single leave-one-benchmark-out evaluation row."""

    held_out: str
    train_sources: list[str]
    auc_baseline: float
    auc_lr: float
    auc_gbm: float
    n_train: int
    n_test: int
    calibration_lr: CalibrationResult
    calibration_gbm: CalibrationResult
    decision_change_lr: DecisionChangeResult
    decision_change_gbm: DecisionChangeResult


def _auc_from_proba(y_true: np.ndarray, y_score: np.ndarray) -> AUCResult:
    """Compute AUC via trapezoidal rule (no sklearn dependency here)."""
    from sklearn.metrics import roc_auc_score

    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return AUCResult(auc=float("nan"), n_positive=n_pos, n_negative=n_neg, n_total=len(y_true))
    auc = float(roc_auc_score(y_true, y_score))
    return AUCResult(auc=auc, n_positive=n_pos, n_negative=n_neg, n_total=len(y_true))


def _calibration_bins(
    y_true: np.ndarray, y_score: np.ndarray
) -> CalibrationResult:
    """Compute reliability diagram bins and ECE."""
    bins = np.linspace(0.0, 1.0, _N_CALIBRATION_BINS + 1)
    bin_mean_pred: list[float] = []
    bin_mean_act: list[float] = []
    bin_counts: list[int] = []
    ece_acc = 0.0
    n = len(y_true)
    for i in range(_N_CALIBRATION_BINS):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_score >= lo) & (y_score < hi) if i < _N_CALIBRATION_BINS - 1 else (y_score >= lo) & (y_score <= hi)
        cnt = int(mask.sum())
        bin_counts.append(cnt)
        if cnt == 0:
            bin_mean_pred.append(float((lo + hi) / 2))
            bin_mean_act.append(0.0)
            continue
        mp = float(y_score[mask].mean())
        ma = float(y_true[mask].mean())
        bin_mean_pred.append(mp)
        bin_mean_act.append(ma)
        ece_acc += (cnt / n) * abs(mp - ma)
    return CalibrationResult(
        bin_mean_predicted=bin_mean_pred,
        bin_mean_actual=bin_mean_act,
        bin_counts=bin_counts,
        ece=ece_acc,
    )


def _match_fire_rate_threshold(model_proba: np.ndarray, fire_rate: float) -> float:
    """Find the threshold that achieves the given fire rate on model_proba."""
    n_fire = int(round(fire_rate * len(model_proba)))
    if n_fire == 0:
        return 1.1
    if n_fire >= len(model_proba):
        return -0.1
    return float(np.sort(model_proba)[::-1][n_fire - 1])


def _decision_changes(
    y_true: np.ndarray,
    baseline_proba: np.ndarray,
    model_proba: np.ndarray,
    baseline_threshold: float,
) -> DecisionChangeResult:
    """Compare model decisions to baseline at the baseline's natural fire rate."""
    baseline_decisions = (baseline_proba >= baseline_threshold).astype(int)
    fire_rate_baseline = float(baseline_decisions.mean())

    model_threshold = _match_fire_rate_threshold(model_proba, fire_rate_baseline)
    model_decisions = (model_proba >= model_threshold).astype(int)
    fire_rate_model = float(model_decisions.mean())

    changed_mask = baseline_decisions != model_decisions
    n_changed = int(changed_mask.sum())

    if n_changed == 0:
        return DecisionChangeResult(
            fire_rate_baseline=fire_rate_baseline,
            fire_rate_model=fire_rate_model,
            n_decisions_changed=0,
            n_changed_correct=0,
            changed_precision=float("nan"),
            baseline_precision_at_changes=float("nan"),
        )

    gold_int = y_true[changed_mask].astype(int)
    model_correct = model_decisions[changed_mask] == gold_int
    baseline_correct = baseline_decisions[changed_mask] == gold_int

    return DecisionChangeResult(
        fire_rate_baseline=fire_rate_baseline,
        fire_rate_model=fire_rate_model,
        n_decisions_changed=n_changed,
        n_changed_correct=int(model_correct.sum()),
        changed_precision=float(model_correct.mean()),
        baseline_precision_at_changes=float(baseline_correct.mean()),
    )


def run_lobo(all_examples: list[LabeledExample]) -> list[LOBOResult]:
    """Run leave-one-benchmark-out evaluation over all unique sources."""
    sources = sorted({e.source for e in all_examples})
    results: list[LOBOResult] = []

    for held_out in sources:
        train_exs = [e for e in all_examples if e.source != held_out]
        test_exs = [e for e in all_examples if e.source == held_out]

        if not train_exs or not test_exs:
            continue

        models = train(train_exs)
        probas = predict_proba(models, test_exs)

        y_true = np.array([e.label for e in test_exs], dtype=np.float64)

        auc_results = {name: _auc_from_proba(y_true, probas[name]) for name in MODEL_NAMES}

        calib_lr = _calibration_bins(y_true, probas["lr"])
        calib_gbm = _calibration_bins(y_true, probas["gbm"])

        dc_lr = _decision_changes(y_true, probas["baseline"], probas["lr"], models.baseline_threshold)
        dc_gbm = _decision_changes(y_true, probas["baseline"], probas["gbm"], models.baseline_threshold)

        results.append(LOBOResult(
            held_out=held_out,
            train_sources=[s for s in sources if s != held_out],
            auc_baseline=auc_results["baseline"].auc,
            auc_lr=auc_results["lr"].auc,
            auc_gbm=auc_results["gbm"].auc,
            n_train=len(train_exs),
            n_test=len(test_exs),
            calibration_lr=calib_lr,
            calibration_gbm=calib_gbm,
            decision_change_lr=dc_lr,
            decision_change_gbm=dc_gbm,
        ))

    return results


def format_auc_table(results: list[LOBOResult]) -> str:
    """Format the AUC table as markdown."""
    header = "| held-out | train sources | n_train | n_test | AUC baseline | AUC LR | AUC GBM |"
    sep = "|---|---|---:|---:|---:|---:|---:|"
    rows = [header, sep]
    for r in results:
        def fmt(v: float) -> str:
            return f"{v:.3f}" if not math.isnan(v) else "n/a"
        row = (
            f"| {r.held_out} | {', '.join(r.train_sources)} | {r.n_train} | {r.n_test} "
            f"| {fmt(r.auc_baseline)} | {fmt(r.auc_lr)} | {fmt(r.auc_gbm)} |"
        )
        rows.append(row)
    return "\n".join(rows)


def format_decision_table(results: list[LOBOResult]) -> str:
    """Format the decision-change table as markdown."""
    header = (
        "| held-out | model | fire_rate_baseline | fire_rate_model "
        "| n_changed | n_changed_correct | changed_precision | baseline_precision_at_changes |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|"
    rows = [header, sep]
    for r in results:
        for model_name, dc in [("lr", r.decision_change_lr), ("gbm", r.decision_change_gbm)]:
            def fmt(v: float) -> str:
                return f"{v:.3f}" if not math.isnan(v) else "n/a"
            row = (
                f"| {r.held_out} | {model_name} "
                f"| {fmt(dc.fire_rate_baseline)} | {fmt(dc.fire_rate_model)} "
                f"| {dc.n_decisions_changed} | {dc.n_changed_correct} "
                f"| {fmt(dc.changed_precision)} | {fmt(dc.baseline_precision_at_changes)} |"
            )
            rows.append(row)
    return "\n".join(rows)
