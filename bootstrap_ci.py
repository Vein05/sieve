#!/usr/bin/env python3
"""Bootstrap 95% CIs for Table 3 (multi-session ceiling) and Table 5 (Pareto efficiency)."""

import json
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent
RUNS = BASE / "results" / "answer_generation_runs"
DATASET = BASE / "dataset-slices" / "longmemeval_bm25_top20_v1.jsonl"

N_BOOT = 10_000
RNG = np.random.default_rng(42)


def load_judge(run_dir):
    """Load judge_run_v2.json, return dict of example_id -> judge_score_v2."""
    path = run_dir / "judge_run_v2.json"
    with open(path) as f:
        data = json.load(f)
    return {o["example_id"]: o["judge_score_v2"] for o in data["outputs"]}


def load_dataset():
    """Load dataset, return list of dicts with example_id and question_type."""
    rows = []
    with open(DATASET) as f:
        for line in f:
            obj = json.loads(line)
            rows.append({"example_id": obj["example_id"], "question_type": obj.get("question_type", obj.get("ability", "unknown"))})
    return rows


def bootstrap_delta_ci(sieve_scores, naive_scores, n_boot=N_BOOT):
    """
    Compute bootstrap 95% CI on accuracy delta (sieve - naive).
    sieve_scores and naive_scores are paired arrays of 0/2.
    Returns (delta, lo, hi).
    """
    n = len(sieve_scores)
    sieve = np.array(sieve_scores)
    naive = np.array(naive_scores)

    delta = (sieve == 2).mean() - (naive == 2).mean()

    deltas = np.empty(n_boot)
    idx = RNG.integers(0, n, size=(n_boot, n))
    for i in range(n_boot):
        s = sieve[idx[i]]
        nv = naive[idx[i]]
        deltas[i] = (s == 2).mean() - (nv == 2).mean()

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return delta, lo, hi


def fmt_pct(v):
    return f"{v*100:+.1f}pp"


def fmt_ci(lo, hi):
    return f"[{lo*100:+.1f}, {hi*100:+.1f}]"


# ── Table 3: Multi-Session Retrieval Ceiling ──

dataset = load_dataset()
id_to_qt = {r["example_id"]: r["question_type"] for r in dataset}

TABLE3_MODELS = {
    "llama8b":  ("paper_llama8b_naive",  "paper_llama8b_sieve"),
    "qwen7b":   ("paper_qwen7b_naive",   "paper_qwen7b_sieve"),
    "llama70b":  ("paper_llama70b_naive",  "paper_llama70b_sieve"),
    "qwen72b":  ("paper_qwen72b_naive",  "paper_qwen72b_sieve"),
}

print("## Table 3: Multi-Session Retrieval Ceiling — Bootstrap 95% CIs on SIEVE Delta")
print()
print("| Model | Non-MS Δ | Non-MS 95% CI | MS Δ | MS 95% CI |")
print("|-------|----------|---------------|------|-----------|")

for model, (naive_dir, sieve_dir) in TABLE3_MODELS.items():
    naive_scores = load_judge(RUNS / naive_dir)
    sieve_scores = load_judge(RUNS / sieve_dir)

    # Split by question type
    ms_ids = [eid for eid, qt in id_to_qt.items() if qt == "multi-session"]
    non_ms_ids = [eid for eid, qt in id_to_qt.items() if qt != "multi-session"]

    # Filter to IDs present in both
    ms_ids = [eid for eid in ms_ids if eid in naive_scores and eid in sieve_scores]
    non_ms_ids = [eid for eid in non_ms_ids if eid in naive_scores and eid in sieve_scores]

    # Non-MS
    ns = [sieve_scores[eid] for eid in non_ms_ids]
    nn = [naive_scores[eid] for eid in non_ms_ids]
    d_nms, lo_nms, hi_nms = bootstrap_delta_ci(ns, nn)

    # MS
    ms_s = [sieve_scores[eid] for eid in ms_ids]
    ms_n = [naive_scores[eid] for eid in ms_ids]
    d_ms, lo_ms, hi_ms = bootstrap_delta_ci(ms_s, ms_n)

    print(f"| {model:8s} | {fmt_pct(d_nms):>8s} | {fmt_ci(lo_nms, hi_nms):>16s} | {fmt_pct(d_ms):>8s} | {fmt_ci(lo_ms, hi_ms):>16s} |")

print()
print(f"  Non-MS n={len(non_ms_ids)}, MS n={len(ms_ids)}")


# ── Table 5: Pareto Efficiency ──

# Map model -> list of (budget_label, naive_dir, sieve_dir)
TABLE5_MODELS = {
    "llama8b": [
        ("@200", "paper_llama8b_naive_at200", "paper_llama8b_sieve_at200"),
        ("@400", "paper_llama8b_naive_at400", "paper_llama8b_sieve_at400"),
        ("@800", "paper_llama8b_naive_at800", "paper_llama8b_sieve_at800"),
        ("uncap", "paper_llama8b_naive", "paper_llama8b_sieve"),
    ],
    "qwen7b": [
        ("@200", "paper_qwen-2-5-7b_naive_at200", "paper_qwen-2-5-7b_sieve_at200"),
        ("@400", "paper_qwen-2-5-7b_naive_at400", "paper_qwen-2-5-7b_sieve_at400"),
        ("@800", "paper_qwen-2-5-7b_naive_at800", "paper_qwen-2-5-7b_sieve_at800"),
        ("uncap", "paper_qwen7b_naive", "paper_qwen7b_sieve"),
    ],
    "gemma12b": [
        ("@200", "paper_gemma12b_naive_at200", "paper_gemma12b_sieve_at200"),
        ("@400", "paper_gemma-3-12b-it_naive_at400", "paper_gemma-3-12b-it_sieve_at400"),
        ("@800", "paper_gemma-3-12b-it_naive_at800", "paper_gemma-3-12b-it_sieve_at800"),
        ("uncap", "paper_gemma12b_naive", "paper_gemma12b_sieve"),
    ],
    "llama70b": [
        ("@200", "paper_llama-3-1-70b_naive_at200", "paper_llama-3-1-70b_sieve_at200"),
        ("@400", "paper_llama-3-1-70b_naive_at400", "paper_llama-3-1-70b_sieve_at400"),
        ("@800", "paper_llama-3-1-70b_naive_at800", "paper_llama-3-1-70b_sieve_at800"),
        ("uncap", "paper_llama70b_naive", "paper_llama70b_sieve"),
    ],
}

print()
print("## Table 5: Pareto Efficiency — Bootstrap 95% CIs on SIEVE Delta")
print()
print("| Model    | Δ@200   | CI@200           | Δ@400   | CI@400           | Δ@800   | CI@800           | Δ uncap | CI uncap         |")
print("|----------|---------|------------------|---------|------------------|---------|------------------|---------|------------------|")

for model, budgets in TABLE5_MODELS.items():
    cells = []
    for label, naive_dir, sieve_dir in budgets:
        naive_scores = load_judge(RUNS / naive_dir)
        sieve_scores = load_judge(RUNS / sieve_dir)

        common_ids = sorted(set(naive_scores.keys()) & set(sieve_scores.keys()))
        ns = [sieve_scores[eid] for eid in common_ids]
        nn = [naive_scores[eid] for eid in common_ids]

        d, lo, hi = bootstrap_delta_ci(ns, nn)
        cells.append(f" {fmt_pct(d):>7s} | {fmt_ci(lo, hi):>16s}")

    print(f"| {model:8s} |{'|'.join(cells)} |")

print()


# ── Cross-Domain Correlation CIs ──

def bootstrap_correlation_ci(baselines, deltas, n_boot=N_BOOT):
    """Bootstrap 95% CI on Pearson r between baselines and deltas."""
    x = np.array(baselines, dtype=float)
    y = np.array(deltas, dtype=float)
    n = len(x)
    from scipy import stats as sp_stats
    r_obs = sp_stats.pearsonr(x, y)[0]

    boot_r = np.empty(n_boot)
    idx = RNG.integers(0, n, size=(n_boot, n))
    for i in range(n_boot):
        xi, yi = x[idx[i]], y[idx[i]]
        # degenerate bootstrap samples (all same x or y) -> skip
        if xi.std() == 0 or yi.std() == 0:
            boot_r[i] = np.nan
        else:
            boot_r[i] = np.corrcoef(xi, yi)[0, 1]

    boot_r = boot_r[~np.isnan(boot_r)]
    lo, hi = np.percentile(boot_r, [2.5, 97.5])
    return r_obs, lo, hi


# Data from paper Table 2 (LME), appendix Table 14 (HotpotQA), appendix Table (MuSiQue),
# appendix Table recomp_hotpot (RECOMP).  Format: (naive_baseline, delta).

DOMAIN_DATA = {
    "LongMemEval SIEVE (n=20)": [
        (27.8, 13.2), (34.4, 12.8), (34.6, 12.0), (35.0, 11.6), (38.6, 12.2),
        (39.0, 11.8), (43.2, 9.2), (43.8, 9.6), (46.4, 8.4), (46.2, 4.8),
        (48.8, 3.6), (47.4, 2.6), (48.4, 7.8), (47.2, 0.0), (49.6, 5.6),
        (50.6, 4.8), (55.6, 2.8), (53.4, 2.8), (56.6, 0.2), (55.4, 0.8),
    ],
    "LongMemEval LLM-Sum (n=20)": [
        (27.8, 16.6), (34.4, 12.8), (34.6, 18.2), (35.0, 6.0), (38.6, 5.6),
        (39.0, 5.2), (43.2, 6.4), (43.8, 0.0), (46.4, 4.8), (46.2, 3.4),
        (48.8, -2.2), (47.4, 1.2), (48.4, 5.8), (47.2, -6.2), (49.6, 1.6),
        (50.6, -1.4), (55.6, -4.6), (53.4, -0.4), (56.6, -4.0), (55.4, -2.0),
    ],
    "HotpotQA LLM-Sum (n=11)": [
        (24.0, 39.2), (40.8, 32.0), (46.0, 33.8), (52.0, 22.4),
        (53.8, 25.8), (59.6, 19.6), (60.6, 17.8), (61.4, 19.4),
        (62.8, 15.4), (66.0, 13.4), (69.4, 11.4),
    ],
    "HotpotQA RECOMP (n=11)": [
        (24.0, 22.2), (40.8, 6.2), (46.0, 4.2), (52.0, -1.6),
        (53.8, -6.6), (59.6, -6.2), (60.6, -12.2), (61.4, -11.4),
        (62.8, -14.8), (66.0, -13.4), (69.4, -17.2),
    ],
    "MuSiQue LLM-Sum (n=11)": [
        (8.0, 32.4), (8.6, 37.4), (9.8, 30.4), (17.8, 26.4),
        (17.8, 24.4), (19.0, 26.6), (20.0, 25.6), (20.6, 25.6),
        (21.4, 23.4), (27.6, 20.0), (31.8, 16.4),
    ],
}

print("## Cross-Domain Correlation Bootstrap 95% CIs")
print()
print(f"| {'Domain':<35s} | {'r':>7s} | {'95% CI':>16s} |")
print(f"|{'-'*37}|{'-'*9}|{'-'*18}|")

for name, pts in DOMAIN_DATA.items():
    baselines = [p[0] for p in pts]
    deltas = [p[1] for p in pts]
    r, lo, hi = bootstrap_correlation_ci(baselines, deltas)
    print(f"| {name:<35s} | {r:>7.3f} | [{lo:>6.3f}, {hi:>6.3f}] |")

print()
print("All CIs: percentile method, 10,000 bootstrap resamples, seed=42.")
