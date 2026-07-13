#!/usr/bin/env python3
"""Compute Pearson/Spearman correlations with leave-one-out stability.

Reads judge-scored runs from results/ and outputs a markdown table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def load_judge_accuracy(run_dir: Path) -> float | None:
    """Load exact_rate from judge_run_v2.json."""
    judge_path = run_dir / "judge_run_v2.json"
    if not judge_path.exists():
        return None
    with open(judge_path) as f:
        data = json.load(f)
    outputs = data.get("outputs", [])
    if not outputs:
        return None
    correct = sum(1 for o in outputs if o.get("judge_score_v2", 0) == 2)
    return round(100 * correct / len(outputs), 1)


# (label, params_str, naive_run_dir, sieve_run_dir)
MODELS = [
    ("Llama 3.1 8B",    "8B",       "paper_llama8b_naive",   "paper_llama8b_sieve"),
    ("Qwen 2.5 7B",     "7B",       "paper_qwen7b_naive",    "paper_qwen7b_sieve"),
    ("OLMo 3.1 32B",    "32B",      "paper_olmo32b_naive",   "paper_olmo32b_sieve"),
    ("GPT-4.1-mini",    "frontier",  "paper_gpt41mini_naive", "paper_gpt41mini_sieve"),
    ("Gemma 3 12B",     "12B",      "paper_gemma12b_naive",  "paper_gemma12b_sieve"),
    ("Llama 3.1 70B",   "70B",      "paper_llama70b_naive",  "paper_llama70b_sieve"),
    ("Qwen 2.5 72B",    "72B",      "paper_qwen72b_naive",   "paper_qwen72b_sieve"),
    ("Grok 4.1-fast",   "frontier",  "paper_grok_naive",      "paper_grok_sieve"),
]


# ── Cross-domain homogeneity tests ────────────────────────────────────

def fisher_z(r: float) -> float:
    """Fisher z-transform of a correlation coefficient."""
    return 0.5 * np.log((1 + r) / (1 - r))


def fisher_z_test(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    """Two-sided test for equality of two Pearson correlations."""
    z1, z2 = fisher_z(r1), fisher_z(r2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z_stat = (z1 - z2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_value


def cochran_q_homogeneity(correlations: list[tuple[float, int]]) -> dict:
    """Cochran Q test: are multiple correlations consistent with one underlying r?

    Args:
        correlations: list of (r, n) pairs

    Returns:
        dict with keys: Q, df, p, combined_r, combined_z
    """
    zs = [fisher_z(r) for r, _ in correlations]
    ws = [n - 3 for _, n in correlations]
    z_avg = np.average(zs, weights=ws)
    Q = sum(w * (z - z_avg) ** 2 for w, z in zip(ws, zs))
    df = len(zs) - 1
    p = 1 - stats.chi2.cdf(Q, df)
    return dict(Q=Q, df=df, p=p, combined_z=z_avg, combined_r=np.tanh(z_avg))


def cohen_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size for two independent groups."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        (group1.var(ddof=1) * (n1 - 1) + group2.var(ddof=1) * (n2 - 1)) / (n1 + n2 - 2)
    )
    return (group1.mean() - group2.mean()) / pooled_std


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - margin, center + margin


def print_cross_domain_stats() -> None:
    """Print Fisher z-tests and Cochran Q for all domain correlations."""
    domain_correlations = [
        ("LongMemEval (SIEVE)", -0.887, 20),
        ("LongMemEval (LLM-Sum)", -0.854, 20),
        ("HotpotQA (LLM-Sum)", -0.959, 11),
        ("RECOMP (in-domain)", -0.987, 11),
        ("MuSiQue (LLM-Sum)", -0.953, 11),
    ]

    print("\n## Cross-Domain Homogeneity\n")

    # Pairwise Fisher z-tests
    print("| Domain A | Domain B | z | p |")
    print("| --- | --- | ---: | ---: |")
    for i in range(len(domain_correlations)):
        for j in range(i + 1, len(domain_correlations)):
            n1, r1, nn1 = domain_correlations[i]
            n2, r2, nn2 = domain_correlations[j]
            z, p = fisher_z_test(r1, nn1, r2, nn2)
            sig = " *" if p < 0.05 else ""
            print(f"| {n1} | {n2} | {z:.2f} | {p:.3f}{sig} |")

    # Cochran Q
    pairs = [(r, n) for _, r, n in domain_correlations]
    result = cochran_q_homogeneity(pairs)
    print(f"\n**Cochran Q** = {result['Q']:.3f}, df = {result['df']}, p = {result['p']:.3f}")
    print(f"**Combined r** = {result['combined_r']:.3f}")
    print(f"Homogeneity: {'PASS' if result['p'] > 0.05 else 'FAIL'}")

    # Cohen's d: weak vs strong readers on LongMemEval
    weak_deltas = np.array([13.2, 12.8, 12.0, 11.6, 12.2, 11.8, 9.2, 9.6])
    strong_deltas = np.array([8.4, 4.8, 2.6, 0.0, 7.8, 3.6, 5.6, 4.8, 2.8, 0.8, 2.8, 0.2])
    d = cohen_d(weak_deltas, strong_deltas)
    print(f"\n## Effect Size")
    print(f"Weak (<45% baseline, n={len(weak_deltas)}): mean Δ = {weak_deltas.mean():.1f}pp")
    print(f"Strong (≥45% baseline, n={len(strong_deltas)}): mean Δ = {strong_deltas.mean():.1f}pp")
    print(f"**Cohen's d** = {d:.2f}")

    # Damage rate CI
    ci = wilson_ci(261, 1711)
    print(f"\n## Damage Rate")
    print(f"261/1711 = 15.3%, Wilson 95% CI: [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")


def main() -> None:
    results_root = Path(__file__).resolve().parent.parent / "results" / "answer_generation_runs"

    rows: list[tuple[str, str, float, float, float]] = []
    for label, params, naive_dir, sieve_dir in MODELS:
        naive_acc = load_judge_accuracy(results_root / naive_dir)
        sieve_acc = load_judge_accuracy(results_root / sieve_dir)
        if naive_acc is None or sieve_acc is None:
            print(f"SKIP {label}: naive={naive_acc}, sieve={sieve_acc}", file=sys.stderr)
            continue
        delta = round(sieve_acc - naive_acc, 1)
        rows.append((label, params, naive_acc, sieve_acc, delta))

    if len(rows) < 3:
        print("Not enough data points", file=sys.stderr)
        sys.exit(1)

    names = [r[0] for r in rows]
    x = np.array([r[2] for r in rows])  # naive accuracy
    y = np.array([r[4] for r in rows])  # delta
    n = len(x)

    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)

    lines = [
        f"# Correlation Analysis (n={n} models)",
        "",
        "## Model Data",
        "",
        "| Model | Params | Naive | SIEVE | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for label, params, naive_acc, sieve_acc, delta in rows:
        lines.append(f"| {label} | {params} | {naive_acc}% | {sieve_acc}% | {delta:+.1f}pp |")

    lines.extend([
        "",
        "## Full Correlation",
        "",
        f"- **Pearson r** = {r_pearson:.3f}, p = {p_pearson:.4f}",
        f"- **Spearman rho** = {r_spearman:.3f}, p = {p_spearman:.4f}",
        "",
        "## Leave-One-Out Stability",
        "",
        "| Dropped | Pearson r | p | Sig? | Spearman rho | p | Sig? |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ])

    all_sig_pearson = True
    all_sig_spearman = True
    worst_pearson_r = 0.0
    worst_pearson_p = 0.0
    worst_pearson_name = ""

    for i in range(n):
        x_loo = np.delete(x, i)
        y_loo = np.delete(y, i)
        r_p, p_p = stats.pearsonr(x_loo, y_loo)
        r_s, p_s = stats.spearmanr(x_loo, y_loo)
        sig_p = "Yes" if p_p < 0.05 else "**No**"
        sig_s = "Yes" if p_s < 0.05 else "**No**"
        if p_p >= 0.05:
            all_sig_pearson = False
        if p_s >= 0.05:
            all_sig_spearman = False
        if abs(r_p) < abs(worst_pearson_r) or worst_pearson_r == 0.0:
            worst_pearson_r = r_p
            worst_pearson_p = p_p
            worst_pearson_name = names[i]
        lines.append(
            f"| {names[i]} | {r_p:.3f} | {p_p:.4f} | {sig_p} | {r_s:.3f} | {p_s:.4f} | {sig_s} |"
        )

    lines.extend([
        "",
        "## Summary",
        "",
        f"- LOO Pearson: {'**all significant**' if all_sig_pearson else 'NOT all significant'} (worst: drop {worst_pearson_name}, r={worst_pearson_r:.3f}, p={worst_pearson_p:.4f})",
        f"- LOO Spearman: {'**all significant**' if all_sig_spearman else 'NOT all significant'}",
        f"- The correlation is {'robust' if all_sig_pearson and all_sig_spearman else 'fragile'} to single-model removal.",
    ])

    output = "\n".join(lines) + "\n"
    print(output)

    out_path = Path(__file__).resolve().parent.parent / "results" / "correlation_analysis.md"
    out_path.write_text(output, encoding="utf-8")
    print(f"\nWrote to {out_path}", file=sys.stderr)

    print_cross_domain_stats()


if __name__ == "__main__":
    main()
