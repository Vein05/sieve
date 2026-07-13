#!/usr/bin/env python3
"""Per-row mechanism attribution for naive vs SIEVE across 7 models."""

from __future__ import annotations

import json
import re
from pathlib import Path

BASE = Path("results/answer_generation_runs")

MODELS = {
    "llama8b":   "Llama-3.1-8B",
    "qwen7b":    "Qwen-2.5-7B",
    "gemma12b":  "Gemma-3-12B",
    "llama70b":  "Llama-3.1-70B",
    "qwen72b":   "Qwen-2.5-72B",
    "grok":      "Grok-3-Mini",
    "gpt41mini": "GPT-4.1-Mini",
}

ABSTENTION_PATTERNS = [
    r"^unknown\.?$",
    r"^i don'?t know\.?$",
    r"^i do not know\.?$",
    r"^cannot determine\.?$",
    r"^not enough information\.?$",
    r"^cannot answer\.?$",
    r"^no information\.?$",
    r"^not provided\.?$",
    r"^not mentioned\.?$",
    r"^unable to (determine|answer)",
    r"^insufficient information",
    r"^n/?a\.?$",
]

def is_abstention(answer: str) -> bool:
    a = answer.strip().lower()
    for pat in ABSTENTION_PATTERNS:
        if re.match(pat, a):
            return True
    return False


def load_model(slug):
    naive_run = json.loads((BASE / f"paper_{slug}_naive" / "run.json").read_text())
    naive_judge = json.loads((BASE / f"paper_{slug}_naive" / "judge_run_v2.json").read_text())
    sieve_judge = json.loads((BASE / f"paper_{slug}_sieve" / "judge_run_v2.json").read_text())

    # Index by example_id for safety
    naive_answers = {o["example_id"]: o["generated_answer"] for o in naive_run["outputs"]}
    naive_scores = {o["example_id"]: o["judge_score_v2"] for o in naive_judge["outputs"]}
    sieve_scores = {o["example_id"]: o["judge_score_v2"] for o in sieve_judge["outputs"]}

    return naive_answers, naive_scores, sieve_scores


def classify_rows(slug):
    naive_answers, naive_scores, sieve_scores = load_model(slug)
    ids = sorted(naive_scores.keys())
    if not ids:
        raise ValueError(f"{slug}: no overlapping example IDs between naive and sieve judge scores")

    counts = {"unknown_correct": 0, "wrong_correct": 0, "correct_wrong": 0, "no_change": 0}
    naive_correct = 0
    sieve_correct = 0
    abstention_count = 0

    for eid in ids:
        ns = naive_scores[eid]
        ss = sieve_scores[eid]
        ans = naive_answers[eid]
        abst = is_abstention(ans)

        if ns == 2:
            naive_correct += 1
        if ss == 2:
            sieve_correct += 1
        if abst:
            abstention_count += 1

        if abst and ns == 0 and ss == 2:
            counts["unknown_correct"] += 1
        elif (not abst) and ns == 0 and ss == 2:
            counts["wrong_correct"] += 1
        elif ns == 2 and ss == 0:
            counts["correct_wrong"] += 1
        else:
            counts["no_change"] += 1

    return {
        "naive_correct": naive_correct,
        "sieve_correct": sieve_correct,
        "abstention_count": abstention_count,
        **counts,
    }


def main():
    results = {}
    for slug, label in MODELS.items():
        r = classify_rows(slug)
        r["label"] = label
        r["slug"] = slug
        results[slug] = r

    # Sort by naive accuracy ascending
    sorted_models = sorted(results.values(), key=lambda x: x["naive_correct"])

    # ── Raw numbers ──
    print("## Raw numbers per model\n")
    print("```")
    for r in sorted_models:
        net = r["unknown_correct"] + r["wrong_correct"] - r["correct_wrong"]
        print(
            f"{r['label']:18s}  naive_acc={r['naive_correct']/5:.1f}%  "
            f"abstentions={r['abstention_count']}  "
            f"U→C={r['unknown_correct']}  W→C={r['wrong_correct']}  "
            f"C→W={r['correct_wrong']}  noΔ={r['no_change']}  "
            f"net={net}"
        )
    print("```\n")

    # ── Table A ──
    print("## Table A: Per-row mechanism attribution (all 7 models)\n")
    print("| Model | Naive Acc | Unknown→Correct | Wrong→Correct | Correct→Wrong | No Change | Net Rescued |")
    print("|:------|----------:|----------------:|--------------:|--------------:|----------:|------------:|")
    for r in sorted_models:
        net = r["unknown_correct"] + r["wrong_correct"] - r["correct_wrong"]
        naive_pct = r["naive_correct"] / 5
        print(
            f"| {r['label']} | {naive_pct:.1f}% | "
            f"{r['unknown_correct']} | {r['wrong_correct']} | {r['correct_wrong']} | "
            f"{r['no_change']} | {net} |"
        )

    print()

    # ── Table B ──
    print("## Table B: Mechanism mix (% of gain from each mechanism)\n")
    print("| Model | Naive Acc | % Abstention Rescue | % Answer Quality | Crossover? |")
    print("|:------|----------:|--------------------:|-----------------:|:-----------|")
    for r in sorted_models:
        total_rescued = r["unknown_correct"] + r["wrong_correct"]
        naive_pct = r["naive_correct"] / 5
        if total_rescued > 0:
            abst_pct = r["unknown_correct"] / total_rescued * 100
            qual_pct = r["wrong_correct"] / total_rescued * 100
        else:
            abst_pct = 0
            qual_pct = 0
        crossover = "Yes" if qual_pct > abst_pct else "No"
        if abs(qual_pct - abst_pct) < 5:
            crossover = "~Even"
        print(
            f"| {r['label']} | {naive_pct:.1f}% | "
            f"{abst_pct:.1f}% | {qual_pct:.1f}% | {crossover} |"
        )

    # ── Verification: accuracy check ──
    print("\n## Verification: accuracy sanity check\n")
    print("```")
    for r in sorted_models:
        naive_pct = r["naive_correct"] / 5
        sieve_pct = r["sieve_correct"] / 5
        delta = sieve_pct - naive_pct
        # Delta should equal (U→C + W→C - C→W) / 5
        mech_delta = (r["unknown_correct"] + r["wrong_correct"] - r["correct_wrong"]) / 5
        match = "OK" if abs(delta - mech_delta) < 0.01 else "MISMATCH"
        print(
            f"{r['label']:18s}  naive={naive_pct:.1f}%  sieve={sieve_pct:.1f}%  "
            f"Δ={delta:+.1f}pp  mech_Δ={mech_delta:+.1f}pp  [{match}]"
        )
    print("```")


if __name__ == "__main__":
    main()
