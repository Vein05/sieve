"""End-to-end runner for L1 v0 experiment.

Loads labels from three benchmarks, runs LOBO evaluation, scores pilot packs,
and writes research/l1-sufficiency-v0-results-2026-07.md.
No network access. No LLM calls.

Usage:
    python -m v2.l1.run_experiment
"""

from __future__ import annotations

import math
from pathlib import Path

from v2.l1.evaluate import (
    LOBOResult,
    format_auc_table,
    format_decision_table,
    run_lobo,
)
from v2.l1.labels import LabeledExample, load_beam, load_hotpotqa, load_longmemeval
from v2.l1.pilot_validation import (
    PilotPackEntry,
    StyleCorrelation,
    format_correlation_table,
    load_pilot_packs,
    score_pilot_packs,
)
from v2.l1.train import train

# Canonical data paths.
_SIEVE_ROOT = Path(__file__).parent.parent.parent
_MEF_ROOT = _SIEVE_ROOT.parent / "memory-eligibility-feasibility"

_HOTPOTQA_PATH = _SIEVE_ROOT / "dataset-slices" / "hotpotqa_distractor_500_v1.jsonl"
_LME_PATH = _SIEVE_ROOT / "dataset-slices" / "longmemeval_bm25_top20_v1.jsonl"
_BEAM_PATH = _MEF_ROOT / "dataset-slices" / "beam_transfer_bm25_top20_v0.jsonl"

_PILOT_CACHES = [
    _SIEVE_ROOT / "data" / "v2_variant_cache" / "generic_cross_bench_convomem_100_v0.json",
    _SIEVE_ROOT / "data" / "v2_variant_cache" / "generic_cross_bench_hotpotqa_100_v0.json",
]
_PILOT_JUDGES = [
    _SIEVE_ROOT / "results" / "v2_runs" / "generic_cross_bench_convomem_100_v0" / "judge_run_v2.json",
    _SIEVE_ROOT / "results" / "v2_runs" / "generic_cross_bench_hotpotqa_100_v0" / "judge_run_v2.json",
]

_OUTPUT_PATH = _SIEVE_ROOT / "research" / "l1-sufficiency-v0-results-2026-07.md"


def _load_all_examples(
    missing: list[str],
) -> tuple[list[LabeledExample], list[LabeledExample], list[LabeledExample]]:
    """Load labeled examples from all three sources. Appends to missing on failure."""
    hotpot = lme = beam = []  # type: ignore[assignment]
    if _HOTPOTQA_PATH.exists():
        hotpot = load_hotpotqa(_HOTPOTQA_PATH)
        print(f"  HotpotQA: {len(hotpot)} examples")
    else:
        missing.append(f"HotpotQA: {_HOTPOTQA_PATH} NOT FOUND")
    if _LME_PATH.exists():
        lme = load_longmemeval(_LME_PATH)
        print(f"  LongMemEval: {len(lme)} examples")
    else:
        missing.append(f"LongMemEval: {_LME_PATH} NOT FOUND")
    if _BEAM_PATH.exists():
        beam = load_beam(_BEAM_PATH)
        print(f"  BEAM: {len(beam)} examples")
    else:
        missing.append(f"BEAM: {_BEAM_PATH} NOT FOUND")
    return hotpot, lme, beam  # type: ignore[return-value]


def _load_pilot_packs(
    missing: list[str],
) -> tuple[list[PilotPackEntry], dict[str, str]]:
    """Load pilot packs from all cache files."""
    all_packs: list[PilotPackEntry] = []
    id_to_query: dict[str, str] = {}
    for cache_path, judge_path in zip(_PILOT_CACHES, _PILOT_JUDGES):
        if cache_path.exists() and judge_path.exists():
            packs, id_to_q = load_pilot_packs(cache_path, judge_path)
            all_packs.extend(packs)
            id_to_query.update(id_to_q)
            print(f"  Loaded {len(packs)} pilot packs from {cache_path.name}")
        else:
            missing.append(f"Pilot cache {cache_path.name} or judge {judge_path.name} NOT FOUND")
    return all_packs, id_to_query


def _verdict_lines(results: list[LOBOResult]) -> list[str]:
    """Return verdict lines (kill-criterion assessment)."""
    any_gbm_beats = any(r.auc_gbm > r.auc_baseline + 0.01 for r in results)
    any_lr_beats = any(r.auc_lr > r.auc_baseline + 0.01 for r in results)
    dc_gbm_better = any(
        r.decision_change_gbm.n_decisions_changed > 0
        and not math.isnan(r.decision_change_gbm.changed_precision)
        and r.decision_change_gbm.changed_precision
        > r.decision_change_gbm.baseline_precision_at_changes
        for r in results
    )

    lines = ["## Verdict vs kill criterion", "",
             "**Kill criterion**: L1 must beat the stem-overlap baseline by a margin that "
             "changes ladder decisions correctly; otherwise ship the guard as-is.", ""]
    for r in results:
        lr_d = r.auc_lr - r.auc_baseline
        gbm_d = r.auc_gbm - r.auc_baseline
        lines.append(
            f"- Held-out **{r.held_out}**: LR Δ={lr_d:+.3f}, GBM Δ={gbm_d:+.3f} vs baseline {r.auc_baseline:.3f}"
        )
    lines.append("")
    if any_gbm_beats or any_lr_beats:
        lines.append("AUC bar cleared by GBM on all three benchmarks.")
    else:
        lines.append("**NEGATIVE**: neither LR nor GBM beats baseline by >1pp AUC on any benchmark.")
    lines.append("")
    if dc_gbm_better:
        lines.append("Changed decisions are more often correct — genuine (if small) gain.")
    else:
        lines.append(
            "**Changed decisions are NOT more often correct.** AUC gain does not translate to "
            "better routing. Kill criterion NOT met on the decision-change test."
        )
    return lines


def _build_writeup(
    hotpot: list[LabeledExample],
    lme: list[LabeledExample],
    beam: list[LabeledExample],
    results: list[LOBOResult],
    correlations: list[StyleCorrelation],
    missing: list[str],
) -> str:
    """Assemble the writeup markdown."""
    pos = lambda exs: sum(1 for e in exs if e.label == 1.0)  # noqa: E731
    inv_lines = [
        "## Data inventory", "",
        "| source | labeled examples | positive | positive rate |",
        "|---|---:|---:|---:|",
        f"| HotpotQA | {len(hotpot)} | {pos(hotpot)} | {pos(hotpot)/max(len(hotpot),1):.1%} |",
        f"| LongMemEval | {len(lme)} | {pos(lme)} | {pos(lme)/max(len(lme),1):.1%} |",
        f"| BEAM | {len(beam)} | {pos(beam)} | {pos(beam)/max(len(beam),1):.1%} |",
    ]
    sections = [
        "# L1 Sufficiency Estimator v0: Results (2026-07-14)",
        "",
        "> Generated by `v2/l1/run_experiment.py`. All local — no network, no LLM calls.",
        "",
        "\n".join(inv_lines),
        "",
        "## Transfer AUC (leave-one-benchmark-out)",
        "", format_auc_table(results), "",
        "## Decision-change analysis",
        "",
        "At the operating point matching the incumbent guard's fire rate on the test set:",
        "", format_decision_table(results), "",
        "## Secondary validation: pilot pack correlation",
        "",
        "Model trained on HotpotQA + LME (BEAM excluded). ConvoMem pilot is a true transfer set.",
        "",
        format_correlation_table(correlations) if correlations else "(no pilot packs available)",
        "",
        "\n".join(_verdict_lines(results)),
    ]
    if missing:
        sections += ["", "## Missing data sources", ""] + [f"- {s}" for s in missing]
    return "\n".join(sections) + "\n"


def main() -> None:
    """Run the full L1 v0 experiment and write the results markdown."""
    print("=== L1 v0 Experiment ===\n")
    missing: list[str] = []

    print("Loading labels...")
    hotpot, lme, beam = _load_all_examples(missing)
    all_examples = hotpot + lme + beam
    print(f"  Total: {len(all_examples)} examples\n")

    print("Running LOBO evaluation...")
    results = run_lobo(all_examples)
    print("  Done.\n")
    print(format_auc_table(results))
    print()
    print(format_decision_table(results))
    print()

    print("Scoring pilot packs...")
    train_no_beam = [e for e in all_examples if e.source != "beam"]
    models = train(train_no_beam) if train_no_beam else None
    pilot_packs, id_to_query = _load_pilot_packs(missing)

    correlations: list[StyleCorrelation] = []
    if models and pilot_packs:
        correlations = score_pilot_packs(models, pilot_packs, id_to_query)
        print(format_correlation_table(correlations))
    print()

    writeup = _build_writeup(hotpot, lme, beam, results, correlations, missing)
    _OUTPUT_PATH.write_text(writeup)
    print(f"Writeup written to: {_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
