"""Part A confirmation: re-run the frozen A/B/C kill-test on the TRUE BEAM 128K store."""

from __future__ import annotations

from pathlib import Path

from analysis.interrogator_beam_store import BEAM_PARQUET, build_true_stores
from analysis.interrogator_reachability import (
    BEAM_SLICE,
    RETRIEVAL_BOUND_ABILITIES,
    RowResult,
    classify_row,
    load_rows,
)
from analysis.interrogator_report import print_report, probe_type_ablation

TRUE_BUDGETS = ((30, 5, 6), (80, 10, 8))  # (B, n_probes, top_m); B=180 exceeds every store


def verify_alignment(rows: list[dict], stores: dict[str, dict[str, str]]) -> int:
    """Fail loud unless every slice unit matches the true-store rendering verbatim."""
    checked = 0
    for row in rows:
        store = stores[str(row["chat_id"])]
        for cand in row["candidate_memories"]:
            if store.get(cand["memory_id"]) != cand["content"]:
                raise ValueError(f"alignment failure: {row['example_id']} {cand['memory_id']}")
            checked += 1
        for gold in row["evidence_sufficiency"].get("answer_bearing_memory_ids") or []:
            if gold not in store:
                raise ValueError(f"gold id absent from true store: {row['example_id']} {gold}")
    return checked


def run_true(
    slice_path: Path = BEAM_SLICE, parquet_path: Path = BEAM_PARQUET
) -> dict[int, list[RowResult]]:
    """Identical frozen protocol with the store swapped to the true parquet-derived chats."""
    rows = load_rows(slice_path)
    stores = build_true_stores(parquet_path)
    verify_alignment(rows, stores)
    out: dict[int, list[RowResult]] = {}
    for spec in TRUE_BUDGETS:
        results: list[RowResult] = []
        for row in rows:
            res = classify_row(row, stores[str(row["chat_id"])], spec)
            if res is not None:
                results.append(res)
        out[spec[0]] = results
    return out


if __name__ == "__main__":
    runs = run_true()
    print_report(runs)
    for budget, results in runs.items():
        rb = [r for r in results if r.ability in RETRIEVAL_BOUND_ABILITIES]
        print(f"B={budget} RETRIEVAL-BOUND probe-type ablation: {probe_type_ablation(rb)}")
