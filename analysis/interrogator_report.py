"""Aggregate and print the A/B/C reachability classification."""

from __future__ import annotations

from collections import defaultdict

from analysis.interrogator_reachability import RETRIEVAL_BOUND_ABILITIES, RowResult

LIVE_THRESHOLD = 0.15
DIE_THRESHOLD = 0.05
SATURATION_FRAC = 0.5  # B/store above this => k-arm trivially saturates the store


def _agg(results: list[RowResult]) -> dict[str, float]:
    missing = sum(r.n_missing for r in results)
    accessible = sum(r.n_missing - r.content_absent for r in results)
    k = sum(r.k_recovered for r in results)
    probe = sum(r.probe_recovered for r in results)
    excl = sum(r.probe_exclusive for r in results)
    k_excl = sum(r.k_exclusive for r in results)
    absent = sum(r.content_absent for r in results)
    both = probe - excl
    neither = accessible - k - excl
    return {
        "rows": float(len(results)),
        "missing": float(missing),
        "accessible": float(accessible),
        "content_absent": float(absent),
        "k_recovered": float(k),
        "probe_recovered": float(probe),
        "probe_exclusive": float(excl),
        "k_exclusive": float(k_excl),
        "both": float(both),
        "unreachable": float(neither + absent),
        "excl_frac": (excl / accessible) if accessible else 0.0,
    }


def _by_ability(results: list[RowResult]) -> dict[str, list[RowResult]]:
    out: dict[str, list[RowResult]] = defaultdict(list)
    for r in results:
        out[r.ability].append(r)
    return out


def _fmt(a: dict[str, float]) -> str:
    return (
        f"rows={int(a['rows'])} missing={int(a['missing'])} "
        f"(accessible={int(a['accessible'])}, content_absent={int(a['content_absent'])}) | "
        f"A k-reach={int(a['k_recovered'])} B probe-excl={int(a['probe_exclusive'])} "
        f"both={int(a['both'])} k-excl={int(a['k_exclusive'])} "
        f"C unreach={int(a['unreachable'])} | B/accessible={a['excl_frac']:.1%}"
    )


def probe_type_ablation(results: list[RowResult]) -> dict[str, int]:
    hits: dict[str, int] = defaultdict(int)
    for r in results:
        for ptype, n in r.probe_type_hits.items():
            hits[ptype] += n
    return dict(hits)


def _saturation_stats(results: list[RowResult], budget: int) -> tuple[int, int]:
    saturated = sum(1 for r in results if budget >= SATURATION_FRAC * r.store_size)
    return saturated, len(results)


def print_report(runs: dict[int, list[RowResult]]) -> None:
    for budget, results in runs.items():
        sat, total = _saturation_stats(results, budget)
        print(f"\n{'=' * 78}\nBUDGET B={budget} extra units\n{'=' * 78}")
        print(
            f"STORE-SATURATION: {sat}/{total} rows have B >= {SATURATION_FRAC:.0%} of their "
            f"store (k-arm trivially saturates; probe-exclusive underestimated there)"
        )
        pooled = _agg(results)
        print(f"POOLED ALL: {_fmt(pooled)}")
        rb = [r for r in results if r.ability in RETRIEVAL_BOUND_ABILITIES]
        print(f"RETRIEVAL-BOUND: {_fmt(_agg(rb))}")
        print("Per ability:")
        for ability, res in sorted(_by_ability(results).items()):
            print(f"  {ability:26s} {_fmt(_agg(res))}")
        print(f"Probe-type ablation (recovered-by-type): {probe_type_ablation(results)}")


def verdict(runs: dict[int, list[RowResult]]) -> str:
    fracs = []
    for budget, results in runs.items():
        sat, total = _saturation_stats(results, budget)
        if total and sat / total > 0.5:  # skip saturated budgets
            continue
        rb = [r for r in results if r.ability in RETRIEVAL_BOUND_ABILITIES]
        fracs.append(_agg(rb)["excl_frac"])
    best = max(fracs) if fracs else 0.0
    if best >= LIVE_THRESHOLD:
        return f"LIVES (exclusive-probe frac {best:.1%} >= {LIVE_THRESHOLD:.0%})"
    if best < DIE_THRESHOLD:
        return f"DIES (exclusive-probe frac {best:.1%} < {DIE_THRESHOLD:.0%})"
    return f"JUDGMENT CALL (exclusive-probe frac {best:.1%})"
