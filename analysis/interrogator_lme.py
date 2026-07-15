"""Part B: honest LongMemEval-S missing-gold A/B/C classification over the full haystack."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from analysis.interrogator_bm25 import BM25Index
from analysis.interrogator_lme_store import (
    LME_HAYSTACK,
    POOL_SIZE,
    Unit,
    build_units,
    load_questions,
    session_date_ranks,
)
from analysis.interrogator_probes import generate_probes
from analysis.interrogator_reachability import (
    ADJACENCY_RADIUS,
    RowResult,
    _k_expansion,
    _probe_expansion,
)
from analysis.interrogator_report import _agg, _fmt, _saturation_stats, probe_type_ablation

LME_BUDGETS = ((30, 5, 6), (80, 10, 8), (180, 15, 12))  # (B, n_probes, top_m)
LME_RETRIEVAL_BOUND = frozenset({"multi-session", "temporal-reasoning"})
ABSTENTION_SUFFIX = "_abs"  # answer_session_ids of _abs rows contain no answer by design
BudgetSpec = tuple[int, int, int]


def _adjacency_units(
    units: list[Unit], pool_ids: frozenset[str], ranks: dict[int, int], budget: int
) -> set[str]:
    """Units of sessions within ADJACENCY_RADIUS of any pooled session in date order."""
    by_rank: dict[int, list[Unit]] = defaultdict(list)
    for u in units:
        by_rank[ranks[u.session_pos]].append(u)
    pool_ranks = sorted({ranks[u.session_pos] for u in units if u.uid in pool_ids})
    wanted: list[str] = []
    seen: set[str] = set(pool_ids)
    for rank in pool_ranks:
        for delta in range(-ADJACENCY_RADIUS, ADJACENCY_RADIUS + 1):
            if delta == 0:
                continue
            for u in by_rank.get(rank + delta, []):
                if u.uid not in seen:
                    wanted.append(u.uid)
                    seen.add(u.uid)
    return set(wanted[:budget])


def _score_sessions(result: RowResult, session_units: dict[str, set[str]],
                    k_ids: set[str], probe_ids: set[str], origin: dict[str, str]) -> None:
    """Session-level gold accounting: a session is recovered if any of its units is acquired."""
    for sid in sorted(session_units):
        s_units = session_units[sid]
        in_k = bool(s_units & k_ids)
        hit_units = sorted(s_units & probe_ids)
        if in_k:
            result.k_recovered += 1
        if hit_units:
            result.probe_recovered += 1
            result.probe_type_hits[origin.get(hit_units[0], "unknown")] += 1
            if not in_k:
                result.probe_exclusive += 1
        elif in_k:
            result.k_exclusive += 1


def classify_lme_row(question: dict, budget_spec: BudgetSpec) -> RowResult | None:
    """Frozen A/B/C protocol at message granularity with session-level gold accounting."""
    budget, n_probes, top_m = budget_spec
    gold_sessions = set(question["answer_session_ids"])
    if not gold_sessions:
        return None
    units = build_units(question)
    index = BM25Index.build([(u.uid, u.text) for u in units])
    query = question["question"]
    ranked = index.rank(query)
    pool_ids = frozenset(uid for uid, _ in ranked[:POOL_SIZE])
    text_of = {u.uid: u.text for u in units}
    sid_of = {u.uid: u.session_id for u in units}
    missing = gold_sessions - {sid_of[uid] for uid in pool_ids}
    if not missing:
        return None
    result = RowResult(
        ability=question["question_type"],
        chat_key=question["question_id"],
        store_size=len(units),
        n_missing=len(missing),
    )
    k_ids = _k_expansion(index, query, pool_ids, budget)
    pool_texts = [text_of[uid] for uid, _ in ranked[:POOL_SIZE]]
    probes = generate_probes(query, pool_texts, n_probes)
    probe_ids, origin = _probe_expansion(index, probes, pool_ids, top_m, budget)
    adj = _adjacency_units(units, pool_ids, session_date_ranks(question), budget)
    session_units: dict[str, set[str]] = {
        sid: {u.uid for u in units if u.session_id == sid} for sid in missing
    }
    missing_unit_ids = set().union(*session_units.values())
    for uid in sorted(adj & missing_unit_ids):
        origin.setdefault(uid, "adjacency")
    probe_ids |= adj
    _score_sessions(result, session_units, k_ids, probe_ids, origin)
    return result


def run(path: Path = LME_HAYSTACK, include_abstention: bool = False) -> dict[int, list[RowResult]]:
    """Run all questions at every budget; rows without gold or without missing gold drop out."""
    questions = [
        q
        for q in load_questions(path)
        if include_abstention or not q["question_id"].endswith(ABSTENTION_SUFFIX)
    ]
    out: dict[int, list[RowResult]] = {}
    for spec in LME_BUDGETS:
        results = [res for q in questions if (res := classify_lme_row(q, spec)) is not None]
        out[spec[0]] = results
    return out


def print_lme_report(runs: dict[int, list[RowResult]]) -> None:
    """LME analogue of the frozen report: pooled, retrieval-bound, per-type, ablation."""
    for budget, results in runs.items():
        sat, total = _saturation_stats(results, budget)
        print(f"\n{'=' * 78}\nLME BUDGET B={budget} extra units\n{'=' * 78}")
        print(f"STORE-SATURATION: {sat}/{total} rows have B >= 50% of their store")
        print(f"POOLED ALL: {_fmt(_agg(results))}")
        rb = [r for r in results if r.ability in LME_RETRIEVAL_BOUND]
        print(f"RETRIEVAL-BOUND (multi-session, temporal-reasoning): {_fmt(_agg(rb))}")
        by_type: dict[str, list[RowResult]] = defaultdict(list)
        for r in results:
            by_type[r.ability].append(r)
        for qtype, res in sorted(by_type.items()):
            print(f"  {qtype:28s} {_fmt(_agg(res))}")
        print(f"Probe-type ablation (all): {probe_type_ablation(results)}")
        print(f"Probe-type ablation (retrieval-bound): {probe_type_ablation(rb)}")


if __name__ == "__main__":
    print_lme_report(run())
