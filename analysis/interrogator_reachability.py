"""Kill-test: classify missing gold as k-reachable / probe-exclusive / unreachable.

Local, deterministic, no network. See research/interrogator-reachability-2026-07.md.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from analysis.interrogator_bm25 import BM25Index
from analysis.interrogator_probes import Probe, generate_probes

BEAM_SLICE = Path(
    "/Users/vein/Documents/research/memory-eligibility-feasibility"
    "/dataset-slices/beam_transfer_bm25_top20_v0.jsonl"
)
POOL_SIZE = 20
BUDGETS = ((30, 5, 6), (80, 10, 8), (180, 15, 12))  # (B, n_probes, top_m)
ADJACENCY_RADIUS = 2
RETRIEVAL_BOUND_ABILITIES = frozenset(
    {"event_ordering", "multi_session_reasoning", "temporal_reasoning", "contradiction_resolution"}
)


@dataclass
class RowResult:
    ability: str
    chat_key: str
    store_size: int
    n_missing: int
    k_recovered: int = 0
    probe_recovered: int = 0
    probe_exclusive: int = 0  # probe AND NOT k
    k_exclusive: int = 0  # k AND NOT probe
    content_absent: int = 0  # missing gold whose content is not in reconstructed store
    probe_type_hits: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def _turn_index(memory_id: str) -> int | None:
    if memory_id.startswith("turn_"):
        try:
            return int(memory_id.split("_")[1])
        except (IndexError, ValueError):
            return None
    return None


def build_chat_stores(rows: list[dict]) -> dict[str, dict[str, str]]:
    """Union each chat's sibling pools into a reconstructed content store."""
    stores: dict[str, dict[str, str]] = defaultdict(dict)
    for row in rows:
        key = f"{row['chat_size']}::{row['chat_id']}"
        for cand in row["candidate_memories"]:
            stores[key][cand["memory_id"]] = cand["content"]
    return stores


def _adjacency_recovered(
    pool_ids: frozenset[str], store: dict[str, str], missing_gold: set[str], budget: int
) -> set[str]:
    """Turns adjacent to in-pool turns (BEAM order signal), capped at budget units."""
    pool_indices = {i for i in (_turn_index(m) for m in pool_ids) if i is not None}
    wanted: list[str] = []
    for idx in sorted(pool_indices):
        for delta in range(-ADJACENCY_RADIUS, ADJACENCY_RADIUS + 1):
            if delta == 0:
                continue
            cand = f"turn_{idx + delta:04d}"
            if cand in store and cand not in pool_ids and cand not in wanted:
                wanted.append(cand)
    acquired = set(wanted[:budget])
    return acquired & missing_gold


def _k_expansion(index: BM25Index, query: str, pool_ids: frozenset[str], budget: int) -> set[str]:
    ranked = index.rank(query, exclude=pool_ids)
    return {doc_id for doc_id, _ in ranked[:budget]}


def _probe_expansion(
    index: BM25Index,
    probes: list[Probe],
    pool_ids: frozenset[str],
    top_m: int,
    budget: int,
) -> tuple[set[str], dict[str, str]]:
    acquired: list[str] = []
    origin: dict[str, str] = {}
    exclude = set(pool_ids)
    for probe in probes:
        if len(acquired) >= budget:
            break
        ranked = index.rank(probe.text, exclude=frozenset(exclude))
        for doc_id, score in ranked[:top_m]:
            if score <= 0 or doc_id in exclude:
                continue
            acquired.append(doc_id)
            origin[doc_id] = probe.ptype
            exclude.add(doc_id)
            if len(acquired) >= budget:
                break
    return set(acquired), origin


def classify_row(row: dict, store: dict[str, str], budget_spec: tuple[int, int, int]) -> RowResult | None:
    budget, n_probes, top_m = budget_spec
    ability = row["ability"]
    pool_ids = frozenset(c["memory_id"] for c in row["candidate_memories"])
    gold = set(row["evidence_sufficiency"].get("answer_bearing_memory_ids") or [])
    missing = gold - pool_ids
    if not missing:
        return None
    result = RowResult(
        ability=ability,
        chat_key=f"{row['chat_size']}::{row['chat_id']}",
        store_size=len(store),
        n_missing=len(missing),
    )
    accessible_missing = {g for g in missing if g in store}
    result.content_absent = len(missing) - len(accessible_missing)
    index = BM25Index.build([(mid, txt) for mid, txt in store.items()])

    k_ids = _k_expansion(index, row["query"], pool_ids, budget)
    pool_texts = [c["content"] for c in row["candidate_memories"]]
    probes = generate_probes(row["query"], pool_texts, n_probes)
    probe_ids, origin = _probe_expansion(index, probes, pool_ids, top_m, budget)
    adj = _adjacency_recovered(pool_ids, store, accessible_missing, budget)
    probe_ids |= adj
    for g in adj:
        origin.setdefault(g, "adjacency")

    for g in accessible_missing:
        in_k = g in k_ids
        in_probe = g in probe_ids
        if in_k:
            result.k_recovered += 1
        if in_probe:
            result.probe_recovered += 1
            result.probe_type_hits[origin.get(g, "unknown")] += 1
        if in_probe and not in_k:
            result.probe_exclusive += 1
        if in_k and not in_probe:
            result.k_exclusive += 1
    return result


def load_rows(path: Path = BEAM_SLICE) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def run(path: Path = BEAM_SLICE) -> dict[int, list[RowResult]]:
    rows = load_rows(path)
    stores = build_chat_stores(rows)
    out: dict[int, list[RowResult]] = {}
    for spec in BUDGETS:
        results: list[RowResult] = []
        for row in rows:
            store = stores[f"{row['chat_size']}::{row['chat_id']}"]
            res = classify_row(row, store, spec)
            if res is not None:
                results.append(res)
        out[spec[0]] = results
    return out


if __name__ == "__main__":
    from analysis.interrogator_report import print_report

    print_report(run())
