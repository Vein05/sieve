"""Tests for the interrogator reachability kill-test harness."""

from __future__ import annotations

from analysis.interrogator_bm25 import BM25Index, tokenize
from analysis.interrogator_probes import generate_probes
from analysis.interrogator_reachability import RowResult, classify_row
from analysis.interrogator_report import _agg


def test_tokenize_lowercase_alphanumeric():
    assert tokenize("Hello, World! 2024/03/29") == ["hello", "world", "2024", "03", "29"]


def test_bm25_ranks_relevant_doc_first():
    docs = [
        ("d1", "the sprint ends on march 29 for user registration"),
        ("d2", "yoga mats are essential for a comfortable practice"),
        ("d3", "analytics dashboard with income and expense breakdown"),
    ]
    index = BM25Index.build(docs)
    ranked = index.rank("when does the sprint end")
    assert ranked[0][0] == "d1"
    assert ranked[0][1] > ranked[1][1]


def test_bm25_idf_rewards_rare_terms():
    docs = [("d1", "common common common rare"), ("d2", "common common common"),
            ("d3", "common common word")]
    index = BM25Index.build(docs)
    # "rare" appears in one doc; querying it should score d1 above zero, others zero
    ranked = dict(index.rank("rare"))
    assert ranked["d1"] > 0
    assert ranked["d2"] == 0.0


def test_bm25_exclude_removes_ids():
    docs = [("a", "sprint deadline"), ("b", "sprint deadline")]
    index = BM25Index.build(docs)
    ranked = index.rank("sprint", exclude=frozenset({"a"}))
    assert [d for d, _ in ranked] == ["b"]


def test_probe_generation_deterministic():
    query = "when does my second sprint end for analytics"
    pool = [
        "Sprint 1 ended on March 29 with user auth by Alice Smith.",
        "Sprint 2 targets Analytics Dashboard by April 19.",
    ]
    p1 = generate_probes(query, pool, cap=10)
    p2 = generate_probes(query, pool, cap=10)
    assert [p.text for p in p1] == [p.text for p in p2]
    assert len(p1) <= 10


def test_probe_cap_respected():
    pool = ["Alice Bob Carol Dave Eve Frank on March 1 and April 2 and May 3."]
    probes = generate_probes("test query words here", pool, cap=3)
    assert len(probes) <= 3


def test_probe_no_gold_leakage():
    # Probes must be derivable from pool + query only; a gold-only token must never appear.
    query = "what did I decide about the database"
    pool = ["We discussed the database schema for users."]
    gold_only_token = "supersecretgoldentity"
    probes = generate_probes(query, pool, cap=20)
    joined = " ".join(p.text for p in probes).lower()
    assert gold_only_token not in joined
    for p in probes:
        for tok in tokenize(p.text):
            assert tok in tokenize(query) or any(tok in tokenize(t) for t in pool)


def test_probe_types_present():
    query = "when did the deadline change for the project timeline"
    pool = [
        "Project Timeline set deadline to March 29, 2024 by Alice.",
        "Project Timeline moved deadline to March 31, 2024.",
    ]
    probes = generate_probes(query, pool, cap=20)
    types = {p.ptype for p in probes}
    assert "entity" in types
    assert "temporal" in types


def _toy_row(pool_ids, gold, query="find the sprint deadline"):
    cands = [{"memory_id": mid, "content": f"turn about {mid} deadline sprint"} for mid in pool_ids]
    return {
        "ability": "event_ordering",
        "chat_size": "128K",
        "chat_id": "1",
        "query": query,
        "candidate_memories": cands,
        "evidence_sufficiency": {"answer_bearing_memory_ids": gold},
    }


def test_classify_returns_none_when_no_missing_gold():
    row = _toy_row(["turn_0001", "turn_0002"], gold=["turn_0001"])
    store = {"turn_0001": "sprint deadline text", "turn_0002": "other"}
    assert classify_row(row, store, (30, 5, 6)) is None


def test_classify_budget_accounting_consistent():
    pool_ids = [f"turn_{i:04d}" for i in range(1, 6)]
    gold = ["turn_0001", "turn_0020", "turn_0021"]  # two missing from pool
    row = _toy_row(pool_ids, gold)
    store = {f"turn_{i:04d}": f"sprint deadline unit {i}" for i in range(1, 40)}
    res = classify_row(row, store, (30, 5, 6))
    assert isinstance(res, RowResult)
    assert res.n_missing == 2
    accessible = res.n_missing - res.content_absent
    # exclusive counts are subsets of their recovered counts
    assert res.probe_exclusive <= res.probe_recovered
    assert res.k_exclusive <= res.k_recovered
    assert res.k_recovered <= accessible
    assert res.probe_recovered <= accessible


def test_content_absent_counted_when_gold_not_in_store():
    row = _toy_row(["turn_0001"], gold=["turn_0001", "turn_0099"])
    store = {"turn_0001": "sprint", "turn_0002": "deadline"}  # turn_0099 absent
    res = classify_row(row, store, (30, 5, 6))
    assert res.content_absent == 1


def test_agg_partitions_accessible():
    row = _toy_row([f"turn_{i:04d}" for i in range(1, 6)], gold=["turn_0021", "turn_0022"])
    store = {f"turn_{i:04d}": f"sprint deadline {i}" for i in range(1, 40)}
    res = classify_row(row, store, (30, 5, 6))
    agg = _agg([res])
    # A(k-only) + B(probe-excl) + both + unreachable-accessible == accessible
    both = agg["both"]
    accessible = agg["accessible"]
    partition = agg["k_exclusive"] + agg["probe_exclusive"] + both
    assert partition <= accessible
