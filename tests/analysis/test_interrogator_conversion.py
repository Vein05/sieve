"""Tests for the interrogator v0 conversion evidence assembly."""

from __future__ import annotations

from analysis import interrogator_conversion as ic


def _row(pool_ids, gold_ids, query="when does the sprint end", pool_content_repeat=1):
    body = "pooled content " * pool_content_repeat
    return {
        "example_id": "beam-test-001",
        "ability": "event_ordering",
        "chat_id": "1",
        "query": query,
        "reference_answer": "March 29",
        "candidate_memories": [
            {"memory_id": mid, "content": f"batch 1 | turn {i} | user: {body}{mid}", "bm25_rank": i + 1}
            for i, mid in enumerate(pool_ids)
        ],
        "evidence_sufficiency": {"answer_bearing_memory_ids": list(gold_ids)},
    }


def _store(ids):
    return {mid: f"batch 1 | turn X | store content {mid} " * 3 for mid in ids}


def test_missing_gold_cohort_classification():
    row = _row(["turn_0001", "turn_0002"], ["turn_0002", "turn_0099"])
    assert ic.cohort_of(row).is_missing_gold is True
    row2 = _row(["turn_0001", "turn_0002"], ["turn_0002"])
    assert ic.cohort_of(row2).is_missing_gold is False


def test_ceiling_acquires_exactly_missing_gold():
    row = _row(["turn_0001"], ["turn_0001", "turn_0050", "turn_0060"])
    store = _store(["turn_0001", "turn_0050", "turn_0060"])
    acquired = ic.acquire(row, store, ic.CONDITION_CEILING)
    assert set(acquired) == {"turn_0050", "turn_0060"}


def test_fixed_acquires_nothing():
    row = _row(["turn_0001"], ["turn_0099"])
    store = _store(["turn_0001", "turn_0099"])
    assert ic.acquire(row, store, ic.CONDITION_FIXED) == []


def test_acquired_units_placed_first_and_survive_cap():
    row = _row(["turn_0001", "turn_0002"], ["turn_0001", "turn_0090"])
    store = _store(["turn_0001", "turn_0002", "turn_0090"])
    ev = ic.assemble_evidence(row, store, ic.CONDITION_CEILING)
    # turn_0090 is the missing gold; it must be acquired and appear before pool text.
    assert "turn_0090" in ev.acquired_ids
    assert ev.source_memory_ids[0] == "turn_0090"
    assert ev.n_acquired_survived >= 1


def test_token_cap_enforced():
    big = {f"turn_{i:04d}": "x " * 50_000 for i in range(1, 40)}
    pool_ids = [f"turn_{i:04d}" for i in range(1, 21)]
    row = _row(pool_ids, ["turn_0030"])
    ev = ic.assemble_evidence(row, big, ic.CONDITION_ADAPTIVE_K)
    assert ev.evidence_token_estimate <= ic.EVIDENCE_TOKEN_CAP + 5


def test_all_conditions_matched_budget_when_pool_overflows():
    big = {f"turn_{i:04d}": "word " * 20_000 for i in range(1, 60)}
    pool_ids = [f"turn_{i:04d}" for i in range(1, 21)]
    row = _row(pool_ids, ["turn_0055"], pool_content_repeat=4000)
    tokens = {
        cond: ic.assemble_evidence(row, big, cond).evidence_token_estimate
        for cond in ic.CONDITIONS
    }
    # When the pool alone overflows E, every condition fills the same budget.
    for cond, tok in tokens.items():
        assert abs(tok - ic.EVIDENCE_TOKEN_CAP) <= ic.EVIDENCE_TOKEN_CAP * 0.05, (cond, tok)


def test_interrogator_deterministic():
    row = _row(["turn_0001", "turn_0002"], ["turn_0080"], query="Alice project deadline")
    store = _store([f"turn_{i:04d}" for i in range(1, 90)])
    a = ic.acquire(row, store, ic.CONDITION_INTERROGATOR)
    b = ic.acquire(row, store, ic.CONDITION_INTERROGATOR)
    assert a == b
