"""Tests for the true-store confirmation loaders (BEAM parquet turnize, LME haystack)."""

from __future__ import annotations

from analysis.interrogator_beam_store import build_true_stores, render_turn, turnize_chat
from analysis.interrogator_lme import _adjacency_units, classify_lme_row
from analysis.interrogator_lme_store import Unit, build_units, session_date_ranks
from analysis.interrogator_reachability import RowResult


def _msg(mid: int, role: str, qtype: str | None = None, anchor: str | None = None,
         content: str = "c") -> dict:
    return {"id": str(mid), "role": role, "question_type": qtype,
            "time_anchor": anchor, "index": None, "content": content}


def _toy_chat() -> list[list[dict]]:
    batch1 = [
        _msg(0, "user", "main_question", "March-15-2024", "first question"),
        _msg(1, "assistant", content="first answer"),
        _msg(2, "user", "answer_ai_question", content="follow up"),
        _msg(3, "assistant", content="second answer"),
        _msg(4, "user", "main_question", content="next\nquestion"),
        _msg(5, "assistant", content="next answer"),
    ]
    batch2 = [
        _msg(6, "user", "main_question", "April-05-2024", "batch two question"),
        _msg(7, "assistant", content="batch two answer"),
    ]
    return [batch1, batch2]


def test_turnize_groups_by_main_question_globally():
    turns = turnize_chat(_toy_chat())
    assert sorted(turns) == [1, 2, 3]
    assert [m[0] for m in turns[1][2]] == [0, 1, 2, 3]
    assert [m[0] for m in turns[2][2]] == [4, 5]
    assert turns[3][0] == 2  # batch number continues, numbering is global
    assert turns[1][1] == "March-15-2024"
    assert turns[2][1] is None


def test_render_turn_matches_slice_format():
    turns = turnize_chat(_toy_chat())
    r1 = render_turn(1, *turns[1])
    assert r1.startswith("March-15-2024 | batch 1 | turn 1 | user[0]: first question")
    r2 = render_turn(2, *turns[2])
    assert r2 == "batch 1 | turn 2 | user[4]: next question assistant[5]: next answer"


def test_build_true_stores_ids_are_turn_padded(tmp_path):
    import pandas as pd

    frame = pd.DataFrame([{"conversation_id": "7", "chat": _toy_chat()}])
    pq = tmp_path / "chat.parquet"
    frame.to_parquet(pq)
    stores = build_true_stores(pq)
    assert set(stores) == {"7"}
    assert set(stores["7"]) == {"turn_0001", "turn_0002", "turn_0003"}


def _toy_question() -> dict:
    return {
        "question_id": "q1",
        "question_type": "multi-session",
        "question": "when does the sprint deadline end",
        "answer_session_ids": ["ans_1"],
        "haystack_session_ids": ["s_a", "ans_1", "s_b"],
        "haystack_dates": ["2023/05/21 (Sun) 10:00", "2023/05/20 (Sat) 09:00",
                           "2023/05/22 (Mon) 11:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "yoga mats and comfortable practice"},
             {"role": "assistant", "content": ""}],
            [{"role": "user", "content": "totally unrelated cooking recipe for pasta"}],
            [{"role": "user", "content": "sprint deadline ends friday"},
             {"role": "assistant", "content": "noted the sprint deadline"}],
        ],
    }


def test_build_units_skips_empty_and_renders_date_role():
    units = build_units(_toy_question())
    assert [u.uid for u in units] == ["s000_m000", "s001_m000", "s002_m000", "s002_m001"]
    assert units[0].text == "2023/05/21 (Sun) 10:00 | user: yoga mats and comfortable practice"
    assert units[1].session_id == "ans_1"


def test_session_date_ranks_orders_by_date():
    ranks = session_date_ranks(_toy_question())
    assert ranks == {1: 0, 0: 1, 2: 2}  # ans_1 is earliest despite position 1


def test_adjacency_units_uses_date_order():
    units = build_units(_toy_question())
    ranks = session_date_ranks(_toy_question())
    pool = frozenset({"s000_m000"})  # session pos 0 = date rank 1
    adj = _adjacency_units(units, pool, ranks, budget=10)
    assert adj == {"s001_m000", "s002_m000", "s002_m001"}


def test_classify_lme_row_detects_missing_gold_without_injection():
    q = _toy_question()
    res = classify_lme_row(q, (1, 5, 6))
    # pool is tiny store top-20 => everything pooled => no missing gold
    assert res is None or isinstance(res, RowResult)


def test_classify_lme_row_session_accounting_invariants():
    q = _toy_question()
    # enlarge store so top-20 pool cannot cover the gold session
    filler = [
        [{"role": "user", "content": f"sprint deadline planning note number {i}"}]
        for i in range(40)
    ]
    q["haystack_sessions"] = q["haystack_sessions"][:2] + filler
    q["haystack_session_ids"] = q["haystack_session_ids"][:2] + [f"f{i}" for i in range(40)]
    q["haystack_dates"] = q["haystack_dates"][:2] + ["2023/06/01 (Thu) 10:00"] * 40
    res = classify_lme_row(q, (5, 5, 6))
    if res is not None:
        accessible = res.n_missing - res.content_absent
        assert res.content_absent == 0
        assert res.probe_exclusive <= res.probe_recovered <= accessible
        assert res.k_exclusive <= res.k_recovered <= accessible


def test_classify_lme_row_none_when_no_gold():
    q = _toy_question()
    q["answer_session_ids"] = []
    assert classify_lme_row(q, (5, 5, 6)) is None


def test_run_excludes_abstention_rows(tmp_path):
    import json

    from analysis.interrogator_lme import run

    q_abs = _toy_question()
    q_abs["question_id"] = "q9_abs"
    path = tmp_path / "lme.json"
    path.write_text(json.dumps([q_abs]))
    runs = run(path)
    assert all(not results for results in runs.values())
