from __future__ import annotations

from pathlib import Path

import pytest

from analysis.wikicontradict import (
    PROMPT_AWARE,
    RENDER_MARKED,
    RENDER_PLAIN,
    ContradictionRow,
    build_prompt,
    condition_cells,
    filter_valid,
    is_valid_row,
    load_rows,
    render_evidence,
    score_response,
)

DATASET = Path("data/wikicontradict/WikiContradict_dataset_v1_rag_qa.csv")


def _row(**overrides: str) -> ContradictionRow:
    base = dict(
        question_id="1",
        question="Which compound is present, apomorphine or aporphine?",
        context1="Apomorphine is the main psychoactive compound.",
        context2="It contains the alkaloid aporphine, not apomorphine.",
        answer1="Apomorphine",
        answer2="Aporphine",
        contradict_type="Implicit (reasoning required)",
    )
    base.update(overrides)
    return ContradictionRow(**base)  # type: ignore[arg-type]


def test_condition_cells_is_2x2() -> None:
    assert len(condition_cells()) == 4


def test_valid_row_accepts_two_sided_conflict() -> None:
    assert is_valid_row(_row()) is True


def test_validity_rejects_identical_answers() -> None:
    assert is_valid_row(_row(answer1="Same", answer2="same")) is False


def test_validity_rejects_empty_side() -> None:
    assert is_valid_row(_row(answer2="")) is False


def test_marked_render_injects_contradiction_marker() -> None:
    text = render_evidence(_row(), RENDER_MARKED)
    assert "[CONTRADICTION]" in text and "[CLAIM A]" in text and "[CLAIM B]" in text


def test_plain_render_has_no_marker() -> None:
    text = render_evidence(_row(), RENDER_PLAIN)
    assert "[CONTRADICTION]" not in text


def test_aware_prompt_requests_all_viewpoints() -> None:
    prompt = build_prompt(_row(), RENDER_PLAIN, PROMPT_AWARE)
    assert "conflicting" in prompt.lower() or "disagree" in prompt.lower()


def test_score_both_sides_and_flag_is_correct() -> None:
    resp = "The sources contradict: one says Apomorphine, the other says Aporphine."
    result = score_response(_row(), resp)
    assert result.both_present and result.flagged and result.det_correct


def test_score_one_side_only_is_not_correct() -> None:
    result = score_response(_row(), "The answer is simply Apomorphine.")
    assert not result.both_present and not result.det_correct


def test_score_both_sides_no_flag_is_judge_candidate() -> None:
    result = score_response(_row(), "Apomorphine. Aporphine.")
    assert result.both_present and not result.flagged and result.judge_candidate


def test_substring_answer_does_not_falsely_match() -> None:
    row = _row(answer1="pin", answer2="Aporphine")
    result = score_response(row, "The answer is Aporphine only.")
    assert not result.both_present  # 'pin' must not match inside 'aporphine'


@pytest.mark.skipif(not DATASET.exists(), reason="dataset not downloaded")
def test_real_dataset_loads_and_filters() -> None:
    rows = load_rows(DATASET)
    assert len(rows) == 253
    valid, dropped = filter_valid(rows)
    assert len(valid) > 100
    assert len(valid) + len(dropped) == len(rows)
