from __future__ import annotations

from pathlib import Path

import pytest

from analysis.format_lever import (
    FORMATS,
    FormatRow,
    build_prompt,
    is_correct,
    load_rows,
    render,
)

SLICE = Path("dataset-slices/longmemeval_bm25_top20_v1.jsonl")


def _row() -> FormatRow:
    return FormatRow(
        example_id="x1",
        query="What degree did I graduate with?",
        reference_answer="Business Administration",
        question_type="single-session-user",
        units=("I graduated with a Business Administration degree.", "The weather was nice."),
    )


def test_content_is_byte_identical_across_formats() -> None:
    row = _row()
    for unit in row.units:
        for fmt in FORMATS:
            assert unit in render(row.units, fmt), f"{unit!r} missing from {fmt}"


def test_formats_are_actually_distinct() -> None:
    row = _row()
    rendered = {render(row.units, fmt) for fmt in FORMATS}
    assert len(rendered) == len(FORMATS)


def test_json_format_is_valid_json_evidence() -> None:
    import json

    parsed = json.loads(render(_row().units, "json"))
    assert parsed["evidence"] == list(_row().units)


def test_prompt_holds_instruction_constant() -> None:
    prompts = {build_prompt(_row(), fmt).split("Question:")[1] for fmt in FORMATS}
    assert len(prompts) == 1  # question + instruction identical across formats


def test_scoring_matches_reference_answer() -> None:
    assert is_correct(_row(), "You graduated with Business Administration.")
    assert not is_correct(_row(), "You graduated with Computer Science.")


def test_scoring_is_token_bounded() -> None:
    row = FormatRow("x", "q", "art", "t", ("u",))
    assert not is_correct(row, "the smart cart departed")  # 'art' not a bare token


@pytest.mark.skipif(not SLICE.exists(), reason="slice not present")
def test_real_slice_loads_with_gold_units() -> None:
    rows = load_rows(SLICE)
    assert len(rows) > 400
    assert all(row.units for row in rows)
