"""Deterministic core for the WikiContradict transfer test (Gate 0, no LLM).

Loads the IBM WikiContradict RAG-QA benchmark, applies a deterministic
row-validity filter, renders the four 2x2 evidence/prompt conditions, and scores
reader outputs by both-sides presence + contradiction flagging. No model calls.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

REF_ANSWER_SEP = "|"
RENDER_PLAIN = "plain"
RENDER_MARKED = "marked"
PROMPT_STANDARD = "standard"
PROMPT_AWARE = "aware"
RENDER_STYLES = (RENDER_PLAIN, RENDER_MARKED)
PROMPT_STYLES = (PROMPT_STANDARD, PROMPT_AWARE)

_WORD_RE = re.compile(r"[a-z0-9]+")
_FLAG_TOKENS = (
    "contradict", "conflict", "however", "differ", "whereas", "disagree",
    "inconsisten", "on the other hand", "two different", "one source",
    "another source", "one passage", "another passage", "while the",
)

_STANDARD_INSTRUCTION = (
    "Answer the question using only the passages above. Be concise."
)
_AWARE_INSTRUCTION = (
    "The passages above may contain contradictory information. If they do, "
    "provide a comprehensive answer that surfaces every conflicting viewpoint "
    "and states explicitly that the sources disagree. Do not silently pick one "
    "side or merge them. Answer using only the passages above."
)


@dataclass(frozen=True)
class ContradictionRow:
    question_id: str
    question: str
    context1: str
    context2: str
    answer1: str
    answer2: str
    contradict_type: str


@dataclass(frozen=True)
class ScoreResult:
    both_present: bool
    flagged: bool
    det_correct: bool  # strict lower bound: both sides AND a conflict signal
    judge_candidate: bool  # both sides present but no lexical flag -> needs judge


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def load_rows(csv_path: Path) -> list[ContradictionRow]:
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            ContradictionRow(
                question_id=str(record["question_ID"]).strip(),
                question=record["question"].strip(),
                context1=record["context1"].strip(),
                context2=record["context2"].strip(),
                answer1=record["answer1"].strip(),
                answer2=record["answer2"].strip(),
                contradict_type=record["contradictType"].strip(),
            )
            for record in reader
        ]


def is_valid_row(row: ContradictionRow) -> bool:
    """Deterministic validity gate: a genuine, scorable two-sided contradiction."""
    fields_present = all(
        (row.question, row.context1, row.context2, row.answer1, row.answer2)
    )
    if not fields_present:
        return False
    norm1, norm2 = _normalize(row.answer1), _normalize(row.answer2)
    if not norm1 or not norm2 or norm1 == norm2:
        return False
    return True


def filter_valid(rows: list[ContradictionRow]) -> tuple[list[ContradictionRow], list[str]]:
    kept, dropped = [], []
    for row in rows:
        (kept if is_valid_row(row) else dropped).append(row)
    return kept, [r.question_id for r in dropped]


def render_evidence(row: ContradictionRow, style: str) -> str:
    if style == RENDER_PLAIN:
        return f"Passage 1: {row.context1}\nPassage 2: {row.context2}"
    if style == RENDER_MARKED:
        return (
            "The following two passages come from equally trustworthy sources "
            "and make incompatible claims about the question.\n"
            f"[CLAIM A] (Passage 1): {row.context1}\n"
            f"[CLAIM B] (Passage 2): {row.context2}\n"
            "[CONTRADICTION] Claim A and Claim B disagree; both are on record."
        )
    raise ValueError(f"Unknown render style: {style}")


def _instruction(prompt_style: str) -> str:
    if prompt_style == PROMPT_STANDARD:
        return _STANDARD_INSTRUCTION
    if prompt_style == PROMPT_AWARE:
        return _AWARE_INSTRUCTION
    raise ValueError(f"Unknown prompt style: {prompt_style}")


def build_prompt(row: ContradictionRow, render_style: str, prompt_style: str) -> str:
    evidence = render_evidence(row, render_style)
    return (
        f"{evidence}\n\n"
        f"Question: {row.question}\n\n"
        f"{_instruction(prompt_style)}"
    )


def _contains_answer(answer: str, response_norm: str) -> bool:
    answer_norm = _normalize(answer)
    if not answer_norm:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(answer_norm)}(?![a-z0-9])", response_norm) is not None


def score_response(row: ContradictionRow, response: str) -> ScoreResult:
    """Deterministic proxy for WikiContradict correctness (both sides + flag)."""
    response_norm = _normalize(response)
    both_present = _contains_answer(row.answer1, response_norm) and _contains_answer(
        row.answer2, response_norm
    )
    flagged = any(token in response_norm for token in _FLAG_TOKENS)
    det_correct = both_present and flagged
    return ScoreResult(
        both_present=both_present,
        flagged=flagged,
        det_correct=det_correct,
        judge_candidate=both_present and not flagged,
    )


def condition_cells() -> tuple[tuple[str, str], ...]:
    return tuple((r, p) for r in RENDER_STYLES for p in PROMPT_STYLES)
