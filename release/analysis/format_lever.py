"""Format-lever probe core (deterministic, no LLM).

Tests whether evidence *format* alone moves readers, with content held
byte-identical. Selects a fixed unit set (gold answer-bearing unit + distractors)
per row and renders it in five syntactic formats that differ only in wrapping.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

GOLD_PREFIX = "answer_"
DEFAULT_N_UNITS = 5
FORMATS = ("prose", "numbered", "json", "table", "headers")
INSTRUCTION = "Answer the question using only the evidence above. Give a short, direct answer."
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class FormatRow:
    example_id: str
    query: str
    reference_answer: str
    question_type: str
    units: tuple[str, ...]


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _select_units(candidates: list[dict], n_units: int) -> tuple[str, ...]:
    gold_idx = [i for i, c in enumerate(candidates) if c["memory_id"].startswith(GOLD_PREFIX)]
    if not gold_idx:
        return ()
    keep = sorted(set(gold_idx) | set(range(min(n_units, len(candidates)))))[:n_units]
    if not any(i in gold_idx for i in keep):  # ensure a gold unit survives the cap
        keep = sorted(set(keep[:-1]) | {gold_idx[0]})
    return tuple(candidates[i]["content"].strip() for i in keep)


def load_rows(slice_path: Path, n_units: int = DEFAULT_N_UNITS) -> list[FormatRow]:
    rows: list[FormatRow] = []
    for line in slice_path.read_text().splitlines():
        record = json.loads(line)
        units = _select_units(record["candidate_memories"], n_units)
        if not units:
            continue
        rows.append(
            FormatRow(
                example_id=record["example_id"],
                query=record["query"],
                reference_answer=record["reference_answer"],
                question_type=record.get("question_type") or "unknown",
                units=units,
            )
        )
    return rows


def render(units: tuple[str, ...], fmt: str) -> str:
    if fmt == "prose":
        return " ".join(units)
    if fmt == "numbered":
        return "\n".join(f"{i}. {u}" for i, u in enumerate(units, 1))
    if fmt == "json":
        return json.dumps({"evidence": list(units)}, indent=2, ensure_ascii=False)
    if fmt == "table":
        head = "| # | Evidence |\n| --- | --- |"
        body = "\n".join(f"| {i} | {u} |" for i, u in enumerate(units, 1))
        return f"{head}\n{body}"
    if fmt == "headers":
        return "\n\n".join(f"### Evidence {i}\n{u}" for i, u in enumerate(units, 1))
    raise ValueError(f"Unknown format: {fmt}")


def build_prompt(row: FormatRow, fmt: str) -> str:
    return f"Evidence:\n{render(row.units, fmt)}\n\nQuestion: {row.query}\n\n{INSTRUCTION}"


def is_correct(row: FormatRow, response: str) -> bool:
    """Lenient: normalized reference answer appears as a token-bounded substring."""
    ref = _normalize(row.reference_answer)
    if not ref:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(ref)}(?![a-z0-9])", _normalize(response)) is not None
