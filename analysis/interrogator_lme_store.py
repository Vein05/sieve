"""LongMemEval-S haystack loader: message-level units, honest BM25 pools (no gold injection)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

LME_HAYSTACK = Path(
    "/Users/vein/Documents/research/sieve/data/longmemeval_source/longmemeval_s"
)
DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"
POOL_SIZE = 20


@dataclass(frozen=True)
class Unit:
    """One haystack message rendered as a retrieval unit."""

    uid: str
    session_pos: int
    session_id: str
    text: str


def load_questions(path: Path = LME_HAYSTACK) -> list[dict]:
    """Load the official LongMemEval-S question list."""
    with path.open() as fh:
        return json.load(fh)


def build_units(question: dict) -> list[Unit]:
    """One unit per non-empty message, rendered as '{date} | {role}: {content}'."""
    units: list[Unit] = []
    sessions = zip(
        question["haystack_session_ids"],
        question["haystack_dates"],
        question["haystack_sessions"],
    )
    for pos, (sid, date, session) in enumerate(sessions):
        for msg_pos, message in enumerate(session):
            content = message.get("content") or ""
            if not content.strip():
                continue
            units.append(
                Unit(
                    uid=f"s{pos:03d}_m{msg_pos:03d}",
                    session_pos=pos,
                    session_id=str(sid),
                    text=f"{date} | {message.get('role', '')}: {content}",
                )
            )
    return units


def _parse_date(raw: str) -> datetime:
    try:
        return datetime.strptime(raw.strip(), DATE_FORMAT)
    except ValueError:
        return datetime.min


def session_date_ranks(question: dict) -> dict[int, int]:
    """Map haystack session position to chronological rank (date, then position)."""
    dates = question["haystack_dates"]
    order = sorted(range(len(dates)), key=lambda pos: (_parse_date(dates[pos]), pos))
    return {pos: rank for rank, pos in enumerate(order)}
