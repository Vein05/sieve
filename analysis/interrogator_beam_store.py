"""True BEAM 128K store loader: turnize parquet chats to match slice memory units."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BEAM_PARQUET = Path(
    "/Users/vein/Documents/research/sieve/data/beam_source/data/100K-00000-of-00001.parquet"
)
MAIN_QUESTION = "main_question"
_NULL_STRS = frozenset({"None", "N/A", ""})

Message = tuple[int, str, str]  # (message_id, role, content)
Turn = tuple[int, str | None, list[Message]]  # (batch_no, time_anchor, messages)


def _is_null(value: object) -> bool:
    return value is None or str(value) in _NULL_STRS


def turnize_chat(chat: list) -> dict[int, Turn]:
    """Group messages into main_question-anchored turns, globally 1-indexed across batches."""
    msgs: list[tuple[int, int, str, str | None, str | None, str]] = []
    for batch_pos, batch in enumerate(chat):
        for m in batch:
            msgs.append(
                (
                    int(m["id"]),
                    batch_pos + 1,
                    m["role"],
                    m.get("question_type"),
                    m.get("time_anchor"),
                    m["content"],
                )
            )
    msgs.sort()
    turns: dict[int, Turn] = {}
    current: int | None = None
    counter = 0
    for mid, batch_no, role, qtype, anchor, content in msgs:
        if role == "user" and qtype == MAIN_QUESTION:
            counter += 1
            current = counter
            turns[counter] = (batch_no, None if _is_null(anchor) else str(anchor), [])
        if current is None:
            continue
        turns[current][2].append((mid, role, content))
    return turns


def render_turn(turn_no: int, batch_no: int, anchor: str | None, messages: list[Message]) -> str:
    """Render a turn exactly as slice candidate_memories content (newlines become spaces)."""
    body = " ".join(f"{role}[{mid}]: {content.replace(chr(10), ' ')}" for mid, role, content in messages)
    prefix = f"{anchor} | " if anchor else ""
    return f"{prefix}batch {batch_no} | turn {turn_no} | {body}"


def build_true_stores(parquet_path: Path = BEAM_PARQUET) -> dict[str, dict[str, str]]:
    """Per-chat true content store: {chat_id: {turn_id: rendered_content}}."""
    frame = pd.read_parquet(parquet_path)
    stores: dict[str, dict[str, str]] = {}
    for _, row in frame.iterrows():
        turns = turnize_chat(list(row["chat"]))
        stores[str(row["conversation_id"])] = {
            f"turn_{n:04d}": render_turn(n, batch_no, anchor, messages)
            for n, (batch_no, anchor, messages) in turns.items()
        }
    return stores
