"""Chain-rendering pilot v0: chain vs resolve-to-latest vs raw chronological.

24 gold-complete BEAM rows (12 knowledge_update, 12 contradiction_resolution).
All conditions consume the SAME chronologically sorted 24K-capped pool; only
the rendering differs. No judge; outputs dumped for manual scoring.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/vein/Documents/research/sieve")
from reader.client import generate_answer  # noqa: E402

SLICE = Path(
    "/Users/vein/Documents/research/memory-eligibility-feasibility"
    "/dataset-slices/beam_transfer_bm25_top20_v0.jsonl"
)
OUT_DIR = Path(__file__).parent / "chain_pilot_out"
ABILITIES = ("knowledge_update", "contradiction_resolution")
N_PER_ABILITY = 12
EVIDENCE_TOKEN_CAP = 24_000
CHARS_PER_TOKEN = 4
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
COMPILER_MODEL = "deepseek/deepseek-chat-v3-0324"
READERS = ("meta-llama/llama-3.1-8b-instruct", "qwen/qwen-2.5-72b-instruct")
CONDITIONS = ("raw_chrono", "resolve_latest", "chain_render")
COMPILE_MAX_TOKENS = 1024
READER_MAX_TOKENS = 512
PARALLELISM = 16
TIMEOUT_S = 180.0

COMPILE_SHARED = """You are compiling evidence from a long-running conversation so that another
model can answer a question. Below are excerpts (turns) from the conversation
in chronological order, followed by the question.

{instruction}

Rules:
- Do NOT answer the question. Output only the fact sheet.
- Cite the turn number for every fact, e.g. (turn 52).
- Keep only facts plausibly relevant to the question.

Conversation excerpts:
{evidence}

Question (do not answer it): {query}

Fact sheet:"""

RESOLVE_INSTRUCTION = (
    "Extract the facts relevant to the question. Where a fact was updated, "
    "revised, or restated over time, keep ONLY the most recent version and "
    "discard all earlier versions."
)

CHAIN_INSTRUCTION = (
    "Extract the facts relevant to the question. Where a fact was updated, "
    "revised, or restated over time, render the FULL update chain in "
    "chronological order on one line, e.g. "
    "\"response time: 800ms (turn 12) -> 300ms (turn 48) -> 250ms (turn 52) [most recent]\". "
    "Never discard an earlier version of a fact. If two statements conflict "
    "and neither is clearly a later update, mark the pair with [CONTRADICTION] "
    "and keep both."
)

READER_PROMPT = """You are answering a question about a long-running conversation with a user,
based on evidence extracted from that conversation.

Question: {query}

Evidence:
{evidence}

Answer the question using the evidence. Facts may have changed over the
conversation; pay attention to WHICH version of a fact the question asks
about (current, earlier, or the change itself). If the evidence contains a
genuine contradiction relevant to the question, point it out explicitly.
Answer in complete sentences; a short narrative is fine.

Answer:"""


def _turn_no(memory_id: str) -> int:
    m = re.match(r"turn_(\d+)", memory_id)
    return int(m.group(1)) if m else 10**9


def load_pilot_rows() -> list[dict]:
    rows = [json.loads(line) for line in SLICE.open()]
    picked: list[dict] = []
    for ability in ABILITIES:
        sub = sorted((r for r in rows if r["ability"] == ability), key=lambda r: r["example_id"])
        complete = [
            r
            for r in sub
            if r["evidence_sufficiency"].get("answer_bearing_memory_ids")
            and set(r["evidence_sufficiency"]["answer_bearing_memory_ids"])
            <= {c["memory_id"] for c in r["candidate_memories"]}
        ]
        picked.extend(complete[:N_PER_ABILITY])
    return picked


def base_evidence(row: dict) -> str:
    pool = sorted(row["candidate_memories"], key=lambda c: _turn_no(c["memory_id"]))
    parts: list[str] = []
    budget = EVIDENCE_TOKEN_CAP
    for cand in pool:
        text = cand["content"]
        cost = len(text) // CHARS_PER_TOKEN
        if budget <= 0:
            break
        if cost > budget:
            text = text[: budget * CHARS_PER_TOKEN]
            cost = budget
        parts.append(text)
        budget -= cost
    return "\n".join(parts)


def _call(model: str, prompt: str, max_tokens: int) -> dict:
    # raw_answer, NOT clean_answer: the repo's cleaner extracts a single span
    # line and destroys narrative/fact-sheet outputs (the known artifact class).
    last: dict = {}
    for attempt in range(3):
        if attempt:
            time.sleep(8 * attempt)
        last = generate_answer(
            provider="openrouter",
            base_url=OPENROUTER_BASE_URL,
            model=model,
            prompt=prompt,
            temperature=0.0,
            timeout_s=TIMEOUT_S,
            max_tokens=max_tokens,
            provider_routing={"allow_fallbacks": False},
        )
        if last.get("success"):
            break
    return last


def compile_evidence(row: dict, condition: str, base: str) -> dict:
    instruction = RESOLVE_INSTRUCTION if condition == "resolve_latest" else CHAIN_INSTRUCTION
    prompt = COMPILE_SHARED.format(instruction=instruction, evidence=base, query=row["query"])
    gen = _call(COMPILER_MODEL, prompt, COMPILE_MAX_TOKENS)
    return {
        "example_id": row["example_id"],
        "condition": condition,
        "text": gen.get("raw_answer") or "",
        "success": bool(gen.get("success")),
        "error": gen.get("error"),
    }


def read_answer(row: dict, condition: str, evidence: str, reader: str) -> dict:
    prompt = READER_PROMPT.format(query=row["query"], evidence=evidence)
    gen = _call(reader, prompt, READER_MAX_TOKENS)
    return {
        "example_id": row["example_id"],
        "ability": row["ability"],
        "condition": condition,
        "reader": reader,
        "question": row["query"],
        "reference_answer": row["reference_answer"],
        "answer": gen.get("raw_answer") or "",
        "evidence_tokens": len(evidence) // CHARS_PER_TOKEN,
        "success": bool(gen.get("success")),
        "error": gen.get("error"),
    }


def run() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rows = load_pilot_rows()
    print(f"pilot rows: {len(rows)}")
    bases = {r["example_id"]: base_evidence(r) for r in rows}

    compile_jobs = [(r, cond) for r in rows for cond in ("resolve_latest", "chain_render")]
    compiled: dict[tuple[str, str], dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLELISM) as pool:
        futs = {
            pool.submit(compile_evidence, r, cond, bases[r["example_id"]]): (r["example_id"], cond)
            for r, cond in compile_jobs
        }
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            compiled[(res["example_id"], res["condition"])] = res
    n_bad = sum(1 for v in compiled.values() if not v["success"] or not v["text"].strip())
    print(f"compiled {len(compiled)} ({n_bad} failed/empty)")
    (OUT_DIR / "compiled.json").write_text(
        json.dumps({f"{k[0]}::{k[1]}": v for k, v in compiled.items()}, indent=1)
    )

    def evidence_for(row: dict, condition: str) -> str:
        if condition == "raw_chrono":
            return bases[row["example_id"]]
        return compiled[(row["example_id"], condition)]["text"]

    reader_jobs = [
        (r, cond, reader) for r in rows for cond in CONDITIONS for reader in READERS
    ]
    outputs: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLELISM) as pool:
        futs = [
            pool.submit(read_answer, r, cond, evidence_for(r, cond), reader)
            for r, cond, reader in reader_jobs
        ]
        for i, fut in enumerate(concurrent.futures.as_completed(futs)):
            outputs.append(fut.result())
            if (i + 1) % 40 == 0:
                print(f"  readers {i + 1}/{len(futs)}")
    outputs.sort(key=lambda o: (o["ability"], o["example_id"], o["condition"], o["reader"]))
    n_fail = sum(1 for o in outputs if not o["success"])
    print(f"reader outputs: {len(outputs)} ({n_fail} failed)")
    (OUT_DIR / "answers.json").write_text(json.dumps(outputs, indent=1))

    lines = ["# Chain pilot v0 — manual scoring sheet\n"]
    for r in rows:
        eid = r["example_id"]
        lines.append(f"\n## {eid} [{r['ability']}]")
        lines.append(f"**Q:** {r['query']}")
        lines.append(f"**GOLD:** {r['reference_answer']}\n")
        for cond in CONDITIONS:
            for reader in READERS:
                ans = next(
                    (o for o in outputs if o["example_id"] == eid and o["condition"] == cond and o["reader"] == reader),
                    None,
                )
                short = reader.split("/")[-1]
                text = (ans["answer"] if ans else "<missing>").strip().replace("\n", " ")
                lines.append(f"- `{cond}` / `{short}`: {text}")
    (OUT_DIR / "review.md").write_text("\n".join(lines))
    print(f"wrote {OUT_DIR}/review.md")


if __name__ == "__main__":
    run()
