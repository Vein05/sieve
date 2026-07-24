"""Oracle-chain arms: perfectly bound version chains from BEAM's own labels.

oracle_chain  = all labeled versions of the queried fact, verbatim, chronological.
oracle_latest = only the latest labeled version (perfect-binding resolve).
No LLM compiler anywhere; same reader prompt as the chain pilot.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/vein/Documents/research/sieve")
from analysis.interrogator_beam_store import build_true_stores  # noqa: E402
from reader.client import generate_answer  # noqa: E402

SLICE = Path(
    "/Users/vein/Documents/research/memory-eligibility-feasibility"
    "/dataset-slices/beam_transfer_bm25_top20_v0.jsonl"
)
PILOT_ANSWERS = Path(
    "/Users/vein/Documents/research/sieve/results/v2_runs/chain_render_pilot_v0/answers.json"
)
OUT_DIR = Path(__file__).parent / "oracle_chain_out"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
READERS = ("meta-llama/llama-3.1-8b-instruct", "qwen/qwen-2.5-72b-instruct")
CONDITIONS = ("oracle_chain", "oracle_latest")
READER_MAX_TOKENS = 512
PARALLELISM = 16
TIMEOUT_S = 180.0

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


def version_ids(row: dict) -> list[str]:
    pp = row.get("perturbation_provenance") or {}
    ids = list(pp.get("stale_memory_ids") or []) + list(pp.get("newer_memory_ids") or [])
    if not ids:
        ids = list(row["evidence_sufficiency"].get("answer_bearing_memory_ids") or [])
    return sorted(set(ids), key=_turn_no)


def turn_text(row: dict, mid: str, stores: dict) -> str:
    for c in row["candidate_memories"]:
        if c["memory_id"] == mid:
            return c["content"]
    return stores[str(row["chat_id"])].get(mid, "")


def render(row: dict, condition: str, stores: dict) -> str:
    ids = version_ids(row)
    if condition == "oracle_latest":
        ids = ids[-1:]
    parts = ["Statements from the conversation about the queried fact, in chronological order:"]
    for i, mid in enumerate(ids, 1):
        parts.append(f"[{i}] (turn {_turn_no(mid)}) {turn_text(row, mid, stores)}")
    return "\n\n".join(parts)


def _call(model: str, prompt: str) -> dict:
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
            max_tokens=READER_MAX_TOKENS,
            provider_routing={"allow_fallbacks": False},
        )
        if last.get("success"):
            break
    return last


def run() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    pilot_ids = sorted({o["example_id"] for o in json.load(PILOT_ANSWERS.open())})
    rows = [r for r in (json.loads(l) for l in SLICE.open()) if r["example_id"] in pilot_ids]
    print(f"rows: {len(rows)}")
    stores = build_true_stores()

    jobs = []
    for r in rows:
        for cond in CONDITIONS:
            evidence = render(r, cond, stores)
            for reader in READERS:
                jobs.append((r, cond, reader, evidence))

    def one(job):
        r, cond, reader, evidence = job
        gen = _call(reader, READER_PROMPT.format(query=r["query"], evidence=evidence))
        return {
            "example_id": r["example_id"],
            "ability": r["ability"],
            "condition": cond,
            "reader": reader,
            "question": r["query"],
            "reference_answer": r["reference_answer"],
            "n_versions": len(version_ids(r)) if cond == "oracle_chain" else 1,
            "answer": gen.get("raw_answer") or "",
            "success": bool(gen.get("success")),
        }

    outputs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLELISM) as pool:
        for res in pool.map(one, jobs):
            outputs.append(res)
    outputs.sort(key=lambda o: (o["ability"], o["example_id"], o["condition"], o["reader"]))
    n_fail = sum(1 for o in outputs if not o["success"])
    print(f"outputs: {len(outputs)} ({n_fail} failed)")
    (OUT_DIR / "answers.json").write_text(json.dumps(outputs, indent=1))

    lines = ["# Oracle-chain arms — manual scoring sheet\n"]
    for eid in sorted({o["example_id"] for o in outputs}):
        sub = [o for o in outputs if o["example_id"] == eid]
        lines.append(f"\n## {eid} [{sub[0]['ability']}]")
        lines.append(f"**Q:** {sub[0]['question']}")
        lines.append(f"**GOLD:** {sub[0]['reference_answer']}\n")
        for o in sub:
            short = o["reader"].split("/")[-1]
            lines.append(f"- `{o['condition']}` / `{short}`: {' '.join(o['answer'].split())}")
    (OUT_DIR / "review.md").write_text("\n".join(lines))
    print(f"wrote {OUT_DIR}/review.md")


if __name__ == "__main__":
    run()
