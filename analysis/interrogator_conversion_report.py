"""Score the interrogator v0 conversion runs against the pre-registered read."""

from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

from analysis.interrogator_conversion import CONDITIONS

RUN_ROOT = Path("results/answer_generation_runs/interrogator_v0_conversion")
SMALL = "meta-llama/llama-3.1-8b-instruct"
STRONG = "qwen/qwen-2.5-72b-instruct"
READERS = (SMALL, STRONG)
LIVE_MARGIN_PP = 3.0
DO_NO_HARM_PP = 1.0
CEILING_FLOOR_PP = 5.0
# One dense Qwen row exceeds the served 32K context in one condition; drop it from
# the Qwen cohort in ALL conditions so the paired comparison stays balanced.
EXCLUDE_QWEN_EIDS = frozenset({"beam-128k-5-temporal_reasoning-001"})

# OpenRouter $/token (fetched 2026-07-14).
PRICE = {
    SMALL: (0.02e-6, 0.03e-6),
    STRONG: (0.36e-6, 0.40e-6),
}


def _load(condition: str, reader: str) -> list[dict]:
    slug = f"{condition}__{reader.replace('/', '__')}"
    path = RUN_ROOT / slug / "judge_run_v2.json"
    return json.loads(path.read_text())["outputs"]


def _eligible(output: dict, reader: str) -> bool:
    if reader == STRONG and output["example_id"] in EXCLUDE_QWEN_EIDS:
        return False
    return True


def _acc(outputs: list[dict]) -> float:
    if not outputs:
        return 0.0
    return 100.0 * sum(1 for o in outputs if int(o.get("judge_score_v2", 0)) == 2) / len(outputs)


def _split(outputs: list[dict]) -> tuple[list[dict], list[dict]]:
    mg = [o for o in outputs if o.get("is_missing_gold")]
    dnh = [o for o in outputs if not o.get("is_missing_gold")]
    return mg, dnh


def condition_table(reader: str) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        outs = [o for o in _load(condition, reader) if _eligible(o, reader)]
        mg, dnh = _split(outs)
        table[condition] = {
            "n_mg": len(mg),
            "n_dnh": len(dnh),
            "mg_acc": _acc(mg),
            "dnh_acc": _acc(dnh),
            "all_acc": _acc(outs),
        }
    return table


def per_ability(reader: str, condition: str) -> dict[str, tuple[int, float]]:
    outs = [o for o in _load(condition, reader) if _eligible(o, reader) and o.get("is_missing_gold")]
    by: dict[str, list[dict]] = defaultdict(list)
    for o in outs:
        by[o["ability"]].append(o)
    return {ab: (len(v), _acc(v)) for ab, v in sorted(by.items())}


def cost_accounting() -> dict[str, dict[str, dict[str, float]]]:
    """Per (reader, condition): tokens retrieved/sent, $/query."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for reader in READERS:
        p_in, p_out = PRICE[reader]
        out[reader] = {}
        for condition in CONDITIONS:
            outs = [o for o in _load(condition, reader) if _eligible(o, reader) and o["success"]]
            n = max(len(outs), 1)
            evid = sum(o["evidence_token_count"] for o in outs) / n
            pin = sum((o["prompt_tokens"] or 0) for o in outs) / n
            pout = sum((o["completion_tokens"] or 0) for o in outs) / n
            acq = sum(o.get("n_acquired", 0) for o in outs) / n
            out[reader][condition] = {
                "evidence_tokens": evid,
                "prompt_tokens": pin,
                "completion_tokens": pout,
                "units_acquired": acq,
                "usd_per_query": pin * p_in + pout * p_out,
            }
    return out


def total_spend() -> float:
    total = 0.0
    p_in_ds, p_out_ds = 0.24e-6, 0.90e-6
    for path in glob.glob(str(RUN_ROOT / "*" / "run.json")):
        outs = json.loads(Path(path).read_text())["outputs"]
        reader = outs[0]["reader_model"]
        p_in, p_out = PRICE[reader]
        for o in outs:
            total += (o["prompt_tokens"] or 0) * p_in + (o["completion_tokens"] or 0) * p_out
    # judge: unique prompts per dir, ~ (question+ref+gen) in, up to 150 out
    for path in glob.glob(str(RUN_ROOT / "*" / "judge_run_v2.json")):
        outs = json.loads(Path(path).read_text())["outputs"]
        seen = set()
        for o in outs:
            key = (o["question"], o["reference_answer"], o["generated_answer"])
            if key in seen:
                continue
            seen.add(key)
            approx_in = len(o["question"] + o["reference_answer"] + o["generated_answer"]) // 4 + 250
            total += approx_in * p_in_ds + 120 * p_out_ds
    return total


def verdict(reader: str, table: dict[str, dict[str, float]]) -> str:
    mg = {c: table[c]["mg_acc"] for c in CONDITIONS}
    dnh = {c: table[c]["dnh_acc"] for c in CONDITIONS}
    beats_k = mg["interrogator"] - mg["adaptive_k"]
    harm = dnh["adaptive_k"] - dnh["interrogator"]  # damage vs the aperture baseline's cohort
    harm_vs_fixed = dnh["fixed"] - dnh["interrogator"]
    ceiling = mg["ceiling"] - mg["fixed"]
    lives = beats_k >= LIVE_MARGIN_PP and harm_vs_fixed <= DO_NO_HARM_PP
    return (
        f"interrogator-vs-adaptive_k(mg): {beats_k:+.1f}pp | "
        f"do-no-harm dmg vs fixed: {harm_vs_fixed:+.1f}pp | "
        f"ceiling(mg): {ceiling:+.1f}pp | "
        f"VERDICT: {'LIVES' if lives else 'DOES NOT CONVERT (kill)'}"
    )


def main() -> None:
    for reader in READERS:
        table = condition_table(reader)
        print(f"\n===== {reader} =====")
        print(f"{'condition':13s} {'mg_acc':>7s} {'dnh_acc':>8s} {'all_acc':>8s}  (n_mg / n_dnh)")
        for c in CONDITIONS:
            t = table[c]
            print(f"{c:13s} {t['mg_acc']:7.1f} {t['dnh_acc']:8.1f} {t['all_acc']:8.1f}   ({t['n_mg']}/{t['n_dnh']})")
        print(verdict(reader, table))
        print("per-ability (missing-gold cohort):")
        for c in CONDITIONS:
            print(f"  {c}: ", {ab: f"{acc:.0f}%(n{n})" for ab, (n, acc) in per_ability(reader, c).items()})
    print("\n===== cost accounting (per query) =====")
    ca = cost_accounting()
    for reader in READERS:
        print(f"-- {reader}")
        for c in CONDITIONS:
            d = ca[reader][c]
            print(
                f"  {c:13s} evid_tok~{d['evidence_tokens']:.0f} prompt_tok~{d['prompt_tokens']:.0f} "
                f"units_acq~{d['units_acquired']:.0f} ${d['usd_per_query']:.5f}/query"
            )
    print(f"\nTOTAL SPEND (readers + judge, est): ${total_spend():.2f}")


if __name__ == "__main__":
    main()
