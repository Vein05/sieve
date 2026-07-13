import os
import json
import argparse
import concurrent.futures
from typing import Dict, Any, List, Tuple

from answer_generation.generation_client import generate_answer


# ---------------------------------------------------------------------------
# Wang et al. (LongMemEval) judge prompts — verbatim from
# xiaowu0162/LongMemEval src/evaluation/evaluate_qa.py
# ---------------------------------------------------------------------------

_LONGMEMEVAL_GENERAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. "
    "If the response only contains a subset of the information required by the answer, answer no.\n\n"
    "Question: {question}\n"
    "Correct Answer: {reference}\n"
    "Model Response: {prediction}\n"
    "Is the model response correct? Answer yes or no only."
)

_LONGMEMEVAL_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. "
    "If the response only contains a subset of the information required by the answer, answer no. "
    "In addition, do not penalize off-by-one errors for the number of days. "
    "If the question asks for the number of days/weeks/months, etc., and the model makes "
    "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response "
    "is still correct.\n\n"
    "Question: {question}\n"
    "Correct Answer: {reference}\n"
    "Model Response: {prediction}\n"
    "Is the model response correct? Answer yes or no only."
)

_LONGMEMEVAL_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. "
    "If the response only contains a subset of the information required by the answer, answer no. "
    "If the response contains some previous information along with an updated answer, "
    "the response should be considered as correct as long as the updated answer is the "
    "required answer.\n\n"
    "Question: {question}\n"
    "Correct Answer: {reference}\n"
    "Model Response: {prediction}\n"
    "Is the model response correct? Answer yes or no only."
)

_LONGMEMEVAL_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a "
    "response from a model. Please answer yes if the response satisfies the desired response. "
    "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
    "The response is correct as long as it recalls and utilizes the user's personal information "
    "correctly.\n\n"
    "Question: {question}\n"
    "Rubric: {reference}\n"
    "Model Response: {prediction}\n"
)

_LONGMEMEVAL_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. "
    "The model could say that the information is incomplete, or some other information is "
    "given but the asked information is not.\n\n"
    "Question: {question}\n"
    "Explanation: {reference}\n"
    "Model Response: {prediction}\n"
    "Does the model correctly identify the question as unanswerable? Answer yes or no only."
)


def build_longmemeval_judge_prompt(
    question: str,
    reference: str,
    prediction: str,
    question_type: str = "",
    is_abstention: bool = False,
) -> str:
    """Select the correct Wang et al. prompt based on question_type."""
    if is_abstention:
        template = _LONGMEMEVAL_ABSTENTION
    elif question_type == "temporal-reasoning":
        template = _LONGMEMEVAL_TEMPORAL
    elif question_type == "knowledge-update":
        template = _LONGMEMEVAL_KNOWLEDGE_UPDATE
    elif question_type == "single-session-preference":
        template = _LONGMEMEVAL_PREFERENCE
    else:
        template = _LONGMEMEVAL_GENERAL
    return template.format(question=question, reference=reference, prediction=prediction)


def parse_longmemeval_judge_response(raw: str) -> int:
    """Extract binary yes/no from judge response. Returns 2 (correct) or 0 (incorrect)."""
    lowered = raw.strip().lower()
    if "yes" in lowered:
        return 2
    return 0


def build_judge_prompt(question: str, reference_answer: str, generated_answer: str) -> str:
    return f"""You are an expert evaluator for short-answer conversational QA.

### TASK:
Compare the GENERATED ANSWER against the REFERENCE ANSWER and decide if it is fully correct, partially correct, or incorrect.

### RULES:
1. Focus on whether the GENERATED ANSWER preserves the core fact(s) of the REFERENCE ANSWER.
2. Ignore harmless stylistic wording differences.
3. Treat safe abstention as correct ONLY if the reference answer itself implies the fact is unavailable or unknown.
4. If the generated answer hallucinates a specific fact that contradicts or goes beyond the reference, score it as incorrect.
5. If the generated answer gives only part of a multi-part answer, score it as partially correct.
6. If the generated answer is vague, incomplete, or misses the key fact, score it as partially correct or incorrect depending on severity.

### SCORING RUBRIC:
- 2 = fully correct
- 1 = partially correct / incomplete but still meaningfully right
- 0 = incorrect / unsupported / wrong abstention

### INPUTS:
Question: {question}
Reference Answer: {reference_answer}
Generated Answer: {generated_answer}

### OUTPUT FORMAT:
Return exactly one JSON object with these keys: "score", "factuality", "completeness", "abstention", "reason".
Scoring: score must be 0, 1, or 2.

### VERDICT (JSON):
"""


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return {
        "score": 0,
        "factuality": "unsupported",
        "completeness": "missing",
        "abstention": "not_applicable",
        "reason": "Failed to parse judge output as JSON.",
    }


def normalize_judge_result(obj: Dict[str, Any]) -> Dict[str, Any]:
    score = obj.get("score", 0)
    try:
        score = int(score)
    except Exception:
        score = 0
    if score not in {0, 1, 2}:
        score = 0

    factuality = str(obj.get("factuality", "unsupported")).strip().lower()
    if factuality not in {"supported", "partially_supported", "unsupported"}:
        factuality = "unsupported"

    completeness = str(obj.get("completeness", "missing")).strip().lower()
    if completeness not in {"complete", "partial", "missing"}:
        completeness = "missing"

    abstention = str(obj.get("abstention", "not_applicable")).strip().lower()
    if abstention not in {"correct", "incorrect", "not_applicable"}:
        abstention = "not_applicable"

    reason = str(obj.get("reason", "")).strip()
    if not reason:
        reason = "No reason provided."

    return {
        "score": score,
        "factuality": factuality,
        "completeness": completeness,
        "abstention": abstention,
        "reason": reason,
    }


def grade_row(pinfo: dict, provider: str, model: str, judge_mode: str = "generic") -> dict:
    if judge_mode == "longmemeval":
        question_type = pinfo.get("question_type", "")
        is_abstention = "_abs" in str(pinfo.get("example_id", "")) or str(pinfo.get("harm_type", "")).lower() == "abstention"
        prompt = build_longmemeval_judge_prompt(
            pinfo["question"],
            pinfo["ref"],
            pinfo["gen"],
            question_type=question_type,
            is_abstention=is_abstention,
        )
        max_tokens = 10
    else:
        prompt = build_judge_prompt(
            pinfo["question"],
            pinfo["ref"],
            pinfo["gen"],
        )
        max_tokens = 150

    base_url = "https://openrouter.ai/api/v1" if provider == "openrouter" else "http://127.0.0.1:11434"

    res = generate_answer(
        provider=provider,
        base_url=base_url,
        model=model,
        prompt=prompt,
        temperature=0.0,
        max_tokens=max_tokens,
    )

    raw = res.get("raw_answer", "") or ""

    if judge_mode == "longmemeval":
        score = parse_longmemeval_judge_response(raw)
        pinfo["judge_score"] = score
        pinfo["judge_factuality"] = "supported" if score == 2 else "unsupported"
        pinfo["judge_completeness"] = "complete" if score == 2 else "missing"
        pinfo["judge_abstention"] = "not_applicable"
        pinfo["judge_reason"] = raw.strip()
    else:
        parsed = safe_json_loads(raw)
        norm = normalize_judge_result(parsed)
        pinfo["judge_score"] = norm["score"]
        pinfo["judge_factuality"] = norm["factuality"]
        pinfo["judge_completeness"] = norm["completeness"]
        pinfo["judge_abstention"] = norm["abstention"]
        pinfo["judge_reason"] = norm["reason"]

    pinfo["judge_raw"] = raw
    return pinfo


def grade_all(payloads: List[dict], provider: str, model: str, parallelism: int, judge_mode: str = "generic") -> List[dict]:
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = {executor.submit(grade_row, p, provider, model, judge_mode): p for p in payloads}
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            done_count += 1
            if done_count % 100 == 0:
                print(f"Judged {done_count}/{len(futures)}...")
    return results


def pairwise_against_naive(system_scores: Dict[str, int], naive_scores: Dict[str, int]) -> Tuple[float, float]:
    wins = 0.0
    constrained = 0.0
    total = 0

    for eid, score in system_scores.items():
        if eid not in naive_scores:
            continue
        total += 1
        n_score = naive_scores[eid]

        if score > n_score:
            wins += 1.0
            constrained += 1.0
        elif score == n_score:
            wins += 0.5
            constrained += 0.5

    if total == 0:
        return 0.5, 0.5
    return wins / total, constrained / total


def summarize_system(outputs: List[dict]) -> Dict[str, Any]:
    n = len(outputs)
    if n == 0:
        return {
            "avg_quality": 0.0,
            "exact_rate": 0.0,
            "partial_or_better_rate": 0.0,
        }

    scores = [int(o.get("judge_score_v2", 0)) for o in outputs]
    exact = sum(1 for s in scores if s == 2)
    partial_or_better = sum(1 for s in scores if s >= 1)

    return {
        "avg_quality": sum(scores) / n,
        "exact_rate": exact / n,
        "partial_or_better_rate": partial_or_better / n,
    }


def _judge_summary_markdown(
    *,
    systems: Dict[str, List[dict]],
    system_score_maps: Dict[str, Dict[str, int]],
    model: str,
    provider: str,
    judge_mode: str = "generic",
) -> str:
    naive_scores = system_score_maps.get("naive_top_k", {})

    md_lines = [
        "",
        "## LLM Judge Summary (v2)",
        "",
        f"- **Judge Model**: `{model}` via `{provider}`",
        f"- **Rubric**: {'0 = incorrect, 2 = correct (binary, Wang et al.)' if judge_mode == 'longmemeval' else '0 = incorrect, 1 = partially correct, 2 = fully correct'}",
        "",
        "| system | avg_quality | exact_rate | partial_or_better | pairwise_vs_naive | constrained_pairwise_vs_naive |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for sys_name, sys_outputs in systems.items():
        summary = summarize_system(sys_outputs)
        score_map = system_score_maps[sys_name]
        pairwise, constrained = pairwise_against_naive(score_map, naive_scores)

        md_lines.append(
            f"| `{sys_name}` | "
            f"{summary['avg_quality']:.3f} | "
            f"{summary['exact_rate']:.3f} | "
            f"{summary['partial_or_better_rate']:.3f} | "
            f"{pairwise:.3f} | "
            f"{constrained:.3f} |"
        )

    md_lines.extend([
        "",
        "### By Category",
        "",
        "| system | category | avg_quality | exact_rate | partial_or_better | pairwise_vs_naive |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])

    for sys_name, sys_outputs in systems.items():
        by_cat: Dict[str, List[dict]] = {}
        by_cat_scores: Dict[str, Dict[str, int]] = {}

        for o in sys_outputs:
            eid = o["example_id"]
            cat = eid.split("-")[0] if "-" in eid else "unknown"
            by_cat.setdefault(cat, []).append(o)
            by_cat_scores.setdefault(cat, {})[eid] = int(o["judge_score_v2"])

        for cat in sorted(by_cat.keys()):
            summary = summarize_system(by_cat[cat])
            score_map = by_cat_scores[cat]
            naive_cat_scores = {eid: naive_scores[eid] for eid in score_map if eid in naive_scores}
            pairwise, _ = pairwise_against_naive(score_map, naive_cat_scores)
            md_lines.append(
                f"| `{sys_name}` | {cat} | "
                f"{summary['avg_quality']:.3f} | "
                f"{summary['exact_rate']:.3f} | "
                f"{summary['partial_or_better_rate']:.3f} | "
                f"{pairwise:.3f} |"
            )

    return "\n".join(md_lines) + "\n"


def _detect_judge_mode(outputs: List[dict]) -> str:
    """Auto-detect judge mode from dataset contents."""
    for o in outputs:
        dn = str(o.get("dataset_name", "") or "").lower()
        if "longmemeval" in dn:
            return "longmemeval"
        eid = str(o.get("example_id", "") or "")
        if eid.startswith("lmf"):
            return "longmemeval"
    return "generic"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to directory containing run.json and summary.md")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model", default="deepseek/deepseek-chat-v3-0324")
    parser.add_argument("--parallelism", type=int, default=100)
    parser.add_argument(
        "--judge-mode",
        choices=["auto", "longmemeval", "generic"],
        default="auto",
        help="Judge prompt style: 'longmemeval' uses Wang et al. binary prompts, 'generic' uses 3-tier rubric, 'auto' detects from dataset",
    )
    args = parser.parse_args()

    run_path = os.path.join(args.run_dir, "run.json")
    summary_path = os.path.join(args.run_dir, "summary.md")
    judge_out_path = os.path.join(args.run_dir, "judge_run_v2.json")

    with open(run_path, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    outputs = run_data.get("outputs", [])
    print(f"Total outputs to judge: {len(outputs)}")

    judge_mode = args.judge_mode
    if judge_mode == "auto":
        judge_mode = _detect_judge_mode(outputs)
    print(f"Judge mode: {judge_mode}")

    unique_payloads: Dict[Tuple[str, str, str], dict] = {}
    payloads: List[dict] = []

    for o in outputs:
        key = (o["question"], o["reference_answer"], o["generated_answer"])
        if key not in unique_payloads:
            pinfo = {
                "question": o["question"],
                "ref": o["reference_answer"],
                "gen": o["generated_answer"],
                "example_id": o.get("example_id", ""),
                "question_type": o.get("question_type", ""),
                "harm_type": o.get("harm_type", ""),
            }
            unique_payloads[key] = pinfo
            payloads.append(pinfo)

    print(f"Unique prompts to judge: {len(payloads)}")

    results = grade_all(payloads, args.provider, args.model, args.parallelism, judge_mode=judge_mode)

    judged_map: Dict[Tuple[str, str, str], dict] = {}
    for r in results:
        key = (r["question"], r["ref"], r["gen"])
        judged_map[key] = r

    systems: Dict[str, List[dict]] = {}
    system_score_maps: Dict[str, Dict[str, int]] = {}

    for o in outputs:
        key = (o["question"], o["reference_answer"], o["generated_answer"])
        judged = judged_map[key]

        o["judge_score_v2"] = judged["judge_score"]
        o["judge_factuality_v2"] = judged["judge_factuality"]
        o["judge_completeness_v2"] = judged["judge_completeness"]
        o["judge_abstention_v2"] = judged["judge_abstention"]
        o["judge_reason_v2"] = judged["judge_reason"]
        o["judge_raw_v2"] = judged["judge_raw"]

        sys_name = o["system"]
        systems.setdefault(sys_name, []).append(o)
        system_score_maps.setdefault(sys_name, {})[o["example_id"]] = judged["judge_score"]

    run_data["judge_model_v2"] = args.model
    with open(judge_out_path, "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=2)

    judge_md = _judge_summary_markdown(
        systems=systems,
        system_score_maps=system_score_maps,
        model=args.model,
        provider=args.provider,
        judge_mode=judge_mode,
    )

    summary_text = ""
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_text = f.read()

    marker = "\n## LLM Judge Summary (v2)\n"
    if marker in summary_text:
        summary_text = summary_text.split(marker, 1)[0].rstrip() + "\n"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text.rstrip() + "\n")
        f.write(judge_md)

    print(f"Saved judged run to {judge_out_path}")
    print(f"Appended v2 judge summary to {summary_path}")


if __name__ == "__main__":
    main()
