"""Scoring helpers for the answer-generation phase."""

from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from functools import lru_cache
import random
from typing import Any

import numpy as np

import assembly_methods.common as am_common


_stem_token = am_common.stem_token

from answer_generation.pali_shims import (
    bleu1,
    normalize_answer_for_scoring,
    normalize_tokens,
    normalized_exact_match,
    token_f1,
)


# Scoring thresholds for correctness classification.
_CORRECT_F1_THRESHOLD = 0.70
_CORRECT_BLEU_THRESHOLD = 0.70
_PARTIAL_F1_THRESHOLD = 0.25
_PARTIAL_BLEU_THRESHOLD = 0.35
_SPAN_MATCH_F1_THRESHOLD = 0.45
_HIGH_F1_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Dataset detection
# ---------------------------------------------------------------------------

def _detect_dataset(row: dict[str, Any]) -> str:
    """Return 'longmemeval' | 'locomo' | 'beam' | 'default'."""
    dn = str(row.get("dataset_name", "") or "").strip().lower()
    if dn == "beam":
        return "beam"
    if "longmemeval" in dn:
        return "longmemeval"
    eid = str(row.get("example_id", "") or "")
    if eid.startswith("locfpc") or eid.startswith("locfrd"):
        return "locomo"
    return "default"


# ---------------------------------------------------------------------------
# LoCoMo paper-aligned F1 (Snap Research, with Porter stemming)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _porter_stemmer():
    from nltk.stem.porter import PorterStemmer
    return PorterStemmer()


_LOCOMO_ARTICLES = re.compile(r"\b(a|an|the|and)\b", re.IGNORECASE)
_LOCOMO_PUNCTUATION = set(string.punctuation)


def _locomo_normalize(text: str) -> str:
    """LoCoMo paper normalization: lowercase, remove punctuation, remove articles+and, whitespace fix."""
    text = text.replace(",", "")
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _LOCOMO_PUNCTUATION)
    text = _LOCOMO_ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _locomo_stemmed_tokens(text: str) -> list[str]:
    ps = _porter_stemmer()
    return [ps.stem(w) for w in _locomo_normalize(text).split()]


def _locomo_f1_single(prediction: str, ground_truth: str) -> float:
    """Single-pair stemmed F1, matching LoCoMo paper exactly."""
    pred_tokens = _locomo_stemmed_tokens(prediction)
    ref_tokens = _locomo_stemmed_tokens(ground_truth)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def locomo_paper_f1(prediction: str, reference: str, category: int) -> float:
    """Compute LoCoMo paper-aligned F1 with category-specific logic.

    Categories: 1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop, 5=adversarial.
    """
    if category == 5:
        # Adversarial: binary keyword check
        lowered = prediction.lower()
        if "no information available" in lowered or "not mentioned" in lowered:
            return 1.0
        return 0.0

    if category == 3:
        # Open-domain: only first answer before ';'
        reference = reference.split(";")[0].strip()

    if category == 1:
        # Multi-hop: comma-split, max-F1-per-ground-truth, mean
        predictions = [p.strip() for p in prediction.split(",")]
        ground_truths = [g.strip() for g in reference.split(",")]
        if not ground_truths:
            return 0.0
        return float(np.mean([
            max([_locomo_f1_single(p, gt) for p in predictions]) if predictions else 0.0
            for gt in ground_truths
        ]))

    # Cat 2, 4 (and fallback): standard stemmed F1
    return _locomo_f1_single(prediction, reference)


def quality_metrics_locomo(
    question: str, answer: str, reference_answer: str, category: int,
) -> dict[str, Any]:
    """LoCoMo paper-aligned scoring. Primary metric = stemmed F1."""
    f1 = locomo_paper_f1(answer, reference_answer, category)

    if f1 >= _HIGH_F1_THRESHOLD:
        label = "correct"
        score = 2
    elif f1 > 0.0:
        label = "partially_correct"
        score = 1
    else:
        label = "incorrect"
        score = 0

    # Also compute our standard metrics for backwards compat
    normalized_pred = normalize_answer_for_scoring(answer, question)
    normalized_ref = normalize_answer_for_scoring(reference_answer, question)

    return {
        "normalized_answer": normalized_pred,
        "normalized_reference_answer": normalized_ref,
        "exact_match": normalized_exact_match(normalized_pred, normalized_ref),
        "token_f1": round(f1, 3),
        "token_f1_paper": round(f1, 3),
        "bleu1": round(bleu1(normalized_pred, normalized_ref), 3),
        "quality_label": label,
        "quality_score": score,
        "reference_alignment_note": f"locomo_cat{category}_paper_f1",
    }


# ---------------------------------------------------------------------------
# LongMemEval heuristic scoring (binary: correct or incorrect, no partial)
# ---------------------------------------------------------------------------

def quality_metrics_longmemeval(
    question: str, answer: str, reference_answer: str,
) -> dict[str, Any]:
    """LongMemEval-aligned heuristic scoring. Binary output only (0 or 2)."""
    base = quality_metrics(question, answer, reference_answer)
    # Collapse partial to incorrect — paper reports binary accuracy
    if base["quality_score"] == 1:
        base["quality_label"] = "incorrect"
        base["quality_score"] = 0
        base["reference_alignment_note"] = base.get("reference_alignment_note", "") + " (partial→incorrect for binary)"
    return base


# ---------------------------------------------------------------------------
# Dataset-aware router
# ---------------------------------------------------------------------------

def quality_metrics_for_row(
    row: dict[str, Any], question: str, answer: str, reference_answer: str,
) -> dict[str, Any]:
    """Route to dataset-appropriate scoring."""
    ds = _detect_dataset(row)
    if ds == "locomo":
        cat = int(row.get("locomo_category", 4))
        return quality_metrics_locomo(question, answer, reference_answer, category=cat)
    elif ds == "longmemeval":
        return quality_metrics_longmemeval(question, answer, reference_answer)
    else:
        return quality_metrics(question, answer, reference_answer)

MONTH_NAME_TO_NUMBER = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

HOLIDAY_MONTH_DAY = {
    "valentine's day": (2, 14),
    "valentines day": (2, 14),
}

DEFAULT_HARM_FLAG_VALUES = {
    "used_stale_memory": 0,
    "used_contradictory_memory": 0,
    "used_redundant_memory": 0,
    "used_unnecessary_personalization": 0,
    "false_suppressed_positive_control": 0,
}


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _is_beam_row(row: dict[str, Any]) -> bool:
    return str(row.get("dataset_name", "")).strip().lower() == "beam"


def _bucket_value(item: dict[str, Any], key: str, fallback: str) -> str:
    value = str(item.get(key, "")).strip()
    return value or fallback


def _bucket_summary(
    items: list[dict[str, Any]],
    *,
    naive_by_example: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    total = len(items)
    quality_total = sum(int(item["quality_score"]) for item in items)
    exact_correct = sum(1 for item in items if item["quality_label"] == "correct")
    partial_or_better = sum(1 for item in items if int(item["quality_score"]) >= 1)
    summary = {
        "rows": total,
        "avg_quality_score": round(_safe_div(quality_total, total), 3),
        "avg_quality_norm": round(_safe_div(quality_total, total * 2), 3),
        "exact_correct_rate": round(_safe_div(exact_correct, total), 3),
        "partial_or_better_rate": round(_safe_div(partial_or_better, total), 3),
    }
    if naive_by_example is not None:
        overlaps = [item for item in items if str(item["example_id"]) in naive_by_example]
        overlap_total = len(overlaps)
        agreement = 0
        regret = 0
        win = 0
        quality_delta = 0.0
        prompt_delta = 0.0
        pairwise_quality_scores: list[float] = []
        constrained_pairwise_scores: list[float] = []
        safety_regressions = 0
        for item in overlaps:
            baseline = naive_by_example[str(item["example_id"])]
            item_quality = int(item["quality_score"])
            baseline_quality = int(baseline["quality_score"])
            quality_delta += item_quality - baseline_quality
            prompt_delta += int(item["prompt_tokens"]) - int(baseline["prompt_tokens"])
            if item_quality == baseline_quality:
                agreement += 1
                pairwise_quality_scores.append(0.5)
            elif item_quality < baseline_quality:
                regret += 1
                pairwise_quality_scores.append(0.0)
            else:
                win += 1
                pairwise_quality_scores.append(1.0)
            safety_regression = _has_safety_regression(item, baseline)
            if safety_regression:
                safety_regressions += 1
            if item_quality > baseline_quality and not safety_regression:
                constrained_pairwise_scores.append(1.0)
            elif item_quality == baseline_quality and not safety_regression:
                constrained_pairwise_scores.append(0.5)
            else:
                constrained_pairwise_scores.append(0.0)
        quality_ci_low, quality_ci_high = _bootstrap_mean_ci(pairwise_quality_scores)
        constrained_ci_low, constrained_ci_high = _bootstrap_mean_ci(constrained_pairwise_scores)
        summary.update(
            {
                "full_context_agreement_rate": round(_safe_div(agreement, overlap_total), 3),
                "full_context_regret_rate": round(_safe_div(regret, overlap_total), 3),
                "full_context_win_rate": round(_safe_div(win, overlap_total), 3),
                "avg_quality_delta_vs_naive_top_k": round(_safe_div(quality_delta, overlap_total), 3),
                "avg_prompt_token_delta_vs_naive_top_k": round(_safe_div(prompt_delta, overlap_total), 3),
                "pairwise_quality_score_vs_naive_top_k": round(_safe_div(sum(pairwise_quality_scores), overlap_total), 3),
                "pairwise_quality_score_ci_low_vs_naive_top_k": round(quality_ci_low, 3),
                "pairwise_quality_score_ci_high_vs_naive_top_k": round(quality_ci_high, 3),
                "constrained_pairwise_score_vs_naive_top_k": round(
                    _safe_div(sum(constrained_pairwise_scores), overlap_total),
                    3,
                ),
                "constrained_pairwise_score_ci_low_vs_naive_top_k": round(constrained_ci_low, 3),
                "constrained_pairwise_score_ci_high_vs_naive_top_k": round(constrained_ci_high, 3),
                "safety_regression_rate_vs_naive_top_k": round(_safe_div(safety_regressions, overlap_total), 3),
            }
        )
    return summary


def _gold_memory_covered_value(output: dict[str, Any]) -> bool | None:
    value = output.get("gold_memory_covered")
    if value is None:
        return None
    return int(value) == 1


def _bootstrap_mean_ci(values: list[float], *, trials: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    sample_size = len(values)
    means: list[float] = []
    for _ in range(trials):
        sample = [values[rng.randrange(sample_size)] for _ in range(sample_size)]
        means.append(sum(sample) / sample_size)
    means.sort()
    low_index = max(0, int(0.025 * trials) - 1)
    high_index = min(trials - 1, int(0.975 * trials))
    return means[low_index], means[high_index]


def _has_safety_regression(item: dict[str, Any], baseline: dict[str, Any]) -> bool:
    for key in DEFAULT_HARM_FLAG_VALUES:
        if key in item and key in baseline and int(item[key]) > int(baseline[key]):
            return True
    return False


def _yes_no_label(text: str) -> str:
    tokens = normalize_tokens(text)
    if tokens[:2] == ["likely", "yes"]:
        return "likely_yes"
    if tokens[:2] == ["likely", "no"]:
        return "likely_no"
    if tokens and tokens[0] in {"yes", "no"}:
        return tokens[0]
    return ""


def _is_yes_no_question(question: str) -> bool:
    lowered = question.strip().lower()
    return lowered.startswith(("would ", "do ", "did ", "is ", "are ", "am ", "can "))


def _is_cumulative_action_question(question: str) -> bool:
    lowered = question.strip().lower()
    return lowered.startswith(("what has ", "what did ")) and (
        " done" in lowered or lowered.endswith(" do") or " do " in lowered
    )


def _tokens(text: str) -> list[str]:
    # Absolute entity normalization: replace hyphens/possessives with spaces and remove articles.
    # lowercase is handled by normalize_tokens, but we ensure tokens are treated strictly.
    normalized_text = str(text).replace("-", " ").replace("'s", "")
    tokens = normalize_tokens(normalized_text)
    # Remove standalone articles
    return [t for t in tokens if t not in {"a", "an", "the"}]


def _token_string(text: str) -> str:
    return " ".join(_tokens(text))




ACTION_CLAUSE_STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "for",
    "of",
    "and",
    "or",
    "my",
    "our",
    "their",
    "his",
    "her",
    "has",
    "have",
    "had",
    "did",
    "do",
    "done",
    "just",
    "really",
    "that",
    "this",
    "so",
    "far",
    "been",
}

NUMBER_WORD_VALUES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

QUANTITY_UNIT_TOKENS = {
    "day": "day",
    "days": "day",
    "week": "week",
    "weeks": "week",
    "month": "month",
    "months": "month",
    "year": "year",
    "years": "year",
    "hour": "hour",
    "hours": "hour",
    "minute": "minute",
    "minutes": "minute",
    "view": "view",
    "views": "view",
    "dollar": "currency",
    "dollars": "currency",
}


def _split_action_clauses(text: str) -> list[str]:
    stripped = text.strip().strip(".")
    if not stripped:
        return []
    clauses = [
        chunk.strip(" .")
        for chunk in re.split(r",|;|\band\b", stripped, flags=re.IGNORECASE)
        if chunk.strip(" .")
    ]
    return clauses if clauses else [stripped]


def _normalized_action_clause_tokens(text: str) -> list[str]:
    # Use lemmatizer to normalize verbs in action clauses "bought" -> "buy"
    tokens = _tokens(text)
    normalized = []
    for t in tokens:
        if t == "nearby":
            normalized.append("local") # Special case for semantic overlap
            continue
        normalized.append(_stem_token(t))
    
    return [token for token in normalized if token not in ACTION_CLAUSE_STOPWORDS]


def _cumulative_action_reference_coverage(answer: str, reference_answer: str) -> tuple[int, int]:
    reference_clauses = _split_action_clauses(reference_answer)
    answer_clauses = _split_action_clauses(answer)
    if not reference_clauses or not answer_clauses:
        return (0, 0)
    covered = 0
    for reference_clause in reference_clauses:
        ref_norm = " ".join(_normalized_action_clause_tokens(reference_clause))
        if not ref_norm:
            continue
        best_f1 = 0.0
        best_overlap = 0
        ref_tokens = set(_normalized_action_clause_tokens(reference_clause))
        for answer_clause in answer_clauses:
            ans_norm = " ".join(_normalized_action_clause_tokens(answer_clause))
            if not ans_norm:
                continue
            best_f1 = max(best_f1, token_f1(ans_norm, ref_norm))
            best_overlap = max(best_overlap, len(set(_normalized_action_clause_tokens(answer_clause)) & ref_tokens))
        if best_f1 >= _SPAN_MATCH_F1_THRESHOLD or best_overlap >= max(1, len(ref_tokens) - 1):
            covered += 1
    return covered, len(reference_clauses)


def _contains_normalized_span(longer: str, shorter: str) -> bool:
    longer_tokens = _tokens(longer)
    shorter_tokens = _tokens(shorter)
    if not longer_tokens or not shorter_tokens:
        return False
    if len(shorter_tokens) > len(longer_tokens):
        return False
    window = len(shorter_tokens)
    for start in range(len(longer_tokens) - window + 1):
        if longer_tokens[start : start + window] == shorter_tokens:
            return True
    return False


def _try_parse_calendar_date(text: str) -> date | None:
    stripped = text.strip().strip(".")
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(stripped, fmt).date()
        except ValueError:
            continue
    return None


def _extract_single_date(text: str) -> date | None:
    lowered = text.lower()
    friday_before = re.search(r"\bfriday before (\d{1,2} [a-z]+ \d{4})\b", lowered)
    if friday_before:
        anchor = _try_parse_calendar_date(friday_before.group(1))
        if anchor is not None:
            current = anchor - timedelta(days=1)
            while current.weekday() != 4:
                current -= timedelta(days=1)
            return current

    direct_patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2} [a-z]+ \d{4}\b",
        r"\b[a-z]+ \d{1,2} \d{4}\b",
    ]
    parsed_dates = []
    for pattern in direct_patterns:
        for match in re.findall(pattern, lowered):
            parsed = _try_parse_calendar_date(match)
            if parsed is not None:
                parsed_dates.append(parsed)

    if len(parsed_dates) == 1:
        return parsed_dates[0]
    return None


def _extract_single_month_day(text: str) -> tuple[int, int] | None:
    lowered = text.lower().strip().strip(".")
    holiday_hits = [month_day for holiday, month_day in HOLIDAY_MONTH_DAY.items() if holiday in lowered]
    if len(holiday_hits) == 1:
        return holiday_hits[0]
    if len(holiday_hits) > 1:
        return None

    patterns = [
        re.compile(r"\b(?P<month>[a-z]+)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\b"),
        re.compile(r"\b(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<month>[a-z]+)\b"),
    ]
    parsed: list[tuple[int, int]] = []
    for pattern in patterns:
        for match in pattern.finditer(lowered):
            month = MONTH_NAME_TO_NUMBER.get(match.group("month"))
            if month is None:
                continue
            parsed.append((month, int(match.group("day"))))
    if len(parsed) == 1:
        return parsed[0]
    return None


def _same_resolved_date(left: str, right: str) -> bool:
    left_date = _extract_single_date(left)
    right_date = _extract_single_date(right)
    if left_date is not None and right_date is not None:
        return left_date == right_date

    left_month_day = (left_date.month, left_date.day) if left_date is not None else _extract_single_month_day(left)
    right_month_day = (right_date.month, right_date.day) if right_date is not None else _extract_single_month_day(right)
    return left_month_day is not None and right_month_day is not None and left_month_day == right_month_day


def _contains_all_tokens(text: str, required_tokens: list[str]) -> bool:
    token_set = set(_tokens(text))
    return bool(required_tokens) and all(token in token_set for token in required_tokens)


def _matches_any_token_group(text: str, token_groups: list[list[str]]) -> bool:
    return any(_contains_all_tokens(text, group) for group in token_groups)


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


def _number_word_value(tokens: list[str]) -> float | None:
    cleaned = [token for token in tokens if token not in {"and", "a", "an", "approximately", "about"}]
    if not cleaned:
        return None
    if cleaned == ["half"]:
        return 0.5
    if cleaned == ["quarter"]:
        return 0.25
    if cleaned[-1] == "half" and len(cleaned) >= 2:
        prefix = _number_word_value(cleaned[:-1])
        if prefix is not None:
            return prefix + 0.5
    total = 0
    used = False
    for token in cleaned:
        if token not in NUMBER_WORD_VALUES:
            return None
        total += NUMBER_WORD_VALUES[token]
        used = True
    return float(total) if used else None


def _extract_quantity_mentions(text: str) -> list[tuple[float, str]]:
    lowered = text.lower().replace("-", " ")
    mentions: list[tuple[float, str]] = []

    for match in re.finditer(r"\$ ?(\d+(?:,\d{3})*(?:\.\d+)?)", text):
        value = float(match.group(1).replace(",", ""))
        mentions.append((value, "currency"))

    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?", lowered)
    for idx, token in enumerate(tokens):
        value: float | None = None
        unit = ""
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            value = float(token)
        else:
            for _window in range(min(4, len(tokens) - idx), 0, -1):
                value = _number_word_value(tokens[idx : idx + _window])
                if value is not None:
                    break
        if value is None:
            continue
        for look_ahead in range(1, 4):
            if idx + look_ahead >= len(tokens):
                break
            maybe_unit = QUANTITY_UNIT_TOKENS.get(tokens[idx + look_ahead], "")
            if maybe_unit:
                unit = maybe_unit
                break
        mentions.append((value, unit))
    return mentions


def _quantity_equivalent(answer: str, reference_answer: str) -> bool:
    ref_mentions = _extract_quantity_mentions(reference_answer)
    ans_mentions = _extract_quantity_mentions(answer)
    if not ref_mentions or not ans_mentions:
        return False
    for ref_value, ref_unit in ref_mentions:
        for ans_value, ans_unit in ans_mentions:
            same_unit = ref_unit == ans_unit or not ref_unit or not ans_unit
            if same_unit and abs(ref_value - ans_value) < 1e-6:
                return True
    return False


def _is_unknown_answer(answer: str) -> bool:
    normalized = _token_string(answer)
    return normalized in {"", "unknown"}


def _is_temporal_reference(reference_answer: str) -> bool:
    lowered = reference_answer.lower()
    return _extract_single_date(reference_answer) is not None or "friday before" in lowered or "around " in lowered


def quality_metrics(question: str, answer: str, reference_answer: str) -> dict[str, Any]:
    normalized_pred = normalize_answer_for_scoring(answer, question)
    normalized_ref = normalize_answer_for_scoring(reference_answer, question)
    em = normalized_exact_match(normalized_pred, normalized_ref)
    f1 = token_f1(normalized_pred, normalized_ref)
    bleu = bleu1(normalized_pred, normalized_ref)

    correct = False
    partial = False
    note = ""

    ref_lower = reference_answer.lower()
    is_neg_ctrl_ref = "did not mention this information" in ref_lower or "information provided is not enough" in ref_lower or "does not specify" in ref_lower

    if is_neg_ctrl_ref:
        if _is_unknown_answer(answer):
            correct = True
            note = "correctly abstained on negative control"
            em = 1.0
            f1 = 1.0
            bleu = 1.0
        else:
            return {
                "normalized_answer": normalized_pred,
                "normalized_reference_answer": normalized_ref,
                "exact_match": 0.0,
                "token_f1": 0.0,
                "bleu1": 0.0,
                "quality_label": "incorrect",
                "quality_score": 0,
                "reference_alignment_note": "hallucinated on negative control",
            }

    if _is_yes_no_question(question):
        pred_label = _yes_no_label(normalized_pred)
        ref_label = _yes_no_label(normalized_ref)
        if pred_label and ref_label and pred_label == ref_label:
            correct = True
            note = "yes/no label matches reference"

    if not correct:
        action_covered, action_total = (0, 0)
        if _is_cumulative_action_question(question):
            action_covered, action_total = _cumulative_action_reference_coverage(answer, reference_answer)
        if action_total > 0 and action_covered == action_total:
            correct = True
            note = "all reference action clauses are covered even with extra true detail"
        elif action_total > 1 and action_covered >= 1:
            partial = True
            note = "some reference action clauses are covered"
        elif em >= 1.0 or f1 >= _CORRECT_F1_THRESHOLD or bleu >= _CORRECT_BLEU_THRESHOLD:
            correct = True
            note = "high lexical match to reference"
        elif _contains_normalized_span(answer, reference_answer) or _contains_normalized_span(
            reference_answer, answer
        ) or _contains_normalized_span(normalized_pred, normalized_ref) or _contains_normalized_span(
            normalized_ref, normalized_pred
        ):
            correct = True
            note = "reference span is preserved in the answer"
        elif _same_resolved_date(answer, reference_answer):
            correct = True
            note = "resolved calendar date matches reference"
        elif _quantity_equivalent(answer, reference_answer):
            correct = True
            note = "numeric quantity matches reference"
        elif f1 >= _PARTIAL_F1_THRESHOLD or bleu >= _PARTIAL_BLEU_THRESHOLD:
            partial = True
            note = "partial lexical overlap with reference"
        else:
            note = "low overlap with reference"

    if correct:
        label = "correct"
        score = 2
    elif partial:
        label = "partially_correct"
        score = 1
    else:
        label = "incorrect"
        score = 0

    return {
        "normalized_answer": normalized_pred,
        "normalized_reference_answer": normalized_ref,
        "exact_match": round(em, 3),
        "token_f1": round(f1, 3),
        "bleu1": round(bleu, 3),
        "quality_label": label,
        "quality_score": score,
        "reference_alignment_note": note,
    }


def _overlap_score(answer: str, text: str) -> float:
    return token_f1(" ".join(normalize_tokens(answer)), " ".join(normalize_tokens(text)))


def harmful_use_flags(
    *,
    row: dict[str, Any],
    generated_answer: str,
    selected_memory_ids: list[str],
    quality_score: int,
) -> dict[str, int]:
    if _is_beam_row(row):
        return dict(DEFAULT_HARM_FLAG_VALUES)
    candidates = row["candidate_memories"]
    gold_decision = row.get("gold_decision") or {}
    gold_selected = set(gold_decision.get("selected_memory_ids") or [])
    gold_suppressed = set(gold_decision.get("suppressed_memory_ids") or [])
    by_memory_id = {str(candidate["memory_id"]): str(candidate["content"]) for candidate in candidates}

    suppressed_scores = [
        _overlap_score(generated_answer, by_memory_id[memory_id])
        for memory_id in gold_suppressed
        if memory_id in by_memory_id
    ]
    max_suppressed_overlap = max(suppressed_scores) if suppressed_scores else 0.0

    flags = dict(DEFAULT_HARM_FLAG_VALUES)

    if row.get("harm_type") == "stale_update" and max_suppressed_overlap >= 0.2:
        flags["used_stale_memory"] = 1
    elif row.get("harm_type") == "contradiction" and max_suppressed_overlap >= 0.2:
        flags["used_contradictory_memory"] = 1
    elif row.get("harm_type") == "redundancy" and selected_memory_ids:
        flags["used_redundant_memory"] = 1
    elif row.get("harm_type") == "unnecessary_personalization" and max_suppressed_overlap >= 0.2:
        flags["used_unnecessary_personalization"] = 1
    elif row.get("harm_type") == "positive_control" and quality_score < 2 and not gold_selected.issubset(selected_memory_ids):
        flags["false_suppressed_positive_control"] = 1

    return flags


def evaluate_run(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_system[row["system"]].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    naive_by_example = {
        str(row["example_id"]): row
        for row in by_system.get("naive_top_k", [])
    }
    for system, items in by_system.items():
        total = len(items)
        quality_total = sum(int(item["quality_score"]) for item in items)
        exact_correct = sum(1 for item in items if item["quality_label"] == "correct")
        partial_or_better = sum(1 for item in items if item["quality_score"] >= 1)
        stale_use = sum(int(item["used_stale_memory"]) for item in items)
        contradiction_use = sum(int(item["used_contradictory_memory"]) for item in items)
        redundant_use = sum(int(item["used_redundant_memory"]) for item in items)
        unnecessary_use = sum(int(item["used_unnecessary_personalization"]) for item in items)
        false_suppress = sum(int(item["false_suppressed_positive_control"]) for item in items)
        package_sufficient = sum(1 for item in items if bool(item.get("package_sufficient_for_reader")))
        correct_unknown = sum(1 for item in items if bool(item.get("unknown_was_correct")))
        false_unknown = sum(
            1
            for item in items
            if _is_unknown_answer(str(item.get("generated_answer") or ""))
            and not bool(item.get("unknown_was_correct"))
        )
        compiler_surface_success = sum(1 for item in items if bool(item.get("compiler_surface_success")))
        avg_prompt_tokens = _safe_div(sum(int(item["prompt_tokens"]) for item in items), total)
        avg_completion_tokens = _safe_div(sum(int(item["completion_tokens"]) for item in items), total)
        avg_latency = _safe_div(sum(float(item["latency_ms"]) for item in items), total)
        avg_selected_count = _safe_div(sum(int(item["selected_memory_count"]) for item in items), total)
        avg_selected_tokens = _safe_div(sum(int(item["selected_memory_tokens"]) for item in items), total)
        avg_budget_target_tokens = _safe_div(sum(int(item.get("budget_target_tokens") or 0) for item in items), total)
        avg_budget_hard_tokens = _safe_div(sum(int(item.get("budget_hard_tokens") or 0) for item in items), total)
        avg_proposal_budget_target_tokens = _safe_div(sum(int(item.get("proposal_budget_target_tokens") or 0) for item in items), total)
        avg_proposal_budget_hard_tokens = _safe_div(sum(int(item.get("proposal_budget_hard_tokens") or 0) for item in items), total)
        avg_llm_compiler_input_tokens = _safe_div(sum(int(item.get("llm_compiler_input_tokens") or 0) for item in items), total)
        compiler_type_counts: Counter[str] = Counter(
            str(item.get("compiler_type", "rule_based")) for item in items
        )

        quality_by_harm: dict[str, Counter[str]] = defaultdict(Counter)
        selection_mode_counts: Counter[str] = Counter()
        failure_stage_counts: Counter[str] = Counter()
        route_reason_counts: Counter[str] = Counter()
        query_family_counts: Counter[str] = Counter()
        budget_overshoot_reason_counts: Counter[str] = Counter()
        proposal_budget_overshoot_reason_counts: Counter[str] = Counter()
        quality_token_buckets: dict[str, list[int]] = defaultdict(list)
        for item in items:
            harm = _bucket_value(item, "harm_type", "unknown")
            quality_by_harm[harm]["rows"] += 1
            if item["quality_label"] == "correct":
                quality_by_harm[harm]["correct"] += 1
            if item["quality_score"] >= 1:
                quality_by_harm[harm]["partial_or_better"] += 1
            failure_stage_counts[_bucket_value(item, "failure_stage", "unknown")] += 1
            route_reason_counts[_bucket_value(item, "reader_route_reason", "unknown")] += 1
            query_family_counts[_bucket_value(item, "query_family", "unknown")] += 1
            for reason in item.get("budget_overshoot_reasons") or []:
                budget_overshoot_reason_counts[str(reason)] += 1
            for reason in item.get("proposal_budget_overshoot_reasons") or []:
                proposal_budget_overshoot_reason_counts[str(reason)] += 1
            selection_meta = item.get("selection_meta") or {}
            raw_mode = (
                item.get("selection_mode")
                or selection_meta.get("selection_mode")
                or selection_meta.get("pack_type")
            )
            mode = str(raw_mode).strip()
            if mode:
                selection_mode_counts[mode] += 1
            prompt_tokens = int(item.get("prompt_tokens") or 0)
            if prompt_tokens <= 128:
                quality_token_buckets["<=128"].append(int(item["quality_score"]))
            elif prompt_tokens <= 256:
                quality_token_buckets["129-256"].append(int(item["quality_score"]))
            elif prompt_tokens <= 512:
                quality_token_buckets["257-512"].append(int(item["quality_score"]))
            else:
                quality_token_buckets[">512"].append(int(item["quality_score"]))

        surgical_modes = {
            "single_anchor",
            "anchor_plus_support",
            "extractive_bundle",
            "top_3_contiguous",
            "anchor_only",
            "timeline_pack",
            "coverage_pack",
        }
        total_surgical = sum(count for mode, count in selection_mode_counts.items() if mode in surgical_modes)
        total_fallback = selection_mode_counts.get("bounded_topk", 0)
        avg_surgical_tokens = _safe_div(
            sum(int(item["prompt_tokens"]) for item in items if (item.get("selection_mode") or (item.get("selection_meta") or {}).get("pack_type")) in surgical_modes),
            total_surgical
        )
        avg_fallback_tokens = _safe_div(
            sum(int(item["prompt_tokens"]) for item in items if (item.get("selection_mode") or (item.get("selection_meta") or {}).get("pack_type")) == "bounded_topk"),
            total_fallback
        )

        summaries[system] = {
            "rows": total,
            "avg_quality_score": round(_safe_div(quality_total, total), 3),
            "avg_quality_norm": round(_safe_div(quality_total, total * 2), 3),
            "exact_correct_rate": round(_safe_div(exact_correct, total), 3),
            "partial_or_better_rate": round(_safe_div(partial_or_better, total), 3),
            "stale_memory_use_rate": round(_safe_div(stale_use, total), 3),
            "contradictory_memory_use_rate": round(_safe_div(contradiction_use, total), 3),
            "redundant_memory_use_rate": round(_safe_div(redundant_use, total), 3),
            "unnecessary_personalization_rate": round(_safe_div(unnecessary_use, total), 3),
            "false_suppression_positive_control_rate": round(_safe_div(false_suppress, total), 3),
            "package_sufficient_rate": round(_safe_div(package_sufficient, total), 3),
            "correct_unknown_rate": round(_safe_div(correct_unknown, total), 3),
            "false_unknown_rate": round(_safe_div(false_unknown, total), 3),
            "compiler_surface_success_rate": round(_safe_div(compiler_surface_success, total), 3),
            "avg_prompt_tokens": round(avg_prompt_tokens, 3),
            "avg_completion_tokens": round(avg_completion_tokens, 3),
            "avg_latency_ms": round(avg_latency, 3),
            "avg_selected_memory_count": round(avg_selected_count, 3),
            "avg_selected_memory_tokens": round(avg_selected_tokens, 3),
            "avg_budget_target_tokens": round(avg_budget_target_tokens, 3),
            "avg_budget_hard_tokens": round(avg_budget_hard_tokens, 3),
            "avg_proposal_budget_target_tokens": round(avg_proposal_budget_target_tokens, 3),
            "avg_proposal_budget_hard_tokens": round(avg_proposal_budget_hard_tokens, 3),
            "avg_llm_compiler_input_tokens": round(avg_llm_compiler_input_tokens, 3),
            "compiler_type_counts": {ct: count for ct, count in sorted(compiler_type_counts.items())},
            "by_harm_type": {
                harm: {
                    "rows": counts["rows"],
                    "exact_correct_rate": round(_safe_div(counts["correct"], counts["rows"]), 3),
                    "partial_or_better_rate": round(
                        _safe_div(counts["partial_or_better"], counts["rows"]),
                        3,
                    ),
                }
                for harm, counts in sorted(quality_by_harm.items())
            },
            "selection_mode_counts": (
                {mode: count for mode, count in sorted(selection_mode_counts.items())}
                if selection_mode_counts
                else None
            ),
            "failure_stage_counts": {stage: count for stage, count in sorted(failure_stage_counts.items())},
            "reader_route_reason_counts": {reason: count for reason, count in sorted(route_reason_counts.items())},
            "query_family_counts": {family: count for family, count in sorted(query_family_counts.items())},
            "budget_overshoot_reason_counts": {reason: count for reason, count in sorted(budget_overshoot_reason_counts.items())},
            "proposal_budget_overshoot_reason_counts": {
                reason: count for reason, count in sorted(proposal_budget_overshoot_reason_counts.items())
            },
            "quality_at_token_buckets": {
                bucket: round(_safe_div(sum(scores), len(scores) * 2), 3)
                for bucket, scores in sorted(quality_token_buckets.items())
                if scores
            },
            "surgical_coverage_rate": round(_safe_div(total_surgical, total), 3),
            "fallback_rate": round(_safe_div(total_fallback, total), 3),
            "avg_tokens_surgical": round(avg_surgical_tokens, 1),
            "avg_tokens_fallback": round(avg_fallback_tokens, 1),
            "notable_failures": [
                {
                    "example_id": item["example_id"],
                    "harm_type": item["harm_type"],
                    "question": item["question"],
                    "answer": item["generated_answer"],
                    "reference_answer": item["reference_answer"],
                    "quality_label": item["quality_label"],
                    "ability": item.get("ability"),
                    "chat_size": item.get("chat_size"),
                }
                for item in items
                if item["quality_score"] == 0
            ][:8],
        }
        if naive_by_example:
            summaries[system].update(
                _bucket_summary(items, naive_by_example=naive_by_example)
            )
            summaries[system]["avg_quality_score"] = round(_safe_div(quality_total, total), 3)
            summaries[system]["avg_quality_norm"] = round(_safe_div(quality_total, total * 2), 3)
            summaries[system]["avg_prompt_tokens"] = round(avg_prompt_tokens, 3)
            summaries[system]["avg_completion_tokens"] = round(avg_completion_tokens, 3)
            summaries[system]["avg_latency_ms"] = round(avg_latency, 3)
            summaries[system]["avg_selected_memory_count"] = round(avg_selected_count, 3)
            summaries[system]["avg_selected_memory_tokens"] = round(avg_selected_tokens, 3)
            summaries[system]["stale_memory_use_rate"] = round(_safe_div(stale_use, total), 3)
            summaries[system]["contradictory_memory_use_rate"] = round(_safe_div(contradiction_use, total), 3)
            summaries[system]["redundant_memory_use_rate"] = round(_safe_div(redundant_use, total), 3)
            summaries[system]["unnecessary_personalization_rate"] = round(_safe_div(unnecessary_use, total), 3)
            summaries[system]["false_suppression_positive_control_rate"] = round(_safe_div(false_suppress, total), 3)
        if any(_is_beam_row(item) for item in items):
            by_ability: dict[str, list[dict[str, Any]]] = defaultdict(list)
            by_chat_size: dict[str, list[dict[str, Any]]] = defaultdict(list)
            by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
            by_query_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in items:
                by_ability[_bucket_value(item, "ability", "unknown")].append(item)
                by_chat_size[_bucket_value(item, "chat_size", "unknown")].append(item)
                by_dataset[_bucket_value(item, "dataset_name", "unknown")].append(item)
                by_query_family[_bucket_value(item, "query_family", "unknown")].append(item)
            summaries[system]["by_ability"] = {
                ability: _bucket_summary(bucket_items, naive_by_example=naive_by_example)
                for ability, bucket_items in sorted(by_ability.items())
            }
            summaries[system]["by_chat_size"] = {
                chat_size: _bucket_summary(bucket_items, naive_by_example=naive_by_example)
                for chat_size, bucket_items in sorted(by_chat_size.items())
            }
            summaries[system]["by_dataset"] = {
                dataset_name: _bucket_summary(bucket_items, naive_by_example=naive_by_example)
                for dataset_name, bucket_items in sorted(by_dataset.items())
            }
            summaries[system]["by_query_family"] = {
                family: _bucket_summary(bucket_items, naive_by_example=naive_by_example)
                for family, bucket_items in sorted(by_query_family.items())
            }
        else:
            by_query_family = defaultdict(list)
            for item in items:
                by_query_family[_bucket_value(item, "query_family", "unknown")].append(item)
            summaries[system]["by_query_family"] = {
                family: _bucket_summary(bucket_items, naive_by_example=naive_by_example)
                for family, bucket_items in sorted(by_query_family.items())
            }

        # --- Dataset-specific paper-aligned aggregation ---
        detected_datasets = {_detect_dataset(item) for item in items}

        if "longmemeval" in detected_datasets:
            lme_items = [item for item in items if _detect_dataset(item) == "longmemeval"]
            lme_total = len(lme_items)
            lme_correct = sum(1 for item in lme_items if int(item["quality_score"]) >= 2)
            by_qtype: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in lme_items:
                qt = str(item.get("question_type", "unknown")).strip() or "unknown"
                by_qtype[qt].append(item)
            per_type_accuracy = {
                qt: round(_safe_div(sum(1 for i in bucket if int(i["quality_score"]) >= 2), len(bucket)), 3)
                for qt, bucket in sorted(by_qtype.items())
            }
            summaries[system]["longmemeval_paper_metrics"] = {
                "rows": lme_total,
                "accuracy": round(_safe_div(lme_correct, lme_total), 3),
                "task_averaged_accuracy": round(
                    _safe_div(sum(per_type_accuracy.values()), len(per_type_accuracy)), 3
                ) if per_type_accuracy else 0.0,
                "by_question_type": {
                    qt: {"rows": len(by_qtype[qt]), "accuracy": acc}
                    for qt, acc in per_type_accuracy.items()
                },
            }

        if "locomo" in detected_datasets:
            loc_items = [item for item in items if _detect_dataset(item) == "locomo"]
            loc_total = len(loc_items)
            f1_values = [float(item.get("token_f1", 0)) for item in loc_items]
            by_cat: dict[int, list[float]] = defaultdict(list)
            for item in loc_items:
                cat = int(item.get("locomo_category", 0))
                by_cat[cat].append(float(item.get("token_f1", 0)))
            summaries[system]["locomo_paper_metrics"] = {
                "rows": loc_total,
                "mean_f1_paper": round(_safe_div(sum(f1_values), loc_total), 3),
                "by_locomo_category": {
                    cat: {"rows": len(vals), "mean_f1": round(_safe_div(sum(vals), len(vals)), 3)}
                    for cat, vals in sorted(by_cat.items())
                },
            }

    if "naive_top_k" in summaries:
        naive_prompt_tokens = float(summaries["naive_top_k"]["avg_prompt_tokens"])
        for summary in summaries.values():
            summary["prompt_token_ratio_vs_naive_top_k"] = round(
                _safe_div(float(summary["avg_prompt_tokens"]), naive_prompt_tokens),
                3,
            )
    return summaries
