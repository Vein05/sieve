"""Scoring utilities for token-level QA metrics.

Vendored from the Pali research workspace (eval_locomo_f1_bleu.py) to avoid
external path dependencies.  Includes the full answer normalization pipeline:
temporal phrase extraction, person answer trimming, date normalization.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# ── Regex constants ──────────────────────────────────────────────────

TOKEN_RE = re.compile(r"[a-z0-9]+")
THINK_BLOCK_RE = re.compile(r"(?is)<think>.*?</think>")
THINK_TAG_RE = re.compile(r"(?i)</?think>")
ANSWER_PREFIX_RE = re.compile(r"(?i)^\s*(?:answer|final answer)\s*:\s*")

TEMPORAL_QUERY_RE = re.compile(r"\b(when|date|time|day|month|year|yesterday|today|tomorrow)\b")
PERSON_QUERY_RE = re.compile(r"\b(who|name|which person|whose)\b")
MULTIHOP_QUERY_RE = re.compile(r"\b(before|after|first|last|both|either|between|together|shared|across|compared to)\b")

MONTH_NAME_RE = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

RELATIVE_DATE_RE = re.compile(
    rf"\b(?:the\s+)?(?:week|month|year|day|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    rf"\s+(?:before|after)\s+\d{{1,2}}\s+{MONTH_NAME_RE}\s*,?\s*\d{{4}}\b",
    re.IGNORECASE,
)
FULL_DATE_RE = re.compile(rf"\b\d{{1,2}}\s+{MONTH_NAME_RE}\s*,?\s*\d{{4}}\b", re.IGNORECASE)
MONTH_DAY_YEAR_RE = re.compile(rf"\b{MONTH_NAME_RE}\s+\d{{1,2}},?\s*\d{{4}}\b", re.IGNORECASE)
MONTH_YEAR_RE = re.compile(rf"\b{MONTH_NAME_RE}\s+\d{{4}}\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
DURATION_RE = re.compile(r"\b\d+\s+(?:years?|months?|weeks?|days?)\b", re.IGNORECASE)
TEMPORAL_SIGNAL_RE = re.compile(
    r"\b(yesterday|today|tomorrow|last\s+(?:week|month|year)|next\s+(?:week|month|year)"
    r"|\d+\s+(?:years?|months?|weeks?|days?)\s+ago)\b",
    re.IGNORECASE,
)

SPEAKER_PREFIX_RE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9 .'\-]{0,80}(?:\s*\([^)]+\))?:\s*")
LEADING_DATE_PREFIX_RE = re.compile(
    rf"(?i)^on\s+\d{{1,2}}\s+{MONTH_NAME_RE}\s+\d{{4}},\s*",
    re.IGNORECASE,
)
SAID_THAT_PREFIX_RE = re.compile(r"(?i)^[A-Z][A-Za-z0-9 .'\-]{0,80}\s+said that\s+", re.IGNORECASE)


# ── Token-level metrics ─────────────────────────────────────────────

def normalize_tokens(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def token_f1(pred: str, ref: str) -> float:
    p = normalize_tokens(pred)
    r = normalize_tokens(ref)
    if not p or not r:
        return 0.0
    cp = Counter(p)
    cr = Counter(r)
    common = sum((cp & cr).values())
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(r)
    return (2 * precision * recall) / (precision + recall)


def bleu1(pred: str, ref: str) -> float:
    p = normalize_tokens(pred)
    r = normalize_tokens(ref)
    if not p or not r:
        return 0.0
    cp = Counter(p)
    cr = Counter(r)
    overlap = sum((cp & cr).values())
    precision = overlap / len(p)
    if precision <= 0:
        return 0.0
    bp = 1.0 if len(p) >= len(r) else math.exp(1 - (len(r) / len(p)))
    return bp * precision


def normalized_exact_match(pred: str, ref: str) -> float:
    p = " ".join(normalize_tokens(pred))
    r = " ".join(normalize_tokens(ref))
    if not p or not r:
        return 0.0
    return 1.0 if p == r else 0.0


# ── Answer cleaning ─────────────────────────────────────────────────

def repair_answer_spacing(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    value = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", value)
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def is_unknown_answer(text: str) -> bool:
    cleaned = (text or "").strip().lower()
    return cleaned in {"", "unknown", "n/a", "na"}


def clean_generated_answer(raw: str) -> str:
    """Strip reasoning artifacts and extract the final answer line."""
    text = (raw or "").replace("\r", "\n")
    text = THINK_BLOCK_RE.sub(" ", text)
    text = THINK_TAG_RE.sub(" ", text)
    text = re.sub(r"(?is)```.*?```", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return "Unknown"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    filtered: list[str] = []
    for line in lines:
        low = line.lower()
        if low in {"answer:", "final answer:"}:
            continue
        if low.startswith("reasoning:") or low.startswith("thought:"):
            continue
        filtered.append(line)

    candidate = filtered[-1] if filtered else lines[-1]
    candidate = ANSWER_PREFIX_RE.sub("", candidate).strip()
    candidate = re.sub(r"^\s*\d+\s*[\.)\s]\s*", "", candidate)
    candidate = candidate.strip(" \"'")
    candidate = repair_answer_spacing(candidate)
    if not candidate:
        return "Unknown"
    if is_unknown_answer(candidate):
        return "Unknown"
    return candidate


# ── Answer normalization for scoring ─────────────────────────────────

def classify_query(query: str) -> tuple[bool, bool, bool]:
    """Classify query as (temporal, person, multihop)."""
    q = (query or "").strip().lower()
    if not q:
        return False, False, False
    temporal = bool(TEMPORAL_QUERY_RE.search(q))
    person = bool(PERSON_QUERY_RE.search(q))
    multihop = bool(MULTIHOP_QUERY_RE.search(q))
    return temporal, person, multihop


def normalize_month_token(token: str) -> str:
    key = token.strip().lower()[:3]
    mapping = {
        "jan": "January", "feb": "February", "mar": "March",
        "apr": "April", "may": "May", "jun": "June",
        "jul": "July", "aug": "August", "sep": "September",
        "oct": "October", "nov": "November", "dec": "December",
    }
    return mapping.get(key, token.strip().title())


def normalize_date_phrase(phrase: str) -> str:
    value = (phrase or "").strip(" \t\r\n.,;:")
    if not value:
        return ""
    # Month Day, Year → Day Month Year
    md = re.match(rf"(?i)^({MONTH_NAME_RE})\s+(\d{{1,2}}),?\s*(\d{{4}})$", value)
    if md:
        return f"{int(md.group(2))} {normalize_month_token(md.group(1))} {md.group(3)}"
    # Day Month Year (already canonical)
    dm = re.match(rf"(?i)^(\d{{1,2}})\s+({MONTH_NAME_RE}),?\s*(\d{{4}})$", value)
    if dm:
        return f"{int(dm.group(1))} {normalize_month_token(dm.group(2))} {dm.group(3)}"
    # Month Year
    my = re.match(rf"(?i)^({MONTH_NAME_RE})\s+(\d{{4}})$", value)
    if my:
        return f"{normalize_month_token(my.group(1))} {my.group(2)}"
    return value


def shape_temporal_answer(question: str, answer: str) -> str:
    value = normalize_date_phrase(answer)
    if not value:
        return ""
    q = (question or "").strip().lower()
    full = re.match(rf"(?i)^(\d{{1,2}})\s+({MONTH_NAME_RE})\s+(\d{{4}})$", value)
    month_year = re.match(rf"(?i)^({MONTH_NAME_RE})\s+(\d{{4}})$", value)
    year_match = YEAR_RE.search(value)
    if ("what year" in q or "which year" in q) and year_match:
        return year_match.group(0)
    if ("what month" in q or "which month" in q) and full:
        return f"{normalize_month_token(full.group(2))} {full.group(3)}"
    if ("what month" in q or "which month" in q) and month_year:
        return f"{normalize_month_token(month_year.group(1))} {month_year.group(2)}"
    return value


def strip_non_temporal_prefixes(text: str) -> str:
    value = SPEAKER_PREFIX_RE.sub("", (text or "").strip()).strip()
    value = LEADING_DATE_PREFIX_RE.sub("", value).strip()
    value = SAID_THAT_PREFIX_RE.sub("", value).strip()
    return value


def extract_temporal_phrase(text: str, question: str, enable_resolver: bool = False) -> str:
    source = (text or "").strip()
    if not source:
        return ""

    candidates: list[tuple[float, str]] = []

    def collect(pattern: re.Pattern[str], base_score: float) -> None:
        for m in pattern.finditer(source):
            phrase = normalize_date_phrase(m.group(0))
            if not phrase:
                continue
            score = base_score + (len(normalize_tokens(phrase)) * 0.01)
            candidates.append((score, phrase))

    collect(RELATIVE_DATE_RE, 1.18)
    collect(FULL_DATE_RE, 1.20)
    collect(MONTH_DAY_YEAR_RE, 1.15)
    collect(MONTH_YEAR_RE, 0.95)
    collect(YEAR_RE, 0.90)
    collect(DURATION_RE, 0.80)

    if not candidates:
        return ""

    candidates.sort(key=lambda x: (-x[0], len(x[1]), x[1].lower()))
    return shape_temporal_answer(question, candidates[0][1])


def normalize_answer_for_scoring(answer: str, question: str, temporal_resolver: bool = False) -> str:
    """Normalize answer for fair scoring: temporal extraction, person trimming."""
    value = repair_answer_spacing(answer)
    if not value:
        return "Unknown"
    temporal, person, _ = classify_query(question)
    if temporal:
        temporal_phrase = extract_temporal_phrase(value, question, enable_resolver=temporal_resolver)
        if temporal_phrase:
            value = shape_temporal_answer(question, temporal_phrase)
    value = strip_non_temporal_prefixes(value)
    if person:
        value = re.split(r"[.;]", value, maxsplit=1)[0].strip()
        words = value.split()
        if len(words) > 6:
            value = " ".join(words[:6]).strip()
    value = re.sub(r"\s+", " ", value).strip(" \"'")
    if is_unknown_answer(value):
        return "Unknown"
    return value if value else "Unknown"


__all__ = [
    "bleu1",
    "clean_generated_answer",
    "normalize_answer_for_scoring",
    "normalize_tokens",
    "normalized_exact_match",
    "token_f1",
]
