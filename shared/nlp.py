"""Shared assembly-method utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import logging
import spacy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NLPProfile:
    tokens: tuple[str, ...]
    tags: tuple[str, ...]
    stems: tuple[str, ...]


@lru_cache(maxsize=1)
def _get_nlp():
    """Load SpaCy pipeline with tagger and sentencizer (no parser/NER)."""
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        nlp.add_pipe("sentencizer")
        logger.info("SpaCy pipeline loaded.")
        return nlp
    except OSError:
        logger.info("SpaCy model not found, downloading en_core_web_sm...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        nlp.add_pipe("sentencizer")
        return nlp

@lru_cache(maxsize=50_000)
def get_nlp_profile(text: str) -> NLPProfile:
    """Perform a single-pass SpaCy parse and return a consolidated profile.

    Layer 1: in-process ``@lru_cache`` (fast).
    Layer 2: SQLite disk cache (survives restarts).
    """
    if not text:
        return NLPProfile(tokens=(), tags=(), stems=())

    from .cache import disk_get, disk_put, text_key
    _key = text_key(text)
    cached = disk_get("nlp_profile", _key)
    if cached is not None:
        return NLPProfile(*cached)

    nlp = _get_nlp()
    doc = nlp(text)

    tokens = []
    tags = []
    stems = []
    for token in doc:
        tokens.append(token.text)
        tags.append(token.tag_)
        stems.append(token.lemma_.lower())

    result = NLPProfile(
        tokens=tuple(tokens),
        tags=tuple(tags),
        stems=tuple(stems),
    )
    disk_put("nlp_profile", _key, (result.tokens, result.tags, result.stems))
    return result


@lru_cache(maxsize=20_000)
def cached_sent_tokenize(text: str) -> list[str]:
    from .cache import disk_get, disk_put, text_key

    _key = text_key(text)
    cached = disk_get("sent_tokenize", _key)
    if cached is not None:
        return cached

    nlp = _get_nlp()
    doc = nlp(text)
    sents = [sent.text for sent in doc.sents]
    disk_put("sent_tokenize", _key, sents)
    return sents

def cached_word_tokenize(text: str) -> list[str]:
    return list(get_nlp_profile(text).tokens)

def cached_pos_tag(tokens_tuple: tuple[str, ...]) -> list[tuple[str, str]]:
    # Join tokens to parse in context for better tagging accuracy
    text = " ".join(tokens_tuple)
    profile = get_nlp_profile(text)
    return list(zip(profile.tokens, profile.tags))

# --- Semantic Constants ---

CONVERSATIONAL_SYNONYMS = {
    "speed": {"plan", "connection", "rate", "mbps", "gbps"},
    "plan": {"speed", "internet", "tier", "subscription"},
    "price": {"cost", "expense", "fare", "fee", "charge", "amount", "total"},
    "cost": {"price", "expense", "fare", "fee", "charge", "amount", "total"},
    "spent": {"cost", "price", "paid", "amount"},
    "buy": {"purchase", "bought", "got", "order", "ordered"},
    "purchase": {"buy", "bought", "got", "order", "ordered"},
    "job": {"work", "position", "role", "career"},
    "work": {"job", "position", "role", "career"},
    "start": {"begin", "commence", "started", "began"},
}

CANONICAL_ATTRIBUTE_ALIASES = {
    "breed": {"breed"},
    "occupation": {"occupation", "role", "job", "profession"},
    "name": {"name", "called", "named", "title"},
    "speed": {"speed", "rate", "bandwidth", "mbps", "gbps"},
}

NOISE_TOKENS = frozenset({"'s", "s", "the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "with", "about", "my", "your", "his", "her"})


_NUMBER_WORDS = {
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
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
}

DEFAULT_V2_BUDGET_CONFIG = {
    "token_budget_min": 448,
    "token_budget_max": 2800,
    "token_budget_steps": [
        {"candidate_tokens_gte": 9000, "candidate_count_gte": 40, "budget": 2150, "match_mode": "and"},
        {"candidate_tokens_gte": 4500, "candidate_count_gte": 24, "budget": 1800, "match_mode": "and"},
        {"candidate_tokens_gte": 2200, "candidate_count_gte": 12, "budget": 1350, "match_mode": "and"},
        {"candidate_tokens_gte": 1120, "candidate_count_gte": 6, "budget": 900, "match_mode": "or"},
        {"candidate_tokens_gte": 0, "candidate_count_gte": 0, "budget": 540, "match_mode": "or"},
    ],
    "active_context_bonus_threshold": 180,
    "active_context_bonus_tokens": 180,
    "family_token_budget_additions": {
        "default": 0,
        "single_anchor": 0,
        "information_extraction": 135,
        "current_state": 225,
        "knowledge_update": 315,
        "conflict_update": 315,
        "temporal": 400,
        "contradiction": 360,
        "ordering": 540,
        "aggregation": 540,
        "multi_session": 540,
        "abstention": 0,
    },
    "family_token_budget_floors": {
        "default": 540,
        "single_anchor": 630,
        "information_extraction": 800,
        "current_state": 1075,
        "knowledge_update": 1250,
        "conflict_update": 1250,
        "temporal": 1400,
        "contradiction": 1350,
        "ordering": 1750,
        "aggregation": 1750,
        "multi_session": 1750,
        "abstention": 0,
    },
    "max_selected_by_family": {
        "default": 8,
        "single_anchor": 8,
        "information_extraction": 10,
        "current_state": 12,
        "knowledge_update": 12,
        "conflict_update": 12,
        "temporal": 10,
        "contradiction": 8,
        "ordering": 10,
        "aggregation": 10,
        "multi_session": 10,
        "abstention": 0,
    },
}


@dataclass
class CandidateView:
    memory_id: str
    text: str
    rank: int
    token_count: int
    profile: Any  # Was TextProfile, now Any to break circular import
    date_key: tuple[int, int, int]
    has_update_marker: bool
    normalized_text: str
    content_tokens: set[str]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=50_000)
def cached_build_profile(
    text: str,
    *,
    use_concept_specs: bool = False,
) -> Any:
    # Profile construction is a repeated hot path during selector training/inference.
    from controller.text_utils import build_profile
    return build_profile(text, use_concept_specs=use_concept_specs)


# --- Core NLP Utilities ---

from nltk.stem import WordNetLemmatizer as _WNL

_WORDNET_LEMMATIZER = _WNL()


def _wordnet_lemma(word: str) -> str:
    """Lookup-based lemmatization via NLTK WordNet — no neural network."""
    w = word.lower()
    v = _WORDNET_LEMMATIZER.lemmatize(w, pos="v")
    if v != w:
        return v
    n = _WORDNET_LEMMATIZER.lemmatize(w, pos="n")
    if n != w:
        return n
    a = _WORDNET_LEMMATIZER.lemmatize(w, pos="a")
    if a != w:
        return a
    return w


@lru_cache(maxsize=20_000)
def stem_token(token: str) -> str:
    """Lemmatize a single token using NLTK WordNet (lookup-based, ~200x faster than spaCy)."""
    if not token or not isinstance(token, str):
        return ""
    return _wordnet_lemma(token.strip().lower())


def canonical_attribute(token: str) -> str | None:
    """Map an attribute-like token to a reusable canonical facet when possible."""
    stem = stem_token(token)
    for canonical, aliases in CANONICAL_ATTRIBUTE_ALIASES.items():
        if stem == canonical or stem in aliases:
            return canonical
    return None


def canonical_attributes_for_text(text: str) -> set[str]:
    attributes: set[str] = set()
    for token in get_clean_tokens(text):
        canonical = canonical_attribute(token)
        if canonical:
            attributes.add(canonical)
    return attributes

def get_clean_tokens(text: str, exclude_noise: bool = True) -> list[str]:
    """Tokenize and clean text, optionally removing possessives and stop-words."""
    if not text:
        return []
    normalized = text.lower()
    tokens = cached_word_tokenize(normalized)
    
    out = []
    for t in tokens:
        if t in {"'s", "s"} and exclude_noise:
            continue
        if not t.isalnum():
            continue
        if exclude_noise and t in NOISE_TOKENS:
            continue
        out.append(t)
    return out

def get_stems_for_text(text: str) -> list[str]:
    """Get a list of stems for the tokens in the text."""
    tokens = get_clean_tokens(text)
    return [stem_token(t) for t in tokens if t]

def check_stem_equivalence(token_a: str, token_b: str) -> bool:
    """Check if two tokens are equivalent via stems or known conversational synonyms."""
    if not token_a or not token_b:
        return False
    stem_a = stem_token(token_a)
    stem_b = stem_token(token_b)
    
    if stem_a == stem_b:
        return True

    canonical_a = canonical_attribute(stem_a)
    canonical_b = canonical_attribute(stem_b)
    if canonical_a and canonical_a == canonical_b:
        return True
        
    # Check synonyms
    syns_a = CONVERSATIONAL_SYNONYMS.get(stem_a, set())
    if stem_b in syns_a:
        return True
        
    syns_b = CONVERSATIONAL_SYNONYMS.get(stem_b, set())
    if stem_a in syns_b:
        return True
        
    return False


def candidate_views(
    row: dict[str, Any],
    *,
    use_concept_specs: bool = False,
) -> list[CandidateView]:
    cache_key = "_v3_cv_concepts" if use_concept_specs else "_v3_cv"
    if cache_key in row:
        return row[cache_key]

    views: list[CandidateView] = []
    from controller.logic import token_count
    for rank, candidate in enumerate(row["candidate_memories"], start=1):
        text = str(candidate["content"])
        profile = cached_build_profile(text, use_concept_specs=use_concept_specs)
        views.append(
            CandidateView(
                memory_id=str(candidate["memory_id"]),
                text=text,
                rank=rank,
                token_count=token_count(text),
                profile=profile,
                date_key=profile.date_key or (0, 0, 0),
                has_update_marker=profile.has_update_marker,
                normalized_text=profile.normalized_text,
                content_tokens=profile.content_tokens,
            )
        )
    row[cache_key] = views
    return views


def shared_result(
    selected: list[CandidateView],
    *,
    budget: int | None,
    budget_mode: str,
    selection_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "selected_memory_ids": [candidate.memory_id for candidate in selected],
        "selected_token_count": sum(candidate.token_count for candidate in selected),
        "selected_count": len(selected),
        "target_token_budget": budget,
        "budget_mode": budget_mode,
    }
    if selection_meta is not None:
        result["selection_meta"] = selection_meta
    return result


def target_token_budget(row: dict[str, Any], context: dict[str, Any]) -> int | None:
    budgets = context.get("controller_budgets", {})
    budget = budgets.get(str(row["example_id"]))
    return int(budget) if budget is not None else None


def _v2_budget_config(context: dict[str, Any]) -> dict[str, Any]:
    config = dict(DEFAULT_V2_BUDGET_CONFIG)
    config.update(context.get("v2_budget_config", {}))
    return config


def _v2_budget_override(
    row: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    overrides = context.get("v2_budget_overrides", {})
    return dict(overrides.get(str(row["example_id"]), {}))


def _row_candidate_token_count(row: dict[str, Any]) -> int:
    from controller.logic import token_count
    return sum(token_count(str(candidate.get("content", ""))) for candidate in row.get("candidate_memories", []))


def _row_active_context_token_count(row: dict[str, Any]) -> int:
    from controller.logic import token_count
    return sum(token_count(str(item)) for item in row.get("active_context", []))


def _normalized_v2_query_family(query_family: str | None) -> str:
    normalized = str(query_family or "").strip().lower()
    return normalized or "default"


def v2_budget_family_hint(row: dict[str, Any]) -> str:
    evidence = row.get("evidence_sufficiency", {})
    if str(evidence.get("label", "")).strip().lower() == "abstention":
        return "abstention"
    if is_event_ordering_query(row):
        return "ordering"
    lowered_harm = str(row.get("harm_type", "")).strip().lower()
    lowered_ability = str(row.get("ability", "")).strip().lower()
    family = lowered_ability or lowered_harm
    if family in {"contradiction", "contradiction_resolution"}:
        return "contradiction"
    if family in {"temporal", "temporal_reasoning"}:
        return "temporal"
    if family in {"knowledge_update", "stale_update"}:
        return "knowledge_update"
    if family in {"multi_session_reasoning", "multi_session"}:
        return "multi_session"
    if family == "information_extraction":
        return "information_extraction"
    return "default"


def v2_target_token_budget(
    row: dict[str, Any],
    context: dict[str, Any],
    *,
    query_family: str | None = None,
) -> int:
    override = _v2_budget_override(row, context)
    if "token_budget" in override:
        return max(0, int(override["token_budget"]))

    family = _normalized_v2_query_family(query_family or v2_budget_family_hint(row))
    if family == "abstention":
        return 0

    config = _v2_budget_config(context)
    candidate_tokens = _row_candidate_token_count(row)
    candidate_count = len(row.get("candidate_memories", []))
    base_budget = int(config["token_budget_min"])
    for step in config.get("token_budget_steps", []):
        token_cond = candidate_tokens >= int(step["candidate_tokens_gte"])
        count_cond = candidate_count >= int(step["candidate_count_gte"])
        match_mode = str(step.get("match_mode", "or")).strip().lower()
        if (token_cond and count_cond) if match_mode == "and" else (token_cond or count_cond):
            base_budget = int(step["budget"])
            break

    if _row_active_context_token_count(row) >= int(config.get("active_context_bonus_threshold", 0)):
        base_budget += int(config.get("active_context_bonus_tokens", 0))

    family_budget = base_budget + int(config.get("family_token_budget_additions", {}).get(family, 0))
    family_budget = max(
        family_budget,
        int(config.get("family_token_budget_floors", {}).get(family, config["token_budget_min"])),
    )
    return min(int(config["token_budget_max"]), max(int(config["token_budget_min"]), family_budget))


def v2_max_selected(
    row: dict[str, Any],
    context: dict[str, Any],
    *,
    query_family: str | None = None,
) -> int:
    override = _v2_budget_override(row, context)
    if "max_selected" in override:
        return max(0, int(override["max_selected"]))

    family = _normalized_v2_query_family(query_family or v2_budget_family_hint(row))
    config = _v2_budget_config(context)
    by_family = config.get("max_selected_by_family", {})
    default_value = int(by_family.get("default", DEFAULT_V2_BUDGET_CONFIG["max_selected_by_family"]["default"]))
    return max(0, int(by_family.get(family, default_value)))


def controller_result(row: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    by_example = context.get("controller_results_by_example", {})
    result = by_example.get(str(row["example_id"]))
    if result is not None:
        return result
    from controller.logic import run_example
    return run_example(row, context["controller_config"])


def append_with_budget(
    selected: list[CandidateView],
    candidate: CandidateView,
    *,
    token_budget: int | None,
    max_selected: int,
    used_tokens: int,
) -> tuple[bool, int]:
    if any(existing.memory_id == candidate.memory_id for existing in selected):
        return False, used_tokens
    if len(selected) >= max_selected:
        return False, used_tokens
    if token_budget is None:
        selected.append(candidate)
        return True, used_tokens + candidate.token_count
    next_used = used_tokens + candidate.token_count
    if next_used <= token_budget:
        selected.append(candidate)
        return True, next_used
    if not selected and token_budget > 0:
        selected.append(candidate)
        return True, next_used
    return False, used_tokens


def requested_item_count(query: str) -> int | None:
    lowered = query.lower()
    digit_match = re.search(r"(?:mention|list|include)\D{0,24}(\d{1,2})\s+items?", lowered)
    if digit_match:
        return int(digit_match.group(1))
    word_match = re.search(
        r"(?:mention|list|include)\D{0,24}"
        r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+items?",
        lowered,
    )
    if word_match:
        return _NUMBER_WORDS.get(word_match.group(1))
    return None


def is_event_ordering_query(row: dict[str, Any]) -> bool:
    if str(row.get("harm_type", "")).strip().lower() == "event_ordering":
        return True
    lowered = str(row.get("query", "")).lower()
    return "in order" in lowered or "order in which" in lowered


def evenly_spaced_indices(total: int, target: int) -> list[int]:
    if total <= 0 or target <= 0:
        return []
    if target >= total:
        return list(range(total))
    raw = [round(index * (total - 1) / max(1, target - 1)) for index in range(target)]
    deduped: list[int] = []
    seen: set[int] = set()
    for value in raw:
        value = min(total - 1, max(0, int(value)))
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    cursor = 0
    while len(deduped) < target and cursor < total:
        if cursor not in seen:
            deduped.append(cursor)
            seen.add(cursor)
        cursor += 1
    return sorted(deduped)
