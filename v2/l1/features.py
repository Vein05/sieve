"""Feature extraction for L1 sufficiency estimator.

All features are cheap, domain-general, computed from (query, pack_texts) only.
No benchmark-identity features. No LLM calls. No network access.

Stem matching uses the same shared.nlp machinery as the v2 scorer but amortises
the cost by pre-computing a stem set for the pack rather than doing an O(|Q|·|P|)
cross-product for each coverage check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import Sequence

from shared.nlp import check_stem_equivalence, get_clean_tokens, stem_token

# Token patterns for entity-ish and date detection.
_RE_ENTITY_TOKEN = re.compile(r"^[A-Z][a-z]*$|^\d+$")
_RE_DATE_TOKEN = re.compile(
    r"\b\d{4}\b"            # year-like
    r"|\b\d{1,2}/\d{1,2}"  # MM/DD
    r"|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?"
    r"|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
    re.IGNORECASE,
)
_CONTENT_STOPWORDS = frozenset(
    {"the", "a", "an", "is", "are", "was", "were", "be", "been",
     "have", "has", "had", "do", "does", "did", "will", "would",
     "could", "should", "may", "might", "i", "you", "he", "she", "it",
     "we", "they", "and", "or", "but", "in", "on", "at", "to", "for",
     "of", "with", "by", "from", "as", "that", "this", "which", "who"}
)

# Rough char budget for concatenated pack (avoids O(n) token overhead on huge packs).
_MAX_PACK_CHARS = 64_000
_MIN_QUERY_TOKENS = 1  # avoid divide-by-zero


@dataclass(frozen=True)
class FeatureVector:
    """One feature vector for (query, pack).

    All values are floats for uniform handling downstream.
    """

    stem_coverage: float            # fraction of query stems hit by any pack unit
    content_word_coverage: float    # coverage restricted to non-stopword tokens
    entity_coverage: float          # coverage of capitalized / numeric query tokens
    date_coverage: float            # coverage of date-like query tokens
    pack_token_count: float         # total whitespace-token count in concatenated pack
    pack_query_length_ratio: float  # pack_tokens / max(query_tokens, 1)
    n_units: float                  # number of candidate units in pack
    max_unit_overlap: float         # max per-unit stem overlap with query
    mean_unit_overlap: float        # mean per-unit stem overlap with query


def _stem_set(tokens: list[str]) -> frozenset[str]:
    """Return the set of stems for a token list (uses cached stem_token)."""
    return frozenset(stem_token(t) for t in tokens if t)


def _coverage_from_stem_sets(
    query_stems: frozenset[str],
    pack_stem_set: frozenset[str],
) -> float:
    """Fraction of query stems present in the pack stem set.

    Returns 0.0 when the query stem set is empty (no query signal to cover).
    """
    if not query_stems:
        return 0.0
    hit = len(query_stems & pack_stem_set)
    return hit / len(query_stems)


def _entity_tokens(tokens: list[str], raw_query: str) -> list[str]:
    """Return query tokens that look like named entities or numbers."""
    raw_words = raw_query.split()
    entity_set: set[str] = set()
    for word in raw_words:
        clean = re.sub(r"[^A-Za-z0-9]", "", word)
        if _RE_ENTITY_TOKEN.match(clean):
            entity_set.add(clean.lower())
    return [t for t in tokens if t in entity_set or t.isdigit()]


def _date_tokens(raw_query: str) -> list[str]:
    """Return lowercased date-like tokens found in the raw query string."""
    return [m.group(0).lower() for m in _RE_DATE_TOKEN.finditer(raw_query)]


def _count_tokens(text: str) -> int:
    """Rough whitespace token count."""
    return len(text.split())


def _query_stems_breakdown(
    query_str: str,
) -> tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
    """Return (all_stems, content_stems, entity_stems, date_stems) for a query."""
    tokens = get_clean_tokens(query_str)
    all_stems = _stem_set(tokens)
    content_stems = _stem_set([t for t in tokens if t not in _CONTENT_STOPWORDS])
    entity_stems = _stem_set(_entity_tokens(tokens, query_str))
    date_stems = _stem_set(_date_tokens(query_str))
    return all_stems, content_stems, entity_stems, date_stems


def _unit_overlaps(query_stems: frozenset[str], pack_texts: Sequence[str]) -> list[float]:
    """Return per-unit stem overlap fractions with the query."""
    overlaps: list[float] = []
    for unit_text in pack_texts:
        unit_tokens = get_clean_tokens(str(unit_text or ""))
        if not unit_tokens or not query_stems:
            overlaps.append(0.0)
            continue
        unit_stems = _stem_set(unit_tokens)
        overlaps.append(len(query_stems & unit_stems) / len(query_stems))
    return overlaps


def extract(query: str, pack_texts: Sequence[str]) -> FeatureVector:
    """Compute the feature vector for a (query, pack) pair."""
    query_str = str(query or "")
    query_tokens = get_clean_tokens(query_str)
    query_token_count = float(max(len(query_tokens), _MIN_QUERY_TOKENS))

    q_stems, content_stems, entity_stems, date_stems = _query_stems_breakdown(query_str)

    pack_concat = " ".join(str(t) for t in pack_texts)[:_MAX_PACK_CHARS]
    pack_stem_set = _stem_set(get_clean_tokens(pack_concat))
    pack_token_count = float(len(pack_concat.split()))

    overlaps = _unit_overlaps(q_stems, pack_texts)
    max_unit = max(overlaps) if overlaps else 0.0
    mean_unit = sum(overlaps) / len(overlaps) if overlaps else 0.0

    return FeatureVector(
        stem_coverage=_coverage_from_stem_sets(q_stems, pack_stem_set),
        content_word_coverage=_coverage_from_stem_sets(content_stems, pack_stem_set),
        entity_coverage=_coverage_from_stem_sets(entity_stems, pack_stem_set),
        date_coverage=_coverage_from_stem_sets(date_stems, pack_stem_set),
        pack_token_count=pack_token_count,
        pack_query_length_ratio=pack_token_count / query_token_count,
        n_units=float(len(pack_texts)),
        max_unit_overlap=max_unit,
        mean_unit_overlap=mean_unit,
    )


def feature_names() -> list[str]:
    """Return field names in the same order as FeatureVector."""
    return [f.name for f in fields(FeatureVector)]


def to_array(fv: FeatureVector) -> list[float]:
    """Convert FeatureVector to a plain list of floats (field order preserved)."""
    return [getattr(fv, f.name) for f in fields(FeatureVector)]
