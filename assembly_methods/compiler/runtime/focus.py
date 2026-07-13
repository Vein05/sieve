"""Query focus helpers for the compiler runtime."""

from __future__ import annotations

import re
from typing import Any

from ...common import cached_build_profile, stem_token, cached_word_tokenize, cached_pos_tag


_QUERY_FOCUS_SKIP_TOKENS = {
    "about",
    "advice",
    "ago",
    "amount",
    "best",
    "been",
    "current",
    "days",
    "did",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "just",
    "know",
    "many",
    "much",
    "my",
    "name",
    "of",
    "often",
    "previous",
    "the",
    "thinking",
    "time",
    "to",
    "total",
    "want",
    "wanted",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "wondering",
    "with",
}

_PREDICATE_FOCUS_SKIP_TOKENS = {"about", "advice", "been", "did", "does", "how", "just", "many", "much", "what", "when", "where", "which", "who", "wondering"}
_WEAK_QUERY_FOCUS_TOKENS = {
    # Pure function / interrogative words — uninformative for focus alignment
    "about",
    "advice",
    "been",
    "did",
    "does",
    "had",
    "have",
    "help",
    "how",
    "just",
    "know",
    "many",
    "much",
    "think",
    "thought",
    "thinking",
    "want",
    "wanted",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "wondering",
}
# Note: state/temporal markers (current, currently, now, new, old, previous,
# prior, still, initially) were removed from this set — they carry real
# semantic signal when matching evidence about state changes, temporal context,
# or "current X" questions.

_COUNT_LOOKUP_OBJECT_SKIP_TOKENS = {"a", "an", "and", "at", "for", "from", "i", "in", "local", "my", "of", "on", "or", "the", "to"}
_COUNT_LOOKUP_PREDICATE_SKIP_TOKENS = {"be", "do", "does", "did", "have", "has", "had", "is", "was", "were"}


def _stem_token(token: str) -> str:
    return stem_token(token)


def _token_stems(tokens: set[str]) -> set[str]:
    return {_stem_token(token) for token in tokens if token}


_QUERY_FOCUS_SYNONYMS: dict[str, set[str]] = {
    "occupation": {"role", "job", "work", "profession", "career"},
}


def _query_focus_tokens(row: dict[str, Any]) -> set[str]:
    profile = cached_build_profile(str(row.get("query", "")))
    focus_tokens = {
        token
        for token in profile.content_tokens | profile.named_tokens
        if token and token not in _QUERY_FOCUS_SKIP_TOKENS and not token.isdigit()
    }
    expanded = set(focus_tokens)
    for token in focus_tokens:
        expanded.update(_QUERY_FOCUS_SYNONYMS.get(token, set()))
    return expanded


def _predicate_focus_tokens(row: dict[str, Any]) -> set[str]:
    return {token for token in _query_focus_tokens(row) if token not in _PREDICATE_FOCUS_SKIP_TOKENS}


def _count_lookup_object_tokens(row: dict[str, Any]) -> set[str]:
    query = str(row.get("query", "")).lower()
    tokens = cached_word_tokenize(query)
    tags = cached_pos_tag(tuple(tokens))

    found_how_many = False
    object_tokens = set()
    for word, tag in tags:
        if word == "many":
            found_how_many = True
            continue
        if found_how_many and tag.startswith(("NN", "JJ")) and word not in _COUNT_LOOKUP_OBJECT_SKIP_TOKENS:
            stemmed = stem_token(word)
            if stemmed not in _WEAK_QUERY_FOCUS_TOKENS:
                object_tokens.add(stemmed)
    if not object_tokens:
        return {token for token in _query_focus_tokens(row) if token not in _PREDICATE_FOCUS_SKIP_TOKENS}
    return object_tokens


def _count_lookup_predicate_tokens(row: dict[str, Any]) -> set[str]:
    query = str(row.get("query", "")).lower()
    tokens = cached_word_tokenize(query)
    tags = cached_pos_tag(tuple(tokens))
    predicates = set()
    for word, tag in tags:
        if tag.startswith("VB") and word not in _COUNT_LOOKUP_PREDICATE_SKIP_TOKENS:
            stemmed = stem_token(word)
            if stemmed not in _WEAK_QUERY_FOCUS_TOKENS:
                predicates.add(stemmed)
    return predicates
