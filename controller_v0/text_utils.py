"""Text normalization and profiling helpers for rule_v0."""

from __future__ import annotations

import re

from controller_v0.constants import (
    CONCEPT_SPECS,
    DATE_RE,
    NEGATION_PATTERNS,
    PERSONAL_LIFE_TOKENS,
    PREFERENCE_MARKERS,
    PREFERENCE_PATTERNS,
    SENSITIVE_PATTERNS,
    STOPWORDS,
    TEMPORAL_HINTS,
    TIME_RE,
    TOKEN_RE,
    UPDATE_MARKERS,
)
from controller_v0.models import TextProfile


def normalize_text(text: str) -> str:
    text = text.lower()
    replacements = {
        "o'hare": "ohare",
        "children's": "childrens",
        "thank-you": "thank you",
        "pet-friendly": "pet friendly",
        "follow-up": "follow up",
        "usb-c": "usb c",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS}


def contains_pattern(pattern: str, normalized_text: str, token_set: set[str]) -> bool:
    if " " in pattern:
        return pattern in normalized_text
    return pattern in token_set


def parse_date_key(text: str) -> tuple[int, int, int] | None:
    match = DATE_RE.search(text)
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3) or 0)
    return (year, month, day)


import assembly_methods.common as am_common

_EXTRACT_NAMED_SKIP = {
    "Am", "Any", "Are", "Can", "Did", "Do", "How", "Is", "Recent", "User", "Team", "Assistant",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "What", "When", "Which", "Who", "Why", "Would", "Actually", "Please", "Thanks", "I", "I'm"
}

def extract_named_tokens(text: str) -> set[set]:
    # Use SpaCy via shared common utilities for better entity detection
    tokens = am_common.cached_word_tokenize(text)
    tags = am_common.cached_pos_tag(tuple(tokens))
    
    return {
        word.lower()
        for word, tag in tags
        if (tag.startswith("NNP") or (tag == "NN" and word[0].isupper()))
        and word not in _EXTRACT_NAMED_SKIP
        and word.isalnum()
    }


from functools import lru_cache

@lru_cache(maxsize=10000)
def build_profile(text: str, *, use_concept_specs: bool = False) -> TextProfile:
    # Use a single-pass parse for everything (Tokens, Tags, NER)
    nlp_prof = am_common.get_nlp_profile(text)
    
    # Heuristic named token detection using the tags from the parse
    named_tokens = {
        word.lower()
        for word, tag in zip(nlp_prof.tokens, nlp_prof.tags)
        if (tag.startswith("NNP") or (tag == "NN" and word[0].isupper()))
        and word not in _EXTRACT_NAMED_SKIP
        and word.isalnum()
    }

    normalized = normalize_text(text)
    token_set = set(nlp_prof.tokens)
    tokens_normalized = list(nlp_prof.tokens) # Keep original SpaCy tokens
    
    concept_scores: dict[str, float] = {}
    if use_concept_specs:
        for concept, spec in CONCEPT_SPECS.items():
            hits = sum(
                1
                for pattern in spec["patterns"]
                if (pattern in normalized if " " in pattern else pattern in token_set)
            )
            if hits:
                concept_scores[concept] = min(1.0, hits / min(2, len(spec["patterns"])))
                
    has_update_marker = any(token.lower() in UPDATE_MARKERS for token in token_set) or any(
        contains_pattern(marker, normalized, token_set)
        for marker in NEGATION_PATTERNS
    )
    has_time_marker = bool(DATE_RE.search(text) or TIME_RE.search(normalized) or (TEMPORAL_HINTS & token_set))
    is_preference = bool(PREFERENCE_MARKERS & token_set) or any(
        pattern in normalized for pattern in PREFERENCE_PATTERNS
    )
    is_sensitive = any(pattern in normalized for pattern in SENSITIVE_PATTERNS)
    is_personal = is_preference or is_sensitive or any(
        token.lower() in token_set for token in PERSONAL_LIFE_TOKENS
    )
    
    return TextProfile(
        text=text,
        normalized_text=normalized,
        tokens=tokens_normalized,
        token_set=token_set,
        content_tokens={token.lower() for token in token_set if token.lower() not in STOPWORDS},
        concept_scores=concept_scores,
        date_key=parse_date_key(text),
        has_update_marker=has_update_marker,
        has_time_marker=has_time_marker,
        is_preference=is_preference,
        is_personal=is_personal,
        is_sensitive=is_sensitive,
        named_tokens=named_tokens,
    )


def merge_profiles(profiles: list[TextProfile]) -> TextProfile:
    """Efficiently merge multiple profiles into one without re-profiling text."""
    if not profiles:
        return build_profile("")
    if len(profiles) == 1:
        return profiles[0]
        
    # Aggregate sets and dicts
    token_set = set()
    content_tokens = set()
    named_tokens = set()
    tokens = []
    text_parts = []
    norm_parts = []
    
    # Concept scores are aggregated by their raw hits if we wanted perfect parity, 
    # but for speed and since they are usually binary in this controller, we use max()
    concept_scores = {}
    
    has_update = False
    has_time = False
    is_pref = False
    is_pers = False
    is_sens = False
    newest_date = None
    
    for p in profiles:
        token_set.update(p.token_set)
        content_tokens.update(p.content_tokens)
        named_tokens.update(p.named_tokens)
        tokens.extend(p.tokens)
        text_parts.append(p.text)
        norm_parts.append(p.normalized_text)
        
        for c, score in p.concept_scores.items():
            concept_scores[c] = max(concept_scores.get(c, 0.0), score)
            
        has_update |= p.has_update_marker
        has_time |= p.has_time_marker
        is_pref |= p.is_preference
        is_pers |= p.is_personal
        is_sens |= p.is_sensitive
        
        if p.date_key:
            if newest_date is None or p.date_key > newest_date:
                newest_date = p.date_key
                
    return TextProfile(
        text=" ".join(text_parts),
        normalized_text=" ".join(norm_parts),
        tokens=tokens,
        token_set=token_set,
        content_tokens=content_tokens,
        concept_scores=concept_scores,
        date_key=newest_date,
        has_update_marker=has_update,
        has_time_marker=has_time,
        is_preference=is_pref,
        is_personal=is_pers,
        is_sensitive=is_sens,
        named_tokens=named_tokens,
    )
