"""Deterministic conversational memory object construction."""

from __future__ import annotations

from dataclasses import replace
import logging
import re
from typing import Any

from controller.constants import DATE_RE, NEGATION_PATTERNS, TEMPORAL_HINTS, TIME_RE, UPDATE_MARKERS

from shared.nlp import (
    CandidateView,
    canonical_attribute,
    cached_sent_tokenize,
    cached_word_tokenize,
    cached_pos_tag,
)
from evidence.memory_objects import MemoryObject, ProvenanceRecord

from functools import lru_cache


_SPEAKER_RE = re.compile(r"(?:^|[\n|])\s*(?P<speaker>[A-Za-z][\w-]{0,31})\s*:\s*")

# Matches JSON dict repr segments: {'role': 'user', 'content': "...", ...}
# These appear in ~8% of LongMemEval candidate memories instead of the clean
# "speaker: text" format.  We parse them into (speaker, content) pairs so the
# downstream sentence splitter never sees raw JSON.
_JSON_TURN_RE = re.compile(
    r"""\{\s*'role'\s*:\s*'(\w+)'\s*,\s*'content'\s*:\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')\s*(?:,\s*'[^']*'\s*:\s*[^}]*)?\}""",
)

# Matches a role header followed by a content key and opening quote, for
# truncated dicts where the content string was cut off before the closing
# quote.  Almost all LongMemEval JSON candidates (~99%) have a truncated
# assistant turn because the text field is capped at ~2000 chars.
_JSON_TRUNCATED_RE = re.compile(
    r"""\{\s*'role'\s*:\s*'(\w+)'\s*,\s*'content'\s*:\s*(['"])""",
)


def _extract_truncated_content(text: str, quote_char: str) -> str:
    """Extract content from a truncated JSON string starting after the opening quote."""
    # Walk character-by-character respecting escapes
    result: list[str] = []
    escaped = False
    for c in text:
        if escaped:
            if c == 'n':
                result.append('\n')
            elif c == 't':
                result.append('\t')
            elif c in ("'", '"', '\\'):
                result.append(c)
            else:
                result.append(c)
            escaped = False
            continue
        if c == '\\':
            escaped = True
            continue
        if c == quote_char:
            # Reached the closing quote — content isn't actually truncated
            break
        result.append(c)
    return ''.join(result).strip()


def _parse_json_turns(text: str) -> list[tuple[str, str, str]] | None:
    """Try to parse JSON dict repr conversation segments from *text*.

    Returns a list of (prefix, speaker, body) tuples if the text contains
    JSON-formatted turns, or ``None`` if the text is not in that format.
    The prefix is extracted from the session header before the first JSON dict.

    Handles truncated content: when the text is cut off mid-string (common in
    LongMemEval where memory text is capped at ~2000 chars), we extract
    whatever content is available from the truncated turn.
    """
    # Quick guard: only attempt if the text contains the JSON role marker
    if "{'role'" not in text and '{"role"' not in text:
        return None

    # Extract the session header prefix (everything before the first JSON dict)
    first_brace = text.find("{")
    if first_brace < 0:
        return None

    prefix = text[:first_brace].rstrip(" |").strip()

    # Phase 1: Extract all complete JSON turns via the strict regex
    turns: list[tuple[str, str, str]] = []
    matched_spans: list[tuple[int, int]] = []
    for m in _JSON_TURN_RE.finditer(text):
        speaker = m.group(1).strip().lower()
        raw_content = m.group(2)
        # Strip surrounding quotes (single or double)
        if (raw_content.startswith('"') and raw_content.endswith('"')) or \
           (raw_content.startswith("'") and raw_content.endswith("'")):
            raw_content = raw_content[1:-1]
        # Unescape common escapes
        content = raw_content.replace("\\'", "'").replace('\\"', '"').replace("\\n", "\n").strip()
        if content:
            turns.append((prefix, speaker, content))
            matched_spans.append((m.start(), m.end()))

    # Phase 2: Find truncated turns that the strict regex missed
    for m in _JSON_TRUNCATED_RE.finditer(text):
        # Skip if this position was already captured by the strict regex
        if any(s <= m.start() < e for s, e in matched_spans):
            continue
        speaker = m.group(1).strip().lower()
        quote_char = m.group(2)
        # Content starts right after the opening quote
        content_start = m.end()
        content = _extract_truncated_content(text[content_start:], quote_char)
        # Strip JSON artifacts that might remain at the end of truncated text
        content = re.sub(r",\s*'[^']*'\s*:\s*\S*$", "", content).strip()
        if content:
            turns.append((prefix, speaker, content))

    return turns if turns else None
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[\"'(\[])?[A-Z0-9])")
_NUMERIC_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_ENTITY_KEY_RE = re.compile(r"\b(?:my|the|our)\s+([a-z][a-z0-9_-]{2,})\b")
_ATTRIBUTE_KEY_RE = re.compile(r"\b(color|status|brand|breed|price|cost|date|time|version|size|name|model|route|plan|role)\b")
_EVENT_KEY_RE = re.compile(r"\b(met|moved|arrived|left|joined|attended|bought|sold|started|finished|visited|traveled|travelled|called|emailed|booked|scheduled|replaced|participated)\b")

_COMPARISON_MARKERS = (
    " compared ",
    " versus ",
    " vs ",
    " more than ",
    " less than ",
    " higher than ",
    " lower than ",
)

_PREFERENCE_MARKERS = (
    " prefer ",
    " preference ",
    " favorite ",
    " favourite ",
    " like ",
    " love ",
    " dislike ",
    " hate ",
)

_STATE_MARKERS = (
    " is ",
    " are ",
    " was ",
    " were ",
    " currently ",
    " current ",
    " now ",
    " still ",
)

_UPDATE_MARKERS = (
    " switched ",
    " changed ",
    " update ",
    " updated ",
    " instead of ",
    " no longer ",
    " used to ",
)

_ID_SANITIZE_RE = re.compile(r"[^a-z0-9]+")
_EVENT_SIGNATURE_SKIP = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "i",
    "in",
    "into",
    "it",
    "my",
    "of",
    "on",
    "our",
    "the",
    "to",
    "user",
}
_VERB_OBJECT_RE = re.compile(
    r"\b(?:met|moved|arrived|left|joined|attended|bought|sold|started|finished|visited|"
    r"traveled|travelled|called|emailed|booked|scheduled|replaced|participated)\b"
    r"(?:\s+(?:to|with|at|for|in|on))?\s+(?:an?|the|my|our)?\s*([a-z][a-z0-9_-]{2,})\b"
)
_ENTITY_STOP_TOKENS = {
    "a",
    "after",
    "am",
    "and",
    "are",
    "as",
    "at",
    "because",
    "before",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "my",
    "now",
    "of",
    "on",
    "or",
    "our",
    "since",
    "than",
    "that",
    "the",
    "to",
    "was",
    "were",
    "while",
    "with",
}
_ENTITY_VERB_STOPS = {
    "am",
    "are",
    "arrived",
    "attended",
    "bought",
    "called",
    "changed",
    "complete",
    "completed",
    "emailed",
    "finished",
    "get",
    "gets",
    "getting",
    "go",
    "got",
    "graduated",
    "is",
    "joined",
    "left",
    "live",
    "lived",
    "lives",
    "met",
    "moved",
    "participated",
    "repaired",
    "replaced",
    "scheduled",
    "serviced",
    "started",
    "stayed",
    "study",
    "studied",
    "take",
    "takes",
    "took",
    "travelled",
    "traveled",
    "visited",
    "was",
    "went",
    "were",
}
_LOCATION_CUE_RE = re.compile(r"\bwhere\b", re.IGNORECASE)
_VALUE_CLEAN_RE = re.compile(r"\s+")
_LOCATION_VALUE_PATTERNS = (
    re.compile(
        r"\b(?:live|lives|lived|reside|resides|resided|move|moves|moved|relocate|relocated|"
        r"stay|stays|stayed|based|located)\b(?:[^.!?;|]{0,40})?\b(?:to|in|at|near|inside|within|under|on)\s+(?P<value>[^.!?;|]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:attend|attended|complete|completed|finish|finished|get|getting|got|go|goes|going|went|"
        r"graduate|graduated|repair|repaired|service|serviced|study|studied|take|takes|took|visit|visited|"
        r"volunteer|volunteered|worship|worshipped)\b(?:[^.!?;|]{0,60})?\b(?:at|in|to|under|inside|on)\s+(?P<value>[^.!?;|]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:trip|vacation|visit|journey|flight|concert|class|classes|course|degree|wedding|party|"
        r"festival|meeting|conference)\b(?:[^.!?;|]{0,40})?\b(?:at|in|to|under|inside|on)\s+(?P<value>[^.!?;|]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:go|goes|going|went|come|came|head|headed|return|returned|make it|made it)\b"
        r"(?:[^.!?;|]{0,40})?\b(?:to|at|in|on)\s+(?P<value>[^.!?;|]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:service|serviced|repair|repaired|fix|fixed)\b(?:[^.!?;|]{0,40})?\b(?:at|from|on)\s+(?P<value>[^.!?;|]+)",
        re.IGNORECASE,
    ),
)
_LOCATION_HEAD_TOKENS = {
    "bed",
    "bedroom",
    "campus",
    "center",
    "centre",
    "church",
    "city",
    "closet",
    "college",
    "downtown",
    "hall",
    "home",
    "house",
    "main",
    "museum",
    "neighborhood",
    "neighbourhood",
    "office",
    "park",
    "paris",
    "shop",
    "shoe",
    "store",
    "street",
    "studio",
    "suburbs",
    "town",
    "university",
}
_TEMPORAL_LOCATION_BLOCKLIST = {
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "today",
    "tomorrow",
    "yesterday",
}
_NON_LOCATION_VALUE_TOKENS = {
    "bandwidth",
    "gbps",
    "internet",
    "mbps",
    "plan",
    "rate",
    "speed",
    "subscription",
    "tier",
}
_DURATION_VALUE_PATTERNS = (
    re.compile(
        r"\b(?:around|about|approximately|roughly)?\s*(?P<number>\d+(?:\.\d+)?)\s+"
        r"(?P<unit>hours?|minutes?|days?|weeks?|months?|years?)(?:\s+each\s+way)?\b",
        re.IGNORECASE,
    ),
)
# Supplemental pattern for word-number durations ("two weeks", "a few months").
# Kept separate because the number group is a word, not a float.
_DURATION_WORD_NUMBER_RE = re.compile(
    r"\b(?P<number>a\s+few|a\s+couple(?:\s+of)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|several|few|couple)\s+"
    r"(?P<unit>hours?|minutes?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_SIZE_VALUE_PATTERNS = (
    re.compile(r"\b(?P<value>\d+(?:\.\d+)?(?:-|\s)?inch)\b", re.IGNORECASE),
)
_PERCENT_VALUE_PATTERNS = (
    re.compile(r"\b(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>%|percent)\b", re.IGNORECASE),
)
_RATIO_VALUE_PATTERNS = (
    re.compile(r"\b(?P<value>\d+(?::\d+)+)\b", re.IGNORECASE),
)
_TIME_VALUE_RE = re.compile(r"\b(?P<value>\d{1,2}(?::\d{2})?\s*(?:am|pm))\b", re.IGNORECASE)
_FAVORITE_NAME_RE = re.compile(r"\bmy\s+favorite\s+(?P<value>[^,.!?;]{1,80})", re.IGNORECASE)
_COLOR_VALUE_RE = re.compile(
    r"\b(?:(?:a|an)\s+)?(?P<value>(?:lighter|darker|deep|bright|soft|muted|pale|warm|cool)\s+shade\s+of\s+"
    r"(?:gray|grey|blue|green|red|yellow|black|white|brown|purple|pink|orange|beige|navy|teal|maroon|gold|silver)|"
    r"(?:gray|grey|blue|green|red|yellow|black|white|brown|purple|pink|orange|beige|navy|teal|maroon|gold|silver))\b",
    re.IGNORECASE,
)
_DIRECT_OBJECT_VALUE_VERBS = {"bake", "bring", "buy", "cook", "make", "order", "prepare", "purchase", "try"}
_DIRECT_OBJECT_TRAILING_PREPS = frozenset({"for", "with", "from", "at", "because", "since", "after", "before"})
_PRODUCT_VALUE_VERBS = frozenset({"buy", "get", "order", "pick", "purchase", "replace", "swap", "use"})
_TRAILING_VALUE_SPLIT_RE = re.compile(
    r"\b(?:and|but|because|which|who|that|where|when|while)\b",
    re.IGNORECASE,
)
_DIRECT_OBJECT_VALUE_CLEANUPS = (
    re.compile(r"\brecipe\b$", re.IGNORECASE),
    re.compile(r"\b(?:service|subscription|plan)\b$", re.IGNORECASE),
)
_LOCATION_MODIFIER_TOKENS = frozenset({"early", "late", "mid"})
_NON_LOCATION_DEVICE_TOKENS = frozenset({
    "android",
    "app",
    "apps",
    "iphone",
    "ipad",
    "kindle",
    "laptop",
    "macbook",
    "netflix",
    "nintendo",
    "phone",
    "playstation",
    "ps4",
    "ps5",
    "roku",
    "service",
    "spotify",
    "steam",
    "tablet",
    "xbox",
    "youtube",
})
_OCCUPATION_VALUE_PATTERNS = (
    re.compile(
        r"\b(?:my|our)\s+(?:current|previous|old|new)?\s*(?:occupation|job|role|profession)\s+(?:is|was|as)\s+(?P<value>[^,.!?;|]+)",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:work(?:ed|ing)?|employed)\s+as\s+(?P<value>[^,.!?;|]+)", re.IGNORECASE),
    re.compile(r"\b(?:role|job|profession)\s+as\s+(?P<value>[^,.!?;|]+)", re.IGNORECASE),
)
_NAME_VALUE_PATTERNS = (
    re.compile(r"\b(?:called|named)\s+(?P<value>[A-Z0-9][^,.!?;|]+)", re.IGNORECASE),
    re.compile(r"\bproduction of\s+(?P<value>[A-Z0-9][^,.!?;|]+)", re.IGNORECASE),
)
_BREED_VALUE_PATTERNS = (
    re.compile(
        r"\b(?:dog|puppy|cat|kitten)\s+(?:is|was)\s+(?:a|an)\s+(?P<value>[^,.!?;|]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmy\s+(?:dog|puppy|cat|kitten)\s+(?:breed\s+is|is)\s+(?P<value>[^,.!?;|]+)",
        re.IGNORECASE,
    ),
)
_SPEED_VALUE_PATTERNS = (
    re.compile(
        r"\b(?:(?:internet|download|upload)\s+)?speed\s+(?:is|was|of)?\s*(?P<value>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>mbps|gbps)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<value>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>mbps|gbps)\s+(?:internet\s+)?(?:plan|speed|tier)\b",
        re.IGNORECASE,
    ),
)
_TITLE_CONTEXT_RE = re.compile(
    r"\s+\b(?:at|in|on)\b\s+(?:the\s+)?"
    r"(?:local|community|school|city|town|downtown|museum|theater|theatre|center|centre|hall|church|park|store|shop|market|festival|conference|campus)\b.*$",
    re.IGNORECASE,
)
_POSSESSIVE_ENTITY_RE = re.compile(
    r"\b(?:my|our|the)\s+(?P<entity>[a-z][a-z0-9_-]*(?:\s+[a-z][a-z0-9_-]*){0,2})\s+"
    r"(?:is|are|was|were|lives|live|lived|moved|move|migrate|migrated|stays|stay|stayed|"
    r"work|works|worked|study|studies|studied|attend|attended|go|goes|went|got|uses|use|used)\b"
)
_LEADING_DETERMINER_RE = re.compile(r"^(?:a|an|the|my|our|your)\s+", re.IGNORECASE)
_PREDICATE_STATE_ATTRIBUTE_MAP = {
    "read": "reading",
    "watch": "watching",
    "study": "studying",
    "use": "using",
}
_WORK_LOCATION_PREPOSITIONS = {"at", "in", "for", "with"}
_LOCATION_PREPOSITIONS = {"at", "in", "near", "inside", "within", "on", "to"}
_BRANDABLE_PRODUCT_HEADS = {
    "shampoo",
    "conditioner",
    "soap",
    "lotion",
    "perfume",
    "headphones",
    "earbuds",
    "phone",
    "laptop",
    "tablet",
    "coffee",
    "tea",
    "sauce",
    "snack",
    "detergent",
}


@lru_cache(maxsize=1)
def _get_spacy_nlp():
    import spacy  # deferred: thinc pulls torch at module level; delay until first parse
    return spacy.load("en_core_web_sm", disable=["ner"])


@lru_cache(maxsize=10000)
def _cached_spacy_doc(text: str):
    """LRU-cached spaCy parse with disk persistence.

    Layer 1: in-process ``@lru_cache`` (fast, lost on restart).
    Layer 2: SQLite disk cache via ``nlp_disk_cache`` (survives restarts).
    Layer 3: actual ``nlp(text)`` call on full miss.
    """
    from shared.perf import get_counters
    from shared.cache import disk_get, disk_put, text_key

    nlp = _get_spacy_nlp()
    _key = text_key(str(text or ""))

    cached_bytes = disk_get("spacy_doc", _key)
    if cached_bytes is not None:
        from spacy.tokens import Doc
        try:
            doc = Doc(nlp.vocab).from_bytes(cached_bytes)
            get_counters().spacy_parses_cache_hit += 1
            return doc
        except Exception:
            pass  # stale/corrupt — fall through to fresh parse

    get_counters().spacy_parses_fresh += 1
    doc = nlp(str(text or ""))
    try:
        disk_put("spacy_doc", _key, doc.to_bytes())
    except Exception:
        pass
    return doc


def _noun_chunk_for_token(doc, token):
    for chunk in doc.noun_chunks:
        if chunk.start <= token.i < chunk.end:
            return chunk
    return None


def _chunk_text_for_token(doc, token) -> str:
    chunk = _noun_chunk_for_token(doc, token)
    if token.pos_ == "PROPN":
        subtree_tokens = [piece.text for piece in token.subtree if piece.dep_ != "punct"]
        subtree_text = " ".join(subtree_tokens).strip()
        if subtree_text:
            return subtree_text
    return str(chunk.text if chunk is not None else token.text).strip()


def _recover_exact_surface(text: str, candidates: list[str] | tuple[str, ...]) -> str | None:
    source_text = str(text or "")
    normalized_candidates = [
        _normalize_surface_form(str(candidate).strip())
        for candidate in candidates
        if str(candidate).strip()
    ]
    if not source_text or not normalized_candidates:
        return None

    lowered_source = source_text.lower()
    spans: list[tuple[int, int]] = []
    for candidate in normalized_candidates:
        lowered_candidate = candidate.lower()
        start = lowered_source.find(lowered_candidate)
        if start >= 0:
            spans.append((start, start + len(candidate)))
    if spans:
        start, end = max(spans, key=lambda item: (item[1] - item[0], -item[0]))
        return source_text[start:end].strip()

    # Only spin up flashtext for larger candidate sets where Aho-Corasick pays off.
    if len(normalized_candidates) > 6:
        try:
            from flashtext import KeywordProcessor
            processor = KeywordProcessor(case_sensitive=False)
            for candidate in normalized_candidates:
                processor.add_keyword(candidate)
            matches = processor.extract_keywords(source_text, span_info=True)
            if matches:
                best = max(matches, key=lambda item: (item[2] - item[1], -item[1]))
                return source_text[best[1]:best[2]].strip()
        except Exception:
            pass

    return None


def _normalize_surface_form(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"\s+'s\b", "'s", cleaned)
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"(?<=\w)\s*-\s*(?=\w)", "-", cleaned)
    return cleaned.strip()


def _recover_exact_chunk_text(doc, token) -> str:
    candidate = _chunk_text_for_token(doc, token)
    recovered = _recover_exact_surface(doc.text, [candidate])
    return _normalize_surface_form(recovered or candidate)


@lru_cache(maxsize=1)
def _ctparse_fn():
    logging.getLogger("ctparse.ctparse").setLevel(logging.ERROR)
    from ctparse import ctparse

    return ctparse


def _dedupe_preserve_order(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _split_speaker_turns(text: str) -> list[tuple[str, str, str]]:
    stripped = str(text or "").strip()
    if not stripped:
        return []

    # Try JSON dict repr parsing first (handles ~8% of LongMemEval memories)
    json_turns = _parse_json_turns(stripped)
    if json_turns is not None:
        return json_turns

    matches = list(_SPEAKER_RE.finditer(stripped))
    if not matches:
        return [("", "unknown", stripped)]

    turns: list[tuple[str, str, str]] = []
    segment_start = 0
    for index, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(stripped)
        prefix = stripped[segment_start:match.start()].strip()
        speaker = match.group("speaker").strip().lower()
        body = stripped[body_start:body_end].strip()
        if body:
            turns.append((prefix, speaker, body))
        segment_start = body_end

    if not turns:
        return [("", "unknown", stripped)]
    return turns


def _split_sentences(text: str) -> list[str]:
    return cached_sent_tokenize(str(text or ""))


def _render_span(prefix: str, speaker: str, sentence: str) -> str:
    parts: list[str] = []
    if prefix.strip():
        parts.append(prefix.strip())
    if speaker and speaker != "unknown":
        parts.append(f"{speaker}:")
    if sentence.strip():
        parts.append(sentence.strip())
    return " ".join(parts).strip()


def _contains_any(normalized: str, markers: tuple[str, ...]) -> bool:
    padded = f" {normalized} "
    return any(marker in padded for marker in markers)


def _extract_value_number(sentence: str) -> tuple[float | None, str | None]:
    for match in _NUMERIC_RE.finditer(sentence):
        token = match.group(0)
        start, end = match.span()
        left_char = sentence[start - 1] if start > 0 else ""
        right_char = sentence[end] if end < len(sentence) else ""
        if left_char == "/" or right_char == "/":
            # Date fragments such as 2023/05/01 should not become numeric facts.
            continue
        cleaned = token.replace("$", "").replace(",", "").replace("%", "")
        try:
            value = float(cleaned)
        except Exception:
            continue
        if token.startswith("$"):
            return value, "usd"
        if token.endswith("%"):
            return value, "percent"
        return value, None
    return None, None


def _extract_entity_key(normalized_sentence: str) -> str | None:
    possessive_match = _POSSESSIVE_ENTITY_RE.search(normalized_sentence)
    if possessive_match:
        matched = _clean_entity_phrase(possessive_match.group("entity"))
        if matched:
            return matched

    # Use cached tokens for better noun phrase detection in sentences like "my black coffee creamer..."
    tokens = cached_word_tokenize(normalized_sentence)
    tags = cached_pos_tag(tuple(tokens))

    # Strategy 1: Find phrases after possessives/determiners (my, the, our)
    for i, (word, tag) in enumerate(tags):
        if word in {"my", "our", "the"}:
            phrase = []
            for j in range(i + 1, len(tags)):
                w, t = tags[j]
                if t.startswith(("NN", "JJ")) or t == "CD":
                    phrase.append(w)
                elif phrase: # Break on first non-noun/adj once we started
                    break
            if phrase:
                return _clean_entity_phrase(" ".join(phrase))
    
    # Strategy 2: Use event marker verbs as object anchors if found
    for i, (word, tag) in enumerate(tags):
        # Using the same verb set as before but checking POS
        if word in _ENTITY_VERB_STOPS and i + 1 < len(tags):
            # Capture the next noun phrase
            phrase = []
            for j in range(i + 1, len(tags)):
                w, t = tags[j]
                if t.startswith(("NN", "JJ")) or t == "CD":
                    phrase.append(w)
                elif t in ("DT", "PRP$") and not phrase:
                    continue
                elif phrase:
                    break
            if phrase:
                return _clean_entity_phrase(" ".join(phrase))

    # Fallback to the first noun/adjective encountered
    fallback = []
    for word, tag in tags:
        if tag.startswith(("NN", "JJ")):
            fallback.append(word)
        elif fallback:
            break
    if fallback:
        return _clean_entity_phrase(" ".join(fallback))
    
    return None


def _clean_entity_phrase(text: str) -> str | None:
    tokens = [token for token in str(text or "").strip().split() if token and token not in _ENTITY_STOP_TOKENS]
    while tokens and tokens[-1] in {"i", "it", "they", "them", "this", "that"}:
        tokens.pop()
    cleaned = " ".join(tokens).strip()
    return cleaned or None


def _should_trim_after_comma(trailing_text: str) -> bool:
    trailing = str(trailing_text or "").strip()
    if not trailing:
        return False
    lowered = trailing.lower()
    if lowered.startswith(("it ", "they ", "he ", "she ", "which ", "who ", "that ", "because ", "but ", "and ")):
        return True
    return trailing[:1].islower()


def _clean_value_phrase(text: str) -> str:
    cleaned = str(text or "").strip()
    if "," in cleaned:
        left, right = re.split(r"\s*,\s*", cleaned, maxsplit=1)
        if _should_trim_after_comma(right):
            cleaned = left
    cleaned = cleaned.strip(".,;:!?")
    cleaned = re.sub(r"^(?:the\s+)?(?:place|spot|venue)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:that|this)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:again|now|today|currently|right now|recently|last week|last month)\b$", "", cleaned, flags=re.IGNORECASE).strip()
    if re.match(r"^the\s+[A-Z]", cleaned):
        cleaned = re.sub(r"^the\s+", "", cleaned)
    cleaned = _VALUE_CLEAN_RE.sub(" ", cleaned)
    return cleaned.strip()


def _clean_attribute_value(text: str) -> str:
    value = _clean_value_phrase(text)
    if not value:
        return ""
    value = _TRAILING_VALUE_SPLIT_RE.split(value, maxsplit=1)[0].strip(" .,;:!?")
    value = re.sub(r"^(?:a|an)\s+", "", value, flags=re.IGNORECASE)
    for pattern in _DIRECT_OBJECT_VALUE_CLEANUPS:
        value = pattern.sub("", value).strip(" .,;:!?")
    return value.strip()


def _clean_frame_value(text: str) -> str:
    value = _clean_attribute_value(text)
    if not value:
        return ""
    value = _LEADING_DETERMINER_RE.sub("", value).strip()
    return value


def _trim_title_context(value: str) -> str:
    return _TITLE_CONTEXT_RE.sub("", str(value or "")).strip()


def _looks_like_location_value(value_text: str) -> bool:
    cleaned = _clean_value_phrase(value_text)
    if not cleaned:
        return False
    if '"' in cleaned or "“" in cleaned or "”" in cleaned:
        return False
    normalized = " ".join(re.findall(r"\b[a-z0-9]+\b", cleaned.lower()))
    if not normalized:
        return False
    tokens = normalized.split()
    if len(tokens) > 8:
        return False
    if tokens and all(token in _TEMPORAL_LOCATION_BLOCKLIST for token in tokens):
        return False
    if tokens and all(token in (_TEMPORAL_LOCATION_BLOCKLIST | _LOCATION_MODIFIER_TOKENS) for token in tokens):
        return False
    if any(token in _NON_LOCATION_VALUE_TOKENS for token in tokens):
        return False
    if any(token in _NON_LOCATION_DEVICE_TOKENS for token in tokens):
        return False
    if any(token in _LOCATION_HEAD_TOKENS for token in tokens):
        return True
    if any(char.isupper() for char in cleaned):
        return True
    return False


_LOCATION_CONTEXT_BREAK_RE = re.compile(
    r"\s+(?:with|and|but|because|since|before|after|during|while|although|though|by|from|for)\b",
    re.IGNORECASE,
)
_QUESTION_OPENERS_RE = re.compile(
    r"^\s*(?:what|when|where|who|which|why|how|can|could|would|should|do|does|did|is|are|was|were|am|have|has|had)\b",
    re.IGNORECASE,
)
_ADVICE_MARKERS_RE = re.compile(
    r"\b(?:for example|consider|recommend|suggest|tips?|resources?|try|use)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_SIGNAL_RE = re.compile(
    r"\b(?:i|i'm|i've|i'd|me|my|mine|we|we're|we've|our|ours)\b",
    re.IGNORECASE,
)
_TEMPORAL_VALUE_RE = re.compile(
    r"\b(?:"
    r"today|tomorrow|yesterday|tonight|morning|afternoon|evening|night|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"last|next|this|ago|back|before|after|during|since|until|"
    r"valentine|valentine'?s|christmas|thanksgiving|easter|halloween|birthday|anniversary|"
    r"day|week|month|year"
    r")\b",
    re.IGNORECASE,
)
_QUOTED_SPAN_RES = (
    re.compile(r'"([^"]+)"'),
    re.compile(r"“([^”]+)”"),
)


def _trim_to_location_head(text: str) -> str:
    """Trim a raw location candidate to its head noun phrase.

    Regex patterns over-capture context clauses (e.g. "Seattle with her partner
    last summer"). This function cuts at the first preposition or temporal word
    that signals a context phrase rather than the location itself.
    """
    # Split at context-signalling prepositions
    m = _LOCATION_CONTEXT_BREAK_RE.search(text)
    if m:
        text = text[: m.start()].strip()
    # Also strip trailing temporal references ("last Sunday", "last summer", "this week")
    _TEMPORAL_ADVERBS = {"last", "this", "next", "every", "each", "ago", "recently", "yesterday", "today", "tomorrow"}
    tokens = text.split()
    trimmed: list[str] = []
    for tok in tokens:
        if tok.lower() in _TEMPORAL_LOCATION_BLOCKLIST or tok.lower() in _TEMPORAL_ADVERBS:
            break
        trimmed.append(tok)
    if trimmed:
        text = " ".join(trimmed).strip(" .,;")
    return text.strip()


def _looks_like_temporal_value(value_text: str) -> bool:
    cleaned = _clean_value_phrase(value_text)
    if not cleaned:
        return False
    normalized = " ".join(re.findall(r"\b[a-z0-9']+\b", cleaned.lower()))
    if not normalized:
        return False
    if not _TEMPORAL_VALUE_RE.search(normalized):
        return False
    tokens = [token for token in normalized.split() if token]
    non_temporal = [
        token
        for token in tokens
        if token not in _TEMPORAL_LOCATION_BLOCKLIST
        and not _TEMPORAL_VALUE_RE.fullmatch(token)
        and token not in {"on", "in", "at", "back"}
    ]
    return len(non_temporal) <= 1


def _candidate_within_quoted_text(sentence: str, candidate: str) -> bool:
    source = str(sentence or "")
    probe = str(candidate or "").strip()
    if not source or not probe:
        return False
    lowered_probe = probe.lower()
    for pattern in _QUOTED_SPAN_RES:
        for match in pattern.finditer(source):
            quoted = str(match.group(1) or "")
            if lowered_probe in quoted.lower():
                return True
    return False


def _skip_supplemental_fact_extraction(*, sentence: str, speaker: str) -> bool:
    text = str(sentence or "").strip()
    lowered_speaker = str(speaker or "").strip().lower()
    if not text:
        return True
    prefix = text.split("?", 1)[0].strip()
    if _QUESTION_OPENERS_RE.match(text):
        return True
    if "?" in text and not _FIRST_PERSON_SIGNAL_RE.search(prefix):
        return True
    if lowered_speaker == "assistant" and "for example" in text.lower():
        return True
    if lowered_speaker == "assistant" and _ADVICE_MARKERS_RE.search(text) and not re.search(r"\b(?:i|i'm|i've|i'd|me|my|mine)\b", text, re.IGNORECASE):
        return True
    return False


def _is_low_authority_sentence(*, sentence: str, speaker: str) -> bool:
    text = str(sentence or "").strip()
    lowered_speaker = str(speaker or "").strip().lower()
    if not text:
        return False
    prefix = text.split("?", 1)[0].strip()
    if _QUESTION_OPENERS_RE.match(text):
        return True
    if "?" in text and not _FIRST_PERSON_SIGNAL_RE.search(prefix):
        return True
    if lowered_speaker == "assistant" and "for example" in text.lower():
        return True
    if lowered_speaker == "assistant" and _ADVICE_MARKERS_RE.search(text) and not re.search(r"\b(?:i|i'm|i've|i'd|me|my|mine)\b", text, re.IGNORECASE):
        return True
    return False


def _is_low_authority_assistant_sentence(*, sentence: str, speaker: str) -> bool:
    return _is_low_authority_sentence(sentence=sentence, speaker=speaker) and str(speaker or "").strip().lower() == "assistant"


def _extract_location_value(sentence: str) -> str | None:
    for pattern in _LOCATION_VALUE_PATTERNS:
        match = pattern.search(sentence)
        if not match:
            continue
        raw = _clean_value_phrase(match.group("value"))
        candidate = _trim_to_location_head(raw) if raw else raw
        if (
            candidate
            and not _candidate_within_quoted_text(sentence, candidate)
            and not _looks_like_temporal_value(candidate)
            and _looks_like_location_value(candidate)
        ):
            return candidate

    doc = _cached_spacy_doc(sentence)
    for token in doc:
        if token.text.lower() not in _LOCATION_PREPOSITIONS:
            continue
        pobj = next((child for child in token.children if child.dep_ in {"pobj", "obj"}), None)
        if pobj is None:
            continue
        candidate = _trim_to_location_head(_recover_exact_chunk_text(doc, pobj))
        if (
            candidate
            and not _candidate_within_quoted_text(sentence, candidate)
            and not _looks_like_temporal_value(candidate)
            and _looks_like_location_value(candidate)
        ):
            return _clean_value_phrase(candidate)

    # Use cached tokens to find "at/in/to" followed by a noun phrase.
    # Stop the phrase at temporal words (days of week, months) so we don't
    # return "Target last Sunday" when "Target" is the location.
    tokens = cached_word_tokenize(sentence)
    tags = cached_pos_tag(tuple(tokens))

    for i, (word, tag) in enumerate(tags):
        if word.lower() in ("at", "in", "to", "inside", "near", "on", "from") and i + 1 < len(tags):
            if word.lower() == "to" and i + 1 < len(tags) and tags[i + 1][1].startswith("VB"):
                continue
            phrase = []
            for j in range(i + 1, len(tags)):
                w, t = tags[j]
                # Temporal or context-signalling words terminate the location phrase
                if w.lower() in _TEMPORAL_LOCATION_BLOCKLIST or w.lower() in ("last", "this", "next", "with", "and", "but"):
                    break
                if t.startswith(("NN", "JJ")) or t == "CD" or t == "DT":
                    phrase.append(w)
                elif phrase:
                    break
            if phrase:
                phrase_text = " ".join(phrase).strip()
                if (
                    not _candidate_within_quoted_text(sentence, phrase_text)
                    and not _looks_like_temporal_value(phrase_text)
                    and _looks_like_location_value(phrase_text)
                ):
                    return _clean_value_phrase(phrase_text)

    return None


def _extract_duration_value(sentence: str) -> tuple[float, str, str] | None:
    phrase_match = re.search(
        r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|few|several|a\s+few|a\s+couple(?:\s+of)?|couple)\s+"
        r"(?:hours?|minutes?|days?|weeks?|months?|years?)(?:\s+each\s+way)?\b",
        sentence,
        re.IGNORECASE,
    )
    if phrase_match:
        suffix_after = sentence[phrase_match.end(): phrase_match.end() + 10].lower().strip()
        if not suffix_after.startswith("ago"):
            try:
                phrase = _normalize_surface_form(phrase_match.group(0).strip())
                parsed = _ctparse_fn()(phrase)
                resolution = getattr(parsed, "resolution", None)
                resolution_text = str(resolution or "")
                if resolution is not None and any(unit in resolution_text.lower() for unit in ("minute", "hour", "day", "week", "month", "year")):
                    numeric = re.search(r"\d+(?:\.\d+)?", phrase)
                    if numeric:
                        return float(numeric.group(0)), resolution_text.split()[-1].lower(), phrase
                    number_word = re.sub(r"\s+", " ", phrase).strip().lower()
                    return -1.0, resolution_text.split()[-1].lower(), number_word
            except Exception:
                pass

    for pattern in _DURATION_VALUE_PATTERNS:
        match = pattern.search(sentence)
        if not match:
            continue
        suffix_after = sentence[match.end(): match.end() + 10].lower().strip()
        if suffix_after.startswith("ago"):
            continue
        raw_number = str(match.group("number") or "").replace(",", "").strip()
        unit = str(match.group("unit") or "").strip().lower()
        try:
            value = float(raw_number)
        except ValueError:
            continue
        label = f"{int(value) if value.is_integer() else value} {unit}"
        matched_text = match.group(0).lower()
        suffix = sentence[match.end(): match.end() + 12].lower()
        if "each way" in matched_text or "each way" in suffix:
            label = f"{label} each way"
        return value, unit, label
    # Fallback: word-number durations ("two weeks", "a few months").
    # Skips relative-temporal phrases like "a few months ago".
    m = _DURATION_WORD_NUMBER_RE.search(sentence)
    if m:
        suffix_after = sentence[m.end(): m.end() + 10].lower().strip()
        if not suffix_after.startswith("ago"):
            number_word = re.sub(r"\s+", " ", m.group("number")).strip().lower()
            unit = m.group("unit").strip().lower()
            label = f"{number_word} {unit}"
            return -1.0, unit, label
    return None


def _extract_time_value(sentence: str) -> str | None:
    match = _TIME_VALUE_RE.search(sentence)
    if not match:
        return None
    candidate = match.group("value").strip()
    try:
        parsed = _ctparse_fn()(candidate)
        if getattr(parsed, "resolution", None) is not None:
            return candidate
    except Exception:
        pass
    return candidate


def _extract_size_value(sentence: str) -> str | None:
    for pattern in _SIZE_VALUE_PATTERNS:
        match = pattern.search(sentence)
        if not match:
            continue
        value = _clean_attribute_value(match.group("value"))
        if value:
            return value
    return None


def _extract_percent_value(sentence: str) -> tuple[float, str, str] | None:
    for pattern in _PERCENT_VALUE_PATTERNS:
        match = pattern.search(sentence)
        if not match:
            continue
        raw_number = str(match.group("number") or "").replace(",", "").strip()
        unit = str(match.group("unit") or "").strip().lower()
        try:
            value = float(raw_number)
        except ValueError:
            continue
        label = f"{int(value) if value.is_integer() else value}%"
        return value, "percent" if unit != "%" else "%", label
    return None


def _extract_ratio_value(sentence: str) -> str | None:
    for pattern in _RATIO_VALUE_PATTERNS:
        match = pattern.search(sentence)
        if not match:
            continue
        value = _clean_attribute_value(match.group("value"))
        if value:
            return value
    return None


def _extract_color_value(sentence: str) -> str | None:
    match = _COLOR_VALUE_RE.search(sentence)
    if not match:
        return None
    value = _clean_attribute_value(match.group("value"))
    if not value:
        return None
    if not re.match(r"^(?:a|an)\s+", value, re.IGNORECASE):
        return f"a {value}" if "shade of" in value.lower() else value
    return value


_PRONOUN_TOKENS = frozenset({
    "it", "this", "that", "them", "those", "these", "something", "anything",
    "everything", "nothing", "one", "ones",
})


def _extract_direct_object_value(sentence: str) -> str | None:
    doc = _cached_spacy_doc(sentence)
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma not in (_DIRECT_OBJECT_VALUE_VERBS | _PRODUCT_VALUE_VERBS):
            continue
        obj = next((child for child in token.children if child.dep_ in {"dobj", "obj", "attr"}), None)
        if obj is None or obj.pos_ == "PRON" or obj.text.lower() in _PRONOUN_TOKENS:
            prep = next((child for child in token.children if child.dep_ == "prep" and child.text.lower() == "with"), None)
            if prep is not None:
                obj = next((child for child in prep.children if child.dep_ in {"pobj", "obj"}), None)
        if obj is None:
            continue
        if obj.pos_ == "PRON" or obj.text.lower() in _PRONOUN_TOKENS:
            continue
        candidate = _recover_exact_chunk_text(doc, obj)
        if not candidate:
            pieces: list[str] = []
            for piece in obj.subtree:
                if piece.dep_ == "punct":
                    continue
                if piece.dep_ == "prep" and piece.text.lower() in _DIRECT_OBJECT_TRAILING_PREPS:
                    break
                pieces.append(piece.text)
            candidate = " ".join(pieces)
        value = _normalize_surface_form(_clean_attribute_value(candidate))
        if value:
            return value
    return None


def _extract_compact_state_value(sentence: str) -> tuple[str | None, str] | None:
    doc = _cached_spacy_doc(sentence)
    for token in doc:
        if token.lemma_.lower() != "be":
            continue
        subject = next((child for child in token.children if child.dep_ in {"nsubj", "nsubjpass"}), None)
        complement = next((child for child in token.children if child.dep_ in {"acomp", "attr", "oprd"}), None)
        if subject is None:
            continue
        if complement is None:
            continue
        subject_chunk = _noun_chunk_for_token(doc, subject)
        subject_text = str(subject_chunk.text if subject_chunk is not None else subject.text)
        lowered_subject = subject_text.lower()
        if not re.search(r"\b(?:my|our)\b", lowered_subject):
            continue
        excluded = {subject.i, token.i}
        if subject_chunk is not None:
            excluded.update(range(subject_chunk.start, subject_chunk.end))
        complement_chunk = _noun_chunk_for_token(doc, complement)
        if complement_chunk is not None:
            value_text = _clean_attribute_value(complement_chunk.text)
        else:
            value_tokens = [piece.text for piece in complement.subtree if piece.i not in excluded and piece.dep_ != "punct"]
            value_text = _clean_attribute_value(" ".join(value_tokens))
        entity_key = _clean_entity_phrase(lowered_subject)
        if entity_key and value_text:
            return entity_key, value_text
    return None


def _extract_predicate_metadata(sentence: str) -> dict[str, Any]:
    doc = _cached_spacy_doc(sentence)
    root = next((token for token in doc if token.dep_ == "ROOT"), None)
    if root is None:
        return {}

    metadata: dict[str, Any] = {
        "predicate_text": root.text,
        "predicate_lemma": root.lemma_.lower(),
        "predicate_pos": root.pos_,
    }

    subject = next((child for child in root.children if child.dep_ in {"nsubj", "nsubjpass"}), None)
    if subject is not None:
        metadata["subject_text"] = _chunk_text_for_token(doc, subject)
        metadata["subject_head"] = subject.lemma_.lower()

    direct_object = next((child for child in root.children if child.dep_ in {"dobj", "obj", "attr"}), None)
    if direct_object is not None:
        metadata["direct_object_text"] = _chunk_text_for_token(doc, direct_object)
        metadata["direct_object_head"] = direct_object.lemma_.lower()

    prep_targets: list[dict[str, str]] = []
    for child in root.children:
        if child.dep_ != "prep":
            continue
        pobj = next((grandchild for grandchild in child.children if grandchild.dep_ == "pobj"), None)
        if pobj is None:
            continue
        prep_targets.append(
            {
                "prep": child.text.lower(),
                "object_text": _chunk_text_for_token(doc, pobj),
                "object_head": pobj.lemma_.lower(),
            }
        )
    if prep_targets:
        metadata["prep_targets"] = prep_targets

    return metadata


def _predicate_frame_facts(base_object: MemoryObject) -> list[dict[str, Any]]:
    predicate_metadata = base_object.provenance.get("predicate_metadata", {})
    if not isinstance(predicate_metadata, dict):
        return []

    predicate_lemma = str(predicate_metadata.get("predicate_lemma") or "").strip().lower()
    direct_object_text = _clean_frame_value(str(predicate_metadata.get("direct_object_text") or ""))
    prep_targets = predicate_metadata.get("prep_targets") or []
    if not isinstance(prep_targets, list):
        prep_targets = []

    facts: list[dict[str, Any]] = []

    attribute_key = _PREDICATE_STATE_ATTRIBUTE_MAP.get(predicate_lemma)
    if attribute_key and direct_object_text:
        value_text = _trim_title_context(direct_object_text) if predicate_lemma in {"read", "watch"} else direct_object_text
        if value_text:
            facts.append(
                {
                    "attribute_key": attribute_key,
                    "value_text": value_text,
                    "entity_key": "user",
                }
            )
            if predicate_lemma == "use":
                object_head = str(predicate_metadata.get("direct_object_head") or "").strip().lower()
                tokens = direct_object_text.split()
                if object_head in _BRANDABLE_PRODUCT_HEADS and len(tokens) >= 2:
                    brand_tokens = tokens[:-1]
                    brand_text = _clean_frame_value(" ".join(brand_tokens))
                    if brand_text and any(char.isupper() for char in brand_text):
                        facts.append(
                            {
                                "attribute_key": "brand",
                                "value_text": brand_text,
                                "entity_key": object_head,
                            }
                        )

    if predicate_lemma == "work":
        for prep_target in prep_targets:
            prep = str(prep_target.get("prep") or "").strip().lower()
            object_text = _clean_frame_value(str(prep_target.get("object_text") or ""))
            if not object_text:
                continue
            if prep == "as":
                facts.append(
                    {
                        "attribute_key": "occupation",
                        "value_text": object_text,
                        "entity_key": "user",
                    }
                )
            elif prep in _WORK_LOCATION_PREPOSITIONS:
                facts.append(
                    {
                        "attribute_key": "employer",
                        "value_text": object_text,
                        "entity_key": "user",
                    }
                )

    if predicate_lemma in {"live", "stay", "reside"}:
        for prep_target in prep_targets:
            prep = str(prep_target.get("prep") or "").strip().lower()
            object_text = _clean_frame_value(str(prep_target.get("object_text") or ""))
            if prep in _LOCATION_PREPOSITIONS and object_text and _looks_like_location_value(object_text):
                facts.append(
                    {
                        "attribute_key": "location",
                        "value_text": object_text,
                        "entity_key": "user",
                    }
                )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for fact in facts:
        key = (
            str(fact.get("entity_key") or ""),
            str(fact.get("attribute_key") or ""),
            str(fact.get("value_text") or ""),
        )
        if not all(key) or key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    return deduped


def _extract_attribute_key(normalized_sentence: str) -> str | None:
    match = _ATTRIBUTE_KEY_RE.search(normalized_sentence)
    if not match:
        return None
    attribute = match.group(1).strip().lower()
    return canonical_attribute(attribute) or attribute


def _extract_event_key(normalized_sentence: str) -> str | None:
    match = _EVENT_KEY_RE.search(normalized_sentence)
    if not match:
        return None
    return match.group(1).strip().lower()


def _infer_update_relation(normalized_sentence: str) -> str | None:
    if " instead of " in f" {normalized_sentence} " or " no longer " in f" {normalized_sentence} ":
        return "overwrite"
    if " now " in f" {normalized_sentence} " or " currently " in f" {normalized_sentence} ":
        return "latest"
    if _contains_any(normalized_sentence, _UPDATE_MARKERS):
        return "update"
    return None


def _normalize_identifier(*parts: str | None) -> str | None:
    normalized_parts: list[str] = []
    for part in parts:
        text = str(part or "").strip().lower()
        if not text:
            continue
        for segment in text.split(":"):
            token = _ID_SANITIZE_RE.sub("_", segment).strip("_")
            if token:
                normalized_parts.append(token)
    if not normalized_parts:
        return None
    return ":".join(normalized_parts)


def _resolved_attribute_key(
    *,
    object_type: str,
    entity_key: str | None,
    attribute_key: str | None,
    value_unit: str | None,
    normalized_sentence: str,
) -> str | None:
    if attribute_key and attribute_key != entity_key:
        return canonical_attribute(attribute_key) or attribute_key
    if object_type in {"state", "update"}:
        return "state"
    if object_type in {"numeric_fact", "comparison_fact"}:
        if value_unit == "usd":
            return "cost"
        if value_unit == "percent":
            return "percent"
        if " total " in f" {normalized_sentence} ":
            return "total"
        return "amount"
    return None


def _attribute_fact_object(
    *,
    view: CandidateView,
    speaker: str,
    turn_index: int,
    sentence_index: int,
    local_order: int,
    sentence: str,
    prefix: str,
    base_object: MemoryObject,
    attribute_key: str,
    value_text: str,
    value_number: float | None = None,
    value_unit: str | None = None,
    entity_key: str | None = None,
    object_type: str = "attribute_fact",
) -> MemoryObject:
    resolved_entity_key = entity_key or base_object.entity_key or "user"
    canonical_entity_id = _normalize_identifier("entity", resolved_entity_key)
    canonical_attribute_id = _normalize_identifier(canonical_entity_id, attribute_key)
    canonical_state_id = _normalize_identifier("state", canonical_attribute_id)
    compact_parts: list[str] = []
    if speaker and speaker != "unknown":
        compact_parts.append(f"{speaker}:")
    if resolved_entity_key and resolved_entity_key != "user":
        compact_parts.append(f"{resolved_entity_key}")
    compact_parts.append(f"{attribute_key}:")
    compact_parts.append(value_text)
    render_text = " ".join(part for part in compact_parts if part).strip()
    object_id = f"{view.memory_id}:t{turn_index:02d}:o{local_order:02d}"
    schema_hints = _schema_hints_for_object(
        object_type=object_type,
        attribute_key=attribute_key,
        event_key=base_object.event_key,
        value_number=value_number,
        value_unit=value_unit,
        predicate_metadata=base_object.provenance.get("predicate_metadata"),
        low_authority=False,
    )
    provenance = base_object.provenance.with_updates(
        object_id=object_id,
        derived_from_object_id=base_object.object_id,
        derived_attribute=attribute_key,
        extracted_value_text=value_text,
        source_text=sentence.strip(),
        source_span_text=sentence.strip(),
        render_text=render_text,
        schema_hints=schema_hints,
    )
    return MemoryObject(
        object_id=object_id,
        memory_id=view.memory_id,
        conversation_id=None,
        speaker=speaker,
        date_key=view.date_key,
        object_type=object_type,
        entity_key=resolved_entity_key,
        attribute_key=attribute_key,
        value_text=value_text,
        value_number=value_number,
        value_unit=value_unit,
        event_key=base_object.event_key,
        update_relation=base_object.update_relation,
        canonical_entity_id=canonical_entity_id,
        canonical_attribute_id=canonical_attribute_id,
        canonical_state_id=canonical_state_id,
        canonical_event_id=base_object.canonical_event_id,
        update_group_id=canonical_state_id if object_type in {"attribute_fact", "state", "update"} else canonical_state_id,
        previous_object_id=None,
        source_text=sentence.strip(),
        render_text=render_text,
        provenance=provenance,
    )


def _extract_occupation_value(sentence: str) -> str | None:
    for pattern in _OCCUPATION_VALUE_PATTERNS:
        match = pattern.search(sentence)
        if not match:
            continue
        value = _clean_attribute_value(match.group("value"))
        if value:
            return value
    return None


def _extract_name_value(sentence: str) -> str | None:
    for pattern in _NAME_VALUE_PATTERNS:
        match = pattern.search(sentence)
        if not match:
            continue
        value = _trim_title_context(_clean_attribute_value(match.group("value")))
        if value:
            return value
    favorite_match = _FAVORITE_NAME_RE.search(sentence)
    if favorite_match:
        value = _normalize_surface_form(_clean_attribute_value(favorite_match.group("value")))
        if value:
            return value
    direct_object_value = _extract_direct_object_value(sentence)
    if direct_object_value:
        return direct_object_value
    return None


def _extract_speed_value(sentence: str) -> tuple[float, str, str] | None:
    for pattern in _SPEED_VALUE_PATTERNS:
        match = pattern.search(sentence)
        if not match:
            continue
        raw_number = str(match.group("value") or "").replace(",", "").strip()
        unit = str(match.group("unit") or "").strip().lower()
        try:
            value = float(raw_number)
        except ValueError:
            continue
        label = f"{int(value) if value.is_integer() else value} {unit.upper()}"
        return value, unit, label
    return None


def _extract_breed_value(sentence: str) -> str | None:
    for pattern in _BREED_VALUE_PATTERNS:
        match = pattern.search(sentence)
        if not match:
            continue
        value = _trim_title_context(_clean_attribute_value(match.group("value")))
        if value:
            return value
    return None


def _event_signature(normalized_sentence: str, event_key: str | None) -> str | None:
    if not event_key:
        return None
    tokens = normalized_sentence.split()
    try:
        start = tokens.index(event_key) + 1
    except ValueError:
        start = 0
    kept = [token for token in tokens[start:] if token not in _EVENT_SIGNATURE_SKIP][:3]
    if not kept:
        kept = [token for token in tokens if token not in _EVENT_SIGNATURE_SKIP][:3]
    return _normalize_identifier(*kept)


def _object_position_key(memory_object: MemoryObject) -> tuple[int, int, int]:
    return (
        int(memory_object.provenance.get("turn_index", 0)),
        int(memory_object.provenance.get("sentence_index", 0)),
        int(memory_object.provenance.get("local_order", 0)),
    )


def _link_memory_objects(objects: list[MemoryObject]) -> list[MemoryObject]:
    if not objects:
        return []

    sorted_objects = sorted(
        objects,
        key=lambda obj: (
            obj.date_key,
            obj.memory_id,
            int(obj.provenance.get("turn_index", 0)),
            int(obj.provenance.get("sentence_index", 0)),
            obj.object_id,
        ),
    )
    previous_by_group: dict[str, MemoryObject] = {}
    linked: list[MemoryObject] = []
    for obj in sorted_objects:
        previous_object_id = None
        if obj.update_group_id:
            previous = previous_by_group.get(obj.update_group_id)
            if previous is not None:
                if previous.memory_id != obj.memory_id or _object_position_key(previous)[:2] != _object_position_key(obj)[:2]:
                    previous_object_id = previous.object_id
        linked_obj = replace(obj, previous_object_id=previous_object_id)
        linked.append(linked_obj)
        if linked_obj.update_group_id:
            previous_by_group[linked_obj.update_group_id] = linked_obj
    linked.sort(key=lambda obj: obj.object_id)
    return linked


def _infer_object_type(
    *,
    normalized_sentence: str,
    has_date: bool,
    value_number: float | None,
    event_key: str | None,
    entity_key: str | None,
    attribute_key: str | None,
) -> str:
    if _contains_any(normalized_sentence, _COMPARISON_MARKERS):
        return "comparison_fact"
    if value_number is not None:
        return "numeric_fact"
    if _contains_any(normalized_sentence, _UPDATE_MARKERS):
        return "update"
    if _contains_any(normalized_sentence, _PREFERENCE_MARKERS):
        return "preference"
    if event_key:
        return "event"
    has_state_anchor = bool(attribute_key) or bool(
        re.search(r"\b(?:my|our|the)\s+[a-z][a-z0-9_-]{2,}\b", normalized_sentence)
    )
    if _contains_any(normalized_sentence, _STATE_MARKERS) and has_state_anchor:
        return "state"
    if has_date:
        return "dated_fact"
    return "state"


def _schema_hints_for_object(
    *,
    object_type: str,
    attribute_key: str | None,
    event_key: str | None,
    value_number: float | None,
    value_unit: str | None,
    predicate_metadata: dict[str, Any] | None,
    low_authority: bool = False,
) -> dict[str, Any]:
    labels: list[str] = []
    plan_kinds: list[str] = []
    answer_type = "text"

    attribute = str(attribute_key or "").strip().lower()
    predicate_lemma = str((predicate_metadata or {}).get("predicate_lemma") or "").strip().lower()
    generic_state_text = object_type == "state" and attribute == "state"

    if generic_state_text:
        labels.append("generic_state_text")
    elif object_type in {"state", "attribute_fact", "preference"}:
        labels.extend(["attribute_value", "direct_answer_candidate"])
        plan_kinds.append("attribute_lookup")
    if object_type == "update":
        labels.extend(["state_transition", "current_state_candidate"])
        plan_kinds.append("current_attribute_lookup")
    if object_type in {"dated_fact", "event"} or event_key:
        labels.append("temporal_anchor")
    if object_type in {"numeric_fact", "comparison_fact"} or value_number is not None:
        labels.append("numeric_operand")
        answer_type = "number"
        if object_type == "comparison_fact":
            plan_kinds.append("difference")
        else:
            plan_kinds.append("sum_operands")
    if object_type == "event":
        labels.append("countable_item")
        plan_kinds.append("count_events")
    if attribute in {"breed", "name", "brand", "location", "occupation", "speed", "date", "time"}:
        labels.append("attribute_lookup_ready")
    if attribute in {"reading", "watching", "studying", "using", "employer"}:
        labels.append("attribute_lookup_ready")
    if attribute in {"location", "date", "time"}:
        answer_type = attribute
    if attribute in {"breed", "brand", "name", "occupation", "reading", "watching", "studying", "using", "employer"}:
        answer_type = "short_text"
    if predicate_lemma in {"read", "watch", "study", "use"}:
        labels.append("consumption_state")
        if "attribute_lookup" not in plan_kinds:
            plan_kinds.append("attribute_lookup")
    if predicate_lemma in {"acquire", "buy", "collect", "get", "order", "pick", "purchase"}:
        labels.append("acquisition_marker")
    if predicate_lemma in {"complete", "finish", "read", "rewatch", "see", "try", "visit", "watch"}:
        labels.append("consumption_or_completion_marker")
    if predicate_lemma in {"complete", "finish", "read", "watch"}:
        labels.append("progress_marker")
    if predicate_lemma in {"work", "live", "stay"}:
        labels.append("profile_state")
        if "attribute_lookup" not in plan_kinds:
            plan_kinds.append("attribute_lookup")
    if value_unit in {"usd", "percent"}:
        answer_type = value_unit
    if low_authority:
        labels.append("low_authority_text")

    return {
        "labels": sorted(set(labels)),
        "plan_kinds": sorted(set(plan_kinds)),
        "answer_type": answer_type,
        "attribute_key": attribute,
    }


def _has_date_signal(text: str, normalized_sentence: str) -> bool:
    if DATE_RE.search(text):
        return True
    if TIME_RE.search(text):
        return True
    tokens = set(re.findall(r"\b[a-z0-9]+\b", normalized_sentence))
    if tokens & set(TEMPORAL_HINTS):
        return True
    return False


def _object_from_sentence(
    *,
    view: CandidateView,
    sentence: str,
    prefix: str,
    speaker: str,
    turn_index: int,
    sentence_index: int,
    local_order: int,
) -> MemoryObject:
    render_text = _render_span(prefix, speaker, sentence)
    analysis_text = sentence.strip() if sentence.strip() else render_text
    normalized = " ".join(re.findall(r"\b[a-z0-9]+\b", analysis_text.lower()))
    has_date = _has_date_signal(render_text, normalized)
    value_number, value_unit = _extract_value_number(analysis_text)
    entity_key = _extract_entity_key(normalized)
    attribute_key = _extract_attribute_key(normalized)
    event_key = _extract_event_key(normalized)
    object_type = _infer_object_type(
        normalized_sentence=normalized,
        has_date=has_date,
        value_number=value_number,
        event_key=event_key,
        entity_key=entity_key,
        attribute_key=attribute_key,
    )
    if _is_low_authority_sentence(sentence=sentence, speaker=speaker) and object_type in {
        "numeric_fact",
        "comparison_fact",
        "state",
        "preference",
    }:
        object_type = "dated_fact"
        value_number = None
        value_unit = None
        attribute_key = None
    update_relation = _infer_update_relation(normalized)
    if object_type != "update":
        update_relation = None
    resolved_attribute_key = _resolved_attribute_key(
        object_type=object_type,
        entity_key=entity_key,
        attribute_key=attribute_key,
        value_unit=value_unit,
        normalized_sentence=normalized,
    )
    canonical_entity_id = _normalize_identifier("entity", entity_key)
    canonical_attribute_id = _normalize_identifier(canonical_entity_id, resolved_attribute_key)
    canonical_state_id = None
    update_group_id = None
    if object_type in {"state", "update", "numeric_fact", "comparison_fact"} and canonical_attribute_id:
        canonical_state_id = _normalize_identifier("state", canonical_attribute_id)
        update_group_id = canonical_state_id
    canonical_event_id = None
    if event_key:
        canonical_event_id = _normalize_identifier(
            "event",
            event_key,
            entity_key,
            _event_signature(normalized, event_key),
        )

    value_text = sentence.strip() if sentence.strip() else None
    predicate_metadata = _extract_predicate_metadata(analysis_text)
    schema_hints = _schema_hints_for_object(
        object_type=object_type,
        attribute_key=resolved_attribute_key,
        event_key=event_key,
        value_number=value_number,
        value_unit=value_unit,
        predicate_metadata=predicate_metadata,
        low_authority=_is_low_authority_sentence(sentence=sentence, speaker=speaker),
    )

    # Typed sub-values extracted from the sentence at build time.
    # Stored directly on the provenance so plan-conditioned slot resolution
    # can select the right value type without re-running extraction per query.
    _place_val = _extract_location_value(sentence)
    _dur_result = _extract_duration_value(sentence)
    _dur_val = _dur_result[2] if _dur_result else None
    _time_val = _extract_time_value(sentence)

    provenance = ProvenanceRecord.from_mapping(
        {
            "memory_id": view.memory_id,
            "object_id": f"{view.memory_id}:t{turn_index:02d}:o{local_order:02d}",
            "speaker": speaker,
            "turn_index": turn_index,
            "sentence_index": sentence_index,
            "local_order": local_order,
            "source_text": sentence.strip(),
            "source_span_text": sentence.strip(),
            "render_text": render_text,
            "memory_text": view.text,
            "date_key": view.date_key,
            "object_type": object_type,
            "canonical_entity_id": canonical_entity_id,
            "canonical_attribute_id": canonical_attribute_id,
            "canonical_state_id": canonical_state_id,
            "canonical_event_id": canonical_event_id,
            "update_group_id": update_group_id,
            "place_value": _place_val,
            "duration_value": _dur_val,
            "time_value": _time_val,
            "temporal_hints": [
                token
                for token in re.findall(r"\b[a-z0-9]+\b", normalized)
                if token in TEMPORAL_HINTS
            ],
            "update_hints": [
                marker
                for marker in NEGATION_PATTERNS
                if (" " in marker and marker in f" {normalized} ") or (" " not in marker and marker in normalized.split())
            ] + [token for token in normalized.split() if token in UPDATE_MARKERS],
            "predicate_metadata": predicate_metadata,
            "schema_hints": schema_hints,
        }
    )

    return MemoryObject(
        object_id=f"{view.memory_id}:t{turn_index:02d}:o{local_order:02d}",
        memory_id=view.memory_id,
        conversation_id=None,
        speaker=speaker,
        date_key=view.date_key,
        object_type=object_type,
        entity_key=entity_key,
        attribute_key=resolved_attribute_key,
        value_text=value_text,
        value_number=value_number,
        value_unit=value_unit,
        event_key=event_key,
        update_relation=update_relation,
        canonical_entity_id=canonical_entity_id,
        canonical_attribute_id=canonical_attribute_id,
        canonical_state_id=canonical_state_id,
        canonical_event_id=canonical_event_id,
        update_group_id=update_group_id,
        previous_object_id=None,
        source_text=sentence.strip(),
        render_text=render_text,
        provenance=provenance,
    )


def _supplemental_objects_from_sentence(
    *,
    view: CandidateView,
    sentence: str,
    prefix: str,
    speaker: str,
    turn_index: int,
    sentence_index: int,
    local_order_start: int,
    base_object: MemoryObject,
) -> list[MemoryObject]:
    supplemental: list[MemoryObject] = []
    next_local_order = local_order_start

    if _skip_supplemental_fact_extraction(sentence=sentence, speaker=speaker):
        return supplemental

    def _append_attribute_fact(
        *,
        attribute_key: str,
        value_text: str,
        entity_key: str | None = None,
        object_type: str = "attribute_fact",
        value_number: float | None = None,
        value_unit: str | None = None,
    ) -> None:
        nonlocal next_local_order
        cleaned_value = str(value_text or "").strip()
        if not cleaned_value:
            return
        supplemental.append(
            _attribute_fact_object(
                view=view,
                speaker=speaker,
                turn_index=turn_index,
                sentence_index=sentence_index,
                local_order=next_local_order,
                sentence=sentence,
                prefix=prefix,
                base_object=base_object,
                attribute_key=attribute_key,
                value_text=cleaned_value,
                entity_key=entity_key,
                object_type=object_type,
                value_number=value_number,
                value_unit=value_unit,
            )
        )
        next_local_order += 1

    location_value = _extract_location_value(sentence)
    if location_value:
        _append_attribute_fact(
            attribute_key="location",
            value_text=location_value,
            entity_key=base_object.entity_key,
        )

    occupation_value = _extract_occupation_value(sentence)
    if occupation_value:
        _append_attribute_fact(
            attribute_key="occupation",
            value_text=occupation_value,
            entity_key="user",
        )

    name_value = _extract_name_value(sentence)
    if name_value:
        name_entity = base_object.entity_key or ("play" if "play" in sentence.lower() else "item")
        _append_attribute_fact(
            attribute_key="name",
            value_text=name_value,
            entity_key=name_entity,
        )

    breed_value = _extract_breed_value(sentence)
    if breed_value:
        breed_entity = base_object.entity_key or "pet"
        _append_attribute_fact(
            attribute_key="breed",
            value_text=breed_value,
            entity_key=breed_entity,
        )

    state_fact = _extract_compact_state_value(sentence)
    if state_fact:
        state_entity, state_value = state_fact
        _append_attribute_fact(
            attribute_key="state",
            value_text=state_value,
            entity_key=state_entity,
        )

    speed_value = _extract_speed_value(sentence)
    if speed_value:
        value_number, value_unit, label = speed_value
        speed_entity = base_object.entity_key or "internet plan"
        _append_attribute_fact(
            attribute_key="speed",
            value_text=label,
            value_number=value_number,
            value_unit=value_unit,
            entity_key=speed_entity,
            object_type="numeric_fact",
        )

    duration_value = _extract_duration_value(sentence)
    if duration_value:
        value_number, value_unit, label = duration_value
        # -1.0 is the sentinel for word-form numbers ("two weeks"); don't propagate
        # as a numeric fact since the value can't participate in arithmetic aggregation.
        if value_number >= 0:
            duration_entity = base_object.entity_key or base_object.event_key or "event"
            _append_attribute_fact(
                attribute_key="time",
                value_text=label,
                value_number=value_number,
                value_unit=value_unit,
                entity_key=duration_entity,
                object_type="numeric_fact",
            )

    size_value = _extract_size_value(sentence)
    if size_value:
        size_entity = base_object.entity_key or "item"
        _append_attribute_fact(
            attribute_key="size",
            value_text=size_value,
            entity_key=size_entity,
        )

    percent_value = _extract_percent_value(sentence)
    if percent_value:
        value_number, value_unit, label = percent_value
        percent_entity = base_object.entity_key or "item"
        _append_attribute_fact(
            attribute_key="percent",
            value_text=label,
            value_number=value_number,
            value_unit=value_unit,
            entity_key=percent_entity,
            object_type="numeric_fact",
        )

    ratio_value = _extract_ratio_value(sentence)
    if ratio_value:
        ratio_entity = base_object.entity_key or "item"
        _append_attribute_fact(
            attribute_key="ratio",
            value_text=ratio_value,
            entity_key=ratio_entity,
        )

    color_value = _extract_color_value(sentence)
    if color_value:
        color_entity = base_object.entity_key or "item"
        _append_attribute_fact(
            attribute_key="color",
            value_text=color_value,
            entity_key=color_entity,
        )

    direct_object_value = _extract_direct_object_value(sentence)
    if direct_object_value:
        object_entity = base_object.entity_key or "item"
        _append_attribute_fact(
            attribute_key="name",
            value_text=direct_object_value,
            entity_key=object_entity,
        )

    for fact in _predicate_frame_facts(base_object):
        _append_attribute_fact(
            attribute_key=str(fact.get("attribute_key") or ""),
            value_text=str(fact.get("value_text") or ""),
            entity_key=str(fact.get("entity_key") or "") or None,
        )

    deduped: list[MemoryObject] = []
    seen_keys: set[tuple[str, str | None, str | None, str | None, float | None, str | None]] = set()
    for obj in supplemental:
        key = (
            obj.object_type,
            obj.entity_key,
            obj.attribute_key,
            obj.value_text,
            obj.value_number,
            obj.value_unit,
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(obj)
    return deduped


def _objects_from_sentence(
    *,
    view: CandidateView,
    sentence: str,
    prefix: str,
    speaker: str,
    turn_index: int,
    sentence_index: int,
    local_order_start: int,
) -> list[MemoryObject]:
    base_object = _object_from_sentence(
        view=view,
        sentence=sentence,
        prefix=prefix,
        speaker=speaker,
        turn_index=turn_index,
        sentence_index=sentence_index,
        local_order=local_order_start,
    )
    supplemental = _supplemental_objects_from_sentence(
        view=view,
        sentence=sentence,
        prefix=prefix,
        speaker=speaker,
        turn_index=turn_index,
        sentence_index=sentence_index,
        local_order_start=local_order_start + 1,
        base_object=base_object,
    )
    return [base_object, *supplemental]


def build_memory_objects(view: CandidateView) -> list[MemoryObject]:
    if not str(view.text or "").strip():
        return []

    from shared.perf import get_counters
    _pc = get_counters()
    _pc.memory_views_built += 1

    objects: list[MemoryObject] = []
    local_order = 0
    turn_index = 0
    for prefix, speaker, body in _split_speaker_turns(view.text):
        turn_index += 1
        sentences = _split_sentences(body)
        if not sentences:
            continue
        for sentence_index, sentence in enumerate(sentences, start=1):
            cleaned = sentence.strip()
            if not cleaned:
                continue
            _pc.sentences_processed += 1
            sentence_objects = _objects_from_sentence(
                view=view,
                sentence=cleaned,
                prefix=prefix,
                speaker=speaker,
                turn_index=turn_index,
                sentence_index=sentence_index,
                local_order_start=local_order,
            )
            objects.extend(sentence_objects)
            local_order += len(sentence_objects)

    final = _link_memory_objects(objects)
    _pc.memory_objects_created += len(final)
    return final


def build_memory_objects_for_views(views: list[CandidateView]) -> list[MemoryObject]:
    ordered_views = sorted(views, key=lambda view: (view.rank, view.memory_id))
    objects: list[MemoryObject] = []
    for view in ordered_views:
        objects.extend(build_memory_objects(view))
    return _link_memory_objects(objects)
