"""Canonical marker constants shared across evidence and retrieval modules."""

from __future__ import annotations

# Comparison markers used for detecting comparative language in unit text.
# Origin: evidence/units.py (_COMPARISON_LANGUAGE_MARKERS)
COMPARISON_LANGUAGE_MARKERS = (
    " compared against ",
    " compared to ",
    " compared with ",
    " difference between ",
    " difference in ",
    " higher than ",
    " less than ",
    " lower than ",
    " more than ",
    " vs ",
    " versus ",
)

# Lightweight comparison fallback markers for query-level detection.
# Origin: evidence/roles.py (_COMPARISON_FALLBACK_MARKERS), retrieval/memory_builder.py (_COMPARISON_MARKERS)
COMPARISON_FALLBACK_MARKERS = (
    " compared ",
    " more ",
    " less ",
    " higher ",
    " lower ",
    " bigger ",
    " smaller ",
    " versus ",
    " vs ",
)

# Comparison markers for memory object type inference (overlap with fallback but uses "than" forms).
# Origin: retrieval/memory_builder.py (_COMPARISON_MARKERS)
COMPARISON_OBJECT_MARKERS = (
    " compared ",
    " versus ",
    " vs ",
    " more than ",
    " less than ",
    " higher than ",
    " lower than ",
)

# Comparison markers (set form) for query target extraction.
# Origin: retrieval/query_targets.py (_COMPARISON_MARKERS)
COMPARISON_QUERY_MARKERS = frozenset({
    "compared against",
    "compared to",
    "compared with",
    "difference between",
    "difference in",
    "greater",
    "higher than",
    "less",
    "less than",
    "lower than",
    "more",
    "more than",
    "smaller",
    "smaller than",
    "than",
    "vs",
    "versus",
})

# Ordering markers for query-level ordering detection in roles.
# Origin: evidence/roles.py (_ORDERING_MARKERS)
ORDERING_ROLE_MARKERS = (
    " first ",
    " second ",
    " third ",
    " before ",
    " after ",
    " order ",
)

# Ordering markers (set form) for query target extraction.
# Origin: retrieval/query_targets.py (_ORDERING_MARKERS)
ORDERING_QUERY_MARKERS = frozenset({
    "order of",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "before",
    "after",
    "earliest",
    "latest",
    "earliest to latest",
    "latest to earliest",
    "from earliest to latest",
    "from latest to earliest",
    "first to last",
    "last to first",
    "most recent",
    "ordered",
    "sequence",
})

# Current state markers for unit text analysis.
# Origin: evidence/units.py (_CURRENT_STATE_LANGUAGE_MARKERS)
CURRENT_STATE_LANGUAGE_MARKERS = (
    " as of now ",
    " current ",
    " current state ",
    " current status ",
    " currently ",
    " now ",
    " right now ",
    " still ",
    " switched ",
    " switched to ",
    " changed to ",
)

# Current state words for role inference.
# Origin: evidence/roles.py (_CURRENT_STATE_WORDS)
CURRENT_STATE_WORDS = (
    "currently",
    "current ",
    "right now",
    "now use",
    "now uses",
    "now using",
    "still ",
    "as of",
)

# Current state markers for query target extraction.
# Origin: retrieval/query_targets.py (_CURRENT_STATE_MARKERS)
CURRENT_STATE_QUERY_MARKERS = (
    "as of now",
    "current ",
    "current state",
    "current status",
    "currently",
    "final state",
    "final status",
    "latest",
    "now ",
    "right now",
    "still ",
    "what is the current",
    "what is the latest",
    "what is the final",
)

# Reference language markers for unit text analysis.
# Origin: evidence/units.py (_REFERENCE_LANGUAGE_MARKERS)
REFERENCE_LANGUAGE_MARKERS = (
    " as of ",
    " ago ",
    " before ",
    " after ",
    " last time ",
    " previous ",
    " reference ",
    " since ",
    " then ",
)

# State markers for memory object type inference.
# Origin: retrieval/memory_builder.py (_STATE_MARKERS)
STATE_OBJECT_MARKERS = (
    " is ",
    " are ",
    " was ",
    " were ",
    " currently ",
    " current ",
    " now ",
    " still ",
)

# Update markers for memory object type inference.
# Origin: retrieval/memory_builder.py (_UPDATE_MARKERS)
UPDATE_OBJECT_MARKERS = (
    " switched ",
    " changed ",
    " update ",
    " updated ",
    " instead of ",
    " no longer ",
    " used to ",
)

# Preference markers for memory object type inference.
# Origin: retrieval/memory_builder.py (_PREFERENCE_MARKERS)
PREFERENCE_MARKERS = (
    " prefer ",
    " preference ",
    " favorite ",
    " favourite ",
    " like ",
    " love ",
    " dislike ",
    " hate ",
)

# Query stop words for evidence rendering (filters common words from query relevance matching).
# Origin: evidence/renderer.py (inline _stop sets)
QUERY_STOP_WORDS = frozenset({
    "how", "many", "much", "did", "do", "does", "i", "the", "in", "a",
    "of", "my", "have", "has", "to", "and", "or", "total", "currently",
    "what", "which", "where", "when", "who", "been", "was", "were", "is",
    "are", "for", "on", "at", "with", "from", "by", "an", "all", "any",
    "past", "few", "last", "first", "new", "different", "types", "number",
    "spend", "spent", "me", "we", "our", "that", "this", "it", "them",
})

# Temporal adverbs for location head trimming.
# Origin: retrieval/memory_builder.py (_TEMPORAL_ADVERBS inside _trim_to_location_head)
TEMPORAL_ADVERBS = frozenset({
    "last", "this", "next", "every", "each", "ago",
    "recently", "yesterday", "today", "tomorrow",
})
