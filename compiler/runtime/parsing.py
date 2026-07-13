"""Numeric text parsing primitives for evidence compiler."""

from __future__ import annotations

import re

_NUMERIC_PARSE_RE = re.compile(r"(?P<prefix>[$€£])?(?P<number>\d[\d,]*(?:\.\d+)?)(?P<suffix>%?)")

_DURATION_TEXT_RE = re.compile(
    r"(?P<number>\d[\d,]*(?:\.\d+)?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty)\b)\s+"
    r"(?P<unit>days?|weeks?|months?|years?|hours?|minutes?|seconds?|semesters?|quarters?)",
    re.IGNORECASE,
)


def _parse_number_word(text: str) -> int | None:
    """Parse English number words to int. Uses word2number for arbitrary
    English numbers, with a fast-path dict for common cases and handling
    for fuzzy quantifiers."""
    normalized = str(text or "").strip().lower()
    if not normalized:
        return None
    if normalized.isdigit():
        return int(normalized)
    # Fast path for the most common single words
    fast = _NUMBER_WORD_TO_INT.get(normalized)
    if fast is not None:
        return fast
    # Fuzzy quantifiers common in conversational memory
    fuzzy = _FUZZY_QUANTIFIERS.get(normalized)
    if fuzzy is not None:
        return fuzzy
    # Arbitrary English number words via word2number
    try:
        from word2number import w2n
        return w2n.word_to_num(normalized)
    except Exception:
        return None


_FUZZY_QUANTIFIERS: dict[str, int] = {
    "a": 1,
    "an": 1,
    "couple": 2,
    "a couple": 2,
    "few": 3,
    "a few": 3,
    "several": 4,
    "half a dozen": 6,
    "a dozen": 12,
    "dozen": 12,
}

_NUMBER_WORD_TO_INT = {
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
}
