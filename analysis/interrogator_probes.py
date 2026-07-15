"""Deterministic probe generation from query + pool content only (no gold leakage)."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from analysis.interrogator_bm25 import tokenize

_CAP_SPAN_RE = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,2})\b")
_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:,?\s+\d{4})?\b|\b\d{4}/\d{2}/\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    "the a an and or but of to in on at for with from by is are was were be been "
    "i you he she it we they my your his her our their this that these those what "
    "when where who how why do does did can could would should will shall me us "
    "as if so not no yes about into over under than then them he's".split()
)
_ATTR_TERMS = ("date", "time", "when", "deadline", "before", "after", "changed", "updated", "now")
MAX_ENTITY_PROBES = 6
MAX_CHAIN_PROBES = 4
MAX_TEMPORAL_PROBES = 4
MAX_DECOMP_PROBES = 4


@dataclass(frozen=True)
class Probe:
    """A generated probe query tagged with its type."""

    text: str
    ptype: str


def _content_words(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 2]


def _entity_probes(query: str, pool_texts: list[str]) -> list[Probe]:
    spans: Counter = Counter()
    for text in pool_texts:
        for m in _CAP_SPAN_RE.findall(text):
            if m.lower() not in _STOPWORDS and len(m) > 2:
                spans[m] += 1
    probes = []
    for span, _ in spans.most_common(MAX_ENTITY_PROBES):
        probes.append(Probe(f"{span} {query}", "entity"))
    return probes


def _chain_probes(pool_texts: list[str]) -> list[Probe]:
    ent_dates: dict[str, set[str]] = {}
    for text in pool_texts:
        dates = _DATE_RE.findall(text)
        if not dates:
            continue
        for span in set(_CAP_SPAN_RE.findall(text)):
            if span.lower() in _STOPWORDS or len(span) <= 2:
                continue
            ent_dates.setdefault(span, set()).update(dates)
    probes = []
    for ent, dates in sorted(ent_dates.items()):
        if len(dates) >= 2:
            probes.append(Probe(f"{ent} {' '.join(_ATTR_TERMS[:4])}", "chain"))
        if len(probes) >= MAX_CHAIN_PROBES:
            break
    return probes


def _temporal_probes(pool_texts: list[str]) -> list[Probe]:
    dates: Counter = Counter()
    for text in pool_texts:
        for d in _DATE_RE.findall(text):
            dates[d] += 1
    probes = []
    for date, _ in dates.most_common(MAX_TEMPORAL_PROBES):
        probes.append(Probe(str(date), "temporal"))
    return probes


def _decomposition_probes(query: str, pool_texts: list[str]) -> list[Probe]:
    covered = set()
    for text in pool_texts:
        covered.update(_content_words(text))
    q_words = _content_words(query)
    probes = []
    for i in range(len(q_words) - 1):
        bigram = f"{q_words[i]} {q_words[i + 1]}"
        if q_words[i] not in covered or q_words[i + 1] not in covered:
            probes.append(Probe(bigram, "decomposition"))
        if len(probes) >= MAX_DECOMP_PROBES:
            break
    return probes


def generate_probes(query: str, pool_texts: list[str], cap: int) -> list[Probe]:
    """Generate deterministic probes from query + pool content, capped at `cap`."""
    ordered = (
        _entity_probes(query, pool_texts)
        + _chain_probes(pool_texts)
        + _temporal_probes(pool_texts)
        + _decomposition_probes(query, pool_texts)
    )
    seen: set[str] = set()
    unique: list[Probe] = []
    for p in ordered:
        key = p.text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(p)
    return unique[:cap]
