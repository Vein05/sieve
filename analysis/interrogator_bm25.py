"""Minimal Okapi BM25 over a fixed document set for reachability analysis."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

BM25_K1 = 1.5
BM25_B = 0.75
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    """Okapi BM25 index over a fixed corpus of (doc_id, text) documents."""

    doc_ids: list[str]
    _tokens: list[list[str]] = field(default_factory=list)
    _df: Counter = field(default_factory=Counter)
    _idf: dict[str, float] = field(default_factory=dict)
    _doc_len: list[int] = field(default_factory=list)
    _avg_len: float = 0.0

    @classmethod
    def build(cls, documents: list[tuple[str, str]]) -> "BM25Index":
        doc_ids = [d[0] for d in documents]
        idx = cls(doc_ids=doc_ids)
        idx._tokens = [tokenize(d[1]) for d in documents]
        idx._doc_len = [len(t) for t in idx._tokens]
        n_docs = max(len(documents), 1)
        idx._avg_len = sum(idx._doc_len) / n_docs
        for toks in idx._tokens:
            for term in set(toks):
                idx._df[term] += 1
        for term, df in idx._df.items():
            idx._idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        return idx

    def score(self, query: str, doc_pos: int) -> float:
        q_terms = tokenize(query)
        toks = self._tokens[doc_pos]
        if not toks:
            return 0.0
        tf = Counter(toks)
        length = self._doc_len[doc_pos]
        norm = BM25_K1 * (1.0 - BM25_B + BM25_B * length / (self._avg_len or 1.0))
        total = 0.0
        for term in q_terms:
            if term not in tf:
                continue
            freq = tf[term]
            idf = self._idf.get(term, 0.0)
            total += idf * (freq * (BM25_K1 + 1.0)) / (freq + norm)
        return total

    def rank(self, query: str, exclude: frozenset[str] = frozenset()) -> list[tuple[str, float]]:
        """Return (doc_id, score) sorted by descending score, excluding given ids."""
        scored = [
            (self.doc_ids[i], self.score(query, i))
            for i in range(len(self.doc_ids))
            if self.doc_ids[i] not in exclude
        ]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored
