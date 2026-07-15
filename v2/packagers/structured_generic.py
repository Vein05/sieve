"""Generic structured packager: schema-free, no v1 cache, no benchmark vocabulary.

Groups candidate passages by query-term overlap into compact labelled sections,
orders them chronologically when a date prefix parses (retrieval rank otherwise),
and renders exact source quotes for values (numbers, dates, capitalised entities).

This is the scientific object of the v3 pilot: it uses only generic operations
(stem-overlap grouping, date parsing, verbatim quoting) so it cannot overfit any
one benchmark's taxonomy. A starvation floor guarantees it never emits fewer than
``MIN_EVIDENCE_TOKENS`` tokens of evidence: if structuring yields less, top
passages are appended verbatim until the floor is met.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from v2.budget import count_tokens, truncate_units_to_budget
from v2.packagers.base import (
    PackagerContext,
    candidate_memories,
    row_example_id,
    row_query,
)
from v2.scorer import StemOverlapScorer
from v2.types import EvidenceVariant

STYLE_NAME = "structured_generic"

# Evidence-starvation floor: the packager never emits fewer than this many tokens
# of evidence. Drives the ConvoMem anti-collapse hypothesis (v1 shipped mean
# 14-token packages that induced spurious abstention).
MIN_EVIDENCE_TOKENS = 120

# A passage joins a topic section if it shares at least this fraction of the
# section's key query terms. Generic threshold, not tuned to any benchmark.
_TOPIC_MIN_OVERLAP = 1

# Date prefix on a memory ("2024/05/01 (Wed) 13:00 ..."); enables chronology.
_DATE_PREFIX = re.compile(r"^(\d{4}/\d{2}/\d{2})(?:\s+\([^)]+\))?(?:\s+(\d{2}:\d{2}))?")

# A "value" worth quoting exactly: a number, a date-like token, or a capitalised
# multi-letter word (named-entity proxy). No benchmark-specific vocabulary.
_VALUE_TOKEN = re.compile(r"\b(?:\d[\d.,:/-]*|[A-Z][A-Za-z0-9]{2,})\b")

_UNGROUPED_LABEL = "other"


@dataclass(frozen=True)
class _Passage:
    rank: int
    memory_id: str
    content: str
    order_key: tuple[int, str, str, int]
    query_overlap: frozenset[str]
    score: float


def _order_key(content: str, rank: int) -> tuple[int, str, str, int]:
    """Chronological when a date prefix parses; retrieval rank otherwise."""
    match = _DATE_PREFIX.match(content.strip())
    if match:
        return 0, match.group(1), match.group(2) or "00:00", rank
    return 1, "", "", rank


def _query_terms(query: str) -> list[str]:
    from shared.nlp import get_clean_tokens

    return list(dict.fromkeys(get_clean_tokens(query)))


def _passage_overlap(content: str, query_terms: list[str]) -> frozenset[str]:
    from shared.nlp import check_stem_equivalence, get_clean_tokens

    passage_tokens = get_clean_tokens(content)
    matched = {
        term
        for term in query_terms
        if any(check_stem_equivalence(term, tok) for tok in passage_tokens)
    }
    return frozenset(matched)


def _build_passages(
    candidates: list[dict[str, Any]], query: str, scorer: Any
) -> list[_Passage]:
    query_terms = _query_terms(query)
    contents = [str(c["content"]).strip() for c in candidates]
    scores = scorer(query, contents) if callable(scorer) else scorer.score(query, contents)
    passages: list[_Passage] = []
    for rank, candidate in enumerate(candidates):
        content = str(candidate["content"]).strip()
        passages.append(
            _Passage(
                rank=rank,
                memory_id=str(candidate.get("memory_id", "")),
                content=content,
                order_key=_order_key(content, rank),
                query_overlap=_passage_overlap(content, query_terms),
                score=float(scores[rank]),
            )
        )
    return passages


def _section_key(passage: _Passage) -> str:
    """Topic label = the passage's highest-priority overlapping query term."""
    if not passage.query_overlap:
        return _UNGROUPED_LABEL
    return sorted(passage.query_overlap)[0]


def _group_sections(passages: list[_Passage]) -> dict[str, list[_Passage]]:
    sections: dict[str, list[_Passage]] = {}
    for passage in passages:
        sections.setdefault(_section_key(passage), []).append(passage)
    for members in sections.values():
        members.sort(key=lambda p: p.order_key)
    return sections


def _quoted_values(content: str) -> str:
    values = list(dict.fromkeys(_VALUE_TOKEN.findall(content)))
    if not values:
        return ""
    return " values: " + ", ".join(f'"{value}"' for value in values)


def _render_passage_line(passage: _Passage) -> str:
    prefix = f"[{passage.memory_id}] " if passage.memory_id else ""
    return f"{prefix}{passage.content}{_quoted_values(passage.content)}"


def _render_units(sections: dict[str, list[_Passage]]) -> list[str]:
    ranked_labels = sorted(
        sections,
        key=lambda label: (
            label == _UNGROUPED_LABEL,
            -max(p.score for p in sections[label]),
            label,
        ),
    )
    units: list[str] = []
    for label in ranked_labels:
        units.append(f"# {label}")
        units.extend(_render_passage_line(p) for p in sections[label])
    return units


def _apply_floor(
    units: list[str], passages: list[_Passage], encoding: str
) -> tuple[list[str], bool]:
    """Append top-scored passages verbatim until the floor is met."""
    text = "\n".join(units)
    if count_tokens(text, encoding=encoding) >= MIN_EVIDENCE_TOKENS:
        return units, False
    present = set(units)
    padded = list(units)
    for passage in sorted(passages, key=lambda p: (-p.score, p.rank)):
        line = passage.content
        if line in present:
            continue
        padded.append(line)
        present.add(line)
        if count_tokens("\n".join(padded), encoding=encoding) >= MIN_EVIDENCE_TOKENS:
            break
    return padded, True


class StructuredGenericPackager:
    """Schema-free generic structuring with a starvation floor."""

    style_name = STYLE_NAME

    def package(
        self, row: Mapping[str, Any], budget: int, ctx: PackagerContext
    ) -> EvidenceVariant:
        candidates = candidate_memories(row)
        scorer = ctx.relevance_scorer or StemOverlapScorer()
        passages = _build_passages(candidates, row_query(row), scorer)
        sections = _group_sections(passages)
        units = _render_units(sections)
        units, floor_fired = _apply_floor(units, passages, ctx.encoding)

        packing = truncate_units_to_budget(units, budget, encoding=ctx.encoding)
        evidence_text = "\n".join(packing.units)
        kept_ids = tuple(
            dict.fromkeys(p.memory_id for p in passages if p.memory_id)
        )
        return EvidenceVariant(
            example_id=row_example_id(row),
            style=self.style_name,
            budget=budget,
            evidence_text=evidence_text,
            token_count=count_tokens(evidence_text, encoding=ctx.encoding),
            source_memory_ids=kept_ids,
            meta={
                "n_candidates": len(candidates),
                "n_sections": len(sections),
                "n_units": len(units),
                "n_kept": len(packing.units),
                "floor_fired": floor_fired,
                "min_evidence_tokens": MIN_EVIDENCE_TOKENS,
                "overflow": packing.overflow,
            },
        )
