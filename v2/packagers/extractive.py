"""Verbatim sentence extraction with true chronological source ordering."""

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
    selected_candidate_memories,
)
from v2.scorer import StemOverlapScorer
from v2.types import EvidenceVariant

STYLE_NAME = "extractive"
SELECTED_STYLE_NAME = "selected_extractive"
DEFAULT_MIN_SENTENCE_CHARS = 10
DATE_PREFIX = re.compile(
    r"^(\d{4}/\d{2}/\d{2})(?:\s+\([^)]+\))?(?:\s+(\d{2}:\d{2}))?"
)


@dataclass(frozen=True)
class _Sentence:
    memory_rank: int
    sentence_pos: int
    memory_id: str
    text: str
    order_key: tuple[int, str, str, int]
    score: float = 0.0


@dataclass(frozen=True)
class _Extraction:
    candidate_count: int
    sentence_count: int
    accepted: tuple[_Sentence, ...]
    overflow: bool


def _split_sentences(text: str) -> list[str]:
    """Reuse v1's cached sentence tokenizer."""
    from shared.nlp import cached_sent_tokenize

    return [sentence.strip() for sentence in cached_sent_tokenize(text) if sentence.strip()]


def _memory_order_key(text: str, rank: int) -> tuple[int, str, str, int]:
    """Order dated memories chronologically and retain rank for undated ties."""
    match = DATE_PREFIX.match(text.strip())
    if match:
        return 0, match.group(1), match.group(2) or "00:00", rank
    return 1, "", "", rank


def _build_sentences(
    candidates: list[dict[str, Any]], min_chars: int
) -> list[_Sentence]:
    sentences: list[_Sentence] = []
    for rank, candidate in enumerate(candidates):
        content = str(candidate["content"])
        order_key = _memory_order_key(content, rank)
        for pos, sentence in enumerate(_split_sentences(content)):
            if len(sentence) >= min_chars:
                sentences.append(
                    _Sentence(
                        rank,
                        pos,
                        str(candidate.get("memory_id", "")),
                        sentence,
                        order_key,
                    )
                )
    return sentences


def _score_sentences(
    query: str,
    sentences: list[_Sentence],
    scorer: Any,
) -> list[_Sentence]:
    texts = [sentence.text for sentence in sentences]
    scores = scorer(query, texts) if callable(scorer) else scorer.score(query, texts)
    return [
        _Sentence(
            sentence.memory_rank,
            sentence.sentence_pos,
            sentence.memory_id,
            sentence.text,
            sentence.order_key,
            float(score),
        )
        for sentence, score in zip(sentences, scores)
    ]


def _emission_key(sentence: _Sentence) -> tuple[object, ...]:
    return (*sentence.order_key, sentence.sentence_pos)


def _select_sentences(
    sentences: list[_Sentence], budget: int, encoding: str
) -> tuple[list[_Sentence], bool]:
    ranked = sorted(
        sentences,
        key=lambda sentence: (-sentence.score, *_emission_key(sentence)),
    )
    if budget == 0:
        return sorted(ranked, key=_emission_key), False
    accepted: list[_Sentence] = []
    for candidate in ranked:
        trial = sorted([*accepted, candidate], key=_emission_key)
        text = "\n".join(sentence.text for sentence in trial)
        if count_tokens(text, encoding=encoding) <= budget:
            accepted = trial
    if accepted:
        return accepted, False
    top = ranked[0]
    packing = truncate_units_to_budget([top.text], budget, encoding=encoding)
    clipped = _Sentence(
        top.memory_rank,
        top.sentence_pos,
        top.memory_id,
        packing.units[0] if packing.units else "",
        top.order_key,
        top.score,
    )
    return ([clipped] if clipped.text else []), packing.overflow


class ExtractivePackager:
    """Select relevant sentences and emit them in source chronology."""

    style_name = STYLE_NAME

    def _candidates(
        self, row: Mapping[str, Any], ctx: PackagerContext
    ) -> list[dict[str, Any]]:
        return candidate_memories(row)

    def package(
        self, row: Mapping[str, Any], budget: int, ctx: PackagerContext
    ) -> EvidenceVariant:
        candidates = self._candidates(row, ctx)
        min_chars = int(ctx.params.get("min_sentence_chars", DEFAULT_MIN_SENTENCE_CHARS))
        sentences = _build_sentences(candidates, min_chars)
        if not sentences:
            result = _Extraction(len(candidates), 0, (), False)
            return self._variant(row, budget, ctx, result)
        scorer = ctx.relevance_scorer or StemOverlapScorer()
        scored = _score_sentences(row_query(row), sentences, scorer)
        accepted, overflow = _select_sentences(scored, budget, ctx.encoding)
        result = _Extraction(
            len(candidates), len(sentences), tuple(accepted), overflow
        )
        return self._variant(row, budget, ctx, result)

    def _variant(
        self,
        row: Mapping[str, Any],
        budget: int,
        ctx: PackagerContext,
        result: _Extraction,
    ) -> EvidenceVariant:
        evidence_text = "\n".join(sentence.text for sentence in result.accepted)
        kept_ids = tuple(
            dict.fromkeys(s.memory_id for s in result.accepted if s.memory_id)
        )
        return EvidenceVariant(
            example_id=row_example_id(row),
            style=self.style_name,
            budget=budget,
            evidence_text=evidence_text,
            token_count=count_tokens(evidence_text, encoding=ctx.encoding),
            source_memory_ids=kept_ids,
            meta={
                "n_candidates": result.candidate_count,
                "n_sentences": result.sentence_count,
                "n_selected": len(result.accepted),
                "order": "source_chronology",
                "overflow": result.overflow,
            },
        )


class SelectedExtractivePackager(ExtractivePackager):
    """Matched-content extraction over the memories selected by SIEVE-v1."""

    style_name = SELECTED_STYLE_NAME

    def _candidates(
        self, row: Mapping[str, Any], ctx: PackagerContext
    ) -> list[dict[str, Any]]:
        return selected_candidate_memories(row, ctx.sieve_entry)
