"""Evidence relevance scorers for the v2 packagers.

Phase 1 ships :class:`StemOverlapScorer`, a thin wrapper over v1's stem-overlap
relevance machinery (``shared.nlp`` / the same signal ``retrieval.proposals``
uses) so ``filtered_raw`` and ``extractive`` run end-to-end offline. Phase 2
will add :class:`CrossEncoderScorer` (a real cross-encoder), which is stubbed
here with the planned model named in its docstring.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EvidenceScorer(Protocol):
    """Scores passages against a query. Higher score == more relevant."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return one relevance score per passage (same length, same order)."""
        ...


class StemOverlapScorer:
    """Lexical stem-overlap relevance, reusing v1 NLP utilities.

    The score of a passage is the fraction of the query's content-word stems
    that appear (via stem equivalence / conversational synonyms) in the
    passage. This is the same family of signal used by
    ``retrieval.proposals._target_matches_text`` and is fully deterministic and
    offline. It is intentionally simple: Phase 1 only needs a stable ordering,
    not a strong ranker.
    """

    name = "stem_overlap"

    def score(self, query: str, passages: list[str]) -> list[float]:
        from shared.nlp import check_stem_equivalence, get_clean_tokens

        query_tokens = get_clean_tokens(str(query or ""))
        if not query_tokens:
            return [0.0 for _ in passages]

        scores: list[float] = []
        for passage in passages:
            passage_tokens = get_clean_tokens(str(passage or ""))
            if not passage_tokens:
                scores.append(0.0)
                continue
            matched = 0
            for q_tok in query_tokens:
                if any(
                    check_stem_equivalence(q_tok, p_tok) for p_tok in passage_tokens
                ):
                    matched += 1
            scores.append(matched / len(query_tokens))
        return scores

    def __call__(self, query: str, passages: list[str]) -> list[float]:
        """Callable form so the scorer can be passed as a ``RelevanceScorer``."""
        return self.score(query, passages)


class CrossEncoderScorer:
    """Phase-2 stub: cross-encoder relevance scorer.

    Planned implementation: a small cross-encoder (``cross-encoder/ms-marco-
    MiniLM-L-6-v2`` for a fast baseline, or a fine-tuned ``deberta-v3-small``
    query/passage relevance head) scoring each (query, passage) pair. It will
    replace :class:`StemOverlapScorer` inside ``filtered_raw`` / ``extractive``
    and feed the completeness estimator described in ``research/idea-v2.md``.

    It remains an explicit later-phase component so Phase 1 has no heavy or
    GPU-only import path and the test-suite stays offline.
    """

    name = "cross_encoder"

    def __init__(self, model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_id = model_id

    def score(self, query: str, passages: list[str]) -> list[float]:  # pragma: no cover - stub
        raise NotImplementedError(
            "CrossEncoderScorer is a Phase-2 stub. Planned model: "
            f"{self.model_id} (or a fine-tuned deberta-v3-small relevance head). "
            "Use StemOverlapScorer for Phase-1 response-surface runs."
        )
