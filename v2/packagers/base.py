"""Packager protocol and shared context for evidence policies and controls.

A *packager* turns one dataset row (plus optional v1 compilation cache entries)
into a single :class:`~v2.types.EvidenceVariant` at a requested token budget.
Every style is a small, deterministic transform; the compile-LLM callable (used
only by the ``summary`` style) is injected via :class:`PackagerContext` so tests
never construct a network client.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from v2.types import EvidenceVariant

# A relevance scorer: (query, passages) -> one score per passage. Higher means
# more relevant. Concrete implementation lives in v2/scorer.py (StemOverlapScorer).
RelevanceScorer = Callable[[str, list[str]], list[float]]

# A compile-LLM callable: (prompt, max_tokens) -> generated text. Injected so the
# summary packager never builds its own client. In tests this is a fake.
CompileLLM = Callable[[str, int], str | Mapping[str, Any]]


@dataclass(frozen=True)
class PackagerContext:
    """Per-example context passed to every packager.

    Attributes:
        sieve_entry: the v1 SIEVE compilation-cache entry for this example_id
            (schema: ``prompt``, ``rendered_evidence_package``,
            ``selected_memory_ids``, ...), or ``None`` if absent.
        naive_entry: the v1 naive compilation-cache entry, or ``None``.
        relevance_scorer: scores passages against the query (stem overlap by
            default; a cross-encoder later). Reused by filtered_raw/extractive.
        compile_llm: injected compile-LLM callable for the ``summary`` style.
            ``None`` for styles that never need it; the summary packager raises
            a clear error if it is missing.
        params: style-specific parameters from the response-surface config
            (e.g. relative_threshold, structured_fraction).
        encoding: tiktoken encoding name for budget accounting.
    """

    sieve_entry: Mapping[str, Any] | None = None
    naive_entry: Mapping[str, Any] | None = None
    relevance_scorer: RelevanceScorer | None = None
    compile_llm: CompileLLM | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    encoding: str = "cl100k_base"


@runtime_checkable
class Packager(Protocol):
    """A single evidence-representation style.

    Implementations are stateless and deterministic: given the same row,
    budget, and context they must produce the same variant (temperature 0,
    sorted iteration, tiktoken counting).
    """

    style_name: str

    def package(
        self, row: Mapping[str, Any], budget: int, ctx: PackagerContext
    ) -> EvidenceVariant:
        """Produce one :class:`EvidenceVariant` for *row* at *budget* tokens."""
        ...


# ---------------------------------------------------------------------------
# Shared row helpers (field names verified against the v1 slice + caches).
# ---------------------------------------------------------------------------

def row_query(row: Mapping[str, Any]) -> str:
    """Return the query text. The v1 slice uses ``query`` (not ``question``);
    the compilation cache entries use ``question``. Accept either."""
    return str(row.get("query") or row.get("question") or "").strip()


def row_example_id(row: Mapping[str, Any]) -> str:
    """Return the example identifier."""
    return str(row.get("example_id") or "").strip()


def candidate_memories(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return candidate memories in retrieval (rank) order.

    Each candidate is a dict with ``content`` and ``memory_id`` (BM25 top-20).
    """
    cands = row.get("candidate_memories") or []
    out: list[dict[str, Any]] = []
    for c in cands:
        if isinstance(c, dict) and str(c.get("content", "")).strip():
            out.append(c)
    return out


def selected_candidate_memories(
    row: Mapping[str, Any],
    sieve_entry: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return row candidates selected by v1, preserving candidate order."""
    if not isinstance(sieve_entry, Mapping):
        return []
    selected = {str(mid) for mid in sieve_entry.get("selected_memory_ids") or []}
    return [
        candidate
        for candidate in candidate_memories(row)
        if str(candidate.get("memory_id", "")) in selected
    ]
