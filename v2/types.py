"""Core value types for the v2 response surface.

These frozen dataclasses are the atomic units of the reader x representation
experiment. ``EvidenceVariant`` is one evidence package for a single
(example_id, style, budget) cell; ``VariantKey`` is its stable cache key; and
``ReaderResult`` is the outcome of running one reader against one variant.

Style conventions follow v1's ``compiler/planner`` (frozen dataclasses for
value types, explicit fields, no behaviour). See ``research/idea-v2.md`` for
the six styles and three budgets.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


def _freeze_meta(meta: Mapping[str, object] | None) -> Mapping[str, object]:
    """Return an immutable, sorted-key view of *meta* for frozen storage."""
    if not meta:
        return MappingProxyType({})
    return MappingProxyType({str(k): meta[k] for k in sorted(meta)})


@dataclass(frozen=True)
class VariantKey:
    """Stable identity of one response-surface cell.

    The string form ``example_id::style::budget`` is used as the cache key and
    as the shard/record identifier in ``results/v2_runs`` so downstream tooling
    (and ``cli/judge.py``) can round-trip it without re-parsing structure.
    """

    example_id: str
    style: str
    budget: int

    def as_str(self) -> str:
        """Deterministic, collision-free string key."""
        return f"{self.example_id}::{self.style}::{self.budget}"

    @classmethod
    def from_str(cls, key: str) -> "VariantKey":
        """Inverse of :meth:`as_str`."""
        example_id, style, budget = key.split("::")
        return cls(example_id=example_id, style=style, budget=int(budget))


@dataclass(frozen=True)
class EvidenceVariant:
    """One compiled evidence package for a single (example_id, style, budget).

    ``evidence_text`` is the reader-facing evidence payload. ``prompt_override``
    is reserved for exact full-pipeline controls such as the published SIEVE-v1
    prompt; ``answer_override`` preserves v1 deterministic routes that did not
    call a reader. Ordinary representation cells leave both ``None``. ``token_count`` is
    measured with tiktoken via
    :func:`v2.budget.count_tokens` (never ``.split()``). ``source_memory_ids``
    records provenance for auditing verbatim-substring guarantees.
    """

    example_id: str
    style: str
    budget: int
    evidence_text: str
    token_count: int
    prompt_override: str | None = None
    answer_override: str | None = None
    source_memory_ids: tuple[str, ...] = ()
    meta: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        # Normalise meta to an immutable, sorted view so equality/hashing are
        # deterministic and the dataclass stays effectively frozen.
        object.__setattr__(self, "meta", _freeze_meta(self.meta))
        object.__setattr__(self, "source_memory_ids", tuple(self.source_memory_ids))

    @property
    def key(self) -> VariantKey:
        """The :class:`VariantKey` identifying this variant."""
        return VariantKey(self.example_id, self.style, self.budget)


@dataclass(frozen=True)
class ReaderResult:
    """Outcome of running one reader model against one :class:`EvidenceVariant`.

    Carries the variant key fields plus the reader identity and generation
    result. ``error`` is ``None`` on success and a short string on failure; the
    runner never bare-excepts (P0 #4 lesson) so failures are always recorded.
    """

    example_id: str
    style: str
    budget: int
    reader_model: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    error: str | None = None
    serving_provider: str | None = None
    served_model: str | None = None

    @property
    def key(self) -> VariantKey:
        """The :class:`VariantKey` identifying the source variant."""
        return VariantKey(self.example_id, self.style, self.budget)
