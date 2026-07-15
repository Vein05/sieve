"""Representation policy: pick (style, budget) for a query given a reader z_r.

Phase 1 ships deterministic baseline policies (``FixedPolicy``,
``HeuristicTierPolicy``) so the response surface can be swept with fixed and
simple-router settings. ``LearnedPolicy`` is the Phase-2 target (a small GBM/MLP
over query features and z_r) and is stubbed with a descriptive
``NotImplementedError``.

The policy is where the v2 thesis lives: it conditions on reader capability, not
model identity (research/idea-v2.md).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from v2.calibration.profile import ReaderProfile

WEAK_PROFILE_MAX = 0.4
STRONG_PROFILE_MIN = 0.7


@runtime_checkable
class RepresentationPolicy(Protocol):
    """Choose an evidence (style, budget) for a query and reader profile."""

    def choose(
        self, query_features: Mapping[str, Any], z_r: ReaderProfile | None
    ) -> tuple[str, int]:
        """Return the chosen ``(style, budget)``."""
        ...


@dataclass(frozen=True)
class FixedPolicy:
    """Always choose the same ``(style, budget)`` — a fixed-representation ablation."""

    style: str
    budget: int

    def choose(
        self, query_features: Mapping[str, Any], z_r: ReaderProfile | None
    ) -> tuple[str, int]:
        return self.style, self.budget


@dataclass(frozen=True)
class HeuristicTierPolicy:
    """Placeholder reader-tier router keyed off mean calibrated capability.

    Maps a coarse reader-capability tier (``weak`` / ``mid`` / ``strong``,
    bucketed from the six-dimensional profile) to a hand-set
    (style, budget). This is a deliberately weak placeholder for the learned
    policy: it exercises the reader-adaptive code path so the response surface
    can compare a simple router against oracle-adaptive. It is NOT the
    contribution — the learned policy is (see :class:`LearnedPolicy`).

    Defaults (from research/idea-v2.md's tier -> style table):
        weak   -> structured @ 400
        mid    -> extractive @ 400
        strong -> filtered_raw @ 800
    """

    weak: tuple[str, int] = ("structured", 400)
    mid: tuple[str, int] = ("extractive", 400)
    strong: tuple[str, int] = ("filtered_raw", 800)

    def choose(
        self, query_features: Mapping[str, Any], z_r: ReaderProfile | None
    ) -> tuple[str, int]:
        if z_r is None:
            raise ValueError("HeuristicTierPolicy requires a ReaderProfile")
        capability = sum(z_r.as_vector()) / len(z_r.as_vector())
        if capability < WEAK_PROFILE_MAX:
            tier = "weak"
        elif capability >= STRONG_PROFILE_MIN:
            tier = "strong"
        else:
            tier = "mid"
        return {"weak": self.weak, "mid": self.mid, "strong": self.strong}.get(
            tier, self.mid
        )


class LearnedPolicy:
    """Phase-2 stub: learned representation policy over (query features, z_r).

    Planned implementation: a small model (gradient-boosted trees or a compact
    MLP) that maps ``(query features, capability profile z_r)`` to a
    ``(style, budget)`` decision, trained on the response-surface utility
    matrix (downstream reader utility, NOT v1 selected/rejected labels;
    research/idea-v2.md "Training signal"). Deliberately simple: the
    contribution is the conditioning signal (z_r), not model complexity.

    Training is intentionally deferred until the response surface establishes
    that oracle-adaptive beats best-fixed by the go/no-go margin.
    """

    def __init__(self, model: Any | None = None) -> None:
        self.model = model

    def choose(
        self, query_features: Mapping[str, Any], z_r: ReaderProfile | None
    ) -> tuple[str, int]:  # pragma: no cover - stub
        raise NotImplementedError(
            "LearnedPolicy is a Phase-2 stub. Planned: a GBM/MLP over "
            "(query features, z_r) trained on the response-surface utility "
            "matrix. Use FixedPolicy or HeuristicTierPolicy for Phase-1 sweeps."
        )
