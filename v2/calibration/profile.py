"""Reader capability profile z_r (Phase-2 stub with real interfaces).

``ReaderProfile`` is the scientific object of v2: a six-dimensional capability
vector keyed by an OPAQUE ``reader_handle`` (never a model ID — conditioning on
model IDs memorises training readers and fails on new ones; research/idea-v2.md
"What NOT to do"). :func:`compute_profile` aggregates probe results into the
vector; :func:`save_profiles` / :func:`load_profiles` persist a battery of them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from v2.calibration.probes import PROBE_DIMENSIONS


@dataclass(frozen=True)
class ReaderProfile:
    """Six-float capability vector for a reader, keyed by an opaque handle.

    The floats are in probe-score units (typically [0, 1]); higher is more
    capable on that dimension. ``reader_handle`` is opaque on purpose: the
    policy conditions on the *profile*, not the model identity.
    """

    reader_handle: str
    distractor_tolerance: float
    multi_hop: float
    temporal: float
    paraphrase_robustness: float
    context_depth: float
    abstention_calibration: float

    def __post_init__(self) -> None:
        for dimension, value in zip(PROBE_DIMENSIONS, self.as_vector()):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{dimension} must be finite and within [0, 1]")

    def as_vector(self) -> tuple[float, ...]:
        """Return the six dimensions in :data:`PROBE_DIMENSIONS` order."""
        return tuple(getattr(self, dim) for dim in PROBE_DIMENSIONS)


def compute_profile(
    probe_results: Mapping[str, float],
    *,
    reader_handle: str,
) -> ReaderProfile:
    """Aggregate per-dimension probe scores into a :class:`ReaderProfile`.

    Expects *probe_results* to already be reduced to one score per dimension.
    Missing dimensions fail loudly so an incomplete calibration cannot silently
    masquerade as a weak reader.

    The separate probe-running harness is responsible for producing
    *probe_results*.
    """
    missing = [dimension for dimension in PROBE_DIMENSIONS if dimension not in probe_results]
    if missing:
        raise ValueError(f"Missing reader-profile dimensions: {missing}")
    values = {dimension: float(probe_results[dimension]) for dimension in PROBE_DIMENSIONS}
    return ReaderProfile(reader_handle=reader_handle, **values)


def save_profiles(path: str | Path, profiles: Mapping[str, ReaderProfile]) -> None:
    """Persist a mapping of reader_handle -> ReaderProfile as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {handle: asdict(profile) for handle, profile in sorted(profiles.items())}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_profiles(path: str | Path) -> dict[str, ReaderProfile]:
    """Load a mapping of reader_handle -> ReaderProfile from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {handle: ReaderProfile(**fields) for handle, fields in data.items()}
