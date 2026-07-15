"""Fail-loud capability profile tests."""

from __future__ import annotations

import pytest

from v2.calibration.probes import PROBE_DIMENSIONS
from v2.calibration.profile import compute_profile
from v2.policy import HeuristicTierPolicy


def _scores(value: float = 0.5) -> dict[str, float]:
    return {dimension: value for dimension in PROBE_DIMENSIONS}


def test_profile_requires_every_dimension():
    scores = _scores()
    scores.pop(PROBE_DIMENSIONS[0])
    with pytest.raises(ValueError, match="Missing"):
        compute_profile(scores, reader_handle="reader")


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_profile_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        compute_profile(_scores(value), reader_handle="reader")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.2, ("structured", 400)), (0.5, ("extractive", 400)), (0.8, ("filtered_raw", 800))],
)
def test_heuristic_policy_uses_profile(value, expected):
    profile = compute_profile(_scores(value), reader_handle="reader")
    assert HeuristicTierPolicy().choose({}, profile) == expected


def test_heuristic_policy_rejects_missing_profile():
    with pytest.raises(ValueError):
        HeuristicTierPolicy().choose({}, None)
