"""Tests for the Phase-2 interface stubs: policy, calibration, scorer."""

from __future__ import annotations

import pytest

from v2.calibration import (
    PROBE_DIMENSIONS,
    ProbeSpec,
    ReaderProfile,
    compute_profile,
    load_probe_battery,
    load_profiles,
    save_profiles,
)
from v2.policy import FixedPolicy, HeuristicTierPolicy, LearnedPolicy
from v2.scorer import CrossEncoderScorer, StemOverlapScorer


class TestScorer:
    def test_stem_overlap_orders_relevant_first(self):
        scorer = StemOverlapScorer()
        scores = scorer.score(
            "what breed is my dog",
            ["My dog is a golden retriever breed", "I ate a sandwich today"],
        )
        assert scores[0] > scores[1]

    def test_empty_query(self):
        assert StemOverlapScorer().score("", ["anything"]) == [0.0]

    def test_cross_encoder_is_stub(self):
        with pytest.raises(NotImplementedError):
            CrossEncoderScorer().score("q", ["p"])


class TestPolicy:
    def test_fixed_policy(self):
        p = FixedPolicy("structured", 400)
        assert p.choose({}, None) == ("structured", 400)

    def test_heuristic_tier_defaults(self):
        p = HeuristicTierPolicy()
        weak = compute_profile({d: 0.2 for d in PROBE_DIMENSIONS}, reader_handle="w")
        mid = compute_profile({d: 0.5 for d in PROBE_DIMENSIONS}, reader_handle="m")
        strong = compute_profile({d: 0.8 for d in PROBE_DIMENSIONS}, reader_handle="s")
        assert p.choose({}, weak) == ("structured", 400)
        assert p.choose({}, mid) == ("extractive", 400)
        assert p.choose({}, strong) == ("filtered_raw", 800)
        with pytest.raises(ValueError):
            p.choose({"reader_tier": "weak"}, None)

    def test_learned_policy_is_stub(self):
        with pytest.raises(NotImplementedError):
            LearnedPolicy().choose({}, None)


class TestCalibration:
    def test_probe_dimensions(self):
        assert len(PROBE_DIMENSIONS) == 6
        assert "distractor_tolerance" in PROBE_DIMENSIONS

    def test_probe_spec_validates_dimension(self):
        with pytest.raises(ValueError):
            ProbeSpec("p1", "not_a_dimension", "{question}", "score_fn")

    def test_empty_battery(self):
        assert load_probe_battery() == []

    def test_compute_profile_and_vector(self):
        scores = {dimension: 0.5 for dimension in PROBE_DIMENSIONS}
        scores["distractor_tolerance"] = 0.8
        prof = compute_profile(scores, reader_handle="rh-1")
        assert isinstance(prof, ReaderProfile)
        assert prof.reader_handle == "rh-1"
        assert prof.distractor_tolerance == 0.8
        assert prof.temporal == 0.5
        assert len(prof.as_vector()) == 6

    def test_save_load_profiles_roundtrip(self, tmp_path):
        prof = compute_profile({d: 0.5 for d in PROBE_DIMENSIONS}, reader_handle="rh")
        path = tmp_path / "profiles.json"
        save_profiles(path, {"rh": prof})
        loaded = load_profiles(path)
        assert loaded["rh"] == prof
