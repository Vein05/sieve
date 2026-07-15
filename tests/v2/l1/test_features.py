"""Tests for v2.l1.features: feature computation and degenerate inputs."""

from __future__ import annotations

import pytest

from v2.l1.features import FeatureVector, extract, feature_names, to_array


class TestExtractFeatures:
    def test_basic_stem_coverage(self):
        fv = extract("What degree did I graduate with?", ["I graduated with a Business Administration degree."])
        assert 0.0 < fv.stem_coverage <= 1.0

    def test_empty_query_returns_zeros(self):
        fv = extract("", ["Some text here."])
        assert fv.stem_coverage == 0.0
        assert fv.content_word_coverage == 0.0  # no query signal to measure

    def test_empty_pack_returns_zeros(self):
        fv = extract("What degree did I graduate with?", [])
        assert fv.stem_coverage == 0.0
        assert fv.n_units == 0.0

    def test_empty_pack_text_strings(self):
        fv = extract("Who directed Big Stone Gap?", ["", ""])
        assert fv.stem_coverage == 0.0
        assert fv.n_units == 2.0

    def test_n_units_matches_pack_size(self):
        fv = extract("query", ["a", "b", "c"])
        assert fv.n_units == 3.0

    def test_pack_token_count_positive_for_nonempty_pack(self):
        fv = extract("query", ["hello world foo"])
        assert fv.pack_token_count > 0.0

    def test_pack_query_length_ratio(self):
        fv = extract("one two three", ["one two three four five six seven eight nine ten"])
        assert fv.pack_query_length_ratio > 1.0

    def test_entity_coverage_nonzero_when_entity_in_pack(self):
        fv = extract("Who is Albert Einstein?", ["Albert Einstein was a physicist."])
        # "Albert" and "Einstein" are capitalized — entity coverage should be > 0.
        assert fv.entity_coverage > 0.0

    def test_date_coverage_nonzero_when_date_in_pack(self):
        fv = extract("What happened in 2023?", ["In 2023, many things occurred."])
        assert fv.date_coverage > 0.0

    def test_max_unit_overlap_ge_mean_unit_overlap(self):
        fv = extract("What is the capital of France?", ["Paris is the capital.", "London is not."])
        assert fv.max_unit_overlap >= fv.mean_unit_overlap

    def test_perfect_coverage_single_unit(self):
        query = "What degree did graduate"
        # Pack contains all query stems.
        fv = extract(query, ["What degree did graduate"])
        assert fv.stem_coverage == pytest.approx(1.0, abs=0.01)

    def test_whitespace_only_query(self):
        fv = extract("   ", ["Some text."])
        assert fv.stem_coverage == 0.0

    def test_returns_feature_vector_instance(self):
        fv = extract("query text", ["passage text"])
        assert isinstance(fv, FeatureVector)


class TestFeatureNamesAndToArray:
    def test_feature_names_length(self):
        names = feature_names()
        assert len(names) == 9  # 9 features defined in FeatureVector

    def test_to_array_length_matches_feature_names(self):
        fv = extract("test query", ["some text"])
        arr = to_array(fv)
        assert len(arr) == len(feature_names())

    def test_to_array_all_floats(self):
        fv = extract("test query", ["some text"])
        arr = to_array(fv)
        assert all(isinstance(v, float) for v in arr)

    def test_first_feature_is_stem_coverage(self):
        names = feature_names()
        assert names[0] == "stem_coverage"

    def test_to_array_values_nonnegative(self):
        fv = extract("What happened?", ["Something happened in 2023."])
        arr = to_array(fv)
        assert all(v >= 0.0 for v in arr)
