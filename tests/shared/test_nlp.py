"""Tests for shared/nlp.py pure utility functions."""

from __future__ import annotations

import pytest

from shared.nlp import (
    NLPProfile,
    stem_token,
    canonical_attribute,
    canonical_attributes_for_text,
    check_stem_equivalence,
    get_clean_tokens,
    get_stems_for_text,
    append_with_budget,
    CandidateView,
    evenly_spaced_indices,
    requested_item_count,
    is_event_ordering_query,
    _normalized_v2_query_family,
    v2_budget_family_hint,
    NOISE_TOKENS,
    CONVERSATIONAL_SYNONYMS,
    CANONICAL_ATTRIBUTE_ALIASES,
)

# ---------------------------------------------------------------------------
# stem_token
# ---------------------------------------------------------------------------

class TestStemToken:
    def test_plural_nouns(self):
        assert stem_token("dogs") == "dog"

    def test_verb_conjugation(self):
        assert stem_token("running") == "run"

    def test_already_base_form(self):
        assert stem_token("cat") == "cat"

    def test_empty_string(self):
        assert stem_token("") == ""

    def test_whitespace_stripped(self):
        assert stem_token("  cats  ") == "cat"

    def test_uppercase_lowered(self):
        assert stem_token("Dogs") == "dog"

    def test_non_string_returns_empty(self):
        assert stem_token(None) == ""  # type: ignore[arg-type]

    def test_adjective_lemmatization(self):
        result = stem_token("happier")
        assert result in {"happy", "happier"}


# ---------------------------------------------------------------------------
# canonical_attribute
# ---------------------------------------------------------------------------

class TestCanonicalAttribute:
    def test_direct_match(self):
        assert canonical_attribute("breed") == "breed"

    def test_alias_match_occupation(self):
        assert canonical_attribute("job") == "occupation"

    def test_alias_match_name(self):
        assert canonical_attribute("name") == "name"

    def test_alias_match_named(self):
        assert canonical_attribute("named") == "name"

    def test_no_match(self):
        assert canonical_attribute("xylophone") is None

    def test_speed_alias(self):
        assert canonical_attribute("bandwidth") == "speed"


# ---------------------------------------------------------------------------
# canonical_attributes_for_text -- needs spaCy for tokenization
# ---------------------------------------------------------------------------

class TestCanonicalAttributesForText:
    def test_extracts_breed_from_sentence(self):
        attrs = canonical_attributes_for_text("what breed is my dog")
        assert "breed" in attrs

    def test_extracts_occupation(self):
        attrs = canonical_attributes_for_text("what is your job title")
        assert "occupation" in attrs

    def test_empty_string(self):
        attrs = canonical_attributes_for_text("")
        assert attrs == set()

    def test_no_attributes(self):
        attrs = canonical_attributes_for_text("hello world")
        assert attrs == set()


# ---------------------------------------------------------------------------
# check_stem_equivalence
# ---------------------------------------------------------------------------

class TestCheckStemEquivalence:
    def test_identical_tokens(self):
        assert check_stem_equivalence("dog", "dog") is True

    def test_plural_and_singular(self):
        assert check_stem_equivalence("dogs", "dog") is True

    def test_synonym_pair_buy_purchase(self):
        assert check_stem_equivalence("buy", "purchase") is True

    def test_synonym_pair_job_work(self):
        assert check_stem_equivalence("job", "work") is True

    def test_unrelated_tokens(self):
        assert check_stem_equivalence("cat", "mountain") is False

    def test_empty_first(self):
        assert check_stem_equivalence("", "dog") is False

    def test_empty_second(self):
        assert check_stem_equivalence("dog", "") is False

    def test_both_empty(self):
        assert check_stem_equivalence("", "") is False

    def test_canonical_alias_equivalence(self):
        # "job" and "profession" both map to canonical "occupation"
        assert check_stem_equivalence("job", "profession") is True


# ---------------------------------------------------------------------------
# get_clean_tokens
# ---------------------------------------------------------------------------

class TestGetCleanTokens:
    def test_basic_tokenization(self):
        tokens = get_clean_tokens("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_noise_removal(self):
        tokens = get_clean_tokens("the cat in the hat", exclude_noise=True)
        assert "the" not in tokens
        assert "in" not in tokens
        assert "cat" in tokens
        assert "hat" in tokens

    def test_noise_kept(self):
        tokens = get_clean_tokens("the cat", exclude_noise=False)
        assert "the" in tokens
        assert "cat" in tokens

    def test_empty_string(self):
        assert get_clean_tokens("") == []

    def test_punctuation_removed(self):
        tokens = get_clean_tokens("hello, world!")
        for t in tokens:
            assert t.isalnum()

    def test_possessive_removed(self):
        tokens = get_clean_tokens("John's cat", exclude_noise=True)
        assert "'s" not in tokens


# ---------------------------------------------------------------------------
# get_stems_for_text
# ---------------------------------------------------------------------------

class TestGetStemsForText:
    def test_basic_stems(self):
        stems = get_stems_for_text("running dogs")
        assert "run" in stems
        assert "dog" in stems

    def test_empty(self):
        assert get_stems_for_text("") == []

    def test_noise_excluded(self):
        stems = get_stems_for_text("the cat")
        # "the" is noise, should be excluded
        assert "the" not in stems


# ---------------------------------------------------------------------------
# evenly_spaced_indices
# ---------------------------------------------------------------------------

class TestEvenlySpacedIndices:
    def test_target_equals_total(self):
        assert evenly_spaced_indices(5, 5) == [0, 1, 2, 3, 4]

    def test_target_greater_than_total(self):
        assert evenly_spaced_indices(3, 10) == [0, 1, 2]

    def test_target_less_than_total(self):
        result = evenly_spaced_indices(10, 3)
        assert len(result) == 3
        assert result[0] == 0
        assert result[-1] == 9
        assert all(0 <= i < 10 for i in result)

    def test_zero_total(self):
        assert evenly_spaced_indices(0, 5) == []

    def test_zero_target(self):
        assert evenly_spaced_indices(5, 0) == []

    def test_single_target(self):
        result = evenly_spaced_indices(10, 1)
        assert len(result) == 1
        assert result[0] == 0

    def test_two_from_ten(self):
        result = evenly_spaced_indices(10, 2)
        assert len(result) == 2
        assert result[0] == 0
        assert result[-1] == 9

    def test_result_is_sorted(self):
        result = evenly_spaced_indices(20, 7)
        assert result == sorted(result)

    def test_no_duplicates(self):
        result = evenly_spaced_indices(20, 7)
        assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# requested_item_count
# ---------------------------------------------------------------------------

class TestRequestedItemCount:
    def test_digit_match(self):
        assert requested_item_count("mention 3 items") == 3

    def test_word_match(self):
        assert requested_item_count("list five items") == 5

    def test_no_match(self):
        assert requested_item_count("what is the weather") is None

    def test_include_digit(self):
        assert requested_item_count("include 7 items from the list") == 7

    def test_word_twelve(self):
        assert requested_item_count("mention twelve items") == 12

    def test_word_one(self):
        assert requested_item_count("list one items") == 1


# ---------------------------------------------------------------------------
# is_event_ordering_query
# ---------------------------------------------------------------------------

class TestIsEventOrderingQuery:
    def test_harm_type_event_ordering(self):
        row = {"harm_type": "event_ordering", "query": "some question"}
        assert is_event_ordering_query(row) is True

    def test_query_contains_in_order(self):
        row = {"query": "list the events in order", "harm_type": ""}
        assert is_event_ordering_query(row) is True

    def test_query_contains_order_in_which(self):
        row = {"query": "what is the order in which they graduated", "harm_type": ""}
        assert is_event_ordering_query(row) is True

    def test_not_ordering(self):
        row = {"query": "what is my dog's name", "harm_type": ""}
        assert is_event_ordering_query(row) is False

    def test_missing_harm_type(self):
        row = {"query": "hello world"}
        assert is_event_ordering_query(row) is False


# ---------------------------------------------------------------------------
# append_with_budget
# ---------------------------------------------------------------------------

def _make_candidate(memory_id: str, token_count: int) -> CandidateView:
    return CandidateView(
        memory_id=memory_id,
        text=f"text_{memory_id}",
        rank=1,
        token_count=token_count,
        profile=None,
        date_key=(0, 0, 0),
        has_update_marker=False,
        normalized_text=f"text_{memory_id}",
        content_tokens=set(),
    )


class TestAppendWithBudget:
    def test_append_within_budget(self):
        selected: list[CandidateView] = []
        ok, used = append_with_budget(
            selected, _make_candidate("a", 10),
            token_budget=100, max_selected=5, used_tokens=0,
        )
        assert ok is True
        assert used == 10
        assert len(selected) == 1

    def test_reject_over_budget(self):
        selected: list[CandidateView] = [_make_candidate("x", 50)]
        ok, used = append_with_budget(
            selected, _make_candidate("b", 60),
            token_budget=100, max_selected=5, used_tokens=50,
        )
        assert ok is False
        assert used == 50
        assert len(selected) == 1

    def test_reject_duplicate(self):
        selected = [_make_candidate("a", 10)]
        ok, used = append_with_budget(
            selected, _make_candidate("a", 10),
            token_budget=100, max_selected=5, used_tokens=10,
        )
        assert ok is False
        assert len(selected) == 1

    def test_reject_max_selected_reached(self):
        selected = [_make_candidate("a", 10)]
        ok, used = append_with_budget(
            selected, _make_candidate("b", 10),
            token_budget=100, max_selected=1, used_tokens=10,
        )
        assert ok is False
        assert len(selected) == 1

    def test_no_budget_always_appends(self):
        selected: list[CandidateView] = []
        ok, used = append_with_budget(
            selected, _make_candidate("a", 999),
            token_budget=None, max_selected=5, used_tokens=0,
        )
        assert ok is True
        assert used == 999

    def test_first_item_allowed_even_if_over_budget(self):
        selected: list[CandidateView] = []
        ok, used = append_with_budget(
            selected, _make_candidate("a", 200),
            token_budget=50, max_selected=5, used_tokens=0,
        )
        assert ok is True
        assert used == 200
        assert len(selected) == 1

    def test_exact_budget_boundary(self):
        selected: list[CandidateView] = []
        ok, used = append_with_budget(
            selected, _make_candidate("a", 100),
            token_budget=100, max_selected=5, used_tokens=0,
        )
        assert ok is True
        assert used == 100


# ---------------------------------------------------------------------------
# _normalized_v2_query_family
# ---------------------------------------------------------------------------

class TestNormalizedV2QueryFamily:
    def test_none_returns_default(self):
        assert _normalized_v2_query_family(None) == "default"

    def test_empty_returns_default(self):
        assert _normalized_v2_query_family("") == "default"

    def test_whitespace_returns_default(self):
        assert _normalized_v2_query_family("   ") == "default"

    def test_normal_family(self):
        assert _normalized_v2_query_family("temporal") == "temporal"

    def test_uppercase_lowered(self):
        assert _normalized_v2_query_family("TEMPORAL") == "temporal"

    def test_whitespace_stripped(self):
        assert _normalized_v2_query_family("  ordering  ") == "ordering"


# ---------------------------------------------------------------------------
# v2_budget_family_hint
# ---------------------------------------------------------------------------

class TestV2BudgetFamilyHint:
    def test_abstention(self):
        row = {"evidence_sufficiency": {"label": "abstention"}}
        assert v2_budget_family_hint(row) == "abstention"

    def test_event_ordering_via_harm_type(self):
        row = {"harm_type": "event_ordering", "query": "list events"}
        assert v2_budget_family_hint(row) == "ordering"

    def test_contradiction(self):
        row = {"harm_type": "", "ability": "contradiction", "query": "q"}
        assert v2_budget_family_hint(row) == "contradiction"

    def test_temporal_reasoning(self):
        row = {"harm_type": "", "ability": "temporal_reasoning", "query": "q"}
        assert v2_budget_family_hint(row) == "temporal"

    def test_information_extraction(self):
        row = {"harm_type": "", "ability": "information_extraction", "query": "q"}
        assert v2_budget_family_hint(row) == "information_extraction"

    def test_default_fallback(self):
        row = {"harm_type": "", "ability": "", "query": "what is my name"}
        assert v2_budget_family_hint(row) == "default"

    def test_knowledge_update_alias(self):
        row = {"harm_type": "", "ability": "stale_update", "query": "q"}
        assert v2_budget_family_hint(row) == "knowledge_update"

    def test_multi_session(self):
        row = {"harm_type": "", "ability": "multi_session_reasoning", "query": "q"}
        assert v2_budget_family_hint(row) == "multi_session"


# ---------------------------------------------------------------------------
# NLPProfile dataclass
# ---------------------------------------------------------------------------

class TestNLPProfile:
    def test_creation(self):
        p = NLPProfile(tokens=("hello",), tags=("NN",), stems=("hello",))
        assert p.tokens == ("hello",)
        assert p.tags == ("NN",)
        assert p.stems == ("hello",)

    def test_frozen(self):
        p = NLPProfile(tokens=(), tags=(), stems=())
        with pytest.raises(AttributeError):
            p.tokens = ("new",)  # type: ignore[misc]

    def test_equality(self):
        a = NLPProfile(tokens=("a",), tags=("NN",), stems=("a",))
        b = NLPProfile(tokens=("a",), tags=("NN",), stems=("a",))
        assert a == b

    def test_empty(self):
        p = NLPProfile(tokens=(), tags=(), stems=())
        assert len(p.tokens) == 0


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------

class TestConstants:
    def test_noise_tokens_contains_the(self):
        assert "the" in NOISE_TOKENS

    def test_noise_tokens_contains_possessive(self):
        assert "'s" in NOISE_TOKENS

    def test_synonyms_are_bidirectional_for_price_cost(self):
        assert "cost" in CONVERSATIONAL_SYNONYMS["price"]
        assert "price" in CONVERSATIONAL_SYNONYMS["cost"]

    def test_canonical_aliases_occupation_includes_job(self):
        assert "job" in CANONICAL_ATTRIBUTE_ALIASES["occupation"]
