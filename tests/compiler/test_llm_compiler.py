"""Tests for compiler/llm_compiler.py pure functions."""

from __future__ import annotations

import pytest

from compiler.llm_compiler import (
    _detect_query_type,
    _select_prompt,
    _strip_think_tags,
    _parse_json_response,
    _date_context,
    format_llm_evidence,
    _ASSISTANT_RECALL_PROMPT,
    _KNOWLEDGE_UPDATE_PROMPT,
    _COUNTING_PROMPT,
    _ORDERING_PROMPT,
    _DEFAULT_PROMPT,
)


# ---------------------------------------------------------------------------
# _detect_query_type
# ---------------------------------------------------------------------------

class TestDetectQueryType:
    def test_assistant_recall_you_suggested(self):
        assert _detect_query_type("you suggested a restaurant") == "assistant_recall"

    def test_assistant_recall_you_mentioned(self):
        assert _detect_query_type("you mentioned something about Python") == "assistant_recall"

    def test_assistant_recall_you_told_me(self):
        assert _detect_query_type("you told me to try that") == "assistant_recall"

    def test_counting_how_many(self):
        assert _detect_query_type("how many dogs do I have") == "counting"

    def test_counting_number_of(self):
        assert _detect_query_type("what is the number of items") == "counting"

    def test_knowledge_update_most_recent(self):
        assert _detect_query_type("what is my most recent address") == "knowledge_update"

    def test_knowledge_update_currently(self):
        assert _detect_query_type("where do I currently live") == "knowledge_update"

    def test_default_simple_question(self):
        assert _detect_query_type("what is my dog's name") == "default"

    def test_ordering_chronological_order(self):
        assert _detect_query_type("list events in chronological order") == "ordering"

    def test_ordering_earliest_to_latest(self):
        assert _detect_query_type("from earliest to latest, what happened") == "ordering"

    def test_ordering_which_happened_first(self):
        assert _detect_query_type("which happened first") == "ordering"

    # -----------------------------------------------------------------------
    # Known bug: ordering queries that don't match _ORDERING_PATTERNS but
    # match _COUNTING_PATTERNS get misclassified as "counting" because
    # counting patterns are checked before ordering-like semantics.
    #
    # This test documents the bug: ordering queries with "total" or "how many"
    # in them would be misclassified. More importantly, ordering queries that
    # DON'T contain explicit ordering keywords but use ordering language
    # (like "list ... in order") could be ambiguous.
    # -----------------------------------------------------------------------

    @pytest.mark.xfail(
        reason="Known bug: ordering queries misclassified as counting when "
               "they use ordering language not covered by _ORDERING_PATTERNS "
               "but contain counting-trigger words like 'how many'",
    )
    def test_ordering_with_how_many_not_counting(self):
        # This query asks about ordering ("in what order") but "how many" triggers
        # counting first. The ordering patterns don't include "in what order".
        query = "how many trips did I take and in what order"
        result = _detect_query_type(query)
        assert result != "counting", f"Got '{result}' but query asks about ordering"
        assert result == "ordering"

    def test_assistant_recall_takes_priority_over_counting(self):
        # assistant_recall is checked first, so this should work
        query = "you mentioned how many items to buy"
        assert _detect_query_type(query) == "assistant_recall"

    def test_case_insensitive(self):
        assert _detect_query_type("HOW MANY cats") == "counting"
        assert _detect_query_type("YOU SUGGESTED a plan") == "assistant_recall"


# ---------------------------------------------------------------------------
# _select_prompt
# ---------------------------------------------------------------------------

class TestSelectPrompt:
    def test_assistant_recall(self):
        assert _select_prompt("assistant_recall") is _ASSISTANT_RECALL_PROMPT

    def test_knowledge_update(self):
        assert _select_prompt("knowledge_update") is _KNOWLEDGE_UPDATE_PROMPT

    def test_counting(self):
        assert _select_prompt("counting") is _COUNTING_PROMPT

    def test_ordering(self):
        assert _select_prompt("ordering") is _ORDERING_PROMPT

    def test_default(self):
        assert _select_prompt("default") is _DEFAULT_PROMPT

    def test_unknown_falls_back_to_default(self):
        assert _select_prompt("nonexistent_type") is _DEFAULT_PROMPT


# ---------------------------------------------------------------------------
# _strip_think_tags
# ---------------------------------------------------------------------------

class TestStripThinkTags:
    def test_removes_think_block(self):
        text = '<think>some reasoning</think>{"facts": []}'
        assert _strip_think_tags(text) == '{"facts": []}'

    def test_no_think_tags(self):
        text = '{"facts": ["a"]}'
        assert _strip_think_tags(text) == '{"facts": ["a"]}'

    def test_multiline_think_block(self):
        text = "<think>\nline 1\nline 2\n</think>\nresult"
        assert _strip_think_tags(text) == "result"

    def test_multiple_think_blocks(self):
        text = "<think>a</think>middle<think>b</think>end"
        assert _strip_think_tags(text) == "middleend"

    def test_empty_think_block(self):
        text = "<think></think>content"
        assert _strip_think_tags(text) == "content"


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_clean_json(self):
        raw = '{"facts": ["a", "b"], "confidence": "high"}'
        result = _parse_json_response(raw)
        assert result["facts"] == ["a", "b"]
        assert result["confidence"] == "high"

    def test_json_with_think_tags(self):
        raw = '<think>reasoning here</think>{"facts": ["x"]}'
        result = _parse_json_response(raw)
        assert result["facts"] == ["x"]

    def test_json_with_surrounding_text(self):
        raw = 'Here is the result: {"facts": ["y"]} end'
        result = _parse_json_response(raw)
        assert result["facts"] == ["y"]

    def test_invalid_json_returns_raw(self):
        raw = "this is not json at all"
        result = _parse_json_response(raw)
        assert "raw" in result

    def test_empty_string(self):
        result = _parse_json_response("")
        assert "raw" in result

    def test_nested_json(self):
        raw = '{"facts": ["a"], "temporal": null, "confidence": "low"}'
        result = _parse_json_response(raw)
        assert result["temporal"] is None
        assert result["confidence"] == "low"


# ---------------------------------------------------------------------------
# _date_context
# ---------------------------------------------------------------------------

class TestDateContext:
    def test_with_date(self):
        result = _date_context("2025-06-15")
        assert "2025-06-15" in result
        assert "relative time" in result.lower() or "Resolve" in result

    def test_none_returns_empty(self):
        assert _date_context(None) == ""

    def test_empty_string_returns_empty(self):
        assert _date_context("") == ""


# ---------------------------------------------------------------------------
# format_llm_evidence
# ---------------------------------------------------------------------------

class TestFormatLlmEvidence:
    def test_basic_formatting(self):
        evidence = {
            "facts": ["fact1", "fact2"],
            "temporal": "2025-01-01",
            "answer_hint": "the answer",
            "confidence": "high",
        }
        result = format_llm_evidence(evidence)
        assert result["compiler_type"] == "llm"
        assert result["pool_augmented"] is False
        assert "answer_hint" in result["slots"]
        assert result["slots"]["answer_hint"]["display_text"] == "the answer"
        # 2 facts + 1 temporal line
        assert len(result["evidence_lines"]) == 3

    def test_no_answer_hint(self):
        evidence = {"facts": ["f1"], "temporal": None, "answer_hint": None}
        result = format_llm_evidence(evidence)
        assert "answer_hint" not in result["slots"]

    def test_no_temporal(self):
        evidence = {"facts": ["f1"], "temporal": None, "answer_hint": "a"}
        result = format_llm_evidence(evidence)
        assert len(result["evidence_lines"]) == 1

    def test_empty_facts(self):
        evidence = {"facts": [], "temporal": None, "answer_hint": None}
        result = format_llm_evidence(evidence)
        assert len(result["evidence_lines"]) == 0
        assert result["slots"] == {}

    def test_evidence_line_text_field(self):
        evidence = {"facts": ["the dog is brown"], "temporal": None, "answer_hint": None}
        result = format_llm_evidence(evidence)
        assert result["evidence_lines"][0]["text"] == "the dog is brown"
