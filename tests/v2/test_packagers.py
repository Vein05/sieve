"""Tests for the six v2 packagers and the style registry.

All offline: no network, no real LLM. The summary packager uses a fake
compile-LLM callable injected via PackagerContext.
"""

from __future__ import annotations

import pytest

from v2.packagers import STYLE_REGISTRY, OFFLINE_STYLES, get_packager
from v2.packagers.base import PackagerContext
from v2.scorer import StemOverlapScorer
from v2.types import EvidenceVariant

from tests.v2.conftest import fake_compile_llm


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registered_styles(self):
        assert set(STYLE_REGISTRY) == {
            "raw", "filtered_raw", "extractive", "selected_raw",
            "selected_extractive", "structured", "structured_quotes",
            "structured_generic", "guarded_structured",
            "summary", "sieve_v1",
        }

    def test_get_packager_returns_matching_style(self):
        for style in STYLE_REGISTRY:
            assert get_packager(style).style_name == style

    def test_unknown_style_raises(self):
        with pytest.raises(KeyError):
            get_packager("does_not_exist")


def _ctx(sieve_entry=None, naive_entry=None, compile_llm=None) -> PackagerContext:
    return PackagerContext(
        sieve_entry=sieve_entry,
        naive_entry=naive_entry,
        relevance_scorer=StemOverlapScorer(),
        compile_llm=compile_llm,
        params={},
        encoding="cl100k_base",
    )


# ---------------------------------------------------------------------------
# raw
# ---------------------------------------------------------------------------

class TestRawPackager:
    def test_respects_budget(self, mini_slice_rows):
        row = mini_slice_rows[0]
        v = get_packager("raw").package(row, budget=200, ctx=_ctx())
        assert isinstance(v, EvidenceVariant)
        assert v.style == "raw"
        assert v.token_count <= 200 or v.meta["overflow"]

    def test_large_budget_reproduces_all_candidates(self, mini_slice_rows):
        row = mini_slice_rows[0]
        v = get_packager("raw").package(row, budget=100_000, ctx=_ctx())
        # At an effectively unbounded budget the control keeps every candidate.
        assert v.meta["n_kept"] == v.meta["n_candidates"]

    def test_zero_budget_is_uncapped_control(self, mini_slice_rows):
        row = mini_slice_rows[0]
        v = get_packager("raw").package(row, budget=0, ctx=_ctx())
        assert v.meta["n_kept"] == v.meta["n_candidates"]
        assert v.meta["uncapped"] is True

    def test_output_lines_are_verbatim(self, mini_slice_rows):
        row = mini_slice_rows[0]
        v = get_packager("raw").package(row, budget=400, ctx=_ctx())
        contents = {str(c["content"]).strip() for c in row["candidate_memories"]}
        for line in v.evidence_text.split("\n"):
            if line.strip():
                assert line in contents


# ---------------------------------------------------------------------------
# filtered_raw
# ---------------------------------------------------------------------------

class TestFilteredRawPackager:
    def test_every_line_is_verbatim_substring_of_some_memory(self, mini_slice_rows):
        row = mini_slice_rows[0]
        v = get_packager("filtered_raw").package(row, budget=400, ctx=_ctx())
        memories = [str(c["content"]) for c in row["candidate_memories"]]
        for line in v.evidence_text.split("\n"):
            if line.strip():
                assert any(line in m for m in memories), line

    def test_respects_budget(self, mini_slice_rows):
        row = mini_slice_rows[0]
        v = get_packager("filtered_raw").package(row, budget=200, ctx=_ctx())
        assert v.token_count <= 200 or v.meta["overflow"]

    def test_is_subset_of_raw(self, mini_slice_rows):
        row = mini_slice_rows[0]
        raw = get_packager("raw").package(row, budget=100_000, ctx=_ctx())
        filt = get_packager("filtered_raw").package(row, budget=100_000, ctx=_ctx())
        assert set(filt.source_memory_ids).issubset(set(raw.source_memory_ids))


# ---------------------------------------------------------------------------
# extractive
# ---------------------------------------------------------------------------

class TestExtractivePackager:
    def test_sentences_are_verbatim_substrings(self, mini_slice_rows):
        row = mini_slice_rows[0]
        v = get_packager("extractive").package(row, budget=400, ctx=_ctx())
        memories = [str(c["content"]) for c in row["candidate_memories"]]
        for line in v.evidence_text.split("\n"):
            if line.strip():
                assert any(line in m for m in memories), line

    def test_respects_budget(self, mini_slice_rows):
        row = mini_slice_rows[0]
        v = get_packager("extractive").package(row, budget=200, ctx=_ctx())
        assert v.token_count <= 200 or v.meta["overflow"]

    def test_output_declares_source_chronology(self, mini_slice_rows):
        row = mini_slice_rows[0]
        v = get_packager("extractive").package(row, budget=800, ctx=_ctx())
        assert v.meta["order"] == "source_chronology"


class TestMatchedContentPackagers:
    @pytest.mark.parametrize("style", ["selected_raw", "selected_extractive"])
    def test_only_uses_v1_selected_ids(
        self, style, mini_slice_rows, sieve_entries_by_id
    ):
        row = mini_slice_rows[0]
        entry = sieve_entries_by_id[row["example_id"]]
        selected = set(entry.get("selected_memory_ids") or [])
        variant = get_packager(style).package(
            row, budget=400, ctx=_ctx(sieve_entry=entry)
        )
        assert set(variant.source_memory_ids).issubset(selected)


# ---------------------------------------------------------------------------
# structured (against real sampled cache entry)
# ---------------------------------------------------------------------------

class TestStructuredPackager:
    def test_uses_cached_rendered_package(self, mini_slice_rows, sieve_entries_by_id):
        row = mini_slice_rows[0]
        entry = sieve_entries_by_id[row["example_id"]]
        v = get_packager("structured").package(row, budget=400, ctx=_ctx(sieve_entry=entry))
        assert v.style == "structured"
        assert v.meta["has_sieve_entry"] is True
        # Content should derive from the cached evidence lines / slots.
        rep = entry["rendered_evidence_package"]
        line_texts = [ln["text"] for ln in rep.get("evidence_lines", []) if ln.get("text")]
        if line_texts:
            assert any(lt in v.evidence_text for lt in line_texts)

    def test_no_sieve_entry_is_empty(self, mini_slice_rows):
        row = mini_slice_rows[0]
        v = get_packager("structured").package(row, budget=400, ctx=_ctx(sieve_entry=None))
        assert v.evidence_text == ""
        assert v.meta["has_sieve_entry"] is False

    def test_route_without_rendered_package_uses_effective_fallback(
        self, mini_slice_rows, sieve_entries_by_id
    ):
        row = mini_slice_rows[0]
        entry = dict(sieve_entries_by_id[row["example_id"]])
        entry["rendered_evidence_package"] = {}
        entry["selected_memory_texts"] = ["fallback evidence"]
        v = get_packager("structured").package(
            row, budget=400, ctx=_ctx(sieve_entry=entry)
        )
        assert v.evidence_text == "fallback evidence"
        assert v.meta["representation_mode"] == "v1_route_fallback"

    def test_respects_budget(self, mini_slice_rows, sieve_entries_by_id):
        # Use the largest fixture entry to exercise truncation.
        for eid, entry in sieve_entries_by_id.items():
            row = next(r for r in mini_slice_rows if r["example_id"] == eid)
            v = get_packager("structured").package(row, budget=50, ctx=_ctx(sieve_entry=entry))
            assert v.token_count <= 50 or v.meta["overflow"]


# ---------------------------------------------------------------------------
# structured_quotes
# ---------------------------------------------------------------------------

class TestStructuredQuotesPackager:
    def test_quotes_are_verbatim(self, mini_slice_rows, sieve_entries_by_id):
        row = mini_slice_rows[0]
        entry = sieve_entries_by_id[row["example_id"]]
        v = get_packager("structured_quotes").package(
            row, budget=800, ctx=_ctx(sieve_entry=entry)
        )
        memories = [str(c["content"]) for c in row["candidate_memories"]]
        # Any quote-block line must be a verbatim substring of some memory.
        in_quotes = False
        for line in v.evidence_text.split("\n"):
            if line == "Exact quotes:":
                in_quotes = True
                continue
            if line == "Structured:":
                in_quotes = False
                continue
            if in_quotes and line.strip():
                assert any(line in m for m in memories), line

    def test_budget_split_meta(self, mini_slice_rows, sieve_entries_by_id):
        row = mini_slice_rows[0]
        entry = sieve_entries_by_id[row["example_id"]]
        v = get_packager("structured_quotes").package(
            row, budget=400, ctx=_ctx(sieve_entry=entry)
        )
        assert v.token_count <= 400
        assert v.meta["n_structured_kept"] >= 1


# ---------------------------------------------------------------------------
# summary (fake compile-LLM; never touches network)
# ---------------------------------------------------------------------------

class TestSummaryPackager:
    def test_uses_injected_llm(self, mini_slice_rows):
        row = mini_slice_rows[0]
        v = get_packager("summary").package(
            row, budget=200, ctx=_ctx(compile_llm=fake_compile_llm)
        )
        assert v.style == "summary"
        assert "Summary" in v.evidence_text
        assert v.token_count <= 200 or v.meta["overflow"]

    def test_missing_llm_raises(self, mini_slice_rows):
        row = mini_slice_rows[0]
        with pytest.raises(ValueError):
            get_packager("summary").package(row, budget=200, ctx=_ctx(compile_llm=None))

    def test_uses_canonical_summarize_prompt(self, mini_slice_rows):
        # Must match the published LLM-Summarize baseline prompt; the earlier
        # one-line prompt cost 8-18pp vs canonical summaries on ConvoMem.
        captured: dict[str, str] = {}

        def spy_llm(prompt: str, max_tokens: int) -> str:
            captured["prompt"] = prompt
            return "Summary."

        row = mini_slice_rows[0]
        get_packager("summary").package(row, budget=200, ctx=_ctx(compile_llm=spy_llm))
        prompt = captured["prompt"]
        assert "You are a precise evidence extractor." in prompt
        assert "Include dates, names, numbers" in prompt
        assert 'say "No relevant evidence found."' in prompt
        assert "Relevant evidence summary:" in prompt
        assert "[Memory 1]" in prompt

    def test_summary_never_builds_client(self):
        # Structural guarantee: the module must not IMPORT a network client.
        import ast
        import inspect
        import v2.packagers.summary as mod

        tree = ast.parse(inspect.getsource(mod))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        assert "reader.client" not in imported
        assert not any(m.startswith("reader.client") for m in imported)


class TestSieveV1Control:
    def test_uses_exact_cached_prompt(self, mini_slice_rows, sieve_entries_by_id):
        row = mini_slice_rows[0]
        entry = sieve_entries_by_id[row["example_id"]]
        variant = get_packager("sieve_v1").package(
            row, budget=0, ctx=_ctx(sieve_entry=entry)
        )
        assert variant.prompt_override == entry["prompt"]
        assert variant.evidence_text == entry["prompt"]
        assert variant.meta["full_prompt_control"] is True

    def test_preserves_deterministic_route(self, mini_slice_rows, sieve_entries_by_id):
        row = mini_slice_rows[0]
        entry = dict(sieve_entries_by_id[row["example_id"]])
        entry.update(
            prompt="",
            is_deterministic=True,
            deterministic_answer="exact cached answer",
        )
        variant = get_packager("sieve_v1").package(
            row, budget=0, ctx=_ctx(sieve_entry=entry)
        )
        assert variant.prompt_override is None
        assert variant.answer_override == "exact cached answer"
        assert variant.meta["deterministic_route"] is True


# ---------------------------------------------------------------------------
# offline styles determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    @pytest.mark.parametrize("style", sorted(OFFLINE_STYLES))
    def test_repeatable(self, style, mini_slice_rows, sieve_entries_by_id):
        row = mini_slice_rows[0]
        entry = sieve_entries_by_id.get(row["example_id"])
        ctx = _ctx(sieve_entry=entry)
        a = get_packager(style).package(row, budget=400, ctx=ctx)
        b = get_packager(style).package(row, budget=400, ctx=ctx)
        assert a.evidence_text == b.evidence_text
        assert a.token_count == b.token_count
        assert a.source_memory_ids == b.source_memory_ids
