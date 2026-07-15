"""Regression guards for the P0 correctness bugs (TODO.md P0 list).

These lock in the fixes for the three P0 bugs that corrupt the per-question-type
metrics the v2 response surface depends on:

  P0 #2 - LLM-summarize baseline must count tokens with tiktoken (token_count),
          never str.split().
  P0 #3 - The baseline reader output must actually SET quality_norm and
          exact_correct (they are read by the summary writer).
  P0 #6 - The judge dedup key must include question_type so identical
          (question, ref, gen) across different question_types do not collide.

These are AST/structural checks so they run offline without API keys.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _module_source(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _function_source(rel: str, func_name: str) -> str:
    """Return the source text of a top-level function via AST (no import).

    Avoids importing the module (which may pull in optional deps like tiktoken
    that are not present in every environment).
    """
    src = _module_source(rel)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return ast.get_source_segment(src, node) or ""
    raise AssertionError(f"function {func_name!r} not found in {rel}")


class TestP0_2_TokenCountNotSplit:
    def test_summarize_baseline_uses_token_count(self):
        src = _module_source("cli/summarize_baseline.py")
        # Imports the tiktoken-backed counter...
        assert "from controller.logic import" in src and "token_count" in src
        # ...and uses it for selected_memory_tokens (efficiency metric).
        assert "token_count(str(t)" in src or "token_count(str(t))" in src

    def test_no_split_based_token_counting(self):
        # Guard: the baseline compile step counts memory tokens via token_count,
        # not len(...split()). (str.split() elsewhere for parsing is fine.)
        src = _function_source("cli/summarize_baseline.py", "_compile_row")
        assert ".split()" not in src
        assert "token_count(" in src


class TestP0_3_BaselineKeysExist:
    def test_read_row_sets_quality_norm_and_exact_correct(self):
        src = _function_source("cli/summarize_baseline.py", "_read_row")
        # The keys must be produced (set), not just read with .get() elsewhere.
        assert '"quality_norm"' in src
        assert '"exact_correct"' in src

    def test_summary_writer_reads_the_same_keys(self):
        src = _function_source("cli/summarize_baseline.py", "_write_summary_markdown")
        assert "quality_norm" in src
        assert "exact_correct" in src


class TestP0_6_JudgeDedupKeyIncludesQuestionType:
    def test_dedup_key_has_question_type(self):
        src = _module_source("cli/judge.py")
        tree = ast.parse(src)

        # Find every 4-element tuple that includes o["question_type"] alongside
        # question/reference_answer/generated_answer -> the dedup/lookup key.
        found_key_with_qtype = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Tuple) or len(node.elts) < 4:
                continue
            subscripts = {
                elt.slice.value
                for elt in node.elts
                if isinstance(elt, ast.Subscript)
                and isinstance(elt.slice, ast.Constant)
                and isinstance(elt.slice.value, str)
            }
            calls = {
                elt.args[0].value
                for elt in node.elts
                if isinstance(elt, ast.Call)
                and getattr(elt.func, "attr", "") == "get"
                and elt.args
                and isinstance(elt.args[0], ast.Constant)
            }
            keys = subscripts | calls
            if {"question", "reference_answer", "generated_answer"} & keys and "question_type" in keys:
                found_key_with_qtype = True
        assert found_key_with_qtype, (
            "judge dedup key must include question_type to avoid the P0 #6 "
            "collision where identical (question, ref, gen) across different "
            "question_types get the wrong judge prompt."
        )
