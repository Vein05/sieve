"""Summary packager: generic abstractive summary via an injected compile-LLM.

Produces a query-focused abstractive summary of the evidence by calling the
compile-LLM callable injected on :class:`PackagerContext`. The packager NEVER
constructs its own network client — the callable is wired at compile time by
the runner (from ``reader.client``) and is a deterministic fake in tests.

The summary text is then budget-truncated on whole-line boundaries so its token
count respects the requested budget.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from v2.budget import count_tokens, hard_truncate_text, truncate_units_to_budget
from v2.packagers.base import (
    PackagerContext,
    candidate_memories,
    row_example_id,
    row_query,
)
from v2.types import EvidenceVariant

STYLE_NAME = "summary"
_DEFAULT_MAX_TOKENS_FLOOR = 64

# Canonical LLM-Summarize prompt from the published baseline
# (predecessor repo release/scripts/run_llm_summarize_baseline.py); the prior
# one-line prompt cost 8-18pp vs canonical summaries on ConvoMem.
_SUMMARY_PROMPT = """\
You are a precise evidence extractor. Given a user question and retrieved conversation memories, extract ONLY the information relevant to answering the question. Output a concise summary of the relevant evidence.

Rules:
- Include dates, names, numbers, and specific details that help answer the question.
- Omit greetings, small talk, and unrelated conversation turns.
- If multiple memories contain relevant information, combine them.
- If no memory is relevant, say "No relevant evidence found."
- Be concise — aim for 2-5 sentences maximum.

Question: {question}

Retrieved memories:
{evidence}

Relevant evidence summary:"""


@dataclass(frozen=True)
class _RenderedSummary:
    text: str
    overflow: bool


def _response_parts(response: Any) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(response, Mapping):
        return str(response or "").strip(), {}
    text = str(
        response.get("text")
        or response.get("clean_answer")
        or response.get("raw_answer")
        or ""
    ).strip()
    return text, response


def _render_summary(summary: str, budget: int, encoding: str) -> _RenderedSummary:
    lines = [line for line in summary.splitlines() if line.strip()]
    units = lines or ([summary] if summary else [])
    packing = truncate_units_to_budget(units, budget, encoding=encoding)
    text = "\n".join(packing.units)
    if budget > 0:
        text = hard_truncate_text(text, budget, encoding=encoding)
    return _RenderedSummary(text, packing.overflow or summary != text)


class SummaryPackager:
    """Generic abstractive summary via the injected compile-LLM callable."""

    style_name = STYLE_NAME

    def package(
        self, row: Mapping[str, Any], budget: int, ctx: PackagerContext
    ) -> EvidenceVariant:
        if ctx.compile_llm is None:
            raise ValueError(
                "SummaryPackager requires ctx.compile_llm (the injected compile-LLM "
                "callable). It is wired by the runner's compile step; it is never "
                "constructed inside the packager."
            )

        query = row_query(row)
        candidates = candidate_memories(row)
        evidence = "\n\n".join(
            f"[Memory {i}]\n{str(c['content']).strip()}"
            for i, c in enumerate(candidates, 1)
            if str(c["content"]).strip()
        )
        prompt = _SUMMARY_PROMPT.format(question=query, evidence=evidence)

        floor = int(ctx.params.get("max_tokens_floor", _DEFAULT_MAX_TOKENS_FLOOR))
        # A generous max-token hint tied to the budget; the compile-LLM may
        # produce fewer, and we budget-truncate the result regardless.
        max_tokens_hint = max(floor, int(budget))

        summary_text, usage = _response_parts(
            ctx.compile_llm(prompt, max_tokens_hint)
        )
        rendered = _render_summary(summary_text, budget, ctx.encoding)
        compiler_config = ctx.params.get("compile_model") or {}

        return EvidenceVariant(
            example_id=row_example_id(row),
            style=self.style_name,
            budget=budget,
            evidence_text=rendered.text,
            token_count=count_tokens(rendered.text, encoding=ctx.encoding),
            source_memory_ids=tuple(str(c.get("memory_id", "")) for c in candidates),
            meta={
                "n_candidates": len(candidates),
                "max_tokens_hint": max_tokens_hint,
                "compiler_model": compiler_config.get("model"),
                "compiler_provider": compiler_config.get("provider"),
                "compiler_input_tokens": usage.get("prompt_tokens")
                or count_tokens(prompt, encoding=ctx.encoding),
                "compiler_output_tokens": usage.get("completion_tokens")
                or count_tokens(summary_text, encoding=ctx.encoding),
                "compiler_latency_ms": usage.get("latency_ms"),
                "raw_summary_tokens": count_tokens(summary_text, encoding=ctx.encoding),
                "overflow": rendered.overflow,
            },
        )
