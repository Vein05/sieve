"""Exact published SIEVE-v1 system control."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from v2.budget import count_tokens
from v2.packagers.base import PackagerContext, row_example_id
from v2.types import EvidenceVariant

STYLE_NAME = "sieve_v1"


class SieveV1Packager:
    """Replay cached reader prompts and deterministic answer routes exactly."""

    style_name = STYLE_NAME

    def package(
        self, row: Mapping[str, Any], budget: int, ctx: PackagerContext
    ) -> EvidenceVariant:
        entry = ctx.sieve_entry
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"sieve_v1 requires a cache entry for {row_example_id(row)!r}"
            )
        prompt = str(entry.get("prompt") or "")
        deterministic = bool(entry.get("is_deterministic"))
        answer = str(entry.get("deterministic_answer") or "")
        if not prompt.strip() and not (deterministic and answer):
            raise ValueError(
                f"sieve_v1 cache entry has no route for {row_example_id(row)!r}"
            )
        evidence = prompt or "\n".join(
            str(text) for text in entry.get("selected_memory_texts") or []
        )
        source_ids = tuple(str(mid) for mid in entry.get("selected_memory_ids") or [])
        return EvidenceVariant(
            example_id=row_example_id(row),
            style=self.style_name,
            budget=budget,
            evidence_text=evidence,
            token_count=count_tokens(evidence, encoding=ctx.encoding),
            prompt_override=prompt or None,
            answer_override=answer if deterministic else None,
            source_memory_ids=source_ids,
            meta={
                "full_prompt_control": True,
                "deterministic_route": deterministic,
                "reader_route_reason": entry.get("reader_route_reason"),
                "compiler_type": entry.get("compiler_type"),
                "uncapped": True,
            },
        )
