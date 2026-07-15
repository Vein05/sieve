"""Prompt controls and durable replay IO for the v2 response surface."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from v2.types import EvidenceVariant, ReaderResult, VariantKey
from v2.variant_cache import VariantCache

RUN_IDENTITY_FILE = "run_identity.json"

SHARED_READER_TEMPLATE = (
    "You are a careful conversational memory assistant.\n"
    "Answer the user's question using ONLY the evidence provided below.\n"
    "Rules:\n"
    "  - Return only the answer span: a name, phrase, number, date, or list.\n"
    "  - Preserve full names and modifiers; do not paraphrase the answer.\n"
    "  - If the evidence is insufficient, answer exactly: Unknown.\n\n"
    "Question: {question}\n"
    "{question_date_block}"
    "\nEvidence:\n{evidence}\n\n"
    "Answer:"
)


def build_reader_prompt(
    *, question: str, question_date: str | None, evidence_text: str
) -> str:
    """Build the one shared prompt used by factorial representation cells."""
    date_block = f"Question date: {question_date}\n" if question_date else ""
    return SHARED_READER_TEMPLATE.format(
        question=str(question or "").strip(),
        question_date_block=date_block,
        evidence=str(evidence_text or "").strip() or "(no evidence)",
    )


def output_record(
    variant: EvidenceVariant,
    row: Mapping[str, Any],
    reader_model: str,
    result: ReaderResult,
    base_system: str,
) -> dict[str, Any]:
    """Assemble one judge-compatible, collision-free condition record."""
    condition = f"{base_system}::{reader_model}::{variant.style}::{variant.budget}"
    return {
        "system": condition,
        "base_system": base_system,
        "example_id": variant.example_id,
        "question": str(row.get("query") or row.get("question") or ""),
        "reference_answer": str(row.get("reference_answer") or ""),
        "generated_answer": result.answer,
        "question_type": str(row.get("question_type") or ""),
        "dataset_name": str(row.get("dataset_name") or "longmemeval"),
        "harm_type": str(row.get("harm_type") or ""),
        "style": variant.style,
        "budget": variant.budget,
        "reader_model": reader_model,
        "query_split": row.get("v2_split"),
        "evidence_token_count": variant.token_count,
        "source_memory_ids": list(variant.source_memory_ids),
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "latency_ms": result.latency_ms,
        "serving_provider": result.serving_provider,
        "served_model": result.served_model,
        "compiler_input_tokens": variant.meta.get("compiler_input_tokens"),
        "compiler_output_tokens": variant.meta.get("compiler_output_tokens"),
        "compiler_latency_ms": variant.meta.get("compiler_latency_ms"),
        "deterministic_route": variant.answer_override is not None,
        "success": result.error is None,
        "error": result.error,
    }


def shard_path(run_dir: Path, key: VariantKey, reader: str) -> Path:
    """Return a filesystem-safe shard path for one reader-condition pair."""
    reader_slug = reader.replace("/", "__").replace(":", "-")
    key_slug = key.as_str().replace("::", "__")
    return run_dir / "shards" / f"{key_slug}__{reader_slug}.json"


def existing_pairs(run_dir: Path) -> set[tuple[str, str]]:
    """Return completed variant-reader pairs from readable shards."""
    done: set[tuple[str, str]] = set()
    for shard in (run_dir / "shards").glob("*.json"):
        try:
            record = json.loads(shard.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not record.get("success", False):
            continue
        key = VariantKey(
            record["example_id"], record["style"], int(record["budget"])
        )
        done.add((key.as_str(), str(record["reader_model"])))
    return done


def ensure_run_identity(
    run_dir: Path,
    cache: VariantCache,
    readers: Sequence[str],
    system: str,
    generation: Mapping[str, Any],
) -> None:
    """Refuse to resume shards from a different experimental condition."""
    template_hash = hashlib.sha256(SHARED_READER_TEMPLATE.encode()).hexdigest()[:16]
    identity = {
        "cache_fingerprint": cache.fingerprint,
        "readers": sorted(set(readers)),
        "system": system,
        "shared_prompt_hash": template_hash,
        "generation": dict(generation),
    }
    path = run_dir / RUN_IDENTITY_FILE
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError(
                "Run directory belongs to a different cache, reader panel, or "
                "generation configuration; choose a new run directory."
            )
        return
    if any((run_dir / "shards").glob("*.json")):
        raise ValueError("Existing shards have no run identity; refusing unsafe resume")
    path.write_text(json.dumps(identity, indent=2, sort_keys=True), encoding="utf-8")


def write_shard(
    run_dir: Path,
    key: VariantKey,
    reader: str,
    record: Mapping[str, Any],
) -> None:
    """Persist one completed pair atomically enough for resumable experiments."""
    shard_path(run_dir, key, reader).write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def collect_outputs(run_dir: Path) -> list[dict[str, Any]]:
    """Read all shards into deterministic output order."""
    records: list[dict[str, Any]] = []
    for shard in sorted((run_dir / "shards").glob("*.json")):
        try:
            records.append(json.loads(shard.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    records.sort(
        key=lambda record: (
            str(record.get("example_id")),
            str(record.get("style")),
            int(record.get("budget", 0)),
            str(record.get("reader_model")),
        )
    )
    return records
