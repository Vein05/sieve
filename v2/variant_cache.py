"""Build, integrity-check, and load the compile-once v2 variant cache."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from v2.packagers import PackagerContext, get_packager
from v2.scorer import StemOverlapScorer
from v2.types import EvidenceVariant, VariantKey


def slice_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash exact query and ordered candidate contents for cache integrity."""
    canonical_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: str(r.get("example_id", ""))):
        candidates = [
            {
                "memory_id": str(candidate.get("memory_id", "")),
                "content": str(candidate.get("content", "")),
            }
            for candidate in row.get("candidate_memories", [])
            if isinstance(candidate, Mapping)
        ]
        canonical_rows.append(
            {
                "example_id": str(row.get("example_id", "")),
                "query": str(row.get("query") or row.get("question") or ""),
                "question_date": str(row.get("question_date", "")),
                "reference_answer": str(row.get("reference_answer", "")),
                "candidate_memories": candidates,
            }
        )
    canonical = json.dumps(canonical_rows, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def config_fingerprint(config: Mapping[str, Any], slice_fp: str) -> str:
    """sha256 over the sorted config plus the slice fingerprint (like v1)."""
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(f"{canonical}|{slice_fp}".encode()).hexdigest()[:16]


@dataclass(frozen=True)
class VariantCache:
    """Loaded variant cache: header fingerprint + variants by VariantKey string."""

    fingerprint: str
    slice_fingerprint: str
    styles: tuple[str, ...]
    budgets: tuple[int, ...]
    style_budgets: Mapping[str, tuple[int, ...]]
    entries: dict[str, EvidenceVariant]

    def get(self, key: VariantKey) -> EvidenceVariant | None:
        return self.entries.get(key.as_str())


@dataclass(frozen=True)
class VariantBuildSpec:
    """Configuration and optional v1 inputs for one cache build."""

    styles: Sequence[str]
    budgets: Sequence[int]
    config: Mapping[str, Any]
    sieve_entries: Mapping[str, Mapping[str, Any]] | None = None
    naive_entries: Mapping[str, Mapping[str, Any]] | None = None
    compile_llm: Callable[[str, int], Any] | None = None

    def fingerprint_config(self) -> dict[str, Any]:
        """Return config plus the effective command-line build grid."""
        merged = dict(self.config)
        merged["_variant_build"] = {
            "styles": list(self.styles),
            "budgets": [int(value) for value in self.budgets],
        }
        return merged


def _context_base(
    example_id: str,
    spec: VariantBuildSpec,
    scorer: StemOverlapScorer,
    encoding: str,
) -> dict[str, Any]:
    return {
        "sieve_entry": (spec.sieve_entries or {}).get(example_id),
        "naive_entry": (spec.naive_entries or {}).get(example_id),
        "relevance_scorer": scorer,
        "compile_llm": spec.compile_llm,
        "encoding": encoding,
    }


def build_variants(
    rows: Sequence[Mapping[str, Any]],
    spec: VariantBuildSpec,
) -> dict[str, EvidenceVariant]:
    """Build every ``(row x style x budget)`` variant, keyed by VariantKey string.

    Iteration is fully sorted (rows by example_id, styles/budgets as given) for
    determinism. The ``summary`` style requires *compile_llm*; if it is absent
    and ``summary`` is requested, a clear error is raised (never a silent skip).
    """
    packager_params = dict(spec.config.get("packager_params", {}))
    style_budgets = dict(spec.config.get("style_budgets", {}))
    encoding = str(spec.config.get("tiktoken_encoding", "cl100k_base"))
    scorer = StemOverlapScorer()

    if "summary" in spec.styles and spec.compile_llm is None:
        raise ValueError(
            "The 'summary' style requires a compile_llm callable. Provide one "
            "(wired from reader.client at compile time) or drop 'summary' from "
            "--styles."
        )

    entries: dict[str, EvidenceVariant] = {}
    for row in sorted(rows, key=lambda r: str(r.get("example_id", ""))):
        eid = str(row.get("example_id", ""))
        ctx_base = _context_base(eid, spec, scorer, encoding)
        for style in spec.styles:
            packager = get_packager(style)
            params = dict(packager_params.get(style, {}))
            if style == "summary":
                params["compile_model"] = dict(spec.config.get("compile_model", {}))
            ctx = PackagerContext(
                params=params,
                **ctx_base,
            )
            cell_budgets = style_budgets.get(style, spec.budgets)
            for budget in dict.fromkeys(int(value) for value in cell_budgets):
                variant = packager.package(row, int(budget), ctx)
                if variant.key.as_str() in entries:
                    raise ValueError(f"Duplicate variant key: {variant.key.as_str()}")
                if budget > 0 and variant.token_count > budget:
                    raise ValueError(
                        f"Budget violation for {variant.key.as_str()}: "
                        f"{variant.token_count} > {budget}"
                    )
                entries[variant.key.as_str()] = variant
    return entries


def _variant_to_json(v: EvidenceVariant) -> dict[str, Any]:
    return {
        "example_id": v.example_id,
        "style": v.style,
        "budget": v.budget,
        "evidence_text": v.evidence_text,
        "token_count": v.token_count,
        "prompt_override": v.prompt_override,
        "answer_override": v.answer_override,
        "source_memory_ids": list(v.source_memory_ids),
        "meta": dict(v.meta),
    }


def _variant_from_json(d: Mapping[str, Any]) -> EvidenceVariant:
    return EvidenceVariant(
        example_id=str(d["example_id"]),
        style=str(d["style"]),
        budget=int(d["budget"]),
        evidence_text=str(d["evidence_text"]),
        token_count=int(d["token_count"]),
        prompt_override=(
            str(d["prompt_override"])
            if d.get("prompt_override") is not None
            else None
        ),
        answer_override=(
            str(d["answer_override"])
            if d.get("answer_override") is not None
            else None
        ),
        source_memory_ids=tuple(d.get("source_memory_ids", [])),
        meta=dict(d.get("meta", {})),
    )


def save_variant_cache(
    path: str | Path,
    entries: Mapping[str, EvidenceVariant],
    spec: VariantBuildSpec,
    slice_fp: str,
) -> str:
    """Write the variant cache to *path*; return the recorded fingerprint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = config_fingerprint(spec.fingerprint_config(), slice_fp)
    actual_style_budgets = {
        style: sorted(
            {variant.budget for variant in entries.values() if variant.style == style}
        )
        for style in spec.styles
    }
    actual_budgets = sorted(
        {budget for cell_budgets in actual_style_budgets.values() for budget in cell_budgets}
    )
    payload = {
        "fingerprint": fingerprint,
        "slice_fingerprint": slice_fp,
        "created": datetime.now(timezone.utc).isoformat(),
        "styles": list(spec.styles),
        "requested_budgets": [int(value) for value in spec.budgets],
        "budgets": actual_budgets,
        "style_budgets": actual_style_budgets,
        "row_count": len({k.split("::")[0] for k in entries}),
        "variant_count": len(entries),
        "entries": {k: _variant_to_json(v) for k, v in sorted(entries.items())},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return fingerprint


def load_variant_cache(
    path: str | Path,
    *,
    expected_config: Mapping[str, Any] | None = None,
    expected_slice_fp: str | None = None,
) -> VariantCache:
    """Load a cache and reject it when supplied integrity inputs disagree."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    stored_fp = str(data.get("fingerprint", ""))
    slice_fp = str(data.get("slice_fingerprint", ""))

    if expected_config is not None and expected_slice_fp is not None:
        fingerprint_config = dict(expected_config)
        fingerprint_config["_variant_build"] = {
            "styles": list(data.get("styles", [])),
            "budgets": list(data.get("requested_budgets", data.get("budgets", []))),
        }
        recomputed = config_fingerprint(fingerprint_config, expected_slice_fp)
        if recomputed != stored_fp:
            raise ValueError(
                f"Variant cache fingerprint mismatch: cache={stored_fp} "
                f"expected={recomputed}. Refusing to replay against a stale "
                f"cache; rebuild with --force."
            )

    entries = {
        k: _variant_from_json(v) for k, v in (data.get("entries") or {}).items()
    }
    return VariantCache(
        fingerprint=stored_fp,
        slice_fingerprint=slice_fp,
        styles=tuple(data.get("styles", [])),
        budgets=tuple(data.get("budgets", [])),
        style_budgets={
            str(style): tuple(int(value) for value in values)
            for style, values in (data.get("style_budgets") or {}).items()
        },
        entries=entries,
    )


def build_and_save(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    spec: VariantBuildSpec,
    *,
    force: bool = False,
) -> tuple[Path, str, bool]:
    """Build + save the variant cache idempotently.

    Returns ``(path, fingerprint, rebuilt)``. If the target already exists with
    a matching fingerprint and ``force`` is false, it is left untouched
    (``rebuilt=False``).
    """
    path = Path(path)
    slice_fp = slice_fingerprint(rows)
    target_fp = config_fingerprint(spec.fingerprint_config(), slice_fp)

    if path.exists() and not force:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if str(existing.get("fingerprint", "")) == target_fp:
                return path, target_fp, False
        except (OSError, json.JSONDecodeError):
            pass  # unreadable/corrupt cache -> rebuild

    entries = build_variants(rows, spec)
    fingerprint = save_variant_cache(
        path,
        entries,
        spec,
        slice_fp=slice_fp,
    )
    return path, fingerprint, True
