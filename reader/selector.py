"""Selection helpers for the answer-generation phase."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
import platform
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import compiler as am
from controller.logic import run_example
from controller.reporting import load_jsonl

DEFAULT_SYSTEMS = list(am.DEFAULT_PAPER_METHODS)
AVAILABLE_SYSTEMS = [
    "naive_top_k",
    "bm25_top_k",
    "bm25_sieve",
]
SYSTEMS = list(DEFAULT_SYSTEMS)

SYSTEM_NOTES = {
    "naive_top_k": "keeps the retriever's top-k memories from the shared candidate pool without reranking.",
    "bm25_top_k": "reranks the shared candidate pool with BM25 and keeps its own top-k memories.",
    "bm25_sieve": "SIEVE evidence compilation: NLP extraction + LLM cascade compiler → budget-aware evidence packaging.",
}

_AUTO_CONTROLLER_PRECOMPUTE_MIN_ROWS = 8
_AUTO_CONTROLLER_PRECOMPUTE_MAX_WORKERS = 6
_AUTO_ROW_SELECTION_MIN_ROWS = 4
_AUTO_ROW_SELECTION_MAX_WORKERS = 8
_MACOS_SAFE_PROCESS_WORKERS = 8
_PROCESS_POOL_BATCH_ROWS_PER_WORKER = 5
_PROCESS_POOL_TASK_ROWS = 5


def _configure_worker_runtime() -> None:
    """Apply conservative thread limits inside subprocesses."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MINILM_FORCE_CPU", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    if platform.system() == "Darwin":
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    try:
        import torch  # optional: only needed when compiler uses neural models
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except ImportError:
        pass


def _resolve_process_worker_policy(
    row_count: int,
    *,
    env_var: str,
    default_cap: int,
    min_rows: int,
) -> tuple[int, str]:
    if row_count <= 1:
        return 1, "single-row"

    override = os.environ.get(env_var)
    if override:
        try:
            requested = int(override)
        except ValueError:
            requested = 1
        requested = max(1, requested)
        capped = min(row_count, requested)
        return capped, f"env {env_var}={requested}"

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 1 or row_count < min_rows:
        return 1, "serial fallback"

    cap = default_cap
    source = f"default cap={default_cap}"
    if platform.system() == "Darwin":
        cap = min(cap, _MACOS_SAFE_PROCESS_WORKERS)
        source = f"macOS safety cap={cap}"
    worker_count = max(1, min(row_count, cpu_count, cap))
    return worker_count, source


def _process_pool_batch_size(worker_count: int) -> int:
    return max(1, worker_count * _PROCESS_POOL_BATCH_ROWS_PER_WORKER)


def _iter_batches(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def _is_answer_bearing_slot_payload(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    memory_id = str(payload.get("memory_id") or "").strip().lower()
    unit_id = str(payload.get("unit_id") or "").strip().lower()
    speaker = str(payload.get("speaker") or "").strip().lower()
    text = str(
        payload.get("display_text")
        or payload.get("text")
        or payload.get("source_span_text")
        or payload.get("source_text")
        or ""
    ).strip()
    if not text:
        return False
    if memory_id == "question_date" or unit_id == "question_date":
        return False
    if speaker == "system" and memory_id in {"", "question_date"}:
        return False
    return True
def load_slice(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def load_controller_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_systems(systems: list[str] | None = None) -> list[str]:
    requested = systems or DEFAULT_SYSTEMS
    normalized: list[str] = []
    seen: set[str] = set()
    for system in requested:
        canonical = am.normalize_method_name(system)
        if canonical not in AVAILABLE_SYSTEMS:
            raise ValueError(f"Unsupported system: {system}")
        if canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    return normalized


def _run_example_worker(payload: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    row, controller_config = payload
    return run_example(row, controller_config)


def _run_example_batch_worker(payloads: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[dict[str, Any]]:
    return [_run_example_worker(payload) for payload in payloads]


def _select_row_selections_worker(
    payload: tuple[int, dict[str, Any], list[str], dict[str, Any]]
) -> tuple[int, str, dict[str, dict[str, Any]], float]:
    row_index, row, selected_systems, shared_context = payload
    example_id, row_selections, elapsed_s = _select_row_selections(
        row=row,
        selected_systems=selected_systems,
        shared_context=shared_context,
    )
    return row_index, example_id, row_selections, elapsed_s


def _select_row_selections_batch_worker(
    payloads: list[tuple[int, dict[str, Any], list[str], dict[str, Any]]]
) -> list[tuple[int, str, dict[str, dict[str, Any]], float]]:
    return [_select_row_selections_worker(payload) for payload in payloads]


def _controller_results_for_rows(
    rows: list[dict[str, Any]],
    controller_config: dict[str, Any],
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    worker_count = _controller_precompute_worker_policy(len(rows))[0]
    if worker_count == 1:
        _configure_worker_runtime()
        return [run_example(row, controller_config) for row in rows]
    batch_size = _process_pool_batch_size(worker_count)
    controller_results: list[dict[str, Any]] = []
    total_rows = len(rows)
    batch_started = perf_counter()
    with ProcessPoolExecutor(
        initializer=_configure_worker_runtime,
        max_workers=worker_count,
    ) as executor:
        for batch_index, (batch_start, batch_rows) in enumerate(_iter_batches(rows, batch_size), start=1):
            if progress_callback is not None:
                progress_callback(
                    "build_system_selections:controller_precompute:batch_start "
                    f"batch={batch_index} rows={len(batch_rows)} workers={worker_count}"
                )
            task_batches = [
                [(row, controller_config) for row in task_rows]
                for _, task_rows in _iter_batches(batch_rows, _PROCESS_POOL_TASK_ROWS)
            ]
            for batch_results in executor.map(_run_example_batch_worker, task_batches):
                controller_results.extend(batch_results)
            if progress_callback is not None:
                progress_callback(
                    "build_system_selections:controller_precompute:batch_done "
                    f"batch={batch_index} completed={min(batch_start + len(batch_rows), total_rows)}/{total_rows} "
                    f"elapsed_s={perf_counter() - batch_started:.3f}"
                )
    return controller_results


def _row_selection_worker_count(row_count: int) -> int:
    return _resolve_process_worker_policy(
        row_count,
        env_var="MEMORY_SELECTOR_ROW_WORKERS",
        default_cap=_AUTO_ROW_SELECTION_MAX_WORKERS,
        min_rows=_AUTO_ROW_SELECTION_MIN_ROWS,
    )[0]


def _controller_precompute_worker_policy(row_count: int) -> tuple[int, str]:
    return _resolve_process_worker_policy(
        row_count,
        env_var="MEMORY_SELECTOR_PRECOMPUTE_WORKERS",
        default_cap=_AUTO_CONTROLLER_PRECOMPUTE_MAX_WORKERS,
        min_rows=_AUTO_CONTROLLER_PRECOMPUTE_MIN_ROWS,
    )


def _row_selection_worker_policy(row_count: int) -> tuple[int, str]:
    return _resolve_process_worker_policy(
        row_count,
        env_var="MEMORY_SELECTOR_ROW_WORKERS",
        default_cap=_AUTO_ROW_SELECTION_MAX_WORKERS,
        min_rows=_AUTO_ROW_SELECTION_MIN_ROWS,
    )


def _select_row_selections(
    *,
    row: dict[str, Any],
    selected_systems: list[str],
    shared_context: dict[str, Any],
) -> tuple[str, dict[str, dict[str, Any]], float]:
    started = perf_counter()
    row_selections: dict[str, dict[str, Any]] = {}
    for system in selected_systems:
        result = am.run_method(
            system,
            row,
            shared_context,
        )
        row_selections[system] = {
            "selected_memory_ids": result["decision_summary"]["predicted_selected_memory_ids"],
            "selection_meta": result.get("selection_meta", {}),
            "decision_summary": result["decision_summary"],
        }
    return str(row["example_id"]), row_selections, perf_counter() - started


def _row_shared_context(
    *,
    row: dict[str, Any],
    controller_result: dict[str, Any],
    shared_context: dict[str, Any],
) -> dict[str, Any]:
    example_id = str(row["example_id"])
    row_overrides = dict(shared_context.get("v2_budget_overrides", {}).get(example_id, {}))
    return {
        "controller_budgets": {
            example_id: int(shared_context["controller_budgets"].get(example_id, 0))
        },
        "max_selected": int(shared_context["max_selected"]),
        "v2_budget_config": dict(shared_context.get("v2_budget_config", {})),
        "v2_budget_overrides": {example_id: row_overrides} if row_overrides else {},
        "use_oracle_memory_labels": bool(shared_context.get("use_oracle_memory_labels", False)),
        "controller_config": shared_context["controller_config"],
        "controller_results_by_example": {
            example_id: controller_result,
        },
        "v2_usefulness_label_model": shared_context.get("v2_usefulness_label_model"),
        "v2_label_pruner_v1_model": shared_context.get("v2_label_pruner_v1_model"),
        "v2_label_selector_v1_model": shared_context.get("v2_label_selector_v1_model"),
    }


def build_system_selections(
    rows: list[dict[str, Any]],
    controller_config: dict[str, Any],
    *,
    systems: list[str] | None = None,
    slice_path: Path | None = None,
    training_slice_paths: list[str] | None = None,
    training_rows: list[dict[str, Any]] | None = None,
    external_prompt_cap: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    _configure_worker_runtime()
    selected_systems = normalize_systems(systems)
    controller_started = perf_counter()
    controller_worker_count, controller_worker_source = _controller_precompute_worker_policy(len(rows))
    if progress_callback is not None:
        progress_callback(
            "build_system_selections:controller_precompute:start "
            f"rows={len(rows)} workers={controller_worker_count} source={controller_worker_source}"
        )
    controller_results = _controller_results_for_rows(
        rows,
        controller_config,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(
            "build_system_selections:controller_precompute:done "
            f"elapsed_s={perf_counter() - controller_started:.3f}"
        )
    context_started = perf_counter()
    if progress_callback is not None:
        progress_callback("build_system_selections:method_context:start")
    shared_context = am.build_method_context(
        controller_results=controller_results,
        controller_config=controller_config,
        slice_path=slice_path,
        training_slice_paths=training_slice_paths,
        training_rows=training_rows,
        progress_callback=progress_callback,
    )
    if external_prompt_cap is not None:
        capped_budget = max(0, int(external_prompt_cap))
        existing_overrides = dict(shared_context.get("v2_budget_overrides", {}))
        shared_context["v2_budget_overrides"] = {
            str(row["example_id"]): {
                **dict(existing_overrides.get(str(row["example_id"]), {})),
                "token_budget": capped_budget,
            }
            for row in rows
        }
        shared_context["external_prompt_cap"] = capped_budget
    if progress_callback is not None:
        progress_callback(
            "build_system_selections:method_context:done "
            f"elapsed_s={perf_counter() - context_started:.3f}"
        )
    am.validate_method_context_for_methods(selected_systems, shared_context)

    by_example_id: dict[str, dict[str, dict[str, Any]]] = {}
    selection_workers, selection_worker_source = _row_selection_worker_policy(len(rows))
    if progress_callback is not None:
        progress_callback(
            "build_system_selections:row_selection:start "
            f"rows={len(rows)} workers={selection_workers} source={selection_worker_source}"
        )
    if selection_workers == 1:
        for row_index, row in enumerate(rows, start=1):
            if progress_callback is not None:
                progress_callback(
                    "build_system_selections:row_selection:row_start "
                    f"{row_index}/{len(rows)} example_id={row.get('example_id')}"
                )
            example_id, row_selections, elapsed_s = _select_row_selections(
                row=row,
                selected_systems=selected_systems,
                shared_context=shared_context,
            )
            by_example_id[example_id] = row_selections
            if progress_callback is not None and (row_index % 50 == 0 or row_index == len(rows)):
                progress_callback(
                    "build_system_selections:row_done "
                    f"{row_index}/{len(rows)} example_id={example_id} "
                    f"elapsed_s={elapsed_s:.3f}"
                )
        return by_example_id

    row_count = len(rows)
    completed = 0
    selection_started = perf_counter()
    row_payloads: list[tuple[int, dict[str, Any], list[str], dict[str, Any]]] = [
        (
            row_index,
            row,
            selected_systems,
            _row_shared_context(
                row=row,
                controller_result=controller_results[row_index - 1],
                shared_context=shared_context,
            ),
        )
        for row_index, row in enumerate(rows, start=1)
    ]
    batch_size = _process_pool_batch_size(selection_workers)
    with ProcessPoolExecutor(
        initializer=_configure_worker_runtime,
        max_workers=selection_workers,
    ) as executor:
        for batch_index, (batch_start, batch_payloads) in enumerate(_iter_batches(row_payloads, batch_size), start=1):
            if progress_callback is not None:
                progress_callback(
                    "build_system_selections:row_selection:batch_start "
                    f"batch={batch_index} rows={len(batch_payloads)} workers={selection_workers}"
                )
            task_batches = [
                task_rows
                for _, task_rows in _iter_batches(batch_payloads, _PROCESS_POOL_TASK_ROWS)
            ]
            batch_results: list[tuple[int, str, dict[str, dict[str, Any]], float]] = []
            for task_results in executor.map(_select_row_selections_batch_worker, task_batches):
                batch_results.extend(task_results)
            for row_index, example_id, row_selections, elapsed_s in batch_results:
                by_example_id[example_id] = row_selections
                completed += 1
                if progress_callback is not None and (completed % 50 == 0 or completed == row_count):
                    progress_callback(
                        "build_system_selections:row_done "
                        f"{completed}/{row_count} example_id={example_id} "
                        f"task_index={row_index} elapsed_s={elapsed_s:.3f}"
                    )
            if progress_callback is not None:
                progress_callback(
                    "build_system_selections:row_selection:batch_done "
                    f"batch={batch_index} completed={completed}/{row_count} "
                    f"elapsed_s={perf_counter() - selection_started:.3f}"
                )

    expected_example_ids = {str(row.get("example_id") or "").strip() for row in rows}
    missing_example_ids = sorted(example_id for example_id in expected_example_ids if example_id and example_id not in by_example_id)
    if missing_example_ids:
        preview = ", ".join(missing_example_ids[:10])
        raise RuntimeError(
            "Selection coverage incomplete after row selection. "
            f"missing_example_ids={preview}"
            + (" ..." if len(missing_example_ids) > 10 else "")
        )

    return by_example_id


def selected_texts_for_system(
    row: dict[str, Any],
    system_selection: dict[str, Any],
) -> list[str]:
    meta = system_selection.get("selection_meta", {})
    compiler_authoritative = bool(meta.get("compiler_authoritative")) or (
        str(meta.get("selection_mode") or "").strip().lower() == "evidence_compiler"
    )
    rendered_package = meta.get("rendered_evidence_package")
    has_bound_structured_evidence = False
    if isinstance(rendered_package, dict):
        slots = rendered_package.get("slots")
        if isinstance(slots, dict):
            has_bound_structured_evidence = any(
                _is_answer_bearing_slot_payload(payload) for payload in slots.values()
            )
        if not has_bound_structured_evidence:
            lines = rendered_package.get("evidence_lines")
            if isinstance(lines, list):
                has_bound_structured_evidence = any(isinstance(item, dict) for item in lines)
    compiled_texts = [
        str(text).strip()
        for text in meta.get("compiled_texts", [])
        if str(text).strip()
    ]
    if compiled_texts:
        return compiled_texts
    if isinstance(rendered_package, dict):
        lines = rendered_package.get("evidence_lines")
        if compiler_authoritative and isinstance(lines, list):
            evidence_texts = [
                str(item.get("text")).strip()
                for item in lines
                if isinstance(item, dict) and str(item.get("text")).strip()
            ]
            if evidence_texts:
                return evidence_texts
        compact_text = str(rendered_package.get("compact_text") or "").strip()
        if compact_text and has_bound_structured_evidence:
            return [compact_text]
    if compiler_authoritative:
        return []
    if meta and "compressed_memory_text" in meta:
        return [str(meta["compressed_memory_text"])]

    by_memory_id = {
        str(candidate["memory_id"]): str(candidate["content"])
        for candidate in row["candidate_memories"]
    }
    return [
        by_memory_id[memory_id]
        for memory_id in system_selection["selected_memory_ids"]
        if memory_id in by_memory_id
    ]
