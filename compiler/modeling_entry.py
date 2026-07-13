"""Model training, caching, and shared feature engineering (Core Gateway)."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Callable

from controller.models import TextProfile
from controller.reporting import load_jsonl

from shared.nlp import (
    DEFAULT_V2_BUDGET_CONFIG,
    cached_build_profile,
    candidate_views,
    controller_result,
    repo_root,
)

from .modeling.config import (
    DEFAULT_TRAINING_SLICE_PATHS,
    METHOD_MODEL_CACHE_PATH,
    METHOD_CACHE_MODEL_KEYS,
    METHOD_MODEL_CACHE_PROTOCOL_VERSION,
    MEMORY_LABEL_TARGETS,
    MEMORY_EFFECT_TARGETS,
    USEFULNESS_PREDICTOR_FEATURES,
    LABEL_SELECTOR_FEATURES,
    ACTIVE_METHOD_MODEL_CACHE_PATH,
    CACHED_METHOD_MODEL_KEYS
)
from .modeling.features import (
    _memory_usefulness_features,
    _candidate_memory_text,
    _usefulness_query_is_binary_verification,
    _answer_shape_signals,
    _memory_label_features,
    _memory_label_score,
    _query_token_weights,
    _lexical_content_tokens
)
from .modeling.training import (
    _dot,
    _best_threshold,
    _train_linear_pruner,
    _fit_linear_keep_model,
    _training_row_context_cache
)
from .modeling.heuristics import (
    _support_rescue_score,
    _postprocess_predicted_memory_usefulness_labels,
)
from .modeling.cache import (
    _method_model_cache_metadata,
    _load_active_method_model_cache,
    _write_active_method_model_cache
)


def resolve_slice_paths(eval_slice_path: Path | None = None, override_paths: list[str] | None = None) -> list[Path]:
    requested = override_paths or DEFAULT_TRAINING_SLICE_PATHS
    eval_resolved = eval_slice_path.resolve() if eval_slice_path else None
    paths: list[Path] = []
    for raw_path in requested:
        candidate = Path(raw_path)
        if not candidate.is_absolute(): candidate = repo_root() / raw_path
        resolved = candidate.resolve()
        if eval_resolved is not None and resolved == eval_resolved: continue
        if resolved.exists(): paths.append(resolved)
    return paths

def load_training_rows(eval_slice_path: Path | None = None, override_paths: list[str] | None = None) -> list[dict[str, Any]]:
    rows_by_example: dict[str, dict[str, Any]] = {}
    for path in resolve_slice_paths(eval_slice_path, override_paths):
        for row in load_jsonl(path): rows_by_example[str(row["example_id"])] = row
    if eval_slice_path is not None and eval_slice_path.exists():
        for row in load_jsonl(eval_slice_path):
            if str(row.get("split", "")).strip().lower(): rows_by_example[str(row["example_id"])] = row
    return list(rows_by_example.values())

def _model_rows_by_split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if not rows: return [], [], "train"
    splits = [str(row.get("split", "")).strip().lower() for row in rows]
    if not any(splits): return rows, rows, "train"
    train = [row for row, s in zip(rows, splits) if s == "train"]
    dev = [row for row, s in zip(rows, splits) if s == "dev"]
    return (train, dev, "dev") if dev else (train, train, "train")

def _usefulness_target_value(memory_labels: dict[str, Any], target_name: str) -> int:
    labels = set(str(label) for label in memory_labels.get("labels", []))
    if target_name in MEMORY_LABEL_TARGETS: return 1 if target_name in labels else 0
    if target_name == "leave_one_out_breaks_sufficiency": return 1 if memory_labels.get("leave_one_out_effect") == "breaks_sufficiency" else 0
    if target_name == "add_one_rescue_restores_sufficiency": return 1 if memory_labels.get("add_one_rescue_effect") == "restores_sufficiency" else 0
    if target_name == "interaction_dependent": return 1 if memory_labels.get("uncertainty") == "interaction_dependent" else 0
    raise KeyError(f"Unsupported usefulness target: {target_name}")

def train_memory_usefulness_model(training_rows: list[dict[str, Any]], controller_config: dict[str, Any], calibration_rows: list[dict[str, Any]] | None = None, threshold_calibration_split: str | None = None, progress_callback: Callable[[str], None] | None = None) -> dict[str, Any] | None:
    use_concept_specs = bool(controller_config.get("profiling", {}).get("use_concept_specs", False))
    calibration_rows = calibration_rows if calibration_rows is not None else training_rows
    
    # Pre-cache results for speed
    all_rows = {str(r.get("example_id", "")): r for r in [*training_rows, *calibration_rows]}.values()
    ctx_cache, q_prof_cache = {}, {}
    for r in sorted(all_rows, key=lambda i: str(i.get("example_id", ""))):
        eid = str(r.get("example_id", ""))
        ctx_cache[eid] = controller_result(r, {"controller_config": controller_config})
        q_prof_cache[eid] = cached_build_profile(str(r.get("query", "")), use_concept_specs=use_concept_specs)

    def collect_examples(rows: list[dict[str, Any]], target: str) -> list[tuple[dict[str, float], int]]:
        examples = []
        for r in rows:
            labels = r.get("memory_usefulness_labels")
            if not labels or "candidate_memories" not in r: continue
            eid = str(r["example_id"])
            q_prof = q_prof_cache[eid]
            cand_views = candidate_views(r, use_concept_specs=use_concept_specs)
            token_weights, total_weight = _query_token_weights(q_prof.content_tokens, [v.content_tokens for v in cand_views])
            views_by_id = {v.memory_id: v for v in cand_views}
            context_token_sets = [_lexical_content_tokens(str(c)) for c in r.get("active_context", [])]
            for c in ctx_cache[eid]["candidates"]:
                mid = str(c["memory_id"])
                l = labels.get(mid)
                if l is None: continue
                view = views_by_id[mid]
                feats = _memory_usefulness_features(row=r, candidate=c, decision_summary=ctx_cache[eid]["decision_summary"], profile=view.profile, query_profile=q_prof, token_weights=token_weights, total_weight=total_weight, context_token_sets=context_token_sets)
                examples.append((feats, _usefulness_target_value(l, target)))
        return examples

    models = {}
    for target in (*MEMORY_LABEL_TARGETS, *MEMORY_EFFECT_TARGETS):
        if progress_callback: progress_callback(f"train_usefulness:target_start target={target}")
        m = _train_linear_pruner(feature_names=USEFULNESS_PREDICTOR_FEATURES, examples=collect_examples(training_rows, target), calibration_examples=collect_examples(calibration_rows, target))
        if m: 
            m.update({"training_rows": len(training_rows), "calibration_rows": len(calibration_rows), "threshold_calibration_split": threshold_calibration_split or "train"})
            models[target] = m
        if progress_callback: progress_callback(f"train_usefulness:target_done target={target} trained={m is not None}")
    
    return {"feature_names": list(USEFULNESS_PREDICTOR_FEATURES), "models": models, "training_rows": len(training_rows), "calibration_rows": len(calibration_rows), "threshold_calibration_split": threshold_calibration_split or "train"} if models else None

def predict_memory_usefulness_labels(row: dict[str, Any], context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    # Import rescue logic here to avoid circular dependencies if any
    from .modeling.heuristics import _apply_support_rescue
    
    model = context.get("v2_usefulness_label_model")
    if not model: return {}
    res = controller_result(row, context)
    use_concept = bool(context["controller_config"].get("profiling", {}).get("use_concept_specs", False))
    q_prof = cached_build_profile(str(row["query"]), use_concept_specs=use_concept)
    
    predicted, payloads = {}, []
    cand_views = candidate_views(row, use_concept_specs=use_concept)
    token_weights, total_weight = _query_token_weights(q_prof.content_tokens, [v.content_tokens for v in cand_views])
    views_by_id = {v.memory_id: v for v in cand_views}
    context_token_sets = [_lexical_content_tokens(str(c)) for c in row.get("active_context", [])]

    for c in res["candidates"]:
        mid = str(c["memory_id"])
        view = views_by_id[mid]
        f = _memory_usefulness_features(row=row, candidate=c, decision_summary=res["decision_summary"], profile=view.profile, query_profile=q_prof, token_weights=token_weights, total_weight=total_weight, context_token_sets=context_token_sets)
        
        target_margins = {t: _dot(model["models"][t]["weights"], f) - float(model["models"][t]["threshold"]) for t in MEMORY_LABEL_TARGETS if t in model["models"]}
        labels = {t for t, m in target_margins.items() if m >= 0.0}
        
        # Conflict resolution between answer/distractor
        if "answer_bearing" in labels and "distractor" in labels:
            if target_margins["answer_bearing"] >= target_margins["distractor"]: labels.discard("distractor")
            else: labels -= {"answer_bearing", "conflict_resolving", "newer_update"}
        if "answer_bearing" in labels: labels.discard("useless_under_budget")
        
        mid = str(c["memory_id"])
        predicted[mid] = {
            "labels": sorted(labels),
            "leave_one_out_effect": "breaks_sufficiency" if _dot(model["models"]["leave_one_out_breaks_sufficiency"]["weights"], f) >= float(model["models"]["leave_one_out_breaks_sufficiency"]["threshold"]) else "no_change",
            "add_one_rescue_effect": "restores_sufficiency" if _dot(model["models"]["add_one_rescue_restores_sufficiency"]["weights"], f) >= float(model["models"]["add_one_rescue_restores_sufficiency"]["threshold"]) else "no_change",
        }
        if "interaction_dependent" in model["models"] and _dot(model["models"]["interaction_dependent"]["weights"], f) >= float(model["models"]["interaction_dependent"]["threshold"]):
            predicted[mid]["uncertainty"] = "interaction_dependent"
            
        payloads.append({"memory_id": mid, "rank": int(c.get("rank", 1)), "support_score": _support_rescue_score(candidate=c, features=f), "features": f, "question_like": 1.0 if "?" in _candidate_memory_text(c) else 0.0, "query": str(row.get("query", "")), "text": _candidate_memory_text(c)})
    
    norm = _postprocess_predicted_memory_usefulness_labels(row=row, decision_summary=res["decision_summary"], labels_by_memory=predicted)
    return _apply_support_rescue(row=row, decision_summary=res["decision_summary"], labels_by_memory=norm, candidate_payloads=payloads)

def resolve_memory_usefulness_labels(row: dict[str, Any], context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if context.get("use_oracle_memory_labels"): return {str(mid): dict(lbls) for mid, lbls in row.get("memory_usefulness_labels", {}).items()}
    cache = context.setdefault("_resolve_memory_usefulness_labels", {})
    key = (str(row.get("example_id", "")), False)
    if key not in cache: cache[key] = predict_memory_usefulness_labels(row, context)
    return cache[key]

def build_method_context(*, controller_results: list[dict[str, Any]], controller_config: dict[str, Any], slice_path: Path | None = None, training_slice_paths: list[str] | None = None, training_rows: list[dict[str, Any]] | None = None, model_cache_path: Path | None = METHOD_MODEL_CACHE_PATH, progress_callback: Callable[[str], None] | None = None) -> dict[str, Any]:
    from .modeling.training import train_selector_model
    train_rows = training_rows if training_rows is not None else load_training_rows(slice_path, training_slice_paths)
    train_rows, calib_rows, calib_split = _model_rows_by_split(train_rows)
    
    cached = None
    if train_rows and model_cache_path:
        meta = _method_model_cache_metadata(training_rows=train_rows, calibration_rows=calib_rows, controller_config=controller_config)
        cached = _load_active_method_model_cache(model_cache_path, expected_metadata=meta)
    
    if cached:
        v2_use, v2_sel = cached.get("v2_usefulness_label_model"), cached.get("v2_label_selector_v1_model")
    else:
        v2_use = train_memory_usefulness_model(train_rows, controller_config, calibration_rows=calib_rows, threshold_calibration_split=calib_split, progress_callback=progress_callback) if train_rows else None
        v2_sel = train_selector_model(train_rows, controller_config, calibration_rows=calib_rows, threshold_calibration_split=calib_split, progress_callback=progress_callback) if train_rows else None
        if train_rows and model_cache_path:
            _write_active_method_model_cache(model_cache_path, metadata=meta, models={"v2_usefulness_label_model": v2_use, "v2_label_selector_v1_model": v2_sel}, source_slice_path=str(slice_path), training_slice_paths=training_slice_paths)

    return {
        "controller_budgets": {r["example_id"]: int(r["decision_summary"]["selected_token_count"]) for r in controller_results},
        "max_selected": int(controller_config["selection"]["max_selected"]), "v2_budget_config": dict(DEFAULT_V2_BUDGET_CONFIG), "v2_budget_overrides": {},
        "use_oracle_memory_labels": False, "controller_config": controller_config, "controller_results_by_example": {str(r["example_id"]): r for r in controller_results},
        "v2_usefulness_label_model": v2_use, "v2_label_pruner_v1_model": None, "v2_label_selector_v1_model": v2_sel,
    }
