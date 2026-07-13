"""Linear training and threshold optimization utilities."""

import numpy as np
from typing import Any
from .common_import import controller_result, cached_build_profile

def _dot(weights: dict[str, float] | np.ndarray, features: dict[str, float] | np.ndarray) -> float:
    if isinstance(weights, np.ndarray) and isinstance(features, np.ndarray):
        return float(np.dot(weights, features))
    if isinstance(weights, dict) and isinstance(features, dict):
        return sum(weights.get(name, 0.0) * value for name, value in features.items())
    # Fallback/Hybrid (rare)
    w_arr = np.array(list(weights.values())) if isinstance(weights, dict) else weights
    f_arr = np.array(list(features.values())) if isinstance(features, dict) else features
    return float(np.dot(w_arr, f_arr))

def _best_threshold(scored_examples: list[tuple[float, int]]) -> float:
    if not scored_examples: return 0.0
    thresholds = sorted({score for score, _ in scored_examples})
    thresholds = [thresholds[0] - 1e-6, *thresholds, thresholds[-1] + 1e-6]
    best_threshold, best_f2 = 0.0, -1.0
    for threshold in thresholds:
        tp, fp, fn = 0, 0, 0
        for score, label in scored_examples:
            predicted = score >= threshold
            if predicted and label == 1: tp += 1
            elif predicted and label == 0: fp += 1
            elif not predicted and label == 1: fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        # Use F2 score to weigh recall 2x as much as precision
        f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) else 0.0
        if f2 > best_f2:
            best_f2, best_threshold = f2, threshold
    return best_threshold

def _train_linear_pruner(*, feature_names: list[str], examples: list[tuple[dict[str, float], int]], calibration_examples: list[tuple[dict[str, float], int]] | None = None) -> dict[str, Any] | None:
    if not examples: return None
    
    # Convert to NumPy for high-speed training
    X = np.zeros((len(examples), len(feature_names)), dtype=np.float32)
    y = np.zeros(len(examples), dtype=np.int32)
    for i, (f, l) in enumerate(examples):
        X[i] = [f.get(name, 0.0) for name in feature_names]
        y[i] = l

    label_values = np.unique(y)
    if len(label_values) == 1:
        constant_label = int(label_values[0])
        return {
            "weights": {name: 0.0 for name in feature_names},
            "threshold": -1e-6 if constant_label == 1 else 1e-6,
            "training_examples": len(examples),
            "constant_label": constant_label,
        }

    weights = np.zeros(len(feature_names), dtype=np.float32)
    lr = 0.25
    for _ in range(12):
        # Vectorized prediction
        scores = np.dot(X, weights)
        preds = (scores >= 0).astype(np.int32)
        errors = y - preds
        if not np.any(errors != 0): break
        # Vectorized update
        weights += lr * np.dot(errors.astype(np.float32), X)
    
    # Calibration
    threshold_examples = calibration_examples or examples
    cal_scores = []
    w_dict = dict(zip(feature_names, weights.tolist()))
    for f, l in threshold_examples:
        cal_scores.append((_dot(w_dict, f), l))
        
    return {
        "weights": w_dict, 
        "threshold": _best_threshold(cal_scores),
        "training_examples": len(examples), 
        "calibration_examples": len(threshold_examples),
    }

def _fit_linear_keep_model(*, feature_names: list[str], training_rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]], collect_examples: Any, threshold_calibration_split: str | None) -> dict[str, Any] | None:
    examples = collect_examples(training_rows)
    calib_examples = collect_examples(calibration_rows)
    model = _train_linear_pruner(feature_names=feature_names, examples=examples, calibration_examples=calib_examples)
    if not model: return None
    model.update({"training_rows": len(training_rows), "calibration_rows": len(calibration_rows), "threshold_calibration_split": threshold_calibration_split or "train"})
    return model

def _training_row_context_cache(*, rows: list[dict[str, Any]], controller_config: dict[str, Any], use_concept_specs: bool) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    res, prof = {}, {}
    for r in sorted(rows, key=lambda i: str(i.get("example_id", ""))):
        eid = str(r.get("example_id", ""))
        if eid in res: continue
        res[eid] = controller_result(r, {"controller_config": controller_config})
        prof[eid] = cached_build_profile(str(r.get("query", "")), use_concept_specs=use_concept_specs)
    return res, prof

def _selector_pairwise_training_examples(*, candidate_payloads: list[tuple[dict[str, float], int, float]]) -> list[tuple[dict[str, float], int]]:
    positives = [item for item in candidate_payloads if item[1] == 1]
    negatives = [item for item in candidate_payloads if item[1] == 0]
    examples: list[tuple[dict[str, float], int]] = []
    if not positives or not negatives: return examples
    from .config import LABEL_SELECTOR_FEATURES
    for pos_feat, _, pos_util in positives:
        reps = max(1, min(3, int(round(max(pos_util, 0.5) * 2))))
        for neg_feat, _, _ in negatives:
            diff = {name: pos_feat.get(name, 0.0) - neg_feat.get(name, 0.0) for name in LABEL_SELECTOR_FEATURES}
            rev_diff = {name: -value for name, value in diff.items()}
            for _ in range(reps):
                examples.extend([(diff, 1), (rev_diff, 0)])
    return examples

def _train_pairwise_ranker(*, feature_names: list[str], examples: list[tuple[dict[str, float], int]]) -> dict[str, Any] | None:
    if not examples: return None
    
    # Convert to NumPy
    X = np.zeros((len(examples), len(feature_names)), dtype=np.float32)
    y = np.array([e[1] for e in examples], dtype=np.float32)
    for i, (f, l) in enumerate(examples):
        X[i] = [f.get(name, 0.0) for name in feature_names]

    weights = np.zeros(len(feature_names), dtype=np.float32)
    lr = 0.2
    l1_lambda = 0.005 
    
    for _ in range(12):
        # Vectorized prediction
        scores = np.dot(X, weights)
        preds = (scores >= 0).astype(np.float32)
        errors = y - preds
        if not np.any(errors != 0): break
        
        # Batch update
        weights += lr * np.dot(errors, X)
        
        # Vectorized L1 proximal step (sparsity)
        weights = np.sign(weights) * np.maximum(0, np.abs(weights) - l1_lambda)
        
    return {
        "weights": dict(zip(feature_names, weights.tolist())), 
        "training_examples": len(examples)
    }

def _collect_selector_training_examples(*, rows: list[dict[str, Any]], controller_config: dict[str, Any], use_concept_specs: bool, include_pairwise: bool, progress_callback: Any = None) -> tuple[list[tuple[dict[str, float], int]], list[tuple[dict[str, float], int]]]:
    res_cache, prof_cache = _training_row_context_cache(rows=rows, controller_config=controller_config, use_concept_specs=use_concept_specs)
    cal_ex, pair_ex = [], []
    from .ranking import _selector_feature_bundle, _selector_keep_label
    from .classification import _selector_family_name
    from .features import _query_token_weights, _lexical_content_tokens
    from .common_import import candidate_views
    total_rows = len(rows)
    for idx, row in enumerate(sorted(rows, key=lambda i: str(i.get("example_id", ""))), start=1):
        if progress_callback and idx % 250 == 0:
            progress_callback(f"collect_selector_examples:progress row={idx}/{total_rows}")
        m_labs = row.get("memory_usefulness_labels")
        if not m_labs or "candidate_memories" not in row: continue
        eid = str(row.get("example_id", ""))
        res, prof = res_cache[eid], prof_cache[eid]
        
        # Precompute row-level token weights
        all_views = candidate_views(row, use_concept_specs=use_concept_specs)
        token_weights, total_weight = _query_token_weights(prof.content_tokens, [v.content_tokens for v in all_views])
        views_by_id = {v.memory_id: v for v in all_views}
        context_token_sets = [_lexical_content_tokens(str(c)) for c in row.get("active_context", [])]

        q_fam = _selector_family_name(row=row, decision_summary=res["decision_summary"], memory_label_sets={str(mid): set(str(l) for l in labs.get("labels", [])) for mid, labs in m_labs.items()})
        payloads = []
        for cand in res["candidates"]:
            mid = str(cand["memory_id"])
            view = views_by_id[mid]
            feat, util, lab = _selector_feature_bundle(row=row, memory_id=mid, candidate=cand, decision_summary=res["decision_summary"], profile=view.profile, query_profile=prof, query_family=q_fam, memory_labels_by_id=m_labs, token_weights=token_weights, total_weight=total_weight, context_token_sets=context_token_sets)
            keep = _selector_keep_label(row=row, memory_labels=lab, query_family=q_fam, decision_summary=res["decision_summary"])
            cal_ex.append((feat, keep))
            payloads.append((feat, keep, util))
        if include_pairwise: pair_ex.extend(_selector_pairwise_training_examples(candidate_payloads=payloads))
    return cal_ex, pair_ex

def train_selector_model(training_rows: list[dict[str, Any]], controller_config: dict[str, Any], calibration_rows: list[dict[str, Any]] | None = None, threshold_calibration_split: str | None = None, progress_callback: Any = None) -> dict[str, Any] | None:
    use_concept_specs = bool(controller_config.get("profiling", {}).get("use_concept_specs", False))
    calib_rows = calibration_rows if calibration_rows is not None else training_rows
    if progress_callback: progress_callback(f"train_selector:collect_start rows={len(training_rows)}")
    cal_ex, pair_ex = _collect_selector_training_examples(rows=training_rows, controller_config=controller_config, use_concept_specs=use_concept_specs, include_pairwise=True, progress_callback=progress_callback)
    if progress_callback: progress_callback(f"train_selector:train_start pairs={len(pair_ex)}")
    if calib_rows != training_rows: 
        if progress_callback: progress_callback(f"train_selector:calib_collect_start rows={len(calib_rows)}")
        cal_ex, _ = _collect_selector_training_examples(rows=calib_rows, controller_config=controller_config, use_concept_specs=use_concept_specs, include_pairwise=False, progress_callback=progress_callback)
    from .config import LABEL_SELECTOR_FEATURES
    model = _train_pairwise_ranker(feature_names=LABEL_SELECTOR_FEATURES, examples=pair_ex)
    if model is None or len(cal_ex) < 12 or len(pair_ex) < 12: return None
    return {
        "weights": model["weights"], "threshold": _best_threshold([(_dot(model["weights"], f), l) for f, l in cal_ex]),
        "training_rows": len(training_rows), "calibration_rows": len(calib_rows), "training_pairs": len(pair_ex), "training_examples": len(cal_ex), "threshold_calibration_split": threshold_calibration_split or "train",
    }
