"""Fingerprinting and model cache management."""

import hashlib
import json
import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import METHOD_MODEL_CACHE_PROTOCOL_VERSION, METHOD_CACHE_MODEL_KEYS

def _stable_json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def _sha256_json(value: Any) -> str:
    return _sha256_text(_stable_json_dumps(value))

def _method_model_source_fingerprint() -> str:
    from . import features, heuristics, training, classification, selection, ranking
    
    # We collect source from all relevant functions to ensure cache invalidation
    relevant_parts = {
        "protocol_version": METHOD_MODEL_CACHE_PROTOCOL_VERSION,
        "cache_scope": "crisp_only_v3",
        "features": {
            "learning": inspect.getsource(features._learning_features),
            "usefulness": inspect.getsource(features._memory_usefulness_features),
        },
        "heuristics": {
            "postprocess": inspect.getsource(heuristics._postprocess_predicted_memory_usefulness_labels),
        },
        "training": {
            "dot": inspect.getsource(training._dot),
            "linear": inspect.getsource(training._train_linear_pruner),
            "pairwise": inspect.getsource(training._train_pairwise_ranker),
        },
        "classification": {
            "family": inspect.getsource(classification._selector_family_name),
        },
        "ranking": {
            "bundle": inspect.getsource(ranking._selector_feature_bundle),
        },
        "selection": {
            "policy": inspect.getsource(selection._selector_policy_select),
        }
    }
    return _sha256_json(relevant_parts)

def _method_model_cache_metadata(*, training_rows: list[dict[str, Any]], calibration_rows: list[dict[str, Any]] | None = None, controller_config: dict[str, Any]) -> dict[str, Any]:
    norm_train = sorted(training_rows, key=lambda r: str(r.get("example_id", "")))
    norm_calib = sorted(calibration_rows if calibration_rows is not None else training_rows, key=lambda r: str(r.get("example_id", "")))
    return {
        "protocol_version": METHOD_MODEL_CACHE_PROTOCOL_VERSION, "source_fingerprint": _method_model_source_fingerprint(),
        "controller_config_fingerprint": _sha256_json(controller_config), "training_rows_fingerprint": _sha256_json(norm_train),
        "training_row_count": len(norm_train), "calibration_rows_fingerprint": _sha256_json(norm_calib),
        "calibration_row_count": len(norm_calib), "cache_scope": "crisp_only",
    }

def _expected_method_model_cache_keys(shared_metadata: dict[str, Any], *, cached_model_keys: tuple[str, ...] = METHOD_CACHE_MODEL_KEYS) -> dict[str, dict[str, Any]]:
    return {key: {**shared_metadata, "method_name": key} for key in cached_model_keys}

def _load_active_method_model_cache(cache_path: Path, *, expected_metadata: dict[str, Any], cached_model_keys: tuple[str, ...] = METHOD_CACHE_MODEL_KEYS) -> dict[str, Any] | None:
    if not cache_path.exists(): return None
    try: payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return None
    if payload.get("metadata") != expected_metadata: return None
    models = payload.get("models")
    if not isinstance(models, dict): return None
    expected = _expected_method_model_cache_keys(expected_metadata, cached_model_keys=cached_model_keys)
    loaded = {}
    for key in cached_model_keys:
        entry = models.get(key)
        if not isinstance(entry, dict) or entry.get("cache_key") != expected[key]: return None
        loaded[key] = entry.get("model")
    return loaded

def _write_active_method_model_cache(cache_path: Path, *, metadata: dict[str, Any], models: dict[str, Any], source_slice_path: str | None = None, training_slice_paths: list[str] | None = None, cached_model_keys: tuple[str, ...] = METHOD_CACHE_MODEL_KEYS) -> None:
    expected = _expected_method_model_cache_keys(metadata, cached_model_keys=cached_model_keys)
    payload = {
        "metadata": metadata, "created_at": datetime.now(timezone.utc).isoformat(),
        "source_slice_path": source_slice_path, "training_slice_paths": list(training_slice_paths or []),
        "models": {key: {"method_name": key, "cache_key": expected[key], "model": models.get(key)} for key in cached_model_keys},
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
