"""Thin Ollama client for answer-generation runs."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from reader.pali_shims import clean_generated_answer


def _json_request(url: str, payload: Any, timeout_s: float) -> tuple[int, dict[str, Any]]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return resp.getcode(), json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            return exc.code, json.loads(body) if body else {}
        except json.JSONDecodeError:
            return exc.code, {"raw": body}
    except Exception as exc:  # pragma: no cover - network/runtime edge case
        return 0, {"error": repr(exc)}


def generate_answer(
    *,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    timeout_s: float = 60.0,
    max_tokens: int = 64,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    started = time.perf_counter()
    code, body = _json_request(base_url.rstrip("/") + "/api/generate", payload, timeout_s)
    latency_ms = (time.perf_counter() - started) * 1000.0

    raw_text = str(body.get("response", "")).strip() if isinstance(body, dict) else ""
    cleaned = clean_generated_answer(raw_text) if raw_text else "Unknown"
    success = code == 200 and bool(raw_text)

    return {
        "success": success,
        "status_code": code,
        "raw_answer": raw_text or "Unknown",
        "clean_answer": cleaned or "Unknown",
        "prompt_tokens": int(body.get("prompt_eval_count") or 0) if isinstance(body, dict) else 0,
        "completion_tokens": int(body.get("eval_count") or 0) if isinstance(body, dict) else 0,
        "latency_ms": round(latency_ms, 3),
        "error": None if success else body,
    }
