"""Thin generation client supporting Ollama and OpenRouter."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from answer_generation.pali_shims import clean_generated_answer

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MAX_RETRIES = 3


def _should_retry_status(code: int) -> bool:
    if code == 0:
        return True
    if code == 429:
        return True
    return 500 <= code < 600


def _json_request(url: str, payload: Any, timeout_s: float, headers: dict[str, str] | None = None) -> tuple[int, dict[str, Any]]:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
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


def _extract_chat_content(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content.strip()
    if isinstance(message_content, list):
        parts: list[str] = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return ""


def _generate_via_ollama(
    *,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout_s: float,
    max_tokens: int,
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


def _generate_via_openrouter(
    *,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    timeout_s: float,
    max_tokens: int,
    api_key_env: str,
    app_url: str,
    app_title: str,
    provider_routing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        return {
            "success": False,
            "status_code": 0,
            "raw_answer": "Unknown",
            "clean_answer": "Unknown",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_ms": 0.0,
            "error": {"error": f"Missing environment variable: {api_key_env}"},
        }

    # Pin to the OpenAI backend via OpenRouter with no fallbacks if using an OpenAI model.
    # Without this, OpenRouter load-balances across multiple backend nodes which
    # produces different outputs even at temperature=0 (FP precision differences
    # across hardware). allow_fallbacks=False ensures the request fails fast rather
    # than silently rerouting to a different provider/node.
    _provider_routing: dict[str, Any] = {}
    if model.startswith("openai/"):
        _provider_routing = {
            "order": ["openai"],
            "allow_fallbacks": False,
        }
        
    if provider_routing:
        _provider_routing.update(provider_routing)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": 42,
    }
    if _provider_routing:
        payload["provider"] = _provider_routing
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": app_url,
        "X-Title": app_title,
    }

    started = time.perf_counter()
    code = 0
    body: dict[str, Any] = {}
    for attempt in range(OPENROUTER_MAX_RETRIES):
        code, body = _json_request(base_url.rstrip("/") + "/chat/completions", payload, timeout_s, headers=headers)
        if code == 200 or not _should_retry_status(code):
            break
        if attempt < OPENROUTER_MAX_RETRIES - 1:
            time.sleep(0.75 * (attempt + 1))
    latency_ms = (time.perf_counter() - started) * 1000.0

    raw_text = ""
    if isinstance(body, dict):
        choices = body.get("choices") or []
        if choices:
            message = choices[0].get("message", {})
            raw_text = _extract_chat_content(message.get("content"))

    cleaned = clean_generated_answer(raw_text) if raw_text else "Unknown"
    success = code == 200 and bool(raw_text)
    usage = body.get("usage", {}) if isinstance(body, dict) else {}

    # Warn if response was truncated — likely a reasoning model that needs more tokens
    if isinstance(body, dict):
        choices = body.get("choices") or []
        if choices:
            finish = choices[0].get("finish_reason", "")
            if finish == "length":
                comp_tokens = int(usage.get("completion_tokens") or 0)
                import sys
                print(
                    f"WARNING: finish_reason=length for {model} "
                    f"(completion={comp_tokens}, max_tokens={max_tokens}). "
                    f"Reasoning models may need --max-tokens 4096+.",
                    file=sys.stderr,
                )

    return {
        "success": success,
        "status_code": code,
        "raw_answer": raw_text or "Unknown",
        "clean_answer": cleaned or "Unknown",
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "latency_ms": round(latency_ms, 3),
        "error": None if success else body,
    }


def generate_answer(
    *,
    provider: str,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    timeout_s: float = 60.0,
    max_tokens: int = 64,
    api_key_env: str = "OPENROUTER_API_KEY",
    app_url: str = "https://anonymous.4open.science/r/from-reliable-to-random-BB62",
    app_title: str = "anonymous-rag-compression-artifact",
    provider_routing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if provider == "ollama":
        return _generate_via_ollama(
            base_url=base_url,
            model=model,
            prompt=prompt,
            temperature=temperature,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
        )
    if provider == "openrouter":
        return _generate_via_openrouter(
            base_url=base_url,
            model=model,
            prompt=prompt,
            temperature=temperature,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            api_key_env=api_key_env,
            app_url=app_url,
            app_title=app_title,
            provider_routing=provider_routing,
        )
    raise ValueError(f"Unsupported provider: {provider}")
