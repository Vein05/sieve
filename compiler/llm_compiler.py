"""LLM-based evidence compiler for cascading compilation.

When the rule-based SIEVE compiler produces garbage (KV fragments, empty
multi-session output), this module calls a small LLM to extract typed
evidence from the top-K candidate memories.  Output matches the
rendered_evidence_package schema so the compact reader sees identical
structure regardless of compiler source.

Query-type-aware prompts:
  - knowledge_update: prioritises most recent state when memories conflict
  - assistant_recall: focuses on assistant-spoken content (suggestions, lists)
  - default: general-purpose fact extraction
"""

from __future__ import annotations

import json
import re
from typing import Any

from reader.client import generate_answer

# ── prompts ───────────────────────────────────────────────────────────

_DEFAULT_PROMPT = """\
You are an evidence compiler for a conversational memory QA system.

Given the user's question and retrieved memory chunks, extract the key evidence needed to answer.
{date_context}
Output a JSON object with these fields:
- "facts": list of concise factual statements extracted from memories (strings)
- "temporal": any dates, durations, or time references relevant to the question (string or null)
- "answer_hint": your best guess at the answer based on the evidence (short phrase)
- "confidence": "high", "medium", or "low"

Be concise. Extract only what is needed to answer the question.

Question: {question}

Retrieved memories:
{memories}

Output JSON only, no markdown fencing:"""

_KNOWLEDGE_UPDATE_PROMPT = """\
You are an evidence compiler for a conversational memory QA system.

Given the user's question and retrieved memory chunks, extract the key evidence needed to answer.
IMPORTANT: Memories may contain outdated AND updated information. When the same topic appears
multiple times, pay attention to dates — the MOST RECENT mention reflects the current state.
If a user says "I switched to X" or "I moved to Y", the answer is X or Y, not the previous state.

Output a JSON object with:
- "facts": list of concise factual statements, prioritizing the MOST RECENT state (strings)
- "temporal": dates showing when information was updated
- "answer_hint": your best guess based on the LATEST evidence (short phrase)
- "confidence": "high", "medium", or "low"
{date_context}
Question: {question}

Retrieved memories:
{memories}

Output JSON only, no markdown fencing:"""

_ASSISTANT_RECALL_PROMPT = """\
You are an evidence compiler for a conversational memory QA system.

Given the user's question and retrieved memory chunks, extract the key evidence needed to answer.
IMPORTANT: The user is asking about something an AI assistant previously said, suggested, or
listed. Focus on ASSISTANT-spoken content — extract specific items, names, lists, or details
that the assistant provided. Include exact names, numbers, and ordinal positions if mentioned.

Output a JSON object with:
- "facts": list of specific items/details the ASSISTANT said or suggested (strings)
- "temporal": any dates or time references
- "answer_hint": the specific detail the user is asking about (short phrase)
- "confidence": "high", "medium", or "low"
{date_context}
Question: {question}

Retrieved memories:
{memories}

Output JSON only, no markdown fencing:"""

_COUNTING_PROMPT = """\
You are an evidence compiler for a conversational memory QA system.

Given the user's question and retrieved memory chunks, extract ALL relevant evidence.
For counting questions, list EVERY distinct item mentioned — do not summarize or group.

Output a JSON object with:
- "facts": list of concise factual statements (one per distinct item/event/entity)
- "temporal": any dates, durations, or time references
- "answer_hint": your best guess at the answer
- "confidence": "high", "medium", or "low"

Be thorough. For counting questions, enumerate every distinct item individually.
{date_context}
Question: {question}

Retrieved memories:
{memories}

Output JSON only, no markdown fencing:"""

_ORDERING_PROMPT = """\
You are an evidence compiler for a conversational memory QA system.

Given the user's question and retrieved memory chunks, extract ALL relevant evidence.
The question asks about chronological or sequential ordering. For each item, extract the
associated date, timestamp, or ordinal position so the correct order can be determined.

Output a JSON object with:
- "facts": list of concise factual statements, each including the date/time or position
- "temporal": dates or time references that establish the ordering
- "answer_hint": the items listed in the requested order (short phrase)
- "confidence": "high", "medium", or "low"

Be precise about dates and ordering signals.
{date_context}
Question: {question}

Retrieved memories:
{memories}

Output JSON only, no markdown fencing:"""

# ── query type detection ──────────────────────────────────────────────

_ASSISTANT_RECALL_PATTERNS = [
    "you suggested", "you recommended", "you mentioned", "can you remind me",
    "our previous chat", "our previous conversation", "we discussed",
    "you said", "you told me", "you listed", "in our chat",
    "check back on", "looking back at", "follow up on",
]

_KNOWLEDGE_UPDATE_PATTERNS = [
    "most recent", "currently", "now", "after", "switch", "moved to",
    "changed to", "updated", "new", "latest", "these days", "still",
]

_COUNTING_PATTERNS = [
    "how many", "how much", "number of", "count of", "total",
]

_ORDERING_PATTERNS = [
    "order of", "from earliest to latest", "which happened first",
    "who graduated first", "starting from the earliest",
    "from first to last", "chronological order",
]


def _detect_query_type(query: str) -> str:
    """Detect query type from surface patterns for prompt selection."""
    q = query.lower()
    if any(p in q for p in _ASSISTANT_RECALL_PATTERNS):
        return "assistant_recall"
    if any(p in q for p in _ORDERING_PATTERNS):
        return "ordering"
    if any(p in q for p in _COUNTING_PATTERNS):
        return "counting"
    if any(p in q for p in _KNOWLEDGE_UPDATE_PATTERNS):
        return "knowledge_update"
    return "default"


def _select_prompt(query_type: str) -> str:
    return {
        "assistant_recall": _ASSISTANT_RECALL_PROMPT,
        "knowledge_update": _KNOWLEDGE_UPDATE_PROMPT,
        "counting": _COUNTING_PROMPT,
        "ordering": _ORDERING_PROMPT,
    }.get(query_type, _DEFAULT_PROMPT)


def _date_context(question_date: str | None) -> str:
    if not question_date:
        return ""
    return (
        f"The question is being asked on {question_date}. "
        "Resolve relative time references (e.g., 'last week', 'a month ago') against this date.\n"
    )


# ── helpers ───────────────────────────────────────────────────────────

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    return _THINK_TAG_RE.sub("", text).strip()


def _parse_json_response(raw: str) -> dict[str, Any]:
    cleaned = _strip_think_tags(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
    return {"raw": cleaned}


def _call_llm(
    prompt: str,
    *,
    model: str,
    provider: str,
    base_url: str,
    timeout_s: float,
    api_key_env: str,
    app_url: str,
    app_title: str,
    max_tokens: int = 300,
) -> dict[str, Any]:
    return generate_answer(
        provider=provider,
        base_url=base_url,
        model=model,
        prompt=prompt,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        temperature=0.0,
        api_key_env=api_key_env,
        app_url=app_url,
        app_title=app_title,
    )


# ── single-call compilation ──────────────────────────────────────────

def llm_compile(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
    model: str = "qwen/qwen3-8b",
    provider: str = "openrouter",
    base_url: str = "https://openrouter.ai/api/v1",
    timeout_s: float = 30.0,
    api_key_env: str = "OPENROUTER_API_KEY",
    app_url: str = "https://anonymous.4open.science/r/from-reliable-to-random-BB62",
    app_title: str = "anonymous-rag-compression-artifact",
    question_date: str | None = None,
) -> dict[str, Any]:
    """Call a small LLM to extract typed evidence from top-K candidates.

    Selects a query-type-aware prompt automatically based on surface
    patterns in the query (knowledge update, assistant recall, counting,
    or default factual extraction).
    """
    query_type = _detect_query_type(query)
    prompt_template = _select_prompt(query_type)

    top = candidates[:top_k]
    mem_block = ""
    for i, c in enumerate(top, 1):
        content = str(c.get("content", ""))[:500]
        mem_block += f"{i}. {content}\n\n"

    prompt = prompt_template.format(
        question=query, memories=mem_block, date_context=_date_context(question_date),
    )
    llm_kwargs = dict(
        model=model, provider=provider, base_url=base_url,
        timeout_s=timeout_s, api_key_env=api_key_env,
        app_url=app_url, app_title=app_title,
    )
    generation = _call_llm(prompt, **llm_kwargs)
    parsed = _parse_json_response(generation.get("raw_answer", ""))

    return {
        "facts": parsed.get("facts", []),
        "temporal": parsed.get("temporal"),
        "answer_hint": parsed.get("answer_hint"),
        "confidence": parsed.get("confidence", "low"),
        "compiler_type": "llm",
        "compiler_model": model,
        "compiler_input_tokens": generation.get("prompt_tokens", 0),
        "compiler_output_tokens": generation.get("completion_tokens", 0),
        "candidates_used": len(top),
        "query_type": query_type,
        "success": generation.get("success", False),
    }


# ── output formatting ────────────────────────────────────────────────

def format_llm_evidence(llm_evidence: dict[str, Any]) -> dict[str, Any]:
    """Convert LLM compiler output to rendered_evidence_package format."""
    slots: dict[str, Any] = {}
    if llm_evidence.get("answer_hint"):
        slots["answer_hint"] = {"display_text": llm_evidence["answer_hint"]}

    evidence_lines = [
        {"text": fact} for fact in (llm_evidence.get("facts") or [])
    ]
    if llm_evidence.get("temporal"):
        evidence_lines.append({"text": f"Temporal: {llm_evidence['temporal']}"})

    return {
        "slots": slots,
        "evidence_lines": evidence_lines,
        "compiler_type": "llm",
        "pool_augmented": False,
    }
