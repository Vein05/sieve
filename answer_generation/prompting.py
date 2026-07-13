"""Prompt builders for the answer-generation phase."""

from __future__ import annotations

import re
from typing import Any


_OBJECT_RENDER_RE = re.compile(
    r"^(?:user|assistant|system):\s+\S[^:]*?(?:location|time|name|state|type|value|date|event):\s+\S",
    re.IGNORECASE,
)


def _is_object_rendered_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if re.match(r"\d{4}/\d{2}/\d{2}", stripped):
        return False
    if re.search(r"\d{4}/\d{2}/\d{2}.*session\s+\S+", stripped, re.IGNORECASE):
        return False
    return bool(_OBJECT_RENDER_RE.match(stripped))


_COMPACT_READER_VARIANTS = {"crisp_compact_v1"}
_PREFERENCE_READER_VARIANTS = {"preference_reader_v1"}
_MAX_SLOT_COUNT = 12
_MAX_SLOT_TEXT_CHARS = 250
_MAX_EVIDENCE_LINE_COUNT = 12
_MAX_EVIDENCE_TEXT_CHARS = 512
_MAX_FALLBACK_LINE_COUNT = 8
_MAX_FALLBACK_TEXT_CHARS = 220


def _clip_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 15)].rstrip() + "... [truncated]"


def _slot_text_limit(slot_name: str) -> int:
    name = str(slot_name or "").strip().lower()
    if name in {"operand_summary", "computed_date_hint"}:
        return 520
    return _MAX_SLOT_TEXT_CHARS


def _render_enumerated_items(payload: dict[str, Any]) -> list[str]:
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    rendered = [str(item).strip() for item in items if str(item).strip()]
    if not rendered:
        return []
    max_items = 24
    lines = [f"- enumerated_items_count: {len(rendered)}"]
    for idx, item in enumerate(rendered[:max_items], start=1):
        lines.append(f"  - ({idx}) {_clip_text(item, 220)}")
    if len(rendered) > max_items:
        lines.append(f"  - ... (+{len(rendered) - max_items} more items)")
    return lines


def _numbered_block(
    lines: list[str],
    *,
    max_lines: int | None = None,
    text_limit: int | None = None,
) -> str:
    clean = [str(line).strip() for line in lines if str(line).strip()]
    if not clean:
        return "1. None"

    if max_lines is not None:
        clean = clean[:max_lines]

    rendered: list[str] = []
    for idx, line in enumerate(clean, start=1):
        text = _clip_text(line, text_limit) if text_limit is not None else line
        rendered.append(f"{idx}. {text}")
    return "\n".join(rendered)


def _compact_reader_block(rendered_evidence_package: dict[str, Any] | None) -> str:
    if not isinstance(rendered_evidence_package, dict):
        return "Validated slots:\n- None\n\nEvidence lines:\n1. None"

    slots = rendered_evidence_package.get("validated_slots") or rendered_evidence_package.get("slots")
    target_entities = [str(item).strip() for item in (rendered_evidence_package.get("target_entities") or []) if str(item).strip()]
    slot_lines: list[str] = []
    if isinstance(slots, dict):
        for slot_name, payload in list(slots.items())[:10]:
            if not isinstance(payload, dict):
                continue
            if str(slot_name).strip() == "enumerated_items":
                expanded = _render_enumerated_items(payload)
                if expanded:
                    slot_lines.extend(expanded)
                    continue
            text = _clip_text(payload.get("display_text") or payload.get("text"), _slot_text_limit(str(slot_name)))
            if text:
                slot_lines.append(f"- {slot_name}: {text}")
    if not slot_lines:
        slot_lines.append("- None")

    evidence_lines: list[str] = []
    rendered_lines = rendered_evidence_package.get("evidence_lines")
    if isinstance(rendered_lines, list):
        clean_lines = [
            item for item in rendered_lines
            if isinstance(item, dict) and not _is_object_rendered_text(str(item.get("text") or ""))
        ]
        for idx, item in enumerate(clean_lines[:_MAX_EVIDENCE_LINE_COUNT], start=1):
            evidence_lines.append(f"{idx}. {_clip_text(item.get('text'), _MAX_EVIDENCE_TEXT_CHARS)}")
    if not evidence_lines:
        evidence_lines.append("1. None")

    target_block = ""
    if target_entities:
        target_block = "Target entities:\n- " + "\n- ".join(target_entities[:6]) + "\n\n"

    additional_block = ""
    additional_context = rendered_evidence_package.get("additional_context")
    if isinstance(additional_context, list) and additional_context:
        ctx_lines: list[str] = []
        for session in additional_context[:6]:
            if not isinstance(session, dict):
                continue
            date = session.get("date") or "?"
            mid = str(session.get("memory_id") or "")[:16]
            ctx_lines.append(f"[{date}] {mid}:")
            for line in (session.get("lines") or [])[:5]:
                raw_text = str(line.get("text") or "").strip()
                if _is_object_rendered_text(raw_text):
                    continue
                text = _clip_text(raw_text, _MAX_EVIDENCE_TEXT_CHARS)
                if text:
                    ctx_lines.append(f"  - {text}")
        if ctx_lines:
            additional_block = "\n\nAdditional compiled context:\n" + "\n".join(ctx_lines)

    return (
        target_block
        + "Validated slots:\n" + "\n".join(slot_lines)
        + "\n\nEvidence lines:\n" + "\n".join(evidence_lines)
        + additional_block
    )


def build_answer_generation_prompt(
    *,
    question: str,
    question_date: str | None = None,
    active_context: list[str],
    retrieved_memories: list[str],
    variant: str = "baseline",
    harm_type: str | None = None,
    insufficient_answer_mode: str = "unknown",
    query_family: str | None = None,
    rendered_evidence_package: dict[str, Any] | None = None,
    question_type: str | None = None,
) -> str:
    lowered_question = question.strip().lower()
    compact_reader = variant in _COMPACT_READER_VARIANTS
    current_block = _numbered_block(
        active_context,
        max_lines=_MAX_FALLBACK_LINE_COUNT if compact_reader else None,
        text_limit=_MAX_FALLBACK_TEXT_CHARS if compact_reader else None,
    )
    memory_block = _numbered_block(
        retrieved_memories,
        max_lines=_MAX_FALLBACK_LINE_COUNT if compact_reader else None,
        text_limit=_MAX_FALLBACK_TEXT_CHARS if compact_reader else None,
    )
    if insufficient_answer_mode == "question_topic_absent_sentence":
        insufficient_rule = (
            "  - If the evidence is insufficient, answer with one brief sentence in this form: "
            "\"Based on the provided chat, there is no information related to <the main topic asked in the question>.\" "
            "Reuse the topic from the question instead of answering generically."
        )
    elif insufficient_answer_mode == "chat_absent_sentence":
        insufficient_rule = (
            "  - If the evidence is insufficient, answer with one brief sentence stating that this information is not available in the chat."
        )
    else:
        insufficient_rule = "  - If the evidence is insufficient, answer exactly: Unknown."
    rules = [
        "  - Prefer current dialogue context over retrieved memories when they conflict.",
        "  - Each dialogue line may start with a date stamp. Resolve words like yesterday, last week, last Friday, this morning, or tonight relative to that line's date.",
        "  - If a dated line describes an event without a relative offset, you may use that line date as the approximate answer.",
        "  - Use personal details only when they directly help answer the question.",
        "  - For who/what/which/where/when questions, copy the most specific answer phrase from the best line and preserve important names and modifiers instead of shortening or paraphrasing them. DO NOT truncate or summarize complex titles, roles, or location descriptions (e.g. 'marketing specialist at a small startup' must be returned in full, not shortened to 'Marketing specialist').",
        "  - When several related lines are present, prefer the one that states the answer most directly instead of nearby background or process details.",
        "  - For factual questions, return only the answer span: a name, noun phrase, number, date, or list. No leading phrase, no explanation, no source mention.",
        insufficient_rule,
    ]
    if compact_reader:
        package = rendered_evidence_package or {}
        _schema_lower = str(package.get("schema_name") or query_family or "").strip().lower()
        _is_multi_session_factlist = _schema_lower == "multisessionfactlist" or str(package.get("mode") or "") == "multi_session_fact_list"
        _is_count_or_sum = _schema_lower in {
            "countlookup", "countevents", "countdistinctitems",
            "sumoperands", "averageaggregate", "deltaaggregate",
            "extremumselection", "multisessionfactlist",
        } or str(query_family or "").strip().lower() in {"aggregation", "multi_session"}
        _qtype = str(question_type or "").strip().lower()
        _is_yes_no = any(lowered_question.startswith(p) for p in (
            "is ", "do ", "does ", "did ", "was ", "were ", "has ", "have ",
            "can ", "are ", "will ", "could ", "should ", "would ",
            "before ",
        )) or any(phrase in lowered_question for phrase in (
            "did i ", "was i ", "have i ", "do i ", "am i ",
        ))

        # --- Compact reader: tight, question-type-specific rules ---
        _is_relative_duration = (
            " ago" in lowered_question
            and lowered_question.startswith("how many ")
            and any(unit in lowered_question for unit in ("day", "week", "month", "year"))
        )

        # Base rules for all compact reader rows (minimal)
        rules = [
            "  - Return only the answer span: a name, phrase, number, date, or list. Preserve full names and modifiers.",
        ]
        if _is_yes_no:
            rules.append("  - For yes/no questions, start with Yes or No.")
        rules.append(insufficient_rule)

        if _is_count_or_sum:
            # Aggregation-specific rules
            rules.extend([
                "  - If an 'enumerated_items' slot is present, count exactly those items.",
                "  - If an 'operand_summary' slot is present, compute the answer from those operands.",
                "  - Otherwise, count or sum from evidence lines AND additional context together.",
                "  - Only count user first-person statements; skip assistant advice.",
                "  - Each line from a different session date may describe a separate item.",
            ])
        else:
            # Lookup-specific rules — trust slots for preference/KU (compiler
            # synthesises the answer there); question slots for other types
            # where extraction errors are common.
            _trust_slots = _qtype in (
                "single-session-preference", "knowledge-update",
                "preference_summary",
            ) or _schema_lower in ("preferencesynthesis",)
            if _trust_slots:
                rules.extend([
                    "  - Answer from validated slots first; copy the answer phrase directly.",
                    "  - If slots are insufficient, check evidence lines and additional context.",
                ])
            else:
                rules.extend([
                    "  - Validated slots below are machine-extracted hints. Cross-check them against the evidence lines before using.",
                    "  - Derive your answer primarily from evidence lines. Use slots only if they are consistent with the evidence.",
                ])

        # Question-type-specific rules (only added when relevant)
        if _qtype == "temporal-reasoning":
            rules.append("  - Evidence lines start with YYYY/MM/DD dates. Use these dates for any time computation.")
            if question_date:
                rules.append(f"  - Question date: {question_date}. Compute differences relative to this date.")
            rules.append("  - If a 'computed_date_hint' slot exists, USE IT directly.")
            rules.append("  - For ordering: earlier YYYY/MM/DD date = happened first. Sort by dates, not list position.")
            if _is_relative_duration:
                rules.append("  - Output exactly '<number> <unit> ago' for relative-duration questions.")

        elif _qtype == "multi-session" or _is_multi_session_factlist:
            rules.append("  - Evidence lines are facts gathered from MULTIPLE conversation sessions. Each line prefixed with [YYYY/MM/DD] is from a different session.")
            rules.append("  - PRIORITY: If a 'computed_aggregate' slot exists, USE THAT VALUE directly — it was computed programmatically and overrides any manual count.")
            rules.append("  - Otherwise, if an 'enumerated_items' slot exists, count exactly those items — each item is one distinct thing.")
            if any(phrase in lowered_question for phrase in ("how much", "how many", "total", "number of", "count")):
                rules.append("  - FIRST list every matching item/event with its date, one per line.")
                rules.append("  - THEN count how many items you listed (or sum amounts if asking 'how much').")
                rules.append("  - FINALLY state your answer. Only count items the user explicitly mentioned doing, having, or experiencing — skip assistant advice.")
            elif any(phrase in lowered_question for phrase in ("average", "difference", "most", "least", "percentage")):
                rules.append("  - FIRST list the relevant numbers from each evidence line.")
                rules.append("  - THEN compute the requested aggregation.")
                rules.append("  - FINALLY state your answer.")
            else:
                rules.append("  - Combine information from ALL evidence lines — the answer requires synthesizing facts across sessions.")

        elif _qtype == "knowledge-update":
            rules.append("  - Information may be UPDATED. Prefer the value from the LATER date.")

        elif _qtype == "single-session-assistant":
            rules.append("  - The answer may be in assistant-spoken content. Check all speakers.")
    else:
        # Non-compact reader: keep original verbose rules
        pass

    if not compact_reader:
        # All the original non-compact rules (unchanged)
        if lowered_question.startswith("why "):
            rules.insert(
                6,
                "  - For why questions, prefer the most direct stated reason and avoid replacing it with broader background motivations.",
            )
        if lowered_question.startswith(("what has ", "what did ")) and (
            " done" in lowered_question or lowered_question.endswith(" do") or " do " in lowered_question
        ):
            rules.insert(
                7,
                "  - When the question asks what someone has done, include all distinct supported actions instead of only the first one.",
            )

        if any(phrase in lowered_question for phrase in ("how much", "how many", "total")):
            rules.insert(
                8,
                "  - If the question asks for a total amount, distance, or cost, mathematically sum the individual numbers found in the text and return ONLY the final computed sum.",
            )
        if harm_type != "contradiction_resolution":
            rules.insert(
                1,
                "  - If retrieved memories conflict, prefer the newer or explicitly updated memory.",
            )
            rules.insert(
                5,
                "  - For yes/no questions, start the answer with Yes or No.",
            )
        else:
            rules.append(
                "  - If the retrieved memories contain unresolved contradictory claims about the asked fact, do not resolve the conflict. Say that the chat contains contradictory information, briefly mention both sides, and ask for clarification."
            )
        if harm_type == "event_ordering":
            rules.extend(
                [
                    "  - For ordering questions, summarize the progression across the retrieved memories in chronological order.",
                    "  - If the question asks for a fixed number of items, give short ordered items that capture major phases rather than repeating low-level details.",
                ]
            )
        if variant == "tail_focus_v1":
            rules.extend(
                [
                    "  - Treat explicit update phrases like switched to, moved to, now uses, now prefers, no longer, and instead as strong evidence of current state.",
                    "  - If the user asks you to draft, write, or phrase something, answer with exactly one short complete sentence.",
                ]
            )
        is_relative_duration_query = (
            " ago" in lowered_question
            and lowered_question.startswith("how many ")
            and any(unit in lowered_question for unit in ("day", "week", "month", "year"))
        )
        if question_date:
            rules.insert(
                4,
                "  - Use the question date as the reference 'now' when the user asks how long ago something happened or when a relative answer depends on when the question was asked.",
            )
        elif is_relative_duration_query:
            rules.insert(
                4,
                "  - If no explicit question date is provided, use the most recent dated line in the provided context as the reference 'now' for relative-time answers.",
            )
        if is_relative_duration_query:
            rules.insert(
                9,
                "  - If the question asks how many days, weeks, months, or years ago something happened, answer with that relative duration in the requested unit instead of giving only a calendar date.",
            )
            rules.insert(
                10,
                "  - For relative-duration questions, output exactly '<number> <unit> ago' and nothing else.",
            )

    question_date_block = f"\nQuestion date: {question_date}\n" if question_date else ""

    if compact_reader:
        compact_block = _compact_reader_block(rendered_evidence_package)
        return (
            "You are a careful conversational memory assistant.\n"
            "Answer the user's question from the validated evidence below.\n"
            "Rules:\n"
            + "\n".join(rules)
            + "\n\n"
            f"Question: {question}\n\n"
            + question_date_block
            + "\n"
            f"Current dialogue context:\n{current_block}\n\n"
            f"Validated evidence package:\n{compact_block}\n\n"
            "Answer:"
        )

    if variant in _PREFERENCE_READER_VARIANTS:
        return (
            "You are a conversational memory assistant that describes user preferences.\n"
            "The user is asking for a recommendation. Use the retrieved memories to identify "
            "what the user likes, prefers, uses, or has mentioned interest in.\n"
            "Rules:\n"
            "  - Answer in EXACTLY this format: \"The user would prefer [responses/suggestions] that [specific preference details from memories].\"\n"
            "  - Include specific details: brands, tools, topics, activities, styles, or constraints the user mentioned.\n"
            "  - Do not give a recommendation. Describe what kind of response the user would prefer.\n"
            "  - Do not fabricate preferences the user never expressed.\n"
            "  - If the memories contain no relevant preference information, answer exactly: Unknown.\n"
            "\n"
            f"Question: {question}\n\n"
            + question_date_block
            + "\n"
            f"Current dialogue context:\n{current_block}\n\n"
            f"Retrieved memories:\n{memory_block}\n\n"
            "Answer:"
        )

    return (
        "You are a careful conversational memory assistant.\n"
        "Answer the user's question using the current dialogue context and retrieved evidence below.\n"
        "Rules:\n"
        + "\n".join(rules)
        + "\n\n"
        f"Question: {question}\n\n"
        + question_date_block
        + "\n"
        f"Current dialogue context:\n{current_block}\n\n"
        f"Retrieved evidence:\n{memory_block}\n\n"
        "Answer:"
    )
