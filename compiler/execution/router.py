"""Execution routing helpers shared by the legacy compiler path."""

from __future__ import annotations

from typing import Any


def normalize_post_contract(answer_contract: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(answer_contract)
    answer_mode = str(normalized.get("answer_mode") or "")
    if answer_mode == "abstain":
        normalized["answerability_level"] = "abstain"
        normalized["deterministic_allowed"] = False
    return normalized


def apply_semantic_reader_downgrade(answer_contract: dict[str, Any], semantic_analysis: dict[str, Any]) -> dict[str, Any]:
    if semantic_analysis.get("semantic_sufficiency_passed", True):
        return answer_contract
    if str(answer_contract.get("answer_mode") or "") not in {"deterministic_span", "deterministic_numeric"}:
        return answer_contract
    downgraded = dict(answer_contract)
    downgraded["answer_mode"] = "reader_from_slots"
    downgraded["answerability_level"] = "reader_only"
    downgraded["answer_text"] = None
    downgraded["deterministic_allowed"] = False
    route_reason = str(downgraded.get("reader_route_reason") or "")
    downgraded["reader_route_reason"] = (
        f"{route_reason}+semantic_support_low".strip("+")
        if route_reason
        else "semantic_support_low"
    )
    return downgraded
