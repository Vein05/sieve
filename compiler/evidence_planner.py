"""Compatibility wrapper for the compiler-first evidence planner."""

from __future__ import annotations

from typing import Any

from .intent.answer_intent import parse_answer_intent
from .planner.planner import build_evidence_plan
from evidence.schema import EvidencePlan


def plan_evidence(row: dict[str, Any], query_family: str) -> EvidencePlan:
    intent = parse_answer_intent(row, query_family)
    return build_evidence_plan(intent, row)
