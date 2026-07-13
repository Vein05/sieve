"""Typed runtime contracts for the compiler-first rewrite."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .intent.answer_intent import AnswerIntent
from .planner.schema import EvidencePlan


@dataclass(frozen=True)
class CandidateBinding:
    slot_name: str
    slot_type: str
    answer_type: str
    memory_id: str
    object_id: str | None
    text: str
    source_text: str
    date_key: tuple[int, int, int] | None
    value_number: float | None
    value_unit: str | None
    entity_match: float
    attribute_match: float
    unit_match: float
    focus_match: float
    confidence: float
    support_ids: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InvariantFailure:
    code: str
    slot_name: str | None
    severity: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundEvidencePackage:
    intent: AnswerIntent
    plan: EvidencePlan
    validated_bindings: dict[str, CandidateBinding | None]
    invalid_bindings: dict[str, str]
    missing_invariants: tuple[InvariantFailure, ...]
    compiled_memory_ids: tuple[str, ...]
    compiled_unit_ids: tuple[str, ...]
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.to_dict(),
            "plan": {
                "family": self.plan.family,
                "schema_name": self.plan.schema_name,
                "requires_latest_resolution": self.plan.requires_latest_resolution,
                "requires_ordering": self.plan.requires_ordering,
                "requires_numeric_reasoning": self.plan.requires_numeric_reasoning,
                "slots": [asdict(slot) for slot in self.plan.slots],
            },
            "validated_bindings": {
                name: binding.to_dict() if binding is not None else None
                for name, binding in self.validated_bindings.items()
            },
            "invalid_bindings": dict(self.invalid_bindings),
            "missing_invariants": [failure.to_dict() for failure in self.missing_invariants],
            "compiled_memory_ids": list(self.compiled_memory_ids),
            "compiled_unit_ids": list(self.compiled_unit_ids),
            "token_count": self.token_count,
        }
