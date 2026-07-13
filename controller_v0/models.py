"""Data models for the rule_v0 controller."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextProfile:
    text: str
    normalized_text: str
    tokens: list[str]
    token_set: set[str]
    content_tokens: set[str]
    concept_scores: dict[str, float]
    date_key: tuple[int, int, int] | None
    has_update_marker: bool
    has_time_marker: bool
    is_preference: bool
    is_personal: bool
    is_sensitive: bool
    named_tokens: set[str]
