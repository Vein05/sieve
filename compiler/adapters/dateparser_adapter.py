"""Small wrapper around dateparser so schema logic stays dependency-agnostic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any

import dateparser
from dateparser.search import search_dates


@dataclass(frozen=True)
class DateSpan:
    text: str
    value: datetime
    start: int
    end: int


@lru_cache(maxsize=1)
def _default_settings() -> dict[str, Any]:
    return {
        "RETURN_AS_TIMEZONE_AWARE": False,
        "PREFER_DAY_OF_MONTH": "first",
        "PREFER_DATES_FROM": "past",
    }


def parse_datetime(text: str, *, relative_base: datetime | None = None) -> datetime | None:
    settings = dict(_default_settings())
    if relative_base is not None:
        settings["RELATIVE_BASE"] = relative_base
    return dateparser.parse(text, settings=settings)


def search_datetimes(text: str, *, relative_base: datetime | None = None) -> tuple[DateSpan, ...]:
    settings = dict(_default_settings())
    if relative_base is not None:
        settings["RELATIVE_BASE"] = relative_base
    matches = search_dates(text, settings=settings) or []
    spans: list[DateSpan] = []
    search_start = 0
    for matched_text, value in matches:
        start = text.find(matched_text, search_start)
        if start < 0:
            start = text.find(matched_text)
        if start < 0:
            continue
        end = start + len(matched_text)
        search_start = end
        spans.append(DateSpan(text=matched_text, value=value, start=start, end=end))
    return tuple(spans)
