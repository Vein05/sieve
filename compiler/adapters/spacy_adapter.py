"""spaCy helpers kept behind a stable adapter boundary."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import spacy


@dataclass(frozen=True)
class EntitySpan:
    text: str
    label: str
    start: int
    end: int


@lru_cache(maxsize=1)
def _load_nlp():
    for model_name in ("en_core_web_sm",):
        try:
            return spacy.load(model_name)
        except OSError:
            continue
    return spacy.blank("en")


def spacy_model_name() -> str:
    return _load_nlp().meta.get("name", "blank:en")


def extract_entities(text: str, *, labels: set[str] | None = None) -> tuple[EntitySpan, ...]:
    doc = _load_nlp()(text)
    entities: list[EntitySpan] = []
    for ent in doc.ents:
        if labels is not None and ent.label_ not in labels:
            continue
        entities.append(EntitySpan(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char))
    return tuple(entities)


def extract_noun_phrases(text: str) -> tuple[str, ...]:
    doc = _load_nlp()(text)
    if not doc.has_annotation("DEP"):
        return ()
    phrases = [chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip()]
    return tuple(dict.fromkeys(phrases))


@lru_cache(maxsize=8192)
def extract_head_nouns(text: str) -> frozenset[str]:
    """Return the set of syntactic head nouns from noun chunks in *text*.

    For "coffee creamer" the head noun is "creamer"; for "cycling club" it is
    "club".  This is used by the entity head-noun compatibility gate to reject
    evidence whose subject entity has no head-noun overlap with the query.
    """
    from shared.cache import disk_get, disk_put, text_key

    _key = text_key(text)
    cached = disk_get("head_nouns", _key)
    if cached is not None:
        return frozenset(cached)

    doc = _load_nlp()(text)
    if not doc.has_annotation("DEP"):
        return frozenset()
    heads: set[str] = set()
    for chunk in doc.noun_chunks:
        root = chunk.root
        lemma = root.lemma_.lower().strip()
        if lemma and root.pos_ in {"NOUN", "PROPN"} and len(lemma) > 1:
            heads.add(lemma)
    result = frozenset(heads)
    disk_put("head_nouns", _key, tuple(result))
    return result
