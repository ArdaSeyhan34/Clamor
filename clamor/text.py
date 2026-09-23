"""Turning raw feedback into analyzable *segments*.

A support ticket is rarely about one thing. It opens with a greeting, mentions how long
the customer has been around, describes one or two problems and signs off. Embedding the
whole ticket blends all of that into one vector, and tickets end up grouped by *writing
style* instead of by *problem*.

Clamor therefore works at the sentence level:

1. split every item into segments (sentences, or the quoted part of call notes),
2. mark segments that are boilerplate (greetings, sign-offs, "please fix asap", ...),
3. cluster only the content segments, and let one item belong to several themes.

Language-specific patterns (greetings, sign-offs, discourse markers) live in
:mod:`clamor.lang`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .lang import ENGLISH, LanguagePack

# Kept for backwards compatibility; the language pack is the source of truth.
BOILERPLATE_PROTOTYPES: tuple[str, ...] = ENGLISH.boilerplate_prototypes

_QUOTE = re.compile(r"[\"“]([^\"”]{15,})[\"”]")
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_segments(
    text: str, product_names: Iterable[str] = (), lang: LanguagePack = ENGLISH
) -> list[str]:
    """Split one feedback item into content-bearing segments.

    If the item quotes the customer (typical for sales or CS call notes), only the quotes
    are kept: the surrounding note is the account manager talking, not the customer.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    quotes = _QUOTE.findall(text)
    if quotes:
        text = " ".join(q.strip() for q in quotes)
    for name in product_names:
        # also drops Turkish suffixes after an apostrophe: "Lezzo'da", "Lezzo'nun"
        text = re.sub(rf"\b{re.escape(name)}(?:['’][^\W\d_]+)?\b", "", text, flags=re.I)
    segments = []
    for raw in _SPLIT.split(text):
        seg = raw.strip(" ,;\t")
        if lang.salutation.match(seg):
            continue
        for _ in range(3):  # "[Bug] Hi team, Also, ..." stacks several layers
            stripped = lang.leading_junk.sub("", seg, count=1)
            if stripped == seg:
                break
            seg = stripped
        seg = re.sub(r"\s{2,}", " ", seg).strip(" ,;-–—")
        if len(seg) >= 3 and re.search(r"[^\W\d_]", seg):
            if seg[1:2].islower():  # re-capitalize after stripping a prefix, keep "iCloud"
                seg = seg[0].upper() + seg[1:]
            segments.append(seg)
    return segments


def segment_feedback(
    feedback: pd.DataFrame, product_names: Iterable[str] = (), lang: LanguagePack = ENGLISH
) -> pd.DataFrame:
    """Explode a feedback table into one row per segment.

    Returns columns: ``doc`` (row position in `feedback`), ``position`` (segment index
    within the item) and ``text``. Items that yield no segment keep their full text so no
    feedback silently disappears.
    """
    names = tuple(product_names)
    rows = []
    for doc, text in enumerate(feedback["text"].tolist()):
        segs = split_segments(text, names, lang) or [str(text).strip() or "(empty)"]
        rows.extend({"doc": doc, "position": i, "text": s} for i, s in enumerate(segs))
    return pd.DataFrame(rows)


def boilerplate_mask(
    segment_vectors: np.ndarray, prototype_vectors: np.ndarray, threshold: float
) -> np.ndarray:
    """True for segments whose closest boilerplate prototype is at least `threshold` similar."""
    if len(segment_vectors) == 0:
        return np.zeros(0, dtype=bool)
    return (segment_vectors @ prototype_vectors.T).max(axis=1) >= threshold
