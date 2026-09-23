"""Turning raw feedback into analyzable *segments*.

A support ticket is rarely about one thing. It opens with a greeting, mentions how long
the customer has been around, describes one or two problems and signs off. Embedding the
whole ticket blends all of that into one vector, and tickets end up grouped by *writing
style* instead of by *problem*.

Clamor therefore works at the sentence level:

1. split every item into segments (sentences, or the quoted part of call notes),
2. mark segments that are boilerplate (greetings, sign-offs, "please fix asap", ...),
3. cluster only the content segments, and let one item belong to several themes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

# Generic sentences that carry tone or context but no product signal. They are matched
# semantically (embedding similarity), so they do not need to be exhaustive.
BOILERPLATE_PROTOTYPES: tuple[str, ...] = (
    "Hi team",
    "Hello support",
    "Thanks in advance",
    "Thank you!",
    "Best regards, Maria",
    "Please fix this asap.",
    "This is really frustrating.",
    "Very annoying.",
    "Hope this gets fixed soon.",
    "Not happy about this.",
    "Would really appreciate it.",
    "Is this on the roadmap?",
    "Keep up the great work!",
    "Five stars.",
    "We use the product every day.",
    "Our team of 20 people relies on it.",
    "I have been a customer for 2 years.",
    "It is central to how we plan our week.",
    "Renewal is in 4 weeks.",
    "Expansion to 30 more seats depends on this.",
    "They mentioned evaluating competitors.",
    "Otherwise very happy with the product.",
    "Anyone else seeing this?",
    "Any update on this?",
    "Let me know if you need more details.",
)

_QUOTE = re.compile(r"[\"“]([^\"”]{15,})[\"”]")
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
# leading noise inside a segment: forum tags, "Feature request:", greetings, discourse markers
_LEADING_JUNK = re.compile(
    r"^(\[[^\]]{1,25}\]\s*"
    r"|[+\w][\w +'-]{1,24}:\s+"
    r"|(hi|hello|hey|dear|good (morning|afternoon))\b[^,.!?]{0,25}[,!.]\s*"
    r"|(also|and|but|plus|btw|ps|fyi|anyway|small thing,? but|one more thing)[,:]?\s+)",
    re.I,
)
# segments that are only a greeting or a sign-off ("Thanks, Maria", "- Omar", "Hi team,")
_SALUTATION = re.compile(
    r"^(?:[-–—~]+|(?i:thanks|thank you|thx|cheers|best|regards|kind regards|best regards"
    r"|sincerely|hi|hello|hey|dear)\b)[\s,.!]*(?:[A-Z][\w.'-]*\s*){0,3}[.!]?$"
)


def split_segments(text: str, product_names: Iterable[str] = ()) -> list[str]:
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
        text = re.sub(rf"\b{re.escape(name)}\b", "", text, flags=re.I)
    segments = []
    for raw in _SPLIT.split(text):
        seg = raw.strip(" ,;\t")
        if _SALUTATION.match(seg):
            continue
        for _ in range(3):  # "[Bug] Hi team, Also, ..." stacks several layers
            stripped = _LEADING_JUNK.sub("", seg, count=1)
            if stripped == seg:
                break
            seg = stripped
        seg = re.sub(r"\s{2,}", " ", seg).strip(" ,;-–—")
        if len(seg) >= 3 and re.search(r"[A-Za-z]", seg):
            if seg[1:2].islower():  # re-capitalize after stripping a prefix, keep "iCloud"
                seg = seg[0].upper() + seg[1:]
            segments.append(seg)
    return segments


def segment_feedback(feedback: pd.DataFrame, product_names: Iterable[str] = ()) -> pd.DataFrame:
    """Explode a feedback table into one row per segment.

    Returns columns: ``doc`` (row position in `feedback`), ``position`` (segment index
    within the item) and ``text``. Items that yield no segment keep their full text so no
    feedback silently disappears.
    """
    names = tuple(product_names)
    rows = []
    for doc, text in enumerate(feedback["text"].tolist()):
        segs = split_segments(text, names) or [str(text).strip() or "(empty)"]
        rows.extend({"doc": doc, "position": i, "text": s} for i, s in enumerate(segs))
    return pd.DataFrame(rows)


def boilerplate_mask(
    segment_vectors: np.ndarray, prototype_vectors: np.ndarray, threshold: float
) -> np.ndarray:
    """True for segments whose closest boilerplate prototype is at least `threshold` similar."""
    if len(segment_vectors) == 0:
        return np.zeros(0, dtype=bool)
    return (segment_vectors @ prototype_vectors.T).max(axis=1) >= threshold
