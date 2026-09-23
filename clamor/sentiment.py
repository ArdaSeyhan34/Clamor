"""A small, transparent sentiment scorer tuned for product feedback.

General-purpose lexicons miss the vocabulary of software complaints: "crashes", "laggy",
"duplicate", "stuck" and "unusable" are neutral words in a movie review. This scorer uses a
compact domain lexicon with negation and intensifier handling, then blends in the star
rating or NPS score when the channel provides one.

It is intentionally simple and explainable; swapping in a transformer classifier would be a
one-function change.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

LEXICON: dict[str, float] = {
    # strongly negative
    "unusable": -3.0,
    "broken": -2.5,
    "crash": -2.5,
    "crashes": -2.5,
    "crashing": -2.5,
    "terrible": -3.0,
    "awful": -3.0,
    "horrible": -3.0,
    "worst": -3.0,
    "useless": -2.8,
    "unacceptable": -2.8,
    "furious": -3.0,
    "hate": -2.8,
    "disappointed": -2.2,
    "frustrating": -2.2,
    "frustrated": -2.2,
    "unfair": -2.0,
    "unreliable": -2.3,
    "fails": -2.0,
    "failed": -2.0,
    "failing": -2.0,
    "error": -1.8,
    "errors": -1.8,
    "bug": -1.8,
    "buggy": -2.2,
    "freezes": -2.2,
    "freeze": -2.0,
    "hangs": -1.8,
    "missed": -1.6,
    "lost": -1.8,
    "disappear": -2.0,
    "disappeared": -2.0,
    "disappears": -2.0,
    "wrong": -1.6,
    "stuck": -1.6,
    "spam": -2.0,
    "drowning": -2.0,
    "painful": -1.8,
    "confusing": -1.6,
    "confused": -1.4,
    "annoying": -1.8,
    "annoyed": -1.8,
    "slow": -1.5,
    "laggy": -1.8,
    "lag": -1.5,
    "drains": -1.6,
    "duplicate": -1.4,
    "duplicates": -1.4,
    "twice": -0.8,
    "expensive": -1.6,
    "pricey": -1.4,
    "overpriced": -2.2,
    "switching": -1.0,
    "risk": -1.0,
    "blocker": -1.5,
    "problem": -1.2,
    "problems": -1.2,
    "issue": -1.0,
    "issues": -1.0,
    "hurts": -1.5,
    "cannot": -0.8,
    "unhappy": -2.0,
    "forever": -0.8,
    "ignore": -1.0,
    "resets": -1.0,
    "lose": -1.5,
    # positive
    "love": 2.6,
    "loving": 2.4,
    "great": 2.2,
    "excellent": 2.8,
    "amazing": 2.8,
    "fantastic": 2.8,
    "awesome": 2.6,
    "best": 2.4,
    "perfect": 2.6,
    "intuitive": 2.0,
    "easy": 1.6,
    "reliable": 1.8,
    "fast": 1.4,
    "clean": 1.2,
    "responsive": 1.6,
    "happy": 1.8,
    "helpful": 1.8,
    "nice": 1.4,
    "good": 1.4,
    "saves": 1.6,
    "worth": 1.4,
    "appreciate": 1.2,
    "thanks": 0.6,
    "thank": 0.6,
    "smooth": 1.4,
    "simple": 1.0,
    "recommend": 1.8,
    "works": 0.6,
    "improved": 1.2,
    "better": 1.0,
    "fixed": 0.8,
}
NEGATIONS = {
    "not",
    "no",
    "never",
    "dont",
    "doesnt",
    "didnt",
    "isnt",
    "wasnt",
    "cant",
    "cannot",
    "wont",
    "without",
    "hardly",
}
INTENSIFIERS = {
    "very": 1.3,
    "really": 1.3,
    "so": 1.2,
    "extremely": 1.5,
    "completely": 1.4,
    "totally": 1.4,
    "super": 1.3,
    "incredibly": 1.5,
    "way": 1.2,
}

_TOKEN = re.compile(r"[a-z]+")


def score_text(text: str) -> float:
    """Sentiment in [-1, 1] for one text."""
    if not isinstance(text, str):
        return 0.0
    tokens = _TOKEN.findall(text.lower().replace("'", "").replace("’", ""))
    total = 0.0
    for i, tok in enumerate(tokens):
        value = LEXICON.get(tok)
        if value is None:
            continue
        window = tokens[max(0, i - 3) : i]
        if any(w in NEGATIONS for w in window):
            value *= -0.6  # "not great" is negative, but milder than "terrible"
        if i > 0 and tokens[i - 1] in INTENSIFIERS:
            value *= INTENSIFIERS[tokens[i - 1]]
        total += value
    exclaim = min(text.count("!"), 3) * 0.15
    total += math.copysign(exclaim, total) if total else 0.0
    return float(total / math.sqrt(total * total + 8.0))  # VADER-style squashing


def rating_to_sentiment(rating: float, channel: str) -> float | None:
    """Map a star rating (1-5) or NPS score (0-10) to [-1, 1]."""
    if rating is None or (isinstance(rating, float) and math.isnan(rating)):
        return None
    if channel == "nps_survey" or rating > 5:
        return float(np.clip((rating - 7.0) / 3.0, -1, 1))  # 7 is the passive midpoint
    return float(np.clip((rating - 3.0) / 2.0, -1, 1))


def score_feedback(feedback: pd.DataFrame, rating_weight: float = 0.4) -> pd.Series:
    """Blend text sentiment with the explicit rating where one exists."""
    text_scores = feedback["text"].map(score_text)
    if "rating" not in feedback:
        return text_scores.rename("sentiment")
    channels = feedback["channel"] if "channel" in feedback else pd.Series("", index=feedback.index)
    rating_scores = [
        rating_to_sentiment(r, c) for r, c in zip(feedback["rating"], channels, strict=True)
    ]
    blended = [
        t if r is None else (1 - rating_weight) * t + rating_weight * r
        for t, r in zip(text_scores, rating_scores, strict=True)
    ]
    return pd.Series(blended, index=feedback.index, name="sentiment")
