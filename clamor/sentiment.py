"""A small, transparent sentiment scorer tuned for product feedback.

General-purpose lexicons miss the vocabulary of software complaints: "crashes", "laggy",
"duplicate", "stuck" and "unusable" are neutral words in a movie review. This scorer uses a
compact domain lexicon per language (see :mod:`clamor.lang`) with negation and intensifier
handling, then blends in the star rating or NPS score when the channel provides one.

Turkish is agglutinative ("çalışmıyor", "çalışmadı", "çalışmıyordu"), so its lexicon is
matched on word *prefixes* after folding away diacritics, and the negation "değil" is
handled after the word it negates.

It is intentionally simple and explainable; swapping in a transformer classifier would be a
one-function change.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd

from .lang import ENGLISH, LanguagePack

# Backwards-compatible aliases (English)
LEXICON = ENGLISH.lexicon


@lru_cache(maxsize=8)
def _sorted_keys(code: str, keys: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(keys, key=len, reverse=True))


def _lookup(tokens: list[str], i: int, lang: LanguagePack) -> tuple[float | None, int]:
    """Sentiment value at position i and how many tokens it consumed."""
    lex = lang.lexicon
    if i + 1 < len(tokens):
        pair = f"{tokens[i]} {tokens[i + 1]}"
        if pair in lex:
            return lex[pair], 2
    tok = tokens[i]
    if tok in lex:
        return lex[tok], 1
    if lang.prefix_lexicon and len(tok) > 3:
        for key in _sorted_keys(lang.code, tuple(lex)):
            if " " not in key and len(key) >= 4 and tok.startswith(key):
                return lex[key], 1
    return None, 1


def score_text(text: str, lang: LanguagePack = ENGLISH) -> float:
    """Sentiment in [-1, 1] for one text."""
    if not isinstance(text, str):
        return 0.0
    cleaned = text.replace("'", "").replace("’", "")
    tokens = lang.tokens(cleaned) if lang.prefix_lexicon else _en_tokens(cleaned)
    total = 0.0
    i = 0
    while i < len(tokens):
        value, used = _lookup(tokens, i, lang)
        if value is not None and value != 0.0:
            window = tokens[max(0, i - 3) : i]
            if any(w in lang.negations_before for w in window):
                value *= -0.6  # "not great" is negative, but milder than "terrible"
            after = tokens[i + used] if i + used < len(tokens) else ""
            if after in lang.negations_after:
                value *= -0.6  # "güzel değil"
            if i > 0 and tokens[i - 1] in lang.intensifiers:
                value *= lang.intensifiers[tokens[i - 1]]
            total += value
        i += used
    exclaim = min(text.count("!"), 3) * 0.15
    total += math.copysign(exclaim, total) if total else 0.0
    return float(total / math.sqrt(total * total + 8.0))  # VADER-style squashing


def _en_tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z]+", text.lower())


def rating_to_sentiment(rating: float, channel: str) -> float | None:
    """Map a star rating (1-5) or NPS score (0-10) to [-1, 1]."""
    if rating is None or (isinstance(rating, float) and math.isnan(rating)):
        return None
    if channel in {"nps_survey", "survey"} or rating > 5:
        return float(np.clip((rating - 7.0) / 3.0, -1, 1))  # 7 is the passive midpoint
    return float(np.clip((rating - 3.0) / 2.0, -1, 1))


def score_feedback(
    feedback: pd.DataFrame, rating_weight: float = 0.4, lang: LanguagePack = ENGLISH
) -> pd.Series:
    """Blend text sentiment with the explicit rating where one exists."""
    text_scores = feedback["text"].map(lambda t: score_text(t, lang))
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
