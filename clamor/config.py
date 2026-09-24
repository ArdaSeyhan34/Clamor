"""Configuration and prioritization presets."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Weights:
    """How much each signal contributes to the opportunity score (normalized internally)."""

    reach: float = 0.25  # how many distinct accounts raise it
    revenue: float = 0.35  # how much recurring revenue those accounts represent
    severity: float = 0.20  # how painful it is (negative sentiment share)
    momentum: float = 0.20  # is it growing right now (statistically significant lift)

    def normalized(self) -> Weights:
        total = self.reach + self.revenue + self.severity + self.momentum
        if total <= 0:
            return Weights(0.25, 0.25, 0.25, 0.25)
        return Weights(
            self.reach / total, self.revenue / total, self.severity / total, self.momentum / total
        )


PRESETS: dict[str, Weights] = {
    "balanced": Weights(),
    "growth": Weights(reach=0.50, revenue=0.15, severity=0.20, momentum=0.15),
    "enterprise": Weights(reach=0.10, revenue=0.60, severity=0.15, momentum=0.15),
    "quality": Weights(reach=0.15, revenue=0.15, severity=0.35, momentum=0.35),
}

# Tuned on the synthetic benchmark (see docs/methodology.md); similarities live on
# different scales for semantic and lexical vectors, hence per-backend defaults.
BACKEND_DEFAULTS: dict[str, dict[str, float]] = {
    "hybrid": {"distance_threshold": 0.70, "boilerplate_threshold": 0.55},
    "hybrid_multilingual": {"distance_threshold": 0.60, "boilerplate_threshold": 0.55},
    "minilm": {"distance_threshold": 0.65, "boilerplate_threshold": 0.60},
    "multilingual": {"distance_threshold": 0.60, "boilerplate_threshold": 0.55},
    "tfidf": {"distance_threshold": 0.90, "boilerplate_threshold": 0.50},
    "st": {"distance_threshold": 0.65, "boilerplate_threshold": 0.60},
}


@dataclass(frozen=True)
class Config:
    language: str = "en"  # "en" or "tr" (see clamor.lang)
    embedding: str | None = None  # None: the language's default (minilm / hybrid)
    redact_pii: bool = True  # mask e-mails, phones, cards, IBANs, national IDs on load
    product_names: tuple[str, ...] = ()
    distance_threshold: float | None = None
    boilerplate_threshold: float | None = None
    min_theme_share: float = 0.005  # themes need >= 0.5% of items (and >= 5 items)
    reassign_similarity: float = 0.5
    consolidate: bool = True  # merge near-duplicate themes (see themes.consolidate)
    segment_sentences: bool = True  # cluster sentences, not whole tickets (see text.py)
    volume_normalization: str = "median_of_ratios"  # or "total" (see stats.py)
    score_window_days: int = 60  # reach/revenue/severity reflect current demand, not history
    recent_days: int = 28
    baseline_days: int = 84
    alpha: float = 0.05
    impact_window_days: int = 28
    release_match_similarity: float = 0.35
    weights: Weights = field(default_factory=Weights)
    random_state: int = 0
    report_language: str | None = None  # of reports and briefs; None: same as `language`

    @property
    def backend(self) -> str:
        if self.embedding:
            return self.embedding
        from .lang import get_language

        return get_language(self.language).default_embedding

    @property
    def output_language(self) -> str:
        from .i18n import OUTPUT_LANGUAGES

        lang = self.report_language or self.language
        return lang if lang in OUTPUT_LANGUAGES else "en"

    def backend_default(self, key: str, backend: str) -> float:
        family = "st" if backend.startswith("st:") else backend
        if family == "hybrid" and self.language != "en":  # multilingual MiniLM inside
            family = "hybrid_multilingual"
        return BACKEND_DEFAULTS.get(family, BACKEND_DEFAULTS["minilm"])[key]

    def with_weights(self, weights: Weights) -> Config:
        return replace(self, weights=weights)
