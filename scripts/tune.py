"""Grid-search clustering thresholds for an embedding backend on a demo scenario.

Embeddings are computed once and cached, so each grid point only re-runs clustering and
evaluation. Used to choose the per-backend defaults in clamor/config.py.

    python scripts/tune.py --scenario lezzo --backend multilingual
    python scripts/tune.py --scenario lezzo --backend hybrid --semantic-weights 0.5,0.65,0.8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clamor import synth  # noqa: E402
from clamor.cli import SCENARIO_SETTINGS  # noqa: E402
from clamor.config import Config  # noqa: E402
from clamor.embeddings import (  # noqa: E402
    HYBRID_SEMANTIC_WEIGHT,
    HybridEmbedder,
    OnnxSentenceEmbedder,
    TfidfEmbedder,
    get_embedder,
)
from clamor.evaluate import evaluate_all  # noqa: E402
from clamor.lang import get_language  # noqa: E402
from clamor.pipeline import analyze  # noqa: E402


class CachingEmbedder:
    def __init__(self, inner):
        self.inner, self.name, self.cache = inner, inner.name, {}

    def fit(self, texts):
        self.inner.fit(texts)
        return self

    def encode(self, texts):
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            for t, v in zip(missing, self.inner.encode(missing), strict=True):
                self.cache[t] = v
        return np.vstack([self.cache[t] for t in texts])


class LowercasingEmbedder:
    """Experiment: lower-case (language-aware) before a cased semantic model."""

    def __init__(self, inner, lower):
        self.inner, self.lower, self.name = inner, lower, inner.name

    def fit(self, texts):
        self.inner.fit([self.lower(t) for t in texts])
        return self

    def encode(self, texts):
        return self.inner.encode([self.lower(t) for t in texts])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="lezzo")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--distances", default="0.5,0.55,0.6,0.65,0.7")
    ap.add_argument("--boilerplate", default="0.5,0.55,0.6,0.65")
    ap.add_argument("--semantic-weights", default=None, help="hybrid only, e.g. 0.5,0.65,0.8")
    ap.add_argument("--lowercase", action="store_true", help="lower-case before embedding")
    args = ap.parse_args()

    settings = SCENARIO_SETTINGS[args.scenario]
    data = settings["data"]
    ds = (synth.SyntheticDataset.load(data) if (data / "feedback.csv").exists()
          else synth.generate_scenario(args.scenario))  # fmt: skip
    lang = get_language(settings["language"])
    backend = args.backend or lang.default_embedding
    if backend == "hybrid":  # the semantic part is cached once, the weight varies
        model = "minilm" if lang.code == "en" else "multilingual"
        weights = args.semantic_weights or str(HYBRID_SEMANTIC_WEIGHT[model])
        semantic = OnnxSentenceEmbedder(model)
        if args.lowercase:
            semantic = LowercasingEmbedder(semantic, lang.lower)
        semantic = CachingEmbedder(semantic)
        embedders = {
            w: HybridEmbedder(semantic, TfidfEmbedder(lang=lang), semantic_weight=w)
            for w in map(float, weights.split(","))
        }
    else:
        inner = get_embedder(backend, fallback=False, lang=lang)
        if args.lowercase:
            inner = LowercasingEmbedder(inner, lang.lower)
        embedders = {"-": CachingEmbedder(inner)}
    print(
        f"scenario={args.scenario} backend={backend} lowercase={args.lowercase} "
        f"items={len(ds.feedback)}"
    )
    print(
        "| semantic weight | distance | boilerplate | themes | ARI | homogeneity | recovered | "
        "releases | spikes | false alarms | spurious | seconds |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    grid = [
        (w, b, d)
        for w in embedders
        for b in map(float, args.boilerplate.split(","))
        for d in map(float, args.distances.split(","))
    ]
    for w, b, d in grid:
        embedder = embedders[w]
        start = time.time()
        cfg = Config(language=lang.code, embedding=backend,
                     product_names=settings["products"], distance_threshold=d,
                     boilerplate_threshold=b)  # fmt: skip
        result = analyze(ds.feedback, ds.accounts, ds.releases, config=cfg,
                         embedder=embedder)  # fmt: skip
        ev = evaluate_all(result, ds)
        t, a, rel = ev["themes"], ev["alerts"]["clamor"], ev["releases"]
        print(
            f"| {w} | {d} | {b} | {t['themes_found']} | {t['adjusted_rand']:.3f} | "
            f"{t['homogeneity']:.3f} | {t['mean_recovered_share']:.0%} | "
            f"{int(rel['correct'].sum())}/{len(rel)} | {a['spikes_detected']}/"
            f"{a['spikes_total']} | {a['false_alarm_episodes']} | "
            f"{a['spurious_trend_days']} | {time.time() - start:.0f} |",
            flush=True,
        )


if __name__ == "__main__":
    main()
