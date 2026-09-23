"""Ablation study: what does each design decision buy?

Runs the full pipeline and evaluation on the demo dataset, switching off one component
at a time, and prints a Markdown table (used in docs/methodology.md).

    python scripts/ablation.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clamor import synth  # noqa: E402
from clamor.config import Config  # noqa: E402
from clamor.evaluate import evaluate_all  # noqa: E402
from clamor.pipeline import analyze  # noqa: E402

BASE = Config(product_names=("Tempo",))
VARIANTS = {
    "Clamor (default)": BASE,
    "Whole tickets instead of sentences": replace(BASE, segment_sentences=False),
    "No boilerplate filter": replace(BASE, boilerplate_threshold=1.01),
    "No theme consolidation": replace(BASE, consolidate=False),
    "Total-volume normalization": replace(BASE, volume_normalization="total"),
    "Lexical embeddings (TF-IDF)": replace(BASE, embedding="tfidf"),
}


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data" / "demo"
    if not (data_dir / "feedback.csv").exists():
        synth.generate().save(data_dir)
    ds = synth.SyntheticDataset.load(data_dir)
    print(
        "| Variant | Themes | ARI | Homogeneity | Recovered | Releases linked | "
        "Spikes caught | False alarms | Spurious trend flags | Runtime |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, cfg in VARIANTS.items():
        start = time.time()
        result = analyze(ds.feedback, ds.accounts, ds.releases, config=cfg)
        ev = evaluate_all(result, ds)
        t, a = ev["themes"], ev["alerts"]["clamor"]
        rel = ev["releases"]
        print(
            f"| {name} | {t['themes_found']} | {t['adjusted_rand']:.3f} | "
            f"{t['homogeneity']:.3f} | {t['mean_recovered_share']:.0%} | "
            f"{int(rel['correct'].sum())}/{len(rel)} | {a['spikes_detected']}/"
            f"{a['spikes_total']} | {a['false_alarm_episodes']} | "
            f"{a['spurious_trend_days']} | "
            f"{time.time() - start:.0f}s |",
            flush=True,
        )


if __name__ == "__main__":
    main()
