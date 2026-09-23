"""Command-line interface: ``clamor demo``, ``clamor analyze``, ``clamor evaluate``."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import typer

from . import llm, synth
from .config import PRESETS, Config
from .insights import headline_insights
from .pipeline import Analysis, analyze, review_with_claude
from .report import write_report

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Clamor: turn customer feedback into a prioritized, evidence-backed roadmap.",
)
DEMO_DATA = Path("data/demo")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _print_summary(analysis: Analysis, seconds: float) -> None:
    typer.secho(
        f"\nAnalyzed {len(analysis.feedback):,} items into {len(analysis.themes)} themes "
        f"in {seconds:.1f}s (backend: {analysis.model.backend})\n",
        bold=True,
    )
    for ins in headline_insights(analysis):
        typer.secho(f"  • {ins.title}", bold=True)
        typer.echo(f"    {ins.detail}")
    road = analysis.roadmap.head(10)
    table = pd.DataFrame(
        {
            "#": road["rank"].astype(int).to_numpy(),
            "theme": road["name"].str.slice(0, 44).to_numpy(),
            "score": road["score"].round(1).to_numpy(),
            "mentions": road["mentions"].astype(int).to_numpy(),
            "trend": road["status"].to_numpy(),
            "by votes": road["vote_rank"].astype(int).to_numpy(),
        }
    )
    typer.echo("\nTop of the roadmap:\n")
    typer.echo(table.to_string(index=False))


def _maybe_review(analysis: Analysis, use_llm: bool | None) -> Analysis:
    if use_llm is False or (use_llm is None and not llm.available()):
        return analysis
    try:
        typer.echo("Asking Claude to review theme names and duplicates...")
        return review_with_claude(analysis)
    except llm.LLMUnavailable as exc:
        typer.secho(f"Claude review skipped: {exc}", fg="yellow")
        return analysis


@app.command()
def generate(
    out: Path = typer.Option(DEMO_DATA, help="Directory for the generated CSV files."),
    seed: int = typer.Option(7, help="Random seed (same seed, same data)."),
    accounts: int = typer.Option(600, help="Number of customer accounts."),
    days: int = typer.Option(synth.DEFAULT_DAYS, help="Days of history to simulate."),
) -> None:
    """Generate the synthetic 'Tempo' dataset with ground truth."""
    data = synth.generate(seed=seed, n_accounts=accounts, days=days)
    data.save(out)
    typer.echo(
        f"Wrote {len(data.feedback):,} feedback items, {len(data.accounts)} accounts and "
        f"{len(data.releases)} releases to {out}/"
    )


@app.command("analyze")
def analyze_cmd(
    feedback: Path = typer.Argument(..., help="CSV/JSON/Excel with at least text and a date."),
    accounts: Path | None = typer.Option(None, help="Accounts with account_id and mrr."),
    releases: Path | None = typer.Option(None, help="Changelog with date and title."),
    out: Path = typer.Option(Path("reports/latest"), help="Output directory."),
    as_of: str | None = typer.Option(None, help="Analyze as if today were this date."),
    backend: str = typer.Option("minilm", help="minilm | hybrid | tfidf | st:<model>"),
    preset: str = typer.Option("balanced", help=f"Weight preset: {', '.join(PRESETS)}"),
    product_name: list[str] = typer.Option([], help="Product name(s) to ignore in text."),
    briefs: int = typer.Option(3, help="Opportunity briefs to write for the top themes."),
    use_llm: bool | None = typer.Option(
        None, "--llm/--no-llm", help="Use Claude (default: if ANTHROPIC_API_KEY is set)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Analyze your own feedback export and write a report."""
    _setup_logging(verbose)
    if preset not in PRESETS:
        raise typer.BadParameter(f"unknown preset {preset!r}")
    config = Config(embedding=backend, product_names=tuple(product_name), weights=PRESETS[preset])
    start = time.time()
    result = analyze(feedback, accounts, releases, config=config, as_of=as_of)
    result = _maybe_review(result, use_llm)
    _print_summary(result, time.time() - start)
    paths = write_report(result, out, n_briefs=briefs, use_llm=use_llm)
    typer.echo(
        f"\nReport: {paths['html']}  (also {paths['markdown'].name}, "
        f"{paths['roadmap'].name}, briefs/)"
    )


@app.command()
def demo(
    out: Path = typer.Option(Path("reports/demo"), help="Output directory for the report."),
    data: Path = typer.Option(DEMO_DATA, help="Where the demo CSVs live (generated if missing)."),
    backend: str = typer.Option("minilm", help="minilm | hybrid | tfidf"),
    use_llm: bool | None = typer.Option(None, "--llm/--no-llm"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full pipeline on the synthetic dataset, evaluate it and write a report."""
    from .evaluate import evaluate_all

    _setup_logging(verbose)
    if not (data / "feedback.csv").exists():
        synth.generate().save(data)
    dataset = synth.SyntheticDataset.load(data)
    config = Config(embedding=backend, product_names=("Tempo",))
    start = time.time()
    result = analyze(dataset.feedback, dataset.accounts, dataset.releases, config=config)
    result = _maybe_review(result, use_llm)
    _print_summary(result, time.time() - start)
    evaluation = evaluate_all(result, dataset)
    paths = write_report(result, out, evaluation=evaluation, use_llm=use_llm)
    t, a = evaluation["themes"], evaluation["alerts"]
    typer.echo(
        f"\nAccuracy vs ground truth: ARI {t['adjusted_rand']:.3f}, NMI {t['nmi']:.3f}, "
        f"homogeneity {t['homogeneity']:.3f}; early warning: "
        f"{a['clamor']['spikes_detected']}/{a['clamor']['spikes_total']} spikes, "
        f"{a['clamor']['false_alarm_episodes']} false alarms "
        f"(naive rule: {a['naive_2x']['false_alarm_episodes']})."
    )
    typer.echo(f"Report: {paths['html']}")


@app.command()
def evaluate(
    data: Path = typer.Option(DEMO_DATA, help="Synthetic dataset directory."),
    backends: list[str] = typer.Option(
        ["tfidf", "minilm", "hybrid"], "--backend", help="Backends to compare (repeatable)."
    ),
    output: Path | None = typer.Option(None, "--json", help="Also write results as JSON."),
) -> None:
    """Benchmark embedding backends against the ground truth."""
    from .evaluate import evaluate_all

    if not (data / "feedback.csv").exists():
        synth.generate().save(data)
    dataset = synth.SyntheticDataset.load(data)
    rows, raw = [], {}
    for name in backends:
        cfg = Config(embedding=name, product_names=("Tempo",))
        start = time.time()
        result = analyze(dataset.feedback, dataset.accounts, dataset.releases, config=cfg)
        if result.model.backend != name:
            typer.secho(f"{name}: unavailable, fell back to {result.model.backend}", fg="yellow")
            continue
        ev = evaluate_all(result, dataset)
        t, a = ev["themes"], ev["alerts"]["clamor"]
        rows.append(
            {
                "backend": name,
                "themes": t["themes_found"],
                "ARI": round(t["adjusted_rand"], 3),
                "NMI": round(t["nmi"], 3),
                "homogeneity": round(t["homogeneity"], 3),
                "recovered": f"{t['mean_recovered_share']:.0%}",
                "releases linked": f"{int(ev['releases']['correct'].sum())}/{len(ev['releases'])}",
                "spikes caught": f"{a['spikes_detected']}/{a['spikes_total']}",
                "false alarms": a["false_alarm_episodes"],
                "seconds": round(time.time() - start, 1),
            }
        )
        raw[name] = {k: v for k, v in ev.items() if isinstance(v, dict)}
    typer.echo(pd.DataFrame(rows).to_string(index=False))
    if output:
        output.write_text(json.dumps(raw, indent=2, default=float))


@app.command()
def brief(
    theme_id: str = typer.Argument(..., help="Theme id from a report, e.g. T03."),
    data: Path = typer.Option(DEMO_DATA, help="Demo dataset directory."),
    use_llm: bool | None = typer.Option(None, "--llm/--no-llm"),
) -> None:
    """Print the opportunity brief for one theme of the demo dataset."""
    from .briefs import write_brief

    dataset = synth.SyntheticDataset.load(data)
    result = analyze(
        dataset.feedback,
        dataset.accounts,
        dataset.releases,
        config=replace(Config(), product_names=("Tempo",)),
    )
    text, source = write_brief(result, theme_id, use_llm=use_llm)
    typer.echo(text)
    typer.secho(f"\n(source: {source})", dim=True)


if __name__ == "__main__":
    app()
