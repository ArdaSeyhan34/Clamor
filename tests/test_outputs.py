from typer.testing import CliRunner

from clamor.cli import app
from clamor.evaluate import evaluate_all
from clamor.insights import headline_insights
from clamor.report import to_html, to_markdown, write_report


def test_evaluation_runs_and_is_sane(analysis, dataset):
    ev = evaluate_all(analysis, dataset)
    assert 0 < ev["themes"]["adjusted_rand"] <= 1
    assert 0.5 < ev["sentiment"]["sign_accuracy"] <= 1
    assert set(ev["alerts"]) == {"clamor", "naive_2x"}
    assert len(ev["releases"]) == len(analysis.releases)


def test_reports_are_written(analysis, dataset, tmp_path):
    paths = write_report(
        analysis, tmp_path, evaluation=evaluate_all(analysis, dataset), n_briefs=2, use_llm=False
    )
    md = paths["markdown"].read_text()
    assert "## Prioritized roadmap" in md and "## Release radar" in md
    assert "Adjusted Rand index" in md
    assert (tmp_path / "briefs").is_dir() and len(list((tmp_path / "briefs").iterdir())) == 2
    assert "plotly" in paths["html"].read_text().lower()


def test_insights_and_html_render(analysis):
    insights = headline_insights(analysis)
    assert insights and insights[0].kind == "priority"
    assert "<h1>Clamor report</h1>" in to_html(analysis)
    assert to_markdown(analysis).startswith("# Clamor report")


def test_cli_generate(tmp_path):
    result = CliRunner().invoke(app, ["generate", "--out", str(tmp_path), "--days", "10"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "feedback.csv").exists()


def test_cli_analyze_tfidf(tmp_path, dataset):
    dataset.save(tmp_path / "data")
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            str(tmp_path / "data" / "feedback.csv"),
            "--accounts",
            str(tmp_path / "data" / "accounts.csv"),
            "--releases",
            str(tmp_path / "data" / "releases.csv"),
            "--out",
            str(tmp_path / "out"),
            "--backend",
            "tfidf",
            "--no-llm",
            "--product-name",
            "Tempo",
            "--briefs",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Top of the roadmap" in result.output
    assert (tmp_path / "out" / "report.html").exists()
