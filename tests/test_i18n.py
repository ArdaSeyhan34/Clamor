"""Output language: every user-facing string has a Turkish translation, English is unchanged."""

import ast
import string
from pathlib import Path

import pytest

from clamor import briefs, charts, i18n, report
from clamor.briefs import template_brief, theme_evidence, write_brief
from clamor.config import PRESETS, Config
from clamor.insights import headline_insights
from clamor.pipeline import rerun
from clamor.report import roadmap_table, to_html, to_markdown

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [*sorted((ROOT / "clamor").glob("*.py")), ROOT / "app" / "streamlit_app.py"]


def translated_literals() -> set[str]:
    """First arguments of t(...) / _(...) / i18n.t(...) that are string literals."""
    found = set()
    for path in SOURCES:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            f = node.func
            name = getattr(f, "id", None) or getattr(f, "attr", "")
            arg = node.args[0]
            if name in {"t", "_"} and isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                found.add(arg.value)
    return found


def looked_up_strings() -> set[str]:
    """Strings translated through a variable rather than a literal."""
    app = ast.parse((ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8"))
    demos = next(
        n.value
        for n in ast.walk(app)
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "DEMOS"
    )
    demo_texts = {
        c.value
        for entry in demos.values
        for c in entry.elts
        if isinstance(c, ast.Constant) and isinstance(c.value, str) and " " in c.value
    }
    return {
        *report.KIND.values(),
        *report.STATUS_ICON.values(),
        *report.HTML_TEXT.values(),
        *briefs.KIND_LABEL.values(),
        *(o for opts in briefs.OPTIONS_BY_KIND.values() for o in opts),
        *charts.VERDICT_STYLE,
        *charts.COMPONENTS,
        *(c.lower() for c in charts.COMPONENTS),
        *PRESETS,
        *report.STATUS_ICON,
        "Not enough data",
        "No matching theme",
        "It appeared recently and was absent before.",
        "It is **growing fast**",
        "It is growing",
        "It is **declining**",
        "Its share of feedback is stable",
        "There is too little recent data to call a trend",
        "Trend unknown",
        "fewer distinct users",
        "milder sentiment",
        "no significant growth",
        "highest-revenue",
        "most frequent",
        "accounts",
        "users",
        "Accounts",
        "Users",
        "accounts (plural)",
        "users (plural)",
        *demo_texts,
    }


def fields(text: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


def test_every_string_has_a_turkish_translation():
    missing = sorted((translated_literals() | looked_up_strings()) - set(i18n.TR))
    assert not missing, "add these to clamor.i18n.TR:\n" + "\n".join(missing)


def test_translations_use_only_known_placeholders():
    wrong = [k for k, v in i18n.TR.items() if not fields(v) <= fields(k)]
    assert not wrong, wrong


def test_turkish_number_formats():
    assert i18n.number(2738, "tr") == "2.738"
    assert i18n.number(3.41, "tr", 2) == "3,41"
    assert i18n.pct(0.917, "tr", 1) == "%91,7"
    assert i18n.pct(2.3, "tr", signed=True) == "+%230"
    assert i18n.pct(-0.29, "en", signed=True) == "-29%"
    assert i18n.q_value(0.0004, "tr") == "< 0,001"
    assert i18n.date("2026-03-23", "tr") == "23 Mart 2026"
    assert i18n.date("2026-03-23", "en") == "Mar 23, 2026"
    assert i18n.money(34_500, "tr", compact=True) == "$34,5 bin"
    assert i18n.money(1_600_000, "en", compact=True) == "$1.6M"


def test_unknown_strings_fall_back_to_english():
    assert i18n.t("Something new", "tr") == "Something new"
    assert i18n.t("Top priority: {name}", "de", name="X") == "Top priority: X"


def test_report_language_follows_the_data_unless_set():
    assert Config().output_language == "en"
    assert Config(language="tr").output_language == "tr"
    assert Config(language="tr", report_language="en").output_language == "en"
    assert Config(report_language="xx").output_language == "en"


@pytest.fixture(scope="module")
def turkish(analysis):
    return rerun(analysis, config=Config(product_names=("Tempo",), report_language="tr"))


def test_turkish_report(turkish):
    md = to_markdown(turkish)
    assert md.startswith("# Clamor raporu")
    for heading in ("## Öne çıkanlar", "## Önceliklendirilmiş yol haritası", "## Sürüm radarı"):
        assert heading in md
    assert "Key insights" not in md and "Release radar" not in md
    html = to_html(turkish)
    assert '<html lang="tr">' in html and "<h1>Clamor raporu</h1>" in html
    assert list(roadmap_table(turkish).columns[:3]) == ["Sıra", "Tema", "Tür"]


def test_turkish_insights_and_brief(turkish):
    insights = headline_insights(turkish)
    assert insights[0].title.startswith("En yüksek öncelik: ")
    tid = turkish.roadmap["theme_id"].iloc[0]
    text, source = write_brief(turkish, tid, use_llm=False)
    assert source == "template"
    for heading in ("### Sorun", "### Kimler etkileniyor", "### Açık sorular"):
        assert heading in text
    # the same evidence renders in either language
    ev = theme_evidence(turkish, tid)
    assert "### Problem" in template_brief(ev, "en")


def test_english_output_is_the_default(analysis):
    assert to_markdown(analysis).startswith("# Clamor report")
    assert headline_insights(analysis, lang="en")[0].title.startswith("Top priority: ")
