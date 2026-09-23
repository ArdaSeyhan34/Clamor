"""Turkish language support, real-world export formats and personal-data masking."""

import pandas as pd
import pytest

from clamor import synth
from clamor.config import Config
from clamor.evaluate import evaluate_all
from clamor.io import combine_feedback, load_feedback, load_releases
from clamor.lang import TURKISH, fold, tr_lower
from clamor.pipeline import analyze
from clamor.privacy import redact, tckn_valid
from clamor.sentiment import score_text
from clamor.text import split_segments


def test_turkish_case_folding():
    assert tr_lower("İSTANBUL ILIK") == "istanbul ılık"
    assert fold("Çalışmıyor, ÖDEME geçmiyor") == "calismiyor, odeme gecmiyor"


@pytest.mark.parametrize(
    "text, sign",
    [
        ("QR ile ödeme çalışmıyor, rezalet", -1),
        ("qr ile odeme calismiyor rezalet", -1),  # typed without Turkish characters
        ("Çok pratik, harika bir uygulama", 1),
        ("Güzel değil", -1),  # negation after the word
        ("Sorunsuz çalışıyor", 1),  # "sorunsuz" must not be read as "sorun"
    ],
)
def test_turkish_sentiment(text, sign):
    assert score_text(text, TURKISH) * sign > 0.2


def test_greetings_do_not_count_as_praise():
    assert score_text("Merhaba, bakiyem yüklenmedi. İyi çalışmalar", TURKISH) < -0.2


def test_turkish_segmentation():
    text = ("Merhaba, Her gün kullanıyorum. QR ile ödeme çalışmıyor. Ayrıca karanlık mod "
            "eklenirse iyi olur. Saygılarımla, Ayşe")  # fmt: skip
    assert split_segments(text, ["Lezzo"], TURKISH) == [
        "Her gün kullanıyorum.",
        "QR ile ödeme çalışmıyor.",
        "Karanlık mod eklenirse iyi olur.",
    ]
    assert split_segments("Lezzo'da QR okumuyor", ["Lezzo"], TURKISH) == ["QR okumuyor"]


def test_personal_data_is_masked():
    text = ("Kart 4111 1111 1111 1111, IBAN TR33 0006 1005 1978 6457 8413 26, "
            "tel 0532 123 45 67, e-posta ali@example.com, TCKN 10000000146. "
            "Sipariş 1234567890123, tutar 150 TL.")  # fmt: skip
    masked = redact(text)
    for tag in ("[card]", "[iban]", "[phone]", "[email]", "[national-id]"):
        assert tag in masked
    assert "1234567890123" in masked and "150 TL" in masked  # not personal data
    assert tckn_valid("10000000146") and not tckn_valid("12345678901")


def test_play_console_export(tmp_path):
    pd.DataFrame({
        "Review Submit Date and Time": ["2026-03-01T10:00:00Z", "2026-03-02T11:00:00Z"],
        "Star Rating": [1, 5],
        "Review Title": ["", "Süper"],
        "Review Text": ["QR çalışmıyor", "Çok pratik"],
    }).to_csv(tmp_path / "reviews.csv", index=False, encoding="utf-16")  # fmt: skip
    fb = load_feedback(tmp_path / "reviews.csv")
    assert fb["text"].tolist() == ["QR çalışmıyor", "Süper. Çok pratik"]
    assert fb["rating"].tolist() == [1, 5]


def test_turkish_excel_export(tmp_path):
    pd.DataFrame({
        "Talep No": [7],
        "Oluşturma Tarihi": ["12.03.2026 10:15"],
        "Konu": ["Giriş"],
        "Açıklama": ["SMS kodu gelmiyor"],
    }).to_csv(tmp_path / "tickets.csv", index=False, sep=";", encoding="cp1254")  # fmt: skip
    fb = load_feedback(tmp_path / "tickets.csv")
    assert fb.loc[0, "created_at"] == pd.Timestamp("2026-03-12 10:15")  # day first
    assert fb.loc[0, "text"] == "Giriş. SMS kodu gelmiyor"
    rel = pd.DataFrame({"Yayın Tarihi": ["10.03.2026"], "Başlık": ["v5.1"]})
    rel.to_csv(tmp_path / "rel.csv", index=False, sep=";")
    assert load_releases(tmp_path / "rel.csv").loc[0, "date"] == pd.Timestamp("2026-03-10")


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["12.03.2026 10:15"], "2026-03-12 10:15"),
        (["14/03/2026", "02/03/2026"], "2026-03-14"),
        (["03/14/2026", "03/02/2026"], "2026-03-14"),
        (["2026-03-14T08:00:00Z"], "2026-03-14 08:00"),
    ],
)
def test_date_order_is_detected(values, expected):
    fb = load_feedback(pd.DataFrame({"text": ["x"] * len(values), "date": values}))
    assert fb["created_at"].max() == pd.Timestamp(expected)


def test_combine_exports_keeps_channels_and_unique_ids(tmp_path):
    pd.DataFrame({
        "Review Submit Date and Time": ["2026-03-01T10:00:00Z"],
        "Review Text": ["QR çalışmıyor"],
    }).to_csv(tmp_path / "reviews.csv", index=False, encoding="utf-16")  # fmt: skip
    pd.DataFrame({
        "Oluşturma Tarihi": ["02.03.2026 09:00", "03.03.2026 09:00"],
        "Açıklama": ["SMS kodu gelmiyor", "Bakiye yüklenmedi"],
    }).to_csv(tmp_path / "tickets.csv", index=False, sep=";", encoding="cp1254")  # fmt: skip
    fb = combine_feedback([tmp_path / "reviews.csv", tmp_path / "tickets.csv"])
    assert fb["channel"].tolist() == ["reviews", "tickets", "tickets"]
    assert fb["feedback_id"].is_unique and fb["account_id"].is_unique
    assert fb["feedback_id"].iloc[0] == "reviews:FB-00001"


@pytest.fixture(scope="module")
def lezzo():
    data = synth.generate_scenario("lezzo", days=140)
    cfg = Config(language="tr", embedding="tfidf", product_names=("Lezzo",))
    return data, analyze(data.feedback, None, data.releases, config=cfg)


def test_lezzo_scenario_has_no_revenue(lezzo):
    data, result = lezzo
    assert data.accounts is None
    assert not result.has_revenue
    assert (result.roadmap["c_revenue"] == 0).all()


def test_lezzo_pipeline_finds_turkish_themes(lezzo):
    data, result = lezzo
    ev = evaluate_all(result, data)
    assert ev["themes"]["homogeneity"] > 0.8
    assert ev["releases"]["correct"].sum() >= 3
    # masked in the analysis, so never shown in quotes or sent to an API
    assert not result.feedback["text"].str.contains("0500 000").any()
    assert result.feedback["text"].str.contains(r"\[phone\]").any()
    # theme names are real sentences, spelled with Turkish characters where customers did
    assert result.themes["name"].str.contains("[çğıöşü]").mean() > 0.5
