"""Loading and validating input tables.

Only ``feedback`` is required. The schema is deliberately forgiving so that raw exports
can be used as they are:

* column names from support tools, survey tools, the Google Play Console and Turkish
  exports are mapped automatically (``Review Text``, ``Description``, ``Yorum``,
  ``Tarih``, ``Puan``, ...);
* a subject or review title is prepended to the body;
* CSV encoding (UTF-8, UTF-16 as used by Play Console exports, Windows-1254) and
  delimiter (``,`` or ``;`` as used by Excel in Turkish locales) are detected;
* day-first dates (``12.03.2026``) are recognized.

feedback  : text (required), created_at (required), feedback_id, account_id, channel, rating
accounts  : account_id (required), mrr (required), plan, seats, company, ...
releases  : date (required), title (required), version, description
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ALIASES = {
    "text": [
        "text", "review text", "body", "comment", "message", "description", "review",
        "content", "feedback", "verbatim", "yorum", "metin", "mesaj", "açıklama", "aciklama",
        "şikayet", "sikayet", "şikayet metni", "talep açıklaması", "yorum metni",
    ],
    "created_at": [
        "created_at", "review submit date and time", "review last update date and time",
        "created", "created at", "created date", "creation date", "date", "timestamp",
        "submitted_at", "time", "tarih", "oluşturma tarihi", "olusturma tarihi",
        "kayıt tarihi", "kayit tarihi", "yorum tarihi",
    ],
    "feedback_id": [
        "feedback_id", "id", "ticket_id", "ticket id", "review_id", "response_id",
        "talep no", "talep numarası", "kayıt no", "kayit no",
    ],
    "account_id": [
        "account_id", "customer_id", "customer id", "company_id", "org_id", "user_id",
        "user id", "kullanıcı id", "kullanici_id", "müşteri no", "musteri no",
    ],
    "channel": ["channel", "source", "origin", "kanal", "kaynak"],
    "rating": [
        "rating", "star rating", "score", "stars", "nps", "puan", "yıldız", "yildiz",
        "memnuniyet puanı",
    ],
    "mrr": ["mrr", "monthly_revenue", "revenue", "gelir", "aylık gelir"],
    "date": ["date", "released_at", "release_date", "shipped_at", "tarih", "yayın tarihi"],
    "title": ["title", "name", "summary", "başlık", "baslik", "sürüm adı"],
    "description": ["description", "notes", "release notes", "açıklama", "aciklama",
                    "sürüm notları"],
}  # fmt: skip
SUBJECT_ALIASES = ["subject", "review title", "konu", "başlık", "baslik"]
FEEDBACK_FIELDS = ["text", "created_at", "feedback_id", "account_id", "channel", "rating"]


class SchemaError(ValueError):
    pass


def _key(name: str) -> str:
    return str(name).replace("İ", "i").replace("I", "i").lower().strip()


def _rename(df: pd.DataFrame, wanted: list[str]) -> pd.DataFrame:
    lower = {_key(c): c for c in df.columns}
    mapping = {}
    for canonical in wanted:
        if canonical in df.columns:
            continue
        for alias in ALIASES.get(canonical, []):
            if alias in lower and lower[alias] not in mapping:
                mapping[lower[alias]] = canonical
                break
    return df.rename(columns=mapping)


def _read_csv(path: Path) -> pd.DataFrame:
    head = path.read_bytes()[:4]
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ["utf-16"]  # Google Play Console review exports
    elif head.startswith(b"\xef\xbb\xbf"):
        encodings = ["utf-8-sig"]
    else:
        encodings = ["utf-8", "cp1254", "latin-1"]  # cp1254: Turkish Windows / Excel
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            with open(path, encoding=encoding) as fh:
                header = fh.readline()
            sep = max([",", ";", "\t", "|"], key=header.count)
            return pd.read_csv(path, sep=sep, encoding=encoding)
        except UnicodeError as exc:
            last_error = exc
    raise SchemaError(f"could not decode {path.name}: {last_error}")


def _read(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    return _read_csv(path)


_DAY_FIRST = re.compile(r"^\s*\d{1,2}[./]\d{1,2}[./]\d{2,4}")


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse timestamps; '12.03.2026' is read as 12 March, as in Turkish exports."""
    sample = values.dropna().astype(str).head(200)
    day_first = len(sample) > 0 and sample.str.match(_DAY_FIRST).mean() > 0.5
    parsed = pd.to_datetime(values, errors="coerce", utc=True, dayfirst=bool(day_first))
    return parsed.dt.tz_localize(None)


def load_feedback(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    raw = _read(source)
    df = _rename(raw, FEEDBACK_FIELDS)
    missing = {"text", "created_at"} - set(df.columns)
    if missing:
        raise SchemaError(
            f"feedback is missing required column(s) {sorted(missing)}; found {list(df.columns)}"
        )
    subject = next((c for c in df.columns if _key(c) in SUBJECT_ALIASES), None)
    if subject is not None and subject != "text":
        title = df[subject].fillna("").astype(str).str.strip()
        body = df["text"].fillna("").astype(str).str.strip()
        joined = [
            b if not t or b.lower().startswith(t.lower()) else (f"{t}. {b}" if b else t)
            for t, b in zip(title, body, strict=True)
        ]
        df["text"] = pd.Series(joined, index=df.index).replace("", None)
    df["created_at"] = _parse_dates(df["created_at"])
    bad = df["created_at"].isna() | df["text"].isna()
    df = df.loc[~bad].copy()
    df["text"] = df["text"].astype(str)
    if "feedback_id" not in df:
        df["feedback_id"] = [f"FB-{i + 1:05d}" for i in range(len(df))]
    if "account_id" not in df:
        df["account_id"] = df["feedback_id"]  # every item is its own "account"
    if "channel" not in df:
        df["channel"] = "unknown"
    if "rating" not in df:
        df["rating"] = float("nan")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["account_id"] = df["account_id"].astype(str)
    df = df.sort_values("created_at", kind="stable").reset_index(drop=True)
    if df.empty:
        raise SchemaError("feedback has no rows with both text and a valid date")
    return df


def load_accounts(source: str | Path | pd.DataFrame | None) -> pd.DataFrame | None:
    if source is None:
        return None
    df = _rename(_read(source), ["account_id", "mrr"])
    missing = {"account_id", "mrr"} - set(df.columns)
    if missing:
        raise SchemaError(f"accounts is missing required column(s) {sorted(missing)}")
    df["account_id"] = df["account_id"].astype(str)
    df["mrr"] = pd.to_numeric(df["mrr"], errors="coerce").fillna(0.0)
    if "plan" not in df:
        df["plan"] = "unknown"
    return df.drop_duplicates("account_id")


def load_releases(source: str | Path | pd.DataFrame | None) -> pd.DataFrame | None:
    if source is None:
        return None
    df = _rename(_read(source), ["date", "title", "description"])
    missing = {"date", "title"} - set(df.columns)
    if missing:
        raise SchemaError(f"releases is missing required column(s) {sorted(missing)}")
    df["date"] = _parse_dates(df["date"])
    if "description" not in df:
        df["description"] = ""
    if "version" not in df:
        df["version"] = df["title"]
    df["description"] = df["description"].fillna("").astype(str)
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
