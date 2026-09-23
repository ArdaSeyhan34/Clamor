"""Loading and validating input tables.

Only ``feedback`` is required. The schema is deliberately forgiving: common column names
from support tools and survey exports are mapped automatically.

feedback  : text (required), created_at (required), feedback_id, account_id, channel, rating
accounts  : account_id (required), mrr (required), plan, seats, company, ...
releases  : date (required), title (required), version, description
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ALIASES = {
    "text": ["text", "body", "comment", "message", "review", "content", "feedback", "verbatim"],
    "created_at": ["created_at", "date", "timestamp", "created", "submitted_at", "time"],
    "feedback_id": ["feedback_id", "id", "ticket_id", "review_id", "response_id"],
    "account_id": ["account_id", "customer_id", "company_id", "org_id", "user_id"],
    "channel": ["channel", "source", "origin"],
    "rating": ["rating", "score", "stars", "nps"],
    "mrr": ["mrr", "monthly_revenue", "revenue"],
    "date": ["date", "released_at", "release_date", "shipped_at"],
    "title": ["title", "name", "summary"],
}


class SchemaError(ValueError):
    pass


def _rename(df: pd.DataFrame, wanted: list[str]) -> pd.DataFrame:
    lower = {c.lower().strip(): c for c in df.columns}
    mapping = {}
    for canonical in wanted:
        if canonical in df.columns:
            continue
        for alias in ALIASES.get(canonical, []):
            if alias in lower and lower[alias] not in mapping:
                mapping[lower[alias]] = canonical
                break
    return df.rename(columns=mapping)


def _read(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    return pd.read_csv(path)


def load_feedback(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    df = _rename(_read(source), list(ALIASES)[:6])
    missing = {"text", "created_at"} - set(df.columns)
    if missing:
        raise SchemaError(
            f"feedback is missing required column(s) {sorted(missing)}; found {list(df.columns)}"
        )
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["created_at"] = df["created_at"].dt.tz_localize(None)
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
    df = _rename(_read(source), ["date", "title"])
    missing = {"date", "title"} - set(df.columns)
    if missing:
        raise SchemaError(f"releases is missing required column(s) {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "description" not in df:
        df["description"] = ""
    if "version" not in df:
        df["version"] = df["title"]
    df["description"] = df["description"].fillna("").astype(str)
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
