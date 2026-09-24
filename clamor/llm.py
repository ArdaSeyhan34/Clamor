"""Optional Claude layer: an analyst that reviews the themes and drafts opportunity briefs.

Everything in Clamor works without an API key; this module adds judgment where statistics
run out:

* **Theme review.** Clustering finds groups but cannot *name* them the way a PM would, and
  it cannot tell that "per-seat pricing" and "price increase" belong on the same slide.
  Claude receives each theme's keywords, statistics and representative quotes, and returns
  a name, a type, a one-line summary and (optionally) a duplicate it should be merged into.
* **Opportunity briefs.** For a selected theme, Claude drafts a one-page brief from the
  evidence only: problem, who is affected, why now, options, success metrics.

Design choices:

* Only aggregates and ~8 representative quotes per theme are sent, never the full corpus:
  a few thousand tokens per run instead of hundreds of thousands, and less customer text
  leaving your systems.
* Structured outputs (JSON schema) make the review machine-readable without parsing prose.
* Any failure (no key, network, refusal) falls back to the deterministic output, so a
  demo never breaks because of an API hiccup.

Set ``ANTHROPIC_API_KEY`` to enable it; ``CLAMOR_MODEL`` overrides the model.
"""

from __future__ import annotations

import json
import logging
import os

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("CLAMOR_MODEL", "claude-opus-5")
KINDS = ["bug", "feature_request", "ux", "pricing", "praise", "other"]

REVIEW_SYSTEM = """You are a senior product manager reviewing the output of a customer \
feedback clustering tool for a software product. Each theme comes with keywords, statistics \
and verbatim customer quotes. Your job is to make the list useful to a product team: give \
each theme a short, specific name in plain language (3 to 6 words, the way you would write \
it on a roadmap), classify it, summarize the underlying user need in one sentence, and flag \
themes that are really the same problem as another theme in the list. Only mark a duplicate \
when a product team would clearly handle both with one initiative; different sub-problems \
with different fixes are not duplicates. Base everything on the evidence provided."""

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme_id": {"type": "string"},
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": KINDS},
                    "summary": {"type": "string"},
                    "duplicate_of": {
                        "type": "string",
                        "description": "theme_id this theme should be merged into, or ''",
                    },
                },
                "required": ["theme_id", "name", "kind", "summary", "duplicate_of"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["themes"],
    "additionalProperties": False,
}

BRIEF_SYSTEM = """You are a senior product manager writing a one-page opportunity brief \
for your team, based strictly on customer feedback evidence prepared by an analytics tool. \
Use only the numbers and quotes provided; never invent metrics, customers or dates. Where \
the evidence is thin, say so and turn it into an open question. Write in concise Markdown \
with these sections: Problem, Who is affected, Evidence, Why now, Options to explore, \
How we will know it worked, Open questions. Keep it under 450 words."""

LANGUAGE_NOTE = {  # appended to the system prompt when the output is not in English
    "tr": {
        "review": "Write every name and summary in Turkish, in the plain words a Turkish "
        "product team would use.",
        "brief": "Write the whole brief in Turkish. Use these section headings: Sorun, Kimler "
        "etkileniyor, Kanıtlar, Neden şimdi, Değerlendirilecek seçenekler, İşe yaradığını "
        "nasıl anlayacağız, Açık sorular. Quote customers verbatim.",
    }
}


def _system(prompt: str, task: str, language: str) -> str:
    note = LANGUAGE_NOTE.get(language, {}).get(task)
    return f"{prompt}\n\n{note}" if note else prompt


class LLMUnavailable(RuntimeError):
    pass


def available() -> bool:
    """True if the SDK is installed and a credential is configured."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


class ClaudeAnalyst:
    def __init__(self, model: str = DEFAULT_MODEL, client=None, effort: str = "medium"):
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise LLMUnavailable("pip install 'clamor[llm]' to enable Claude") from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort

    def _call(self, system: str, prompt: str, schema: dict | None = None) -> str:
        import anthropic

        output_config: dict = {"effort": self.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config=output_config,
                # if a safety classifier declines, retry server-side on the recommended model
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as exc:
            raise LLMUnavailable("Anthropic API key missing or invalid") from exc
        except anthropic.RateLimitError as exc:
            raise LLMUnavailable("rate limited by the Anthropic API, try again later") from exc
        except anthropic.APIStatusError as exc:
            raise LLMUnavailable(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMUnavailable("could not reach the Anthropic API") from exc
        if response.stop_reason == "refusal":
            raise LLMUnavailable("the model declined this request")
        if response.stop_reason == "max_tokens":
            raise LLMUnavailable("response was truncated")
        return "".join(b.text for b in response.content if b.type == "text")

    def review_themes(self, themes: pd.DataFrame, language: str = "en") -> pd.DataFrame:
        """Name, classify and de-duplicate themes. Returns one row per theme_id.

        ``language`` is the language of the names and summaries ("en" or "tr").
        """
        payload = []
        for _, t in themes.iterrows():
            payload.append(
                {
                    "theme_id": t["theme_id"],
                    "keywords": list(t["keywords"])[:6],
                    "mentions": int(t.get("mentions_all_time", t.get("mentions", 0))),
                    "mean_sentiment": round(float(t.get("mean_sentiment", 0) or 0), 2),
                    "top_plans": t.get("plan_mix") if isinstance(t.get("plan_mix"), dict) else {},
                    "quotes": list(t["examples"])[:8],
                }
            )
        prompt = (
            "Review these feedback themes. Return exactly one entry per theme_id.\n\n"
            + json.dumps(payload, indent=1, default=str)
        )
        data = json.loads(
            self._call(_system(REVIEW_SYSTEM, "review", language), prompt, REVIEW_SCHEMA)
        )
        out = pd.DataFrame(data["themes"])
        known = set(themes["theme_id"])
        out = out[out["theme_id"].isin(known)].drop_duplicates("theme_id")
        out["duplicate_of"] = out["duplicate_of"].where(out["duplicate_of"].isin(known), "")
        out.loc[out["duplicate_of"] == out["theme_id"], "duplicate_of"] = ""
        return out

    def write_brief(self, evidence: dict, language: str = "en") -> str:
        prompt = "Write the opportunity brief for this theme. Evidence (JSON):\n\n" + json.dumps(
            evidence, indent=1, default=str
        )
        return self._call(_system(BRIEF_SYSTEM, "brief", language), prompt).strip()


def apply_review(themes: pd.DataFrame, review: pd.DataFrame) -> pd.DataFrame:
    """Overlay names, kinds and summaries from a review onto a theme table."""
    out = themes.copy()
    r = review.set_index("theme_id")
    has = out["theme_id"].isin(r.index)
    out.loc[has, "name"] = out.loc[has, "theme_id"].map(r["name"])
    out.loc[has, "kind"] = out.loc[has, "theme_id"].map(r["kind"])
    out.loc[has, "summary"] = out.loc[has, "theme_id"].map(r["summary"])
    return out


def merge_suggestions(review: pd.DataFrame) -> dict[str, str]:
    dup = review[review["duplicate_of"].astype(bool)]
    return dict(zip(dup["theme_id"], dup["duplicate_of"], strict=True))
