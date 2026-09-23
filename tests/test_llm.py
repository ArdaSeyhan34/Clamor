"""The Claude layer is tested with a fake client: no network, no API key."""

import json
from types import SimpleNamespace

import pytest

from clamor import llm
from clamor.briefs import template_brief, theme_evidence, write_brief
from clamor.pipeline import review_with_claude


class FakeMessages:
    def __init__(self, reply, stop_reason="end_turn"):
        self.reply, self.stop_reason, self.calls = reply, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.reply(kwargs) if callable(self.reply) else self.reply
        return SimpleNamespace(
            stop_reason=self.stop_reason, content=[SimpleNamespace(type="text", text=text)]
        )


def fake_client(reply, stop_reason="end_turn"):
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(reply, stop_reason)))


def test_review_names_classifies_and_merges(analysis):
    ids = analysis.themes["theme_id"].tolist()

    def reply(kwargs):
        payload = json.loads(kwargs["messages"][0]["content"].split("\n\n", 1)[1])
        assert {p["theme_id"] for p in payload} == set(ids)
        return json.dumps(
            {
                "themes": [
                    {
                        "theme_id": t,
                        "name": f"Theme {t}",
                        "kind": "bug",
                        "summary": "s",
                        "duplicate_of": ids[0] if t == ids[1] else "",
                    }
                    for t in ids
                ]
            }
        )

    client = fake_client(reply)
    reviewed = review_with_claude(analysis, llm.ClaudeAnalyst(client=client))
    kwargs = client.beta.messages.calls[0]
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["fallbacks"] == "default"
    assert ids[1] not in set(reviewed.themes["theme_id"])  # merged into ids[0]
    assert (reviewed.themes["name"].str.startswith("Theme ")).all()


def test_invalid_duplicate_targets_are_ignored(analysis):
    ids = analysis.themes["theme_id"].tolist()[:2]
    reply = json.dumps(
        {
            "themes": [
                {
                    "theme_id": ids[0],
                    "name": "A",
                    "kind": "ux",
                    "summary": "",
                    "duplicate_of": "T99",
                },
                {
                    "theme_id": ids[1],
                    "name": "B",
                    "kind": "ux",
                    "summary": "",
                    "duplicate_of": ids[1],
                },
                {
                    "theme_id": "T77",
                    "name": "ghost",
                    "kind": "ux",
                    "summary": "",
                    "duplicate_of": "",
                },
            ]
        }
    )
    review = llm.ClaudeAnalyst(client=fake_client(reply)).review_themes(analysis.themes.head(2))
    assert set(review["theme_id"]) == set(ids)
    assert llm.merge_suggestions(review) == {}


def test_refusal_raises_and_brief_falls_back_to_template(analysis):
    tid = analysis.roadmap["theme_id"].iloc[0]
    analyst = llm.ClaudeAnalyst(client=fake_client("", stop_reason="refusal"))
    with pytest.raises(llm.LLMUnavailable):
        analyst.write_brief({"x": 1})
    text, source = write_brief(analysis, tid, use_llm=True, analyst=analyst)
    assert source == "template"
    assert text.startswith("## ")


def test_brief_uses_claude_when_available(analysis):
    tid = analysis.roadmap["theme_id"].iloc[0]
    client = fake_client("## Brief\nBody")
    text, source = write_brief(analysis, tid, analyst=llm.ClaudeAnalyst(client=client))
    assert source == "claude" and text == "## Brief\nBody"
    evidence = json.loads(
        client.beta.messages.calls[0]["messages"][0]["content"].split("\n\n", 1)[1]
    )
    assert evidence["theme_id"] == tid and evidence["quotes"]


def test_template_brief_contains_evidence(analysis):
    tid = analysis.roadmap["theme_id"].iloc[0]
    ev = theme_evidence(analysis, tid)
    text = template_brief(ev)
    for section in ("### Problem", "### Who is affected", "### Evidence", "### Why now"):
        assert section in text
    assert ev["quotes"][0] in text
