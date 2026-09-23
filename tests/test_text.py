import pandas as pd

from clamor.text import segment_feedback, split_segments


def test_strips_greetings_signoffs_and_tags():
    text = "[Bug] Hi team, The Android app is so slow. Please fix this asap. - Priya"
    assert split_segments(text) == ["The Android app is so slow.", "Please fix this asap."]


def test_call_notes_keep_only_the_customer_quote():
    text = 'Call notes (Acme): the CTO said "We need SAML single sign-on." Renewal in 3 weeks.'
    assert split_segments(text) == ["We need SAML single sign-on."]


def test_product_name_is_removed_and_casing_restored():
    assert split_segments("feature request: please add Tempo dark mode", ["Tempo"]) == [
        "Please add dark mode"
    ]
    assert split_segments("iCloud sync fails.") == ["iCloud sync fails."]


def test_praise_that_starts_like_a_signoff_is_kept():
    assert split_segments("Best app ever! Thanks, Maria") == ["Best app ever!"]


def test_items_without_content_keep_their_text():
    seg = segment_feedback(pd.DataFrame({"text": ["Thanks!", "Sync broke. Also, add dark mode."]}))
    assert seg[seg["doc"] == 0]["text"].tolist() == ["Thanks!"]
    assert seg[seg["doc"] == 1]["text"].tolist() == ["Sync broke.", "Add dark mode."]
