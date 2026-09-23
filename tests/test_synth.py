from clamor import synth


def test_generation_is_deterministic():
    a = synth.generate(seed=11, n_accounts=50, days=21)
    b = synth.generate(seed=11, n_accounts=50, days=21)
    assert a.feedback.equals(b.feedback)
    assert a.accounts.equals(b.accounts)


def test_tables_are_consistent(dataset):
    fb, truth = dataset.feedback, dataset.ground_truth
    assert set(fb.columns) >= {"feedback_id", "created_at", "channel", "account_id", "text"}
    assert fb["feedback_id"].is_unique
    assert set(truth["feedback_id"]) == set(fb["feedback_id"])
    assert set(fb["account_id"]) <= set(dataset.accounts["account_id"])
    assert fb["created_at"].is_monotonic_increasing
    # the free plan pays nothing, enterprise pays the most per account
    mrr = dataset.accounts.groupby("plan")["mrr"].mean()
    assert mrr["free"] == 0 and mrr["enterprise"] > mrr["business"] > mrr["pro"]


def test_planted_regression_is_visible(dataset):
    truth = dataset.feedback.merge(dataset.ground_truth)
    sync = truth[truth["true_theme"] == "calendar_sync"].set_index("created_at")
    weekly = sync.resample("W").size()
    # the v3.2 sync regression (mid-March) should dwarf the January baseline
    assert weekly["2026-03-16":"2026-04-12"].mean() > 3 * weekly[:"2026-02-28"].mean()


def test_save_and_load_roundtrip(tmp_path):
    data = synth.generate(seed=1, n_accounts=30, days=14)
    data.save(tmp_path)
    loaded = synth.SyntheticDataset.load(tmp_path)
    assert len(loaded.feedback) == len(data.feedback)
    assert list(loaded.releases.columns) == ["date", "version", "title", "description"]
