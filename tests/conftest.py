"""Shared fixtures. Tests use the TF-IDF backend so they run offline and fast in CI."""

from __future__ import annotations

import pytest

from clamor import synth
from clamor.config import Config
from clamor.pipeline import analyze


@pytest.fixture(scope="session")
def dataset() -> synth.SyntheticDataset:
    return synth.generate(seed=3, n_accounts=250, days=140)


@pytest.fixture(scope="session")
def config() -> Config:
    return Config(embedding="tfidf", product_names=("Tempo",))


@pytest.fixture(scope="session")
def analysis(dataset, config):
    return analyze(dataset.feedback, dataset.accounts, dataset.releases, config=config)
