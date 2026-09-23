import numpy as np

from clamor.themes import OTHER, classify_kind, cluster_vectors, consolidate


def _blobs(sizes, dim=16, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.eye(dim)[: len(sizes)]
    X = np.vstack(
        [c + noise * rng.standard_normal((n, dim)) for c, n in zip(centers, sizes, strict=True)]
    )
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.repeat(np.arange(len(sizes)), sizes)
    return X, y


def test_finds_unbalanced_clusters_and_orders_by_size():
    X, y = _blobs([200, 60, 12])  # very unbalanced, like real feedback
    labels = cluster_vectors(X, distance_threshold=0.5, min_size=5)
    assert len(set(labels) - {OTHER}) == 3
    assert (labels == 0).sum() == 200 and (labels == 2).sum() == 12


def test_tiny_clusters_are_dissolved():
    X, _ = _blobs([100, 3])
    labels = cluster_vectors(X, distance_threshold=0.5, min_size=5, reassign_similarity=0.9)
    assert set(labels) == {0, OTHER}
    assert (labels == OTHER).sum() == 3


def test_consolidate_requires_meaning_and_vocabulary_to_agree():
    X, _ = _blobs([20, 20, 20])
    X[20:40] = X[:20]  # clusters 0 and 1 mean the same thing...
    labels = np.repeat([0, 1, 2], 20)
    texts = ["sso saml login"] * 20 + ["saml sso okta"] * 20 + ["sso saml login"] * 20
    merged = consolidate(labels, texts, X, min_semantic=0.9, min_lexical=0.3)
    assert len(set(merged)) == 2  # 0+1 merged; 2 shares words with 0 but not meaning
    assert merged[0] == merged[25] != merged[45]


def test_kind_heuristics():
    assert classify_kind(["the app crashes on startup", "sync is broken"], -0.6) == "bug"
    assert classify_kind(["please add a dark mode", "would love an export option"], 0.1) == (
        "feature_request"
    )
    assert classify_kind(["the price increase is too expensive"], -0.5) == "pricing"
    assert classify_kind(["love it, super intuitive"], 0.8) == "praise"
