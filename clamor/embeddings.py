"""Text embedding backends.

``minilm`` (default)
    `all-MiniLM-L6-v2 <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>`_
    run through ONNX Runtime. Semantic, fast on a laptop CPU (~3k sentences in a few
    seconds) and without the PyTorch dependency. The ~90 MB model is downloaded once to
    ``~/.cache/clamor`` (override with ``CLAMOR_MODEL_DIR``).

``multilingual``
    `paraphrase-multilingual-MiniLM-L12-v2
    <https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2>`_,
    the same idea for 50+ languages including Turkish (default for ``language="tr"``).
    Larger download (~470 MB), still CPU-friendly.

``hybrid``
    MiniLM (semantic) and TF-IDF (lexical) vectors concatenated with weights 0.8 / 0.2,
    the same idea as hybrid search. It was the default until theme consolidation was
    added; since then it performs on par with plain MiniLM on the benchmark (see
    ``clamor evaluate``), so the simpler model is the default. Worth trying on jargon-heavy
    feedback, where exact vocabulary matters more.

``tfidf``
    TF-IDF + truncated SVD (LSA). Needs no download; used in CI and as an automatic fallback
    when the model cannot be fetched. Purely lexical, so it will not know that "dark theme"
    and "night mode" are the same request. The evaluation quantifies how much that costs.

``st:<model-name>``
    Any `sentence-transformers` model, if that package is installed.
"""

from __future__ import annotations

import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

HF = "https://huggingface.co/"
ONNX_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
MODEL_FILES = {"model.onnx": "onnx/model.onnx", "tokenizer.json": "tokenizer.json"}
# kept for backwards compatibility
MINILM_REPO = HF + ONNX_MODELS["minilm"] + "/resolve/main/"
MINILM_FILES = MODEL_FILES


class Embedder(Protocol):
    name: str

    def fit(self, texts: list[str]) -> Embedder: ...

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def model_dir(name: str = "minilm") -> Path:
    """Local folder of an ONNX model: ~/.cache/clamor/<model> (override with env vars).

    ``CLAMOR_MODEL_HOME`` moves the whole cache; ``CLAMOR_MODEL_DIR`` points directly at
    the MiniLM folder (kept for backwards compatibility).
    """
    if name == "minilm" and os.environ.get("CLAMOR_MODEL_DIR"):
        return Path(os.environ["CLAMOR_MODEL_DIR"])
    home = Path(os.environ.get("CLAMOR_MODEL_HOME", Path.home() / ".cache" / "clamor"))
    return home / ONNX_MODELS[name].split("/")[-1]


def ensure_model(name: str = "minilm", directory: Path | None = None) -> Path:
    """Download an ONNX model and its tokenizer if they are not cached yet."""
    directory = directory or model_dir(name)
    directory.mkdir(parents=True, exist_ok=True)
    for local, remote in MODEL_FILES.items():
        target = directory / local
        if target.exists() and target.stat().st_size > 0:
            continue
        url = f"{HF}{ONNX_MODELS[name]}/resolve/main/{remote}"
        log.info("Downloading %s -> %s", url, target)
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.replace(target)
    return directory


def ensure_minilm(directory: Path | None = None) -> Path:
    return ensure_model("minilm", directory)


class OnnxSentenceEmbedder:
    """A sentence-transformers model with mean pooling, via ONNX Runtime (no PyTorch)."""

    def __init__(
        self,
        name: str = "minilm",
        directory: Path | None = None,
        batch_size: int = 64,
        max_length: int = 128,
    ):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.name = name
        path = ensure_model(name, directory)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        # BERT vocabularies pad with [PAD]=0, XLM-R (multilingual) with <pad>=1
        pad = next(
            (t for t in ("[PAD]", "<pad>") if self.tokenizer.token_to_id(t) is not None), "[PAD]"
        )
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id(pad) or 0, pad_token=pad)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(path / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.batch_size = batch_size
        self.dim: int | None = None

    def fit(self, texts: list[str]) -> OnnxSentenceEmbedder:
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 384), dtype=np.float32)
        order = np.argsort([len(t) for t in texts])  # similar lengths -> less padding
        out: np.ndarray | None = None
        for start in range(0, len(texts), self.batch_size):
            idx = order[start : start + self.batch_size]
            enc = self.tokenizer.encode_batch([texts[i] for i in idx])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feeds)[0]
            if out is None:
                self.dim = hidden.shape[-1]
                out = np.zeros((len(texts), self.dim), dtype=np.float32)
            m = mask[..., None].astype(np.float32)
            out[idx] = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        return _normalize(out)


class MiniLMEmbedder(OnnxSentenceEmbedder):
    """all-MiniLM-L6-v2 (English)."""

    def __init__(self, directory: Path | None = None, batch_size: int = 64, max_length: int = 128):
        super().__init__("minilm", directory, batch_size, max_length)


class TfidfEmbedder:
    """Lexical baseline: word + character n-gram TF-IDF compressed with LSA."""

    name = "tfidf"

    def __init__(self, n_components: int = 128, random_state: int = 0, lang=None):
        self.n_components = n_components
        self.random_state = random_state
        self.lang = lang
        self._pipeline = None

    def fit(self, texts: list[str]) -> TfidfEmbedder:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline, make_union

        if self.lang is None or self.lang.code == "en":
            word_opts: dict = {"stop_words": "english"}
            char_opts: dict = {}
        else:
            word_opts = {
                "stop_words": sorted(self.lang.stop_words),
                "preprocessor": self.lang.lower,
            }
            char_opts = {"preprocessor": self.lang.lower}
        union = make_union(
            TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_df=0.4, sublinear_tf=True, **word_opts
            ),
            TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, **char_opts
            ),
        )
        n_features = union.fit_transform(texts).shape[1]
        n_comp = max(2, min(self.n_components, n_features - 1, len(texts) - 1))
        self._pipeline = make_pipeline(
            union, TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        )
        self._pipeline.fit(texts)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("TfidfEmbedder must be fitted before encoding")
        if not texts:
            return np.zeros((0, self._pipeline[-1].n_components))
        return _normalize(self._pipeline.transform(texts))


class HybridEmbedder:
    """Weighted concatenation of a semantic and a lexical embedder (unit norm by design)."""

    name = "hybrid"

    def __init__(self, semantic: Embedder, lexical: Embedder, semantic_weight: float = 0.8):
        self.semantic = semantic
        self.lexical = lexical
        self.semantic_weight = semantic_weight

    def fit(self, texts: list[str]) -> HybridEmbedder:
        self.semantic.fit(texts)
        self.lexical.fit(texts)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        a = self.semantic_weight
        sem = self.semantic.encode(texts)
        lex = self.lexical.encode(texts)
        self._semantic_dims = sem.shape[1]
        return np.hstack([np.sqrt(a) * sem, np.sqrt(1.0 - a) * lex])

    def semantic_part(self, vectors: np.ndarray) -> np.ndarray:
        """Recover the (unit-norm) semantic block from hybrid vectors."""
        return vectors[:, : self._semantic_dims] / np.sqrt(self.semantic_weight)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.name = f"st:{model_name}"
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> SentenceTransformerEmbedder:
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


def get_embedder(name: str = "minilm", fallback: bool = True, lang=None) -> Embedder:
    """Build an embedder by name, falling back to TF-IDF if the model is unavailable.

    `lang` (a :class:`clamor.lang.LanguagePack`) sets stop words for the lexical backends
    and picks the semantic model inside ``hybrid``.
    """
    try:
        if name == "hybrid":
            semantic = "multilingual" if lang is not None and lang.code != "en" else "minilm"
            return HybridEmbedder(OnnxSentenceEmbedder(semantic), TfidfEmbedder(lang=lang))
        if name in ONNX_MODELS:
            return OnnxSentenceEmbedder(name)
        if name == "tfidf":
            return TfidfEmbedder(lang=lang)
        if name.startswith("st:"):
            return SentenceTransformerEmbedder(name[3:])
    except Exception as exc:  # network down, missing optional package, ...
        if not fallback:
            raise
        log.warning("Embedding backend %r unavailable (%s); falling back to TF-IDF.", name, exc)
        return TfidfEmbedder(lang=lang)
    raise ValueError(
        f"Unknown embedding backend: {name!r} "
        "(use minilm, multilingual, hybrid, tfidf or st:<model>)"
    )
