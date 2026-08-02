"""Embedding utilities for the RAG Agentic System."""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from .config import EmbeddingConfig

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer  # pragma: no cover
    from sklearn.feature_extraction.text import TfidfVectorizer  # pragma: no cover

logger = logging.getLogger(__name__)
VectorArray = NDArray[np.float32]


@dataclass(slots=True)
class EmbeddingBackend:
    """Wrapper around embedding backends with optional TF-IDF fallback."""

    config: EmbeddingConfig
    _st_model: SentenceTransformer | None = field(default=None, init=False, repr=False)
    _vectorizer: TfidfVectorizer | None = field(default=None, init=False, repr=False)

    @property
    def has_sentence_transformer(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401

            return True
        except ImportError:
            return False

    def _load_sentence_transformer(self) -> None:
        if self._st_model is not None:
            return

        if not self.has_sentence_transformer:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with `pip install sentence-transformers` or enable TF-IDF fallback."
            )

        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self.config.model_name)
        self._st_model = SentenceTransformer(self.config.model_name, device=self.config.device)

    def _ensure_tfidf(self) -> None:
        if self._vectorizer is not None:
            return

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required for TF-IDF fallback. "
                "Install with `pip install scikit-learn`."
            ) from exc

        logger.warning("Using TF-IDF fallback embedding backend.")
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=2048,
            ngram_range=(1, 2),
        )

    def fit(self, corpus: Iterable[str]) -> None:
        """Fit the fallback vectorizer on a corpus."""
        if not self.config.use_tfidf_fallback:
            return

        corpus_list = list(corpus)
        if not corpus_list:
            return

        self._ensure_tfidf()
        assert self._vectorizer is not None
        logger.info("Fitting TF-IDF vectorizer on corpus of %d documents.", len(corpus_list))
        self._vectorizer.fit(corpus_list)

    def embed(self, texts: Sequence[str]) -> VectorArray:
        """Embed a list of texts."""
        if not texts:
            return np.zeros((0, self.config.vector_dimension), dtype=np.float32)

        if self.has_sentence_transformer:
            self._load_sentence_transformer()
            assert self._st_model is not None
            embeddings = self._st_model.encode(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return embeddings.astype(np.float32)

        if not self.config.use_tfidf_fallback:
            raise RuntimeError(
                "No embedding backend available (sentence-transformers missing "
                "and TF-IDF fallback disabled)."
            )

        self._ensure_tfidf()
        vectorizer = self._vectorizer
        if vectorizer is None:
            raise RuntimeError("TF-IDF vectorizer failed to initialize.")

        if not hasattr(vectorizer, "vocabulary_"):
            logger.info("TF-IDF vectorizer not fitted. Fitting on provided texts (%d).", len(texts))
            try:
                vectorizer.fit(list(texts))
            except ValueError:
                # e.g. texts contain only stop words -> empty vocabulary.
                logger.warning("Could not fit TF-IDF on provided texts; returning zero vectors.")
                return np.zeros((len(texts), self.config.vector_dimension), dtype=np.float32)

        matrix = vectorizer.transform(list(texts))
        dense_matrix = cast(Any, matrix).toarray()
        embeddings = np.asarray(dense_matrix, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def embed_single(self, text: str) -> VectorArray:
        vectors = self.embed([text])
        if len(vectors):
            return vectors[0]
        return np.zeros((self.config.vector_dimension,), dtype=np.float32)


__all__ = ["EmbeddingBackend", "VectorArray"]