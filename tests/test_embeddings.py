"""Tests for the embedding backend, exercising the TF-IDF fallback path only."""

import numpy as np
import pytest

from rag_agent.config import EmbeddingConfig
from rag_agent.embeddings import EmbeddingBackend

CORPUS = [
    "python programming language for data science",
    "machine learning models require training data",
    "cats and dogs are common household pets",
]


def make_backend(**overrides):
    defaults = {"use_tfidf_fallback": True, "vector_dimension": 8}
    defaults.update(overrides)
    return EmbeddingBackend(config=EmbeddingConfig(**defaults))


def test_embed_empty_input_returns_empty_matrix():
    backend = make_backend()

    result = backend.embed([])

    assert result.shape == (0, 8)
    assert result.dtype == np.float32


def test_tfidf_fallback_returns_normalized_vectors(force_tfidf):
    backend = make_backend()
    backend.fit(CORPUS)

    vectors = backend.embed(CORPUS)

    assert vectors.shape[0] == len(CORPUS)
    assert vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_tfidf_auto_fits_when_not_fitted(force_tfidf):
    backend = make_backend()

    vectors = backend.embed(CORPUS)

    assert vectors.shape[0] == len(CORPUS)
    assert np.any(vectors != 0)


def test_embed_single_returns_one_dimensional_vector(force_tfidf):
    backend = make_backend()
    backend.fit(CORPUS)

    vector = backend.embed_single("python programming for data science")

    assert vector.ndim == 1
    assert np.any(vector != 0)


def test_similar_texts_have_higher_cosine_similarity(force_tfidf):
    backend = make_backend()
    backend.fit(CORPUS)

    query = backend.embed_single("python data science programming")
    related = backend.embed_single(CORPUS[0])
    unrelated = backend.embed_single(CORPUS[2])

    assert float(np.dot(query, related)) > float(np.dot(query, unrelated))


def test_embed_raises_when_no_backend_available(force_tfidf):
    backend = make_backend(use_tfidf_fallback=False)

    with pytest.raises(RuntimeError, match="No embedding backend available"):
        backend.embed(["some text"])


def test_fit_is_noop_when_fallback_disabled(force_tfidf):
    backend = make_backend(use_tfidf_fallback=False)

    backend.fit(CORPUS)

    assert backend._vectorizer is None


def test_fit_is_noop_on_empty_corpus(force_tfidf):
    backend = make_backend()

    backend.fit([])

    assert backend._vectorizer is None
