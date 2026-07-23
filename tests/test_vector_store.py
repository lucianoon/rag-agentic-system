"""Tests for the in-memory vector store with cosine similarity search."""

import numpy as np

from rag_agent.config import VectorStoreConfig
from rag_agent.types import Document
from rag_agent.vector_store import VectorStore


def make_store(top_k=5, dimension=3):
    return VectorStore(config=VectorStoreConfig(embedding_dimension=dimension, top_k=top_k))


def make_doc(idx):
    return Document(id=f"doc-{idx}", content=f"content {idx}")


def vec(*values):
    return np.array(values, dtype=np.float32)


def test_add_stores_documents_and_embeddings():
    store = make_store()
    store.add([(make_doc(0), vec(1, 0, 0)), (make_doc(1), vec(0, 1, 0))])

    assert set(store.documents) == {"doc-0", "doc-1"}
    assert set(store.embeddings) == {"doc-0", "doc-1"}
    assert store.embeddings["doc-0"].dtype == np.float32


def test_search_ranks_by_cosine_similarity():
    store = make_store()
    store.add(
        [
            (make_doc(0), vec(1, 0, 0)),
            (make_doc(1), vec(0, 1, 0)),
            (make_doc(2), vec(0.9, 0.1, 0)),
        ]
    )

    results = store.search(vec(1, 0, 0))

    assert [r.document.id for r in results] == ["doc-0", "doc-2", "doc-1"]
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert abs(scores[0] - 1.0) < 1e-6


def test_search_respects_explicit_top_k():
    store = make_store(top_k=5)
    store.add([(make_doc(i), vec(1, float(i), 0)) for i in range(4)])

    results = store.search(vec(1, 0, 0), top_k=2)

    assert len(results) == 2


def test_search_uses_config_top_k_by_default():
    store = make_store(top_k=2)
    store.add([(make_doc(i), vec(1, float(i), 0)) for i in range(4)])

    results = store.search(vec(1, 0, 0))

    assert len(results) == 2


def test_search_on_empty_store_returns_empty():
    store = make_store()

    assert store.search(vec(1, 0, 0)) == []


def test_search_with_empty_query_vector_returns_empty():
    store = make_store()
    store.add([(make_doc(0), vec(1, 0, 0))])

    assert store.search(np.zeros((0,), dtype=np.float32)) == []


def test_delete_removes_documents():
    store = make_store()
    store.add([(make_doc(0), vec(1, 0, 0)), (make_doc(1), vec(0, 1, 0))])

    store.delete(["doc-0", "missing-id"])

    assert set(store.documents) == {"doc-1"}
    assert set(store.embeddings) == {"doc-1"}


def test_clear_empties_the_store():
    store = make_store()
    store.add([(make_doc(0), vec(1, 0, 0))])

    store.clear()

    assert store.documents == {}
    assert store.embeddings == {}
