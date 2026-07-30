"""Tests for text chunking, document ingestion, and filesystem retrieval."""

import pytest

from rag_agent.config import EmbeddingConfig, RetrievalConfig, VectorStoreConfig
from rag_agent.embeddings import EmbeddingBackend
from rag_agent.retrieval import DocumentIngestor, FileSystemRetriever, chunk_text
from rag_agent.vector_store import VectorStore

# --- chunk_text -----------------------------------------------------------


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("", chunk_size=4, overlap=1) == []
    assert chunk_text("   ", chunk_size=4, overlap=1) == []


def test_chunk_text_short_text_returns_single_chunk():
    assert chunk_text("just three words", chunk_size=10, overlap=2) == ["just three words"]


def test_chunk_text_rejects_non_positive_chunk_size():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("a b c", chunk_size=0, overlap=0)


def test_chunk_text_rejects_negative_overlap():
    with pytest.raises(ValueError, match="overlap must be non-negative"):
        chunk_text("a b c", chunk_size=2, overlap=-1)


def test_chunk_text_adjusts_overlap_when_too_large():
    # overlap >= chunk_size is reduced to chunk_size // 3, so this must terminate.
    chunks = chunk_text("a b c d e f", chunk_size=3, overlap=3)

    assert chunks == ["a b c", "c d e", "e f"]


def test_chunk_text_covers_all_words():
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_text(" ".join(words), chunk_size=10, overlap=3)

    assert chunks[0].split()[0] == "w0"
    assert chunks[-1].split()[-1] == "w24"
    # Consecutive chunks share the overlap region.
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]


# --- DocumentIngestor -----------------------------------------------------


def make_retrieval_config(tmp_path, **overrides):
    defaults = {
        "sources": [str(tmp_path)],
        "file_extensions": [".txt", ".md"],
        "chunk_size": 8,
        "chunk_overlap": 2,
    }
    defaults.update(overrides)
    return RetrievalConfig(**defaults)


def test_ingestor_loads_only_supported_extensions(tmp_path):
    (tmp_path / "a.txt").write_text("plain text document", encoding="utf-8")
    (tmp_path / "b.md").write_text("markdown document", encoding="utf-8")
    (tmp_path / "c.png").write_text("binary-ish content", encoding="utf-8")

    documents = DocumentIngestor(make_retrieval_config(tmp_path)).load_documents()

    names = sorted(doc.metadata["path"] for doc in documents)
    assert len(documents) == 2
    assert names[0].endswith("a.txt")
    assert names[1].endswith("b.md")


def test_ingestor_scans_subdirectories_recursively(tmp_path):
    nested = tmp_path / "sub" / "deeper"
    nested.mkdir(parents=True)
    (nested / "deep.txt").write_text("deeply nested document", encoding="utf-8")

    documents = DocumentIngestor(make_retrieval_config(tmp_path)).load_documents()

    assert len(documents) == 1
    assert documents[0].content == "deeply nested document"


def test_ingestor_missing_source_directory_returns_empty(tmp_path):
    config = make_retrieval_config(tmp_path, sources=[str(tmp_path / "missing")])

    assert DocumentIngestor(config).load_documents() == []
    assert DocumentIngestor(config).load_chunks() == []


def test_ingestor_load_chunks_sets_chunk_metadata(tmp_path):
    words = " ".join(f"word{i}" for i in range(20))
    (tmp_path / "long.txt").write_text(words, encoding="utf-8")

    chunks = DocumentIngestor(make_retrieval_config(tmp_path)).load_chunks()

    assert len(chunks) > 1
    for idx, chunk in enumerate(chunks):
        assert chunk.id.endswith(f"#chunk-{idx}")
        assert chunk.metadata["chunk_index"] == idx
        assert chunk.metadata["chunk_size"] == len(chunk.content)
        assert chunk.metadata["source"] == "filesystem"


# --- FileSystemRetriever --------------------------------------------------


def make_retriever(tmp_path):
    embeddings = EmbeddingBackend(
        config=EmbeddingConfig(use_tfidf_fallback=True, vector_dimension=8)
    )
    vector_store = VectorStore(config=VectorStoreConfig(embedding_dimension=8, top_k=3))
    # chunk_size large enough that each test file stays a single chunk.
    config = make_retrieval_config(tmp_path, chunk_size=64)
    retriever = FileSystemRetriever(config, embeddings, vector_store)
    return retriever, vector_store


def test_retriever_ingest_and_search_finds_relevant_document(tmp_path, force_tfidf):
    (tmp_path / "python.txt").write_text(
        "Python is a programming language used for data science.", encoding="utf-8"
    )
    (tmp_path / "cooking.txt").write_text(
        "Bread recipes require flour water yeast and an oven.", encoding="utf-8"
    )

    retriever, vector_store = make_retriever(tmp_path)
    retriever.ingest()

    assert len(vector_store.documents) == 2

    results = retriever.search("python programming language", top_k=1)

    assert len(results) == 1
    assert "python.txt" in results[0].document.id
    assert results[0].score > 0


def test_retriever_search_blank_query_returns_empty(tmp_path, force_tfidf):
    retriever, _ = make_retriever(tmp_path)

    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_retriever_ingest_with_no_documents_leaves_store_empty(tmp_path, force_tfidf):
    retriever, vector_store = make_retriever(tmp_path)

    retriever.ingest()

    assert vector_store.documents == {}


def test_retriever_clear_empties_vector_store(tmp_path, force_tfidf):
    (tmp_path / "doc.txt").write_text("some indexed content here", encoding="utf-8")
    retriever, vector_store = make_retriever(tmp_path)
    retriever.ingest()
    assert vector_store.documents

    retriever.clear()

    assert vector_store.documents == {}
    assert vector_store.embeddings == {}
