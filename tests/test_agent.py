"""Tests for document ingestion and agent behavior."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_agent import (
    AgenticRAG,
    AppConfig,
    ExecutionContext,
    FileSystemRetriever,
    MemoryStore,
    VectorStore,
)
from rag_agent.config import EmbeddingConfig, MemoryConfig, RetrievalConfig, VectorStoreConfig
from rag_agent.retrieval import chunk_text


class DummyEmbeddings:
    def __init__(self):
        self.config = EmbeddingConfig(use_tfidf_fallback=True, vector_dimension=3)
        self.fit_calls = []

    def fit(self, corpus):
        self.fit_calls.append(list(corpus))

    def embed_single(self, text):
        return np.array([
            float(len(text) % 11),
            float(sum(ord(ch) for ch in text) % 17),
            1.0,
        ], dtype=np.float32)


def build_agent(tmp_path):
    config = AppConfig(
        embeddings=EmbeddingConfig(use_tfidf_fallback=True, vector_dimension=3),
        vector_store=VectorStoreConfig(embedding_dimension=3, top_k=3),
        retrieval=RetrievalConfig(
            sources=[str(tmp_path)],
            file_extensions=[".txt", ".md"],
            chunk_size=8,
            chunk_overlap=2,
        ),
        memory=MemoryConfig(enabled=False, database_path=str(tmp_path / "memory.db")),
    )
    embeddings = DummyEmbeddings()
    vector_store = VectorStore(config.vector_store)
    retriever = FileSystemRetriever(config.retrieval, embeddings, vector_store)
    memory = MemoryStore(config.memory)
    context = ExecutionContext(config, embeddings, retriever, vector_store, memory)
    return AgenticRAG(context), embeddings


def test_chunk_text_uses_overlap():
    chunks = chunk_text("one two three four five six seven", chunk_size=4, overlap=2)
    assert chunks == ["one two three four", "three four five six", "five six seven"]


def test_add_documents_indexes_supported_files(tmp_path):
    doc = tmp_path / "knowledge.txt"
    doc.write_text("Python is useful for data science and automation.", encoding="utf-8")
    ignored = tmp_path / "image.png"
    ignored.write_text("not indexed", encoding="utf-8")

    agent, embeddings = build_agent(tmp_path)
    added = agent.add_documents([str(doc), str(ignored), str(tmp_path / "missing.txt")])

    assert added >= 1
    assert embeddings.fit_calls
    assert len(agent.context.vector_store.documents) == added
    assert all("knowledge.txt" in doc_id for doc_id in agent.context.vector_store.documents)


def test_query_returns_references_after_add_documents(tmp_path):
    doc = tmp_path / "rag.md"
    doc.write_text("RAG combines retrieval with generation for better answers.", encoding="utf-8")

    agent, _ = build_agent(tmp_path)
    agent.add_documents([str(doc)])
    response = agent.query("What does RAG combine?")

    assert "Based on the retrieved documents" in response.answer
    assert response.references
