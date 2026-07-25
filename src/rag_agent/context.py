"""Factory for the runtime execution context."""
from __future__ import annotations

from .config import AppConfig
from .embeddings import EmbeddingBackend
from .memory import MemoryStore
from .pipeline import ExecutionContext
from .retrieval import FileSystemRetriever
from .vector_store import VectorStore


def create_context(config: AppConfig) -> ExecutionContext:
    """Create the execution context from configuration."""
    embeddings = EmbeddingBackend(config=config.embeddings)
    vector_store = VectorStore(config=config.vector_store)
    retriever = FileSystemRetriever(
        config=config.retrieval,
        embeddings=embeddings,
        vector_store=vector_store,
    )
    memory = MemoryStore(config=config.memory)

    return ExecutionContext(
        config=config,
        embeddings=embeddings,
        retriever=retriever,
        vector_store=vector_store,
        memory=memory,
    )


__all__ = ["create_context"]
