"""Simple vector store implementation for the RAG Agentic System."""
from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity

from .config import VectorStoreConfig
from .types import Document, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VectorStore:
    """In-memory vector store with cosine similarity search."""

    config: VectorStoreConfig
    embeddings: dict[str, NDArray[np.float32]] = field(default_factory=dict)
    documents: dict[str, Document] = field(default_factory=dict)

    def add(self, items: Sequence[tuple[Document, NDArray[np.float32]]]) -> None:
        """Add documents and their embeddings to the store."""
        for doc, vector in items:
            if vector.shape[-1] != self.config.embedding_dimension:
                logger.warning(
                    "Embedding dimension mismatch for %s: expected %d got %d",
                    doc.id,
                    self.config.embedding_dimension,
                    vector.shape[-1],
                )
            self.documents[doc.id] = doc
            self.embeddings[doc.id] = vector.astype(np.float32)

        logger.info("Indexed %d documents. Total stored: %d", len(items), len(self.documents))

    def delete(self, doc_ids: Iterable[str]) -> None:
        for identifier in doc_ids:
            self.embeddings.pop(identifier, None)
            self.documents.pop(identifier, None)

    def _matrix(self) -> NDArray[np.float32]:
        if not self.embeddings:
            return np.zeros((0, self.config.embedding_dimension), dtype=np.float32)
        return np.stack(list(self.embeddings.values()))

    def search(
        self, query_vector: NDArray[np.float32], top_k: int | None = None
    ) -> list[RetrievalResult]:
        if query_vector.size == 0 or not self.embeddings:
            return []

        matrix = self._matrix()
        similarities = cosine_similarity(query_vector.reshape(1, -1), matrix)[0]
        identifiers = list(self.embeddings.keys())

        top_k = top_k or self.config.top_k
        best_indices = np.argsort(similarities)[::-1][:top_k]

        results: list[RetrievalResult] = []
        for idx in best_indices:
            doc_id = identifiers[idx]
            doc = self.documents[doc_id]
            score = float(similarities[idx])
            results.append(RetrievalResult(document=doc, score=score))

        return results

    def clear(self) -> None:
        self.embeddings.clear()
        self.documents.clear()


__all__ = ["VectorStore"]