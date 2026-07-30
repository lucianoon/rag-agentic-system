"""Document ingestion and retrieval utilities."""
from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from .config import RetrievalConfig
from .embeddings import EmbeddingBackend, VectorArray
from .types import Document, RetrievalResult
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple text chunker operating on words."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        logger.warning("overlap >= chunk_size; adjusting overlap to chunk_size // 3")
        overlap = max(chunk_size // 3, 0)

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += step

    return chunks


class DocumentIngestor:
    """Loads documents from filesystem and prepares them for indexing."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config

    def _iter_files(self) -> Iterable[Path]:
        directories = [Path(path) for path in self.config.sources]
        for directory in directories:
            if not directory.exists():
                logger.warning("Source directory does not exist: %s", directory)
                continue

            for path in directory.rglob("*"):
                if path.is_file() and path.suffix.lower() in self.config.file_extensions:
                    yield path

    def load_documents(self) -> list[Document]:
        documents: list[Document] = []
        for path in self._iter_files():
            try:
                document = Document.from_path(path)
                documents.append(document)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to ingest %s: %s", path, exc)

        logger.info("Loaded %d raw documents from sources.", len(documents))
        return documents

    def load_chunks(self) -> list[Document]:
        chunked: list[Document] = []
        for document in self.load_documents():
            chunks = chunk_text(document.content, self.config.chunk_size, self.config.chunk_overlap)
            if not chunks:
                continue

            for idx, chunk in enumerate(chunks):
                chunk_id = f"{document.id}#chunk-{idx}"
                metadata = dict(document.metadata)
                metadata.update({"chunk_index": idx, "chunk_size": len(chunk)})
                chunked.append(Document(id=chunk_id, content=chunk, metadata=metadata))

        logger.info("Created %d chunked documents.", len(chunked))
        return chunked


class FileSystemRetriever:
    """Retrieves documents using vector search over filesystem-ingested data."""

    def __init__(
        self,
        config: RetrievalConfig,
        embeddings: EmbeddingBackend,
        vector_store: VectorStore,
    ) -> None:
        self.config = config
        self.embeddings = embeddings
        self.vector_store = vector_store

    def ingest(self) -> None:
        ingestor = DocumentIngestor(self.config)
        documents = ingestor.load_chunks()
        if not documents:
            logger.warning("No documents ingested; vector store unchanged.")
            return

        corpus = [doc.content for doc in documents]
        if self.embeddings.config.use_tfidf_fallback:
            self.embeddings.fit(corpus)

        vectors: list[tuple[Document, VectorArray]] = []
        for doc in documents:
            vector = self.embeddings.embed_single(doc.content)
            vectors.append((doc, vector))

        self.vector_store.add(vectors)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        if not query.strip():
            return []

        query_vector = self.embeddings.embed_single(query)
        return self.vector_store.search(query_vector, top_k=top_k)

    def clear(self) -> None:
        self.vector_store.clear()


__all__ = ["DocumentIngestor", "FileSystemRetriever", "chunk_text"]