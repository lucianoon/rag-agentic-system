"""Core pipeline orchestration for the RAG Agentic System."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import AppConfig
from .embeddings import EmbeddingBackend
from .memory import MemoryStore
from .retrieval import FileSystemRetriever
from .types import AgentResponse, Document, TaskLog, TaskStep
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExecutionContext:
    """Holds the runtime dependencies of the pipeline."""

    config: AppConfig
    embeddings: EmbeddingBackend
    retriever: FileSystemRetriever
    vector_store: VectorStore
    memory: MemoryStore

    def ingest_if_empty(self) -> None:
        if self.vector_store.documents:
            return
        logger.info("Vector store empty. Starting ingestion.")
        self.retriever.ingest()
        if not self.vector_store.documents:
            logger.warning("No documents were ingested. Queries will return empty results.")


class Pipeline:
    """Coordinates retrieval, reasoning, and response synthesis."""

    def __init__(self, context: ExecutionContext) -> None:
        self.context = context

    def initialize(self) -> None:
        """Ensure that required resources are ready."""
        self.context.ingest_if_empty()

    def retrieve_documents(self, query: str, top_k: int | None = None) -> list[Document]:
        results = self.context.retriever.search(query, top_k=top_k)
        documents = [result.document for result in results]
        logger.debug("Retrieved %d documents for query '%s'", len(documents), query)
        return documents

    def build_task_log(self, query: str, documents: list[Document], answer: str) -> TaskLog:
        log = TaskLog(task_id=query, query=query)
        log.add_step(
            description="Retrieved documents",
            output=f"{len(documents)} documents retrieved",
            references=[doc.id for doc in documents],
        )
        log.add_step(description="Generated answer", output=answer)
        return log

    def respond(self, query: str, answer: str, documents: list[Document]) -> AgentResponse:
        return AgentResponse(
            answer=answer,
            references=[doc.id for doc in documents],
            steps=[
                TaskStep(description="Retrieved documents", output=str(len(documents))),
                TaskStep(description="Generated answer", output=answer),
            ],
        )

    def process(self, query: str, answer: str, documents: list[Document]) -> AgentResponse:
        response = self.respond(query, answer, documents)
        log = self.build_task_log(query, documents, answer)
        self.context.memory.store(log)
        return response