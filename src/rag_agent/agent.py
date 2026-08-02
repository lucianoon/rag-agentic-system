"""Main RAG Agent implementation with agentic behavior."""
from __future__ import annotations

import logging
from pathlib import Path

from .llm import AgentModel, build_agent_model
from .loop import AgentLoop
from .pipeline import ExecutionContext, Pipeline
from .retrieval import chunk_text
from .types import AgentResponse, Document, TaskLog

logger = logging.getLogger(__name__)


class AgenticRAG:
    """RAG agent driven by a tool-use loop.

    The model behind the loop is pluggable (``config.llm``): Claude with
    native tool use when a key is available, or a deterministic scripted
    model that keeps the exact same loop running offline.
    """

    def __init__(self, context: ExecutionContext, model: AgentModel | None = None) -> None:
        self.context = context
        self.pipeline = Pipeline(context)
        self._model = model
        self._initialized = False

    @property
    def model(self) -> AgentModel:
        if self._model is None:
            self._model = build_agent_model(
                self.context.config.llm,
                temperature=self.context.config.agent.temperature,
            )
            logger.info("Agent model: %s", self._model.mode)
        return self._model

    def initialize(self) -> None:
        """Initialize the agent and prepare resources."""
        if self._initialized:
            return

        logger.info("Initializing RAG Agentic System")
        self.pipeline.initialize()
        self._initialized = True
        logger.info("RAG Agentic System ready")

    def query(self, question: str, top_k: int | None = None) -> AgentResponse:
        """Run a task through the agent loop and return the final answer."""
        if not self._initialized:
            self.initialize()

        if not question.strip():
            return AgentResponse(
                answer="Please provide a valid question.",
                references=[],
                steps=[],
            )

        logger.info("Processing query: %s", question)

        loop = AgentLoop(self.context, self.model, default_top_k=top_k)
        result = loop.run(question)

        metadata = {
            "agent_mode": self.model.mode,
            "iterations": result.iterations,
            "tool_calls": result.tool_calls,
        }
        if result.confidence is not None:
            metadata["confidence"] = result.confidence
            metadata["verified"] = result.verified

        log = TaskLog(task_id=question, query=question, steps=result.steps, metadata=metadata)
        self.context.memory.store(log)

        logger.info(
            "Query processed: iterations=%d tool_calls=%d mode=%s",
            result.iterations,
            result.tool_calls,
            self.model.mode,
        )
        return AgentResponse(
            answer=result.answer,
            references=result.references,
            steps=result.steps,
            metadata=metadata,
        )

    def add_documents(self, file_paths: list[str]) -> int:
        """Add specific text/Markdown documents to the active vector store.

        Args:
            file_paths: Paths to .txt/.md files that should be chunked and indexed.

        Returns:
            Number of chunks added to the vector store.
        """
        if not file_paths:
            return 0

        allowed_extensions = {ext.lower() for ext in self.context.config.retrieval.file_extensions}
        chunked_documents: list[Document] = []

        for raw_path in file_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.exists() or not path.is_file():
                logger.warning("Skipping missing document: %s", path)
                continue
            if path.suffix.lower() not in allowed_extensions:
                logger.warning("Skipping unsupported document extension: %s", path)
                continue

            document = Document.from_path(path)
            chunks = chunk_text(
                document.content,
                self.context.config.retrieval.chunk_size,
                self.context.config.retrieval.chunk_overlap,
            )
            for idx, chunk in enumerate(chunks):
                metadata = dict(document.metadata)
                metadata.update({"chunk_index": idx, "chunk_size": len(chunk)})
                chunked_documents.append(
                    Document(
                        id=f"{document.id}#chunk-{idx}",
                        content=chunk,
                        metadata=metadata,
                    )
                )

        if not chunked_documents:
            logger.warning("No valid documents were added.")
            return 0

        # Refit TF-IDF fallback on the full corpus to keep vector dimensions stable.
        corpus = [doc.content for doc in self.context.vector_store.documents.values()]
        corpus.extend(doc.content for doc in chunked_documents)
        if self.context.embeddings.config.use_tfidf_fallback:
            self.context.embeddings.fit(corpus)

        all_documents = list(self.context.vector_store.documents.values()) + chunked_documents
        vectors = [
            (doc, self.context.embeddings.embed_single(doc.content))
            for doc in all_documents
        ]
        self.context.vector_store.clear()
        self.context.vector_store.add(vectors)
        logger.info(
            "Added %d document chunks from %d files",
            len(chunked_documents),
            len(file_paths),
        )
        return len(chunked_documents)

    def clear_memory(self) -> None:
        """Clear the vector store and reset."""
        self.pipeline.context.retriever.clear()
        logger.info("Vector store cleared")

    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "total_documents": len(self.context.vector_store.documents),
            "embeddings_stored": len(self.context.vector_store.embeddings),
            "memory_enabled": self.context.memory.config.enabled,
        }


__all__ = ["AgenticRAG"]