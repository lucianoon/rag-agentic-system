"""Integration-style tests for the pipeline and the agent (no LLM, no network)."""

from rag_agent import AgenticRAG, AppConfig, ExecutionContext, FileSystemRetriever, MemoryStore, VectorStore
from rag_agent.config import EmbeddingConfig, MemoryConfig, RetrievalConfig, VectorStoreConfig
from rag_agent.embeddings import EmbeddingBackend
from rag_agent.pipeline import Pipeline
from rag_agent.types import Document


def build_context(tmp_path, memory_enabled=True):
    config = AppConfig(
        embeddings=EmbeddingConfig(use_tfidf_fallback=True, vector_dimension=8),
        vector_store=VectorStoreConfig(embedding_dimension=8, top_k=3),
        retrieval=RetrievalConfig(
            sources=[str(tmp_path / "docs")],
            file_extensions=[".txt", ".md"],
            chunk_size=16,
            chunk_overlap=4,
        ),
        memory=MemoryConfig(enabled=memory_enabled, database_path=str(tmp_path / "memory.db")),
    )
    embeddings = EmbeddingBackend(config=config.embeddings)
    vector_store = VectorStore(config=config.vector_store)
    retriever = FileSystemRetriever(config.retrieval, embeddings, vector_store)
    memory = MemoryStore(config=config.memory)
    return ExecutionContext(config, embeddings, retriever, vector_store, memory)


def write_docs(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "python.txt").write_text(
        "Python is a programming language widely used for data science and automation.",
        encoding="utf-8",
    )
    (docs_dir / "cooking.txt").write_text(
        "Bread recipes require flour water yeast salt and a hot oven to bake.",
        encoding="utf-8",
    )
    return docs_dir


def test_ingest_if_empty_populates_store_once(tmp_path, force_tfidf):
    write_docs(tmp_path)
    context = build_context(tmp_path)

    context.ingest_if_empty()
    assert len(context.vector_store.documents) == 2

    # A second call must not re-ingest (store already populated).
    marker = Document(id="marker", content="marker")
    context.vector_store.documents["marker"] = marker
    context.ingest_if_empty()
    assert "marker" in context.vector_store.documents
    assert len(context.vector_store.documents) == 3


def test_pipeline_process_returns_response_and_stores_task_log(tmp_path, force_tfidf):
    write_docs(tmp_path)
    context = build_context(tmp_path)
    pipeline = Pipeline(context)
    pipeline.initialize()

    documents = pipeline.retrieve_documents("python programming", top_k=1)
    assert len(documents) == 1

    response = pipeline.process("python programming", "the answer", documents)

    assert response.answer == "the answer"
    assert response.references == [doc.id for doc in documents]
    assert [step.description for step in response.steps] == [
        "Retrieved documents",
        "Generated answer",
    ]

    logs = context.memory.recent()
    assert len(logs) == 1
    assert logs[0].query == "python programming"
    assert len(logs[0].steps) == 2


def test_agent_end_to_end_query_uses_ingested_documents(tmp_path, force_tfidf):
    write_docs(tmp_path)
    agent = AgenticRAG(build_context(tmp_path))
    agent.initialize()

    response = agent.query("What is Python used for?")

    assert "Based on the retrieved documents" in response.answer
    assert response.references
    # The query was logged to memory.
    assert len(agent.context.memory.recent()) == 1


def test_agent_empty_question_returns_validation_message(tmp_path, force_tfidf):
    agent = AgenticRAG(build_context(tmp_path))

    response = agent.query("   ")

    assert response.answer == "Please provide a valid question."
    assert response.references == []


def test_agent_query_without_documents_returns_fallback_message(tmp_path, force_tfidf):
    (tmp_path / "docs").mkdir()
    agent = AgenticRAG(build_context(tmp_path))
    agent.initialize()

    response = agent.query("quantum computing basics?")

    assert "couldn't find relevant information" in response.answer
    assert response.references == []


def test_agent_stopword_only_query_on_empty_system_does_not_crash(tmp_path, force_tfidf):
    # Regression test: a stop-word-only query against an empty system used to
    # crash while auto-fitting the TF-IDF vectorizer.
    (tmp_path / "docs").mkdir()
    agent = AgenticRAG(build_context(tmp_path))
    agent.initialize()

    response = agent.query("anything at all?")

    assert "couldn't find relevant information" in response.answer


def test_agent_add_documents_returns_zero_for_empty_or_unsupported(tmp_path, force_tfidf):
    agent = AgenticRAG(build_context(tmp_path))

    assert agent.add_documents([]) == 0

    unsupported = tmp_path / "image.png"
    unsupported.write_text("not text", encoding="utf-8")
    assert agent.add_documents([str(unsupported)]) == 0
    assert agent.add_documents([str(tmp_path / "missing.txt")]) == 0


def test_agent_stats_and_clear_memory(tmp_path, force_tfidf):
    write_docs(tmp_path)
    agent = AgenticRAG(build_context(tmp_path))
    agent.initialize()

    stats = agent.get_stats()
    assert stats["total_documents"] == 2
    assert stats["embeddings_stored"] == 2
    assert stats["memory_enabled"] is True

    agent.clear_memory()

    stats = agent.get_stats()
    assert stats["total_documents"] == 0
    assert stats["embeddings_stored"] == 0
