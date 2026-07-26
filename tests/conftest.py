"""Shared pytest fixtures for the test suite."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from rag_agent.embeddings import EmbeddingBackend  # noqa: E402

_LLM_ENV = (
    "RAG_LLM_BASE_URL",
    "RAG_LLM_API_KEY",
    "RAG_LLM_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)


@pytest.fixture(autouse=True)
def offline_llm_env(monkeypatch):
    """Keep backend selection out of the developer's environment.

    Without this, a machine with ANTHROPIC_API_KEY or OPENAI_API_KEY exported
    resolves a live backend and the suite stops matching what CI runs.
    """
    for name in _LLM_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def force_tfidf(monkeypatch):
    """Force the TF-IDF fallback so tests never download sentence-transformers models."""
    monkeypatch.setattr(
        EmbeddingBackend,
        "has_sentence_transformer",
        property(lambda self: False),
    )


@pytest.fixture(autouse=True)
def no_anthropic_key(monkeypatch):
    """Keep tests deterministic: the auto provider must resolve to scripted."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
