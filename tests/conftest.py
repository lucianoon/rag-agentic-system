"""Shared pytest fixtures for the test suite."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from rag_agent.embeddings import EmbeddingBackend  # noqa: E402


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
