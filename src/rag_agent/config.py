"""Configuration management for the RAG Agentic System."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str | None = None
    use_tfidf_fallback: bool = True
    vector_dimension: int = 384


@dataclass(slots=True)
class VectorStoreConfig:
    backend: str = "simple"
    embedding_dimension: int = 384
    similarity_metric: str = "cosine"
    top_k: int = 5


@dataclass(slots=True)
class RetrievalConfig:
    sources: list[str] = field(default_factory=lambda: ["data/processed"])
    file_extensions: list[str] = field(default_factory=lambda: [".txt", ".md"])
    chunk_size: int = 512
    chunk_overlap: int = 64


@dataclass(slots=True)
class MemoryConfig:
    enabled: bool = True
    database_path: str = "data/memory.db"
    cleanup_days: int = 30
    importance_threshold: float = 0.3


@dataclass(slots=True)
class VerificationConfig:
    enabled: bool = True
    factual_checks: bool = True
    consistency_checks: bool = True
    min_confidence: float = 0.55


@dataclass(slots=True)
class AgentConfig:
    max_iterations: int = 6
    retry_attempts: int = 2
    temperature: float = 0.2
    reasoning_steps: int = 3


@dataclass(slots=True)
class AppConfig:
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    llm: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        return cls(
            embeddings=EmbeddingConfig(**data.get("embeddings", {})),
            vector_store=VectorStoreConfig(**data.get("vector_store", {})),
            retrieval=RetrievalConfig(**data.get("retrieval", {})),
            memory=MemoryConfig(**data.get("memory", {})),
            verification=VerificationConfig(**data.get("verification", {})),
            agent=AgentConfig(**data.get("agent", {})),
            llm=data.get("llm", {}),
        )


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from YAML file or defaults."""
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return AppConfig()

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return AppConfig.from_dict(data)


__all__ = ["AppConfig", "DEFAULT_CONFIG_PATH", "EmbeddingConfig", "load_config"]