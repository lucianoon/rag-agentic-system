"""Tests for YAML configuration loading."""

from pathlib import Path

from rag_agent.config import DEFAULT_CONFIG_PATH, AppConfig, load_config


def test_load_config_missing_file_returns_defaults(tmp_path):
    config = load_config(tmp_path / "does-not-exist.yaml")

    assert config == AppConfig()


def test_load_config_empty_file_returns_defaults(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")

    config = load_config(empty)

    assert config == AppConfig()


def test_load_config_partial_yaml_keeps_defaults_for_missing_sections(tmp_path):
    partial = tmp_path / "partial.yaml"
    partial.write_text(
        "retrieval:\n  chunk_size: 128\n  chunk_overlap: 16\n",
        encoding="utf-8",
    )

    config = load_config(partial)

    assert config.retrieval.chunk_size == 128
    assert config.retrieval.chunk_overlap == 16
    # Untouched sections fall back to defaults.
    assert config.retrieval.sources == ["data/processed"]
    assert config.embeddings.use_tfidf_fallback is True
    assert config.vector_store.top_k == 5


def test_load_config_full_yaml_overrides(tmp_path):
    full = tmp_path / "full.yaml"
    full.write_text(
        """
embeddings:
  model_name: "custom-model"
  use_tfidf_fallback: false
  vector_dimension: 128
vector_store:
  embedding_dimension: 128
  top_k: 2
memory:
  enabled: false
  database_path: "custom.db"
llm:
  provider: "openai"
""",
        encoding="utf-8",
    )

    config = load_config(full)

    assert config.embeddings.model_name == "custom-model"
    assert config.embeddings.use_tfidf_fallback is False
    assert config.embeddings.vector_dimension == 128
    assert config.vector_store.embedding_dimension == 128
    assert config.vector_store.top_k == 2
    assert config.memory.enabled is False
    assert config.memory.database_path == "custom.db"
    assert config.llm == {"provider": "openai"}


def test_default_config_file_exists_and_loads():
    assert DEFAULT_CONFIG_PATH.exists()

    config = load_config()

    assert config.retrieval.chunk_size == 512
    assert config.retrieval.chunk_overlap == 64
    assert config.retrieval.file_extensions == [".txt", ".md"]
    assert config.vector_store.similarity_metric == "cosine"
    assert config.embeddings.use_tfidf_fallback is True


def test_from_dict_ignores_unknown_top_level_keys():
    config = AppConfig.from_dict({"agent": {"max_iterations": 9}, "unknown": {"x": 1}})

    assert config.agent.max_iterations == 9
