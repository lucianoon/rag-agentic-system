"""Tests for the add_documents.py helper script (document management on disk)."""

from pathlib import Path

import pytest

import add_documents


def test_create_sample_documents_writes_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    created = add_documents.create_sample_documents()

    assert len(created) == 4
    for path in created:
        file_path = Path(path)
        assert file_path.exists()
        assert file_path.suffix == ".txt"
        assert file_path.read_text(encoding="utf-8").strip()
    assert (tmp_path / "data" / "processed").is_dir()


def test_add_custom_document_writes_content(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    path = add_documents.add_custom_document("nota.md", "conteudo de teste")

    file_path = Path(path)
    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8") == "conteudo de teste"


def test_add_custom_document_rejects_unsupported_extension(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match=".txt ou .md"):
        add_documents.add_custom_document("nota.pdf", "conteudo")


def test_list_documents_returns_empty_when_directory_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert add_documents.list_documents() == []


def test_list_documents_finds_txt_and_md_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "a.txt").write_text("a", encoding="utf-8")
    (processed / "b.md").write_text("b", encoding="utf-8")
    (processed / "c.pdf").write_text("c", encoding="utf-8")

    documents = add_documents.list_documents()

    assert sorted(doc.name for doc in documents) == ["a.txt", "b.md"]
