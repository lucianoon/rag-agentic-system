"""Tests for the core data models."""

from rag_agent.types import Document, TaskLog


def test_document_from_path_reads_content_and_metadata(tmp_path):
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello world", encoding="utf-8")

    document = Document.from_path(file_path)

    assert document.id == str(file_path)
    assert document.content == "hello world"
    assert document.metadata["source"] == "filesystem"
    assert document.metadata["path"] == str(file_path)
    assert "created_at" in document.metadata


def test_task_log_add_step_appends_steps_in_order():
    log = TaskLog(task_id="t1", query="q")

    log.add_step("first", "out1")
    log.add_step("second", "out2", references=["ref-a"])

    assert [step.description for step in log.steps] == ["first", "second"]
    assert log.steps[0].references == []
    assert log.steps[1].references == ["ref-a"]
