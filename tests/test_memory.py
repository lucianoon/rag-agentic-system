"""Tests for the SQLite-backed memory store."""

import sqlite3
from datetime import datetime, timedelta

from rag_agent.config import MemoryConfig
from rag_agent.memory import MemoryStore
from rag_agent.types import TaskLog


def make_store(tmp_path, **overrides):
    defaults = {"enabled": True, "database_path": str(tmp_path / "memory.db")}
    defaults.update(overrides)
    return MemoryStore(config=MemoryConfig(**defaults))


def make_log(query="what is rag?"):
    log = TaskLog(task_id="task-1", query=query, metadata={"lang": "pt"})
    log.add_step("Retrieved documents", "2 documents retrieved", references=["a.txt", "b.txt"])
    log.add_step("Generated answer", "final answer")
    return log


def test_store_and_recent_roundtrip(tmp_path):
    store = make_store(tmp_path)
    store.store(make_log())

    logs = store.recent()

    assert len(logs) == 1
    log = logs[0]
    assert log.task_id == "task-1"
    assert log.query == "what is rag?"
    assert log.metadata == {"lang": "pt"}
    assert [step.description for step in log.steps] == ["Retrieved documents", "Generated answer"]
    assert log.steps[0].references == ["a.txt", "b.txt"]
    assert log.steps[1].references == []


def test_recent_orders_newest_first_and_respects_limit(tmp_path):
    store = make_store(tmp_path)
    for i in range(5):
        store.store(TaskLog(task_id=f"task-{i}", query=f"query {i}"))

    logs = store.recent(limit=3)

    assert [log.task_id for log in logs] == ["task-4", "task-3", "task-2"]


def test_disabled_store_ignores_writes_and_reads(tmp_path):
    store = make_store(tmp_path, enabled=False)
    store.store(make_log())

    assert store.recent() == []

    # Nothing should have been written to the database either.
    with sqlite3.connect(tmp_path / "memory.db") as conn:
        count = conn.execute("SELECT COUNT(*) FROM task_logs").fetchone()[0]
    assert count == 0


def test_creates_missing_parent_directory(tmp_path):
    db_path = tmp_path / "nested" / "dir" / "memory.db"

    make_store(tmp_path, database_path=str(db_path))

    assert db_path.exists()


def test_cleanup_removes_old_entries_and_keeps_recent_ones(tmp_path):
    store = make_store(tmp_path, cleanup_days=30)
    old_log = TaskLog(
        task_id="old",
        query="old query",
        created_at=datetime.utcnow() - timedelta(days=40),
    )
    store.store(old_log)
    store.store(TaskLog(task_id="new", query="new query"))

    store.cleanup()

    logs = store.recent()
    assert [log.task_id for log in logs] == ["new"]


def test_cleanup_is_noop_when_disabled_by_config(tmp_path):
    store = make_store(tmp_path, cleanup_days=0)
    store.store(
        TaskLog(task_id="old", query="old", created_at=datetime.utcnow() - timedelta(days=400))
    )

    store.cleanup()

    assert len(store.recent()) == 1
