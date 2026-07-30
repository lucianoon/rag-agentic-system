"""Simple SQLite-backed memory store for the agent."""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .config import MemoryConfig
from .types import TaskLog, TaskStep

logger = logging.getLogger(__name__)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS task_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    query TEXT NOT NULL,
    steps TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT NOT NULL
);
"""

_INSERT_LOG_SQL = """
INSERT INTO task_logs (task_id, query, steps, metadata, created_at)
VALUES (?, ?, ?, ?, ?);
"""

_SELECT_RECENT_SQL = """
SELECT task_id, query, steps, metadata, created_at
FROM task_logs
ORDER BY id DESC
LIMIT ?;
"""

_DELETE_OLD_SQL = """
DELETE FROM task_logs
WHERE created_at < datetime('now', ?);
"""


@dataclass(slots=True)
class MemoryStore:
    """Lightweight SQLite-backed storage for task logs."""

    config: MemoryConfig
    _path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._path = Path(self.config.database_path)
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()

    def store(self, log: TaskLog) -> None:
        if not self.config.enabled:
            return

        serialized_steps = json.dumps(
            [
                {
                    "description": step.description,
                    "output": step.output,
                    "references": step.references,
                    "timestamp": step.timestamp.isoformat(),
                }
                for step in log.steps
            ]
        )
        metadata = json.dumps(log.metadata)

        with self._connect() as conn:
            conn.execute(
                _INSERT_LOG_SQL,
                (log.task_id, log.query, serialized_steps, metadata, log.created_at.isoformat()),
            )
            conn.commit()

    def recent(self, limit: int = 10) -> list[TaskLog]:
        if not self.config.enabled:
            return []

        with self._connect() as conn:
            rows = conn.execute(_SELECT_RECENT_SQL, (limit,)).fetchall()

        logs: list[TaskLog] = []
        for task_id, query, steps_json, metadata_json, _created_at in rows:
            steps_data = json.loads(steps_json)
            steps = [
                TaskStep(
                    description=item["description"],
                    output=item["output"],
                    references=item.get("references", []),
                )
                for item in steps_data
            ]
            metadata = json.loads(metadata_json) if metadata_json else {}
            log = TaskLog(
                task_id=task_id,
                query=query,
                steps=steps,
                metadata=metadata,
            )
            logs.append(log)

        return logs

    def cleanup(self) -> None:
        if not self.config.enabled or self.config.cleanup_days <= 0:
            return

        days = f"-{self.config.cleanup_days} day"
        with self._connect() as conn:
            conn.execute(_DELETE_OLD_SQL, (days,))
            conn.commit()


__all__ = ["MemoryStore"]