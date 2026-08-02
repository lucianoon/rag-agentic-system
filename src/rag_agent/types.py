"""Core data models used across the RAG Agentic System."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Document:
    """Represents a raw or processed document."""
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path, source: str = "filesystem") -> Document:
        text = path.read_text(encoding="utf-8")
        metadata = {
            "source": source,
            "path": str(path),
            "created_at": datetime.utcnow().isoformat(),
        }
        return cls(id=str(path), content=text, metadata=metadata)


@dataclass(slots=True)
class RetrievalResult:
    """Result of a vector retrieval operation."""
    document: Document
    score: float


@dataclass(slots=True)
class TaskStep:
    """Represents a single reasoning or action step taken by the agent."""
    description: str
    output: str
    references: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class TaskLog:
    """Tracks the sequence of steps and results for a task."""
    task_id: str
    query: str
    steps: list[TaskStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self, description: str, output: str, references: Iterable[str] | None = None
    ) -> None:
        self.steps.append(
            TaskStep(
                description=description,
                output=output,
                references=list(references or []),
            )
        )


@dataclass(slots=True)
class AgentResponse:
    """Final response returned to the user after task completion."""
    answer: str
    references: list[str]
    steps: list[TaskStep]
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "AgentResponse",
    "Document",
    "RetrievalResult",
    "TaskLog",
    "TaskStep",
]