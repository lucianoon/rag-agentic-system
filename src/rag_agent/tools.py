"""Tools the agent can call, and their executor.

Tool schemas follow the Anthropic tool-use format so they can be passed
directly to the API; the scripted model uses the same definitions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .llm import NO_RESULTS_MARKER
from .pipeline import ExecutionContext

logger = logging.getLogger(__name__)

MAX_EXCERPT_CHARS = 500

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "search_documents",
        "description": (
            "Search the indexed document corpus by semantic similarity. "
            "Returns the most relevant document excerpts with their ids and scores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "top_k": {
                    "type": "integer",
                    "description": "How many documents to return (default from config).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_task_history",
        "description": (
            "List recent tasks the agent has completed, from persistent memory. "
            "Useful to check whether a similar question was already answered."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Maximum entries to return (default 5)."}
            },
            "required": [],
        },
    },
]


@dataclass(slots=True)
class ToolExecutor:
    """Executes tool calls against the runtime context.

    ``execute`` returns ``(result_text, references)`` — the text goes back to
    the model as the tool result; the references (document ids) accumulate on
    the final response.
    """

    context: ExecutionContext
    default_top_k: int | None = None

    def execute(self, name: str, tool_input: Dict[str, Any]) -> Tuple[str, List[str]]:
        if name == "search_documents":
            return self._search_documents(tool_input)
        if name == "get_task_history":
            return self._get_task_history(tool_input)
        logger.warning("Agent requested unknown tool: %s", name)
        return f"Error: unknown tool '{name}'.", []

    def _search_documents(self, tool_input: Dict[str, Any]) -> Tuple[str, List[str]]:
        query = str(tool_input.get("query", "")).strip()
        top_k = tool_input.get("top_k") or self.default_top_k
        results = self.context.retriever.search(query, top_k=top_k)
        if not results:
            return NO_RESULTS_MARKER, []

        lines = []
        references = []
        for result in results:
            doc = result.document
            source = doc.metadata.get("path", doc.id)
            excerpt = doc.content[:MAX_EXCERPT_CHARS]
            lines.append(f"From {source} (score {result.score:.3f}):\n{excerpt}")
            references.append(doc.id)
        return "\n\n".join(lines), references

    def _get_task_history(self, tool_input: Dict[str, Any]) -> Tuple[str, List[str]]:
        limit = int(tool_input.get("limit") or 5)
        logs = self.context.memory.recent(limit=limit)
        if not logs:
            return "No previous tasks in memory.", []
        lines = [f"- {log.query} ({len(log.steps)} steps)" for log in logs]
        return "Recent tasks:\n" + "\n".join(lines), []


__all__ = ["TOOL_DEFINITIONS", "ToolExecutor"]
