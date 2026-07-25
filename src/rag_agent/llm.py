"""Agent model backends: Claude with tool use, and a scripted offline model.

The agent loop only depends on the small :class:`AgentModel` protocol, so the
same loop machinery runs against two very different backends:

- ``ClaudeModel`` — real agentic behavior: Claude decides which tools to call,
  reads the results and iterates until it can answer.
- ``ScriptedModel`` — a deterministic stand-in that always searches once and
  then answers extractively from the tool result. It keeps the loop, tool
  execution, memory logging and verification genuinely exercised in CI and on
  machines without an API key.

``build_agent_model`` selects a backend from ``config.llm`` and the
environment.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-8"

NO_RESULTS_MARKER = "No matching documents found."

NO_RESULTS_ANSWER = (
    "I couldn't find relevant information to answer your question. "
    "Try adding more documents to the system."
)


@dataclass(slots=True)
class ToolCallRequest:
    """One tool invocation requested by the model."""

    id: str
    name: str
    input: Dict[str, Any]


@dataclass(slots=True)
class ModelTurn:
    """One assistant turn: free text and/or tool calls.

    ``raw_content`` is the assistant message content in Anthropic block
    format, appended verbatim to the conversation so tool_use ids stay
    consistent across iterations.
    """

    text: str = ""
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    raw_content: List[Dict[str, Any]] = field(default_factory=list)


class AgentModel(Protocol):
    """Produces assistant turns for the agent loop."""

    mode: str

    def create_turn(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> ModelTurn:
        ...


class ScriptedModel:
    """Deterministic offline model: search once, then answer extractively.

    Turn 1 requests a ``search_documents`` call for the user's task. Turn 2
    reads the tool result and produces the same extractive answer the
    pre-agentic pipeline used, so behavior without an API key is unchanged —
    but it now flows through the real loop, tools and verification.
    """

    mode = "scripted"

    def create_turn(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> ModelTurn:
        tool_result = self._last_tool_result(messages)
        if tool_result is None:
            task = self._last_user_text(messages)
            call = ToolCallRequest(
                id="scripted-search-1",
                name="search_documents",
                input={"query": task},
            )
            return ModelTurn(
                tool_calls=[call],
                raw_content=[
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
                ],
            )

        if tool_result.startswith(NO_RESULTS_MARKER):
            return ModelTurn(text=NO_RESULTS_ANSWER)

        return ModelTurn(text=f"Based on the retrieved documents:\n\n{tool_result}")

    @staticmethod
    def _last_user_text(messages: List[Dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                return message["content"]
        return ""

    @staticmethod
    def _last_tool_result(messages: List[Dict[str, Any]]) -> str | None:
        for message in reversed(messages):
            content = message.get("content")
            if message.get("role") == "user" and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        return str(block.get("content", ""))
        return None


class ClaudeModel:
    """Claude as the agent brain, using native tool use."""

    mode = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ):
        import anthropic  # imported lazily so the SDK stays optional

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = anthropic.Anthropic()

    def create_turn(
        self,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> ModelTurn:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=messages,
            tools=tools,
        )
        text_parts: List[str] = []
        tool_calls: List[ToolCallRequest] = []
        raw_content: List[Dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
                raw_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_calls.append(ToolCallRequest(id=block.id, name=block.name, input=block.input))
                raw_content.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                )
        return ModelTurn(text="".join(text_parts).strip(), tool_calls=tool_calls, raw_content=raw_content)


def _anthropic_available() -> bool:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def build_agent_model(llm_config: Dict[str, Any], temperature: float = 0.2) -> AgentModel:
    """Select an agent model from ``config.llm`` and the environment.

    ``llm.provider``:
        - ``null`` / ``"auto"`` (default) — Claude when the SDK and
          ``ANTHROPIC_API_KEY`` are available, else the scripted model.
        - ``"anthropic"`` — always Claude (raises if the SDK/key is missing).
        - ``"scripted"`` — always the offline deterministic model.
    """
    provider = (llm_config.get("provider") or "auto").lower()
    model = llm_config.get("model") or DEFAULT_MODEL
    max_tokens = int(llm_config.get("max_tokens") or 1024)

    if provider == "scripted":
        return ScriptedModel()
    if provider == "anthropic":
        return ClaudeModel(model=model, max_tokens=max_tokens, temperature=temperature)
    if provider == "auto":
        if _anthropic_available():
            return ClaudeModel(model=model, max_tokens=max_tokens, temperature=temperature)
        logger.info("No Anthropic key/SDK available; using the scripted offline model.")
        return ScriptedModel()
    raise ValueError(
        f"Unknown llm.provider={provider!r}. Expected one of: auto, anthropic, scripted."
    )


__all__ = [
    "AgentModel",
    "ClaudeModel",
    "ModelTurn",
    "NO_RESULTS_ANSWER",
    "NO_RESULTS_MARKER",
    "ScriptedModel",
    "ToolCallRequest",
    "build_agent_model",
]
