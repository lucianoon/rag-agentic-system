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

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"

#: Sent when a base URL is set but no credential is (Ollama, LM Studio, vLLM).
LOCAL_PLACEHOLDER_KEY = "not-needed"

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
    input: dict[str, Any]


@dataclass(slots=True)
class ModelTurn:
    """One assistant turn: free text and/or tool calls.

    ``raw_content`` is the assistant message content in Anthropic block
    format, appended verbatim to the conversation so tool_use ids stay
    consistent across iterations.
    """

    text: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    raw_content: list[dict[str, Any]] = field(default_factory=list)


class AgentModel(Protocol):
    """Produces assistant turns for the agent loop."""

    mode: str

    def create_turn(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
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
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
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
    def _last_user_text(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                return message["content"]
        return ""

    @staticmethod
    def _last_tool_result(messages: list[dict[str, Any]]) -> str | None:
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
        temperature: float | None = None,
    ):
        import anthropic  # imported lazily so the SDK stays optional

        self.model = model
        self.max_tokens = max_tokens
        # temperature is accepted for signature compatibility but never sent:
        # sampling parameters were removed on Claude Opus 4.7 and later, and a
        # request carrying one is rejected with a 400.
        self.temperature = temperature
        self._client = anthropic.Anthropic()

    def create_turn(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelTurn:
        # A interface é neutra de provedor (dicts simples, compartilhados com o
        # backend OpenAI-compatible), então os TypedDicts do SDK da Anthropic
        # entram apenas na fronteira da chamada.
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=cast(Any, messages),
            tools=cast(Any, tools),
        )
        text_parts: list[str] = []
        tool_calls: list[ToolCallRequest] = []
        raw_content: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
                raw_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                tool_calls.append(ToolCallRequest(id=block.id, name=block.name, input=block.input))
                raw_content.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                )
        return ModelTurn(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            raw_content=raw_content,
        )


class OpenAICompatibleModel:
    """Agent brain on any endpoint speaking the OpenAI chat-completions API.

    Covers OpenAI, OpenRouter, Groq, Together, DeepInfra, Fireworks, vLLM,
    Ollama and LM Studio with no per-provider code. The agent loop speaks
    Anthropic shapes, so this backend translates in both directions: tool
    schemas and message history on the way out, tool calls on the way back.
    """

    mode = "openai"

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        max_tokens: int = 1024,
        temperature: float | None = 0.2,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        try:
            import openai  # imported lazily so the SDK stays optional
        except ImportError as exc:  # pragma: no cover - depends on the env
            raise RuntimeError(
                "The OpenAI-compatible backend needs the 'openai' package. "
                "Install it with: pip install openai"
            ) from exc

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"],
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _messages_to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate the loop's Anthropic-shaped history to OpenAI shapes.

        Anthropic packs tool results into a ``user`` turn as content blocks;
        OpenAI wants one ``tool`` message per result. An assistant turn carries
        its calls in ``tool_calls`` rather than as ``tool_use`` blocks.
        """
        out: list[dict[str, Any]] = []
        for message in messages:
            role, content = message["role"], message["content"]
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            if role == "assistant":
                text = "".join(b["text"] for b in content if b.get("type") == "text")
                calls = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {
                            "name": b["name"],
                            "arguments": json.dumps(b.get("input") or {}),
                        },
                    }
                    for b in content
                    if b.get("type") == "tool_use"
                ]
                entry: dict[str, Any] = {"role": "assistant", "content": text or None}
                if calls:
                    entry["tool_calls"] = calls
                out.append(entry)
                continue
            for block in content:
                if block.get("type") == "tool_result":
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": str(block.get("content", "")),
                        }
                    )
        return out

    def create_turn(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system}]
            + self._messages_to_openai(messages),
            "tools": self._tools_to_openai(tools),
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        response = self._client.chat.completions.create(**payload)

        choice = response.choices[0].message
        text = (choice.content or "").strip()
        tool_calls: list[ToolCallRequest] = []
        raw_content: list[dict[str, Any]] = []
        if text:
            raw_content.append({"type": "text", "text": text})
        for call in choice.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                # A malformed argument blob must not kill the loop — surface it
                # as an empty input so the tool reports the failure instead.
                logger.warning(
                    "Tool %s returned unparseable arguments: %r",
                    call.function.name,
                    call.function.arguments,
                )
                arguments = {}
            tool_calls.append(
                ToolCallRequest(id=call.id, name=call.function.name, input=arguments)
            )
            raw_content.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.function.name,
                    "input": arguments,
                }
            )
        return ModelTurn(text=text, tool_calls=tool_calls, raw_content=raw_content)


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _anthropic_available() -> bool:
    if not _env("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _openai_available() -> bool:
    """True when an OpenAI-compatible endpoint is reachable from the env.

    A base URL alone is enough: local servers (Ollama, LM Studio, vLLM) take
    no credential.
    """
    if not _env("RAG_LLM_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True


def _build_openai_model(model: str | None, max_tokens: int, temperature: float) -> AgentModel:
    base_url = _env("RAG_LLM_BASE_URL", "OPENAI_BASE_URL")
    api_key = _env("RAG_LLM_API_KEY", "OPENAI_API_KEY") or (
        LOCAL_PLACEHOLDER_KEY if base_url else None
    )
    return OpenAICompatibleModel(
        model=model or os.getenv("RAG_LLM_MODEL") or DEFAULT_OPENAI_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )


def build_agent_model(llm_config: dict[str, Any], temperature: float = 0.2) -> AgentModel:
    """Select an agent model from ``config.llm`` and the environment.

    ``llm.provider``:
        - ``null`` / ``"auto"`` (default) — Claude when its SDK and key are
          available; else an OpenAI-compatible endpoint when one is configured;
          else the scripted model.
        - ``"anthropic"`` — always Claude (raises if the SDK/key is missing).
        - ``"openai"`` — always an OpenAI-compatible endpoint, selected by
          ``RAG_LLM_BASE_URL`` / ``OPENAI_BASE_URL`` and
          ``RAG_LLM_API_KEY`` / ``OPENAI_API_KEY``. This covers OpenAI,
          OpenRouter, Groq, Together, vLLM, Ollama and LM Studio.
        - ``"scripted"`` — always the offline deterministic model.

    ``llm.model`` overrides the model id for either live backend; without it
    each backend falls back to its own default.
    """
    provider = (llm_config.get("provider") or "auto").lower()
    configured_model = llm_config.get("model") or None
    max_tokens = int(llm_config.get("max_tokens") or 1024)

    if provider == "scripted":
        return ScriptedModel()
    if provider == "anthropic":
        return ClaudeModel(model=configured_model or DEFAULT_MODEL, max_tokens=max_tokens)
    if provider == "openai":
        return _build_openai_model(configured_model, max_tokens, temperature)
    if provider == "auto":
        if _anthropic_available():
            return ClaudeModel(model=configured_model or DEFAULT_MODEL, max_tokens=max_tokens)
        if _openai_available():
            return _build_openai_model(configured_model, max_tokens, temperature)
        logger.info("No live backend configured; using the scripted offline model.")
        return ScriptedModel()
    raise ValueError(
        f"Unknown llm.provider={provider!r}. "
        "Expected one of: auto, anthropic, openai, scripted."
    )


__all__ = [
    "AgentModel",
    "ClaudeModel",
    "OpenAICompatibleModel",
    "ModelTurn",
    "NO_RESULTS_ANSWER",
    "NO_RESULTS_MARKER",
    "ScriptedModel",
    "ToolCallRequest",
    "build_agent_model",
]
