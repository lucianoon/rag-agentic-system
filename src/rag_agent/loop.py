"""The agent loop: model turns, tool execution, iteration cap, verification.

This is what makes the system *agentic*: the model decides which tools to
call, reads their results and iterates until it can answer — bounded by
``agent.max_iterations``. After the final answer, the loop optionally checks
that the answer is grounded in the evidence the tools actually returned
(``verification.factual_checks`` / ``verification.min_confidence``), flagging
answers that drift from the retrieved documents.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .llm import AgentModel
from .pipeline import ExecutionContext
from .tools import TOOL_DEFINITIONS, ToolExecutor
from .types import TaskStep

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a research agent over a private document corpus. Use the "
    "search_documents tool to gather evidence before answering; refine the "
    "query and search again if the first results are not enough. Answer using "
    "ONLY information from tool results, citing the source paths. If the "
    "corpus does not contain the answer, say so plainly instead of guessing."
)

ITERATION_LIMIT_ANSWER = (
    "I could not complete this task within the configured iteration limit. "
    "Partial evidence was gathered but no final answer was produced."
)

_STOPWORDS = frozenset(
    [
        "a", "an", "and", "are", "as", "at", "be", "based", "by", "for",
        "from", "has", "have", "if", "in", "into", "is", "it", "its", "of",
        "on", "or", "should", "that", "the", "their", "this", "to", "was",
        "were", "will", "with", "must", "not",
    ]
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def grounding_score(answer: str, evidence: list[str], threshold: float = 0.5) -> float:
    """Fraction of answer sentences lexically supported by the evidence.

    A deterministic proxy for groundedness (not semantic entailment): a
    sentence counts as supported when at least ``threshold`` of its content
    tokens appear in the concatenated tool results.
    """
    evidence_tokens: set[str] = set()
    for text in evidence:
        evidence_tokens |= _content_tokens(text)

    sentences = _sentences(answer)
    if not sentences or not evidence_tokens:
        return 0.0

    supported = 0
    for sentence in sentences:
        tokens = _content_tokens(sentence)
        if not tokens or len(tokens & evidence_tokens) / len(tokens) >= threshold:
            supported += 1
    return round(supported / len(sentences), 4)


@dataclass(slots=True)
class LoopResult:
    """Outcome of one agent-loop run."""

    answer: str
    steps: list[TaskStep] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    iterations: int = 0
    tool_calls: int = 0
    confidence: float | None = None
    verified: bool | None = None


class AgentLoop:
    """Runs a task through model turns and tool executions."""

    def __init__(
        self,
        context: ExecutionContext,
        model: AgentModel,
        default_top_k: int | None = None,
    ):
        self.context = context
        self.model = model
        self.executor = ToolExecutor(context, default_top_k=default_top_k)

    def run(self, task: str) -> LoopResult:
        config = self.context.config
        max_iterations = max(1, config.agent.max_iterations)

        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        result = LoopResult(answer="")
        evidence: list[str] = []
        seen_references: set[str] = set()

        for iteration in range(1, max_iterations + 1):
            result.iterations = iteration
            turn = self.model.create_turn(SYSTEM_PROMPT, messages, TOOL_DEFINITIONS)

            if not turn.tool_calls:
                result.answer = turn.text
                result.steps.append(
                    TaskStep(description="Generated answer", output=turn.text[:300])
                )
                break

            messages.append({"role": "assistant", "content": turn.raw_content})
            tool_results = []
            for call in turn.tool_calls:
                result.tool_calls += 1
                output, references = self.executor.execute(call.name, call.input)
                evidence.append(output)
                for reference in references:
                    if reference not in seen_references:
                        seen_references.add(reference)
                        result.references.append(reference)
                result.steps.append(
                    TaskStep(
                        description=f"Tool call: {call.name}",
                        output=output[:300],
                        references=references,
                    )
                )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": call.id, "content": output}
                )
            messages.append({"role": "user", "content": tool_results})
        else:
            logger.warning("Agent hit the iteration limit (%d) without answering.", max_iterations)
            result.answer = ITERATION_LIMIT_ANSWER
            result.steps.append(
                TaskStep(description="Iteration limit reached", output=result.answer)
            )

        self._verify(result, evidence)
        return result

    def _verify(self, result: LoopResult, evidence: list[str]) -> None:
        verification = self.context.config.verification
        if not (verification.enabled and verification.factual_checks):
            return
        if not result.answer or not result.references:
            return

        result.confidence = grounding_score(result.answer, evidence)
        result.verified = result.confidence >= verification.min_confidence
        result.steps.append(
            TaskStep(
                description="Verified answer grounding",
                output=f"confidence={result.confidence} verified={result.verified}",
            )
        )
        if not result.verified:
            result.answer += (
                "\n\nWarning: parts of this answer could not be confirmed against "
                f"the retrieved documents (confidence {result.confidence:.2f} < "
                f"{verification.min_confidence})."
            )


__all__ = ["AgentLoop", "LoopResult", "SYSTEM_PROMPT", "grounding_score"]
