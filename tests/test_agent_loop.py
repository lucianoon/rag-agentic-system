"""Tests for the agent loop, tools and model selection (no LLM, no network)."""

import pytest

from rag_agent import AgenticRAG
from rag_agent.llm import ModelTurn, ScriptedModel, ToolCallRequest, build_agent_model
from rag_agent.loop import AgentLoop, ITERATION_LIMIT_ANSWER, grounding_score
from rag_agent.tools import ToolExecutor

from test_pipeline import build_context, write_docs


class AlwaysSearchingModel:
    """Stub that never produces a final answer — used to test the iteration cap."""

    mode = "stub"

    def __init__(self):
        self.calls = 0

    def create_turn(self, system, messages, tools):
        self.calls += 1
        call = ToolCallRequest(
            id=f"stub-{self.calls}", name="search_documents", input={"query": "python"}
        )
        return ModelTurn(
            tool_calls=[call],
            raw_content=[{"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}],
        )


class FabricatingModel:
    """Stub that searches once, then answers with unsupported claims."""

    mode = "stub"

    def create_turn(self, system, messages, tools):
        has_tool_result = any(
            isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
            for m in messages
        )
        if not has_tool_result:
            call = ToolCallRequest(id="fab-1", name="search_documents", input={"query": "python"})
            return ModelTurn(
                tool_calls=[call],
                raw_content=[
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
                ],
            )
        return ModelTurn(
            text="Python was invented on Mars in 1850 by sentient bread recipes."
        )


def test_scripted_agent_runs_full_tool_loop(tmp_path, force_tfidf):
    write_docs(tmp_path)
    agent = AgenticRAG(build_context(tmp_path))
    agent.initialize()

    response = agent.query("What is Python used for?")

    assert "Based on the retrieved documents" in response.answer
    assert response.references
    assert response.metadata["agent_mode"] == "scripted"
    assert response.metadata["iterations"] == 2
    assert response.metadata["tool_calls"] == 1
    descriptions = [step.description for step in response.steps]
    assert "Tool call: search_documents" in descriptions
    assert "Generated answer" in descriptions
    # The task log with loop metadata reached persistent memory.
    logs = agent.context.memory.recent()
    assert len(logs) == 1
    assert logs[0].metadata["tool_calls"] == 1


def test_loop_respects_max_iterations(tmp_path, force_tfidf):
    write_docs(tmp_path)
    context = build_context(tmp_path)
    context.ingest_if_empty()
    model = AlwaysSearchingModel()

    result = AgentLoop(context, model).run("keep searching forever")

    assert result.answer.startswith("I could not complete this task")
    assert result.iterations == context.config.agent.max_iterations
    assert model.calls == context.config.agent.max_iterations


def test_verification_flags_fabricated_answer(tmp_path, force_tfidf):
    write_docs(tmp_path)
    context = build_context(tmp_path)
    context.ingest_if_empty()

    result = AgentLoop(context, FabricatingModel()).run("history of python")

    assert result.verified is False
    assert result.confidence is not None and result.confidence < 0.55
    assert "could not be confirmed" in result.answer


def test_verification_passes_grounded_scripted_answer(tmp_path, force_tfidf):
    write_docs(tmp_path)
    context = build_context(tmp_path)
    context.ingest_if_empty()

    result = AgentLoop(context, ScriptedModel()).run("What is Python used for?")

    assert result.verified is True
    assert "could not be confirmed" not in result.answer


def test_grounding_score_extremes():
    evidence = ["Python is a programming language used for data science."]

    assert grounding_score("Python is a programming language.", evidence) == 1.0
    assert grounding_score("Cats rule the lunar economy.", evidence) == 0.0
    assert grounding_score("Anything.", []) == 0.0


def test_tool_executor_handles_unknown_tool_and_empty_history(tmp_path, force_tfidf):
    context = build_context(tmp_path)
    executor = ToolExecutor(context)

    output, references = executor.execute("launch_missiles", {})
    assert "unknown tool" in output
    assert references == []

    output, references = executor.execute("get_task_history", {})
    assert output == "No previous tasks in memory."
    assert references == []


def test_build_agent_model_selection(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert build_agent_model({}).mode == "scripted"  # auto without key
    assert build_agent_model({"provider": "scripted"}).mode == "scripted"

    with pytest.raises(ValueError, match="llm.provider"):
        build_agent_model({"provider": "openai"})
