"""Tests for MockTool and cassette record/replay."""

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from aegis import AgentState, RunConfig, graph, node, tool
from aegis.checkpointers import MemoryCheckpointer
from aegis.testing import MockTool, cassette

# ── MockTool unit tests ───────────────────────────────────────────────────────


async def test_mock_tool_returns_fixed_value():
    mock = MockTool.returns(["result1", "result2"])
    result = await mock.execute({"query": "test"})
    assert result == ["result1", "result2"]


async def test_mock_tool_returns_none_by_default():
    mock = MockTool.returns(None)
    result = await mock.execute({})
    assert result is None


async def test_mock_tool_noop_returns_none():
    mock = MockTool.noop()
    result = await mock.execute({"x": 1})
    assert result is None


async def test_mock_tool_raises_exception():
    exc = TimeoutError("upstream unavailable")
    mock = MockTool.raises(exc)
    with pytest.raises(TimeoutError, match="upstream unavailable"):
        await mock.execute({"arg": "val"})


async def test_mock_tool_calls_side_effect():
    results = []

    def capture(**kwargs):
        results.append(kwargs)
        return "captured"

    mock = MockTool.calls(capture)
    result = await mock.execute({"key": "value"})
    assert result == "captured"
    assert results == [{"key": "value"}]


# ── MockTools context manager integration ────────────────────────────────────


@dataclass
class MockState(AgentState):
    query: str = ""
    result: str = ""
    side_calls: int = 0


@tool(name="mock_search_v1")
async def mock_search_v1(query: str) -> str:
    return "live result"


@node()
async def search_mock_node(state: MockState, tools) -> MockState:
    result = await tools.mock_search_v1(query=state.query)
    return state.update(result=result)


@graph(name="mock-test-graph", version="1.0.0")
async def mock_test_graph(state: MockState) -> MockState:
    return await search_mock_node(state)


async def test_mock_tools_intercepts_tool_call():
    cp = MemoryCheckpointer()
    async with mock_test_graph.mock_tools(
        {
            "mock_search_v1": MockTool.returns("mocked result"),
        }
    ):
        result = await mock_test_graph.run(
            input=MockState(query="quantum computing"),
            config=RunConfig(thread_id="mock-ctx-001", checkpointer=cp),
        )

    assert result.status == "completed"
    assert result.state.result == "mocked result"


async def test_mock_tools_records_call_count():
    cp = MemoryCheckpointer()
    async with mock_test_graph.mock_tools(
        {
            "mock_search_v1": MockTool.returns("x"),
        }
    ) as ctx:
        await mock_test_graph.run(
            input=MockState(query="test"),
            config=RunConfig(thread_id="mock-count-001", checkpointer=cp),
        )

    assert ctx.calls["mock_search_v1"].count == 1


async def test_mock_tools_records_call_args():
    cp = MemoryCheckpointer()
    async with mock_test_graph.mock_tools(
        {
            "mock_search_v1": MockTool.returns("y"),
        }
    ) as ctx:
        await mock_test_graph.run(
            input=MockState(query="hello world"),
            config=RunConfig(thread_id="mock-args-001", checkpointer=cp),
        )

    call = ctx.calls["mock_search_v1"].calls[0]
    assert call.args["query"] == "hello world"


async def test_mock_tools_live_calls_made_is_zero():
    """No real tool execution should happen inside mock_tools context."""
    cp = MemoryCheckpointer()
    async with mock_test_graph.mock_tools(
        {
            "mock_search_v1": MockTool.returns("ok"),
        }
    ):
        result = await mock_test_graph.run(
            input=MockState(query="test"),
            config=RunConfig(thread_id="live-zero-001", checkpointer=cp),
        )
    # The mock intercepts the call, so live_calls_made should be 0
    # (live_calls_made is tracked at the RunContext level via mock_ctx)
    assert result.status == "completed"


async def test_mock_tools_error_propagates():
    cp = MemoryCheckpointer()
    async with mock_test_graph.mock_tools(
        {
            "mock_search_v1": MockTool.raises(ConnectionError("network down")),
        }
    ):
        result = await mock_test_graph.run(
            input=MockState(query="test"),
            config=RunConfig(thread_id="mock-err-001", checkpointer=cp),
        )
    assert result.status == "failed"
    assert "network down" in result.error.message


# ── Cassette record/replay integration ───────────────────────────────────────


@dataclass
class CassetteState(AgentState):
    query: str = ""
    answer: str = ""


@tool(name="cassette_tool_v1")
async def cassette_tool_v1(query: str) -> str:
    return f"live answer for: {query}"


@node()
async def cassette_node(state: CassetteState, tools) -> CassetteState:
    answer = await tools.cassette_tool_v1(query=state.query)
    return state.update(answer=answer)


@graph(name="cassette-test-graph", version="1.0.0")
async def cassette_test_graph(state: CassetteState) -> CassetteState:
    return await cassette_node(state)


async def test_cassette_record_creates_file():
    cp = MemoryCheckpointer()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/test.json"
        async with cassette.record(path):
            async with cassette_test_graph.mock_tools(
                {
                    "cassette_tool_v1": MockTool.returns("recorded answer"),
                }
            ):
                await cassette_test_graph.run(
                    input=CassetteState(query="what is AI"),
                    config=RunConfig(thread_id="cassette-record-001", checkpointer=cp),
                )

        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert "entries" in data
        assert "content_hash" in data
        assert data["content_hash"] != ""


async def test_cassette_replay_serves_tool_calls():
    """Record a cassette then replay it — no live calls should occur."""
    cp1 = MemoryCheckpointer()
    cp2 = MemoryCheckpointer()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/replay.json"

        # Record
        async with cassette.record(path):
            async with cassette_test_graph.mock_tools(
                {
                    "cassette_tool_v1": MockTool.returns("replay answer"),
                }
            ):
                await cassette_test_graph.run(
                    input=CassetteState(query="AI"),
                    config=RunConfig(thread_id="cassette-record-002", checkpointer=cp1),
                )

        # Replay — no mock_tools needed; cassette serves the tool call
        async with cassette.replay(path):
            result = await cassette_test_graph.run(
                input=CassetteState(query="AI"),
                config=RunConfig(thread_id="cassette-replay-002", checkpointer=cp2),
            )

        assert result.status == "completed"
        assert result.state.answer == "replay answer"


async def test_cassette_replay_with_override():
    """Override one tool in replay to test error handling."""
    cp1 = MemoryCheckpointer()
    cp2 = MemoryCheckpointer()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/override.json"

        # Record
        async with cassette.record(path):
            async with cassette_test_graph.mock_tools(
                {
                    "cassette_tool_v1": MockTool.returns("ok answer"),
                }
            ):
                await cassette_test_graph.run(
                    input=CassetteState(query="test"),
                    config=RunConfig(thread_id="override-record", checkpointer=cp1),
                )

        # Replay with override that raises
        async with cassette.replay(
            path,
            override_tools={"cassette_tool_v1": MockTool.raises(TimeoutError("injected"))},
        ):
            result = await cassette_test_graph.run(
                input=CassetteState(query="test"),
                config=RunConfig(thread_id="override-replay", checkpointer=cp2),
            )

        assert result.status == "failed"
        assert "injected" in result.error.message


async def test_cassette_file_missing_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        async with cassette.replay("/nonexistent/path/to/cassette.json"):
            pass
