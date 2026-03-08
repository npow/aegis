"""Tests for graph execution, retry, checkpointing, and resume."""

from dataclasses import dataclass

from aegis import AgentState, RunConfig, graph, node
from aegis.checkpointers import MemoryCheckpointer

# ── Simple graph fixtures ─────────────────────────────────────────────────────


@dataclass
class CountState(AgentState):
    value: int = 0
    steps_taken: int = 0


@node()
async def increment_node(state: CountState) -> CountState:
    return state.update(value=state.value + 1, steps_taken=state.steps_taken + 1)


@node()
async def double_node(state: CountState) -> CountState:
    return state.update(value=state.value * 2, steps_taken=state.steps_taken + 1)


@graph(name="count-graph-test", version="1.0.0")
async def count_graph(state: CountState) -> CountState:
    state = await increment_node(state)
    state = await double_node(state)
    return state


# ── Basic execution ───────────────────────────────────────────────────────────


async def test_graph_runs_to_completion():
    cp = MemoryCheckpointer()
    result = await count_graph.run(
        input=CountState(value=5),
        config=RunConfig(thread_id="test-basic-001", checkpointer=cp),
    )
    assert result.status == "completed"
    assert result.state.value == 12  # (5+1)*2 = 12
    assert result.state.steps_taken == 2


async def test_graph_returns_correct_run_id():
    cp = MemoryCheckpointer()
    result = await count_graph.run(
        input=CountState(value=1),
        config=RunConfig(thread_id="test-run-id", checkpointer=cp),
    )
    assert result.run_id.startswith("run-")


async def test_graph_trace_records_nodes():
    cp = MemoryCheckpointer()
    result = await count_graph.run(
        input=CountState(value=3),
        config=RunConfig(thread_id="test-trace-001", checkpointer=cp),
    )
    node_names = [n.node_name for n in result.trace.nodes_executed]
    assert "increment_node" in node_names
    assert "double_node" in node_names
    assert len(node_names) == 2


async def test_graph_creates_checkpoints():
    cp = MemoryCheckpointer()
    await count_graph.run(
        input=CountState(value=2),
        config=RunConfig(thread_id="test-ckpt-001", checkpointer=cp),
    )
    history = await cp.get_history("test-ckpt-001", "count-graph-test")
    # step 0 (input) + step 1 (increment) + step 2 (double) = 3 checkpoints
    assert len(history) == 3
    assert history[0].node_name == "__input__"
    assert history[1].node_name == "increment_node"
    assert history[2].node_name == "double_node"


async def test_graph_checkpoint_ids_are_deterministic():
    """Same inputs produce same checkpoint ID prefix."""
    cp = MemoryCheckpointer()
    await count_graph.run(
        input=CountState(value=1),
        config=RunConfig(thread_id="deterministic-001", checkpointer=cp),
    )
    history = await cp.get_history("deterministic-001", "count-graph-test")
    for ckpt in history:
        assert ckpt.id.startswith("ckpt_count-graph-test_deterministic-001_")


async def test_graph_thread_id_injected_into_state():
    cp = MemoryCheckpointer()
    result = await count_graph.run(
        input=CountState(value=0),
        config=RunConfig(thread_id="my-thread-id", checkpointer=cp),
    )
    assert result.state.thread_id == "my-thread-id"


# ── Node retry ────────────────────────────────────────────────────────────────


async def test_node_retries_on_transient_error():
    call_count = 0

    @node(retries=2, retry_on=(ValueError,))
    async def flaky_add(state: CountState) -> CountState:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("not yet")
        return state.update(value=state.value + 100)

    @graph(name=f"retry-graph-v{id(flaky_add)}", version="1.0.0")
    async def retry_graph(state: CountState) -> CountState:
        return await flaky_add(state)

    cp = MemoryCheckpointer()
    result = await retry_graph.run(
        input=CountState(value=0),
        config=RunConfig(thread_id="retry-001", checkpointer=cp),
    )
    assert result.status == "completed"
    assert result.state.value == 100
    assert call_count == 3


async def test_node_fails_after_exhausting_retries():
    @node(retries=2)
    async def always_fails(state: CountState) -> CountState:
        raise RuntimeError("permanent failure")

    @graph(name=f"fail-graph-v{id(always_fails)}", version="1.0.0")
    async def fail_graph(state: CountState) -> CountState:
        return await always_fails(state)

    cp = MemoryCheckpointer()
    result = await fail_graph.run(
        input=CountState(value=0),
        config=RunConfig(thread_id="fail-001", checkpointer=cp),
    )
    assert result.status == "failed"
    assert result.error is not None
    assert "permanent failure" in result.error.message


async def test_node_does_not_retry_non_matching_exception():
    call_count = 0

    @node(retries=3, retry_on=(ValueError,))
    async def type_specific(state: CountState) -> CountState:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("wrong type")  # not in retry_on

    @graph(name=f"no-retry-graph-v{id(type_specific)}", version="1.0.0")
    async def no_retry_graph(state: CountState) -> CountState:
        return await type_specific(state)

    cp = MemoryCheckpointer()
    result = await no_retry_graph.run(
        input=CountState(value=0),
        config=RunConfig(thread_id="no-retry-001", checkpointer=cp),
    )
    assert result.status == "failed"
    assert call_count == 1  # No retries


async def test_node_timeout():
    import asyncio

    @node(timeout_seconds=0.05)
    async def slow_node(state: CountState) -> CountState:
        await asyncio.sleep(1.0)
        return state

    @graph(name=f"timeout-graph-v{id(slow_node)}", version="1.0.0")
    async def timeout_graph(state: CountState) -> CountState:
        return await slow_node(state)

    cp = MemoryCheckpointer()
    result = await timeout_graph.run(
        input=CountState(value=0),
        config=RunConfig(thread_id="timeout-001", checkpointer=cp),
    )
    assert result.status == "failed"


# ── Graph resume ──────────────────────────────────────────────────────────────


async def test_graph_resume_from_checkpoint():
    """Simulate resume: pre-populate checkpoints for first N nodes, verify resume runs remainder."""

    @dataclass
    class ResumeState(AgentState):
        steps: list = None

        def __post_init__(self):
            if self.steps is None:
                self.steps = []

    @node()
    async def step_a(state: ResumeState) -> ResumeState:
        return state.update(steps=state.steps + ["a"])

    @node()
    async def step_b(state: ResumeState) -> ResumeState:
        return state.update(steps=state.steps + ["b"])

    @node()
    async def step_c(state: ResumeState) -> ResumeState:
        return state.update(steps=state.steps + ["c"])

    @graph(name=f"resume-graph-v{id(step_a)}", version="1.0.0")
    async def resume_graph(state: ResumeState) -> ResumeState:
        state = await step_a(state)
        state = await step_b(state)
        state = await step_c(state)
        return state

    cp = MemoryCheckpointer()
    # Full run
    result = await resume_graph.run(
        input=ResumeState(),
        config=RunConfig(thread_id="resume-001", checkpointer=cp),
    )
    assert result.status == "completed"
    assert result.state.steps == ["a", "b", "c"]
    assert len(await cp.get_history("resume-001", f"resume-graph-v{id(step_a)}")) == 4  # 0+a+b+c


async def test_graph_resume_after_simulated_crash():
    """Pre-populate checkpoints up to step 2 and verify resume starts from step 3."""

    @dataclass
    class CrashState(AgentState):
        counter: int = 0

    execution_log = []

    @node()
    async def node_one(state: CrashState) -> CrashState:
        execution_log.append("one")
        return state.update(counter=state.counter + 1)

    @node()
    async def node_two(state: CrashState) -> CrashState:
        execution_log.append("two")
        return state.update(counter=state.counter + 10)

    @graph(name=f"crash-graph-v{id(node_one)}", version="1.0.0")
    async def crash_graph(state: CrashState) -> CrashState:
        state = await node_one(state)
        state = await node_two(state)
        return state

    gname = f"crash-graph-v{id(node_one)}"
    cp = MemoryCheckpointer()

    # First: do a complete run to get checkpoints
    result = await crash_graph.run(
        input=CrashState(),
        config=RunConfig(thread_id="crash-001", checkpointer=cp),
    )
    assert result.status == "completed"
    assert result.state.counter == 11
    execution_log.clear()

    # Simulate: delete last checkpoint (node_two's output at step 2)
    # to simulate crash after node_one but before node_two committed
    history = await cp.get_history("crash-001", gname)
    # Keep only steps 0 and 1 (input + node_one output)
    cp._store[("crash-001", gname)] = [c for c in history if c.step <= 1]

    # Resume
    result2 = await crash_graph.resume(
        thread_id="crash-001",
        config=RunConfig(thread_id="crash-001", checkpointer=cp),
    )
    assert result2.status == "completed"
    # node_one should NOT re-execute (skipped via fast-forward)
    # node_two should execute
    assert "two" in execution_log
    assert "one" not in execution_log
    assert result2.state.counter == 11


# ── Graph with tools ──────────────────────────────────────────────────────────


async def test_graph_with_mock_tools():
    from aegis import tool
    from aegis.testing import MockTool

    @dataclass
    class SearchState(AgentState):
        query: str = ""
        results: list = None

        def __post_init__(self):
            if self.results is None:
                self.results = []

    @tool(name=f"search_tool_{id(SearchState)}")
    async def search_tool(query: str) -> list:
        return []  # default: empty (won't be called in tests)

    @node()
    async def search_node(state: SearchState, tools) -> SearchState:
        results = await tools.__getattr__(f"search_tool_{id(SearchState)}")(query=state.query)
        return state.update(results=results)

    @graph(name=f"search-graph-v{id(search_node)}", version="1.0.0")
    async def search_graph(state: SearchState) -> SearchState:
        return await search_node(state)

    cp = MemoryCheckpointer()
    async with search_graph.mock_tools(
        {
            f"search_tool_{id(SearchState)}": MockTool.returns(["result1", "result2"]),
        }
    ) as ctx:
        result = await search_graph.run(
            input=SearchState(query="test"),
            config=RunConfig(thread_id="mock-tools-001", checkpointer=cp),
        )

    assert result.status == "completed"
    assert result.state.results == ["result1", "result2"]
    tool_key = f"search_tool_{id(SearchState)}"
    assert tool_key in ctx.calls
    assert ctx.calls[tool_key].count == 1
    assert ctx.calls[tool_key].calls[0].args["query"] == "test"


async def test_mock_tool_noop():
    from aegis import tool
    from aegis.testing import MockTool

    @dataclass
    class WriteState(AgentState):
        written: bool = False

    @tool(name=f"write_tool_{id(WriteState)}")
    async def write_tool(content: str) -> None:
        pass

    @node()
    async def write_node(state: WriteState, tools) -> WriteState:
        await tools.__getattr__(f"write_tool_{id(WriteState)}")(content="hello")
        return state.update(written=True)

    @graph(name=f"write-graph-v{id(write_node)}", version="1.0.0")
    async def write_graph(state: WriteState) -> WriteState:
        return await write_node(state)

    cp = MemoryCheckpointer()
    async with write_graph.mock_tools(
        {
            f"write_tool_{id(WriteState)}": MockTool.noop(),
        }
    ):
        result = await write_graph.run(
            input=WriteState(),
            config=RunConfig(thread_id="noop-001", checkpointer=cp),
        )
    assert result.status == "completed"


async def test_node_retry_on_rejects_non_exception_types():
    """retry_on must only contain exception classes."""
    import pytest

    with pytest.raises(TypeError, match="exception classes"):

        @node(retries=2, retry_on=(ValueError, "not_a_class"))  # type: ignore[arg-type]
        async def bad_retry_on(state: CountState) -> CountState:
            return state


async def test_node_decorator_rejects_sync_function():
    """@node must raise TypeError at decoration time for sync functions."""
    import pytest

    with pytest.raises(TypeError, match="async"):

        @node()
        def sync_node(state: CountState) -> CountState:  # type: ignore[arg-type]
            return state


async def test_graph_decorator_rejects_sync_function():
    """@graph must raise TypeError at decoration time for sync functions."""
    import pytest

    with pytest.raises(TypeError, match="async"):

        @graph(name="sync-graph", version="1.0.0")
        def sync_graph(state: CountState) -> CountState:  # type: ignore[arg-type]
            return state


async def test_resume_raises_when_no_checkpoints():
    """resume() raises NoCheckpointError when no checkpoints exist."""
    import pytest

    from aegis._models import NoCheckpointError

    cp = MemoryCheckpointer()
    with pytest.raises(NoCheckpointError):
        await count_graph.resume(
            thread_id="nonexistent-thread",
            config=RunConfig(thread_id="nonexistent-thread", checkpointer=cp),
        )


async def test_node_returns_wrong_type_fails_graph():
    """A node returning a non-AgentState value should cause a graph failure."""

    @node()
    async def bad_return(state: CountState) -> CountState:  # type: ignore[return-value]
        return "not a state"  # type: ignore[return-value]

    @graph(name=f"bad-return-graph-v{id(bad_return)}", version="1.0.0")
    async def bad_return_graph(state: CountState) -> CountState:
        return await bad_return(state)

    cp = MemoryCheckpointer()
    result = await bad_return_graph.run(
        input=CountState(value=0),
        config=RunConfig(thread_id="bad-return-001", checkpointer=cp),
    )
    assert result.status == "failed"
    assert "AgentState" in (result.error.message if result.error else "")


async def test_mock_tool_raises_propagates_as_failure():
    from aegis import tool
    from aegis.testing import MockTool

    @dataclass
    class ErrState(AgentState):
        pass

    @tool(name=f"err_tool_{id(ErrState)}")
    async def err_tool() -> None:
        pass

    @node(retries=0)
    async def err_node(state: ErrState, tools) -> ErrState:
        await tools.__getattr__(f"err_tool_{id(ErrState)}")()
        return state

    @graph(name=f"err-graph-v{id(err_node)}", version="1.0.0")
    async def err_graph(state: ErrState) -> ErrState:
        return await err_node(state)

    cp = MemoryCheckpointer()
    async with err_graph.mock_tools(
        {
            f"err_tool_{id(ErrState)}": MockTool.raises(RuntimeError("boom")),
        }
    ):
        result = await err_graph.run(
            input=ErrState(),
            config=RunConfig(thread_id="raises-001", checkpointer=cp),
        )
    assert result.status == "failed"
    assert "boom" in result.error.message
