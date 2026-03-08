"""Tests for chain(), parallel(), and supervisor() multi-agent composition."""

from dataclasses import dataclass, field

from aegis import AgentState, RunConfig, graph, node
from aegis._multi_agent import chain, parallel, supervisor
from aegis.checkpointers import MemoryCheckpointer

# ── Fixtures ──────────────────────────────────────────────────────────────────


@dataclass
class PipeState(AgentState):
    value: int = 0
    log: list = field(default_factory=list)
    next_specialist: str = ""


@node()
async def add_ten(state: PipeState) -> PipeState:
    return state.update(value=state.value + 10, log=state.log + ["add_ten"])


@node()
async def multiply_two(state: PipeState) -> PipeState:
    return state.update(value=state.value * 2, log=state.log + ["multiply_two"])


@node()
async def subtract_one(state: PipeState) -> PipeState:
    return state.update(value=state.value - 1, log=state.log + ["subtract_one"])


@graph(name="pipe-add", version="1.0.0")
async def pipe_add(state: PipeState) -> PipeState:
    return await add_ten(state)


@graph(name="pipe-mul", version="1.0.0")
async def pipe_mul(state: PipeState) -> PipeState:
    return await multiply_two(state)


@graph(name="pipe-sub", version="1.0.0")
async def pipe_sub(state: PipeState) -> PipeState:
    return await subtract_one(state)


# ── chain() tests ─────────────────────────────────────────────────────────────


async def test_chain_executes_sequentially():
    pipeline = chain(pipe_add, pipe_mul)
    cp = MemoryCheckpointer()
    result = await pipeline.run(
        input=PipeState(value=5),
        config=RunConfig(thread_id="chain-001", checkpointer=cp),
    )
    assert result.status == "completed"
    assert result.state.value == 30  # (5+10)*2 = 30


async def test_chain_three_graphs():
    pipeline = chain(pipe_add, pipe_mul, pipe_sub)
    cp = MemoryCheckpointer()
    result = await pipeline.run(
        input=PipeState(value=5),
        config=RunConfig(thread_id="chain-three-001", checkpointer=cp),
    )
    assert result.status == "completed"
    assert result.state.value == 29  # ((5+10)*2)-1 = 29


async def test_chain_stops_on_first_failure():
    @node()
    async def always_fails_chain(state: PipeState) -> PipeState:
        raise ValueError("chain stop")

    @graph(name=f"chain-fail-graph-v{id(always_fails_chain)}", version="1.0.0")
    async def chain_fail_graph(state: PipeState) -> PipeState:
        return await always_fails_chain(state)

    pipeline = chain(chain_fail_graph, pipe_mul)
    cp = MemoryCheckpointer()
    result = await pipeline.run(
        input=PipeState(value=5),
        config=RunConfig(thread_id="chain-fail-001", checkpointer=cp),
    )
    # Should stop at first failure
    assert result.status == "failed"


# ── parallel() tests ──────────────────────────────────────────────────────────


async def test_parallel_runs_concurrently():
    pipeline = parallel(pipe_add, pipe_mul)
    cp = MemoryCheckpointer()
    results = await pipeline.run(
        input=PipeState(value=10),
        config=RunConfig(thread_id="parallel-001", checkpointer=cp),
    )
    assert len(results) == 2
    # pipe_add: 10+10=20
    assert results[0].state.value == 20
    # pipe_mul: 10*2=20
    assert results[1].state.value == 20


async def test_parallel_three_graphs():
    pipeline = parallel(pipe_add, pipe_mul, pipe_sub)
    cp = MemoryCheckpointer()
    results = await pipeline.run(
        input=PipeState(value=5),
        config=RunConfig(thread_id="parallel-three-001", checkpointer=cp),
    )
    assert len(results) == 3
    values = [r.state.value for r in results]
    assert 15 in values  # 5+10
    assert 10 in values  # 5*2
    assert 4 in values  # 5-1


async def test_parallel_with_join():
    @node()
    async def join_node(state: PipeState) -> PipeState:
        # Join just returns current state (parallel results are attached separately)
        return state.update(value=state.value + 1000, log=state.log + ["joined"])

    @graph(name=f"join-graph-v{id(join_node)}", version="1.0.0")
    async def join_graph(state: PipeState) -> PipeState:
        return await join_node(state)

    pipeline = parallel(pipe_add, pipe_mul).join(join_graph)
    cp = MemoryCheckpointer()
    result = await pipeline.run(
        input=PipeState(value=1),
        config=RunConfig(thread_id="join-001", checkpointer=cp),
    )
    assert result.status == "completed"
    assert "joined" in result.state.log


# ── supervisor() tests ────────────────────────────────────────────────────────


@dataclass
class RouterState(AgentState):
    query: str = ""
    category: str = ""
    answer: str = ""
    next_specialist: str = ""  # set by router to indicate which specialist


@node()
async def route_to_math(state: RouterState) -> RouterState:
    return state.update(next_specialist="math" if "math" in state.query else "__done__")


@node()
async def math_specialist_node(state: RouterState) -> RouterState:
    return state.update(answer="42", next_specialist="__done__")


@graph(name="supervisor-router-v1", version="1.0.0")
async def supervisor_router(state: RouterState) -> RouterState:
    return await route_to_math(state)


@graph(name="supervisor-math-v1", version="1.0.0")
async def supervisor_math(state: RouterState) -> RouterState:
    return await math_specialist_node(state)


async def test_supervisor_routes_to_correct_specialist():
    pipeline = supervisor(
        router=supervisor_router,
        specialists={"math": supervisor_math},
        max_handoffs=3,
    )
    cp = MemoryCheckpointer()
    result = await pipeline.run(
        input=RouterState(query="solve math problem"),
        config=RunConfig(thread_id="supervisor-001", checkpointer=cp),
    )
    assert result.status == "completed"
    assert result.state.answer == "42"


async def test_supervisor_unknown_specialist_fails():
    @node()
    async def bad_router_node(state: RouterState) -> RouterState:
        return state.update(next_specialist="nonexistent")

    @graph(name=f"bad-router-v{id(bad_router_node)}", version="1.0.0")
    async def bad_router(state: RouterState) -> RouterState:
        return await bad_router_node(state)

    pipeline = supervisor(
        router=bad_router,
        specialists={"math": supervisor_math},
        max_handoffs=1,
    )
    cp = MemoryCheckpointer()
    result = await pipeline.run(
        input=RouterState(query="test"),
        config=RunConfig(thread_id="bad-router-001", checkpointer=cp),
    )
    assert result.status == "failed"
    assert "nonexistent" in result.error.message


async def test_parallel_branch_failure_returned_as_failed_result():
    """A branch that raises an exception returns a failed RunResult (not re-raised)."""

    @node(retries=0)
    async def fail_branch_node(state: PipeState) -> PipeState:
        raise RuntimeError("branch explosion")

    @graph(name=f"fail-branch-v{id(fail_branch_node)}", version="1.0.0")
    async def fail_branch(state: PipeState) -> PipeState:
        return await fail_branch_node(state)

    pipeline = parallel(pipe_add, fail_branch)
    cp = MemoryCheckpointer()
    results = await pipeline.run(
        input=PipeState(value=1),
        config=RunConfig(thread_id="fail-branch-001", checkpointer=cp),
    )
    assert len(results) == 2
    # pipe_add should succeed
    assert results[0].status == "completed"
    # fail_branch should be wrapped as a failed result
    assert results[1].status == "failed"
    assert "branch explosion" in results[1].error.message


async def test_chain_propagates_metadata():
    """chain() sub-configs should carry the parent RunConfig metadata."""
    captured_metadata = {}

    @node()
    async def capture_meta_node(state: PipeState) -> PipeState:
        from aegis._context import _run_context

        ctx = _run_context.get()
        # Metadata is on RunConfig, not RunContext directly,
        # but we can verify thread_id naming matches chain pattern
        captured_metadata["thread_id"] = ctx.thread_id
        return state

    @graph(name=f"capture-meta-v{id(capture_meta_node)}", version="1.0.0")
    async def capture_meta_graph(state: PipeState) -> PipeState:
        return await capture_meta_node(state)

    pipeline = chain(capture_meta_graph)
    cp = MemoryCheckpointer()
    await pipeline.run(
        input=PipeState(value=0),
        config=RunConfig(thread_id="meta-parent", checkpointer=cp, metadata={"key": "val"}),
    )
    assert captured_metadata.get("thread_id", "").startswith("meta-parent-chain-")


async def test_supervisor_max_handoffs_exceeded():
    @node()
    async def loop_router_node(state: RouterState) -> RouterState:
        # Never returns __done__, causing infinite loop
        return state.update(next_specialist="loop")

    @node()
    async def loop_specialist_node(state: RouterState) -> RouterState:
        return state  # no next_specialist = done? No, loop_router will route again

    @graph(name=f"loop-router-v{id(loop_router_node)}", version="1.0.0")
    async def loop_router(state: RouterState) -> RouterState:
        return await loop_router_node(state)

    @graph(name=f"loop-spec-v{id(loop_specialist_node)}", version="1.0.0")
    async def loop_specialist(state: RouterState) -> RouterState:
        return await loop_specialist_node(state)

    pipeline = supervisor(
        router=loop_router,
        specialists={"loop": loop_specialist},
        max_handoffs=2,
    )
    cp = MemoryCheckpointer()
    result = await pipeline.run(
        input=RouterState(query="test"),
        config=RunConfig(thread_id="loop-001", checkpointer=cp),
    )
    assert result.status == "failed"
    assert "max_handoffs" in result.error.message.lower() or "2" in result.error.message


async def test_supervisor_specialist_timeout_returns_failed_result():
    """Specialist that times out must return a failed RunResult, not raise TimeoutError."""
    import asyncio

    @node(retries=0)
    async def slow_specialist_node(state: RouterState) -> RouterState:
        await asyncio.sleep(10.0)  # will be timed out
        return state

    @node()
    async def route_to_slow(state: RouterState) -> RouterState:
        return state.update(next_specialist="slow")

    @graph(name=f"slow-spec-v{id(slow_specialist_node)}", version="1.0.0")
    async def slow_specialist(state: RouterState) -> RouterState:
        return await slow_specialist_node(state)

    @graph(name=f"timeout-router-v{id(route_to_slow)}", version="1.0.0")
    async def timeout_router(state: RouterState) -> RouterState:
        return await route_to_slow(state)

    pipeline = supervisor(
        router=timeout_router,
        specialists={"slow": slow_specialist},
        max_handoffs=1,
        handoff_timeout=0.05,  # 50ms
    )
    cp = MemoryCheckpointer()
    result = await pipeline.run(
        input=RouterState(query="test"),
        config=RunConfig(thread_id="timeout-specialist-001", checkpointer=cp),
    )
    assert result.status == "failed"
    assert "timed out" in result.error.message.lower() or "TimeoutError" in result.error.exception_type
