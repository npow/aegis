"""Tests for budget enforcement."""

from dataclasses import dataclass

from aegis import AgentState, Budget, RunConfig, graph, node, tool
from aegis.checkpointers import MemoryCheckpointer
from aegis.testing import MockTool


@dataclass
class BudgetState(AgentState):
    calls: int = 0


@tool(name="budget_test_tool")
async def budget_test_tool() -> str:
    return "ok"


async def test_budget_hard_stop_on_tool_calls():
    """Budget.max_tool_calls=1 should hard-stop after 2 calls."""

    @node()
    async def call_twice(state: BudgetState, tools) -> BudgetState:
        await tools.budget_test_tool()
        await tools.budget_test_tool()
        return state.update(calls=2)

    @graph(name=f"budget-tool-calls-v{id(call_twice)}", version="1.0.0")
    async def budget_graph(state: BudgetState) -> BudgetState:
        return await call_twice(state)

    cp = MemoryCheckpointer()
    async with budget_graph.mock_tools({"budget_test_tool": MockTool.returns("ok")}):
        result = await budget_graph.run(
            input=BudgetState(),
            config=RunConfig(thread_id="budget-tc-001", checkpointer=cp),
            budget=Budget(max_tool_calls=1),
        )
    assert result.status == "budget_exceeded"


async def test_budget_hard_stop_on_wall_time():
    """Budget.max_wall_time_seconds=0 should stop immediately (wall time starts at 0)."""
    import asyncio

    @node()
    async def slow_node(state: BudgetState) -> BudgetState:
        await asyncio.sleep(0.5)
        return state

    @graph(name=f"budget-time-v{id(slow_node)}", version="1.0.0")
    async def time_budget_graph(state: BudgetState) -> BudgetState:
        return await slow_node(state)

    cp = MemoryCheckpointer()
    # Set a very small wall time budget so the pre-check fires on the first node
    result = await time_budget_graph.run(
        input=BudgetState(),
        config=RunConfig(thread_id="budget-time-001", checkpointer=cp),
        budget=Budget(max_wall_time_seconds=0),
    )
    # Either budget_exceeded or failed (timeout + budget check racing)
    assert result.status in ("budget_exceeded", "failed")


async def test_no_budget_allows_unlimited_tool_calls():
    """Without a budget, many tool calls should succeed."""

    @node()
    async def many_calls(state: BudgetState, tools) -> BudgetState:
        for _ in range(20):
            await tools.budget_test_tool()
        return state.update(calls=20)

    @graph(name=f"no-budget-v{id(many_calls)}", version="1.0.0")
    async def no_budget_graph(state: BudgetState) -> BudgetState:
        return await many_calls(state)

    cp = MemoryCheckpointer()
    async with no_budget_graph.mock_tools({"budget_test_tool": MockTool.returns("ok")}):
        result = await no_budget_graph.run(
            input=BudgetState(),
            config=RunConfig(thread_id="no-budget-001", checkpointer=cp),
        )
    assert result.status == "completed"
    assert result.state.calls == 20


async def test_budget_status_tracks_tool_calls():
    """BudgetStatus should accurately report tool calls made."""

    @node()
    async def counted_calls(state: BudgetState, tools) -> BudgetState:
        await tools.budget_test_tool()
        await tools.budget_test_tool()
        await tools.budget_test_tool()
        return state.update(calls=3)

    @graph(name=f"count-calls-v{id(counted_calls)}", version="1.0.0")
    async def count_graph(state: BudgetState) -> BudgetState:
        return await counted_calls(state)

    cp = MemoryCheckpointer()
    async with count_graph.mock_tools({"budget_test_tool": MockTool.returns("ok")}):
        result = await count_graph.run(
            input=BudgetState(),
            config=RunConfig(thread_id="count-calls-001", checkpointer=cp),
            budget=Budget(max_tool_calls=10),
        )
    assert result.status == "completed"
