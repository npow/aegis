"""Targeted tests to close coverage gaps across eval, runtime, budget, and http_intercept."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from aegis import AgentState, Budget, RunConfig, graph, node, tool
from aegis.checkpointers import MemoryCheckpointer
from aegis.testing import MockTool

# ── Shared fixtures ────────────────────────────────────────────────────────────


@dataclass
class CovState(AgentState):
    value: int = 0
    calls: int = 0


@tool(name="cov_tool")
async def cov_tool() -> str:
    return "ok"


@node()
async def cov_node(state: CovState, tools) -> CovState:
    await tools.cov_tool()
    return state.update(value=state.value + 1, calls=state.calls + 1)


@graph(name="cov-graph", version="1.0.0")
async def cov_graph(state: CovState) -> CovState:
    return await cov_node(state)


# ── eval/_assertions.py coverage ──────────────────────────────────────────────


def _make_trace(tool_calls=None):
    from aegis._models import NodeTrace, RunTrace

    tcs = tool_calls or []
    node_trace = NodeTrace(
        node_name="n",
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        input_state={},
        output_state={},
        tool_calls=tcs,
        status="completed",
    )
    return RunTrace(
        run_id="r1",
        thread_id="t1",
        graph_name="g",
        graph_version="1.0",
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        status="completed",
        nodes_executed=[node_trace],
    )


def _make_tool_call(name="my_tool", **args):
    from aegis._models import ToolCall

    return ToolCall(
        call_id="c1",
        tool_name=name,
        args=args,
        result=None,
        error=None,
        latency_ms=1,
        timestamp=datetime.utcnow(),
        node_name="n",
    )


def test_assertion_tool_call_called_false_but_was_called():
    """called=False but the tool was called → fail."""
    from aegis._models import ToolCallAssertion
    from aegis.eval._assertions import evaluate_assertion

    trace = _make_trace([_make_tool_call("my_tool")])
    assertion = ToolCallAssertion(description="d", tool_name="my_tool", called=False)
    state = CovState()
    passed, msg = evaluate_assertion(assertion, state, trace)
    assert not passed
    assert "1 time" in msg or "called" in msg.lower()


def test_assertion_tool_call_min_times_not_reached():
    """Tool called once but min_times=3 → fail."""
    from aegis._models import ToolCallAssertion
    from aegis.eval._assertions import evaluate_assertion

    trace = _make_trace([_make_tool_call("my_tool")])
    assertion = ToolCallAssertion(description="d", tool_name="my_tool", called=True, min_times=3)
    passed, msg = evaluate_assertion(assertion, CovState(), trace)
    assert not passed
    assert "1" in msg


def test_assertion_tool_call_args_match_fails():
    """args_match specified but no call matches → fail with actual args."""
    from aegis._models import ToolCallAssertion
    from aegis.eval._assertions import evaluate_assertion

    tc = _make_tool_call("my_tool", query="hello")
    trace = _make_trace([tc])
    assertion = ToolCallAssertion(
        description="d",
        tool_name="my_tool",
        called=True,
        args_match={"query": "world"},  # won't match "hello"
    )
    passed, msg = evaluate_assertion(assertion, CovState(), trace)
    assert not passed
    assert "args_match" in msg or "world" in msg or "hello" in msg


def test_assertion_tool_call_args_match_passes():
    """args_match where at least one call satisfies all expected args → pass."""
    from aegis._models import ToolCallAssertion
    from aegis.eval._assertions import evaluate_assertion

    tc = _make_tool_call("my_tool", query="world")
    trace = _make_trace([tc])
    assertion = ToolCallAssertion(
        description="d",
        tool_name="my_tool",
        called=True,
        args_match={"query": "world"},
    )
    passed, _ = evaluate_assertion(assertion, CovState(), trace)
    assert passed


def test_assertion_unknown_type():
    """An unrecognized assertion type returns (False, error message)."""
    from aegis._models import EvalAssertion
    from aegis.eval._assertions import evaluate_assertion

    trace = _make_trace()
    assertion = EvalAssertion(description="mystery")
    passed, msg = evaluate_assertion(assertion, CovState(), trace)
    assert not passed
    assert "Unknown" in msg


def test_assertion_none_trace_for_trace_snapshot():
    """TraceSnapshotAssertion with trace=None → False."""
    from aegis._models import TraceSnapshotAssertion
    from aegis.eval._assertions import evaluate_assertion

    assertion = TraceSnapshotAssertion(description="snap", golden_trace_path="/tmp/golden.json")
    passed, msg = evaluate_assertion(assertion, CovState(), trace=None)
    assert not passed
    assert "None" in msg or "trace" in msg.lower()


def test_assertion_none_trace_for_tool_call():
    """ToolCallAssertion with trace=None → False."""
    from aegis._models import ToolCallAssertion
    from aegis.eval._assertions import evaluate_assertion

    assertion = ToolCallAssertion(description="d", tool_name="my_tool")
    passed, msg = evaluate_assertion(assertion, CovState(), trace=None)
    assert not passed
    assert "None" in msg or "trace" in msg.lower()


def test_assertion_schema_predicate_raises():
    """SchemaAssertion where predicate raises → False with error message."""
    from aegis._models import SchemaAssertion
    from aegis.eval._assertions import evaluate_assertion

    trace = _make_trace()

    def boom(s):
        raise AttributeError("no such field")

    assertion = SchemaAssertion(description="d", predicate=boom)
    passed, msg = evaluate_assertion(assertion, CovState(), trace)
    assert not passed
    assert "AttributeError" in msg or "raised" in msg.lower()


def test_trace_snapshot_assertion_diverged_sequence():
    """Golden trace has different tool sequence → fail with diff message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from aegis._models import TraceSnapshotAssertion
        from aegis.eval._assertions import evaluate_assertion

        golden_path = f"{tmpdir}/golden.json"
        # Write a golden trace with tool "tool_a"
        golden = [{"tool_name": "tool_a", "node_name": "n", "args": {}}]
        Path(golden_path).write_text(json.dumps(golden, indent=2))

        # Actual trace has "tool_b"
        tc = _make_tool_call("tool_b")
        trace = _make_trace([tc])
        assertion = TraceSnapshotAssertion(description="d", golden_trace_path=golden_path)
        passed, msg = evaluate_assertion(assertion, CovState(), trace)
        assert not passed
        assert "diverged" in msg.lower() or "tool_a" in msg or "tool_b" in msg


def test_trace_snapshot_assertion_args_diverged():
    """Same tool sequence, different args → fail with args diverged message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from aegis._models import TraceSnapshotAssertion
        from aegis.eval._assertions import evaluate_assertion

        golden_path = f"{tmpdir}/golden.json"
        golden = [{"tool_name": "tool_a", "node_name": "n", "args": {"x": 1}}]
        Path(golden_path).write_text(json.dumps(golden, indent=2))

        tc = _make_tool_call("tool_a", x=2)  # same name, different arg
        trace = _make_trace([tc])
        assertion = TraceSnapshotAssertion(
            description="d",
            golden_trace_path=golden_path,
            normalize_fields=[],  # don't strip any fields so args are compared
        )
        passed, msg = evaluate_assertion(assertion, CovState(), trace)
        assert not passed
        assert "diverged" in msg.lower() or "args" in msg.lower()


# ── eval/_suite.py coverage ───────────────────────────────────────────────────


async def test_eval_suite_status_mismatch():
    """EvalCase.expected_status='failed' but graph completed → fail."""
    from aegis.eval import EvalCase, EvalSuite, SchemaAssertion

    suite = EvalSuite(
        name="status-mismatch-suite",
        graph=cov_graph,
        cases=[
            EvalCase(
                id="mismatch",
                input=CovState(),
                assertions=[SchemaAssertion(predicate=lambda s: True, description="ok")],
                expected_status="failed",  # graph will complete, not fail
            ),
        ],
        pass_rate_gate=1.0,
    )
    async with cov_graph.mock_tools({"cov_tool": MockTool.returns("ok")}):
        results = await suite.run(base_thread_id="status-mismatch")
    assert results.pass_rate == 0.0
    assert not results.gate_passed


async def test_eval_suite_graph_raises_exception():
    """Graph raises unexpected exception → case fails gracefully."""
    from aegis.eval import EvalCase, EvalSuite, SchemaAssertion

    @node(retries=0)
    async def explode(state: CovState) -> CovState:
        raise RuntimeError("kaboom!")

    @graph(name=f"explode-graph-v{id(explode)}", version="1.0.0")
    async def explode_graph(state: CovState) -> CovState:
        return await explode(state)

    suite = EvalSuite(
        name="exception-suite",
        graph=explode_graph,
        cases=[
            EvalCase(
                id="exc-case",
                input=CovState(),
                assertions=[SchemaAssertion(predicate=lambda s: True, description="ok")],
            ),
        ],
        pass_rate_gate=0.0,
    )
    results = await suite.run(base_thread_id="exc-run")
    # Graph fails (returns RunResult with status=failed), not raises
    # so we just check the suite ran without crashing
    assert results.total_cases == 1


async def test_eval_suite_graph_run_raises_directly():
    """EvalSuite._run_case handles a graph whose .run() raises (not just returns failed)."""
    from aegis.eval import EvalCase, EvalSuite, SchemaAssertion

    class _RaisingGraph:
        """Mock graph object that raises directly from .run()."""

        async def run(self, input, config, **kwargs):
            raise ValueError("direct graph crash")

    suite = EvalSuite(
        name="direct-raise-suite",
        graph=_RaisingGraph(),
        cases=[
            EvalCase(
                id="raise-case",
                input=CovState(),
                assertions=[SchemaAssertion(predicate=lambda s: True, description="ok")],
            ),
        ],
        pass_rate_gate=0.0,
    )
    results = await suite.run(base_thread_id="direct-raise")
    assert results.total_cases == 1
    assert results.passed_cases == 0
    result = results.case_results[0]
    assert not result.passed
    assert result.trace is None
    assert any("ValueError" in msg for _, ok, msg in result.assertion_results if not ok)


async def test_eval_suite_no_cassette_counts_live_calls():
    """Without a cassette, non-mocked tool calls are counted as live_calls_made."""
    from aegis.eval import EvalCase, EvalSuite, SchemaAssertion, ToolCallAssertion

    suite = EvalSuite(
        name="live-calls-suite",
        graph=cov_graph,
        cases=[
            EvalCase(
                id="live",
                input=CovState(),
                assertions=[
                    ToolCallAssertion(description="cov_tool called", tool_name="cov_tool"),
                    SchemaAssertion(predicate=lambda s: s.value == 1, description="value incremented"),
                ],
                cassette=None,  # no cassette — live path
            ),
        ],
    )
    async with cov_graph.mock_tools({"cov_tool": MockTool.returns("ok")}):
        results = await suite.run(base_thread_id="live-calls")
    assert results.total_cases == 1
    # cov_tool was mocked, so was_mocked=True → live_calls_made should be 0
    assert results.case_results[0].live_calls_made == 0


# ── _runtime.py: streaming ─────────────────────────────────────────────────────


async def test_graph_stream_yields_events():
    """stream() yields one event per node completion."""
    from aegis._runtime import GraphEvent

    @dataclass
    class StreamState(AgentState):
        step: int = 0

    @node()
    async def s1(state: StreamState) -> StreamState:
        return state.update(step=1)

    @node()
    async def s2(state: StreamState) -> StreamState:
        return state.update(step=2)

    @graph(name=f"stream-graph-v{id(s1)}", version="1.0.0")
    async def stream_graph(state: StreamState) -> StreamState:
        state = await s1(state)
        state = await s2(state)
        return state

    cp = MemoryCheckpointer()
    events = []
    async for event in stream_graph.stream(
        input=StreamState(),
        config=RunConfig(thread_id="stream-001", checkpointer=cp),
    ):
        events.append(event)

    assert len(events) == 2
    assert events[0].node_name == "s1"
    assert events[1].node_name == "s2"
    assert all(isinstance(e, GraphEvent) for e in events)
    assert all(e.type == "node_completed" for e in events)


async def test_graph_stream_early_exit():
    """Consumer breaking out of the stream loop does not hang."""

    @dataclass
    class SState(AgentState):
        x: int = 0

    @node()
    async def n1(state: SState) -> SState:
        return state.update(x=1)

    @node()
    async def n2(state: SState) -> SState:
        return state.update(x=2)

    @graph(name=f"stream-exit-graph-v{id(n1)}", version="1.0.0")
    async def stream_exit_graph(state: SState) -> SState:
        state = await n1(state)
        state = await n2(state)
        return state

    cp = MemoryCheckpointer()
    events = []
    async for event in stream_exit_graph.stream(
        input=SState(),
        config=RunConfig(thread_id="stream-exit-001", checkpointer=cp),
    ):
        events.append(event)
        break  # early exit — should not deadlock

    assert len(events) == 1


# ── _runtime.py: fork ─────────────────────────────────────────────────────────


async def test_graph_fork_replays_from_checkpoint():
    """fork() replays from a specified checkpoint into a new thread.

    Uses CovState (module-level) so _infer_state_type can resolve the annotation.
    """

    @node()
    async def fincr(state: CovState) -> CovState:
        return state.update(value=state.value + 1)

    @node()
    async def fdouble(state: CovState) -> CovState:
        return state.update(value=state.value * 2)

    @graph(name=f"fork-cov-graph-v{id(fincr)}", version="1.0.0")
    async def fork_cov_graph(state: CovState) -> CovState:
        state = await fincr(state)
        state = await fdouble(state)
        return state

    cp = MemoryCheckpointer()
    await fork_cov_graph.run(
        input=CovState(value=3),
        config=RunConfig(thread_id="fork-src", checkpointer=cp),
    )
    gname = f"fork-cov-graph-v{id(fincr)}"
    history = await cp.get_history("fork-src", gname)
    # Fork from step 1 (after fincr — value=4)
    fork_ckpt = history[1]

    fork_result = await fork_cov_graph.fork(
        thread_id="fork-src",
        checkpoint_id=fork_ckpt.id,
        new_thread_id="fork-new",
        config=RunConfig(thread_id="fork-src", checkpointer=cp),
    )
    assert fork_result.status == "completed"
    # fincr already ran (value=4), fdouble runs → 4*2=8
    assert fork_result.state.value == 8


async def test_graph_fork_missing_checkpoint_raises():
    """fork() raises NoCheckpointError for non-existent checkpoint_id."""
    from aegis._models import NoCheckpointError

    @dataclass
    class FState2(AgentState):
        x: int = 0

    @node()
    async def fnode2(state: FState2) -> FState2:
        return state

    @graph(name=f"fork-missing-v{id(fnode2)}", version="1.0.0")
    async def fork_missing_graph(state: FState2) -> FState2:
        return await fnode2(state)

    cp = MemoryCheckpointer()
    await fork_missing_graph.run(
        input=FState2(),
        config=RunConfig(thread_id="fork-miss-src", checkpointer=cp),
    )
    with pytest.raises(NoCheckpointError):
        await fork_missing_graph.fork(
            thread_id="fork-miss-src",
            checkpoint_id="nonexistent-id",
            new_thread_id="fork-miss-new",
            config=RunConfig(thread_id="fork-miss-src", checkpointer=cp),
        )


# ── _runtime.py: _compute_backoff ─────────────────────────────────────────────


def test_compute_backoff_none():
    from aegis._runtime import _compute_backoff

    assert _compute_backoff("none", 1) == 0.0
    assert _compute_backoff("none", 5) == 0.0


def test_compute_backoff_linear():
    from aegis._runtime import _compute_backoff

    assert _compute_backoff("linear", 1) == 1.0
    assert _compute_backoff("linear", 3) == 3.0


def test_compute_backoff_exponential():
    from aegis._runtime import _compute_backoff

    assert _compute_backoff("exponential", 1) == 1.0  # 2^0
    assert _compute_backoff("exponential", 3) == 4.0  # 2^2


def test_compute_backoff_unknown_defaults_to_zero():
    from aegis._runtime import _compute_backoff

    assert _compute_backoff("random_strategy", 2) == 0.0


# ── _runtime.py: _infer_state_type raises ────────────────────────────────────


def test_infer_state_type_raises_for_unannotated():
    """_infer_state_type raises TypeError when no AgentState subclass is found."""
    from aegis._decorators import GraphDef
    from aegis._runtime import _infer_state_type

    async def unannotated_fn(state):  # no annotations
        return state

    gdef = object.__new__(GraphDef)
    gdef.fn = unannotated_fn
    gdef.name = "unannotated-graph"

    with pytest.raises(TypeError, match="Cannot infer AgentState subclass"):
        _infer_state_type(gdef)


# ── Budget: downgrade_model policy ────────────────────────────────────────────


async def test_budget_downgrade_model_does_not_stop():
    """downgrade_model policy allows the graph to complete."""

    @node()
    async def two_calls(state: CovState, tools) -> CovState:
        await tools.cov_tool()
        await tools.cov_tool()  # 2nd call triggers budget → downgrade
        return state.update(calls=2)

    @graph(name=f"downgrade-v{id(two_calls)}", version="1.0.0")
    async def downgrade_graph(state: CovState) -> CovState:
        return await two_calls(state)

    cp = MemoryCheckpointer()
    async with downgrade_graph.mock_tools({"cov_tool": MockTool.returns("ok")}):
        result = await downgrade_graph.run(
            input=CovState(),
            config=RunConfig(thread_id="downgrade-001", checkpointer=cp),
            budget=Budget(
                max_tool_calls=1,
                on_exceeded="downgrade_model",
                downgrade_to="gpt-3.5-turbo",
            ),
        )
    # downgrade_model must not stop the graph
    assert result.status == "completed"
    assert result.state.calls == 2


async def test_budget_handler_extend():
    """Budget exceeded handler returning BudgetDecision.extend() continues the run."""
    from aegis._models import BudgetDecision

    @node()
    async def three_calls(state: CovState, tools) -> CovState:
        await tools.cov_tool()
        await tools.cov_tool()
        await tools.cov_tool()
        return state.update(calls=3)

    @graph(name=f"extend-v{id(three_calls)}", version="1.0.0")
    async def extend_graph(state: CovState) -> CovState:
        return await three_calls(state)

    @extend_graph.on_budget_exceeded
    async def handler(event):
        return BudgetDecision.extend(event.budget, max_tool_calls=10)

    cp = MemoryCheckpointer()
    async with extend_graph.mock_tools({"cov_tool": MockTool.returns("ok")}):
        result = await extend_graph.run(
            input=CovState(),
            config=RunConfig(thread_id="extend-001", checkpointer=cp),
            budget=Budget(max_tool_calls=1),
        )
    assert result.status == "completed"
    assert result.state.calls == 3


async def test_budget_handler_downgrade_decision():
    """Budget exceeded handler returning BudgetDecision.downgrade() continues."""
    from aegis._models import BudgetDecision

    @node()
    async def two_more(state: CovState, tools) -> CovState:
        await tools.cov_tool()
        await tools.cov_tool()
        return state.update(calls=2)

    @graph(name=f"handler-downgrade-v{id(two_more)}", version="1.0.0")
    async def handler_downgrade_graph(state: CovState) -> CovState:
        return await two_more(state)

    @handler_downgrade_graph.on_budget_exceeded
    async def handler(event):
        return BudgetDecision.downgrade("gpt-3.5-turbo")

    cp = MemoryCheckpointer()
    async with handler_downgrade_graph.mock_tools({"cov_tool": MockTool.returns("ok")}):
        result = await handler_downgrade_graph.run(
            input=CovState(),
            config=RunConfig(thread_id="handler-downgrade-001", checkpointer=cp),
            budget=Budget(max_tool_calls=1),
        )
    assert result.status == "completed"


async def test_budget_handler_raises_falls_back_to_hard_stop():
    """When budget handler raises, BudgetExceededError is propagated."""

    @node()
    async def call_once(state: CovState, tools) -> CovState:
        await tools.cov_tool()
        await tools.cov_tool()
        return state.update(calls=2)

    @graph(name=f"handler-raises-v{id(call_once)}", version="1.0.0")
    async def handler_raises_graph(state: CovState) -> CovState:
        return await call_once(state)

    @handler_raises_graph.on_budget_exceeded
    async def bad_handler(event):
        raise RuntimeError("handler exploded")

    cp = MemoryCheckpointer()
    async with handler_raises_graph.mock_tools({"cov_tool": MockTool.returns("ok")}):
        result = await handler_raises_graph.run(
            input=CovState(),
            config=RunConfig(thread_id="handler-raises-001", checkpointer=cp),
            budget=Budget(max_tool_calls=1),
        )
    assert result.status == "budget_exceeded"


# ── _http_intercept.py: inside-run paths ──────────────────────────────────────


async def test_http_intercept_inside_run_allows_permitted_domain():
    """Inside a graph run with a permitted domain, _intercept should not raise."""
    from aegis._context import RunContext, _run_context
    from aegis._http_intercept import _intercept
    from aegis._models import NetworkPermission, PermissionScope, RunTrace

    trace = RunTrace(
        run_id="r1",
        thread_id="t1",
        graph_name="g",
        graph_version="1.0",
        started_at=datetime.utcnow(),
        completed_at=None,
        status="running",
    )
    cp = MemoryCheckpointer()
    ctx = RunContext(
        run_id="r1",
        thread_id="t1",
        graph_name="g",
        graph_version="1.0",
        checkpointer=cp,
        trace=trace,
        permission_scope=PermissionScope(
            network=NetworkPermission(
                allowed_domains=["api.example.com"],
                deny_all_others=True,
            )
        ),
        budget=None,
    )
    token = _run_context.set(ctx)
    try:
        _intercept("https://api.example.com/path")  # should not raise
    finally:
        _run_context.reset(token)


async def test_http_intercept_inside_run_blocks_forbidden_domain():
    """Inside a graph run, _intercept raises PermissionDeniedError for non-whitelisted domain."""
    from aegis._context import RunContext, _run_context
    from aegis._http_intercept import _intercept
    from aegis._models import NetworkPermission, PermissionDeniedError, PermissionScope, RunTrace

    trace = RunTrace(
        run_id="r1",
        thread_id="t1",
        graph_name="g",
        graph_version="1.0",
        started_at=datetime.utcnow(),
        completed_at=None,
        status="running",
    )
    cp = MemoryCheckpointer()
    ctx = RunContext(
        run_id="r1",
        thread_id="t1",
        graph_name="g",
        graph_version="1.0",
        checkpointer=cp,
        trace=trace,
        permission_scope=PermissionScope(
            network=NetworkPermission(
                allowed_domains=["safe.example.com"],
                deny_all_others=True,
            )
        ),
        budget=None,
    )
    token = _run_context.set(ctx)
    try:
        with pytest.raises(PermissionDeniedError):
            _intercept("https://evil.example.com/steal")
    finally:
        _run_context.reset(token)


async def test_http_intercept_no_permission_scope_allows_all():
    """Inside a graph run with no permission_scope, all domains are allowed."""
    from aegis._context import RunContext, _run_context
    from aegis._http_intercept import _intercept
    from aegis._models import RunTrace

    trace = RunTrace(
        run_id="r1",
        thread_id="t1",
        graph_name="g",
        graph_version="1.0",
        started_at=datetime.utcnow(),
        completed_at=None,
        status="running",
    )
    cp = MemoryCheckpointer()
    ctx = RunContext(
        run_id="r1",
        thread_id="t1",
        graph_name="g",
        graph_version="1.0",
        checkpointer=cp,
        trace=trace,
        permission_scope=None,  # no scope = allow all
        budget=None,
    )
    token = _run_context.set(ctx)
    try:
        _intercept("https://any-domain.com/path")  # should not raise
    finally:
        _run_context.reset(token)


# ── _context.py: GraphContext ─────────────────────────────────────────────────


async def test_graph_context_subgraph_call():
    """GraphContext allows calling registered sub-graphs from within a node."""
    from aegis._context import GraphContext, RunContext
    from aegis._models import RunTrace

    @dataclass
    class SubState(AgentState):
        done: bool = False

    @node()
    async def sub_node(state: SubState) -> SubState:
        return state.update(done=True)

    @graph(name=f"sub-graph-ctx-v{id(sub_node)}", version="1.0.0")
    async def sub_graph_ctx(state: SubState) -> SubState:
        return await sub_node(state)

    cp = MemoryCheckpointer()
    trace = RunTrace(
        run_id="r1",
        thread_id="t1",
        graph_name="outer",
        graph_version="1.0",
        started_at=datetime.utcnow(),
        completed_at=None,
        status="running",
    )
    ctx = RunContext(
        run_id="r1",
        thread_id="t1",
        graph_name="outer",
        graph_version="1.0",
        checkpointer=cp,
        trace=trace,
        permission_scope=None,
        budget=None,
    )
    gc = GraphContext(ctx)
    proxy = gc.__getattr__(f"sub-graph-ctx-v{id(sub_node)}")
    result = await proxy.run(
        input=SubState(),
        config=RunConfig(thread_id="gc-sub-001", checkpointer=cp),
    )
    assert result.status == "completed"
    assert result.state.done is True


def test_graph_context_raises_for_unknown_graph():
    """GraphContext raises KeyError for unregistered graph name."""
    from aegis._context import GraphContext, RunContext
    from aegis._models import RunTrace

    cp = MemoryCheckpointer()
    trace = RunTrace(
        run_id="r1",
        thread_id="t1",
        graph_name="outer",
        graph_version="1.0",
        started_at=datetime.utcnow(),
        completed_at=None,
        status="running",
    )
    ctx = RunContext(
        run_id="r1",
        thread_id="t1",
        graph_name="outer",
        graph_version="1.0",
        checkpointer=cp,
        trace=trace,
        permission_scope=None,
        budget=None,
    )
    gc = GraphContext(ctx)
    with pytest.raises(KeyError, match="not found"):
        gc.__getattr__("nonexistent-graph-xyz")


def test_graph_context_private_attr_raises():
    """GraphContext raises AttributeError for underscore-prefixed names."""
    from aegis._context import GraphContext, RunContext
    from aegis._models import RunTrace

    cp = MemoryCheckpointer()
    trace = RunTrace(
        run_id="r1",
        thread_id="t1",
        graph_name="outer",
        graph_version="1.0",
        started_at=datetime.utcnow(),
        completed_at=None,
        status="running",
    )
    ctx = RunContext(
        run_id="r1",
        thread_id="t1",
        graph_name="outer",
        graph_version="1.0",
        checkpointer=cp,
        trace=trace,
        permission_scope=None,
        budget=None,
    )
    gc = GraphContext(ctx)
    with pytest.raises(AttributeError):
        gc.__getattr__("_private_attr")


# ── _decorators.py: edge cases ────────────────────────────────────────────────


async def test_graph_checkpoint_history_via_api():
    """get_checkpoint_history() returns checkpoints in step order."""
    cp = MemoryCheckpointer()
    await cov_graph.run(
        input=CovState(value=1),
        config=RunConfig(thread_id="ckpt-hist-001", checkpointer=cp),
    )
    history = await cov_graph.get_checkpoint_history(
        "ckpt-hist-001", config=RunConfig(thread_id="ckpt-hist-001", checkpointer=cp)
    )
    assert len(history) >= 2
    assert history[0].step == 0
    assert history[0].node_name == "__input__"


# ── _models.py: BudgetDecision edge cases ────────────────────────────────────


def test_budget_decision_downgrade_factory():
    """BudgetDecision.downgrade() creates correct action/model."""
    from aegis._models import BudgetDecision

    bd = BudgetDecision.downgrade("gpt-3.5-turbo")
    assert bd.action == "downgrade"
    assert bd.updated_budget is not None
    assert bd.updated_budget.downgrade_to == "gpt-3.5-turbo"


def test_budget_status_compute_pct_all_dimensions():
    """BudgetStatus.compute_pct() populates all percentage keys."""
    from aegis._models import Budget, BudgetStatus

    budget = Budget(
        max_tokens=1000,
        max_llm_cost_usd=5.0,
        max_tool_calls=10,
        max_wall_time_seconds=60,
    )
    status = BudgetStatus(
        tokens_used=500,
        cost_usd=2.5,
        tool_calls_made=5,
        wall_time_seconds=30,
    )
    status.compute_pct(budget)
    assert status.pct_consumed["tokens"] == pytest.approx(0.5)
    assert status.pct_consumed["cost"] == pytest.approx(0.5)
    assert status.pct_consumed["tool_calls"] == pytest.approx(0.5)
    assert status.pct_consumed["wall_time"] == pytest.approx(0.5)
