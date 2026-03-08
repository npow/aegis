"""Tests for EvalSuite, EvalCase, and assertion types."""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from aegis import AgentState, RunConfig, graph, node, tool
from aegis.checkpointers import MemoryCheckpointer
from aegis.eval import (
    EvalCase,
    EvalGateFailure,
    EvalSuite,
    SchemaAssertion,
    ToolCallAssertion,
    TraceSnapshotAssertion,
)
from aegis.testing import MockTool

# ── Fixtures ──────────────────────────────────────────────────────────────────


@dataclass
class EvalState(AgentState):
    query: str = ""
    summary: str = ""
    status: str = "idle"


@tool(name="eval_search_tool")
async def eval_search_tool(query: str) -> list:
    return []


@node()
async def eval_research_node(state: EvalState, tools) -> EvalState:
    await tools.eval_search_tool(query=state.query)
    return state.update(status="searched")


@node()
async def eval_summarize_node(state: EvalState) -> EvalState:
    return state.update(
        summary=f"Summary of: {state.query}" * 5,  # >100 chars
        status="done",
    )


@graph(name="eval-pipeline", version="1.0.0")
async def eval_pipeline(state: EvalState) -> EvalState:
    state = await eval_research_node(state)
    state = await eval_summarize_node(state)
    return state


# ── Assertion unit tests ──────────────────────────────────────────────────────


def test_tool_call_assertion_passes_when_called():
    from datetime import datetime

    from aegis._models import NodeTrace, RunTrace, ToolCall
    from aegis.eval._assertions import evaluate_assertion

    tool_call = ToolCall(
        call_id="c1",
        tool_name="eval_search_tool",
        args={"query": "test"},
        result=[],
        error=None,
        latency_ms=10,
        timestamp=datetime.utcnow(),
        node_name="eval_research_node",
    )
    node_trace = NodeTrace(
        node_name="eval_research_node",
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        input_state={},
        output_state={},
        tool_calls=[tool_call],
        status="completed",
    )
    trace = RunTrace(
        run_id="r1",
        thread_id="t1",
        graph_name="eval-pipeline",
        graph_version="1.0",
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        status="completed",
        nodes_executed=[node_trace],
    )
    assertion = ToolCallAssertion(
        description="must call eval_search_tool",
        tool_name="eval_search_tool",
        called=True,
        min_times=1,
    )
    state = EvalState(query="test", status="done")
    passed, msg = evaluate_assertion(assertion, state, trace)
    assert passed is True


def test_tool_call_assertion_fails_when_not_called():
    from datetime import datetime

    from aegis._models import RunTrace
    from aegis.eval._assertions import evaluate_assertion

    trace = RunTrace(
        run_id="r1",
        thread_id="t1",
        graph_name="g",
        graph_version="1.0",
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        status="completed",
    )
    assertion = ToolCallAssertion(
        description="must call eval_search_tool",
        tool_name="eval_search_tool",
        called=True,
        min_times=1,
    )
    state = EvalState()
    passed, msg = evaluate_assertion(assertion, state, trace)
    assert passed is False
    assert "not called" in msg.lower() or "0" in msg


def test_schema_assertion_passes():
    from datetime import datetime

    from aegis._models import RunTrace
    from aegis.eval._assertions import evaluate_assertion

    trace = RunTrace(
        run_id="r1",
        thread_id="t1",
        graph_name="g",
        graph_version="1.0",
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        status="completed",
    )
    assertion = SchemaAssertion(
        predicate=lambda s: s.status == "done",
        description="must reach done status",
    )
    state = EvalState(status="done")
    passed, msg = evaluate_assertion(assertion, state, trace)
    assert passed is True


def test_schema_assertion_fails():
    from datetime import datetime

    from aegis._models import RunTrace
    from aegis.eval._assertions import evaluate_assertion

    trace = RunTrace(
        run_id="r1",
        thread_id="t1",
        graph_name="g",
        graph_version="1.0",
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        status="completed",
    )
    assertion = SchemaAssertion(
        predicate=lambda s: len(s.summary) > 100,
        description="summary must be >100 chars",
    )
    state = EvalState(summary="short")
    passed, msg = evaluate_assertion(assertion, state, trace)
    assert passed is False


def test_tool_call_assertion_max_times():
    from datetime import datetime

    from aegis._models import NodeTrace, RunTrace, ToolCall
    from aegis.eval._assertions import evaluate_assertion

    def make_tc(i):
        return ToolCall(
            call_id=f"c{i}",
            tool_name="eval_search_tool",
            args={},
            result=[],
            error=None,
            latency_ms=1,
            timestamp=datetime.utcnow(),
            node_name="n",
        )

    node_trace = NodeTrace(
        node_name="n",
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        input_state={},
        output_state={},
        tool_calls=[make_tc(i) for i in range(5)],  # 5 calls
        status="completed",
    )
    trace = RunTrace(
        run_id="r1",
        thread_id="t1",
        graph_name="g",
        graph_version="1.0",
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        status="completed",
        nodes_executed=[node_trace],
    )
    assertion = ToolCallAssertion(
        description="max 3 calls",
        tool_name="eval_search_tool",
        called=True,
        min_times=1,
        max_times=3,
    )
    passed, msg = evaluate_assertion(assertion, EvalState(), trace)
    assert passed is False
    assert "5" in msg or "at most" in msg


# ── EvalSuite integration tests ───────────────────────────────────────────────


async def test_eval_suite_passes_when_all_cases_pass():
    cp = MemoryCheckpointer()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/eval.json"

        # We'll use a mock-based cassette approach: record with mocks
        from aegis.testing import cassette as _cassette

        async with _cassette.record(path):
            async with eval_pipeline.mock_tools(
                {
                    "eval_search_tool": MockTool.returns(["r1"]),
                }
            ):
                await eval_pipeline.run(
                    input=EvalState(query="quantum"),
                    config=RunConfig(thread_id="eval-record", checkpointer=cp),
                )

        suite = EvalSuite(
            name="test-eval-suite",
            graph=eval_pipeline,
            cases=[
                EvalCase(
                    id="basic",
                    input=EvalState(query="quantum"),
                    cassette=path,
                    assertions=[
                        SchemaAssertion(
                            predicate=lambda s: s.status == "done",
                            description="must reach done status",
                        ),
                        ToolCallAssertion(
                            description="must call eval_search_tool",
                            tool_name="eval_search_tool",
                            called=True,
                            min_times=1,
                        ),
                    ],
                ),
            ],
            pass_rate_gate=1.0,
        )

        results = await suite.run(base_thread_id="eval-run")
        assert results.pass_rate == 1.0
        assert results.gate_passed is True
        results.assert_gates()  # should not raise


async def test_eval_suite_fails_gate_when_case_fails():
    cp = MemoryCheckpointer()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/eval_fail.json"

        from aegis.testing import cassette as _cassette

        async with _cassette.record(path):
            async with eval_pipeline.mock_tools(
                {
                    "eval_search_tool": MockTool.returns([]),
                }
            ):
                await eval_pipeline.run(
                    input=EvalState(query="test"),
                    config=RunConfig(thread_id="eval-fail-record", checkpointer=cp),
                )

        suite = EvalSuite(
            name="failing-suite",
            graph=eval_pipeline,
            cases=[
                EvalCase(
                    id="impossible-assertion",
                    input=EvalState(query="test"),
                    cassette=path,
                    assertions=[
                        SchemaAssertion(
                            predicate=lambda s: False,  # always fails
                            description="this always fails",
                        ),
                    ],
                ),
            ],
            pass_rate_gate=1.0,
        )

        results = await suite.run(base_thread_id="eval-fail-run")
        assert results.pass_rate == 0.0
        assert results.gate_passed is False

        with pytest.raises(EvalGateFailure):
            results.assert_gates()


async def test_eval_suite_summary_contains_case_results():
    cp = MemoryCheckpointer()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/summary.json"

        from aegis.testing import cassette as _cassette

        async with _cassette.record(path):
            async with eval_pipeline.mock_tools(
                {
                    "eval_search_tool": MockTool.returns([]),
                }
            ):
                await eval_pipeline.run(
                    input=EvalState(query="test"),
                    config=RunConfig(thread_id="summary-record", checkpointer=cp),
                )

        suite = EvalSuite(
            name="summary-suite",
            graph=eval_pipeline,
            cases=[
                EvalCase(
                    id="my-case",
                    input=EvalState(query="test"),
                    cassette=path,
                    assertions=[
                        SchemaAssertion(predicate=lambda s: True, description="pass"),
                    ],
                ),
            ],
        )

        results = await suite.run()
        summary = results.summary()
        assert "summary-suite" in summary
        assert "my-case" in summary


async def test_trace_snapshot_assertion_creates_golden_on_first_run():
    """First run creates the golden file and passes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from datetime import datetime

        from aegis._models import NodeTrace, RunTrace, ToolCall
        from aegis.eval._assertions import evaluate_assertion

        golden_path = f"{tmpdir}/golden.json"

        tool_call = ToolCall(
            call_id="c1",
            tool_name="eval_search_tool",
            args={"query": "test"},
            result=[],
            error=None,
            latency_ms=10,
            timestamp=datetime.utcnow(),
            node_name="n",
        )
        node_trace = NodeTrace(
            node_name="n",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            input_state={},
            output_state={},
            tool_calls=[tool_call],
            status="completed",
        )
        trace = RunTrace(
            run_id="r1",
            thread_id="t1",
            graph_name="g",
            graph_version="1.0",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            status="completed",
            nodes_executed=[node_trace],
        )
        assertion = TraceSnapshotAssertion(
            description="trace regression test",
            golden_trace_path=golden_path,
        )
        passed, msg = evaluate_assertion(assertion, EvalState(), trace)
        assert passed is True
        assert Path(golden_path).exists()

        # Second run with same trace should also pass
        passed2, _ = evaluate_assertion(assertion, EvalState(), trace)
        assert passed2 is True
