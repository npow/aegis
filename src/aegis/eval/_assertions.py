"""Assertion types for the Aegis eval pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .._models import (
    AgentState,
    EvalAssertion,
    RunTrace,
    SchemaAssertion,
    ToolCallAssertion,
    TraceSnapshotAssertion,
)


def evaluate_assertion(
    assertion: EvalAssertion,
    final_state: AgentState,
    trace: RunTrace | None,
) -> tuple[bool, str]:
    """Evaluate a single assertion. Returns (passed, message)."""
    if isinstance(assertion, (ToolCallAssertion, TraceSnapshotAssertion)) and trace is None:
        return False, f"Cannot evaluate {type(assertion).__name__}: trace is None (graph failed)"
    if isinstance(assertion, ToolCallAssertion):
        return _evaluate_tool_call(assertion, trace)  # type: ignore[arg-type]
    elif isinstance(assertion, SchemaAssertion):
        return _evaluate_schema(assertion, final_state)
    elif isinstance(assertion, TraceSnapshotAssertion):
        return _evaluate_trace_snapshot(assertion, trace)  # type: ignore[arg-type]
    else:
        return False, f"Unknown assertion type: {type(assertion).__name__}"


def _evaluate_tool_call(assertion: ToolCallAssertion, trace: RunTrace) -> tuple[bool, str]:
    all_tool_calls = [
        tc
        for node in trace.nodes_executed
        for tc in node.tool_calls
        if tc.tool_name == assertion.tool_name
    ]
    call_count = len(all_tool_calls)

    if assertion.called and call_count == 0:
        return False, (
            f"Tool '{assertion.tool_name}' was not called "
            f"(expected at least {assertion.min_times} time(s))"
        )
    if not assertion.called and call_count > 0:
        return (
            False,
            f"Tool '{assertion.tool_name}' was called {call_count} time(s) but expected not to be",
        )

    if call_count < assertion.min_times:
        return False, (
            f"Tool '{assertion.tool_name}' called {call_count} time(s), "
            f"expected at least {assertion.min_times}"
        )

    if assertion.max_times is not None and call_count > assertion.max_times:
        return False, (
            f"Tool '{assertion.tool_name}' called {call_count} time(s), "
            f"expected at most {assertion.max_times}"
        )

    if assertion.args_match:
        # At least one call must satisfy ALL of the expected args
        matching = [
            tc
            for tc in all_tool_calls
            if all(tc.args.get(k) == v for k, v in assertion.args_match.items())
        ]
        if not matching:
            expected_str = ", ".join(f"{k}={v!r}" for k, v in assertion.args_match.items())
            actual_str = "; ".join(
                "{" + ", ".join(f"{k}={tc.args.get(k)!r}" for k in assertion.args_match) + "}"
                for tc in all_tool_calls
            )
            return False, (
                f"Tool '{assertion.tool_name}': no call matched args_match {{{expected_str}}}. "
                f"Actual call args: [{actual_str}]"
            )

    return True, assertion.description or f"Tool '{assertion.tool_name}' called correctly"


def _evaluate_schema(assertion: SchemaAssertion, final_state: AgentState) -> tuple[bool, str]:
    try:
        result = assertion.predicate(final_state)
    except Exception as exc:
        return False, f"Schema assertion raised {type(exc).__name__}: {exc}"

    if result:
        return True, assertion.description or "Schema assertion passed"
    return False, assertion.description or "Schema assertion failed"


def _evaluate_trace_snapshot(
    assertion: TraceSnapshotAssertion, trace: RunTrace
) -> tuple[bool, str]:
    """Compare the actual tool call sequence to a golden trace file."""
    golden_path = Path(assertion.golden_trace_path)
    if not golden_path.exists():
        # First run: write the golden trace and pass
        actual = _trace_to_comparable(trace, assertion.normalize_fields)
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(actual, indent=2))
        return True, f"Golden trace written to {golden_path}"

    golden = json.loads(golden_path.read_text())
    actual = _trace_to_comparable(trace, assertion.normalize_fields)

    if golden == actual:
        return True, assertion.description or "Trace matches golden"

    # Produce a diff-like description
    g_seq = [e["tool_name"] for e in golden]
    a_seq = [e["tool_name"] for e in actual]
    if g_seq != a_seq:
        return False, (
            f"Tool call sequence diverged from golden.\n  Expected: {g_seq}\n  Actual:   {a_seq}"
        )
    return False, (
        f"Trace args diverged from golden (same tool order). "
        f"Check {assertion.golden_trace_path} for details."
    )


def _trace_to_comparable(trace: RunTrace, normalize_fields: list[str]) -> list[dict[str, Any]]:
    """Flatten tool calls to a normalized list for comparison."""
    entries = []
    for node in trace.nodes_executed:
        for tc in node.tool_calls:
            entry: dict[str, Any] = {
                "tool_name": tc.tool_name,
                "node_name": tc.node_name,
                "args": tc.args,
            }
            # Remove normalized (volatile) fields
            for f in normalize_fields:
                entry.pop(f, None)
            entries.append(entry)
    return entries
