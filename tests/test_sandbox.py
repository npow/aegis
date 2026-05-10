"""Tests for subprocess sandboxing (@node sandbox=True)."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import pytest

import rampart

# ── State types ────────────────────────────────────────────────────────────────


@dataclass
class SbState(rampart.AgentState):
    x: int = 0
    label: str = ""


# ── Module-level sandbox nodes (must be at module level for pickling) ──────────


@rampart.node(sandbox=True)
async def _sb_double(state: SbState) -> SbState:
    return state.update(x=state.x * 2)


@rampart.node(sandbox=True)
async def _sb_label(state: SbState) -> SbState:
    return state.update(label=f"sandboxed:{state.x}")


@rampart.node(sandbox=True, retries=0)
async def _sb_crash(state: SbState) -> SbState:
    raise ValueError("subprocess boom")


@rampart.node(sandbox=True)
async def _sb_with_tools(state: SbState, tools: rampart.ToolContext) -> SbState:
    """Sandbox=True but declares tools — should fall back to in-process with warning."""
    return state.update(x=99)


@rampart.graph(name="_sb_graph_double")
async def _sb_graph_double(state: SbState) -> SbState:
    return await _sb_double(state)


@rampart.graph(name="_sb_graph_label")
async def _sb_graph_label(state: SbState) -> SbState:
    return await _sb_label(state)


@rampart.graph(name="_sb_graph_crash")
async def _sb_graph_crash(state: SbState) -> SbState:
    return await _sb_crash(state)


@rampart.graph(name="_sb_graph_with_tools")
async def _sb_graph_with_tools(state: SbState) -> SbState:
    return await _sb_with_tools(state)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sandbox_basic_execution():
    """A sandboxed node runs in subprocess and returns correct state."""
    cfg = rampart.RunConfig(thread_id="sb-basic")
    result = await _sb_graph_double.run(SbState(x=7), cfg)
    assert result.status == "completed"
    assert result.state.x == 14


@pytest.mark.asyncio
async def test_sandbox_string_field():
    cfg = rampart.RunConfig(thread_id="sb-label")
    result = await _sb_graph_label.run(SbState(x=5), cfg)
    assert result.state.label == "sandboxed:5"


@pytest.mark.asyncio
async def test_sandbox_exception_propagates():
    """Exceptions raised in the subprocess surface as run failures."""
    cfg = rampart.RunConfig(thread_id="sb-crash")
    result = await _sb_graph_crash.run(SbState(), cfg)
    assert result.status == "failed"
    assert result.error is not None
    assert "boom" in result.error.message


@pytest.mark.asyncio
async def test_sandbox_warns_and_falls_back_with_injected_context():
    """sandbox=True + tools parameter: warn and run in-process."""
    cfg = rampart.RunConfig(thread_id="sb-tools")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await _sb_graph_with_tools.run(SbState(x=0), cfg)

    assert result.state.x == 99  # ran in-process
    warn_msgs = [str(w.message) for w in caught]
    assert any("sandbox" in m.lower() for m in warn_msgs)


# ── Unit tests for the sandbox internals (no subprocess) ─────────────────────
# These complement the integration tests above by exercising the helper
# functions directly so coverage reflects what's actually tested.


def test_configure_sandbox_updates_module_state():
    """configure_sandbox writes max_workers / start_method to module globals."""
    from rampart import _sandbox

    original_workers = _sandbox._max_workers
    original_method = _sandbox._start_method
    try:
        _sandbox.configure_sandbox(max_workers=8, start_method="spawn")
        assert _sandbox._max_workers == 8
        assert _sandbox._start_method == "spawn"
    finally:
        _sandbox.configure_sandbox(
            max_workers=original_workers, start_method=original_method
        )


def test_pool_initializer_resets_signal_handlers():
    """_pool_initializer restores default SIGINT/SIGTERM handling.

    Workers inherit the parent's signal handlers via fork; the initializer
    must reset them so workers respond normally to SIGINT/SIGTERM rather
    than running the orchestrator's pyflow handlers.
    """
    import signal

    from rampart._sandbox import _pool_initializer

    # Capture and restore original handlers around the call.
    orig_int = signal.getsignal(signal.SIGINT)
    orig_term = signal.getsignal(signal.SIGTERM)
    custom = lambda signum, frame: None  # noqa: E731
    signal.signal(signal.SIGINT, custom)
    signal.signal(signal.SIGTERM, custom)
    try:
        _pool_initializer()
        # Both should now be SIG_DFL.
        assert signal.getsignal(signal.SIGINT) == signal.SIG_DFL
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
    finally:
        signal.signal(signal.SIGINT, orig_int)
        signal.signal(signal.SIGTERM, orig_term)


def test_run_node_in_subprocess_executes_async_fn_and_returns_dict():
    """_run_node_in_subprocess: reconstructs state, runs fn, returns dict.

    Called directly in-process here (no actual fork) — sufficient to cover
    the state reconstruction + asyncio.run + asdict path that pickling
    in a real subprocess exercises.
    """
    from rampart._sandbox import _run_node_in_subprocess

    async def double(state: SbState) -> SbState:
        return state.update(x=state.x * 2)

    result = _run_node_in_subprocess(
        double,
        state_dict={"x": 21, "label": "hello"},
        state_class=SbState,
        max_memory_mb=None,
        max_cpu_seconds=None,
    )
    assert result["x"] == 42
    assert result["label"] == "hello"


def test_run_node_in_subprocess_filters_unknown_state_keys():
    """The state_dict may carry keys the AgentState subclass doesn't declare
    (e.g. older snapshots after a schema migration). Those keys must be
    silently dropped during reconstruction; only known fields are passed
    to the dataclass __init__."""
    from rampart._sandbox import _run_node_in_subprocess

    async def identity(state: SbState) -> SbState:
        return state

    result = _run_node_in_subprocess(
        identity,
        state_dict={"x": 5, "label": "ok", "removed_field": "garbage"},
        state_class=SbState,
        max_memory_mb=None,
        max_cpu_seconds=None,
    )
    assert "removed_field" not in result
    assert result["x"] == 5


def test_run_node_in_subprocess_warns_on_invalid_resource_limits(monkeypatch):
    """When setrlimit raises (e.g. wrong platform or bad value), the function
    must NOT propagate the error — it warns and continues. The async fn
    still runs."""
    import warnings as _warnings

    from rampart._sandbox import _run_node_in_subprocess

    async def trivial(state: SbState) -> SbState:
        return state

    # Monkeypatch the resource module's setrlimit to raise OSError for a
    # known cause (insufficient privileges / bad value). The sandbox code
    # imports `resource` lazily, so patch via sys.modules.
    import resource as _real_res

    def boom(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(_real_res, "setrlimit", boom)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result = _run_node_in_subprocess(
            trivial,
            state_dict={"x": 1, "label": ""},
            state_class=SbState,
            max_memory_mb=64,
            max_cpu_seconds=1,
        )
    assert result["x"] == 1
    # At least one warning should mention the resource-limit failure.
    warn_msgs = [str(w.message) for w in caught]
    assert any("resource limits" in m for m in warn_msgs)
