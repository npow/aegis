"""Graph execution engine: node scheduling, retry, checkpointing, resume, stream."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ._artifacts import ArtifactContext
from ._context import (
    GraphContext,
    LLMContext,
    RunContext,
    ToolContext,
    _run_context,
)
from ._decorators import _TOOL_REGISTRY, GraphDef, NodeDef
from ._models import (
    AgentState,
    Budget,
    BudgetExceededError,
    Checkpoint,
    NoCheckpointError,
    NodeTrace,
    RunConfig,
    RunError,
    RunResult,
    RunTrace,
)

# ── Streaming event ────────────────────────────────────────────────────────────


@dataclass
class GraphEvent:
    """Emitted for each node completion during stream()."""

    type: str  # "node_completed" | "node_failed"
    node_name: str
    state: AgentState
    error: str | None = None


# ── Main entry points ─────────────────────────────────────────────────────────


async def _run_graph(
    graph_def: GraphDef,
    input_state: AgentState,
    config: RunConfig,
    budget: Budget | None = None,
    *,
    fast_forward_to_step: int = -1,
    existing_checkpoints: dict[int, Checkpoint] | None = None,
    is_resume: bool = False,
    parent_run_id: str | None = None,
    stream_queue: asyncio.Queue | None = None,  # type: ignore[type-arg]
) -> RunResult:
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    checkpointer = graph_def._resolve_checkpointer(config)

    # Resolve OTel tracer and artifact store from global config
    from . import _globals

    # Artifact store: RunConfig override first, then global default
    artifact_store: Any | None = getattr(config, "artifact_store", None)
    if artifact_store is None:
        artifact_store = _globals.DEFAULT_ARTIFACT_STORE

    otel_tracer: Any | None = None
    if _globals.DEFAULT_TRACER is not None:
        try:
            otel_tracer = _globals.DEFAULT_TRACER._get_tracer()
        except Exception:
            pass

    trace = RunTrace(
        run_id=run_id,
        thread_id=config.thread_id,
        graph_name=graph_def.name,
        graph_version=graph_def.version,
        started_at=datetime.now(timezone.utc),
        completed_at=None,
        status="running" if not is_resume else "resumed",
        parent_run_id=parent_run_id,
    )

    # Pull in any testing overrides (mock_tools / cassette)
    mock_tools, mock_ctx, cassette_mode, cassette, cassette_override_tools = _get_testing_state()

    ctx = RunContext(
        run_id=run_id,
        thread_id=config.thread_id,
        graph_name=graph_def.name,
        graph_version=graph_def.version,
        checkpointer=checkpointer,
        trace=trace,
        permission_scope=graph_def.permissions,
        budget=budget,
        tool_registry=dict(_TOOL_REGISTRY),
        mock_tools=mock_tools,
        mock_ctx=mock_ctx,
        cassette_mode=cassette_mode,
        cassette=cassette,
        cassette_override_tools=cassette_override_tools,
        _fast_forward_to_step=fast_forward_to_step,
        _existing_checkpoints=existing_checkpoints or {},
        _budget_start_time=time.monotonic(),
        _budget_exceeded_handler=graph_def._budget_exceeded_handler,
        stream_queue=stream_queue,
        otel_tracer=otel_tracer,
        artifact_store=artifact_store,
    )

    # Inject run metadata into state
    import dataclasses as _dc

    input_state = _dc.replace(input_state, thread_id=config.thread_id, run_id=run_id)

    # Checkpoint the initial input state (step 0)
    initial_ckpt = _make_checkpoint(
        ctx=ctx,
        step=0,
        node_name="__input__",
        state=input_state,
        parent_id=None,
    )
    # Only save if not already in existing checkpoints (resume case)
    if 0 not in ctx._existing_checkpoints:
        await checkpointer.save(initial_ckpt)
        ctx.last_checkpoint_id = initial_ckpt.id
    else:
        ctx.last_checkpoint_id = ctx._existing_checkpoints[0].id

    # Reserve step 0 for input
    ctx._step_counter = 1

    token = _run_context.set(ctx)
    result_state = input_state
    error: RunError | None = None

    try:
        # Wrap in OTel span if configured
        if otel_tracer is not None:
            with otel_tracer.start_as_current_span(f"rampart.graph.{graph_def.name}") as span:
                span.set_attribute("rampart.graph_name", graph_def.name)
                span.set_attribute("rampart.graph_version", graph_def.version)
                span.set_attribute("rampart.run_id", run_id)
                span.set_attribute("rampart.thread_id", config.thread_id)
                # Capture OTel trace ID for observability linkage
                try:
                    trace.otel_trace_id = format(span.get_span_context().trace_id, "032x")
                except Exception:
                    pass
                result_state = await graph_def.fn(input_state)
        else:
            result_state = await graph_def.fn(input_state)
        trace.status = "completed"
    except BudgetExceededError as exc:
        trace.status = "budget_exceeded"
        trace.error = str(exc)
        error = RunError(
            message=str(exc),
            exception_type=type(exc).__name__,
        )
    except asyncio.CancelledError:
        trace.status = "cancelled"
        raise
    except Exception as exc:
        import traceback as tb

        trace.status = "failed"
        trace.error = str(exc)
        error = RunError(
            message=str(exc),
            exception_type=type(exc).__name__,
            traceback=tb.format_exc(),
        )
    finally:
        _run_context.reset(token)
        now = datetime.now(timezone.utc)
        trace.completed_at = now
        trace.wall_time_seconds = (now - trace.started_at).total_seconds()
        trace.final_state = _serialize_state(result_state)

        if trace.status in ("failed", "cancelled", "budget_exceeded"):
            try:
                terminal_ckpt = _make_checkpoint(ctx=ctx, step=ctx._step_counter, node_name="__terminal__", state=result_state, parent_id=ctx.last_checkpoint_id)
                await checkpointer.save(terminal_ckpt)
            except Exception:
                pass  # Best-effort terminal checkpoint

        # Finalize cassette recording
        if cassette_mode == "record" and cassette is not None:
            cassette.update_hash()

    return RunResult(
        state=result_state,
        trace=trace,
        status=trace.status,  # type: ignore[arg-type]
        error=error,
    )


async def _resume_graph(
    graph_def: GraphDef,
    thread_id: str,
    config: RunConfig,
) -> RunResult:
    """Resume execution from the last committed checkpoint."""
    checkpointer = graph_def._resolve_checkpointer(config)
    history = await checkpointer.get_history(thread_id, graph_def.name)

    if not history:
        raise NoCheckpointError(
            f"No checkpoints found for thread '{thread_id}' in graph '{graph_def.name}'. "
            "Cannot resume."
        )

    if history and history[0].graph_version != graph_def.version:
        from ._models import GraphVersionConflict

        raise GraphVersionConflict(
            f"Graph version mismatch: checkpoints are version '{history[0].graph_version}' "
            f"but graph is now version '{graph_def.version}'. "
            "Use graph.fork() to migrate, or delete old checkpoints."
        )

    # Build existing checkpoints map
    existing = {c.step: c for c in history}
    last_step = max(existing.keys())

    # Get the initial input state from step 0 checkpoint
    input_ckpt = existing.get(0)
    if input_ckpt is None:
        raise NoCheckpointError(
            "Missing step-0 (input) checkpoint; cannot reconstruct initial state."
        )

    # We need to reconstruct state_type from the graph's function annotation
    state_type = _infer_state_type(graph_def)
    input_state = _deserialize_state(input_ckpt.state_snapshot, state_type)

    # Preserve the previous run_id as parent for lineage tracking
    prev_run_id = existing[last_step].run_id

    return await _run_graph(
        graph_def=graph_def,
        input_state=input_state,
        config=config,
        fast_forward_to_step=last_step,
        existing_checkpoints=existing,
        is_resume=True,
        parent_run_id=prev_run_id,
    )


async def _fork_graph(
    graph_def: GraphDef,
    thread_id: str,
    checkpoint_id: str,
    inject_state: dict[str, Any],
    new_thread_id: str,
    config: RunConfig | None = None,
) -> RunResult:
    """Fork from a specific checkpoint with optional state injection.

    ``config`` controls which checkpointer is used to look up the original
    thread's history. Falls back to the graph-level default when omitted.
    """
    checkpointer = graph_def._resolve_checkpointer(config)
    history = await checkpointer.get_history(thread_id, graph_def.name)

    # Find the target checkpoint
    target = next((c for c in history if c.id == checkpoint_id), None)
    if target is None:
        raise NoCheckpointError(f"Checkpoint '{checkpoint_id}' not found for thread '{thread_id}'.")

    # Build existing checkpoints up to the fork point
    existing = {c.step: c for c in history if c.step <= target.step}

    # Inject state changes into the target checkpoint's state
    merged_state_dict = {**target.state_snapshot, **inject_state}

    # Get input state from step 0
    input_ckpt = existing.get(0)
    if input_ckpt is None:
        raise NoCheckpointError("Missing step-0 (input) checkpoint.")

    state_type = _infer_state_type(graph_def)

    # Rewrite the fork target's state
    fork_run_id = f"fork-{uuid.uuid4().hex[:8]}"
    existing[target.step] = Checkpoint(
        id=checkpoint_id,
        thread_id=new_thread_id,
        run_id=fork_run_id,
        graph_name=graph_def.name,
        graph_version=graph_def.version,
        step=target.step,
        node_name=target.node_name,
        state_snapshot=merged_state_dict,
        created_at=datetime.now(timezone.utc),
        parent_checkpoint_id=target.parent_checkpoint_id,
        is_fork_root=True,
    )

    # Save fork checkpoints to new thread
    new_config = RunConfig(thread_id=new_thread_id, checkpointer=checkpointer)
    for c in existing.values():
        fork_c = Checkpoint(
            id=c.id,
            thread_id=new_thread_id,
            run_id=c.run_id,
            graph_name=c.graph_name,
            graph_version=c.graph_version,
            step=c.step,
            node_name=c.node_name,
            state_snapshot=c.state_snapshot,
            created_at=c.created_at,
            parent_checkpoint_id=c.parent_checkpoint_id,
            is_fork_root=c.is_fork_root,
        )
        await checkpointer.save(fork_c)

    input_state = _deserialize_state(input_ckpt.state_snapshot, state_type)
    return await _run_graph(
        graph_def=graph_def,
        input_state=input_state,
        config=new_config,
        fast_forward_to_step=target.step,
        existing_checkpoints={**existing},
        parent_run_id=target.run_id,
    )


async def _stream_graph(
    graph_def: GraphDef,
    input_state: AgentState,
    config: RunConfig,
    budget: Budget | None = None,
) -> AsyncIterator[GraphEvent]:
    """Stream node events in real-time during graph execution.

    Yields a ``GraphEvent`` immediately after each node completes (or fails),
    rather than waiting for the entire graph to finish.
    """
    queue: asyncio.Queue[GraphEvent | None] = asyncio.Queue(maxsize=512)

    async def _run_and_signal() -> None:
        try:
            await _run_graph(graph_def, input_state, config, budget, stream_queue=queue)
        finally:
            await queue.put(None)  # sentinel — signals end of stream

    task = asyncio.create_task(_run_and_signal())

    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event
    finally:
        # Cancel the background task if the consumer exits early (e.g., break or exception),
        # otherwise await would deadlock if the queue is full and no one is consuming.
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ── Node execution ────────────────────────────────────────────────────────────


async def _execute_node_in_context(
    node_def: NodeDef,
    state: AgentState,
    ctx: RunContext,
) -> AgentState:
    """Execute a single node with full Rampart machinery."""
    step = ctx.next_step()

    # Fast-forward: return cached state if this step has a completed checkpoint
    if step <= ctx._fast_forward_to_step and step in ctx._existing_checkpoints:
        cached = ctx._existing_checkpoints[step]
        if cached.node_name != node_def.name:
            import warnings

            warnings.warn(
                f"Fast-forward step {step}: cached node '{cached.node_name}' != "
                f"current node '{node_def.name}'. Graph structure may have changed.",
                stacklevel=2,
            )
        state_type = type(state)
        return _deserialize_state(cached.state_snapshot, state_type)

    # Budget pre-check (wall time)
    await ctx.check_budget(ctx.last_checkpoint_id or "")

    # Create node trace
    node_trace = NodeTrace(
        node_name=node_def.name,
        started_at=datetime.now(timezone.utc),
        completed_at=None,
        input_state=_serialize_state(state),
        output_state=None,
        attempt=1,
        status="running",
    )
    ctx.trace.nodes_executed.append(node_trace)
    prev_node_name = ctx.current_node_name
    prev_node_trace = ctx.current_node_trace
    ctx.current_node_name = node_def.name
    ctx.current_node_trace = node_trace

    try:
        # Wrap in OTel span if configured
        if ctx.otel_tracer is not None:
            with ctx.otel_tracer.start_as_current_span(f"rampart.node.{node_def.name}") as span:
                span.set_attribute("rampart.node_name", node_def.name)
                span.set_attribute("rampart.graph_name", ctx.graph_name)
                span.set_attribute("rampart.run_id", ctx.run_id)
                span.set_attribute("rampart.step", step)
                result = await _execute_with_retry(node_def, state, ctx, node_trace)
        else:
            result = await _execute_with_retry(node_def, state, ctx, node_trace)
    finally:
        ctx.current_node_name = prev_node_name
        ctx.current_node_trace = prev_node_trace

    # Write checkpoint
    ckpt = _make_checkpoint(
        ctx=ctx,
        step=step,
        node_name=node_def.name,
        state=result,
        parent_id=ctx.last_checkpoint_id,
    )
    await ctx.checkpointer.save(ckpt)
    ctx.last_checkpoint_id = ckpt.id

    # Emit streaming event if a consumer is attached
    if ctx.stream_queue is not None:
        await ctx.stream_queue.put(
            GraphEvent(
                type="node_completed" if node_trace.status == "completed" else "node_failed",
                node_name=node_def.name,
                state=result,
                error=node_trace.error,
            )
        )

    return result


async def _execute_with_retry(
    node_def: NodeDef,
    state: AgentState,
    ctx: RunContext,
    node_trace: NodeTrace,
) -> AgentState:
    """Execute a node function with retry logic and timeout."""
    max_attempts = node_def.retries + 1
    last_exc: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        node_trace.attempt = attempt
        node_trace.status = "running"

        try:
            coro = _call_node_fn(node_def, state, ctx)
            if node_def.timeout_seconds:
                result = await asyncio.wait_for(coro, timeout=node_def.timeout_seconds)
            else:
                result = await coro

            node_trace.status = "completed"
            node_trace.completed_at = datetime.now(timezone.utc)
            node_trace.output_state = _serialize_state(result)
            return result

        except BudgetExceededError:
            # Never retry on budget exceeded — propagate immediately
            node_trace.status = "failed"
            node_trace.completed_at = datetime.now(timezone.utc)
            raise

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break

            # Check if this exception type is retriable
            if node_def.retry_on is not None:
                if not isinstance(exc, tuple(node_def.retry_on)):
                    break

            node_trace.status = "retrying"
            backoff_secs = _compute_backoff(node_def.retry_backoff, attempt)
            if backoff_secs > 0:
                await asyncio.sleep(backoff_secs)

    # All retries exhausted
    node_trace.status = "failed"
    node_trace.completed_at = datetime.now(timezone.utc)
    node_trace.error = str(last_exc)
    if last_exc is None:
        raise RuntimeError(
            f"Node '{node_def.name}' exhausted all retries but no exception was captured"
        )
    raise last_exc


async def _call_node_fn(
    node_def: NodeDef,
    state: AgentState,
    ctx: RunContext,
) -> AgentState:
    """Build kwargs and call the node's original function."""
    has_injected = node_def._needs_tools or node_def._needs_llm or node_def._needs_graphs or node_def._needs_artifacts

    # Sandbox: run state-only nodes in an isolated subprocess
    if node_def.sandbox and not has_injected:
        from ._sandbox import run_sandboxed

        return await run_sandboxed(
            fn=node_def.fn,
            state=state,
            state_class=type(state),
        )
    elif node_def.sandbox and has_injected:
        import warnings

        warnings.warn(
            f"Node '{node_def.name}' has sandbox=True but declares injected context "
            f"parameters (tools/llm/graphs/artifacts). Sandbox is not supported for "
            f"nodes with injected contexts — running in-process instead.",
            stacklevel=4,
        )

    kwargs: dict[str, Any] = {}
    if node_def._needs_tools:
        kwargs["tools"] = ToolContext(ctx)
    if node_def._needs_llm:
        kwargs["llm"] = LLMContext(ctx)
    if node_def._needs_graphs:
        kwargs["graphs"] = GraphContext(ctx)
    if node_def._needs_artifacts:
        kwargs["artifacts"] = ArtifactContext(ctx)
    result: AgentState = await node_def.fn(state, **kwargs)
    if not isinstance(result, AgentState):
        raise TypeError(
            f"Node '{node_def.name}' returned {type(result).__name__!r} "
            f"but must return an AgentState subclass"
        )
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_checkpoint(
    ctx: RunContext,
    step: int,
    node_name: str,
    state: AgentState,
    parent_id: str | None,
) -> Checkpoint:
    state_dict = _serialize_state(state)
    state_hash = hashlib.sha256(
        json.dumps(state_dict, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    ckpt_id = f"ckpt_{ctx.graph_name}_{ctx.thread_id}_{step}_{state_hash}"
    return Checkpoint(
        id=ckpt_id,
        thread_id=ctx.thread_id,
        run_id=ctx.run_id,
        graph_name=ctx.graph_name,
        graph_version=ctx.graph_version,
        step=step,
        node_name=node_name,
        state_snapshot=state_dict,
        created_at=datetime.now(timezone.utc),
        parent_checkpoint_id=parent_id,
    )


def _serialize_state(state: AgentState) -> dict[str, Any]:
    import dataclasses

    d = dataclasses.asdict(state)
    d["__type__"] = f"{type(state).__module__}.{type(state).__qualname__}"
    return d


def _deserialize_state(data: dict[str, Any], state_type: type) -> AgentState:
    import dataclasses

    known = {f.name: f for f in dataclasses.fields(state_type)}
    filtered = {}
    for k, v in data.items():
        if k == "__type__":
            continue
        if k in known:
            field_type = known[k].type
            # Attempt to reconstruct nested dataclasses
            if isinstance(v, dict) and isinstance(field_type, type) and dataclasses.is_dataclass(field_type):
                try:
                    v = field_type(**{fk: fv for fk, fv in v.items() if fk in {f.name for f in dataclasses.fields(field_type)}})
                except Exception:
                    pass
            filtered[k] = v
    return state_type(**filtered)


def _infer_state_type(graph_def: GraphDef) -> type:
    """Infer the AgentState subtype from the graph function's annotations.

    Raises ``TypeError`` if no concrete AgentState subclass can be found, rather
    than silently falling back to bare ``AgentState`` (which would lose all
    custom fields when deserializing checkpoint data).
    """
    import inspect
    import typing

    # get_type_hints resolves string annotations (from __future__ annotations)
    try:
        hints = typing.get_type_hints(graph_def.fn)
    except Exception:
        hints = getattr(graph_def.fn, "__annotations__", {})

    params = list(inspect.signature(graph_def.fn).parameters.keys())
    candidates = ["return"] + params
    for key in candidates:
        if key and key in hints:
            t = hints[key]
            if isinstance(t, type) and issubclass(t, AgentState) and t is not AgentState:
                return t

    raise TypeError(
        f"Cannot infer AgentState subclass for graph '{graph_def.name}'. "
        "Annotate the graph function's first parameter or return type with your "
        "AgentState subclass so that checkpoint data can be correctly deserialized "
        "on resume/fork. Example: `async def my_graph(state: MyState) -> MyState:`"
    )


def _compute_backoff(strategy: str, attempt: int) -> float:
    """Return seconds to sleep before the next retry attempt."""
    if strategy == "none":
        return 0.0
    elif strategy == "linear":
        return float(attempt)
    elif strategy == "exponential":
        return 2.0 ** (attempt - 1)
    return 0.0


def _get_testing_state() -> tuple[
    dict[str, Any] | None,
    Any | None,
    str | None,
    Any | None,
    dict[str, Any] | None,
]:
    """Read active testing context (mock_tools / cassette) from contextvars."""
    from .testing._mock_tools import _mock_testing_context

    state = _mock_testing_context.get()
    if state is None:
        return None, None, None, None, None

    from ._models import MockContext

    mock_ctx = MockContext()
    if state.mock_ctx_holder is not None:
        state.mock_ctx_holder.calls = mock_ctx.calls  # share reference

    return (
        state.mock_tools,
        mock_ctx,
        state.cassette_mode,
        state.cassette,
        state.cassette_override_tools,
    )
