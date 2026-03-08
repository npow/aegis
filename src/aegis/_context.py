"""RunContext, ToolContext, LLMContext, GraphContext, and the active-run contextvar."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ._models import (
    AgentState,
    BudgetExceededError,
    BudgetStatus,
    LLMCall,
    LLMNotConfiguredError,
    NodeTrace,
    PermissionScope,
    RunTrace,
    ToolCall,
)

if TYPE_CHECKING:
    from ._models import Budget, CassetteRecord, MockContext, RunConfig
    from .checkpointers._base import CheckpointerBase
    from .testing._mock_tools import MockTool


# ── Active run contextvar ─────────────────────────────────────────────────────

_run_context: ContextVar[RunContext | None] = ContextVar("aegis_run_context", default=None)


# ── RunContext ────────────────────────────────────────────────────────────────


@dataclass
class RunContext:
    """All mutable state for a single graph run."""

    run_id: str
    thread_id: str
    graph_name: str
    graph_version: str
    checkpointer: CheckpointerBase
    trace: RunTrace
    permission_scope: PermissionScope | None
    budget: Budget | None

    # Tool registry: tool_name -> ToolDef
    tool_registry: dict[str, Any] = field(default_factory=dict)

    # Mock overrides: tool_name -> MockTool (set via mock_tools() context)
    mock_tools: dict[str, MockTool] | None = None
    mock_ctx: MockContext | None = None

    # Cassette state
    cassette_mode: str | None = None  # "record" | "replay" | None
    cassette: CassetteRecord | None = None
    cassette_index: int = 0  # next entry to serve in replay
    cassette_override_tools: dict[str, MockTool] | None = None

    # Step tracking
    _step_counter: int = 0
    _fast_forward_to_step: int = -1  # replay through this step
    _existing_checkpoints: dict[int, Any] = field(default_factory=dict)
    last_checkpoint_id: str | None = None

    # Budget tracking (mutable)
    _budget_start_time: float = field(default_factory=time.monotonic)
    _budget_tokens_used: int = 0
    _budget_cost_usd: float = 0.0
    _budget_tool_calls: int = 0
    _budget_exceeded_handler: Any | None = None  # async callable
    _downgraded: bool = False  # True once downgrade_model policy fires

    # Current node context
    current_node_name: str | None = None
    current_node_trace: NodeTrace | None = None

    # LLM model override (for downgrade_model budget policy)
    llm_model_override: str | None = None

    # Streaming: optional queue for real-time node events
    stream_queue: asyncio.Queue | None = None  # type: ignore[type-arg]

    # OTel tracer (opentelemetry Tracer instance, if configured)
    otel_tracer: Any | None = None

    def next_step(self) -> int:
        step = self._step_counter
        self._step_counter += 1
        return step

    def budget_status(self) -> BudgetStatus:
        elapsed = time.monotonic() - self._budget_start_time
        status = BudgetStatus(
            tokens_used=self._budget_tokens_used,
            cost_usd=self._budget_cost_usd,
            tool_calls_made=self._budget_tool_calls,
            wall_time_seconds=elapsed,
        )
        if self.budget:
            status.compute_pct(self.budget)
        return status

    async def check_budget(self, checkpoint_id: str = "") -> None:
        """Raise BudgetExceededError (or apply policy) if any budget dimension is exceeded."""
        if not self.budget:
            return
        status = self.budget_status()
        exceeded: str | None = None

        if self.budget.max_tokens is not None and status.tokens_used > self.budget.max_tokens:
            exceeded = "tokens"
        elif (
            self.budget.max_llm_cost_usd is not None
            and status.cost_usd > self.budget.max_llm_cost_usd
        ):
            exceeded = "cost"
        elif (
            self.budget.max_tool_calls is not None
            and status.tool_calls_made > self.budget.max_tool_calls
        ):
            exceeded = "tool_calls"
        elif self.budget.max_wall_time_seconds is not None:
            elapsed = time.monotonic() - self._budget_start_time
            if elapsed > self.budget.max_wall_time_seconds:
                exceeded = "wall_time"

        if not exceeded:
            return

        from ._models import BudgetExceededEvent

        event = BudgetExceededEvent(
            run_id=self.run_id,
            thread_id=self.thread_id,
            exceeded_dimension=exceeded,
            budget=self.budget,
            current_status=status,
            checkpoint_id=checkpoint_id or self.last_checkpoint_id or "",
        )

        policy = self.budget.on_exceeded

        # ── downgrade_model ────────────────────────────────────────────────────
        if policy == "downgrade_model":
            if not self._downgraded and self.budget.downgrade_to:
                self._downgraded = True
                self.llm_model_override = self.budget.downgrade_to
            # Don't raise — continue with the downgraded model
            return

        # ── pause_and_notify / hard_stop / compress_context ────────────────────
        # Call user-registered handler if present; honour its BudgetDecision.
        if self._budget_exceeded_handler is not None:
            try:
                decision = await self._budget_exceeded_handler(event)
                if decision is not None:
                    if decision.action == "extend" and decision.updated_budget is not None:
                        self.budget = decision.updated_budget
                        return
                    elif decision.action == "downgrade":
                        if decision.updated_budget and decision.updated_budget.downgrade_to:
                            self.llm_model_override = decision.updated_budget.downgrade_to
                        return
                    # else "hard_stop" — fall through to raise
            except Exception as handler_exc:
                import warnings

                warnings.warn(
                    f"Budget exceeded handler raised {type(handler_exc).__name__}: {handler_exc}. "
                    "Falling back to BudgetExceededError.",
                    stacklevel=2,
                )

        raise BudgetExceededError(event)

    async def execute_tool(self, tool_name: str, kwargs: dict[str, Any]) -> Any:
        """Dispatch a tool call through permission checks, mocks, cassette, and real execution."""
        import time as _time

        start = _time.monotonic()
        node_name = self.current_node_name or "unknown"
        call_id = str(uuid.uuid4())[:8]

        # Track budget for ALL tool calls (mocked or real) before dispatch
        self._budget_tool_calls += 1
        await self.check_budget(self.last_checkpoint_id or "")

        # 1. Cassette override_tools take highest priority
        if self.cassette_override_tools and tool_name in self.cassette_override_tools:
            mock = self.cassette_override_tools[tool_name]
            return await self._execute_mock_tool(tool_name, kwargs, mock, call_id, node_name, start)

        # 2. mock_tools() context
        if self.mock_tools and tool_name in self.mock_tools:
            mock = self.mock_tools[tool_name]
            return await self._execute_mock_tool(
                tool_name, kwargs, mock, call_id, node_name, start, record_in_mock_ctx=True
            )

        # 3. Cassette replay
        if self.cassette_mode == "replay" and self.cassette:
            return await self._serve_tool_from_cassette(
                tool_name, kwargs, call_id, node_name, start
            )

        # 4. Permission check
        self._check_tool_permission(tool_name, node_name)

        # 5. Real tool execution
        tool_def = self.tool_registry.get(tool_name)
        if tool_def is None:
            raise KeyError(f"Tool '{tool_name}' not found in registry")

        error: str | None = None
        result: Any = None
        try:
            result = await tool_def.fn(**kwargs)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            latency_ms = int((_time.monotonic() - start) * 1000)
            tc = ToolCall(
                call_id=call_id,
                tool_name=tool_name,
                args=kwargs,
                result=result,
                error=error,
                latency_ms=latency_ms,
                timestamp=datetime.utcnow(),
                node_name=node_name,
                permission_checked=True,
                permission_granted=True,
            )
            if self.current_node_trace:
                self.current_node_trace.tool_calls.append(tc)

            # Record in cassette if recording
            if self.cassette_mode == "record" and self.cassette is not None:
                from ._models import CassetteEntry

                entry = CassetteEntry(
                    type="tool_call",
                    call_id=call_id,
                    step=self._step_counter - 1,
                    node_name=node_name,
                    request={"tool_name": tool_name, "args": _safe_serialize(kwargs)},
                    response={"result": _safe_serialize(result), "error": error},
                    timestamp=datetime.utcnow(),
                )
                self.cassette.entries.append(entry)

        return result

    async def _execute_mock_tool(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        mock: MockTool,
        call_id: str,
        node_name: str,
        start: float,
        record_in_mock_ctx: bool = False,
    ) -> Any:
        import time as _time

        error: str | None = None
        result: Any = None
        try:
            result = await mock.execute(kwargs)
            return result
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            latency_ms = int((_time.monotonic() - start) * 1000)
            tc = ToolCall(
                call_id=call_id,
                tool_name=tool_name,
                args=kwargs,
                result=result,
                error=error,
                latency_ms=latency_ms,
                timestamp=datetime.utcnow(),
                node_name=node_name,
                was_mocked=True,
                permission_checked=False,
                permission_granted=True,
            )
            if self.current_node_trace:
                self.current_node_trace.tool_calls.append(tc)

            if record_in_mock_ctx and self.mock_ctx is not None:
                from ._models import MockCallRecord

                if tool_name not in self.mock_ctx.calls:
                    self.mock_ctx.calls[tool_name] = MockCallRecord(tool_name=tool_name)
                rec = self.mock_ctx.calls[tool_name]
                rec.count += 1
                rec.calls.append(tc)

            # Record mock result in cassette if we're in record mode
            if self.cassette_mode == "record" and self.cassette is not None:
                from ._models import CassetteEntry

                cassette_entry = CassetteEntry(
                    type="tool_call",
                    call_id=call_id,
                    step=self._step_counter - 1,
                    node_name=node_name,
                    request={"tool_name": tool_name, "args": _safe_serialize(kwargs)},
                    response={"result": _safe_serialize(result), "error": error},
                    timestamp=datetime.utcnow(),
                )
                self.cassette.entries.append(cassette_entry)

    async def _serve_tool_from_cassette(
        self,
        tool_name: str,
        kwargs: dict[str, Any],
        call_id: str,
        node_name: str,
        start: float,
    ) -> Any:
        import time as _time

        from ._models import AegisCassetteStaleError

        if self.cassette is None:  # pragma: no cover
            raise RuntimeError("_serve_tool_from_cassette called without active cassette")
        # Find next tool_call entry in cassette
        entries = self.cassette.entries
        idx = self.cassette_index
        while idx < len(entries) and entries[idx].type != "tool_call":
            idx += 1
        if idx >= len(entries):
            raise AegisCassetteStaleError(
                f"Cassette exhausted: no more tool calls recorded "
                f"(tool='{tool_name}', step={self._step_counter})"
            )

        entry = entries[idx]
        self.cassette_index = idx + 1
        self._sync_cassette_index()

        # Validate tool name match
        recorded_tool = entry.request.get("tool_name")
        if recorded_tool != tool_name:
            raise AegisCassetteStaleError(
                f"Cassette divergence at index {idx}: "
                f"expected tool '{recorded_tool}', got '{tool_name}'"
            )

        if entry.response.get("error"):
            raise RuntimeError(entry.response["error"])

        result = entry.response.get("result")
        latency_ms = int((_time.monotonic() - start) * 1000)
        tc = ToolCall(
            call_id=call_id,
            tool_name=tool_name,
            args=kwargs,
            result=result,
            error=None,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
            node_name=node_name,
            was_mocked=False,
        )
        if self.current_node_trace:
            self.current_node_trace.tool_calls.append(tc)
        return result

    def _sync_cassette_index(self) -> None:
        """Propagate cassette_index back to the testing context's mutable tracker."""
        from .testing._mock_tools import _mock_testing_context

        ts = _mock_testing_context.get()
        if ts is not None:
            ts.cassette_index_ref[0] = self.cassette_index

    def _check_tool_permission(self, tool_name: str, node_name: str) -> None:
        if not self.permission_scope:
            return
        scope = self.permission_scope
        if scope.tools is not None and tool_name not in scope.tools:
            from datetime import datetime

            from ._models import PermissionDeniedError, PermissionViolationEvent

            event = PermissionViolationEvent(
                run_id=self.run_id,
                thread_id=self.thread_id,
                node_name=node_name,
                violation_type="tool_not_in_whitelist",
                attempted_action=f"call tool '{tool_name}'",
                declared_scope=scope,
                timestamp=datetime.utcnow(),
            )
            raise PermissionDeniedError(event)

    async def execute_llm_call(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute an LLM call, serving from cassette if in replay mode."""
        import time as _time

        # Apply model override from budget downgrade
        effective_model = self.llm_model_override or model
        node_name = self.current_node_name or "unknown"
        call_id = str(uuid.uuid4())[:8]
        start = _time.monotonic()

        # Cassette replay
        if self.cassette_mode == "replay" and self.cassette:
            return await self._serve_llm_from_cassette(
                effective_model, prompt, system, call_id, node_name, start, kwargs
            )

        # Live LLM call — try litellm
        try:
            import litellm  # type: ignore[import]
        except ImportError as exc:
            raise LLMNotConfiguredError(
                "No LLM provider configured. Install aegis[litellm] and set your API key, "
                "or use cassette replay / mock_tools() for testing."
            ) from exc

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await litellm.acompletion(model=effective_model, messages=messages, **kwargs)
        text = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cached_tokens = getattr(response.usage, "cache_read_input_tokens", 0)
        cost_usd = getattr(response, "_hidden_params", {}).get("response_cost", 0.0) or 0.0

        latency_ms = int((_time.monotonic() - start) * 1000)
        llm_call = LLMCall(
            call_id=call_id,
            model=effective_model,
            system_prompt=system,
            user_prompt=prompt,
            response=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
            node_name=node_name,
        )

        await self._record_llm_call(llm_call)

        # Record in cassette if recording
        if self.cassette_mode == "record" and self.cassette is not None:
            from ._models import CassetteEntry

            entry = CassetteEntry(
                type="llm_call",
                call_id=call_id,
                step=self._step_counter - 1,
                node_name=node_name,
                request=_safe_serialize(
                    {"model": effective_model, "system": system, "prompt": prompt, **kwargs}
                ),
                response=_safe_serialize(
                    {
                        "text": text,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cost_usd": cost_usd,
                    }
                ),
                timestamp=datetime.utcnow(),
            )
            self.cassette.entries.append(entry)

        return LLMResponse(text=text, call=llm_call)

    async def _serve_llm_from_cassette(
        self,
        model: str,
        prompt: str,
        system: str | None,
        call_id: str,
        node_name: str,
        start: float,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> LLMResponse:
        import time as _time

        from ._models import AegisCassetteStaleError

        if self.cassette is None:  # pragma: no cover
            raise RuntimeError("_serve_llm_from_cassette called without active cassette")
        entries = self.cassette.entries
        idx = self.cassette_index
        while idx < len(entries) and entries[idx].type != "llm_call":
            idx += 1
        if idx >= len(entries):
            raise AegisCassetteStaleError(
                f"Cassette exhausted: no more LLM calls recorded (node={node_name})"
            )

        entry = entries[idx]
        self.cassette_index = idx + 1
        self._sync_cassette_index()

        # Stale detection: compare request hash (model + prompts + extra kwargs)
        import hashlib
        import json as _json

        current_req = {"model": model, "system": system, "prompt": prompt, **(extra_kwargs or {})}
        current_hash = hashlib.sha256(
            _json.dumps(current_req, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        recorded_hash = hashlib.sha256(
            _json.dumps(entry.request, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        if current_hash != recorded_hash:
            raise AegisCassetteStaleError(
                f"Cassette '{node_name}' is stale at index {idx} (llm_call):\n"
                f"  prompt hash changed\n"
                f"  Recorded:  sha256:{recorded_hash}\n"
                f"  Current:   sha256:{current_hash}\n\n"
                f"Re-record with: pytest --aegis-record"
            )

        resp = entry.response
        latency_ms = int((_time.monotonic() - start) * 1000)
        llm_call = LLMCall(
            call_id=call_id,
            model=model,
            system_prompt=system,
            user_prompt=prompt,
            response=resp.get("text", ""),
            input_tokens=resp.get("input_tokens", 0),
            output_tokens=resp.get("output_tokens", 0),
            cached_tokens=0,
            cost_usd=resp.get("cost_usd", 0.0),
            latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
            node_name=node_name,
            was_replayed=True,
        )
        await self._record_llm_call(llm_call)
        return LLMResponse(text=resp.get("text", ""), call=llm_call)

    async def _record_llm_call(self, llm_call: LLMCall) -> None:
        self._budget_tokens_used += llm_call.input_tokens + llm_call.output_tokens
        self._budget_cost_usd += llm_call.cost_usd
        self.trace.add_llm_call(llm_call)
        if self.current_node_trace:
            self.current_node_trace.llm_calls.append(llm_call)
        await self.check_budget(self.last_checkpoint_id or "")


# ── LLM Response ──────────────────────────────────────────────────────────────


@dataclass
class LLMResponse:
    text: str
    call: LLMCall


# ── ToolContext ───────────────────────────────────────────────────────────────


class ToolContext:
    """Injected into node functions via 'tools' parameter.

    Attribute access returns an async callable that dispatches to the named tool.
    """

    def __init__(self, ctx: RunContext) -> None:
        self._ctx = ctx

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        async def _dispatch(**kwargs: Any) -> Any:
            return await self._ctx.execute_tool(name, kwargs)

        _dispatch.__name__ = name
        return _dispatch


# ── LLMContext ────────────────────────────────────────────────────────────────


class LLMContext:
    """Injected into node functions via 'llm' parameter."""

    def __init__(self, ctx: RunContext) -> None:
        self._ctx = ctx

    async def complete(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return await self._ctx.execute_llm_call(model=model, prompt=prompt, system=system, **kwargs)


# ── GraphContext ──────────────────────────────────────────────────────────────


class GraphContext:
    """Injected into node functions via 'graphs' parameter.

    Allows calling sub-graphs from within a node with automatic OTel propagation.
    """

    def __init__(self, ctx: RunContext) -> None:
        self._ctx = ctx

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        # Return a proxy that delegates .run() etc. to the named graph
        from ._decorators import _GRAPH_REGISTRY

        graph_def = _GRAPH_REGISTRY.get(name)
        if graph_def is None:
            raise KeyError(f"Graph '{name}' not found in registry")

        _gdef = graph_def  # narrow to non-Optional for closure

        class _SubGraphProxy:
            async def run(proxy_self, input: AgentState, config: RunConfig, **kw: Any) -> Any:  # noqa: N805
                return await _gdef.run(input=input, config=config, **kw)

        return _SubGraphProxy()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _safe_serialize(obj: Any) -> Any:
    """Best-effort JSON-safe serialization."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    try:
        import dataclasses as dc

        if dc.is_dataclass(obj) and not isinstance(obj, type):
            return dc.asdict(obj)
    except Exception:
        pass
    return str(obj)
