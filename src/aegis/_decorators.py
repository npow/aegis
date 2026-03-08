"""@graph, @node, and @tool decorators plus their descriptor classes."""

from __future__ import annotations

import inspect
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from ._models import (
    AgentState,
    Budget,
    PermissionScope,
    RunConfig,
    RunResult,
)

F = TypeVar("F", bound=Callable[..., Any])

# ── Global registries ─────────────────────────────────────────────────────────

_TOOL_REGISTRY: dict[str, ToolDef] = {}
_GRAPH_REGISTRY: dict[str, GraphDef] = {}


def get_tool_registry() -> dict[str, ToolDef]:
    return dict(_TOOL_REGISTRY)


def get_graph_registry() -> dict[str, GraphDef]:
    return dict(_GRAPH_REGISTRY)


# ── ToolDef ───────────────────────────────────────────────────────────────────


class ToolDef:
    """Descriptor wrapping a @tool-decorated function."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str,
        description: str = "",
        permissions: Any | None = None,  # ToolPermission
        require_human_approval: bool = False,
        approval_timeout_seconds: int = 3600,
        approval_on_timeout: str = "hard_stop",
    ) -> None:
        self.fn = fn
        self.name = name
        self.description = description
        self.permissions = permissions
        self.require_human_approval = require_human_approval
        self.approval_timeout_seconds = approval_timeout_seconds
        self.approval_on_timeout = approval_on_timeout
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

    async def __call__(self, **kwargs: Any) -> Any:
        return await self.fn(**kwargs)


# ── NodeDef ───────────────────────────────────────────────────────────────────


class NodeDef:
    """Descriptor wrapping a @node-decorated function."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        retries: int = 0,
        retry_backoff: str = "none",
        retry_on: tuple[type, ...] | None = None,
        timeout_seconds: float | None = None,
        sandbox: bool = False,
    ) -> None:
        if retry_on is not None:
            bad = [
                t for t in retry_on if not (isinstance(t, type) and issubclass(t, BaseException))
            ]
            if bad:
                raise TypeError(f"@node retry_on must contain only exception classes, got: {bad!r}")
        self.fn = fn
        self.name = fn.__name__
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.retry_on = retry_on
        self.timeout_seconds = timeout_seconds
        self.sandbox = sandbox
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__

        # Inspect parameter names at decoration time
        sig = inspect.signature(fn)
        self._params = list(sig.parameters.keys())
        self._needs_tools = "tools" in self._params
        self._needs_llm = "llm" in self._params
        self._needs_graphs = "graphs" in self._params

    async def __call__(self, state: AgentState, **kwargs: Any) -> AgentState:
        from ._context import _run_context

        ctx = _run_context.get()
        if ctx is None:
            # Standalone — inject whatever was provided, execute directly
            return await self._call_direct(state, **kwargs)
        else:
            # Inside a graph run — full framework machinery
            from ._runtime import _execute_node_in_context

            return await _execute_node_in_context(self, state, ctx)

    async def _call_direct(self, state: AgentState, **kwargs: Any) -> AgentState:
        """Execute without framework machinery (for standalone / direct tests)."""
        call_kwargs: dict[str, Any] = dict(kwargs)
        result: AgentState = await self.fn(state, **call_kwargs)
        return result


# ── GraphDef ──────────────────────────────────────────────────────────────────


class GraphDef:
    """Descriptor wrapping a @graph-decorated function.

    Provides run(), resume(), fork(), stream(), and testing helpers.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str,
        version: str = "1.0.0",
        checkpointer: str = "memory",
        permissions: PermissionScope | None = None,
        budget: Budget | None = None,
    ) -> None:
        self.fn = fn
        self.name = name
        self.version = version
        self._checkpointer_type = checkpointer
        self.permissions = permissions
        self.default_budget = budget
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self._budget_exceeded_handler: Callable[..., Any] | None = None
        # Register in global registry
        _GRAPH_REGISTRY[name] = self

    # ── Core methods ──────────────────────────────────────────────────────────

    async def run(
        self,
        input: AgentState,
        config: RunConfig,
        budget: Budget | None = None,
    ) -> RunResult:
        from ._runtime import _run_graph

        return await _run_graph(
            self,
            input_state=input,
            config=config,
            budget=budget or self.default_budget,
        )

    async def resume(self, thread_id: str, config: RunConfig | None = None) -> RunResult:
        """Resume from the last committed checkpoint for this thread."""
        from ._runtime import _resume_graph

        if config is None:
            config = RunConfig(thread_id=thread_id)
        return await _resume_graph(self, thread_id=thread_id, config=config)

    async def fork(
        self,
        thread_id: str,
        checkpoint_id: str,
        inject_state: dict[str, Any] | None = None,
        new_thread_id: str | None = None,
        config: RunConfig | None = None,
    ) -> RunResult:
        """Fork from a specific checkpoint, optionally injecting state changes.

        Pass ``config`` to specify which checkpointer holds the original thread's
        history (required when you ran the graph with a custom ``config.checkpointer``).
        """
        from ._runtime import _fork_graph

        return await _fork_graph(
            self,
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            inject_state=inject_state or {},
            new_thread_id=new_thread_id or f"{thread_id}-fork-{uuid.uuid4().hex[:6]}",
            config=config,
        )

    async def get_checkpoint_history(
        self, thread_id: str, config: RunConfig | None = None
    ) -> list[Any]:
        """Return all checkpoints for a thread, ordered by step.

        Pass ``config`` to specify which checkpointer to read from when you used
        a custom ``config.checkpointer`` during the original run.
        """
        checkpointer = self._resolve_checkpointer(config)
        return await checkpointer.get_history(thread_id, self.name)  # type: ignore[no-any-return]

    async def stream(
        self,
        input: AgentState,
        config: RunConfig,
        budget: Budget | None = None,
    ) -> AsyncIterator[Any]:
        """Stream intermediate node events during execution."""
        from ._runtime import _stream_graph

        async for event in _stream_graph(self, input_state=input, config=config, budget=budget):
            yield event

    # ── Testing helpers ───────────────────────────────────────────────────────

    @asynccontextmanager
    async def mock_tools(self, overrides: dict[str, Any]) -> AsyncIterator[MockToolsContext]:
        """Context manager: replace named tools with MockTool instances for this block."""
        ctx = MockToolsContext(overrides)
        async with ctx:
            yield ctx

    # ── Budget exceeded handler ───────────────────────────────────────────────

    def on_budget_exceeded(self, handler: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator: register a handler called when the budget is exceeded."""
        self._budget_exceeded_handler = handler
        return handler

    # ── Permissions display ───────────────────────────────────────────────────

    def get_permissions(self) -> PermissionScope | None:
        return self.permissions

    # ── Internal ──────────────────────────────────────────────────────────────

    def _resolve_checkpointer(self, config: RunConfig | None) -> Any:  # CheckpointerBase
        """Return the checkpointer to use, in priority order:
        1. config.checkpointer (per-run override)
        2. aegis.configure(checkpointer=...) global default
        3. graph-level default (memory or sqlite based on @graph(checkpointer=...))
        """
        from .checkpointers._memory import MemoryCheckpointer
        from .checkpointers._sqlite import SqliteCheckpointer

        if config is not None and config.checkpointer is not None:
            return config.checkpointer

        # Check global default set via aegis.configure()
        from . import _globals

        if _globals.DEFAULT_CHECKPOINTER is not None:
            return _globals.DEFAULT_CHECKPOINTER

        if self._checkpointer_type == "memory":
            # Use a shared in-process store keyed by graph name
            if not hasattr(self, "_memory_checkpointer"):
                self._memory_checkpointer = MemoryCheckpointer()
            return self._memory_checkpointer
        elif self._checkpointer_type == "sqlite":
            if not hasattr(self, "_sqlite_checkpointer"):
                self._sqlite_checkpointer = SqliteCheckpointer()
            return self._sqlite_checkpointer
        else:
            # postgres / redis / dynamodb — default to memory with a warning
            import warnings

            warnings.warn(
                f"Checkpointer type '{self._checkpointer_type}' not configured; "
                "falling back to in-memory. Set config.checkpointer explicitly.",
                stacklevel=3,
            )
            if not hasattr(self, "_memory_checkpointer"):
                self._memory_checkpointer = MemoryCheckpointer()
            return self._memory_checkpointer


# ── @graph decorator ──────────────────────────────────────────────────────────


def graph(
    name: str,
    *,
    version: str = "1.0.0",
    checkpointer: str = "memory",
    permissions: PermissionScope | None = None,
    budget: Budget | None = None,
) -> Callable[[F], GraphDef]:
    """Decorator that turns an async function into an Aegis graph."""

    def decorator(fn: F) -> GraphDef:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@graph: '{fn.__name__}' must be an async function (use 'async def')")
        return GraphDef(
            fn,
            name=name,
            version=version,
            checkpointer=checkpointer,
            permissions=permissions,
            budget=budget,
        )

    return decorator


# ── @node decorator ───────────────────────────────────────────────────────────


def node(
    _fn: Callable[..., Any] | None = None,
    *,
    retries: int = 0,
    retry_backoff: str = "none",
    retry_on: tuple[type, ...] | None = None,
    timeout_seconds: float | None = None,
    sandbox: bool = False,
) -> Any:
    """Decorator that wraps an async node function with Aegis execution machinery."""

    def decorator(fn: Callable[..., Any]) -> NodeDef:
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"@node: '{fn.__name__}' must be an async function (use 'async def')")
        return NodeDef(
            fn,
            retries=retries,
            retry_backoff=retry_backoff,
            retry_on=retry_on,
            timeout_seconds=timeout_seconds,
            sandbox=sandbox,
        )

    if _fn is not None:
        # Called as @node without arguments
        return decorator(_fn)
    return decorator


# ── @tool decorator ───────────────────────────────────────────────────────────


def tool(
    _fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str = "",
    permissions: Any | None = None,
    require_human_approval: bool = False,
    approval_timeout_seconds: int = 3600,
    approval_on_timeout: str = "hard_stop",
) -> Any:
    """Decorator that registers a function as an Aegis tool."""

    def decorator(fn: Callable[..., Any]) -> ToolDef:
        tool_name = name or fn.__name__
        td = ToolDef(
            fn,
            name=tool_name,
            description=description,
            permissions=permissions,
            require_human_approval=require_human_approval,
            approval_timeout_seconds=approval_timeout_seconds,
            approval_on_timeout=approval_on_timeout,
        )
        _TOOL_REGISTRY[tool_name] = td
        return td

    if _fn is not None:
        return decorator(_fn)
    return decorator


# ── MockToolsContext (used by GraphDef.mock_tools()) ─────────────────────────


class MockToolsContext:
    """Async context manager returned by graph.mock_tools()."""

    def __init__(self, overrides: dict[str, Any]) -> None:
        self._overrides = overrides
        self._mock_ctx_var_token: Any = None
        self.calls: dict[str, Any] = {}  # populated during run

    async def __aenter__(self) -> MockToolsContext:
        from .testing._mock_tools import _mock_testing_context, _TestingState

        # Merge with existing state (e.g., preserve cassette context)
        existing = _mock_testing_context.get()
        if existing is not None:
            import dataclasses

            state = dataclasses.replace(
                existing,
                mock_tools={**(existing.mock_tools or {}), **self._overrides},
                mock_ctx_holder=self,
            )
        else:
            state = _TestingState(mock_tools=self._overrides, mock_ctx_holder=self)
        self._mock_ctx_var_token = _mock_testing_context.set(state)
        return self

    async def __aexit__(self, *args: Any) -> None:
        from .testing._mock_tools import _mock_testing_context

        if self._mock_ctx_var_token is not None:
            _mock_testing_context.reset(self._mock_ctx_var_token)
