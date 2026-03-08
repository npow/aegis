"""MockTool — first-class tool mocking for Aegis tests."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

# ── Shared testing contextvar ─────────────────────────────────────────────────


@dataclass
class _TestingState:
    mock_tools: dict[str, MockTool] | None = None
    mock_ctx_holder: Any | None = None  # MockToolsContext
    cassette_mode: str | None = None
    cassette: Any | None = None
    cassette_override_tools: dict[str, MockTool] | None = None
    # Mutable box tracking actual cassette entries served (updated by RunContext)
    cassette_index_ref: list = field(default_factory=lambda: [0])


_mock_testing_context: ContextVar[_TestingState | None] = ContextVar(
    "aegis_testing_context", default=None
)


# ── MockTool ──────────────────────────────────────────────────────────────────


class MockTool:
    """A tool replacement that returns a fixed value, raises an exception, or executes a custom callable."""

    def __init__(
        self,
        *,
        return_value: Any = None,
        raise_exception: BaseException | None = None,
        side_effect: Callable[..., Any] | None = None,
        noop: bool = False,
    ) -> None:
        self._return_value = return_value
        self._raise_exception = raise_exception
        self._side_effect = side_effect
        self._noop = noop

    async def execute(self, kwargs: dict[str, Any]) -> Any:
        if self._raise_exception is not None:
            raise self._raise_exception
        if self._noop:
            return None
        if self._side_effect is not None:
            import asyncio

            result = self._side_effect(**kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return self._return_value

    # ── Factory methods (matches PRD API) ─────────────────────────────────────

    @classmethod
    def returns(cls, value: Any) -> MockTool:
        """Return a fixed value on every call."""
        return cls(return_value=value)

    @classmethod
    def noop(cls) -> MockTool:
        """Return None on every call (no-op)."""
        return cls(noop=True)

    @classmethod
    def raises(cls, exc: BaseException) -> MockTool:
        """Raise the given exception on every call."""
        return cls(raise_exception=exc)

    @classmethod
    def calls(cls, fn: Callable[..., Any]) -> MockTool:
        """Delegate to a callable (sync or async)."""
        return cls(side_effect=fn)
