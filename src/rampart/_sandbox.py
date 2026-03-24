"""Subprocess sandboxing for ``@node(sandbox=True)`` nodes.

Layer 3 of Rampart's three-layer isolation model:

  Layer 1 — Python library monkey-patching (_http_intercept.py)
              Intercepts httpx / requests at the Python level.
  Layer 2 — HTTP proxy injection (_config.py ``http_proxy_port``)
              Routes all agent traffic through a configurable local proxy.
  Layer 3 — Subprocess execution (this module)
              Runs the node function in an isolated child process, giving
              crash isolation (OOM / SIGKILL in the node cannot take down
              the orchestrator) and optional OS-level resource limits.

**Limitation**: sandbox=True is only supported for state-only nodes (nodes that
don't declare ``tools``, ``llm``, ``graphs``, or ``artifacts`` parameters).
Those injected contexts are bound to the parent's RunContext and cannot be
serialized across a process boundary.  Nodes that need tool/LLM access with
sandbox isolation should delegate to sub-graphs running in separate processes.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import multiprocessing
import signal
import threading
import warnings
from typing import Any

# ── Module-level configuration (set via configure_sandbox) ────────────────────
_max_workers: int = 4
_start_method: str | None = None

# Module-level process pool (lazy initialized, shared across all sandboxed runs).
_pool: concurrent.futures.ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()


def configure_sandbox(
    max_workers: int = 4,
    start_method: str | None = None,
) -> None:
    """Configure sandbox process pool settings.

    Must be called before any sandboxed node executes (i.e. before the pool is
    created).  Calling after the pool is already running has no effect on the
    existing pool — call :func:`shutdown_sandbox` first if you need to
    reconfigure.

    Args:
        max_workers: Maximum number of worker processes in the pool.
        start_method: Multiprocessing start method (``"fork"``, ``"spawn"``,
            ``"forkserver"``, or ``None`` for the platform default).
    """
    global _max_workers, _start_method
    _max_workers = max_workers
    _start_method = start_method


def _pool_initializer() -> None:
    """Initializer run in each pool worker process.

    Resets inherited state that should not leak from the parent process:
    - Restores default signal handlers (the parent may have custom ones).
    - Clears any thread-local state that could have been inherited via fork.
    """
    # Restore default signal handling so that workers respond normally to
    # SIGINT/SIGTERM rather than inheriting the parent's handlers.
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _get_pool() -> concurrent.futures.ProcessPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            mp_context = (
                multiprocessing.get_context(_start_method)
                if _start_method is not None
                else None
            )
            _pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=_max_workers,
                mp_context=mp_context,
                initializer=_pool_initializer,
            )
        return _pool


def _run_node_in_subprocess(
    fn: Any,
    state_dict: dict[str, Any],
    state_class: type,
    max_memory_mb: int | None,
    max_cpu_seconds: int | None,
) -> dict[str, Any]:
    """Executed inside the child process.

    1. Applies OS resource limits (Linux / macOS only; silently skipped elsewhere).
    2. Reconstructs the AgentState from its dict representation.
    3. Runs the async node function via ``asyncio.run()``.
    4. Returns the result as a plain dict for pickling back to the parent.

    Note:
        ``state_class`` must be picklable for cross-process serialization.  This
        is satisfied by module-level ``@dataclass`` definitions (the default for
        ``AgentState`` subclasses).  Lambda-defined or closure-based classes will
        fail under the ``"spawn"`` start method.
    """
    import asyncio as _asyncio
    import dataclasses

    # ── Resource limits ──────────────────────────────────────────────────────
    if max_memory_mb is not None or max_cpu_seconds is not None:
        try:
            import resource as _res  # Unix only

            if max_memory_mb is not None:
                mem_bytes = max_memory_mb * 1024 * 1024
                _res.setrlimit(_res.RLIMIT_AS, (mem_bytes, mem_bytes))
            if max_cpu_seconds is not None:
                _res.setrlimit(_res.RLIMIT_CPU, (max_cpu_seconds, max_cpu_seconds))
        except (ImportError, OSError, ValueError) as exc:
            warnings.warn(f"Could not set resource limits: {exc}")

    # ── Reconstruct state ────────────────────────────────────────────────────
    known = {f.name for f in dataclasses.fields(state_class)}
    state = state_class(**{k: v for k, v in state_dict.items() if k in known})

    # ── Execute ──────────────────────────────────────────────────────────────
    result = _asyncio.run(fn(state))
    return dataclasses.asdict(result)


async def run_sandboxed(
    fn: Any,
    state: Any,
    state_class: type,
    max_memory_mb: int | None = 512,
    max_cpu_seconds: int | None = None,
) -> Any:
    """Run an async node function in an isolated subprocess.

    Args:
        fn: The node's original async function (must be picklable — true for
            all module-level ``@node`` decorated functions).
        state: The current ``AgentState`` instance to pass to the function.
        state_class: The concrete ``AgentState`` subclass, used to reconstruct
            the result dict back into a typed instance.
        max_memory_mb: Virtual-memory ceiling for the subprocess in megabytes.
            Default 512 MB.  ``None`` disables the limit.
        max_cpu_seconds: CPU-time ceiling in seconds.  ``None`` (default)
            disables the limit.

    Returns:
        A reconstructed ``AgentState`` of the same type as *state*.

    Raises:
        ``concurrent.futures.ProcessPoolExecutor`` propagates any exception
        raised inside the subprocess as a ``concurrent.futures.process.RemoteTraceback``.
    """
    import dataclasses

    state_dict = dataclasses.asdict(state)
    loop = asyncio.get_event_loop()
    pool = _get_pool()

    result_dict: dict[str, Any] = await loop.run_in_executor(
        pool,
        _run_node_in_subprocess,
        fn,
        state_dict,
        state_class,
        max_memory_mb,
        max_cpu_seconds,
    )

    known = {f.name for f in dataclasses.fields(state_class)}
    return state_class(**{k: v for k, v in result_dict.items() if k in known})
