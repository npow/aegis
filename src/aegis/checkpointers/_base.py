"""Abstract base protocol for checkpoint backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .._models import Checkpoint


@runtime_checkable
class CheckpointerBase(Protocol):
    """Protocol that all checkpoint backends must implement.

    Backends must also support use as an async context manager so callers
    can guarantee cleanup on shutdown::

        async with PostgresCheckpointer(url) as cp:
            aegis.configure(checkpointer=cp)
            ...
    """

    async def save(self, checkpoint: Checkpoint) -> None:
        """Persist a checkpoint atomically."""
        ...

    async def get_latest(self, thread_id: str, graph_name: str) -> Checkpoint | None:
        """Return the most recent checkpoint for a thread, or None."""
        ...

    async def get_by_step(self, thread_id: str, graph_name: str, step: int) -> Checkpoint | None:
        """Return the checkpoint at a specific step, or None."""
        ...

    async def get_history(self, thread_id: str, graph_name: str) -> list[Checkpoint]:
        """Return all checkpoints for a thread, ordered by step ascending."""
        ...

    async def delete_thread(self, thread_id: str, graph_name: str) -> None:
        """Delete all checkpoints for a thread (e.g., for cleanup after fork)."""
        ...

    async def close(self) -> None:
        """Release all resources (connections, file handles, etc.)."""
        ...

    async def __aenter__(self) -> CheckpointerBase:
        """Enter async context manager — returns self."""
        ...

    async def __aexit__(self, *args: object) -> None:
        """Exit async context manager — calls close()."""
        ...
