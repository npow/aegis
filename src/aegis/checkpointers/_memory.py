"""In-memory checkpoint backend — for development and testing."""

from __future__ import annotations

import asyncio

from .._models import Checkpoint


class MemoryCheckpointer:
    """In-process checkpoint store. No persistence across process restarts.

    All mutations are protected by an asyncio.Lock to prevent race conditions
    when multiple nodes or parallel branches checkpoint concurrently.

    Supports use as an async context manager::

        async with MemoryCheckpointer() as cp:
            await cp.save(checkpoint)
    """

    def __init__(self) -> None:
        # Key: (thread_id, graph_name) -> list[Checkpoint] ordered by step
        self._store: dict[tuple[str, str], list[Checkpoint]] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    async def close(self) -> None:
        """No-op for in-memory store (no resources to release)."""

    async def __aenter__(self) -> MemoryCheckpointer:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def save(self, checkpoint: Checkpoint) -> None:
        key = (checkpoint.thread_id, checkpoint.graph_name)
        async with self._lock:
            if key not in self._store:
                self._store[key] = []
            # Remove any existing checkpoint at the same step (idempotent save)
            self._store[key] = [c for c in self._store[key] if c.step != checkpoint.step]
            self._store[key].append(checkpoint)
            self._store[key].sort(key=lambda c: c.step)

    async def get_latest(self, thread_id: str, graph_name: str) -> Checkpoint | None:
        key = (thread_id, graph_name)
        checkpoints = self._store.get(key, [])
        return checkpoints[-1] if checkpoints else None

    async def get_by_step(self, thread_id: str, graph_name: str, step: int) -> Checkpoint | None:
        key = (thread_id, graph_name)
        for ckpt in self._store.get(key, []):
            if ckpt.step == step:
                return ckpt
        return None

    async def get_history(self, thread_id: str, graph_name: str) -> list[Checkpoint]:
        key = (thread_id, graph_name)
        return list(self._store.get(key, []))

    async def delete_thread(self, thread_id: str, graph_name: str) -> None:
        key = (thread_id, graph_name)
        async with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Clear all checkpoints (useful between tests)."""
        self._store.clear()
