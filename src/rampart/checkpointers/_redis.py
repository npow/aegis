"""Redis checkpoint backend — for production multi-process deployments."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .._models import Checkpoint


class RedisCheckpointer:
    """Checkpoint store backed by Redis (requires redis>=5.0: ``pip install rampart[redis]``).

    Uses one Redis hash per (graph_name, thread_id) pair. The hash field is the
    step number (as a string); the value is a JSON-serialized checkpoint.  This
    gives O(1) save / get-by-step with correct idempotent-replace semantics.

    Example::

        async with RedisCheckpointer("redis://localhost:6379") as cp:
            rampart.configure(checkpointer=cp)
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        key_prefix: str = "rampart",
        ttl_days: int | None = None,
    ) -> None:
        self._url = url
        self._key_prefix = key_prefix
        self._ttl_seconds = int(ttl_days * 86400) if ttl_days else None
        self._client: Any | None = None
        self._client_lock: Any = None  # asyncio.Lock, created lazily

    async def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        import asyncio as _asyncio
        if self._client_lock is None:
            self._client_lock = _asyncio.Lock()
        async with self._client_lock:
            if self._client is not None:
                return self._client
            return await self._init_client()

    async def _init_client(self) -> Any:
        try:
            from redis.asyncio import Redis  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "redis is required for RedisCheckpointer. "
                "Install it with: pip install rampart[redis]"
            ) from exc
        self._client = await Redis.from_url(self._url, decode_responses=True)
        return self._client

    def _hash_key(self, thread_id: str, graph_name: str) -> str:
        return f"{self._key_prefix}:{graph_name}:{thread_id}"

    @staticmethod
    def _serialize(checkpoint: Checkpoint) -> str:
        import dataclasses

        data = dataclasses.asdict(checkpoint)
        data["created_at"] = checkpoint.created_at.isoformat()
        return json.dumps(data, default=str)

    @staticmethod
    def _deserialize(raw: str) -> Checkpoint:
        data = json.loads(raw)
        return Checkpoint(
            id=data["id"],
            thread_id=data["thread_id"],
            run_id=data["run_id"],
            graph_name=data["graph_name"],
            graph_version=data["graph_version"],
            step=data["step"],
            node_name=data["node_name"],
            state_snapshot=data["state_snapshot"],
            created_at=datetime.fromisoformat(data["created_at"]),
            parent_checkpoint_id=data.get("parent_checkpoint_id"),
            is_fork_root=bool(data.get("is_fork_root", False)),
        )

    async def save(self, checkpoint: Checkpoint) -> None:
        r = await self._get_client()
        key = self._hash_key(checkpoint.thread_id, checkpoint.graph_name)
        await r.hset(key, str(checkpoint.step), self._serialize(checkpoint))
        if self._ttl_seconds:
            await r.expire(key, self._ttl_seconds)

    async def get_latest(self, thread_id: str, graph_name: str) -> Checkpoint | None:
        r = await self._get_client()
        key = self._hash_key(thread_id, graph_name)
        all_fields: dict[str, str] = await r.hgetall(key)
        if not all_fields:
            return None
        max_field = max(all_fields, key=lambda k: int(k))
        return self._deserialize(all_fields[max_field])

    async def get_by_step(self, thread_id: str, graph_name: str, step: int) -> Checkpoint | None:
        r = await self._get_client()
        key = self._hash_key(thread_id, graph_name)
        raw: str | None = await r.hget(key, str(step))
        return self._deserialize(raw) if raw else None

    async def get_history(self, thread_id: str, graph_name: str) -> list[Checkpoint]:
        r = await self._get_client()
        key = self._hash_key(thread_id, graph_name)
        all_fields: dict[str, str] = await r.hgetall(key)
        if not all_fields:
            return []
        sorted_items = sorted(all_fields.items(), key=lambda kv: int(kv[0]))
        return [self._deserialize(v) for _, v in sorted_items]

    async def delete_thread(self, thread_id: str, graph_name: str) -> None:
        r = await self._get_client()
        key = self._hash_key(thread_id, graph_name)
        await r.delete(key)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> RedisCheckpointer:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
