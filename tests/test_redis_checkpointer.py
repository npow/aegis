"""Tests for RedisCheckpointer using a fake Redis client."""
from __future__ import annotations

from datetime import datetime

import pytest

from rampart import Checkpoint
from rampart.checkpointers import RedisCheckpointer


def _make_ckpt(step: int, thread_id: str = "t1", graph_name: str = "g") -> Checkpoint:
    return Checkpoint(
        id=f"ckpt_{graph_name}_{thread_id}_{step}_abc",
        thread_id=thread_id,
        run_id="r1",
        graph_name=graph_name,
        graph_version="1.0",
        step=step,
        node_name=f"node_{step}",
        state_snapshot={"value": step, "thread_id": thread_id, "run_id": "r1"},
        created_at=datetime(2025, 1, 1, 12, 0, 0),
        parent_checkpoint_id=None,
    )


def _make_fake_redis():
    """Build an in-memory fake Redis client that mirrors the hset/hget/hgetall API."""
    store: dict[str, dict[str, str]] = {}

    class FakeRedis:
        async def hset(self, key: str, field: str, value: str) -> None:
            store.setdefault(key, {})[field] = value

        async def hget(self, key: str, field: str) -> str | None:
            return store.get(key, {}).get(field)

        async def hgetall(self, key: str) -> dict[str, str]:
            return dict(store.get(key, {}))

        async def expire(self, key: str, seconds: int) -> None:
            pass  # TTL not simulated

        async def delete(self, key: str) -> None:
            store.pop(key, None)

        async def aclose(self) -> None:
            pass

    fake = FakeRedis()

    async def from_url(url: str, **kwargs: object) -> FakeRedis:
        return fake

    return fake, from_url


@pytest.fixture
def fake_redis_checkpointer():
    """Return a RedisCheckpointer wired to a fake in-memory Redis."""
    fake, from_url = _make_fake_redis()
    cp = RedisCheckpointer(url="redis://fake:6379", key_prefix="test")

    # _get_client does "from redis.asyncio import Redis" then "Redis.from_url(...)".
    # We inject the fake by pre-setting the internal _client directly.
    cp._client = fake
    yield cp


# ── Key format ────────────────────────────────────────────────────────────────


def test_hash_key_format():
    cp = RedisCheckpointer(key_prefix="rp")
    assert cp._hash_key("thread1", "my_graph") == "rp:my_graph:thread1"


# ── Round-trip save / get_latest / get_by_step ────────────────────────────────


@pytest.mark.asyncio
async def test_save_and_get_latest(fake_redis_checkpointer):
    cp = fake_redis_checkpointer
    await cp.save(_make_ckpt(0))
    await cp.save(_make_ckpt(1))
    await cp.save(_make_ckpt(2))

    latest = await cp.get_latest("t1", "g")
    assert latest is not None
    assert latest.step == 2


@pytest.mark.asyncio
async def test_get_by_step(fake_redis_checkpointer):
    cp = fake_redis_checkpointer
    await cp.save(_make_ckpt(0))
    await cp.save(_make_ckpt(3))

    ckpt = await cp.get_by_step("t1", "g", 3)
    assert ckpt is not None
    assert ckpt.step == 3

    missing = await cp.get_by_step("t1", "g", 99)
    assert missing is None


@pytest.mark.asyncio
async def test_get_history_ordered(fake_redis_checkpointer):
    cp = fake_redis_checkpointer
    for step in [2, 0, 1]:
        await cp.save(_make_ckpt(step))

    history = await cp.get_history("t1", "g")
    assert [c.step for c in history] == [0, 1, 2]


@pytest.mark.asyncio
async def test_get_latest_returns_none_when_empty(fake_redis_checkpointer):
    cp = fake_redis_checkpointer
    result = await cp.get_latest("no-such-thread", "g")
    assert result is None


@pytest.mark.asyncio
async def test_get_history_returns_empty_when_missing(fake_redis_checkpointer):
    cp = fake_redis_checkpointer
    result = await cp.get_history("ghost", "g")
    assert result == []


@pytest.mark.asyncio
async def test_delete_thread(fake_redis_checkpointer):
    cp = fake_redis_checkpointer
    await cp.save(_make_ckpt(0))
    await cp.save(_make_ckpt(1))
    await cp.delete_thread("t1", "g")

    history = await cp.get_history("t1", "g")
    assert history == []


@pytest.mark.asyncio
async def test_idempotent_replace(fake_redis_checkpointer):
    """Saving the same step twice replaces the earlier entry."""
    cp = fake_redis_checkpointer
    ckpt_a = _make_ckpt(1)
    ckpt_b = _make_ckpt(1)
    ckpt_b = Checkpoint(
        **{**ckpt_b.__dict__, "node_name": "updated_node"}
    )

    await cp.save(ckpt_a)
    await cp.save(ckpt_b)

    result = await cp.get_by_step("t1", "g", 1)
    assert result is not None
    assert result.node_name == "updated_node"


# ── Serialization round-trip ──────────────────────────────────────────────────


def test_serialize_deserialize_round_trip():
    ckpt = _make_ckpt(5)
    raw = RedisCheckpointer._serialize(ckpt)
    restored = RedisCheckpointer._deserialize(raw)

    assert restored.id == ckpt.id
    assert restored.step == 5
    assert restored.state_snapshot == ckpt.state_snapshot
    assert restored.created_at == ckpt.created_at


# ── TTL ───────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ttl_expire_called():
    cp = RedisCheckpointer(url="redis://fake", ttl_days=7)

    expire_calls = []

    class TrackingRedis:
        async def hset(self, key, field, value):
            pass

        async def expire(self, key, seconds):
            expire_calls.append((key, seconds))

        async def aclose(self):
            pass

    cp._client = TrackingRedis()
    await cp.save(_make_ckpt(0))

    assert len(expire_calls) == 1
    assert expire_calls[0][1] == 7 * 86400


@pytest.mark.asyncio
async def test_close_resets_client_to_none():
    """RedisCheckpointer.close() must call aclose() and clear the client field
    so a fresh connection is opened on next use."""
    cp = RedisCheckpointer(url="redis://fake")

    aclose_calls: list[bool] = []

    class FakeClient:
        async def aclose(self) -> None:
            aclose_calls.append(True)

    cp._client = FakeClient()
    await cp.close()
    assert aclose_calls == [True]
    assert cp._client is None


@pytest.mark.asyncio
async def test_close_is_noop_when_no_client():
    """close() must not raise if no client was ever connected."""
    cp = RedisCheckpointer(url="redis://fake")
    assert cp._client is None
    await cp.close()  # must not raise
    assert cp._client is None


@pytest.mark.asyncio
async def test_async_context_manager_closes_on_exit():
    """`async with` invocation must call close() automatically."""
    cp = RedisCheckpointer(url="redis://fake")

    aclose_calls: list[bool] = []

    class FakeClient:
        async def aclose(self) -> None:
            aclose_calls.append(True)

    cp._client = FakeClient()
    async with cp as entered:
        assert entered is cp
    assert aclose_calls == [True]
    assert cp._client is None
