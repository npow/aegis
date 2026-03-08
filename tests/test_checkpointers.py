"""Tests for MemoryCheckpointer and SqliteCheckpointer."""

import tempfile
from datetime import datetime

from aegis import Checkpoint
from aegis.checkpointers import MemoryCheckpointer, SqliteCheckpointer


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
        created_at=datetime.utcnow(),
        parent_checkpoint_id=None,
    )


# ── MemoryCheckpointer ────────────────────────────────────────────────────────


async def test_memory_save_and_get_latest():
    cp = MemoryCheckpointer()
    await cp.save(_make_ckpt(0))
    await cp.save(_make_ckpt(1))
    await cp.save(_make_ckpt(2))
    latest = await cp.get_latest("t1", "g")
    assert latest is not None
    assert latest.step == 2


async def test_memory_get_by_step():
    cp = MemoryCheckpointer()
    for i in range(3):
        await cp.save(_make_ckpt(i))
    ckpt = await cp.get_by_step("t1", "g", 1)
    assert ckpt is not None
    assert ckpt.step == 1
    assert ckpt.node_name == "node_1"


async def test_memory_get_history_ordered():
    cp = MemoryCheckpointer()
    for i in [2, 0, 1]:
        await cp.save(_make_ckpt(i))
    history = await cp.get_history("t1", "g")
    assert [c.step for c in history] == [0, 1, 2]


async def test_memory_returns_none_for_missing():
    cp = MemoryCheckpointer()
    assert await cp.get_latest("missing", "g") is None
    assert await cp.get_by_step("missing", "g", 5) is None


async def test_memory_get_history_empty():
    cp = MemoryCheckpointer()
    assert await cp.get_history("empty", "g") == []


async def test_memory_delete_thread():
    cp = MemoryCheckpointer()
    await cp.save(_make_ckpt(0))
    await cp.save(_make_ckpt(1))
    await cp.delete_thread("t1", "g")
    assert await cp.get_history("t1", "g") == []


async def test_memory_idempotent_save():
    """Saving same step twice should update, not duplicate."""
    cp = MemoryCheckpointer()
    c1 = _make_ckpt(1)
    c2 = Checkpoint(
        id="new-id",
        thread_id="t1",
        run_id="r2",
        graph_name="g",
        graph_version="1.0",
        step=1,
        node_name="node_1_updated",
        state_snapshot={"value": 99},
        created_at=datetime.utcnow(),
        parent_checkpoint_id=None,
    )
    await cp.save(c1)
    await cp.save(c2)  # Replace step 1
    history = await cp.get_history("t1", "g")
    assert len(history) == 1
    assert history[0].node_name == "node_1_updated"


async def test_memory_separate_threads():
    cp = MemoryCheckpointer()
    await cp.save(_make_ckpt(0, thread_id="thread-a"))
    await cp.save(_make_ckpt(0, thread_id="thread-b"))
    a = await cp.get_history("thread-a", "g")
    b = await cp.get_history("thread-b", "g")
    assert len(a) == 1
    assert len(b) == 1


# ── SqliteCheckpointer ────────────────────────────────────────────────────────


async def test_sqlite_save_and_get_latest():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        async with SqliteCheckpointer(f.name) as cp:
            await cp.save(_make_ckpt(0))
            await cp.save(_make_ckpt(1))
            latest = await cp.get_latest("t1", "g")
            assert latest is not None
            assert latest.step == 1


async def test_sqlite_get_by_step():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        async with SqliteCheckpointer(f.name) as cp:
            for i in range(3):
                await cp.save(_make_ckpt(i))
            ckpt = await cp.get_by_step("t1", "g", 2)
            assert ckpt is not None
            assert ckpt.step == 2


async def test_sqlite_get_history_ordered():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        async with SqliteCheckpointer(f.name) as cp:
            for i in [1, 0, 2]:
                await cp.save(_make_ckpt(i))
            history = await cp.get_history("t1", "g")
            assert [c.step for c in history] == [0, 1, 2]


async def test_sqlite_delete_thread():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        async with SqliteCheckpointer(f.name) as cp:
            await cp.save(_make_ckpt(0))
            await cp.delete_thread("t1", "g")
            assert await cp.get_history("t1", "g") == []


async def test_sqlite_returns_none_for_missing():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        async with SqliteCheckpointer(f.name) as cp:
            assert await cp.get_latest("no-thread", "g") is None


async def test_sqlite_state_snapshot_roundtrip():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        async with SqliteCheckpointer(f.name) as cp:
            ckpt = _make_ckpt(0)
            ckpt.state_snapshot["nested"] = {"a": 1, "b": [1, 2, 3]}
            await cp.save(ckpt)
            loaded = await cp.get_by_step("t1", "g", 0)
            assert loaded is not None
            assert loaded.state_snapshot["nested"]["a"] == 1
            assert loaded.state_snapshot["nested"]["b"] == [1, 2, 3]


# ── PostgresCheckpointer table_name validation ────────────────────────────────


def test_postgres_checkpointer_rejects_invalid_table_name():
    """PostgresCheckpointer must reject table names with SQL injection characters."""
    import pytest

    from aegis._config import PostgresCheckpointer

    with pytest.raises(ValueError, match="invalid"):
        PostgresCheckpointer("postgres://localhost/db", table_name="bad; DROP TABLE foo")

    with pytest.raises(ValueError, match="invalid"):
        PostgresCheckpointer("postgres://localhost/db", table_name="1starts_with_digit")

    with pytest.raises(ValueError, match="invalid"):
        PostgresCheckpointer("postgres://localhost/db", table_name="has-hyphen")


def test_postgres_checkpointer_accepts_valid_table_name():
    """PostgresCheckpointer should accept valid SQL identifiers."""
    from aegis._config import PostgresCheckpointer

    # Should not raise
    cp = PostgresCheckpointer("postgres://localhost/db", table_name="aegis_checkpoints")
    assert cp.table_name == "aegis_checkpoints"

    cp2 = PostgresCheckpointer("postgres://localhost/db", table_name="_private_table")
    assert cp2.table_name == "_private_table"
