"""Tests for artifact versioning (_artifacts.py)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import rampart
from rampart._artifacts import ArtifactNotFoundError, MemoryArtifactStore, SqliteArtifactStore

# ── Shared state type ──────────────────────────────────────────────────────────


@dataclass
class ArtState(rampart.AgentState):
    text: str = ""
    saved: bool = False


# ── MemoryArtifactStore unit tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_memory_store_save_and_get():
    store = MemoryArtifactStore()
    async with store:
        from datetime import datetime

        from rampart._artifacts import Artifact

        art = Artifact(
            id="art_test_t1_r1_summary_abc123",
            name="summary",
            run_id="r1",
            thread_id="t1",
            graph_name="g",
            graph_version="1.0",
            node_name="n",
            step=1,
            data={"result": "hello"},
            tags=["v1"],
            created_at=datetime.utcnow(),
            size_bytes=20,
            data_type="dict",
        )
        await store.save(art)

        fetched = await store.get("t1", "g", "summary")
        assert fetched is not None
        assert fetched.data == {"result": "hello"}
        assert fetched.tags == ["v1"]


@pytest.mark.asyncio
async def test_memory_store_latest_by_insertion():
    store = MemoryArtifactStore()
    from datetime import datetime

    from rampart._artifacts import Artifact

    async with store:
        for i in range(3):
            await store.save(
                Artifact(
                    id=f"art_g_t1_r1_summary_{i}",
                    name="summary",
                    run_id="r1",
                    thread_id="t1",
                    graph_name="g",
                    graph_version="1.0",
                    node_name="n",
                    step=i,
                    data=f"v{i}",
                    tags=[],
                    created_at=datetime.utcnow(),
                    size_bytes=2,
                    data_type="str",
                )
            )

        latest = await store.get("t1", "g", "summary")
        assert latest is not None
        assert latest.data == "v2"


@pytest.mark.asyncio
async def test_memory_store_list_all_and_by_name():
    store = MemoryArtifactStore()
    from datetime import datetime

    from rampart._artifacts import Artifact

    async with store:
        for name in ["alpha", "beta", "alpha"]:
            await store.save(
                Artifact(
                    id=f"art_g_t1_r1_{name}_x",
                    name=name,
                    run_id="r1",
                    thread_id="t1",
                    graph_name="g",
                    graph_version="1.0",
                    node_name="n",
                    step=0,
                    data=name,
                    tags=[],
                    created_at=datetime.utcnow(),
                    size_bytes=5,
                    data_type="str",
                )
            )

        all_arts = await store.list("t1", "g")
        assert len(all_arts) == 3

        alpha = await store.list("t1", "g", name="alpha")
        assert len(alpha) == 2


@pytest.mark.asyncio
async def test_memory_store_get_none_when_missing():
    store = MemoryArtifactStore()
    async with store:
        result = await store.get("no_thread", "g", "nope")
        assert result is None


# ── SqliteArtifactStore unit tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sqlite_store_round_trip(tmp_path):
    db = tmp_path / "artifacts.db"
    from datetime import datetime

    from rampart._artifacts import Artifact

    async with SqliteArtifactStore(db) as store:
        art = Artifact(
            id="art_g_t1_r1_key_aabbcc",
            name="key",
            run_id="r1",
            thread_id="t1",
            graph_name="g",
            graph_version="1.0",
            node_name="node1",
            step=2,
            data=[1, 2, 3],
            tags=["prod"],
            created_at=datetime(2025, 1, 1, 12, 0, 0),
            size_bytes=10,
            data_type="list",
        )
        await store.save(art)

        fetched = await store.get("t1", "g", "key")
        assert fetched is not None
        assert fetched.data == [1, 2, 3]
        assert fetched.tags == ["prod"]
        assert fetched.step == 2

        listing = await store.list("t1", "g")
        assert len(listing) == 1


# ── ArtifactContext integration (full graph run) ───────────────────────────────


@rampart.node
async def _save_node(state: ArtState, artifacts: rampart.ArtifactContext) -> ArtState:
    await artifacts.save("result", state.text, tags=["integration"])
    return state.update(saved=True)


@rampart.graph(name="_art_integration_graph")
async def _art_integration_graph(state: ArtState) -> ArtState:
    return await _save_node(state)


@pytest.mark.asyncio
async def test_artifact_saved_and_retrieved_via_graph():
    store = MemoryArtifactStore()
    async with store:
        cfg = rampart.RunConfig(thread_id="art-t1", artifact_store=store)
        result = await _art_integration_graph.run(ArtState(text="hello artifact"), cfg)
        assert result.state.saved is True

        data = await _art_integration_graph.get_artifact("art-t1", "result", store=store)
        assert data == "hello artifact"

        arts = await _art_integration_graph.list_artifacts("art-t1", store=store)
        assert len(arts) == 1
        assert arts[0].tags == ["integration"]


@pytest.mark.asyncio
async def test_artifact_not_found_error():
    store = MemoryArtifactStore()
    async with store:
        with pytest.raises(ArtifactNotFoundError):
            await _art_integration_graph.get_artifact("missing-thread", "nope", store=store)


@pytest.mark.asyncio
async def test_artifact_context_load():
    """ArtifactContext.load() returns the data of a previously saved artifact."""
    store = MemoryArtifactStore()

    @rampart.node
    async def _save_then_load(state: ArtState, artifacts: rampart.ArtifactContext) -> ArtState:
        await artifacts.save("msg", "world")
        msg = await artifacts.load("msg")
        return state.update(text=msg)

    @rampart.graph(name="_save_load_graph")
    async def _save_load_graph(state: ArtState) -> ArtState:
        return await _save_then_load(state)

    async with store:
        cfg = rampart.RunConfig(thread_id="sl-t1", artifact_store=store)
        result = await _save_load_graph.run(ArtState(), cfg)
        assert result.state.text == "world"


@pytest.mark.asyncio
async def test_artifact_context_load_raises_when_missing():
    store = MemoryArtifactStore()

    @rampart.node
    async def _load_missing(state: ArtState, artifacts: rampart.ArtifactContext) -> ArtState:
        await artifacts.load("doesnt_exist")
        return state

    @rampart.graph(name="_load_missing_graph")
    async def _load_missing_graph(state: ArtState) -> ArtState:
        return await _load_missing(state)

    async with store:
        cfg = rampart.RunConfig(thread_id="lm-t1", artifact_store=store)
        result = await _load_missing_graph.run(ArtState(), cfg)
        # The graph run should fail
        assert result.status == "failed"
        assert "ArtifactNotFoundError" in (result.error.exception_type if result.error else "")


@pytest.mark.asyncio
async def test_artifact_list_returns_empty_without_store():
    """list() with no artifact store configured returns []."""

    @rampart.node
    async def _list_no_store(state: ArtState, artifacts: rampart.ArtifactContext) -> ArtState:
        arts = await artifacts.list()
        return state.update(text=str(len(arts)))

    @rampart.graph(name="_list_no_store_graph")
    async def _list_no_store_graph(state: ArtState) -> ArtState:
        return await _list_no_store(state)

    cfg = rampart.RunConfig(thread_id="ns-t1")
    result = await _list_no_store_graph.run(ArtState(), cfg)
    assert result.state.text == "0"
