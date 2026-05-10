"""Artifact versioning for Rampart.

Artifacts are named, tagged outputs saved by nodes during a graph run.
Unlike checkpoints (which are internal step snapshots used for resume/replay),
artifacts are first-class user-defined outputs with stable names that can be
retrieved across runs.

Example::

    @node
    async def summarize(state: MyState, llm: LLMContext, artifacts: ArtifactContext) -> MyState:
        response = await llm.complete("gpt-4o", state.text)
        await artifacts.save("summary", response.text, tags=["v1"])
        return state.update(summary=response.text)

    # Retrieve later (even from a different run):
    summary = await my_graph.get_artifact(thread_id="t1", name="summary")
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ── Artifact model ─────────────────────────────────────────────────────────────


@dataclass
class Artifact:
    """A named, versioned output saved during a graph run."""

    id: str            # "art_{graph}_{thread}_{run}_{name}_{hash[:6]}"
    name: str          # user-defined label, e.g. "summary", "embeddings"
    run_id: str
    thread_id: str
    graph_name: str
    graph_version: str
    node_name: str
    step: int
    data: Any          # JSON-serializable payload
    tags: list[str]
    created_at: datetime
    size_bytes: int
    data_type: str     # type(data).__name__


class ArtifactNotFoundError(Exception):
    """Raised when ``ArtifactContext.load()`` finds no matching artifact."""


# ── ArtifactStoreBase protocol ─────────────────────────────────────────────────


@runtime_checkable
class ArtifactStoreBase(Protocol):
    """Protocol that all artifact store backends must implement."""

    async def save(self, artifact: Artifact) -> None:
        """Persist an artifact."""
        ...

    async def get(
        self,
        thread_id: str,
        graph_name: str,
        name: str,
        run_id: str | None = None,
    ) -> Artifact | None:
        """Return the most-recent artifact with *name* for a thread, or None."""
        ...

    async def list(
        self,
        thread_id: str,
        graph_name: str,
        name: str | None = None,
    ) -> list[Artifact]:
        """Return all artifacts for a thread, optionally filtered by name."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...

    async def __aenter__(self) -> ArtifactStoreBase:
        ...

    async def __aexit__(self, *args: object) -> None:
        ...


# ── MemoryArtifactStore ────────────────────────────────────────────────────────


class MemoryArtifactStore:
    """In-process artifact store. Data is lost when the process exits."""

    def __init__(self) -> None:
        # {(thread_id, graph_name): list[Artifact]} — ordered by insertion
        self._store: dict[tuple[str, str], list[Artifact]] = {}

    async def save(self, artifact: Artifact) -> None:
        key = (artifact.thread_id, artifact.graph_name)
        self._store.setdefault(key, []).append(artifact)

    async def get(
        self,
        thread_id: str,
        graph_name: str,
        name: str,
        run_id: str | None = None,
    ) -> Artifact | None:
        key = (thread_id, graph_name)
        entries = self._store.get(key, [])
        matches = [a for a in entries if a.name == name]
        if run_id is not None:
            matches = [a for a in matches if a.run_id == run_id]
        return matches[-1] if matches else None  # latest by insertion order

    async def list(
        self,
        thread_id: str,
        graph_name: str,
        name: str | None = None,
    ) -> list[Artifact]:
        key = (thread_id, graph_name)
        entries = list(self._store.get(key, []))
        if name is not None:
            entries = [a for a in entries if a.name == name]
        return entries

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> MemoryArtifactStore:
        return self

    async def __aexit__(self, *_: object) -> None:
        pass


# ── SqliteArtifactStore ────────────────────────────────────────────────────────


_DEFAULT_DB_PATH = Path.home() / ".rampart" / "artifacts.db"


class SqliteArtifactStore:
    """SQLite-backed artifact store (requires ``pip install rampart[sqlite]``)."""

    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: Any | None = None

    async def _get_db(self) -> Any:
        if self._db is not None:
            return self._db
        try:
            import aiosqlite  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "aiosqlite is required for SqliteArtifactStore. "
                "Install it with: pip install rampart[sqlite]"
            ) from exc
        db = await aiosqlite.connect(str(self._db_path))
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await self._init_schema(db)
        self._db = db
        return db

    async def _init_schema(self, db: Any) -> None:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS rampart_artifacts (
                id            TEXT NOT NULL PRIMARY KEY,
                name          TEXT NOT NULL,
                run_id        TEXT NOT NULL,
                thread_id     TEXT NOT NULL,
                graph_name    TEXT NOT NULL,
                graph_version TEXT NOT NULL,
                node_name     TEXT NOT NULL,
                step          INTEGER NOT NULL,
                data          TEXT NOT NULL,
                tags          TEXT NOT NULL DEFAULT '[]',
                created_at    TEXT NOT NULL,
                size_bytes    INTEGER NOT NULL DEFAULT 0,
                data_type     TEXT NOT NULL DEFAULT 'unknown'
            )
            """
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_art_thread_graph_name "
            "ON rampart_artifacts(thread_id, graph_name, name)"
        )
        await db.commit()

    @staticmethod
    def _row_to_artifact(row: Any) -> Artifact:
        return Artifact(
            id=row["id"],
            name=row["name"],
            run_id=row["run_id"],
            thread_id=row["thread_id"],
            graph_name=row["graph_name"],
            graph_version=row["graph_version"],
            node_name=row["node_name"],
            step=row["step"],
            data=json.loads(row["data"]),
            tags=json.loads(row["tags"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            size_bytes=row["size_bytes"],
            data_type=row["data_type"],
        )

    async def save(self, artifact: Artifact) -> None:
        db = await self._get_db()
        await db.execute(
            """
            INSERT OR REPLACE INTO rampart_artifacts
              (id, name, run_id, thread_id, graph_name, graph_version, node_name,
               step, data, tags, created_at, size_bytes, data_type)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                artifact.id,
                artifact.name,
                artifact.run_id,
                artifact.thread_id,
                artifact.graph_name,
                artifact.graph_version,
                artifact.node_name,
                artifact.step,
                json.dumps(artifact.data, default=str),
                json.dumps(artifact.tags),
                artifact.created_at.isoformat(),
                artifact.size_bytes,
                artifact.data_type,
            ),
        )
        await db.commit()

    async def get(
        self,
        thread_id: str,
        graph_name: str,
        name: str,
        run_id: str | None = None,
    ) -> Artifact | None:
        db = await self._get_db()
        if run_id is not None:
            q = (
                "SELECT * FROM rampart_artifacts "
                "WHERE thread_id=? AND graph_name=? AND name=? AND run_id=? "
                "ORDER BY created_at DESC LIMIT 1"
            )
            params: tuple[Any, ...] = (thread_id, graph_name, name, run_id)
        else:
            q = (
                "SELECT * FROM rampart_artifacts "
                "WHERE thread_id=? AND graph_name=? AND name=? "
                "ORDER BY created_at DESC LIMIT 1"
            )
            params = (thread_id, graph_name, name)
        async with db.execute(q, params) as cursor:
            row = await cursor.fetchone()
            return self._row_to_artifact(row) if row else None

    async def list(
        self,
        thread_id: str,
        graph_name: str,
        name: str | None = None,
    ) -> list[Artifact]:
        db = await self._get_db()
        if name is not None:
            q = (
                "SELECT * FROM rampart_artifacts "
                "WHERE thread_id=? AND graph_name=? AND name=? ORDER BY created_at ASC"
            )
            params = (thread_id, graph_name, name)
        else:
            q = (
                "SELECT * FROM rampart_artifacts "
                "WHERE thread_id=? AND graph_name=? ORDER BY created_at ASC"
            )
            params = (thread_id, graph_name)
        async with db.execute(q, params) as cursor:
            rows = await cursor.fetchall()
            return [self._row_to_artifact(r) for r in rows]

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> SqliteArtifactStore:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


# ── ArtifactContext ────────────────────────────────────────────────────────────


class ArtifactContext:
    """Injected into node functions via the ``artifacts`` parameter.

    Example::

        @node
        async def extract(state: S, artifacts: ArtifactContext) -> S:
            await artifacts.save("result", state.output, tags=["production"])
            return state
    """

    def __init__(self, ctx: Any) -> None:  # ctx: RunContext
        self._ctx = ctx

    async def save(
        self,
        name: str,
        data: Any,
        *,
        tags: list[str] | None = None,
    ) -> Artifact:
        """Persist a named artifact for the current run.

        Args:
            name: Stable label, e.g. ``"summary"`` or ``"embeddings_v2"``.
            data: JSON-serializable payload.
            tags: Optional list of string tags for filtering.

        Returns:
            The saved ``Artifact`` instance.
        """
        if self._ctx.artifact_store is None:
            raise RuntimeError(
                "No artifact store configured.  Pass artifact_store= to "
                "rampart.configure() or RunConfig to enable artifact persistence."
            )
        raw = json.dumps(data, default=str)
        data_hash = hashlib.sha256(raw.encode()).hexdigest()[:6]
        artifact = Artifact(
            id=(
                f"art_{self._ctx.graph_name}_{self._ctx.thread_id}"
                f"_{self._ctx.run_id}_{name}_{data_hash}"
            ),
            name=name,
            run_id=self._ctx.run_id,
            thread_id=self._ctx.thread_id,
            graph_name=self._ctx.graph_name,
            graph_version=self._ctx.graph_version,
            node_name=self._ctx.current_node_name or "unknown",
            step=self._ctx._step_counter - 1,
            data=data,
            tags=tags or [],
            created_at=datetime.utcnow(),
            size_bytes=len(raw.encode()),
            data_type=type(data).__name__,
        )
        await self._ctx.artifact_store.save(artifact)
        return artifact

    async def load(
        self,
        name: str,
        run_id: str | None = None,
    ) -> Any:
        """Return the data payload of the most-recent artifact with *name*.

        Args:
            name: The artifact label to look up.
            run_id: If given, restrict to this specific run.

        Raises:
            ``ArtifactNotFoundError`` if no matching artifact exists.
        """
        if self._ctx.artifact_store is None:
            raise RuntimeError("No artifact store configured.")
        artifact = await self._ctx.artifact_store.get(
            thread_id=self._ctx.thread_id,
            graph_name=self._ctx.graph_name,
            name=name,
            run_id=run_id,
        )
        if artifact is None:
            msg = f"No artifact named '{name}' for thread '{self._ctx.thread_id}'"
            if run_id:
                msg += f" run '{run_id}'"
            raise ArtifactNotFoundError(msg)
        return artifact.data

    async def list(self, name: str | None = None) -> list[Artifact]:
        """Return all artifacts for the current thread, optionally filtered by name."""
        if self._ctx.artifact_store is None:
            return []
        return await self._ctx.artifact_store.list(
            thread_id=self._ctx.thread_id,
            graph_name=self._ctx.graph_name,
            name=name,
        )
