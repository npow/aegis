"""Global Aegis configuration helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


def configure(
    *,
    checkpointer: Any | None = None,
    tracer: Any | None = None,
    http_proxy_port: int | None = None,
) -> None:
    """Configure global Aegis defaults.

    In production, call this at application startup::

        import aegis, os
        aegis.configure(
            checkpointer=aegis.PostgresCheckpointer(os.environ["DATABASE_URL"]),
            tracer=aegis.OTelTracer(endpoint=os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]),
        )

    Args:
        checkpointer: Default checkpointer for all graphs (overridden by per-run config).
        tracer: OTel tracer that creates spans for every graph and node execution.
        http_proxy_port: Local port of an HTTP/HTTPS proxy that all agent traffic is
            routed through (sets HTTP_PROXY / HTTPS_PROXY env vars so that httpx and
            requests automatically use it for newly created sessions).
    """
    import aegis._globals as _g

    if checkpointer is not None:
        _g.DEFAULT_CHECKPOINTER = checkpointer
    if tracer is not None:
        _g.DEFAULT_TRACER = tracer
    if http_proxy_port is not None:
        _g.HTTP_PROXY_PORT = http_proxy_port
        proxy_url = f"http://localhost:{http_proxy_port}"
        os.environ.setdefault("HTTP_PROXY", proxy_url)
        os.environ.setdefault("HTTPS_PROXY", proxy_url)


# ── PostgresCheckpointer ──────────────────────────────────────────────────────


class PostgresCheckpointer:
    """Postgres-backed checkpoint store (requires asyncpg: ``pip install aegis[postgres]``)."""

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS {table} (
            id                   TEXT NOT NULL,
            thread_id            TEXT NOT NULL,
            run_id               TEXT NOT NULL,
            graph_name           TEXT NOT NULL,
            graph_version        TEXT NOT NULL,
            step                 INTEGER NOT NULL,
            node_name            TEXT NOT NULL,
            state_snapshot       TEXT NOT NULL,
            created_at           TEXT NOT NULL,
            parent_checkpoint_id TEXT,
            is_fork_root         BOOLEAN NOT NULL DEFAULT FALSE,
            PRIMARY KEY (thread_id, graph_name, step)
        )
    """
    _CREATE_INDEX = (
        "CREATE INDEX IF NOT EXISTS idx_{table}_thread_graph ON {table}(thread_id, graph_name)"
    )

    def __init__(
        self,
        connection_string: str,
        table_name: str = "aegis_checkpoints",
        pool_min: int = 1,
        pool_max: int = 5,
    ) -> None:
        import re as _re

        if not _re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", table_name):
            raise ValueError(
                f"PostgresCheckpointer: table_name {table_name!r} is invalid. "
                "Use only letters, digits, and underscores, starting with a letter or underscore."
            )
        self.connection_string = connection_string
        self.table_name = table_name
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._pool: Any | None = None
        self._pool_lock: Any = None  # asyncio.Lock — created lazily

    async def _get_pool(self) -> Any:
        # Fast path: pool already created
        if self._pool is not None:
            return self._pool

        # Lazy-create the lock (avoids issues constructing in non-async context)
        import asyncio as _asyncio

        if self._pool_lock is None:
            self._pool_lock = _asyncio.Lock()

        async with self._pool_lock:
            # Re-check after acquiring lock (double-checked locking)
            if self._pool is not None:
                return self._pool

            try:
                import asyncpg  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "asyncpg is required for PostgresCheckpointer. "
                    "Install it with: pip install aegis[postgres]"
                ) from exc
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=self._pool_min,
                max_size=self._pool_max,
            )
            async with self._pool.acquire() as conn:
                await conn.execute(self._CREATE_TABLE.format(table=self.table_name))
                await conn.execute(self._CREATE_INDEX.format(table=self.table_name))
        return self._pool

    @staticmethod
    def _row_to_checkpoint(row: Any) -> Any:
        from ._models import Checkpoint

        return Checkpoint(
            id=row["id"],
            thread_id=row["thread_id"],
            run_id=row["run_id"],
            graph_name=row["graph_name"],
            graph_version=row["graph_version"],
            step=row["step"],
            node_name=row["node_name"],
            state_snapshot=json.loads(row["state_snapshot"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            parent_checkpoint_id=row["parent_checkpoint_id"],
            is_fork_root=bool(row["is_fork_root"]),
        )

    async def save(self, checkpoint: Any) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self.table_name}
                  (id, thread_id, run_id, graph_name, graph_version, step,
                   node_name, state_snapshot, created_at, parent_checkpoint_id,
                   is_fork_root)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                ON CONFLICT (thread_id, graph_name, step) DO UPDATE SET
                    id                   = EXCLUDED.id,
                    run_id               = EXCLUDED.run_id,
                    node_name            = EXCLUDED.node_name,
                    state_snapshot       = EXCLUDED.state_snapshot,
                    created_at           = EXCLUDED.created_at,
                    parent_checkpoint_id = EXCLUDED.parent_checkpoint_id,
                    is_fork_root         = EXCLUDED.is_fork_root
                """,
                checkpoint.id,
                checkpoint.thread_id,
                checkpoint.run_id,
                checkpoint.graph_name,
                checkpoint.graph_version,
                checkpoint.step,
                checkpoint.node_name,
                json.dumps(checkpoint.state_snapshot, default=str),
                checkpoint.created_at.isoformat(),
                checkpoint.parent_checkpoint_id,
                checkpoint.is_fork_root,
            )

    async def get_latest(self, thread_id: str, graph_name: str) -> Any | None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self.table_name} "
                "WHERE thread_id=$1 AND graph_name=$2 ORDER BY step DESC LIMIT 1",
                thread_id,
                graph_name,
            )
            return self._row_to_checkpoint(row) if row else None

    async def get_by_step(self, thread_id: str, graph_name: str, step: int) -> Any | None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT * FROM {self.table_name} WHERE thread_id=$1 AND graph_name=$2 AND step=$3",
                thread_id,
                graph_name,
                step,
            )
            return self._row_to_checkpoint(row) if row else None

    async def get_history(self, thread_id: str, graph_name: str) -> list[Any]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM {self.table_name} "
                "WHERE thread_id=$1 AND graph_name=$2 ORDER BY step ASC",
                thread_id,
                graph_name,
            )
            return [self._row_to_checkpoint(r) for r in rows]

    async def delete_thread(self, thread_id: str, graph_name: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self.table_name} WHERE thread_id=$1 AND graph_name=$2",
                thread_id,
                graph_name,
            )

    async def close(self) -> None:
        """Close the connection pool. Call on application shutdown."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def __aenter__(self) -> PostgresCheckpointer:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


# ── OTelTracer ────────────────────────────────────────────────────────────────


class OTelTracer:
    """OpenTelemetry tracer that instruments every graph run and node execution.

    Requires ``pip install aegis[otel]``. Falls back to a no-op if the
    opentelemetry packages are not installed.

    Example::

        aegis.configure(
            tracer=aegis.OTelTracer(
                endpoint="http://otel-collector:4317",
                service_name="my-agent",
            )
        )
    """

    def __init__(
        self,
        endpoint: str | None = None,
        service_name: str = "aegis",
    ) -> None:
        self.endpoint = endpoint
        self.service_name = service_name
        self._tracer: Any | None = None
        self._initialized = False

    def _get_tracer(self) -> Any | None:
        """Return an opentelemetry Tracer, or None if otel is not installed."""
        if self._initialized:
            return self._tracer
        self._initialized = True

        try:
            from opentelemetry import trace  # type: ignore[import]
            from opentelemetry.sdk.resources import Resource  # type: ignore[import]
            from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import]

            resource = Resource.create({"service.name": self.service_name})
            provider = TracerProvider(resource=resource)

            if self.endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import]
                        OTLPSpanExporter,
                    )
                    from opentelemetry.sdk.trace.export import (
                        BatchSpanProcessor,  # type: ignore[import]
                    )

                    exporter = OTLPSpanExporter(endpoint=self.endpoint)
                    provider.add_span_processor(BatchSpanProcessor(exporter))
                except ImportError:
                    import warnings

                    warnings.warn(
                        "opentelemetry-exporter-otlp-proto-grpc not installed; "
                        "spans will not be exported. Install aegis[otel].",
                        stacklevel=2,
                    )

            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer("aegis", tracer_provider=provider)

        except ImportError:
            import warnings

            warnings.warn(
                "opentelemetry-api / opentelemetry-sdk not installed; "
                "tracing disabled. Install aegis[otel] to enable.",
                stacklevel=2,
            )
            self._tracer = None

        return self._tracer
