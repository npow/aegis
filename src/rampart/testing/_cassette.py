"""Cassette record and replay for deterministic, zero-cost testing."""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .._models import CassetteRecord, CassetteReplayContext
from ._mock_tools import MockTool, _mock_testing_context, _TestingState


class _CassetteNS:
    """Namespace object so users write ``cassette.record()`` and ``cassette.replay()``."""

    @asynccontextmanager
    async def record(self, path: str) -> AsyncIterator[CassetteRecord]:
        """Context manager: record all LLM and tool calls to a cassette file."""
        cassette = CassetteRecord(
            graph_name="",
            graph_version="",
            recorded_at=datetime.now(timezone.utc),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
        state = _TestingState(
            cassette_mode="record",
            cassette=cassette,
        )
        token = _mock_testing_context.set(state)
        try:
            yield cassette
        finally:
            _mock_testing_context.reset(token)
            cassette.update_hash()
            _save_cassette(cassette, path)

    @asynccontextmanager
    async def replay(
        self,
        path: str,
        override_tools: dict[str, MockTool] | None = None,
    ) -> AsyncIterator[CassetteReplayContext]:
        """Context manager: serve all LLM and tool calls from a cassette file."""
        cassette = _load_cassette(path)
        replay_ctx = CassetteReplayContext(
            cassette=cassette,
            total_recorded_calls=len(cassette.entries),
        )
        state = _TestingState(
            cassette_mode="replay",
            cassette=cassette,
            cassette_override_tools=override_tools,
            # cassette_index_ref starts at [0] — updated by RunContext as entries are served
        )
        token = _mock_testing_context.set(state)
        try:
            yield replay_ctx
        finally:
            _mock_testing_context.reset(token)
            # Use the actual served count tracked by RunContext via cassette_index_ref
            replay_ctx.replay_calls_served = state.cassette_index_ref[0]


cassette = _CassetteNS()


# ── Serialization ─────────────────────────────────────────────────────────────


def _save_cassette(record: CassetteRecord, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "format_version": record.format_version,
        "graph_name": record.graph_name,
        "graph_version": record.graph_version,
        "recorded_at": record.recorded_at.isoformat(),
        "python_version": record.python_version,
        "content_hash": record.content_hash,
        "entries": [_entry_to_dict(e) for e in record.entries],
    }
    p.write_text(json.dumps(data, indent=2, default=str))


def _load_cassette(path: str) -> CassetteRecord:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Cassette file not found: {path}\n"
            "Record it first with: async with cassette.record(path): ..."
        )
    data = json.loads(p.read_text())
    from .._models import CassetteEntry

    entries = [
        CassetteEntry(
            type=e["type"],
            call_id=e["call_id"],
            step=e["step"],
            node_name=e["node_name"],
            request=e["request"],
            response=e["response"],
            timestamp=datetime.fromisoformat(e["timestamp"]),
        )
        for e in data.get("entries", [])
    ]
    record = CassetteRecord(
        format_version=data.get("format_version", "1.0"),
        graph_name=data.get("graph_name", ""),
        graph_version=data.get("graph_version", ""),
        recorded_at=datetime.fromisoformat(data.get("recorded_at", datetime.now(timezone.utc).isoformat())),
        python_version=data.get("python_version", ""),
        entries=entries,
        content_hash=data.get("content_hash", ""),
    )
    if record.content_hash:
        actual_hash = record.compute_hash()
        if actual_hash != record.content_hash:
            import warnings

            warnings.warn(
                f"Cassette content hash mismatch: file may have been manually edited. "
                f"Expected {record.content_hash[:16]}..., got {actual_hash[:16]}...",
                stacklevel=2,
            )
    return record


def _entry_to_dict(entry: Any) -> dict[str, Any]:
    return {
        "type": entry.type,
        "call_id": entry.call_id,
        "step": entry.step,
        "node_name": entry.node_name,
        "request": entry.request,
        "response": entry.response,
        "timestamp": entry.timestamp.isoformat()
        if hasattr(entry.timestamp, "isoformat")
        else str(entry.timestamp),
    }
