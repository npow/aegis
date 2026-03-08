"""HTTP transport Layer 1 — Python library monkey-patching.

Intercepts outbound HTTP calls when inside an Aegis graph run and checks
them against the active permission scope. Covers httpx and requests.
Layer 2 (proxy injection) and Layer 3 (sandbox) are stubs in this release.
"""

from __future__ import annotations

import threading
from typing import Any

_installed = False
_lock = threading.Lock()

# Store original send methods so we can restore them in tests
_originals: dict[str, Any] = {}


def install() -> None:
    """Monkey-patch HTTP libraries. Called once at aegis import time."""
    global _installed
    with _lock:
        if _installed:
            return
        _patch_httpx()
        _patch_requests()
        _installed = True


def uninstall() -> None:
    """Restore original HTTP methods (for testing)."""
    global _installed
    with _lock:
        _restore_httpx()
        _restore_requests()
        _installed = False


def _patch_httpx() -> None:
    try:
        import httpx  # type: ignore[import]
    except ImportError:
        return

    orig_sync = httpx.Client.send
    orig_async = httpx.AsyncClient.send
    _originals["httpx_sync"] = orig_sync
    _originals["httpx_async"] = orig_async

    def _patched_sync(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        _intercept(str(request.url))
        return orig_sync(self, request, *args, **kwargs)

    async def _patched_async(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        _intercept(str(request.url))
        return await orig_async(self, request, *args, **kwargs)

    httpx.Client.send = _patched_sync  # type: ignore[method-assign]
    httpx.AsyncClient.send = _patched_async  # type: ignore[method-assign]


def _restore_httpx() -> None:
    try:
        import httpx  # type: ignore[import]
    except ImportError:
        return
    if "httpx_sync" in _originals:
        httpx.Client.send = _originals.pop("httpx_sync")  # type: ignore[method-assign]
    if "httpx_async" in _originals:
        httpx.AsyncClient.send = _originals.pop("httpx_async")  # type: ignore[method-assign]


def _patch_requests() -> None:
    try:
        import requests  # type: ignore[import]
    except ImportError:
        return

    orig = requests.Session.send
    _originals["requests"] = orig

    def _patched_requests(self: Any, request: Any, **kwargs: Any) -> Any:
        _intercept(request.url)
        return orig(self, request, **kwargs)

    requests.Session.send = _patched_requests  # type: ignore[method-assign]


def _restore_requests() -> None:
    try:
        import requests  # type: ignore[import]
    except ImportError:
        return
    if "requests" in _originals:
        requests.Session.send = _originals.pop("requests")  # type: ignore[method-assign]


def _intercept(url: str) -> None:
    """Check the URL against the active run's permission scope."""
    from ._context import _run_context

    ctx = _run_context.get()
    if ctx is None:
        return  # Not inside a graph run — allow all traffic

    if not ctx.permission_scope:
        return

    from ._permissions import check_network_permission

    try:
        check_network_permission(
            url=url,
            scope=ctx.permission_scope,
            run_id=ctx.run_id,
            thread_id=ctx.thread_id,
            node_name=ctx.current_node_name or "unknown",
        )
    except Exception as _exc:
        from datetime import datetime

        from ._models import PermissionViolationEvent

        event = PermissionViolationEvent(
            run_id=ctx.run_id,
            thread_id=ctx.thread_id,
            node_name=ctx.current_node_name or "unknown",
            violation_type="http_intercept_blocked",
            attempted_action=f"HTTP {url}",
            declared_scope=ctx.permission_scope,
            timestamp=datetime.utcnow(),
        )
        # Re-raise as PermissionDeniedError
        from ._models import PermissionDeniedError

        raise PermissionDeniedError(event) from _exc
