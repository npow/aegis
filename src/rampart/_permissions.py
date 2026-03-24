"""Permission scope enforcement for tool calls and network access."""

from __future__ import annotations

import fnmatch
from datetime import datetime, timezone
from urllib.parse import urlparse

from ._models import (
    PermissionDeniedError,
    PermissionScope,
    PermissionViolationEvent,
)


def check_tool_permission(
    tool_name: str,
    scope: PermissionScope | None,
    run_id: str,
    thread_id: str,
    node_name: str,
) -> None:
    """Raise PermissionDeniedError if the tool is not in the declared scope."""
    if scope is None or scope.tools is None:
        return
    if tool_name not in scope.tools:
        event = PermissionViolationEvent(
            run_id=run_id,
            thread_id=thread_id,
            node_name=node_name,
            violation_type="tool_not_in_whitelist",
            attempted_action=f"call tool '{tool_name}'",
            declared_scope=scope,
            timestamp=datetime.now(timezone.utc),
        )
        raise PermissionDeniedError(event)


def check_network_permission(
    url: str,
    scope: PermissionScope | None,
    run_id: str,
    thread_id: str,
    node_name: str,
) -> None:
    """Raise PermissionDeniedError if the domain is not in the declared scope."""
    if scope is None:
        return
    net = scope.network
    if not net or not net.deny_all_others:
        return

    domain = _extract_domain(url)
    if not domain:
        # Deny requests with unparseable URLs when deny_all_others is active
        event = PermissionViolationEvent(
            run_id=run_id,
            thread_id=thread_id,
            node_name=node_name,
            violation_type="network_domain_denied",
            attempted_action=f"HTTP request to {url!r} (unparseable URL)",
            declared_scope=scope,
            timestamp=datetime.now(timezone.utc),
        )
        raise PermissionDeniedError(event)

    if _domain_matches_any(domain, net.allowed_domains):
        return

    event = PermissionViolationEvent(
        run_id=run_id,
        thread_id=thread_id,
        node_name=node_name,
        violation_type="network_domain_denied",
        attempted_action=f"HTTP request to {url}",
        declared_scope=scope,
        timestamp=datetime.now(timezone.utc),
    )
    raise PermissionDeniedError(event)


def check_filesystem_permission(
    path: str,
    operation: str,  # "read" | "write"
    scope: PermissionScope | None,
    run_id: str,
    thread_id: str,
    node_name: str,
) -> None:
    """Raise PermissionDeniedError if the path is not in the declared scope."""
    if scope is None:
        return
    fs = scope.filesystem
    if operation == "read":
        if not fs.read:
            _deny_fs(path, operation, scope, run_id, thread_id, node_name)
        if fs.read_allowed_paths and not _path_matches_any(path, fs.read_allowed_paths):
            _deny_fs(path, operation, scope, run_id, thread_id, node_name)
    elif operation == "write":
        if not fs.write:
            _deny_fs(path, operation, scope, run_id, thread_id, node_name)
        if fs.write_allowed_paths and not _path_matches_any(path, fs.write_allowed_paths):
            _deny_fs(path, operation, scope, run_id, thread_id, node_name)


def _deny_fs(
    path: str,
    operation: str,
    scope: PermissionScope,
    run_id: str,
    thread_id: str,
    node_name: str,
) -> None:
    event = PermissionViolationEvent(
        run_id=run_id,
        thread_id=thread_id,
        node_name=node_name,
        violation_type="filesystem_path_denied",
        attempted_action=f"{operation} '{path}'",
        declared_scope=scope,
        timestamp=datetime.now(timezone.utc),
    )
    raise PermissionDeniedError(event)


def _extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.hostname or ""
    except Exception:
        return ""


def _domain_matches_any(domain: str, patterns: list[str]) -> bool:
    """Return True if domain matches any pattern in the list.

    ``*`` matches exactly one DNS label (no dots), so ``*.example.com``
    matches ``api.example.com`` but NOT ``a.b.example.com``.
    """
    import re

    for pattern in patterns:
        regex = re.escape(pattern).replace(r"\*", r"[^.]+")
        if re.fullmatch(regex, domain):
            return True
    return False


def _path_matches_any(path: str, patterns: list[str]) -> bool:
    """Return True if path matches any glob pattern in the list."""
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern):
            return True
    return False
