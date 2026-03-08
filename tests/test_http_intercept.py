"""Tests for HTTP transport Layer 1 interception."""

import pytest

from aegis import (
    NetworkPermission,
    PermissionDeniedError,
    PermissionScope,
)
from aegis._permissions import check_network_permission

# ── Unit tests ────────────────────────────────────────────────────────────────


def test_intercept_allows_whitelisted_domain():
    scope = PermissionScope(
        network=NetworkPermission(allowed_domains=["api.openai.com"], deny_all_others=True)
    )
    # Should not raise
    check_network_permission("https://api.openai.com/v1/chat", scope, "r1", "t1", "n1")


def test_intercept_blocks_unlisted_domain():
    scope = PermissionScope(
        network=NetworkPermission(allowed_domains=["api.openai.com"], deny_all_others=True)
    )
    with pytest.raises(PermissionDeniedError) as exc_info:
        check_network_permission("https://exfiltrate.evil.com/data", scope, "r1", "t1", "n1")
    assert exc_info.value.event.violation_type == "network_domain_denied"


def test_intercept_no_scope_allows_all():
    """No scope = no interception."""
    check_network_permission("https://anything.example.com", None, "r1", "t1", "n")


def test_intercept_no_deny_all_others_allows_all():
    """deny_all_others=False means unmatched domains are allowed."""
    scope = PermissionScope(
        network=NetworkPermission(
            allowed_domains=["api.openai.com"],
            deny_all_others=False,
        )
    )
    # Should not raise for unlisted domain
    check_network_permission("https://any-other.com/path", scope, "r1", "t1", "n")


def test_intercept_wildcard_subdomain():
    scope = PermissionScope(
        network=NetworkPermission(
            allowed_domains=["*.anthropic.com"],
            deny_all_others=True,
        )
    )
    check_network_permission("https://api.anthropic.com/v1", scope, "r1", "t1", "n")

    with pytest.raises(PermissionDeniedError):
        check_network_permission("https://evil.com/path", scope, "r1", "t1", "n")


# ── HTTP intercept installed at import ────────────────────────────────────────


def test_http_intercept_installed():
    """The HTTP intercept should be installed at aegis import time."""
    from aegis._http_intercept import _installed

    assert _installed is True


def test_http_intercept_does_not_block_outside_run():
    """Outside a graph run, HTTP calls should be allowed (no RunContext)."""
    from aegis._context import _run_context
    from aegis._http_intercept import _intercept

    # No run context set — should not raise
    assert _run_context.get() is None
    _intercept("https://anything-at-all.com")  # must not raise
