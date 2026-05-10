"""Tests for HTTP transport Layer 1 interception."""

import pytest

from rampart import (
    NetworkPermission,
    PermissionDeniedError,
    PermissionScope,
)
from rampart._permissions import check_network_permission

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
    """The HTTP intercept should be installed at rampart import time."""
    from rampart._http_intercept import _installed

    assert _installed is True


def test_http_intercept_does_not_block_outside_run():
    """Outside a graph run, HTTP calls should be allowed (no RunContext)."""
    from rampart._context import _run_context
    from rampart._http_intercept import _intercept

    # No run context set — should not raise
    assert _run_context.get() is None
    _intercept("https://anything-at-all.com")  # must not raise


# ── install/uninstall round-trip ──────────────────────────────────────────────


@pytest.fixture
def fresh_install():
    """Cycle install state so test bodies can re-install/uninstall safely."""
    from rampart import _http_intercept

    was_installed = _http_intercept._installed
    yield _http_intercept
    _http_intercept.uninstall()
    if was_installed:
        _http_intercept.install()


def test_install_is_idempotent(fresh_install):
    """install() called twice must not double-wrap (would otherwise fire the
    intercept twice per request)."""
    fresh_install.install()
    saved = dict(fresh_install._originals)
    fresh_install.install()  # second call must be a no-op
    assert fresh_install._originals == saved


def test_uninstall_restores_httpx_send(fresh_install):
    import httpx

    fresh_install.uninstall()
    pristine = httpx.Client.send
    fresh_install.install()
    assert httpx.Client.send is not pristine, "install should swap send"
    fresh_install.uninstall()
    assert httpx.Client.send is pristine, "uninstall should restore send"


def test_uninstall_restores_requests_send(fresh_install):
    requests = pytest.importorskip(
        "requests",
        reason="requests is not installed; the patcher silently no-ops it",
    )
    fresh_install.uninstall()
    pristine = requests.Session.send
    fresh_install.install()
    assert requests.Session.send is not pristine
    fresh_install.uninstall()
    assert requests.Session.send is pristine


def test_uninstall_clears_installed_flag(fresh_install):
    fresh_install.install()
    assert fresh_install._installed is True
    fresh_install.uninstall()
    assert fresh_install._installed is False


# ── _intercept inside a run ──────────────────────────────────────────────────


def _make_ctx(*, scope=None, current_node="n1"):
    """Build a minimal RunContext for unit tests."""
    from datetime import datetime, timezone

    from rampart._context import RunContext
    from rampart._models import RunTrace
    from rampart.checkpointers import MemoryCheckpointer

    trace = RunTrace(
        run_id="r1",
        thread_id="t1",
        graph_name="g",
        graph_version="1",
        started_at=datetime.now(timezone.utc),
        completed_at=None,
        status="running",
    )
    return RunContext(
        run_id="r1",
        thread_id="t1",
        graph_name="g",
        graph_version="1",
        checkpointer=MemoryCheckpointer(),
        trace=trace,
        permission_scope=scope,
        budget=None,
        current_node_name=current_node,
    )


def test_intercept_allows_when_scope_is_none():
    """Inside a run with no permission_scope set, _intercept must allow."""
    from rampart._context import _run_context
    from rampart._http_intercept import _intercept

    token = _run_context.set(_make_ctx(scope=None))
    try:
        _intercept("https://example.com")  # must not raise
    finally:
        _run_context.reset(token)


def test_intercept_raises_on_denied_url():
    """When the active scope denies a URL, _intercept raises
    PermissionDeniedError carrying a populated PermissionViolationEvent."""
    from rampart import (
        NetworkPermission,
        PermissionDeniedError,
        PermissionScope,
    )
    from rampart._context import _run_context
    from rampart._http_intercept import _intercept

    scope = PermissionScope(
        network=NetworkPermission(allowed_domains=["allowed.com"], deny_all_others=True)
    )
    ctx = _make_ctx(scope=scope, current_node="n1")
    token = _run_context.set(ctx)
    try:
        with pytest.raises(PermissionDeniedError) as exc_info:
            _intercept("https://blocked.example.com/x")
        event = exc_info.value.event
        assert event.run_id == "r1"
        assert event.node_name == "n1"
        assert event.violation_type == "http_intercept_blocked"
        assert "blocked.example.com" in event.attempted_action
    finally:
        _run_context.reset(token)
