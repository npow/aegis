"""Tests for human-approval delivery channels (_approval.py)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rampart._approval import _resolve_timeout, request_approval
from rampart._models import ApprovalPolicy, PermissionDeniedError


# ── _resolve_timeout ───────────────────────────────────────────────────────────


def test_resolve_timeout_approve():
    policy = ApprovalPolicy(delivery="webhook", on_timeout="approve", timeout_seconds=30)
    assert _resolve_timeout(policy) is True


def test_resolve_timeout_deny():
    policy = ApprovalPolicy(delivery="webhook", on_timeout="deny", timeout_seconds=30)
    assert _resolve_timeout(policy) is False


def test_resolve_timeout_hard_stop():
    policy = ApprovalPolicy(delivery="webhook", on_timeout="hard_stop", timeout_seconds=30)
    with pytest.raises(PermissionDeniedError):
        _resolve_timeout(policy)


# ── Webhook channel ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_webhook_approved():
    policy = ApprovalPolicy(
        delivery="webhook",
        delivery_target="http://localhost:9999/approve",
        timeout_seconds=5,
        on_timeout="deny",
    )

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"approved": True}

    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_cls.return_value = mock_client

        result = await request_approval(
            tool_name="delete_file",
            args={"path": "/etc/passwd"},
            run_id="r1",
            thread_id="t1",
            node_name="agent",
            call_id="c1",
            policy=policy,
        )

    assert result is True


@pytest.mark.asyncio
async def test_webhook_denied():
    policy = ApprovalPolicy(
        delivery="webhook",
        delivery_target="http://localhost:9999/approve",
        timeout_seconds=5,
        on_timeout="deny",
    )

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"approved": False}

    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_cls.return_value = mock_client

        result = await request_approval(
            tool_name="nuke",
            args={},
            run_id="r1",
            thread_id="t1",
            node_name="agent",
            call_id="c2",
            policy=policy,
        )

    assert result is False


@pytest.mark.asyncio
async def test_webhook_timeout_falls_back_to_policy():
    policy = ApprovalPolicy(
        delivery="webhook",
        delivery_target="http://localhost:9999/approve",
        timeout_seconds=1,
        on_timeout="approve",
    )

    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_cls.return_value = mock_client

        result = await request_approval(
            tool_name="scan",
            args={},
            run_id="r1",
            thread_id="t1",
            node_name="agent",
            call_id="c3",
            policy=policy,
        )

    assert result is True  # on_timeout="approve"


@pytest.mark.asyncio
async def test_webhook_no_target_falls_back():
    """Missing delivery_target applies on_timeout policy."""
    import warnings

    policy = ApprovalPolicy(
        delivery="webhook",
        delivery_target=None,
        timeout_seconds=5,
        on_timeout="deny",
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = await request_approval(
            tool_name="tool",
            args={},
            run_id="r1",
            thread_id="t1",
            node_name="n",
            call_id="c4",
            policy=policy,
        )

    assert result is False


# ── Slack channel ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_slack_notify_then_apply_timeout():
    """Slack is one-way: sends notification then applies on_timeout."""
    policy = ApprovalPolicy(
        delivery="slack",
        delivery_target="https://hooks.slack.com/services/X/Y/Z",
        timeout_seconds=60,
        on_timeout="deny",
    )

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_cls.return_value = mock_client

        result = await request_approval(
            tool_name="send_email",
            args={"to": "ceo@example.com"},
            run_id="r1",
            thread_id="t1",
            node_name="mailer",
            call_id="c5",
            policy=policy,
        )

    assert result is False  # on_timeout="deny"
    # Slack POST was called
    mock_client.post.assert_awaited_once()


# ── Email channel ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_email_no_smtp_url_warns_and_applies_timeout():
    """Email without RAMPART_SMTP_URL warns and applies on_timeout."""
    import warnings

    policy = ApprovalPolicy(
        delivery="email",
        delivery_target="ops@example.com",
        timeout_seconds=30,
        on_timeout="approve",
    )

    with patch.dict("os.environ", {}, clear=False):
        # Ensure RAMPART_SMTP_URL is absent
        import os

        os.environ.pop("RAMPART_SMTP_URL", None)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await request_approval(
                tool_name="rm_rf",
                args={},
                run_id="r1",
                thread_id="t1",
                node_name="n",
                call_id="c6",
                policy=policy,
            )

    assert result is True  # on_timeout="approve"


# ── Unknown channel ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_channel_denies():
    """ApprovalPolicy now validates delivery at construction time."""
    with pytest.raises(ValueError, match="Invalid delivery"):
        ApprovalPolicy(
            delivery="carrier_pigeon",  # type: ignore[arg-type]
            timeout_seconds=5,
            on_timeout="approve",
        )
