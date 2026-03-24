"""Approval workflow delivery — webhook, Slack, and email channels.

All three channels notify a human that a tool call is pending. The delivery
contract differs by channel:

  webhook  — POST payload to delivery_target; blocks until the endpoint
             responds with ``{"approved": bool}``.  The timeout is the
             HTTP request timeout.  This is the only channel that supports
             synchronous approval; use it when you control the endpoint.

  slack    — POST a notification to a Slack incoming-webhook URL (one-way).
             Since Slack incoming webhooks cannot return a structured reply,
             Rampart immediately applies the ``on_timeout`` policy after
             sending the notification.

  email    — Send a plain-text SMTP email to delivery_target.  Requires
             the ``RAMPART_SMTP_URL`` environment variable
             (``smtp://user:pass@host:port``).  Like Slack, approval is
             resolved via ``on_timeout`` after the notification is sent.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ApprovalDeliveryError(Exception):
    """Raised when a synchronous approval delivery channel fails."""


async def request_approval(
    tool_name: str,
    args: dict[str, Any],
    run_id: str,
    thread_id: str,
    node_name: str,
    call_id: str,
    policy: Any,  # ApprovalPolicy
) -> bool:
    """Send an approval request and return True (approved) or False (denied).

    Raises ``PermissionDeniedError`` when the policy is ``hard_stop`` and no
    approval was received in time.
    """
    payload: dict[str, Any] = {
        "tool_name": tool_name,
        "args": _safe_json(args),
        "run_id": run_id,
        "thread_id": thread_id,
        "node_name": node_name,
        "call_id": call_id,
    }

    if policy.delivery == "webhook":
        return await _webhook_approval(payload, policy)
    elif policy.delivery == "slack":
        await _slack_notify(payload, policy)
        return _resolve_timeout(
            policy,
            run_id=payload["run_id"],
            thread_id=payload["thread_id"],
            node_name=payload["node_name"],
        )
    elif policy.delivery == "email":
        await _email_notify(payload, policy)
        return _resolve_timeout(
            policy,
            run_id=payload["run_id"],
            thread_id=payload["thread_id"],
            node_name=payload["node_name"],
        )

    # Unknown channel — deny and warn
    import warnings

    warnings.warn(
        f"Unknown ApprovalPolicy.delivery channel {policy.delivery!r}; denying tool call.",
        stacklevel=2,
    )
    return False


# ── Webhook ────────────────────────────────────────────────────────────────────


async def _webhook_approval(payload: dict[str, Any], policy: Any) -> bool:
    """POST payload; expect ``{"approved": bool}`` in the response body."""
    if not policy.delivery_target:
        import warnings

        warnings.warn(
            "ApprovalPolicy.delivery='webhook' but delivery_target is not set; "
            "denying tool call by default.",
            stacklevel=3,
        )
        return _resolve_timeout(
            policy,
            run_id=payload.get("run_id", ""),
            thread_id=payload.get("thread_id", ""),
            node_name=payload.get("node_name", ""),
        )

    try:
        import httpx  # type: ignore[import]
    except ImportError:
        import warnings

        warnings.warn(
            "httpx is required for webhook approval delivery. "
            "Install it with: pip install httpx. "
            f"Falling back to on_timeout='{policy.on_timeout}'.",
            stacklevel=3,
        )
        return _resolve_timeout(
            policy,
            run_id=payload.get("run_id", ""),
            thread_id=payload.get("thread_id", ""),
            node_name=payload.get("node_name", ""),
        )

    try:
        timeout = float(policy.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                policy.delivery_target,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        response.raise_for_status()
        data = response.json()
        return bool(data.get("approved", False))

    except asyncio.TimeoutError:
        return _resolve_timeout(
            policy,
            run_id=payload.get("run_id", ""),
            thread_id=payload.get("thread_id", ""),
            node_name=payload.get("node_name", ""),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "Webhook approval request to %r failed: %s. "
            "Applying on_timeout='%s'.",
            policy.delivery_target,
            exc,
            policy.on_timeout,
        )
        raise ApprovalDeliveryError(
            f"Webhook approval request to {policy.delivery_target!r} failed: {exc}"
        ) from exc


# ── Slack ──────────────────────────────────────────────────────────────────────


async def _slack_notify(payload: dict[str, Any], policy: Any) -> None:
    """Post a notification message to a Slack incoming-webhook URL."""
    if not policy.delivery_target:
        import warnings

        warnings.warn(
            "ApprovalPolicy.delivery='slack' but delivery_target is not set; "
            "skipping Slack notification.",
            stacklevel=3,
        )
        return

    message = {
        "text": (
            ":warning: *Rampart: human approval required*\n"
            f"*Tool:* `{payload['tool_name']}`\n"
            f"*Run:* `{payload['run_id']}`  |  *Thread:* `{payload['thread_id']}`\n"
            f"*Node:* `{payload['node_name']}`\n"
            f"*Args:*\n```{json.dumps(payload['args'], indent=2)}```\n"
            f"_Timeout policy: `{policy.on_timeout}` after {policy.timeout_seconds}s_"
        )
    }

    try:
        import httpx  # type: ignore[import]

        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(policy.delivery_target, json=message)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Slack approval notification failed: %s", exc)


# ── Email ──────────────────────────────────────────────────────────────────────


async def _email_notify(payload: dict[str, Any], policy: Any) -> None:
    """Send a plain-text approval-request email via SMTP.

    Requires ``RAMPART_SMTP_URL=smtp://user:pass@host:port``.
    """
    if not policy.delivery_target:
        import warnings

        warnings.warn(
            "ApprovalPolicy.delivery='email' but delivery_target is not set; "
            "skipping email notification.",
            stacklevel=3,
        )
        return

    smtp_url = _get_smtp_url()
    if smtp_url is None:
        import warnings

        warnings.warn(
            "ApprovalPolicy.delivery='email' requires the RAMPART_SMTP_URL environment "
            "variable (e.g. smtp://user:pass@smtp.example.com:587). "
            "Skipping email notification.",
            stacklevel=3,
        )
        return

    subject = f"[Rampart] Approval required: {payload['tool_name']}"
    body = (
        f"A Rampart agent is waiting for human approval to call:\n\n"
        f"  Tool:      {payload['tool_name']}\n"
        f"  Run ID:    {payload['run_id']}\n"
        f"  Thread ID: {payload['thread_id']}\n"
        f"  Node:      {payload['node_name']}\n"
        f"  Call ID:   {payload['call_id']}\n\n"
        f"  Args:\n{json.dumps(payload['args'], indent=4)}\n\n"
        f"This request will time out after {policy.timeout_seconds}s.  "
        f"On timeout, the policy is: '{policy.on_timeout}'.\n"
    )

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            _send_smtp_sync,
            smtp_url,
            policy.delivery_target,
            subject,
            body,
        )
    except Exception as exc:
        logger.warning("Email approval notification failed: %s", exc)


def _send_smtp_sync(smtp_url: str, to_addr: str, subject: str, body: str) -> None:
    import smtplib
    from email.mime.text import MIMEText
    from urllib.parse import urlparse

    parsed = urlparse(smtp_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 587
    user = parsed.username
    password = parsed.password
    from_addr = f"{user}@{host}" if user else f"rampart@{host}"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr

    with smtplib.SMTP(host, port) as smtp:
        smtp.starttls()
        if user and password:
            smtp.login(user, password)
        smtp.send_message(msg)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _resolve_timeout(
    policy: Any,
    *,
    run_id: str = "",
    thread_id: str = "",
    node_name: str = "",
) -> bool:
    """Apply the on_timeout policy after a notification is sent.

    Returns True (auto-approve), False (auto-deny), or raises PermissionDeniedError
    (hard_stop).
    """
    if policy.on_timeout == "approve":
        return True
    elif policy.on_timeout == "deny":
        return False
    else:  # "hard_stop"
        from datetime import datetime, timezone

        from ._models import PermissionDeniedError, PermissionViolationEvent

        event = PermissionViolationEvent(
            run_id=run_id,
            thread_id=thread_id,
            node_name=node_name,
            violation_type="tool_not_in_whitelist",
            attempted_action="approval timed out (hard_stop policy)",
            declared_scope=None,  # type: ignore[arg-type]
            timestamp=datetime.now(timezone.utc),
        )
        raise PermissionDeniedError(event)


def _get_smtp_url() -> str | None:
    import os

    return os.environ.get("RAMPART_SMTP_URL")


def _safe_json(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    return str(obj)
