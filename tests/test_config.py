"""Tests for the global configuration helpers (rampart.configure)."""

from __future__ import annotations

import os

import pytest

import rampart
from rampart import _globals


@pytest.fixture(autouse=True)
def _restore_globals():
    """Capture and restore module-level rampart globals around every test."""
    original = {
        "DEFAULT_CHECKPOINTER": _globals.DEFAULT_CHECKPOINTER,
        "DEFAULT_TRACER": _globals.DEFAULT_TRACER,
        "DEFAULT_ARTIFACT_STORE": _globals.DEFAULT_ARTIFACT_STORE,
        "HTTP_PROXY_PORT": getattr(_globals, "HTTP_PROXY_PORT", None),
    }
    proxy_env = (os.environ.get("HTTP_PROXY"), os.environ.get("HTTPS_PROXY"))
    yield
    for key, value in original.items():
        setattr(_globals, key, value)
    for var, value in zip(("HTTP_PROXY", "HTTPS_PROXY"), proxy_env, strict=True):
        if value is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = value


# ----------------------------------------------------------------------------
# configure() — sets module-level defaults
# ----------------------------------------------------------------------------


def test_configure_sets_checkpointer():
    sentinel = object()
    rampart.configure(checkpointer=sentinel)
    assert _globals.DEFAULT_CHECKPOINTER is sentinel


def test_configure_sets_tracer_and_artifact_store():
    tracer_sentinel = object()
    store_sentinel = object()
    rampart.configure(tracer=tracer_sentinel, artifact_store=store_sentinel)
    assert _globals.DEFAULT_TRACER is tracer_sentinel
    assert _globals.DEFAULT_ARTIFACT_STORE is store_sentinel


def test_configure_only_overwrites_explicit_kwargs():
    """Passing checkpointer=None must NOT clobber a previously-configured one;
    each kwarg is opt-in."""
    initial = object()
    _globals.DEFAULT_CHECKPOINTER = initial

    rampart.configure(tracer=object())  # checkpointer not passed
    assert _globals.DEFAULT_CHECKPOINTER is initial


def test_configure_http_proxy_sets_env_vars():
    rampart.configure(http_proxy_port=8888)
    assert _globals.HTTP_PROXY_PORT == 8888
