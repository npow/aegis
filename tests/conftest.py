"""Shared fixtures for Aegis tests."""

import pytest

from aegis.checkpointers import MemoryCheckpointer


@pytest.fixture
def mem_cp():
    """Fresh MemoryCheckpointer per test."""
    return MemoryCheckpointer()
