"""Tests for permission scope enforcement."""

from dataclasses import dataclass

import pytest

from aegis import (
    AgentState,
    FilesystemPermission,
    NetworkPermission,
    PermissionDeniedError,
    PermissionScope,
    RunConfig,
    graph,
    node,
    tool,
)
from aegis._permissions import (
    _domain_matches_any,
    _path_matches_any,
    check_filesystem_permission,
    check_network_permission,
    check_tool_permission,
)
from aegis.checkpointers import MemoryCheckpointer
from aegis.testing import MockTool

# ── Unit tests for permission helpers ─────────────────────────────────────────


def test_domain_matching_exact():
    assert _domain_matches_any("arxiv.org", ["arxiv.org"]) is True
    assert _domain_matches_any("arxiv.org", ["example.com"]) is False


def test_domain_matching_wildcard():
    assert _domain_matches_any("en.wikipedia.org", ["*.wikipedia.org"]) is True
    assert _domain_matches_any("wikipedia.org", ["*.wikipedia.org"]) is False
    assert _domain_matches_any("a.b.c.wikipedia.org", ["*.wikipedia.org"]) is False


def test_path_matching_glob():
    assert _path_matches_any("/tmp/output/file.txt", ["/tmp/output/**"]) is True
    assert _path_matches_any("/etc/passwd", ["/tmp/output/**"]) is False
    assert _path_matches_any("/tmp/reports/q1.pdf", ["/tmp/reports/**"]) is True


def test_check_tool_permission_allows_whitelisted():
    scope = PermissionScope(tools=["web_search", "write_file"])
    # Should not raise
    check_tool_permission("web_search", scope, "r1", "t1", "node")


def test_check_tool_permission_blocks_non_whitelisted():
    scope = PermissionScope(tools=["web_search"])
    with pytest.raises(PermissionDeniedError) as exc_info:
        check_tool_permission("delete_database", scope, "r1", "t1", "node")
    assert exc_info.value.event.violation_type == "tool_not_in_whitelist"


def test_check_tool_permission_none_scope_allows_all():
    # None scope means no restrictions
    check_tool_permission("any_tool", None, "r1", "t1", "node")


def test_check_tool_permission_none_tools_list_allows_all():
    scope = PermissionScope(tools=None)  # None = all tools allowed
    check_tool_permission("any_tool", scope, "r1", "t1", "node")


def test_check_network_permission_allows_whitelisted_domain():
    scope = PermissionScope(
        network=NetworkPermission(
            allowed_domains=["arxiv.org", "*.wikipedia.org"],
            deny_all_others=True,
        )
    )
    check_network_permission("https://arxiv.org/paper", scope, "r1", "t1", "node")
    check_network_permission("https://en.wikipedia.org/wiki", scope, "r1", "t1", "node")


def test_check_network_permission_blocks_unknown_domain():
    scope = PermissionScope(
        network=NetworkPermission(
            allowed_domains=["arxiv.org"],
            deny_all_others=True,
        )
    )
    with pytest.raises(PermissionDeniedError) as exc_info:
        check_network_permission("https://evil.com/steal", scope, "r1", "t1", "node")
    assert exc_info.value.event.violation_type == "network_domain_denied"


def test_check_filesystem_permission_allows_whitelisted_path():
    scope = PermissionScope(
        filesystem=FilesystemPermission(
            write=True,
            write_allowed_paths=["/tmp/output/**"],
        )
    )
    check_filesystem_permission("/tmp/output/report.pdf", "write", scope, "r1", "t1", "node")


def test_check_filesystem_permission_blocks_unlisted_path():
    scope = PermissionScope(
        filesystem=FilesystemPermission(
            write=True,
            write_allowed_paths=["/tmp/output/**"],
        )
    )
    with pytest.raises(PermissionDeniedError) as exc_info:
        check_filesystem_permission("/etc/cron/evil", "write", scope, "r1", "t1", "node")
    assert exc_info.value.event.violation_type == "filesystem_path_denied"


def test_check_filesystem_permission_blocks_write_when_disabled():
    scope = PermissionScope(filesystem=FilesystemPermission(write=False))
    with pytest.raises(PermissionDeniedError):
        check_filesystem_permission("/tmp/anything", "write", scope, "r1", "t1", "node")


# ── Integration: permission scope enforced during graph run ───────────────────


@dataclass
class PermState(AgentState):
    action: str = ""
    done: bool = False


@tool(name="allowed_tool_perm_test")
async def allowed_tool_perm_test() -> str:
    return "ok"


@tool(name="forbidden_tool_perm_test")
async def forbidden_tool_perm_test() -> str:
    return "should not reach"


async def test_graph_blocks_non_whitelisted_tool_in_scope():
    @node()
    async def call_forbidden(state: PermState, tools) -> PermState:
        await tools.forbidden_tool_perm_test()
        return state.update(done=True)

    @graph(
        name=f"perm-block-graph-v{id(call_forbidden)}",
        version="1.0.0",
        permissions=PermissionScope(tools=["allowed_tool_perm_test"]),  # forbidden not listed
    )
    async def perm_block_graph(state: PermState) -> PermState:
        return await call_forbidden(state)

    cp = MemoryCheckpointer()
    # No mock: even if it were mocked, permission check happens BEFORE mock dispatch
    result = await perm_block_graph.run(
        input=PermState(),
        config=RunConfig(thread_id="perm-block-001", checkpointer=cp),
    )
    assert result.status == "failed"
    assert result.error is not None
    assert (
        "Permission denied" in result.error.message
        or "tool_not_in_whitelist" in result.error.message
    )


async def test_graph_allows_whitelisted_tool_in_scope():
    @node()
    async def call_allowed(state: PermState, tools) -> PermState:
        await tools.allowed_tool_perm_test()
        return state.update(done=True)

    @graph(
        name=f"perm-allow-graph-v{id(call_allowed)}",
        version="1.0.0",
        permissions=PermissionScope(tools=["allowed_tool_perm_test"]),
    )
    async def perm_allow_graph(state: PermState) -> PermState:
        return await call_allowed(state)

    cp = MemoryCheckpointer()
    async with perm_allow_graph.mock_tools({"allowed_tool_perm_test": MockTool.returns("ok")}):
        result = await perm_allow_graph.run(
            input=PermState(),
            config=RunConfig(thread_id="perm-allow-001", checkpointer=cp),
        )
    assert result.status == "completed"
    assert result.state.done is True


async def test_graph_no_scope_allows_all_tools():
    """A graph without permissions allows any tool."""

    @node()
    async def call_any(state: PermState, tools) -> PermState:
        await tools.forbidden_tool_perm_test()
        return state.update(done=True)

    @graph(name=f"no-scope-graph-v{id(call_any)}", version="1.0.0")  # no permissions=
    async def no_scope_graph(state: PermState) -> PermState:
        return await call_any(state)

    cp = MemoryCheckpointer()
    async with no_scope_graph.mock_tools({"forbidden_tool_perm_test": MockTool.returns("ok")}):
        result = await no_scope_graph.run(
            input=PermState(),
            config=RunConfig(thread_id="no-scope-001", checkpointer=cp),
        )
    assert result.status == "completed"
