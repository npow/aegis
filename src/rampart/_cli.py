"""CLI for Rampart — rampart run, resume, history, eval, trace, permissions."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click


@click.group()
@click.version_option(package_name="rampart")
def main() -> None:
    """Rampart — production-safe LLM agent runtime."""


@main.command()
@click.argument("project_name")
def init(project_name: str) -> None:
    """Initialize a new Rampart project."""

    root = Path(project_name)
    if root.exists():
        click.echo(f"Directory '{project_name}' already exists.", err=True)
        raise SystemExit(1)

    root.mkdir(parents=True)
    (root / "src" / project_name).mkdir(parents=True)
    (root / "tests").mkdir()

    (root / "pyproject.toml").write_text(
        f'[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n\n'
        f'[project]\nname = "{project_name}"\nversion = "0.1.0"\ndependencies = ["rampart"]\n\n'
        f'[tool.pytest.ini_options]\nasyncio_mode = "auto"\ntestpaths = ["tests"]\n'
    )
    (root / "src" / project_name / "__init__.py").write_text("")
    (root / "src" / project_name / "agent.py").write_text(
        '"""Example Rampart agent."""\n\nfrom dataclasses import dataclass\nfrom rampart import AgentState, graph, node\n\n\n@dataclass\nclass MyState(AgentState):\n    query: str = ""\n    result: str = ""\n\n\n@node()\nasync def process_node(state: MyState) -> MyState:\n    return state.update(result=f"Processed: {state.query}")\n\n\n@graph(name="my-agent", version="1.0.0")\nasync def my_agent(state: MyState) -> MyState:\n    return await process_node(state)\n'
    )
    (root / "tests" / "test_agent.py").write_text(
        f'"""Tests for my_agent."""\n\nimport pytest\nfrom {project_name}.agent import MyState, my_agent\nfrom rampart import RunConfig\nfrom rampart.checkpointers import MemoryCheckpointer\n\n\nasync def test_my_agent():\n    result = await my_agent.run(\n        input=MyState(query="hello"),\n        config=RunConfig(thread_id="test-001", checkpointer=MemoryCheckpointer()),\n    )\n    assert result.status == "completed"\n    assert result.state.result == "Processed: hello"\n'
    )

    click.echo(f"✓ Created project '{project_name}'")
    click.echo(f"  cd {project_name} && pip install -e '.[dev]' && pytest")


@main.command()
@click.argument("graph_name")
@click.option("--input", "input_json", required=True, help="JSON input state")
@click.option("--thread-id", default=None, help="Thread ID (auto-generated if omitted)")
@click.option("--checkpointer", default="sqlite", show_default=True)
def run(graph_name: str, input_json: str, thread_id: str | None, checkpointer: str) -> None:
    """Run a graph with the given input."""
    import uuid as _uuid

    from rampart._decorators import _GRAPH_REGISTRY

    _thread_id = thread_id or f"cli-{_uuid.uuid4().hex[:8]}"

    graph_def = _GRAPH_REGISTRY.get(graph_name)
    if graph_def is None:
        click.echo(f"Graph '{graph_name}' not found. Did you import your module?", err=True)
        raise SystemExit(1)

    try:
        input_data = json.loads(input_json)
    except json.JSONDecodeError as exc:
        click.echo(f"Invalid JSON input: {exc}", err=True)
        raise SystemExit(1) from None

    from rampart._models import RunConfig
    from rampart._runtime import _deserialize_state, _infer_state_type
    from rampart.checkpointers import MemoryCheckpointer, SqliteCheckpointer

    state_type = _infer_state_type(graph_def)
    input_state = _deserialize_state(input_data, state_type)
    cp = SqliteCheckpointer() if checkpointer == "sqlite" else MemoryCheckpointer()
    config = RunConfig(thread_id=_thread_id, checkpointer=cp)

    result = asyncio.run(graph_def.run(input=input_state, config=config))

    click.echo(
        json.dumps(
            {
                "status": result.status,
                "thread_id": _thread_id,
                "run_id": result.run_id,
                "state": result.trace.final_state,
                "cost_usd": result.trace.total_cost_usd,
                "wall_time_seconds": result.trace.wall_time_seconds,
            },
            indent=2,
        )
    )


@main.command()
@click.argument("graph_name")
@click.option("--thread-id", required=True, help="Thread ID to resume")
@click.option("--checkpointer", default="sqlite", show_default=True)
def resume(graph_name: str, thread_id: str, checkpointer: str) -> None:
    """Resume a failed or interrupted run from its last checkpoint."""
    from rampart._decorators import _GRAPH_REGISTRY
    from rampart._models import RunConfig
    from rampart.checkpointers import MemoryCheckpointer, SqliteCheckpointer

    graph_def = _GRAPH_REGISTRY.get(graph_name)
    if graph_def is None:
        click.echo(f"Graph '{graph_name}' not found.", err=True)
        raise SystemExit(1)

    cp = SqliteCheckpointer() if checkpointer == "sqlite" else MemoryCheckpointer()
    config = RunConfig(thread_id=thread_id, checkpointer=cp)
    result = asyncio.run(graph_def.resume(thread_id=thread_id, config=config))

    click.echo(
        json.dumps(
            {
                "status": result.status,
                "thread_id": thread_id,
                "run_id": result.run_id,
            },
            indent=2,
        )
    )


@main.command()
@click.argument("graph_name")
@click.option("--thread-id", required=True, help="Thread ID")
def history(graph_name: str, thread_id: str) -> None:
    """Show checkpoint history for a thread."""
    from rampart._decorators import _GRAPH_REGISTRY
    from rampart.checkpointers import SqliteCheckpointer

    graph_def = _GRAPH_REGISTRY.get(graph_name)
    if graph_def is None:
        click.echo(f"Graph '{graph_name}' not found.", err=True)
        raise SystemExit(1)

    cp = SqliteCheckpointer()
    checkpoints = asyncio.run(cp.get_history(thread_id, graph_name))
    if not checkpoints:
        click.echo(f"No checkpoints found for thread '{thread_id}'.")
        return

    for ckpt in checkpoints:
        click.echo(
            f"  step={ckpt.step:3d}  node={ckpt.node_name:<20s}  "
            f"at={ckpt.created_at.isoformat()[:19]}  id={ckpt.id}"
        )


@main.command("eval")
@click.argument("graph_name")
@click.option("--suite", required=True, help="Path to Python file defining the EvalSuite")
def run_eval(graph_name: str, suite: str) -> None:
    """Run an eval suite and print results."""
    import importlib.util

    spec_obj = importlib.util.spec_from_file_location("_eval_suite", suite)
    if spec_obj is None:
        click.echo(f"Cannot load eval suite from '{suite}'", err=True)
        raise SystemExit(1)
    mod = importlib.util.module_from_spec(spec_obj)
    if spec_obj.loader is None:
        click.echo(f"Cannot execute eval suite '{suite}': no loader found", err=True)
        raise SystemExit(1)
    spec_obj.loader.exec_module(mod)  # type: ignore[union-attr]

    # Find a variable named 'suite' in the module
    suite_obj = getattr(mod, "suite", None)
    if suite_obj is None:
        click.echo("No variable named 'suite' found in the eval file.", err=True)
        raise SystemExit(1)

    from rampart._models import EvalGateFailure

    results = asyncio.run(suite_obj.run())
    click.echo(results.summary())

    try:
        results.assert_gates()
    except EvalGateFailure as exc:
        click.echo(f"\n✗ Eval gate FAILED: {exc}", err=True)
        raise SystemExit(1) from None

    click.echo(f"\n✓ Eval gate passed ({results.pass_rate:.0%})")


@main.command()
@click.argument("graph_name")
@click.option("--thread-id", required=True, help="Thread ID to trace")
@click.option("--follow", is_flag=True, help="Follow live updates (not implemented in v1)")
def trace(graph_name: str, thread_id: str, follow: bool) -> None:
    """Show the execution trace for a thread."""
    from rampart._decorators import _GRAPH_REGISTRY
    from rampart.checkpointers import SqliteCheckpointer

    graph_def = _GRAPH_REGISTRY.get(graph_name)
    if graph_def is None:
        click.echo(f"Graph '{graph_name}' not found.", err=True)
        raise SystemExit(1)

    cp = SqliteCheckpointer()
    checkpoints = asyncio.run(cp.get_history(thread_id, graph_name))
    if not checkpoints:
        click.echo(f"No trace found for thread '{thread_id}'.")
        return

    click.echo(f"Thread: {thread_id}  Graph: {graph_name}")
    click.echo("─" * 60)
    for ckpt in checkpoints:
        click.echo(f"  [{ckpt.step:3d}] {ckpt.node_name:<25s}  {ckpt.created_at.isoformat()[:19]}")


@main.command()
@click.argument("graph_name")
def permissions(graph_name: str) -> None:
    """Show the declared permission scope for a graph."""
    from rampart._decorators import _GRAPH_REGISTRY

    graph_def = _GRAPH_REGISTRY.get(graph_name)
    if graph_def is None:
        click.echo(f"Graph '{graph_name}' not found.", err=True)
        raise SystemExit(1)

    scope = graph_def.permissions
    if scope is None:
        click.echo(
            f"Graph '{graph_name}' has no permission scope declared (all tools and network allowed)."
        )
        return

    click.echo(f"Permission scope for '{graph_name}':")
    if scope.tools is not None:
        click.echo(f"  Tools:   {scope.tools}")
    else:
        click.echo("  Tools:   (all registered tools)")
    if scope.network:
        net = scope.network
        click.echo(
            f"  Network: allowed_domains={net.allowed_domains}, deny_all_others={net.deny_all_others}"
        )
    if scope.filesystem:
        fs = scope.filesystem
        click.echo(f"  Files:   read={fs.read}, write={fs.write}")
        if fs.read_allowed_paths:
            click.echo(f"           read_paths={fs.read_allowed_paths}")
        if fs.write_allowed_paths:
            click.echo(f"           write_paths={fs.write_allowed_paths}")

    # Show per-tool permissions
    from rampart._decorators import _TOOL_REGISTRY

    tools_with_perms = {
        name: td
        for name, td in _TOOL_REGISTRY.items()
        if td.require_human_approval or td.permissions is not None
    }
    if tools_with_perms:
        click.echo("  Per-tool:")
        for name, td in tools_with_perms.items():
            flags: list[str] = []
            if td.require_human_approval:
                flags.append(
                    f"require_human_approval (timeout={td.approval_timeout_seconds}s, on_timeout={td.approval_on_timeout})"
                )
            if td.permissions:
                flags.append(f"permissions={td.permissions}")
            click.echo(f"    {name}: {', '.join(flags)}")


@main.command()
@click.argument("graph_name")
@click.option("--input", "input_json", required=True, help="JSON input state")
@click.option("--cassette", required=True, help="Path to save the cassette file")
@click.option("--thread-id", default=None)
def record(graph_name: str, input_json: str, cassette: str, thread_id: str | None) -> None:
    """Record a cassette from a live graph run."""
    import uuid as _uuid

    from rampart._decorators import _GRAPH_REGISTRY
    from rampart._models import RunConfig
    from rampart._runtime import _deserialize_state, _infer_state_type
    from rampart.checkpointers import MemoryCheckpointer
    from rampart.testing import cassette as _cassette

    _thread_id = thread_id or f"cassette-{_uuid.uuid4().hex[:8]}"
    graph_def = _GRAPH_REGISTRY.get(graph_name)
    if graph_def is None:
        click.echo(f"Graph '{graph_name}' not found.", err=True)
        raise SystemExit(1)

    input_data = json.loads(input_json)
    state_type = _infer_state_type(graph_def)
    input_state = _deserialize_state(input_data, state_type)
    config = RunConfig(thread_id=_thread_id, checkpointer=MemoryCheckpointer())

    async def _run() -> None:
        async with _cassette.record(cassette):
            await graph_def.run(input=input_state, config=config)

    asyncio.run(_run())
    click.echo(f"✓ Cassette written to {cassette}")
