# Rampart

[![CI](https://github.com/npow/rampart/actions/workflows/ci.yml/badge.svg)](https://github.com/npow/rampart/actions/workflows/ci.yml)
[![Release](https://github.com/npow/rampart/actions/workflows/release.yml/badge.svg)](https://github.com/npow/rampart/actions/workflows/release.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![Docs](https://img.shields.io/badge/docs-mintlify-18a34a?style=flat-square)](https://mintlify.com/npow/rampart)

**A runtime that makes LLM agents production-safe by default.**

> Built for teams who need more than a prototype framework when shipping agents to production.

---

## The problem

Prototypes work in demos. Production is different.

- A process crash at step 7 of 12 loses everything — you restart from zero.
- A prompt change silently breaks tool selection; nobody notices until a user report.
- A runaway loop burns $800 in API credits before anyone wakes up.
- The Replit incident (July 2025) — an agent deleted 1,200+ production records despite an explicit code-and-action freeze — wasn't an edge case. It was the predictable result of a framework with no permission enforcement model.

Every team shipping agents today independently assembles Temporal + LangGraph + Langfuse + `unittest.mock`, paying the integration tax on every project. Rampart ships all of it in the box.

---

## Quickstart

```bash
pip install rampart
pip install "rampart[sqlite]"   # SQLite checkpointing
```

```python
from dataclasses import dataclass, field
from rampart import graph, node, tool, AgentState, RunConfig
from rampart.checkpointers import SqliteCheckpointer

@dataclass
class ResearchState(AgentState):
    query: str = ""
    results: list[str] = field(default_factory=list)
    summary: str = ""

@tool(name="web_search")
async def web_search(query: str) -> list[str]:
    ...  # your implementation

@node(retries=3, retry_on=(TimeoutError,), timeout_seconds=30)
async def search_node(state: ResearchState, tools) -> ResearchState:
    results = await tools.web_search(query=state.query)
    return state.update(results=results)

@node(timeout_seconds=60)
async def summarize_node(state: ResearchState) -> ResearchState:
    return state.update(summary=f"Found {len(state.results)} results for: {state.query}")

@graph(name="research", version="1.0.0")
async def research_pipeline(state: ResearchState) -> ResearchState:
    state = await search_node(state)
    state = await summarize_node(state)
    return state

result = await research_pipeline.run(
    input=ResearchState(query="quantum computing"),
    config=RunConfig(
        thread_id="session-001",
        checkpointer=SqliteCheckpointer("./agent.db"),
    ),
)
print(result.state.summary)
print(f"Cost: ${result.trace.total_cost_usd:.4f}")
```

---

## Features

### Durable execution — crash anywhere, resume from the last step

State is checkpointed after every node. A process restart, OOM kill, or rolling deployment resumes from the last committed step without re-executing completed work.

```python
result = await research_pipeline.resume(
    thread_id="session-001",
    config=RunConfig(checkpointer=SqliteCheckpointer("./agent.db")),
)
```

LangGraph checkpoints within a process. Rampart checkpoints survive restarts.

### Testing without live APIs

`mock_tools()` and cassette record/replay are built into the framework. Your CI pipeline makes no network calls and produces identical results on every run.

```python
# Record once against live APIs
async with cassette.record("tests/fixtures/research.json"):
    await research_pipeline.run(input=ResearchState(query="quantum computing"))

# Replay forever — zero cost, zero flakiness
async with cassette.replay("tests/fixtures/research.json"):
    result = await research_pipeline.run(input=ResearchState(query="quantum computing"))
assert result.state.summary != ""
```

```python
# Mock specific tools for unit tests
async with research_pipeline.mock_tools({
    "web_search": MockTool.returns([{"text": "result 1"}]),
    "send_email": MockTool.noop(),
}) as ctx:
    result = await research_pipeline.run(input=state)

assert ctx.calls["web_search"].count == 1
assert ctx.calls["send_email"].count == 0
```

### Eval as a deployment gate

`EvalSuite` runs in CI and blocks promotion when assertions fail. Plug it into your pipeline the same way you'd block on a failing test.

```python
suite = EvalSuite(
    name="research-pipeline-v2",
    graph=research_pipeline,
    cases=[
        EvalCase(
            id="basic-query",
            input=ResearchState(query="quantum computing"),
            cassette="tests/fixtures/research.json",
            assertions=[
                ToolCallAssertion(tool_name="web_search", called=True, min_times=1),
                SchemaAssertion(
                    predicate=lambda s: len(s.summary) > 100,
                    description="summary must be substantive",
                ),
            ],
        ),
    ],
    pass_rate_gate=1.0,
)

results = await suite.run()
results.assert_gates()  # raises EvalGateFailure if any case fails
```

### Permission scopes

Every graph declares what it's allowed to do. Violations are blocked and logged before execution.

```python
@graph(
    name="data-analyst",
    permissions=PermissionScope(
        tools=["query_db", "write_report"],
        network=NetworkPermission(
            allowed_domains=["api.internal.company.com"],
            deny_all_others=True,
        ),
    ),
)
async def analyst_pipeline(state): ...
```

Enforcement runs at three layers: the tool call boundary, the Python HTTP transport (`httpx`, `requests`, and `urllib` are patched at import time), and an optional sandboxed execution context. An agent cannot escape by calling `httpx.get()` directly.

### Budget envelopes

Budgets are hard runtime constraints checked before each node, not alerts you configure after the first incident.

```python
result = await research_pipeline.run(
    input=state,
    config=RunConfig(thread_id="session-xyz"),
    budget=Budget(
        max_llm_cost_usd=0.50,
        max_tool_calls=20,
        max_wall_time_seconds=120,
        on_exceeded="hard_stop",
    ),
)
# result.status == "budget_exceeded" — never silently exceeded
```

---

## How it compares

| | **Rampart** | **LangGraph** | **CrewAI** | **AutoGen** |
|---|---|---|---|---|
| Crash recovery (survives restart) | ✅ | ⚠️ in-process only | ❌ | ❌ |
| Testing without live APIs | ✅ built-in | ⚠️ roll your own | ⚠️ roll your own | ⚠️ roll your own |
| CI eval gate | ✅ built-in | ❌ | ❌ | ❌ |
| Permission enforcement | ✅ enforced | ❌ | ❌ | ❌ |
| HTTP transport interception | ✅ | ❌ | ❌ | ❌ |
| Budget hard limits | ✅ | ❌ | ❌ | ❌ |
| Multi-agent composition | ✅ | ✅ | ✅ | ✅ |

LangGraph is the closest comparison. Rampart has the same decorator ergonomics and adds durable multi-process recovery, built-in testing primitives, eval gates, and permission enforcement. CrewAI and AutoGen prioritize role-based multi-agent patterns and leave infrastructure concerns to you.

---

## Multi-agent composition

```python
from rampart import chain, parallel, supervisor

# Sequential
pipeline = chain(fetch_graph, analyze_graph, report_graph)

# Fan-out then join
pipeline = parallel(search_graph, db_graph).join(synthesize_graph)

# Router + specialists
pipeline = supervisor(
    router=classifier_graph,
    specialists={"billing": billing_graph, "support": support_graph},
    max_handoffs=5,
)

result = await pipeline.run(input=state, config=config)
```

---

## Checkpoint backends

| Backend | Install | Use case |
|---|---|---|
| `MemoryCheckpointer` | built-in | Tests, local dev |
| `SqliteCheckpointer` | `pip install "rampart[sqlite]"` | Single-process, local |
| `PostgresCheckpointer` | `pip install "rampart[postgres]"` | Production, multi-process |
