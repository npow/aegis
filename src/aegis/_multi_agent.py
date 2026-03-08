"""Multi-agent composition: chain(), parallel(), supervisor()."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ._decorators import GraphDef
from ._models import AgentState, RunConfig, RunResult

# ── chain() ───────────────────────────────────────────────────────────────────


class ChainGraph:
    """Executes a sequence of graphs, passing state from one to the next."""

    def __init__(self, *graphs: GraphDef) -> None:
        self.graphs = list(graphs)

    async def run(
        self,
        input: AgentState,
        config: RunConfig,
        **kwargs: Any,
    ) -> RunResult:
        if not self.graphs:
            raise ValueError("ChainGraph requires at least one graph")
        state = input
        last_result: RunResult | None = None
        for i, g in enumerate(self.graphs):
            sub_config = RunConfig(
                thread_id=f"{config.thread_id}-chain-{i}",
                checkpointer=config.checkpointer,
                metadata=config.metadata,
            )
            last_result = await g.run(input=state, config=sub_config, **kwargs)
            if last_result.status != "completed":
                return last_result
            state = last_result.state
        if last_result is None:  # pragma: no cover — guarded by empty check above
            raise RuntimeError("ChainGraph produced no result")
        return last_result


def chain(*graphs: GraphDef) -> ChainGraph:
    """Execute graphs sequentially, piping state through each."""
    return ChainGraph(*graphs)


# ── parallel() ────────────────────────────────────────────────────────────────


class ParallelGraph:
    """Fan-out: run multiple graphs concurrently with the same input."""

    def __init__(self, *graphs: GraphDef) -> None:
        self.graphs = list(graphs)
        self._join_graph: GraphDef | None = None

    def join(self, join_graph: GraphDef) -> ParallelWithJoin:
        return ParallelWithJoin(self.graphs, join_graph)

    async def run(
        self,
        input: AgentState,
        config: RunConfig,
        **kwargs: Any,
    ) -> list[RunResult]:
        tasks = []
        for i, g in enumerate(self.graphs):
            sub_config = RunConfig(
                thread_id=f"{config.thread_id}-par-{i}",
                checkpointer=config.checkpointer,
                metadata=config.metadata,
            )
            tasks.append(asyncio.create_task(g.run(input=input, config=sub_config, **kwargs)))
        try:
            return list(await asyncio.gather(*tasks))
        except Exception:
            # Cancel any still-running branches to avoid orphaned tasks
            for t in tasks:
                t.cancel()
            # Drain cancellations
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


class ParallelWithJoin:
    """Fan-out + join: run parallel graphs then synthesize with a join graph."""

    def __init__(self, graphs: list[GraphDef], join_graph: GraphDef) -> None:
        self.graphs = graphs
        self.join_graph = join_graph

    async def run(
        self,
        input: AgentState,
        config: RunConfig,
        **kwargs: Any,
    ) -> RunResult:
        # Run parallel branches
        tasks = []
        for i, g in enumerate(self.graphs):
            sub_config = RunConfig(
                thread_id=f"{config.thread_id}-par-{i}",
                checkpointer=config.checkpointer,
                metadata=config.metadata,
            )
            tasks.append(asyncio.create_task(g.run(input=input, config=sub_config, **kwargs)))
        try:
            results: list[RunResult] = list(await asyncio.gather(*tasks))
        except Exception:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        # Merge results into a list annotation on the state
        # Convention: join graph receives the state from the first branch;
        # parallel results are attached as a special attribute for the join graph
        # to access via its first node's state.
        merged_state = _merge_parallel_results(input, results)
        join_config = RunConfig(
            thread_id=f"{config.thread_id}-join",
            checkpointer=config.checkpointer,
            metadata=config.metadata,
        )
        return await self.join_graph.run(input=merged_state, config=join_config, **kwargs)


def parallel(*graphs: GraphDef) -> ParallelGraph:
    """Run multiple graphs in parallel with the same input."""
    return ParallelGraph(*graphs)


# ── supervisor() ──────────────────────────────────────────────────────────────


class SupervisorGraph:
    """Router + specialists: the router graph decides which specialist handles each request."""

    def __init__(
        self,
        router: GraphDef,
        specialists: dict[str, GraphDef],
        *,
        max_handoffs: int = 5,
        handoff_timeout: int = 300,
    ) -> None:
        self.router = router
        self.specialists = specialists
        self.max_handoffs = max_handoffs
        self.handoff_timeout = handoff_timeout

    async def run(
        self,
        input: AgentState,
        config: RunConfig,
        **kwargs: Any,
    ) -> RunResult:
        state = input
        for handoff in range(self.max_handoffs):
            # Run the router to decide which specialist
            router_config = RunConfig(
                thread_id=f"{config.thread_id}-router-{handoff}",
                checkpointer=config.checkpointer,
                metadata=config.metadata,
            )
            router_result = await self.router.run(input=state, config=router_config, **kwargs)
            if router_result.status != "completed":
                return router_result

            # Router should set a `next_specialist` field on the state
            specialist_key = getattr(router_result.state, "next_specialist", None)
            if specialist_key is None or specialist_key == "__done__":
                return router_result

            specialist = self.specialists.get(specialist_key)
            if specialist is None:
                import uuid as _uuid
                from datetime import datetime

                from ._models import RunError, RunTrace

                trace = RunTrace(
                    run_id=f"run-{_uuid.uuid4().hex[:8]}",
                    thread_id=config.thread_id,
                    graph_name="supervisor",
                    graph_version="1.0.0",
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    status="failed",
                    error=f"Unknown specialist: '{specialist_key}'",
                )
                return RunResult(
                    state=router_result.state,
                    trace=trace,
                    status="failed",
                    error=RunError(
                        message=f"Router returned unknown specialist key: '{specialist_key}'",
                        exception_type="KeyError",
                    ),
                )

            spec_config = RunConfig(
                thread_id=f"{config.thread_id}-spec-{specialist_key}-{handoff}",
                checkpointer=config.checkpointer,
                metadata=config.metadata,
            )
            try:
                spec_result = await asyncio.wait_for(
                    specialist.run(input=router_result.state, config=spec_config, **kwargs),
                    timeout=self.handoff_timeout,
                )
            except asyncio.TimeoutError:
                import uuid as _uuid
                from datetime import datetime

                from ._models import RunError, RunTrace

                trace = RunTrace(
                    run_id=f"run-{_uuid.uuid4().hex[:8]}",
                    thread_id=config.thread_id,
                    graph_name="supervisor",
                    graph_version="1.0.0",
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    status="failed",
                    error=f"Specialist '{specialist_key}' timed out after {self.handoff_timeout}s",
                )
                return RunResult(
                    state=router_result.state,
                    trace=trace,
                    status="failed",
                    error=RunError(
                        message=f"Specialist '{specialist_key}' timed out after {self.handoff_timeout}s",
                        exception_type="TimeoutError",
                    ),
                )
            state = spec_result.state
            if spec_result.status != "completed":
                return spec_result

            # If specialist signals done, return immediately
            next_after_spec = getattr(spec_result.state, "next_specialist", None)
            if next_after_spec == "__done__":
                return spec_result

        # Max handoffs reached
        import uuid as _uuid
        from datetime import datetime

        from ._models import RunError, RunTrace

        trace = RunTrace(
            run_id=f"run-{_uuid.uuid4().hex[:8]}",
            thread_id=config.thread_id,
            graph_name="supervisor",
            graph_version="1.0.0",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            status="failed",
            error=f"Max handoffs ({self.max_handoffs}) exceeded",
        )
        return RunResult(
            state=state,
            trace=trace,
            status="failed",
            error=RunError(
                message=f"Supervisor exceeded max_handoffs ({self.max_handoffs})",
                exception_type="MaxHandoffsError",
            ),
        )


def supervisor(
    router: GraphDef,
    specialists: dict[str, GraphDef],
    *,
    max_handoffs: int = 5,
    handoff_timeout: int = 300,
) -> SupervisorGraph:
    """Create a supervisor that routes to one of several specialist graphs."""
    return SupervisorGraph(
        router=router,
        specialists=specialists,
        max_handoffs=max_handoffs,
        handoff_timeout=handoff_timeout,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _merge_parallel_results(original: AgentState, results: list[RunResult]) -> AgentState:
    """Attach parallel results to the state as _parallel_results attribute."""
    # We use a simple convention: attach the results as a dynamic attribute
    # The join graph node can access state._parallel_results
    import dataclasses

    @dataclass
    class _MergedState(type(original)):  # type: ignore[misc]
        _parallel_results: list[RunResult] = field(default_factory=list)

    try:
        merged = _MergedState(
            **{f.name: getattr(original, f.name) for f in dataclasses.fields(original)},
            _parallel_results=results,
        )
        return merged  # type: ignore[return-value]
    except Exception:
        # Fallback: return original with results as attribute
        result = dataclasses.replace(original)
        object.__setattr__(result, "_parallel_results", results)
        return result
