"""All data models for Rampart — translated from PRD section 8."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, get_args

# ── State ─────────────────────────────────────────────────────────────────────


@dataclass
class AgentState:
    """Base class for all graph state objects.

    Must be JSON-serializable. Fields must be typed.
    Subclasses define domain-specific state.
    """

    thread_id: str = ""
    run_id: str = ""

    def update(self, **kwargs: Any) -> AgentState:
        """Return a new instance with specified fields replaced. Does not mutate self."""
        return dataclasses.replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        """Deserialize from a dict, ignoring unknown fields."""
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ── Run Config ────────────────────────────────────────────────────────────────


@dataclass
class RunConfig:
    """Configuration for a single graph run."""

    thread_id: str
    # Allow test/local overrides without touching graph definition
    checkpointer: Any | None = None  # CheckpointerBase instance
    artifact_store: Any | None = None  # ArtifactStoreBase instance
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Checkpointing ─────────────────────────────────────────────────────────────


@dataclass
class Checkpoint:
    """Persisted, immutable snapshot of AgentState at a specific step."""

    id: str  # "ckpt_{graph}_{thread}_{step}_{state_hash}"
    thread_id: str
    run_id: str
    graph_name: str
    graph_version: str
    step: int  # monotonically increasing within a run
    node_name: str  # node that just completed (or "__input__" for step 0)
    state_snapshot: dict[str, Any]  # JSON-serialized AgentState
    created_at: datetime
    parent_checkpoint_id: str | None  # None for step 0; set for sequential steps
    is_fork_root: bool = False  # True if created via fork()


@dataclass
class CheckpointBackendConfig:
    type: Literal["memory", "sqlite", "postgres", "redis", "dynamodb"]
    connection_string: str | None = None
    table_name: str = "rampart_checkpoints"
    ttl_days: int | None = None  # None = retain indefinitely


# ── Execution Tracing ─────────────────────────────────────────────────────────


@dataclass
class LLMCall:
    call_id: str
    model: str
    system_prompt: str | None
    user_prompt: str
    response: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float
    latency_ms: int
    timestamp: datetime
    node_name: str
    was_replayed: bool = False


@dataclass
class ToolCall:
    call_id: str
    tool_name: str
    args: dict[str, Any]
    result: Any
    error: str | None
    latency_ms: int
    timestamp: datetime
    node_name: str
    was_mocked: bool = False
    permission_checked: bool = True
    permission_granted: bool = True
    required_human_approval: bool = False
    human_approved: bool | None = None


@dataclass
class NodeTrace:
    node_name: str
    started_at: datetime
    completed_at: datetime | None
    input_state: dict[str, Any]
    output_state: dict[str, Any] | None
    llm_calls: list[LLMCall] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    attempt: int = 1
    status: Literal["running", "completed", "failed", "retrying"] = "running"
    error: str | None = None


@dataclass
class RunTrace:
    run_id: str
    thread_id: str
    graph_name: str
    graph_version: str
    started_at: datetime
    completed_at: datetime | None
    status: Literal["running", "completed", "failed", "paused", "budget_exceeded", "resumed", "cancelled"]
    nodes_executed: list[NodeTrace] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: float = 0.0
    wall_time_seconds: float = 0.0
    final_state: dict[str, Any] | None = None
    error: str | None = None
    parent_run_id: str | None = None
    otel_trace_id: str = ""

    def add_llm_call(self, call: LLMCall) -> None:
        self.total_input_tokens += call.input_tokens
        self.total_output_tokens += call.output_tokens
        self.total_cached_tokens += call.cached_tokens
        self.total_cost_usd += call.cost_usd


# ── Run Result ────────────────────────────────────────────────────────────────


@dataclass
class RunError:
    message: str
    exception_type: str
    traceback: str | None = None


@dataclass
class RunResult:
    """Result of a graph run."""

    state: AgentState
    trace: RunTrace
    status: Literal["completed", "failed", "budget_exceeded", "paused"]
    error: RunError | None = None

    @property
    def run_id(self) -> str:
        return self.trace.run_id


# ── Budget ────────────────────────────────────────────────────────────────────


@dataclass
class Budget:
    max_tokens: int | None = None
    max_llm_cost_usd: float | None = None
    max_tool_calls: int | None = None
    max_wall_time_seconds: int | None = None
    on_exceeded: Literal[
        "hard_stop",
        "pause_and_notify",
        "downgrade_model",
        "compress_context",
    ] = "hard_stop"
    downgrade_to: str | None = None  # required if on_exceeded="downgrade_model"
    notify_at_pct: float = 0.80


@dataclass
class BudgetStatus:
    tokens_used: int = 0
    cost_usd: float = 0.0
    tool_calls_made: int = 0
    wall_time_seconds: float = 0.0
    pct_consumed: dict[str, float] = field(default_factory=dict)
    exceeded_dimension: str | None = None

    def compute_pct(self, budget: Budget) -> None:
        pct: dict[str, float] = {}
        if budget.max_tokens:
            pct["tokens"] = self.tokens_used / budget.max_tokens
        if budget.max_llm_cost_usd:
            pct["cost"] = self.cost_usd / budget.max_llm_cost_usd
        if budget.max_tool_calls:
            pct["tool_calls"] = self.tool_calls_made / budget.max_tool_calls
        if budget.max_wall_time_seconds:
            pct["wall_time"] = self.wall_time_seconds / budget.max_wall_time_seconds
        self.pct_consumed = pct


@dataclass
class BudgetExceededEvent:
    run_id: str
    thread_id: str
    exceeded_dimension: str
    budget: Budget
    current_status: BudgetStatus
    checkpoint_id: str


@dataclass
class BudgetDecision:
    action: Literal["hard_stop", "extend", "downgrade"]
    updated_budget: Budget | None = None

    def __post_init__(self) -> None:
        allowed = get_args(self.__dataclass_fields__["action"].type)
        if not allowed:
            allowed = ("hard_stop", "extend", "downgrade")
        if self.action not in allowed:
            raise ValueError(
                f"Invalid action {self.action!r}; must be one of {allowed}"
            )

    @staticmethod
    def hard_stop() -> BudgetDecision:
        return BudgetDecision(action="hard_stop")

    @staticmethod
    def extend(current_budget: Budget, **overrides: Any) -> BudgetDecision:
        """Extend the *current* budget by overriding specific fields.

        Example (inside a budget exceeded handler)::

            return BudgetDecision.extend(event.budget, max_tokens=event.budget.max_tokens + 5000)
        """
        import dataclasses

        updated = dataclasses.replace(current_budget, **overrides)
        return BudgetDecision(action="extend", updated_budget=updated)

    @staticmethod
    def downgrade(model: str) -> BudgetDecision:
        return BudgetDecision(action="downgrade", updated_budget=Budget(downgrade_to=model))


# ── Permissions ───────────────────────────────────────────────────────────────


@dataclass
class NetworkPermission:
    allowed_domains: list[str] = field(default_factory=list)
    deny_all_others: bool = True
    max_bytes_out_per_run: int | None = None
    max_bytes_in_per_run: int | None = None


@dataclass
class FilesystemPermission:
    read: bool = False
    write: bool = False
    read_allowed_paths: list[str] = field(default_factory=list)
    write_allowed_paths: list[str] = field(default_factory=list)


@dataclass
class ApprovalPolicy:
    require_for_patterns: list[str] = field(default_factory=list)
    timeout_seconds: int = 3600
    on_timeout: Literal["hard_stop", "deny", "approve"] = "hard_stop"
    delivery: Literal["webhook", "slack", "email"] = "webhook"
    delivery_target: str | None = None

    def __post_init__(self) -> None:
        allowed = get_args(self.__dataclass_fields__["delivery"].type)
        if not allowed:
            allowed = ("webhook", "slack", "email")
        if self.delivery not in allowed:
            raise ValueError(
                f"Invalid delivery {self.delivery!r}; must be one of {allowed}"
            )


@dataclass
class PermissionScope:
    tools: list[str] | None = None  # None = all registered tools allowed
    network: NetworkPermission = field(default_factory=NetworkPermission)
    filesystem: FilesystemPermission = field(default_factory=FilesystemPermission)
    approval: ApprovalPolicy = field(default_factory=ApprovalPolicy)


@dataclass
class PermissionViolationEvent:
    run_id: str
    thread_id: str
    node_name: str
    violation_type: Literal[
        "tool_not_in_whitelist",
        "network_domain_denied",
        "filesystem_path_denied",
        "http_intercept_blocked",
    ]
    attempted_action: str
    declared_scope: PermissionScope
    timestamp: datetime
    callback: Callable[..., Any] | None = None


# ── Testing ───────────────────────────────────────────────────────────────────


@dataclass
class CassetteEntry:
    type: Literal["llm_call", "tool_call"]
    call_id: str
    step: int
    node_name: str
    request: dict[str, Any]
    response: dict[str, Any]
    timestamp: datetime


@dataclass
class CassetteRecord:
    format_version: str = "1.0"
    graph_name: str = ""
    graph_version: str = ""
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    python_version: str = ""
    entries: list[CassetteEntry] = field(default_factory=list)
    content_hash: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 of all entry request data."""
        data = json.dumps(
            [e.request for e in self.entries],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def update_hash(self) -> None:
        self.content_hash = self.compute_hash()


@dataclass
class MockCallRecord:
    tool_name: str
    count: int = 0
    calls: list[ToolCall] = field(default_factory=list)


@dataclass
class MockContext:
    calls: dict[str, MockCallRecord] = field(default_factory=dict)
    live_calls_made: int = 0


@dataclass
class CassetteReplayContext:
    cassette: CassetteRecord
    replay_calls_served: int = 0
    total_recorded_calls: int = 0
    live_calls_made: int = 0


# ── Eval ──────────────────────────────────────────────────────────────────────


@dataclass
class EvalAssertion:
    description: str


@dataclass
class ToolCallAssertion(EvalAssertion):
    tool_name: str = ""
    called: bool = True
    min_times: int = 1
    max_times: int | None = None
    args_match: dict[str, Any] | None = None


@dataclass
class SchemaAssertion(EvalAssertion):
    predicate: Callable[[AgentState], bool] = field(default=lambda s: True)
    description: str = ""


@dataclass
class TraceSnapshotAssertion(EvalAssertion):
    """Fails if the tool call sequence diverges from the golden trace file."""

    golden_trace_path: str = ""
    normalize_fields: list[str] = field(
        default_factory=lambda: ["timestamp", "latency_ms", "run_id", "call_id", "cost_usd"]
    )


@dataclass
class EvalCase:
    id: str
    input: AgentState
    assertions: list[EvalAssertion]
    cassette: str | None = None
    tags: list[str] = field(default_factory=list)
    expected_status: Literal["completed", "failed"] = "completed"


@dataclass
class EvalCaseResult:
    case_id: str
    passed: bool
    assertion_results: list[tuple[EvalAssertion, bool, str]]
    trace: RunTrace
    duration_seconds: float
    live_calls_made: int


@dataclass
class EvalSuiteResult:
    suite_name: str
    total_cases: int
    passed_cases: int
    pass_rate: float
    llm_judge_score: float | None
    case_results: list[EvalCaseResult]
    gate_passed: bool
    duration_seconds: float
    total_cost_usd: float

    def assert_gates(self) -> None:
        """Raises EvalGateFailure if pass_rate_gate not met."""
        if not self.gate_passed:
            failed = [r for r in self.case_results if not r.passed]
            msgs = "\n".join(
                f"  - [{r.case_id}] "
                + "; ".join(msg for _, ok, msg in r.assertion_results if not ok)
                for r in failed
            )
            raise EvalGateFailure(
                f"Eval gate failed: {self.passed_cases}/{self.total_cases} cases passed "
                f"({self.pass_rate:.0%})\n{msgs}"
            )

    def summary(self) -> str:
        lines = [
            f"EvalSuite '{self.suite_name}': "
            f"{self.passed_cases}/{self.total_cases} passed "
            f"({self.pass_rate:.0%}) in {self.duration_seconds:.1f}s",
        ]
        for r in self.case_results:
            icon = "✓" if r.passed else "✗"
            lines.append(f"  {icon} [{r.case_id}] ({r.duration_seconds:.2f}s)")
            for _assertion, ok, msg in r.assertion_results:
                if not ok:
                    lines.append(f"      FAIL: {msg}")
        return "\n".join(lines)


# ── Custom Exceptions ─────────────────────────────────────────────────────────


class RampartError(Exception):
    """Base class for all Rampart exceptions."""


class BudgetExceededError(RampartError):
    def __init__(self, event: BudgetExceededEvent) -> None:
        self.event = event
        super().__init__(
            f"Budget exceeded: {event.exceeded_dimension} limit reached (run_id={event.run_id})"
        )


class PermissionDeniedError(RampartError):
    def __init__(self, event: PermissionViolationEvent) -> None:
        self.event = event
        super().__init__(f"Permission denied: {event.violation_type} — {event.attempted_action}")


class RampartCassetteStaleError(RampartError):
    pass


class EvalGateFailure(RampartError):
    pass


class GraphVersionConflict(RampartError):
    pass


class NoCheckpointError(RampartError):
    pass


class LLMNotConfiguredError(RampartError):
    pass
