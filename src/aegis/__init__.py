"""Aegis — A runtime that makes LLM agents production-safe by default."""

from __future__ import annotations

# Install HTTP intercept at import time (no-op if httpx/requests not installed)
from ._http_intercept import install as _install_intercept

_install_intercept()

# ── Public API ────────────────────────────────────────────────────────────────

# Configuration helper (global defaults)
from ._config import OTelTracer, PostgresCheckpointer, configure  # noqa: E402
from ._context import (  # noqa: E402
    GraphContext,
    LLMContext,
    LLMResponse,
    ToolContext,
)
from ._decorators import (  # noqa: E402
    GraphDef,
    NodeDef,
    ToolDef,
    get_graph_registry,
    get_tool_registry,
    graph,
    node,
    tool,
)
from ._models import (  # noqa: E402
    AegisCassetteStaleError,
    # Exceptions
    AegisError,
    # State
    AgentState,
    ApprovalPolicy,
    # Budget
    Budget,
    BudgetDecision,
    BudgetExceededError,
    BudgetExceededEvent,
    BudgetStatus,
    CassetteEntry,
    # Testing models
    CassetteRecord,
    CassetteReplayContext,
    # Checkpointing
    Checkpoint,
    CheckpointBackendConfig,
    # Eval models
    EvalAssertion,
    EvalCase,
    EvalCaseResult,
    EvalGateFailure,
    EvalSuiteResult,
    FilesystemPermission,
    GraphVersionConflict,
    # Tracing
    LLMCall,
    LLMNotConfiguredError,
    MockCallRecord,
    MockContext,
    # Permissions
    NetworkPermission,
    NoCheckpointError,
    NodeTrace,
    PermissionDeniedError,
    PermissionScope,
    PermissionViolationEvent,
    RunConfig,
    RunError,
    RunResult,
    RunTrace,
    SchemaAssertion,
    ToolCall,
    ToolCallAssertion,
    TraceSnapshotAssertion,
)
from ._multi_agent import chain, parallel, supervisor  # noqa: E402
from .checkpointers import (  # noqa: E402
    CheckpointerBase,
    MemoryCheckpointer,
    SqliteCheckpointer,
)

__all__ = [
    # State
    "AgentState",
    "RunConfig",
    "RunResult",
    "RunError",
    # Decorators
    "graph",
    "node",
    "tool",
    "GraphDef",
    "NodeDef",
    "ToolDef",
    "get_tool_registry",
    "get_graph_registry",
    # Context
    "ToolContext",
    "LLMContext",
    "GraphContext",
    "LLMResponse",
    # Checkpointing
    "Checkpoint",
    "CheckpointBackendConfig",
    "CheckpointerBase",
    "MemoryCheckpointer",
    "SqliteCheckpointer",
    # Tracing
    "LLMCall",
    "ToolCall",
    "NodeTrace",
    "RunTrace",
    # Budget
    "Budget",
    "BudgetStatus",
    "BudgetExceededEvent",
    "BudgetDecision",
    # Permissions
    "NetworkPermission",
    "FilesystemPermission",
    "ApprovalPolicy",
    "PermissionScope",
    "PermissionViolationEvent",
    # Testing
    "CassetteRecord",
    "CassetteEntry",
    "MockCallRecord",
    "MockContext",
    "CassetteReplayContext",
    # Eval
    "EvalAssertion",
    "ToolCallAssertion",
    "SchemaAssertion",
    "TraceSnapshotAssertion",
    "EvalCase",
    "EvalCaseResult",
    "EvalSuiteResult",
    # Multi-agent
    "chain",
    "parallel",
    "supervisor",
    # Configuration
    "configure",
    "PostgresCheckpointer",
    "OTelTracer",
    # Exceptions
    "AegisError",
    "BudgetExceededError",
    "PermissionDeniedError",
    "AegisCassetteStaleError",
    "EvalGateFailure",
    "GraphVersionConflict",
    "NoCheckpointError",
    "LLMNotConfiguredError",
]
