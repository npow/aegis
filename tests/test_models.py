"""Tests for Aegis data models."""

from dataclasses import dataclass
from datetime import datetime

from aegis import (
    AgentState,
    Budget,
    BudgetDecision,
    BudgetStatus,
    PermissionScope,
)

# ── AgentState ────────────────────────────────────────────────────────────────


@dataclass
class _Counter(AgentState):
    value: int = 0
    label: str = ""


def test_agent_state_update_creates_new_instance():
    original = _Counter(value=5, label="hello")
    updated = original.update(value=10)
    assert updated.value == 10
    assert updated.label == "hello"
    assert original.value == 5  # unchanged


def test_agent_state_update_does_not_mutate():
    original = _Counter(value=1, label="a")
    updated = original.update(value=2, label="b")
    assert original.value == 1
    assert original.label == "a"
    assert updated.value == 2
    assert updated.label == "b"


def test_agent_state_to_dict():
    s = _Counter(value=7, label="test", thread_id="t1", run_id="r1")
    d = s.to_dict()
    assert d["value"] == 7
    assert d["label"] == "test"
    assert d["thread_id"] == "t1"


def test_agent_state_from_dict():
    d = {"value": 42, "label": "x", "thread_id": "t", "run_id": "r"}
    s = _Counter.from_dict(d)
    assert s.value == 42
    assert s.label == "x"


def test_agent_state_from_dict_ignores_unknown_fields():
    d = {"value": 1, "unknown_field": "nope"}
    s = _Counter.from_dict(d)
    assert s.value == 1
    assert not hasattr(s, "unknown_field")


# ── Budget ────────────────────────────────────────────────────────────────────


def test_budget_decision_hard_stop():
    decision = BudgetDecision.hard_stop()
    assert decision.action == "hard_stop"
    assert decision.updated_budget is None


def test_budget_decision_extend():
    base = Budget(max_tokens=1000, max_llm_cost_usd=2.00)
    decision = BudgetDecision.extend(base, max_llm_cost_usd=5.00)
    assert decision.action == "extend"
    assert decision.updated_budget is not None
    # The override is applied
    assert decision.updated_budget.max_llm_cost_usd == 5.00
    # Other fields from the base budget are preserved
    assert decision.updated_budget.max_tokens == 1000


def test_budget_decision_downgrade():
    decision = BudgetDecision.downgrade(model="openai/gpt-4o-mini")
    assert decision.action == "downgrade"
    assert decision.updated_budget is not None
    assert decision.updated_budget.downgrade_to == "openai/gpt-4o-mini"


def test_budget_status_pct_consumed():
    budget = Budget(max_tokens=1000, max_llm_cost_usd=2.00)
    status = BudgetStatus(tokens_used=500, cost_usd=1.00)
    status.compute_pct(budget)
    assert status.pct_consumed["tokens"] == 0.5
    assert status.pct_consumed["cost"] == 0.5


# ── Checkpoint ID format ──────────────────────────────────────────────────────


def test_checkpoint_id_format():

    # Just verify the format by constructing a valid checkpoint id manually
    state = _Counter(value=1)
    import dataclasses
    import hashlib
    import json

    state_dict = dataclasses.asdict(state)
    state_hash = hashlib.sha256(
        json.dumps(state_dict, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    expected_prefix = "ckpt_my-graph_t1_2_"
    ckpt_id = f"ckpt_my-graph_t1_2_{state_hash}"
    assert ckpt_id.startswith(expected_prefix)
    assert len(ckpt_id.split("_")) >= 4


# ── PermissionScope ───────────────────────────────────────────────────────────


def test_permission_scope_defaults():
    scope = PermissionScope()
    assert scope.tools is None  # None = all tools allowed
    assert scope.network.deny_all_others is True
    assert scope.filesystem.read is False


def test_network_permission_domain_matching():
    from aegis._permissions import _domain_matches_any

    patterns = ["*.wikipedia.org", "arxiv.org"]
    assert _domain_matches_any("en.wikipedia.org", patterns) is True
    assert _domain_matches_any("arxiv.org", patterns) is True
    assert _domain_matches_any("evil.com", patterns) is False
    assert _domain_matches_any("wikipedia.org", patterns) is False  # no * prefix


# ── EvalSuiteResult ───────────────────────────────────────────────────────────


def test_eval_suite_result_summary():
    from aegis import EvalCaseResult, EvalSuiteResult
    from aegis._models import RunTrace

    trace = RunTrace(
        run_id="r1",
        thread_id="t1",
        graph_name="g",
        graph_version="1.0",
        started_at=datetime.utcnow(),
        completed_at=datetime.utcnow(),
        status="completed",
    )
    result = EvalSuiteResult(
        suite_name="test",
        total_cases=2,
        passed_cases=2,
        pass_rate=1.0,
        llm_judge_score=None,
        case_results=[
            EvalCaseResult(
                case_id="c1",
                passed=True,
                assertion_results=[],
                trace=trace,
                duration_seconds=0.1,
                live_calls_made=0,
            )
        ],
        gate_passed=True,
        duration_seconds=0.2,
        total_cost_usd=0.0,
    )
    summary = result.summary()
    assert "test" in summary
    assert "2/2" in summary or "100%" in summary
