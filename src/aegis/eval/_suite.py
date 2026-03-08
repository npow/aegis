"""EvalSuite, EvalCase, EvalSuiteResult, and EvalGateFailure."""

from __future__ import annotations

import time
from typing import Any

from .._models import (
    EvalCase,
    EvalCaseResult,
    EvalSuiteResult,
    RunConfig,
)
from ._assertions import evaluate_assertion


class EvalSuite:
    """Collection of EvalCases that gate deployment."""

    def __init__(
        self,
        name: str,
        graph: Any,  # GraphDef
        cases: list[EvalCase],
        *,
        pass_rate_gate: float = 1.0,
        llm_judge_gate: float = 0.85,
        llm_judge_model: str | None = None,
    ) -> None:
        self.name = name
        self.graph = graph
        self.cases = cases
        self.pass_rate_gate = pass_rate_gate
        self.llm_judge_gate = llm_judge_gate
        self.llm_judge_model = llm_judge_model

    async def run(self, *, base_thread_id: str = "eval") -> EvalSuiteResult:
        """Run all eval cases and return a result summary."""
        suite_start = time.monotonic()
        case_results: list[EvalCaseResult] = []
        total_cost = 0.0

        for case in self.cases:
            result = await self._run_case(case, base_thread_id=base_thread_id)
            case_results.append(result)
            total_cost += result.trace.total_cost_usd if result.trace is not None else 0.0

        passed = sum(1 for r in case_results if r.passed)
        total = len(case_results)
        pass_rate = passed / total if total > 0 else 0.0
        gate_passed = pass_rate >= self.pass_rate_gate

        # LLM-as-judge scoring (advisory; also gates if configured)
        judge_score = await self._compute_llm_judge_score(case_results)
        if judge_score is not None and judge_score < self.llm_judge_gate:
            gate_passed = False

        return EvalSuiteResult(
            suite_name=self.name,
            total_cases=total,
            passed_cases=passed,
            pass_rate=pass_rate,
            llm_judge_score=judge_score,
            case_results=case_results,
            gate_passed=gate_passed,
            duration_seconds=time.monotonic() - suite_start,
            total_cost_usd=total_cost,
        )

    async def _run_case(self, case: EvalCase, base_thread_id: str) -> EvalCaseResult:
        thread_id = f"{base_thread_id}-{case.id}"
        config = RunConfig(thread_id=thread_id)
        case_start = time.monotonic()
        live_calls = 0

        # Set up cassette replay if provided
        try:
            if case.cassette:
                from ..testing._cassette import cassette as _cassette

                async with _cassette.replay(case.cassette) as replay_ctx:
                    run_result = await self.graph.run(input=case.input, config=config)
                    live_calls = replay_ctx.live_calls_made
            else:
                run_result = await self.graph.run(input=case.input, config=config)
                # When no cassette, every non-mocked tool call is a live call
                live_calls = sum(
                    1
                    for n in run_result.trace.nodes_executed
                    for tc in n.tool_calls
                    if not tc.was_mocked
                )
        except Exception as exc:
            # Graph raised an exception rather than returning a failed RunResult.
            # Record it as a failed case so the suite continues.
            return EvalCaseResult(
                case_id=case.id,
                passed=False,
                assertion_results=[(None, False, f"Graph raised {type(exc).__name__}: {exc}")],  # type: ignore[list-item]
                trace=None,  # type: ignore[arg-type]
                duration_seconds=time.monotonic() - case_start,
                live_calls_made=0,
            )

        # Status assertion
        assertion_results = []
        if run_result.status != case.expected_status:
            assertion_results.append(
                (
                    None,
                    False,
                    f"Expected status '{case.expected_status}', got '{run_result.status}'",
                )
            )

        # Run all assertions
        for assertion in case.assertions:
            ok, msg = evaluate_assertion(assertion, run_result.state, run_result.trace)
            assertion_results.append((assertion, ok, msg))  # type: ignore[arg-type]

        overall_passed = all(ok for _, ok, _ in assertion_results)

        return EvalCaseResult(
            case_id=case.id,
            passed=overall_passed,
            assertion_results=assertion_results,  # type: ignore[arg-type]
            trace=run_result.trace,
            duration_seconds=time.monotonic() - case_start,
            live_calls_made=live_calls,
        )

    async def _compute_llm_judge_score(self, case_results: list[EvalCaseResult]) -> float | None:
        """Call an LLM to holistically score the agent's behaviour (0.0–1.0).

        Returns ``None`` when ``llm_judge_model`` is not configured, litellm is
        not installed, or the LLM response cannot be parsed as a float.
        """
        if self.llm_judge_model is None:
            return None

        try:
            import litellm  # type: ignore[import]
        except ImportError:
            return None

        # Build a concise summary for the judge
        lines = [f"Eval suite: {self.name}", f"Total cases: {len(case_results)}", ""]
        for r in case_results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"[{r.case_id}] {status}")
            for _, ok, msg in r.assertion_results:
                if not ok:
                    lines.append(f"  - FAILED: {msg}")

        summary = "\n".join(lines)
        prompt = (
            "You are evaluating an AI agent's test results. "
            "Score the overall quality of the agent's behaviour from 0.0 (terrible) "
            "to 1.0 (excellent). Consider correctness, reliability, and efficiency.\n\n"
            f"{summary}\n\n"
            "Reply with ONLY a decimal number between 0.0 and 1.0. No explanation."
        )

        try:
            response = await litellm.acompletion(
                model=self.llm_judge_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            score = float(raw)
            return max(0.0, min(1.0, score))
        except Exception:
            return None
