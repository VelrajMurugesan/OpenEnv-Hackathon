"""Core environment engine — manages sessions, state, and step logic."""

from __future__ import annotations

import uuid
from typing import Any

from app.graders import grade_findings
from app.models import (
    ActionType,
    AgentAction,
    EnvState,
    GraderResult,
    GroundTruthIssue,
    Invoice,
    StepResponse,
    TaskInfo,
)
from app.tasks.easy import get_easy_task
from app.tasks.hard import get_hard_task
from app.tasks.medium import get_medium_task


ALL_TASK_IDS = [
    "easy_1", "easy_2", "easy_3",
    "medium_1", "medium_2", "medium_3",
    "hard_1", "hard_2", "hard_3",
]


def _get_task(task_id: str) -> tuple[TaskInfo, list[Invoice], list[GroundTruthIssue]]:
    """Load a task by ID."""
    if task_id.startswith("easy_"):
        return get_easy_task(task_id)
    elif task_id.startswith("medium_"):
        return get_medium_task(task_id)
    elif task_id.startswith("hard_"):
        return get_hard_task(task_id)
    else:
        raise ValueError(f"Unknown task ID: {task_id}")


class Session:
    """Represents an active environment session for one task."""

    def __init__(self, task_id: str) -> None:
        self.session_id = str(uuid.uuid4())
        self.task_info, self.invoices, self.ground_truth = _get_task(task_id)
        self.findings: list[dict[str, Any]] = []
        self.approved_invoices: set[str] = set()
        self.step_count = 0
        self.done = False
        self.score: float | None = None
        self.grader_result: GraderResult | None = None

    def get_state(self) -> EnvState:
        """Return the current environment state."""
        return EnvState(
            task_id=self.task_info.task_id,
            task_description=self.task_info.description,
            difficulty=self.task_info.difficulty.value,
            invoices=[inv.model_dump() for inv in self.invoices],
            findings=self.findings,
            step_count=self.step_count,
            max_steps=self.task_info.max_steps,
            done=self.done,
            score=self.score,
        )

    def step(self, action: AgentAction) -> StepResponse:
        """Process an agent action and return the updated state."""
        if self.done:
            return StepResponse(
                state=self.get_state(),
                reward=0.0,
                done=True,
                info={"message": "Session already finished. Call reset() to start a new task."},
            )

        self.step_count += 1
        reward = 0.0
        info: dict[str, Any] = {}

        if action.action == ActionType.FLAG_ISSUE:
            finding = {
                "invoice_id": action.invoice_id,
                "field": action.field,
                "category": action.category,
                "severity": action.severity,
                "description": action.description,
            }
            self.findings.append(finding)
            info["message"] = f"Issue flagged on invoice {action.invoice_id}"

            # Give intermediate reward hint: does this finding match any ground truth?
            from app.graders import _match_finding_to_issue
            matched = any(
                _match_finding_to_issue(finding, gt)
                for gt in self.ground_truth
            )
            reward = 0.05 if matched else -0.02
            info["hint"] = "Finding recorded"

        elif action.action == ActionType.APPROVE:
            self.approved_invoices.add(action.invoice_id)
            info["message"] = f"Invoice {action.invoice_id} approved"

            # Check if approved invoice actually has issues
            invoice_issues = [
                gt for gt in self.ground_truth
                if gt.invoice_id == action.invoice_id
            ]
            if invoice_issues:
                reward = -0.1  # Penalty for approving a bad invoice
                info["hint"] = "Invoice approved"
            else:
                reward = 0.05  # Reward for correctly approving a clean invoice
                info["hint"] = "Invoice approved"

        elif action.action == ActionType.SUBMIT_REPORT:
            # Final submission — grade and end session
            self.grader_result = grade_findings(
                self.findings, self.ground_truth, self.task_info.task_id
            )
            self.score = self.grader_result.score
            self.done = True
            reward = self.score
            info["grader_result"] = self.grader_result.model_dump()
            info["message"] = "Report submitted and graded"

        # Auto-end if max steps reached
        if self.step_count >= self.task_info.max_steps and not self.done:
            self.grader_result = grade_findings(
                self.findings, self.ground_truth, self.task_info.task_id
            )
            self.score = self.grader_result.score
            self.done = True
            reward = self.score
            info["grader_result"] = self.grader_result.model_dump()
            info["message"] = "Max steps reached — auto-graded"

        return StepResponse(
            state=self.get_state(),
            reward=reward,
            done=self.done,
            info=info,
        )


class EnvironmentManager:
    """Manages multiple sessions across tasks."""

    def __init__(self) -> None:
        self.sessions: dict[str, Session] = {}
        self.active_session: Session | None = None

    def reset(self, task_id: str) -> EnvState:
        """Reset the environment with a new task."""
        if task_id not in ALL_TASK_IDS:
            raise ValueError(
                f"Unknown task_id: {task_id}. Available: {ALL_TASK_IDS}"
            )
        session = Session(task_id)
        self.sessions[session.session_id] = session
        self.active_session = session
        return session.get_state()

    def state(self) -> EnvState:
        """Return the current state."""
        if not self.active_session:
            raise RuntimeError("No active session. Call reset() first.")
        return self.active_session.get_state()

    def step(self, action: AgentAction) -> StepResponse:
        """Process an action in the current session."""
        if not self.active_session:
            raise RuntimeError("No active session. Call reset() first.")
        return self.active_session.step(action)

    def get_task_list(self) -> list[TaskInfo]:
        """Return all available tasks."""
        tasks = []
        for task_id in ALL_TASK_IDS:
            info, _, _ = _get_task(task_id)
            tasks.append(info)
        return tasks

    def get_grader_result(self) -> GraderResult | None:
        """Return the grader result for the current session."""
        if self.active_session:
            return self.active_session.grader_result
        return None
