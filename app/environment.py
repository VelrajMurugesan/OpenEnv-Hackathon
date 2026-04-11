"""OpenEnv Environment subclass for the GST Invoice Compliance Checker.

This wraps the existing Session/EnvironmentManager logic from app/engine.py
into the interface that openenv-core's `create_app()` expects. All scoring,
grading, and task logic remain unchanged — only the HTTP surface changes.
"""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.interfaces import Environment

from app.engine import ALL_TASK_IDS, Session
from app.models import ActionType, AgentAction
from app.openenv_types import GSTAction, GSTObservation, GSTState


class GSTComplianceEnvironment(Environment[GSTAction, GSTObservation, GSTState]):
    """OpenEnv-compatible environment for Indian GST invoice compliance audit.

    Each episode corresponds to one task (e.g. ``easy_1``, ``hard_4``).
    The agent receives invoices, flags compliance issues, approves clean
    invoices, and submits a final report. The grader returns a severity-
    weighted F1 score clamped strictly into (0.0001, 0.9998).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._session: Session | None = None
        self._state = GSTState()

    # ── Required abstract methods ────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str = "easy_1",
        **kwargs: Any,
    ) -> GSTObservation:
        """Start a new audit episode for the given task.

        Parameters
        ----------
        task_id : str
            One of the 10 task IDs (easy_1 … hard_4).  Passed through from
            the JSON body of ``POST /reset`` via the framework's
            ``extra="allow"`` handling on ``ResetRequest``.
        """
        if task_id not in ALL_TASK_IDS:
            raise ValueError(
                f"Unknown task_id: {task_id}. Available: {ALL_TASK_IDS}"
            )

        self._session = Session(task_id)
        self._state = GSTState(
            episode_id=episode_id or self._session.session_id,
            step_count=0,
            task_id=task_id,
            done=False,
            score=None,
        )

        return self._make_observation(reward=0.0)

    def step(
        self,
        action: GSTAction,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> GSTObservation:
        """Process one agent action and return the updated observation."""
        if self._session is None:
            raise RuntimeError("No active session. Call reset() first.")

        agent_action = AgentAction(
            action=ActionType(action.action),
            invoice_id=action.invoice_id,
            field=action.field,
            category=action.category,
            severity=action.severity,
            description=action.description,
        )

        step_resp = self._session.step(agent_action)

        self._state.step_count = self._session.step_count
        self._state.done = self._session.done
        self._state.score = self._session.score

        return self._make_observation(
            reward=step_resp.reward,
            info=step_resp.info,
        )

    @property
    def state(self) -> GSTState:
        """Return lightweight state (used by ``GET /state``)."""
        return self._state

    # ── Helpers ──────────────────────────────────────────────────────────

    def _make_observation(
        self,
        reward: float = 0.0,
        info: dict[str, Any] | None = None,
    ) -> GSTObservation:
        if self._session is None:
            return GSTObservation(done=True, reward=0.0)

        env_state = self._session.get_state()
        return GSTObservation(
            done=env_state.done,
            reward=reward,
            task_id=env_state.task_id,
            task_description=env_state.task_description,
            difficulty=env_state.difficulty,
            invoices=env_state.invoices,
            findings=env_state.findings,
            step_count=env_state.step_count,
            max_steps=env_state.max_steps,
            score=env_state.score,
            metadata=info or {},
        )
