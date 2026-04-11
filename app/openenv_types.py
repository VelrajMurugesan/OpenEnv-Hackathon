"""OpenEnv-compatible Action and Observation types for the GST Compliance Checker.

These subclass the framework's base types so that `create_app()` can
auto-generate `/schema`, JSON-RPC for `/mcp`, and proper OpenAPI docs.
"""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class GSTAction(Action):
    """Agent action in the GST compliance audit environment.

    action types:
      - flag_issue:    report a compliance violation on a specific invoice
      - approve:       mark an invoice as compliant
      - submit_report: finalize the audit and receive a score
    """

    action: str = Field(
        default="submit_report",
        description="One of: flag_issue, approve, submit_report",
    )
    invoice_id: str = Field(default="", description="Target invoice ID")
    field: str = Field(default="", description="Invoice field with the issue")
    category: str = Field(
        default="",
        description="Issue category: missing_field, invalid_format, wrong_value, "
        "tax_mismatch, compliance_violation, inconsistency, duplicate",
    )
    severity: str = Field(
        default="",
        description="Issue severity: critical, major, minor",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the issue",
    )


class GSTObservation(Observation):
    """Observation returned after reset() or step().

    Inherits `done`, `reward`, and `metadata` from the framework base class.
    """

    task_id: str = Field(default="", description="Current task ID")
    task_description: str = Field(default="", description="Task instructions")
    difficulty: str = Field(default="", description="easy, medium, or hard")
    invoices: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Invoices to audit (JSON dicts)",
    )
    findings: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Issues flagged so far",
    )
    step_count: int = Field(default=0, description="Steps taken this episode")
    max_steps: int = Field(default=0, description="Maximum steps for this task")
    score: float | None = Field(
        default=None,
        description="F1 score (set after submit_report)",
    )


class GSTState(State):
    """Minimal state exposed via GET /state.

    Inherits `episode_id` and `step_count` from the framework base class.
    """

    task_id: str = ""
    done: bool = False
    score: float | None = None
