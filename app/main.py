"""FastAPI application — OpenEnv API endpoints for GST Invoice Compliance Checker."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.engine import EnvironmentManager
from app.models import (
    AgentAction,
    EnvironmentInfo,
    EnvState,
    GraderResult,
    ResetRequest,
    StepRequest,
    StepResponse,
    TaskInfo,
)

app = FastAPI(
    title="GST Invoice Compliance Checker — OpenEnv",
    description=(
        "An OpenEnv environment where AI agents audit GST invoices for "
        "compliance violations. 9 tasks across easy/medium/hard difficulty."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = EnvironmentManager()


@app.get("/")
def root() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "environment": "GST Invoice Compliance Checker",
        "version": "1.0.0",
    }


@app.get("/info", response_model=EnvironmentInfo)
def get_info() -> EnvironmentInfo:
    """Return environment metadata and available tasks."""
    return EnvironmentInfo(tasks=env.get_task_list())


@app.get("/tasks", response_model=list[TaskInfo])
def list_tasks() -> list[TaskInfo]:
    """List all available tasks with their metadata."""
    return env.get_task_list()


@app.post("/reset", response_model=EnvState)
def reset(request: ResetRequest) -> EnvState:
    """Reset the environment with a specific task.

    This initializes a new session with the given task's invoices.
    The agent should then read the state and begin auditing.
    """
    try:
        return env.reset(request.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=EnvState)
def get_state() -> EnvState:
    """Return the current environment state.

    Includes task description, invoices to audit, findings so far,
    step count, and whether the session is complete.
    """
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    """Submit an agent action and receive the updated state.

    Action types:
    - flag_issue: Flag a compliance issue on a specific invoice
    - approve: Mark an invoice as compliant
    - submit_report: Finalize the audit and receive a score

    Returns the new state, a reward signal, and whether the session is done.
    """
    try:
        return env.step(request.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/grade", response_model=GraderResult | None)
def get_grade() -> GraderResult | None:
    """Return the grader result for the current session (after submit_report)."""
    return env.get_grader_result()
