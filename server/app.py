"""Server entry point using openenv-core's create_app().

This replaces the hand-rolled FastAPI server with the framework's standard
app factory, which auto-registers /health, /metadata, /schema, /mcp, /ws,
/docs, and /openapi.json in addition to /reset, /step, /state.

Custom convenience endpoints (/tasks, /info, /grade) are added on top.
"""

from __future__ import annotations

import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from openenv.core.env_server.http_server import create_app

from app.environment import GSTComplianceEnvironment
from app.openenv_types import GSTAction, GSTObservation

# ── Create the standard OpenEnv app ─────────────────────────────────────────
#
# The GST audit is multi-step (reset → flag_issue × N → approve × M →
# submit_report). The framework's HTTP server is stateless-per-request by
# default (each request creates a fresh Environment via the factory). To
# maintain state across HTTP calls, we pass a factory that always returns
# the SAME singleton instance. The singleton's .close() is the default
# no-op, so lifecycle management by the server is safe.
#
# For concurrent-session support (WebSocket), the framework manages its own
# instance pool, but HTTP traffic shares this single instance.

_shared_env = GSTComplianceEnvironment()

app = create_app(
    lambda: _shared_env,
    GSTAction,
    GSTObservation,
    env_name="gst-invoice-compliance-checker",
)

# CORS — required for browser-based tools and the Colab notebook
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Custom convenience endpoints (not part of the OpenEnv spec) ─────────────

@app.get("/")
def root() -> dict:
    """Legacy health check (kept for backward compat with HF Spaces probes)."""
    return {
        "status": "ok",
        "environment": "GST Invoice Compliance Checker",
        "version": "1.0.0",
    }


@app.get("/tasks")
def list_tasks() -> list[dict]:
    """List all available tasks with metadata."""
    from app.engine import EnvironmentManager

    mgr = EnvironmentManager()
    return [t.model_dump() for t in mgr.get_task_list()]


@app.get("/info")
def get_info() -> dict:
    """Return environment metadata and task list."""
    from app.engine import EnvironmentManager

    mgr = EnvironmentManager()
    tasks = [t.model_dump() for t in mgr.get_task_list()]
    return {
        "name": "GST Invoice Compliance Checker",
        "version": "1.0.0",
        "description": (
            "An OpenEnv environment where an AI agent audits GST invoices "
            "for compliance violations. Tasks range from basic field validation "
            "to complex multi-invoice audits with reverse charge and ITC rules."
        ),
        "tasks": tasks,
    }


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    """Run the OpenEnv environment server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
