"""Multi-model benchmark runner for the GST Invoice Compliance Checker.

Runs `inference.py` once per configured model against the live (or local)
OpenEnv environment, parses the canonical `[END] task=X score=Y.YYYY steps=N`
log lines, and emits a markdown leaderboard table that can be pasted into the
README.

Usage
-----

    # Benchmark against the live HF Space (default)
    uv run python benchmark.py

    # Benchmark against a local server
    ENV_URL=http://localhost:7860 uv run python benchmark.py

    # Restrict to specific models
    uv run python benchmark.py --models gpt-4o-mini,gpt-4o

Environment variables
---------------------

    ENV_URL          Base URL of the running OpenEnv environment.
                     Defaults to the live HF Space deployment.
    API_BASE_URL     LLM endpoint (default: https://api.openai.com/v1).
    OPENAI_API_KEY   Used as the LLM credential. If absent, the LLM review
                     pass inside inference.py fails open and the score
                     reflects the programmatic-only baseline (which is
                     itself a meaningful row in the leaderboard).

Output
------

    benchmark_results.md    Markdown table ready to paste into the README.
    benchmark_results.json  Raw per-task scores for downstream tooling.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_URL = os.environ.get(
    "ENV_URL",
    "https://velrajmurugesan-gst-invoice-compliance-checker.hf.space",
)

# Models to benchmark by default. The "programmatic" pseudo-model uses no LLM
# at all and represents the deterministic-rules-only baseline. Other rows
# require valid API credentials for the corresponding provider.
DEFAULT_MODELS = [
    {"id": "programmatic", "label": "Programmatic Only (no LLM)", "model_name": "none"},
    {"id": "gpt-4o-mini", "label": "GPT-4o-mini (hybrid)", "model_name": "gpt-4o-mini"},
    {"id": "gpt-4o", "label": "GPT-4o (hybrid)", "model_name": "gpt-4o"},
]

END_LINE_RE = re.compile(
    r"^\[END\]\s+task=(\S+)\s+score=([0-9.]+)\s+steps=(\d+)",
    re.MULTILINE,
)


@dataclass
class ModelResult:
    """Aggregated benchmark result for a single model configuration."""
    model_id: str
    label: str
    per_task: dict[str, float] = field(default_factory=dict)
    durations: dict[str, float] = field(default_factory=dict)
    error: str = ""

    @property
    def average(self) -> float:
        if not self.per_task:
            return 0.0
        return sum(self.per_task.values()) / len(self.per_task)


def run_inference_for_model(model_cfg: dict, env_url: str) -> ModelResult:
    """Run inference.py as a subprocess with the given model config and parse
    the resulting [END] lines into a ModelResult."""
    result = ModelResult(
        model_id=model_cfg["id"],
        label=model_cfg["label"],
    )

    env = os.environ.copy()
    env["ENV_URL"] = env_url
    env["MODEL_NAME"] = model_cfg["model_name"]
    env["PYTHONIOENCODING"] = "utf-8"
    if model_cfg["id"] == "programmatic":
        # Force the LLM review pass inside inference.py to fail open by
        # giving it a guaranteed-bad endpoint and key. The deterministic
        # programmatic_audit() pass still runs to completion.
        env["API_BASE_URL"] = "http://127.0.0.1:1"
        env["OPENAI_API_KEY"] = "DISABLED"
        env["HF_TOKEN"] = "DISABLED"

    print(f"\n=== {model_cfg['label']} ===", flush=True)
    print(f"  ENV_URL    = {env_url}", flush=True)
    print(f"  MODEL_NAME = {model_cfg['model_name']}", flush=True)

    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "inference.py")],
            env=env,
            capture_output=True,
            text=True,
            timeout=1200,  # 20 min hard cap
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        result.error = "timeout (>20m)"
        return result
    except Exception as exc:
        result.error = f"subprocess error: {exc}"
        return result

    elapsed = time.time() - start

    if proc.returncode != 0:
        result.error = f"exit={proc.returncode}: {proc.stderr.strip()[-200:]}"
        return result

    # Parse [END] lines from stdout
    matches = END_LINE_RE.findall(proc.stdout)
    if not matches:
        result.error = "no [END] lines found in stdout"
        return result

    for task_id, score_str, _steps in matches:
        try:
            result.per_task[task_id] = float(score_str)
        except ValueError:
            continue

    print(f"  Tasks parsed: {len(result.per_task)}/{len(matches)}", flush=True)
    print(f"  Average:      {result.average:.4f}", flush=True)
    print(f"  Elapsed:      {elapsed:.1f}s", flush=True)
    return result


def render_markdown_table(results: list[ModelResult]) -> str:
    """Render the leaderboard as a markdown table sorted by average score."""
    if not results:
        return "_(no benchmark results)_"

    # Collect the union of task IDs across all results, in canonical order.
    canonical = [
        "easy_1", "easy_2", "easy_3",
        "medium_1", "medium_2", "medium_3",
        "hard_1", "hard_2", "hard_3", "hard_4",
    ]
    seen = set()
    for r in results:
        seen.update(r.per_task.keys())
    task_order = [t for t in canonical if t in seen] + sorted(seen - set(canonical))

    sorted_results = sorted(results, key=lambda r: r.average, reverse=True)

    header = "| Model | Avg | " + " | ".join(task_order) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(task_order)) + "|"
    rows: list[str] = [header, sep]
    for r in sorted_results:
        if r.error:
            cells = ["—"] * len(task_order)
            row = f"| **{r.label}** | _{r.error}_ | " + " | ".join(cells) + " |"
        else:
            cells = [
                f"{r.per_task[t]:.4f}" if t in r.per_task else "—"
                for t in task_order
            ]
            row = f"| **{r.label}** | **{r.average:.4f}** | " + " | ".join(cells) + " |"
        rows.append(row)
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-url",
        default=DEFAULT_ENV_URL,
        help=f"OpenEnv server URL (default: {DEFAULT_ENV_URL})",
    )
    parser.add_argument(
        "--models",
        default="",
        help=(
            "Comma-separated list of model ids to benchmark "
            "(default: programmatic,gpt-4o-mini,gpt-4o)"
        ),
    )
    parser.add_argument(
        "--out-md",
        default="benchmark_results.md",
        help="Output markdown table path",
    )
    parser.add_argument(
        "--out-json",
        default="benchmark_results.json",
        help="Output raw JSON path",
    )
    args = parser.parse_args()

    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        models = [m for m in DEFAULT_MODELS if m["id"] in wanted]
        if not models:
            print(f"No matching models in {[m['id'] for m in DEFAULT_MODELS]}")
            return 2
    else:
        models = DEFAULT_MODELS

    print("=" * 60)
    print("GST Invoice Compliance Checker — Multi-Model Benchmark")
    print(f"Environment: {args.env_url}")
    print(f"Models:      {[m['id'] for m in models]}")
    print("=" * 60)

    results: list[ModelResult] = []
    for cfg in models:
        results.append(run_inference_for_model(cfg, args.env_url))

    md = render_markdown_table(results)
    Path(args.out_md).write_text(md + "\n", encoding="utf-8")
    print("\n" + "=" * 60)
    print("LEADERBOARD")
    print("=" * 60)
    print(md)

    raw = {
        "env_url": args.env_url,
        "results": [
            {
                "model_id": r.model_id,
                "label": r.label,
                "average": r.average,
                "per_task": r.per_task,
                "error": r.error,
            }
            for r in results
        ],
    }
    Path(args.out_json).write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out_md} and {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
