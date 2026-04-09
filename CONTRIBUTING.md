# Contributing to GST Invoice Compliance Checker

Thank you for considering a contribution! This repository hosts an
[OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that benchmarks
LLM/RL agents on Indian GST (Goods and Services Tax) invoice compliance
reasoning.

The most useful contributions are:

1. **New tasks** that exercise GST rules not yet covered (place-of-supply edge
   cases, special economic zone supplies, exports with LUT, etc.).
2. **More HSN/SAC codes and rate mappings** in `data/hsn_codes.py`.
3. **Stronger baseline agents** in `inference.py` (better LLM prompts, fewer
   false positives, fewer missed issues).
4. **Adversarial / precision-testing tasks** (see `hard_4` for the pattern).
5. **Multi-model benchmark results** — run `benchmark.py` against models you
   have access to and PR the updated leaderboard.

## Local development

```bash
# Install dependencies (uv is recommended)
uv sync

# Run the test suite
uv run python tests/test_env.py

# Start the env server locally
uv run uvicorn app.main:app --host 0.0.0.0 --port 7860

# Run baseline inference against the local server
ENV_URL=http://localhost:7860 \
  MODEL_NAME=gpt-4o-mini \
  OPENAI_API_KEY=sk-... \
  uv run python inference.py
```

## Adding a new task

1. Add task data to `app/tasks/easy.py`, `medium.py`, or `hard.py`. Each task
   has a `TaskInfo`, a list of `Invoice` objects, and ground truth either from
   `data.gst_rules.run_*_validation()` (auto) or hand-coded.
2. Register the task ID in `app/engine.py:ALL_TASK_IDS`.
3. Register the task ID in `inference.py:TASK_IDS`.
4. Add a row in `openenv.yaml:tasks`.
5. Update `tests/test_env.py:test_all_tasks_load` and
   `test_task_difficulties` if the totals change.
6. Run `uv run python tests/test_env.py` and verify all tests pass.
7. Run `uv run python benchmark.py --models programmatic` to confirm the new
   task scores cleanly in the (0, 1) interval.

## Hard rules

These exist because the OpenEnv hackathon Phase 2 validator enforces them and
will reject the submission otherwise:

- **Task scores must be strictly inside (0, 1)** — never 0.0 or 1.0. The
  grader clamps to `[0.0001, 0.9998]` in `app/graders.py`. Do not loosen this.
- **`inference.py` must emit `[START]`/`[STEP]`/`[END]` log lines on stdout**
  in the canonical space-separated `key=value` format. Diagnostic output goes
  to **stderr** via `log_diagnostic()`. Do not switch to JSON.
- **`/reset` must accept an empty body** (the validator pings it that way).
- **Inference must complete in under 20 minutes** on 2 vCPU / 8 GB RAM.

## Reporting issues

Open a GitHub issue with:
- Steps to reproduce
- Expected vs. actual behavior
- Environment (OS, Python version, `uv --version` output)

## License

By contributing you agree that your contributions will be licensed under the
[MIT License](LICENSE).
