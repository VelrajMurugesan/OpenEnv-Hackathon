"""Build notebooks/training.ipynb programmatically.

We assemble the notebook with `nbformat` so that we never have to hand-escape
JSON for Python source cells. Run once after editing this file:

    uv run python scripts/build_training_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "training.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


cells: list[nbf.NotebookNode] = []

# ─── Title + intro ────────────────────────────────────────────────────────
cells.append(md(r"""
# Training agents against the GST Invoice Compliance Checker

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VelrajMurugesan/OpenEnv-Hackathon/blob/master/notebooks/training.ipynb)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Space-blue)](https://huggingface.co/spaces/VelrajMurugesan/gst-invoice-compliance-checker)

This notebook demonstrates two complementary recipes for training an LLM agent against the [GST Invoice Compliance Checker](https://huggingface.co/spaces/VelrajMurugesan/gst-invoice-compliance-checker) OpenEnv environment.

## Recipes

1. **Supervised distillation** — use the deterministic GST rules engine in `data/gst_rules.py` as a teacher and fine-tune **Qwen2.5-0.5B-Instruct** with **LoRA** to imitate it. This trains in well under an hour on a single Colab T4 and turns a tiny model into a competent GST auditor.
2. **Online RL with GRPO** — use the env's reward signal directly as the training target via TRL's `GRPOTrainer`. The reward function is just a Python wrapper around the live HF Space `/step` endpoint. This is the production recipe and is recommended on A100-or-larger hardware.

## Why both?

SFT gets you to ~0.99 fast and cheap by distilling explicit rules. GRPO is for the next mile: teaching the agent the *long tail* of issues the deterministic rules don't yet encode — phrasing variations, contextual fraud signals, and the kind of judgment a senior CA brings to a borderline invoice. The two are complementary, not alternatives.

## Hardware

| Recipe | Min. GPU | Time | Notes |
|---|---|---|---|
| SFT (Qwen 0.5B + LoRA, 3 epochs) | T4 16 GB | ~15 min | Free Colab is fine |
| Inference + eval | T4 16 GB | ~5 min | Free Colab is fine |
| GRPO (online RL) | A100 40 GB+ | hours | Pro/Pro+ Colab or own GPU |
"""))

# ─── Setup ───────────────────────────────────────────────────────────────
cells.append(md(r"""
## 1. Install dependencies

We pin to recent versions of `trl`, `peft`, and `transformers` because the GRPO API in TRL is still moving fast.
"""))

cells.append(code(r"""
!pip install -q "transformers>=4.45" "peft>=0.13" "trl>=0.12" \
    "datasets>=3.0" "accelerate>=0.34" "bitsandbytes>=0.43" \
    "pydantic>=2.0" httpx
"""))

cells.append(code(r"""
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:        ", torch.cuda.get_device_name(0))
    print(f"VRAM:           {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("BF16 supported:", torch.cuda.is_bf16_supported())
"""))

# ─── Part 1: Connect to the env ──────────────────────────────────────────
cells.append(md(r"""
## 2. Connect to the live OpenEnv environment

We talk to the running HF Space over plain HTTP. No SDK required.
"""))

cells.append(code(r"""
import json
import httpx

ENV_URL = "https://velrajmurugesan-gst-invoice-compliance-checker.hf.space"
client = httpx.Client(base_url=ENV_URL, timeout=120)

print("Health:", client.get("/").json())
print()

tasks = client.get("/tasks").json()
print(f"Found {len(tasks)} tasks:")
for t in tasks:
    print(f"  [{t['difficulty']:6s}] {t['task_id']:9s} -- {t['name']}")
"""))

# ─── Part 2: The reward shape ────────────────────────────────────────────
cells.append(md(r"""
## 3. Inspect the reward shape

The env returns a dense intermediate reward at every step plus a sparse final reward (the F1 score) when the agent calls `submit_report`. Both signals are usable as RL targets.

| Action | Reward |
|---|---|
| Correct `flag_issue` | +0.05 |
| False-positive `flag_issue` | -0.02 |
| `approve` a clean invoice | +0.05 |
| `approve` a dirty invoice | -0.10 |
| `submit_report` | severity-weighted F1, clamped to (0.0001, 0.9998) |

Let's see it for ourselves on `easy_1`.
"""))

cells.append(code(r"""
resp = client.post("/reset", json={"task_id": "easy_1"}).json()
state = resp.get("observation", resp)  # unwrap create_app() response
print(f"Reset to task: {state['task_id']}, invoices: {len(state['invoices'])}")
print()

# Action 1: a real issue
r1 = client.post("/step", json={"action": {
    "action": "flag_issue",
    "invoice_id": "INV-E1-001",
    "field": "invoice_number",
    "category": "missing_field",
    "severity": "critical",
    "description": "Invoice number is missing",
}}).json()
print(f"Real flag    -> reward {r1['reward']:+.4f}")

# Action 2: a fake issue
r2 = client.post("/step", json={"action": {
    "action": "flag_issue",
    "invoice_id": "INV-E1-001",
    "field": "supplier_gstin",
    "category": "invalid_format",
    "severity": "minor",
    "description": "(this is not really wrong)",
}}).json()
print(f"False flag   -> reward {r2['reward']:+.4f}")

# Action 3: submit report -- final F1 score
r3 = client.post("/step", json={"action": {"action": "submit_report"}}).json()
obs3 = r3.get("observation", r3.get("state", {}))
print(f"Submit       -> reward {r3['reward']:+.4f}  (final score = {obs3.get('score')})")
"""))

# ─── Part 3: SFT data generation ─────────────────────────────────────────
cells.append(md(r"""
## 4. Distill the rules engine into a training set

The GST rules engine in `data/gst_rules.py` already produces ground-truth findings for every task. We treat those findings as **expert demonstrations** and pair each invoice batch with its gold list of violations. This gives us a clean supervised distillation dataset for free, with no human annotation.

We clone the repo so we can import the rules engine directly.
"""))

cells.append(code(r"""
!git clone -q https://github.com/VelrajMurugesan/OpenEnv-Hackathon.git /content/OpenEnv-Hackathon

import sys
sys.path.insert(0, "/content/OpenEnv-Hackathon")

from app.tasks.easy import get_easy_task
from app.tasks.medium import get_medium_task
from app.tasks.hard import get_hard_task

ALL_TASK_IDS = [
    "easy_1", "easy_2", "easy_3",
    "medium_1", "medium_2", "medium_3",
    "hard_1", "hard_2", "hard_3", "hard_4",
]


def get_task(task_id):
    if task_id.startswith("easy"):
        return get_easy_task(task_id)
    if task_id.startswith("medium"):
        return get_medium_task(task_id)
    if task_id.startswith("hard"):
        return get_hard_task(task_id)
    raise ValueError(task_id)


SYSTEM_INSTRUCTION = (
    "You are a senior Indian GST compliance auditor. Given GST invoices in JSON, "
    "list every compliance violation as a JSON array. Each finding must have "
    "the keys invoice_id, field, category, severity, description. "
    "Categories: missing_field, invalid_format, wrong_value, tax_mismatch, "
    "compliance_violation, inconsistency, duplicate. "
    "Severities: critical, major, minor. "
    "Respond with ONLY the JSON array, no prose."
)


def make_sft_example(task_id):
    info, invoices, gt = get_task(task_id)
    invoices_json = json.dumps([inv.model_dump() for inv in invoices], indent=2)
    findings = [
        {
            "invoice_id": issue.invoice_id,
            "field": issue.field,
            "category": issue.category.value,
            "severity": issue.severity.value,
            "description": issue.description,
        }
        for issue in gt
    ]
    user_msg = f"INVOICES:\n{invoices_json}\n\nReturn the JSON array of findings."
    assistant_msg = json.dumps(findings, indent=2)
    return {
        "task_id": task_id,
        "system": SYSTEM_INSTRUCTION,
        "user": user_msg,
        "assistant": assistant_msg,
        "num_findings": len(findings),
    }


dataset = [make_sft_example(t) for t in ALL_TASK_IDS]
print(f"Generated {len(dataset)} training examples")
print(f"Total expert findings: {sum(d['num_findings'] for d in dataset)}")
print()
print("Findings per task:")
for d in dataset:
    print(f"  {d['task_id']:10s}  {d['num_findings']:3d} findings")
"""))

# ─── Part 4: Load Qwen2.5-0.5B + LoRA ────────────────────────────────────
cells.append(md(r"""
## 5. Load Qwen2.5-0.5B-Instruct with LoRA

Qwen2.5-0.5B is the smallest production-quality instruction model that fits comfortably on a free Colab T4 and still has enough capacity to learn structured JSON output. We attach a LoRA adapter so we only train ~1% of parameters.
"""))

cells.append(code(r"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
"""))

# ─── Part 5: Format dataset for SFTTrainer ───────────────────────────────
cells.append(md(r"""
## 6. Format the dataset for `SFTTrainer`

We render each (system, user, assistant) triple through the Qwen chat template and hand the resulting text strings to `SFTTrainer`.
"""))

cells.append(code(r"""
from datasets import Dataset


def to_chat_text(example):
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


hf_dataset = Dataset.from_list(dataset).map(to_chat_text, remove_columns=["task_id", "system", "user", "assistant", "num_findings"])
print(f"Dataset rows:  {len(hf_dataset)}")
print(f"Sample length: {len(hf_dataset[0]['text'])} characters")
print()
print(hf_dataset[0]["text"][:600], "...")
"""))

# ─── Part 6: SFT training ────────────────────────────────────────────────
cells.append(md(r"""
## 7. Run SFT (3 epochs, ~15 min on T4)

The 10-task dataset is small, so we train for 3 epochs with a generous learning rate. On a T4 this completes in well under an hour. On an A100 it's a few minutes.

If you only want to verify the recipe runs, set `num_train_epochs=1` for a faster pass.
"""))

cells.append(code(r"""
from trl import SFTConfig, SFTTrainer
from transformers import TrainingArguments

# TRL versions differ on parameter names. We detect which one to use.
import inspect
_sft_params = inspect.signature(SFTConfig).parameters

sft_kwargs = dict(
    output_dir="./gst-auditor-qwen-0.5b",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=1,
    save_strategy="epoch",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    report_to="none",
)

# max_seq_length vs max_length — name changed across TRL versions
if "max_seq_length" in _sft_params:
    sft_kwargs["max_seq_length"] = 4096
elif "max_length" in _sft_params:
    sft_kwargs["max_length"] = 4096

# dataset_text_field — removed in newer TRL versions
if "dataset_text_field" in _sft_params:
    sft_kwargs["dataset_text_field"] = "text"

sft_config = SFTConfig(**sft_kwargs)

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    args=sft_config,
)

trainer.train()
"""))

# ─── Part 7: Inference with the trained model ───────────────────────────
cells.append(md(r"""
## 8. Generate findings with the distilled auditor

We hand the trained model an unseen invoice batch and let it produce a JSON array of findings.
"""))

cells.append(code(r"""
def generate_findings(task_id, max_new_tokens=2048):
    info, invoices, _ = get_task(task_id)
    invoices_json = json.dumps([inv.model_dump() for inv in invoices], indent=2)
    user_msg = f"INVOICES:\n{invoices_json}\n\nReturn the JSON array of findings."

    # Build the prompt text, then tokenize separately (avoids BatchEncoding
    # compatibility issues across transformers versions).
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_msg},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response


print("Trained Qwen 0.5B output for hard_4 (the adversarial task):\n")
print(generate_findings("hard_4")[:1500])
"""))

# ─── Part 8: End-to-end eval ─────────────────────────────────────────────
cells.append(md(r"""
## 9. Evaluate end-to-end against the live env

We submit the model's findings to the live HF Space and read back the F1 score that the env grader assigns. This is exactly the same scoring path the OpenEnv hackathon validator uses.

If the trained model has learned the rules correctly, it should approach the **0.9911** average from the deterministic baseline. Anything closer than 0.97 is a strong result for a 0.5B parameter model.
"""))

cells.append(code(r"""
def parse_findings_from_response(response):
    start = response.find("[")
    end = response.rfind("]") + 1
    if start < 0 or end <= start:
        return []
    try:
        parsed = json.loads(response[start:end])
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def evaluate_task(task_id):
    client.post("/reset", json={"task_id": task_id})
    response = generate_findings(task_id)
    findings = parse_findings_from_response(response)

    for finding in findings:
        if not isinstance(finding, dict):
            continue
        client.post("/step", json={"action": {
            "action": "flag_issue",
            "invoice_id": finding.get("invoice_id", ""),
            "field": finding.get("field", ""),
            "category": finding.get("category", ""),
            "severity": finding.get("severity", "major"),
            "description": finding.get("description", ""),
        }})

    result = client.post("/step", json={"action": {"action": "submit_report"}}).json()
    obs = result.get("observation", result.get("state", {}))
    return obs.get("score", 0.0), len(findings)


print(f"{'Task':12s}  {'Score':>8s}  {'Flagged':>8s}")
print("-" * 32)
scores = []
for task_id in ALL_TASK_IDS:
    score, n_flagged = evaluate_task(task_id)
    scores.append(score)
    print(f"{task_id:12s}  {score:8.4f}  {n_flagged:8d}")

print("-" * 32)
print(f"{'Average':12s}  {sum(scores)/len(scores):8.4f}")
"""))

# ─── Part 9: GRPO recipe ─────────────────────────────────────────────────
cells.append(md(r"""
## 10. Online RL via GRPO (advanced)

GRPO (**Group Relative Policy Optimization**, the algorithm popularized by DeepSeek and now in TRL) is well suited to this env because:

1. The env has a fast, callable reward function — we just hit the live HF Space `/step` endpoint
2. The reward is dense enough at intermediate steps and sharp at the end (F1 score)
3. The action space is structured JSON, which group-relative advantages handle better than raw token-level PPO

The reward function below is a thin wrapper around the env: parse the model's JSON output, submit each finding via `/step`, call `submit_report`, return the F1 score. TRL's `GRPOTrainer` will sample several rollouts per prompt, normalize advantages within the group, and apply a policy update.

> **Compute note.** GRPO is real online RL. Even with a 0.5B model, doing this properly wants an A100 or similar. The cell below sets it up but does not call `.train()` so you can read the recipe without burning credits. Uncomment the bottom lines once you're on adequate hardware.
"""))

cells.append(code(r"""
# This cell is a RECIPE — it shows how to set up GRPO but does NOT run
# training (.train() is commented out). If trl is unavailable due to a
# Colab kernel restart, we skip gracefully.
try:
    import inspect
    from trl import GRPOConfig, GRPOTrainer
except ImportError:
    print("trl not available (Colab may have restarted the kernel).")
    print("The GRPO recipe is visible in the cell source code above.")
    print("To run it: restart runtime, run cell 1 (install), then this cell.")
    GRPOConfig = None

if GRPOConfig is not None:
    PROMPTS = [
        {
            "task_id": d["task_id"],
            "prompt": tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": d["user"]},
                ],
                tokenize=False,
                add_generation_prompt=True,
            ),
        }
        for d in dataset
    ]
    prompt_dataset = Dataset.from_list(PROMPTS)

    def env_reward_function(prompts, completions, task_id, **kwargs):
        rewards = []
        for tid, completion in zip(task_id, completions):
            client.post("/reset", json={"task_id": tid})
            findings = parse_findings_from_response(completion)
            for finding in findings:
                if not isinstance(finding, dict):
                    continue
                client.post("/step", json={"action": {
                    "action": "flag_issue",
                    "invoice_id": finding.get("invoice_id", ""),
                    "field": finding.get("field", ""),
                    "category": finding.get("category", ""),
                    "severity": finding.get("severity", "major"),
                    "description": finding.get("description", ""),
                }})
            result = client.post("/step", json={"action": {"action": "submit_report"}}).json()
            obs = result.get("observation", result.get("state", {}))
            rewards.append(float(obs.get("score", 0.0)))
        return rewards

    _grpo_params = inspect.signature(GRPOConfig).parameters
    grpo_kwargs = dict(
        output_dir="./gst-auditor-qwen-0.5b-grpo",
        learning_rate=5e-6,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
    )
    for name, val in [
        ("num_generations", 4),
        ("max_prompt_length", 2048),
        ("max_completion_length", 1024),
    ]:
        if name in _grpo_params:
            grpo_kwargs[name] = val

    grpo_config = GRPOConfig(**grpo_kwargs)

    # Uncomment once you have adequate GPU memory (A100 40GB or larger):
    #
    # grpo_trainer = GRPOTrainer(
    #     model=model,
    #     reward_funcs=env_reward_function,
    #     args=grpo_config,
    #     train_dataset=prompt_dataset,
    # )
    # grpo_trainer.train()

print("GRPO trainer configured. Uncomment the .train() call once on adequate GPU.")
"""))

# ─── Part 10: Save + share ───────────────────────────────────────────────
cells.append(md(r"""
## 11. Save the trained adapter

We save just the LoRA adapter (a few MB) rather than the full base model. To restore: load Qwen2.5-0.5B-Instruct and `PeftModel.from_pretrained(model, adapter_dir)`.
"""))

cells.append(code(r"""
ADAPTER_DIR = "./gst-auditor-qwen-0.5b-lora"

model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

import os
print(f"Saved {ADAPTER_DIR}/")
for f in sorted(os.listdir(ADAPTER_DIR)):
    size_kb = os.path.getsize(os.path.join(ADAPTER_DIR, f)) / 1024
    print(f"  {f:40s}  {size_kb:8.1f} KB")

# Optional -- push to the HF Hub
# from huggingface_hub import login
# login(token="hf_...")
# model.push_to_hub("YourUsername/gst-auditor-qwen-0.5b-lora")
# tokenizer.push_to_hub("YourUsername/gst-auditor-qwen-0.5b-lora")
"""))

# ─── Part 11: Recap ──────────────────────────────────────────────────────
cells.append(md(r"""
## What we built

| Step | What it does |
|---|---|
| 4. Distill | Use the deterministic GST rules engine to generate expert findings, with no human annotation |
| 5. Load | Qwen2.5-0.5B-Instruct + LoRA (~1% of parameters trainable) |
| 7. SFT | Fine-tune on the distilled trajectories — fits in a free Colab T4 |
| 9. Evaluate | Score the trained model end-to-end against the live HF Space |
| 10. GRPO | Online RL recipe using the env's reward signal directly |

## Where to go next

- Run [`benchmark.py`](https://github.com/VelrajMurugesan/OpenEnv-Hackathon/blob/master/benchmark.py) with your trained model to compare against the GPT-4o baseline.
- Add more tasks (see `CONTRIBUTING.md`) — the rules engine will produce expert trajectories for them automatically.
- Push your trained adapter to the HF Hub and link it from the README leaderboard.
- Experiment with larger base models (Qwen2.5-1.5B or Qwen2.5-3B) for SFT and see how far you can push past the 0.9911 deterministic ceiling.

## Citation

If you train against this environment, please cite the repo:

```
@misc{gst-invoice-compliance-checker-2026,
  title  = {GST Invoice Compliance Checker: an OpenEnv benchmark for Indian GST audit reasoning},
  author = {Velraj Murugesan},
  year   = {2026},
  url    = {https://huggingface.co/spaces/VelrajMurugesan/gst-invoice-compliance-checker},
}
```
"""))


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
    "accelerator": "GPU",
    "colab": {"provenance": [], "machine_shape": "hm"},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as fh:
    nbf.write(nb, fh)

print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB, {len(cells)} cells)")
