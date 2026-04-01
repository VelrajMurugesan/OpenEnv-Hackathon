"""Baseline inference script for the GST Invoice Compliance Checker OpenEnv.

Uses an LLM via OpenAI-compatible client to audit GST invoices.
Logs structured output following [START]/[STEP]/[END] format.

Configuration via environment variables:
  - API_BASE_URL: LLM endpoint (e.g. https://api.openai.com/v1)
  - MODEL_NAME: Model identifier (e.g. gpt-4o)
  - HF_TOKEN: Hugging Face token for authentication
"""

from __future__ import annotations

import json
import os
import sys
import time

import httpx
from openai import OpenAI

# ── Configuration ───────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or os.environ.get("OPENAI_API_KEY", ""),
)

TASK_IDS = [
    "easy_1", "easy_2", "easy_3",
    "medium_1", "medium_2", "medium_3",
    "hard_1", "hard_2", "hard_3",
]

# ── Helpers ─────────────────────────────────────────────────────────────────

def log(tag: str, data: dict) -> None:
    """Print structured log line."""
    print(f"[{tag}] {json.dumps(data)}", flush=True)


def env_request(method: str, endpoint: str, payload: dict | None = None) -> dict:
    """Make a request to the OpenEnv environment."""
    url = f"{ENV_URL}{endpoint}"
    with httpx.Client(timeout=60) as http:
        if method == "GET":
            resp = http.get(url)
        else:
            resp = http.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


def build_audit_prompt(state: dict) -> str:
    """Build the LLM prompt from the current environment state."""
    invoices_text = json.dumps(state["invoices"], indent=2)

    return f"""You are an expert GST (Goods and Services Tax) compliance auditor for India.

TASK: {state['task_description']}
DIFFICULTY: {state['difficulty']}

INVOICES TO AUDIT:
{invoices_text}

GST RULES TO CHECK:
1. MANDATORY FIELDS: invoice_number, invoice_date (YYYY-MM-DD), supplier_name, supplier_gstin, recipient_name, place_of_supply. For B2B: recipient_gstin is mandatory.
2. GSTIN FORMAT: 15 characters — 2-digit state code + 5 alpha (PAN) + 4 digits + 1 alpha + 1 alphanumeric + Z + 1 checksum.
3. HSN/SAC CODES: Must be valid codes from the GST HSN database.
4. TAX RATES: Must be one of 0%, 5%, 12%, 18%, 28% and must match the HSN code's prescribed rate.
5. INTER vs INTRA STATE: If supplier state != place of supply → IGST. If same state → CGST + SGST (each = half of total rate).
6. ARITHMETIC: qty × unit_price = taxable_value; tax = rate × taxable_value; line total = taxable + tax; invoice total = sum of lines.
7. E-WAY BILL: Required for inter-state supply > INR 50,000. Must be 12 digits.
8. REVERSE CHARGE: HSN codes 9961, 9962, 9971, 9973, 9985 require reverse_charge=true.
9. COMPOSITION SCHEME: Cannot do inter-state supply. Tax rate max 5%. Cannot issue reverse charge invoices.
10. DUPLICATES: Same supplier GSTIN + same invoice number = duplicate.

Respond with a JSON array of findings. Each finding must have:
{{
  "invoice_id": "the invoice ID",
  "field": "the field with the issue",
  "category": "missing_field|invalid_format|wrong_value|tax_mismatch|compliance_violation|inconsistency|duplicate",
  "severity": "critical|major|minor",
  "description": "clear description of the issue"
}}

If an invoice is fully compliant, include an approval:
{{
  "invoice_id": "the invoice ID",
  "action": "approve"
}}

Return ONLY the JSON array, no other text."""


def parse_llm_response(response_text: str) -> list[dict]:
    """Parse the LLM response into a list of findings."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        findings = json.loads(text)
        if isinstance(findings, list):
            return findings
        return [findings]
    except json.JSONDecodeError:
        # Try to extract JSON array from the text
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return []


# ── Main Inference Loop ─────────────────────────────────────────────────────

def run_task(task_id: str) -> dict:
    """Run inference on a single task."""
    log("START", {"task_id": task_id, "model": MODEL_NAME})

    # Reset environment
    state = env_request("POST", "/reset", {"task_id": task_id})
    step_num = 0

    log("STEP", {
        "task_id": task_id,
        "step": step_num,
        "action": "reset",
        "num_invoices": len(state["invoices"]),
    })

    # Get LLM analysis
    prompt = build_audit_prompt(state)

    llm_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert Indian GST compliance auditor. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
    )

    response_text = llm_response.choices[0].message.content or "[]"
    findings = parse_llm_response(response_text)

    step_num += 1
    log("STEP", {
        "task_id": task_id,
        "step": step_num,
        "action": "llm_analysis",
        "findings_count": len(findings),
    })

    # Submit each finding as a step
    for i, finding in enumerate(findings):
        if finding.get("action") == "approve":
            action = {
                "action": "approve",
                "invoice_id": finding.get("invoice_id", ""),
            }
        else:
            action = {
                "action": "flag_issue",
                "invoice_id": finding.get("invoice_id", ""),
                "field": finding.get("field", ""),
                "category": finding.get("category", ""),
                "severity": finding.get("severity", "major"),
                "description": finding.get("description", ""),
            }

        result = env_request("POST", "/step", {"action": action})
        step_num += 1

        log("STEP", {
            "task_id": task_id,
            "step": step_num,
            "action": action["action"],
            "invoice_id": action.get("invoice_id", ""),
            "reward": result.get("reward", 0),
            "done": result.get("done", False),
        })

        if result.get("done", False):
            break

    # Submit final report
    if not result.get("done", False):
        result = env_request("POST", "/step", {"action": {"action": "submit_report"}})
        step_num += 1

    score = result.get("state", {}).get("score", 0.0)
    grader_info = result.get("info", {}).get("grader_result", {})

    log("END", {
        "task_id": task_id,
        "score": score,
        "steps": step_num,
        "precision": grader_info.get("details", {}).get("precision", 0),
        "recall": grader_info.get("details", {}).get("recall", 0),
        "true_positives": grader_info.get("details", {}).get("true_positives", 0),
        "false_positives": grader_info.get("details", {}).get("false_positives", 0),
        "missed_issues": grader_info.get("details", {}).get("missed_issues", 0),
    })

    return {"task_id": task_id, "score": score, "steps": step_num}


def main() -> None:
    """Run inference on all tasks."""
    print("=" * 60, flush=True)
    print("GST Invoice Compliance Checker — Baseline Inference", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"Environment: {ENV_URL}", flush=True)
    print("=" * 60, flush=True)

    results = []
    total_score = 0.0

    for task_id in TASK_IDS:
        try:
            result = run_task(task_id)
            results.append(result)
            total_score += result["score"]
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", flush=True)
            log("END", {"task_id": task_id, "score": 0.0, "error": str(e)})
            results.append({"task_id": task_id, "score": 0.0, "error": str(e)})

    # Summary
    avg_score = total_score / len(TASK_IDS) if TASK_IDS else 0.0
    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        status = "PASS" if r.get("score", 0) >= 0.5 else "FAIL"
        print(f"  [{status}] {r['task_id']}: score={r.get('score', 0):.4f}", flush=True)
    print(f"\n  Average Score: {avg_score:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
