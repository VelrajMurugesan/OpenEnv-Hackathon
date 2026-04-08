"""Baseline inference script for the GST Invoice Compliance Checker OpenEnv.

Hybrid approach: programmatic validation for deterministic rules + LLM for
complex interpretation.

This script is what the OpenEnv hackathon Phase 2 deep-validator runs as a
subprocess. It MUST emit `[START]`, `[STEP]` and `[END]` log lines on stdout
in space-separated `key=value` format (matching the official sample
inference.py from ArjunMadhava/meta-hackathon-2026). The validator regex-parses
these lines to extract per-task scores and asserts that every score is
strictly inside the open interval (0, 1) — never exactly 0.0 or 1.0.

All non-protocol diagnostic output goes to stderr via `log_diagnostic` so it
cannot pollute the parser.

Configuration via environment variables:
  - API_BASE_URL: LLM endpoint (e.g. https://api.openai.com/v1)
  - MODEL_NAME:   Model identifier (e.g. gpt-4o)
  - HF_TOKEN:     Hugging Face token for authentication
  - ENV_URL:      Base URL of the running OpenEnv environment
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback

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

# Action ID mapping for the [STEP] log — the validator expects an integer
# action= field even though our env uses string action types.
ACTION_ID = {
    "reset": -1,
    "flag_issue": 0,
    "approve": 1,
    "submit_report": 2,
}

# OpenEnv validator requires per-task scores strictly in (0, 1) — never 0.0
# or 1.0 — so we clamp client-side as a defense-in-depth layer on top of the
# server's grader clamp.
SCORE_MIN = 0.0001
SCORE_MAX = 0.9998


def clamp_score(value: float) -> float:
    """Clamp any score into the strict open interval (0, 1)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return SCORE_MIN
    if v != v:  # NaN
        return SCORE_MIN
    if v <= SCORE_MIN:
        return SCORE_MIN
    if v >= SCORE_MAX:
        return SCORE_MAX
    return v


# ── HSN → Valid Tax Rates (exact copy from data/hsn_codes.py) ───────────────

HSN_RATES: dict[str, list[float]] = {
    "0201": [5.0], "0401": [0.0], "0402": [5.0], "0713": [0.0],
    "0902": [5.0], "0901": [5.0], "1001": [0.0], "1006": [5.0],
    "1101": [5.0], "1701": [5.0], "1905": [18.0], "2106": [18.0],
    "2201": [18.0], "2202": [28.0], "2203": [28.0],
    "5208": [5.0], "6109": [5.0, 12.0], "6203": [12.0], "6204": [12.0],
    "8471": [18.0], "8517": [18.0], "8528": [18.0], "8443": [18.0],
    "8415": [28.0], "8418": [18.0], "8450": [18.0], "8516": [18.0],
    "8703": [28.0], "8711": [28.0], "8714": [18.0, 28.0], "4011": [28.0],
    "3004": [5.0, 12.0], "3003": [12.0], "3005": [12.0], "3006": [12.0],
    "9018": [12.0], "4802": [12.0], "4820": [12.0], "9608": [18.0],
    "2523": [28.0], "7213": [18.0], "6802": [12.0, 18.0],
    "9401": [18.0], "9403": [18.0],
    "9983": [18.0], "9971": [18.0], "9973": [18.0], "9981": [18.0],
    "9982": [18.0], "9984": [18.0], "9985": [18.0], "9986": [0.0, 5.0],
    "9987": [18.0], "9988": [18.0], "9961": [5.0, 12.0], "9962": [5.0, 12.0],
    "9963": [5.0, 12.0, 18.0], "9964": [5.0], "9972": [12.0, 18.0],
    "9991": [0.0], "9992": [0.0], "9993": [0.0], "9995": [0.0],
    "9996": [0.0], "9997": [18.0],
}

VALID_TAX_RATES = [0.0, 5.0, 12.0, 18.0, 28.0]
VALID_STATE_CODES = {f"{i:02d}" for i in range(1, 39)}
RCM_HSN_CODES = {"9961", "9962", "9971", "9973", "9985"}
GSTIN_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")


# ── Validator-protocol log emitters ─────────────────────────────────────────


def log_diagnostic(message: str) -> None:
    """Send free-form diagnostic info to stderr so it never pollutes the
    validator's stdout parser."""
    print(message, file=sys.stderr, flush=True)


def emit_start(task_id: str) -> None:
    """Emit the canonical [START] line for a task."""
    print(f"[START] task={task_id}", flush=True)


def emit_step(
    task_id: str,
    step_count: int,
    reward: float,
    action_id: int,
    done: bool,
) -> None:
    """Emit the canonical [STEP] line for a single env step."""
    print(
        "[STEP] "
        f"task={task_id} "
        f"step={step_count} "
        f"reward={float(reward):.4f} "
        f"action={action_id} "
        f"done={str(bool(done)).lower()}",
        flush=True,
    )


def emit_end(task_id: str, score: float, steps: int, status: str) -> None:
    """Emit the canonical [END] line for a finished task. The score is always
    clamped strictly into (0, 1) to satisfy the Phase 2 validator."""
    safe = clamp_score(score)
    print(
        f"[END] task={task_id} score={safe:.4f} steps={steps} status={status}",
        flush=True,
    )


def emit_setup_failure(task_id: str, reason: str) -> None:
    """Emit a fully valid [START]/[STEP]/[END] sequence for a task that
    crashed before it could legitimately produce a score. The score is still
    clamped to the (0, 1) floor so the validator does not flag it as out of
    range."""
    emit_start(task_id)
    emit_step(task_id=task_id, step_count=0, reward=0.0, action_id=-1, done=False)
    emit_end(task_id, SCORE_MIN, 0, reason)


# ── HTTP helper ─────────────────────────────────────────────────────────────


def env_request(method: str, endpoint: str, payload: dict | None = None) -> dict:
    url = f"{ENV_URL}{endpoint}"
    with httpx.Client(timeout=60) as http:
        if method == "GET":
            resp = http.get(url)
        else:
            resp = http.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


# ── Programmatic Validators ─────────────────────────────────────────────────


def programmatic_audit(invoices: list[dict]) -> list[dict]:
    """Run deterministic rule-based checks on all invoices."""
    findings: list[dict] = []

    def flag(inv_id: str, field: str, cat: str, sev: str, desc: str) -> None:
        findings.append({
            "invoice_id": inv_id,
            "field": field,
            "category": cat,
            "severity": sev,
            "description": desc,
        })

    for inv in invoices:
        inv_id = inv.get("invoice_id", "")

        # ── Step 1: Mandatory fields ──
        mandatory = {
            "invoice_number": "Invoice Number",
            "invoice_date": "Invoice Date",
            "supplier_name": "Supplier Name",
            "supplier_gstin": "Supplier GSTIN",
            "recipient_name": "Recipient Name",
            "place_of_supply": "Place of Supply",
        }
        if inv.get("supply_type") == "B2B":
            mandatory["recipient_gstin"] = "Recipient GSTIN"

        for field_key, label in mandatory.items():
            val = inv.get(field_key, "")
            if not val or str(val).strip() == "":
                flag(inv_id, field_key, "missing_field", "critical",
                     f"{label} is missing")

        if not inv.get("line_items"):
            flag(inv_id, "line_items", "missing_field", "critical",
                 "Invoice has no line items")

        # ── Step 2: GSTIN format ──
        for gstin_field in ["supplier_gstin", "recipient_gstin"]:
            gstin = inv.get(gstin_field, "")
            if gstin and not GSTIN_PATTERN.match(gstin):
                flag(inv_id, gstin_field, "invalid_format", "critical",
                     f"GSTIN '{gstin}' has invalid format (expected 15-char pattern)")

        # ── Step 3: State code validation ──
        for sc_field, label in [
            ("supplier_state_code", "Supplier state code"),
            ("recipient_state_code", "Recipient state code"),
            ("place_of_supply", "Place of supply"),
        ]:
            code = inv.get(sc_field, "")
            if code and code not in VALID_STATE_CODES:
                flag(inv_id, sc_field, "invalid_format", "major",
                     f"{label} '{code}' is not a valid Indian state code")

        # GSTIN-state consistency
        supplier_gstin = inv.get("supplier_gstin", "")
        if supplier_gstin and len(supplier_gstin) >= 2:
            gstin_state = supplier_gstin[:2]
            declared = inv.get("supplier_state_code", "")
            if declared and gstin_state != declared:
                flag(inv_id, "supplier_state_code", "inconsistency", "major",
                     f"Supplier GSTIN state ({gstin_state}) != declared state ({declared})")

        recipient_gstin = inv.get("recipient_gstin", "")
        if recipient_gstin and len(recipient_gstin) >= 2:
            gstin_state = recipient_gstin[:2]
            declared = inv.get("recipient_state_code", "")
            if declared and gstin_state != declared:
                flag(inv_id, "recipient_state_code", "inconsistency", "major",
                     f"Recipient GSTIN state ({gstin_state}) != declared state ({declared})")

        # ── Step 4: Invoice date format ──
        date_val = inv.get("invoice_date", "")
        if date_val and not re.match(r"^\d{4}-\d{2}-\d{2}$", date_val):
            flag(inv_id, "invoice_date", "invalid_format", "minor",
                 f"Invoice date '{date_val}' not in YYYY-MM-DD format")

        # ── Step 5: Line item checks ──
        supplier_state = inv.get("supplier_state_code", "")
        pos = inv.get("place_of_supply", "")
        is_interstate = supplier_state != pos

        computed_taxable = 0.0
        computed_tax = 0.0

        for idx, item in enumerate(inv.get("line_items", [])):
            hsn = item.get("hsn_code", "")
            tax_rate = item.get("tax_rate", 0)
            taxable_value = item.get("taxable_value", 0)
            quantity = item.get("quantity", 0)
            unit_price = item.get("unit_price", 0)
            tax_type = item.get("tax_type", "")

            # HSN code validity
            if hsn and hsn not in HSN_RATES:
                flag(inv_id, f"line_items[{idx}].hsn_code", "invalid_format", "major",
                     f"HSN/SAC code '{hsn}' is not recognized")

            # Tax rate validity
            if tax_rate not in VALID_TAX_RATES:
                flag(inv_id, f"line_items[{idx}].tax_rate", "wrong_value", "critical",
                     f"Tax rate {tax_rate}% is not a valid GST rate")

            # Tax rate vs HSN match
            if hsn in HSN_RATES and tax_rate not in HSN_RATES[hsn]:
                flag(inv_id, f"line_items[{idx}].tax_rate", "tax_mismatch", "critical",
                     f"Tax rate {tax_rate}% doesn't match HSN {hsn} (valid: {HSN_RATES[hsn]})")

            # Inter/intra state tax type
            if is_interstate and tax_type != "IGST":
                flag(inv_id, f"line_items[{idx}].tax_type", "tax_mismatch", "critical",
                     f"Inter-state supply should use IGST, not {tax_type}")
            elif not is_interstate and tax_type != "CGST+SGST":
                flag(inv_id, f"line_items[{idx}].tax_type", "tax_mismatch", "critical",
                     f"Intra-state supply should use CGST+SGST, not {tax_type}")

            # Tax amount checks
            expected_tax = round(taxable_value * tax_rate / 100, 2)
            if is_interstate:
                actual_igst = item.get("igst_amount", 0)
                if abs(actual_igst - expected_tax) > 0.01:
                    flag(inv_id, f"line_items[{idx}].igst_amount", "wrong_value", "major",
                         f"IGST amount {actual_igst} != expected {expected_tax}")
            else:
                half_tax = round(expected_tax / 2, 2)
                actual_cgst = item.get("cgst_amount", 0)
                actual_sgst = item.get("sgst_amount", 0)
                if abs(actual_cgst - half_tax) > 0.01:
                    flag(inv_id, f"line_items[{idx}].cgst_amount", "wrong_value", "major",
                         f"CGST amount {actual_cgst} != expected {half_tax}")
                if abs(actual_sgst - half_tax) > 0.01:
                    flag(inv_id, f"line_items[{idx}].sgst_amount", "wrong_value", "major",
                         f"SGST amount {actual_sgst} != expected {half_tax}")

            # Arithmetic: taxable_value = qty * unit_price
            expected_taxable = round(quantity * unit_price, 2)
            if abs(taxable_value - expected_taxable) > 0.01:
                flag(inv_id, f"line_items[{idx}].taxable_value", "wrong_value", "major",
                     f"Taxable value {taxable_value} != qty({quantity}) x price({unit_price}) = {expected_taxable}")

            # Arithmetic: total_amount
            actual_total = item.get("total_amount", 0)
            expected_total = round(taxable_value + expected_tax, 2)
            if abs(actual_total - expected_total) > 0.01:
                flag(inv_id, f"line_items[{idx}].total_amount", "wrong_value", "major",
                     f"Line total {actual_total} != taxable({taxable_value}) + tax({expected_tax}) = {expected_total}")

            actual_tax_total = item.get("igst_amount", 0) + item.get("cgst_amount", 0) + item.get("sgst_amount", 0)
            computed_taxable += taxable_value
            computed_tax += actual_tax_total

        # Invoice-level arithmetic
        computed_taxable = round(computed_taxable, 2)
        computed_tax = round(computed_tax, 2)

        if abs(inv.get("total_taxable_value", 0) - computed_taxable) > 0.01:
            flag(inv_id, "total_taxable_value", "wrong_value", "major",
                 f"Total taxable {inv.get('total_taxable_value', 0)} != sum of lines {computed_taxable}")

        if abs(inv.get("total_tax", 0) - computed_tax) > 0.01:
            flag(inv_id, "total_tax", "wrong_value", "major",
                 f"Total tax {inv.get('total_tax', 0)} != sum of line taxes {computed_tax}")

        expected_inv_total = round(computed_taxable + computed_tax, 2)
        if abs(inv.get("total_invoice_value", 0) - expected_inv_total) > 0.01:
            flag(inv_id, "total_invoice_value", "wrong_value", "major",
                 f"Invoice total {inv.get('total_invoice_value', 0)} != {expected_inv_total}")

        # ── Step 6: E-way bill ──
        if is_interstate and inv.get("total_invoice_value", 0) > 50000:
            eway = inv.get("eway_bill_number", "")
            if not eway or eway.strip() == "":
                flag(inv_id, "eway_bill_number", "compliance_violation", "critical",
                     f"E-way bill required for inter-state supply > INR 50,000 (value: {inv.get('total_invoice_value', 0)})")
            elif len(eway) != 12 or not eway.isdigit():
                flag(inv_id, "eway_bill_number", "invalid_format", "major",
                     f"E-way bill number must be 12 digits, got '{eway}'")

        # ── Step 7: Reverse charge ──
        for idx, item in enumerate(inv.get("line_items", [])):
            if item.get("hsn_code", "") in RCM_HSN_CODES:
                if not inv.get("reverse_charge", False):
                    flag(inv_id, "reverse_charge", "compliance_violation", "critical",
                         f"HSN {item['hsn_code']} requires reverse charge but invoice not marked as RCM")
                    break  # One flag per invoice for RCM

        # ── Step 8: Composition scheme ──
        if inv.get("is_composition_scheme", False):
            if is_interstate:
                flag(inv_id, "supply_type", "compliance_violation", "critical",
                     "Composition scheme dealers cannot make inter-state supplies")
            for idx, item in enumerate(inv.get("line_items", [])):
                if item.get("tax_rate", 0) > 5.0:
                    flag(inv_id, f"line_items[{idx}].tax_rate", "compliance_violation", "major",
                         f"Composition scheme tax rate cannot exceed 5%, found {item['tax_rate']}%")
            if inv.get("reverse_charge", False):
                flag(inv_id, "is_composition_scheme", "compliance_violation", "critical",
                     "Composition scheme dealers cannot issue reverse charge invoices")

    # ── Step 9: Duplicate detection (cross-invoice) ──
    seen: dict[str, str] = {}
    for inv in invoices:
        key = f"{inv.get('supplier_gstin', '')}|{inv.get('invoice_number', '')}"
        inv_id = inv.get("invoice_id", "")
        if key in seen:
            flag(inv_id, "invoice_number", "duplicate", "critical",
                 f"Duplicate invoice number '{inv.get('invoice_number', '')}' from same supplier (also in {seen[key]})")
        else:
            seen[key] = inv_id

    return findings


def build_llm_review_prompt(invoices: list[dict], programmatic_findings: list[dict]) -> str:
    """Ask LLM to review programmatic findings and add any missed issues."""
    findings_summary = json.dumps(programmatic_findings, indent=2)
    invoices_text = json.dumps(invoices, indent=2)

    return f"""You are a senior GST compliance auditor. A rule-based system has already audited these invoices and found the issues listed below.

INVOICES:
{invoices_text}

ISSUES ALREADY FOUND BY AUTOMATED SYSTEM:
{findings_summary}

Review the invoices and the automated findings. The automated system is very accurate, so:
1. Do NOT repeat any finding already listed above
2. Only add findings if you spot something the rules genuinely missed
3. Focus on: unusual patterns, context-dependent issues, or business logic violations

If the automated system found everything, return an empty array: []

Return ONLY a JSON array of additional findings (or []):
{{"invoice_id": "...", "field": "...", "category": "...", "severity": "...", "description": "..."}}"""


def parse_llm_response(response_text: str) -> list[dict]:
    """Parse the LLM response into a list of findings."""
    text = response_text.strip()
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
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return []


# ── Main Inference Loop ─────────────────────────────────────────────────────


def call_llm_review(invoices: list[dict], findings: list[dict]) -> list[dict]:
    """Best-effort LLM review pass. Returns [] on any error."""
    try:
        review_prompt = build_llm_review_prompt(invoices, findings)
        llm_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior Indian GST compliance auditor. "
                        "An automated system already found most issues. "
                        "Only flag genuinely missed violations. Be precise, not verbose. "
                        "Respond only with valid JSON."
                    ),
                },
                {"role": "user", "content": review_prompt},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        response_text = llm_response.choices[0].message.content or "[]"
        return parse_llm_response(response_text)
    except Exception as exc:
        log_diagnostic(f"[WARN] LLM review failed: {exc}")
        return []


def run_task(task_id: str) -> float:
    """Run inference on a single task. Always emits exactly one [START],
    one or more [STEP]s, and exactly one [END] line. The returned score is
    clamped to the open interval (0, 1)."""
    emit_start(task_id)

    step_count = 0
    final_score = SCORE_MIN
    status = "completed"

    try:
        # Reset environment
        state = env_request("POST", "/reset", {"task_id": task_id})
        invoices = state.get("invoices", [])
        log_diagnostic(f"[INFO] {task_id}: reset ok ({len(invoices)} invoices)")

        # ── Pass 1: Programmatic rule-based audit ──
        findings = programmatic_audit(invoices)
        log_diagnostic(f"[INFO] {task_id}: programmatic_audit found {len(findings)} issues")

        # ── Pass 2: LLM review (best effort) ──
        extra_findings = call_llm_review(invoices, findings)

        # Merge LLM findings (deduplicate)
        seen = {(f.get("invoice_id", ""), f.get("field", "")) for f in findings}
        added = 0
        for ef in extra_findings:
            if not isinstance(ef, dict):
                continue
            if ef.get("action") == "approve":
                continue
            key = (ef.get("invoice_id", ""), ef.get("field", ""))
            if key not in seen:
                findings.append(ef)
                seen.add(key)
                added += 1
        log_diagnostic(f"[INFO] {task_id}: llm_review added {added} extras (total={len(findings)})")

        # ── Identify clean invoices and approve them ──
        invoice_ids = {inv.get("invoice_id", "") for inv in invoices}
        flagged_ids = {f.get("invoice_id", "") for f in findings}
        clean_ids = invoice_ids - flagged_ids

        # Submit findings as flag_issue actions
        result: dict | None = None
        for finding in findings:
            action = {
                "action": "flag_issue",
                "invoice_id": finding.get("invoice_id", ""),
                "field": finding.get("field", ""),
                "category": finding.get("category", ""),
                "severity": finding.get("severity", "major"),
                "description": finding.get("description", ""),
            }
            result = env_request("POST", "/step", {"action": action})
            step_count += 1
            emit_step(
                task_id=task_id,
                step_count=step_count,
                reward=result.get("reward", 0.0),
                action_id=ACTION_ID["flag_issue"],
                done=bool(result.get("done", False)),
            )
            if result.get("done", False):
                break

        # Approve clean invoices
        if result is None or not result.get("done", False):
            for clean_id in clean_ids:
                result = env_request("POST", "/step", {
                    "action": {"action": "approve", "invoice_id": clean_id}
                })
                step_count += 1
                emit_step(
                    task_id=task_id,
                    step_count=step_count,
                    reward=result.get("reward", 0.0),
                    action_id=ACTION_ID["approve"],
                    done=bool(result.get("done", False)),
                )
                if result.get("done", False):
                    break

        # Submit final report if not already done
        if result is None or not result.get("done", False):
            result = env_request("POST", "/step", {"action": {"action": "submit_report"}})
            step_count += 1
            emit_step(
                task_id=task_id,
                step_count=step_count,
                reward=result.get("reward", 0.0),
                action_id=ACTION_ID["submit_report"],
                done=bool(result.get("done", False)),
            )

        raw_score = result.get("state", {}).get("score") if result else None
        if raw_score is None:
            log_diagnostic(f"[WARN] {task_id}: no score in final state, defaulting to floor")
            status = "incomplete"
        final_score = clamp_score(raw_score if raw_score is not None else SCORE_MIN)

        grader_info = (result or {}).get("info", {}).get("grader_result", {}).get("details", {})
        log_diagnostic(
            f"[INFO] {task_id}: score={final_score:.4f} "
            f"precision={grader_info.get('precision', 0)} "
            f"recall={grader_info.get('recall', 0)} "
            f"tp={grader_info.get('true_positives', 0)} "
            f"fp={grader_info.get('false_positives', 0)} "
            f"missed={grader_info.get('missed_issues', 0)}"
        )

    except Exception as exc:
        status = "error"
        log_diagnostic(f"[ERROR] {task_id}: {exc}")
        log_diagnostic(traceback.format_exc())
        # Make sure we still emit at least one step so the protocol is complete.
        if step_count == 0:
            emit_step(task_id=task_id, step_count=0, reward=0.0, action_id=-1, done=False)
            step_count = 1
        final_score = SCORE_MIN

    emit_end(task_id, final_score, step_count, status)
    return clamp_score(final_score)


def main() -> None:
    """Run inference on all tasks. Diagnostics go to stderr; only the
    canonical [START]/[STEP]/[END] log lines go to stdout."""
    log_diagnostic("=" * 60)
    log_diagnostic("GST Invoice Compliance Checker — Inference")
    log_diagnostic(f"Model:       {MODEL_NAME}")
    log_diagnostic(f"Environment: {ENV_URL}")
    log_diagnostic("=" * 60)

    scores: list[float] = []
    for task_id in TASK_IDS:
        try:
            score = run_task(task_id)
        except Exception as exc:
            log_diagnostic(f"[FATAL] {task_id}: unhandled exception: {exc}")
            log_diagnostic(traceback.format_exc())
            emit_setup_failure(task_id, "fatal_error")
            score = SCORE_MIN
        scores.append(clamp_score(score))

    avg = sum(scores) / len(scores) if scores else 0.0
    log_diagnostic("=" * 60)
    log_diagnostic("RESULTS SUMMARY")
    log_diagnostic("=" * 60)
    for task_id, s in zip(TASK_IDS, scores):
        tag = "PASS" if s >= 0.5 else "FAIL"
        log_diagnostic(f"  [{tag}] {task_id}: score={s:.4f}")
    log_diagnostic(f"\n  Average Score: {avg:.4f}")
    log_diagnostic("=" * 60)


if __name__ == "__main__":
    main()
