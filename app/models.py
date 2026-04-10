"""Pydantic models for the GST Invoice Compliance Checker OpenEnv environment."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Invoice Models ──────────────────────────────────────────────────────────

class TaxType(str, Enum):
    IGST = "IGST"
    CGST_SGST = "CGST+SGST"


class InvoiceLineItem(BaseModel):
    description: str
    hsn_code: str
    quantity: float
    unit_price: float
    taxable_value: float
    tax_rate: float  # percentage e.g. 18.0
    tax_type: TaxType
    cgst_amount: float = 0.0
    sgst_amount: float = 0.0
    igst_amount: float = 0.0
    total_amount: float


class Invoice(BaseModel):
    invoice_id: str
    invoice_number: str
    invoice_date: str  # YYYY-MM-DD
    supplier_name: str
    supplier_gstin: str
    supplier_state_code: str
    recipient_name: str
    recipient_gstin: str
    recipient_state_code: str
    place_of_supply: str  # state code
    supply_type: str  # "B2B" or "B2C"
    reverse_charge: bool = False
    line_items: list[InvoiceLineItem]
    total_taxable_value: float
    total_tax: float
    total_invoice_value: float
    eway_bill_number: str = ""
    is_composition_scheme: bool = False
    notes: str = ""


# ── Issue Models ────────────────────────────────────────────────────────────

class IssueSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class IssueCategory(str, Enum):
    MISSING_FIELD = "missing_field"
    INVALID_FORMAT = "invalid_format"
    WRONG_VALUE = "wrong_value"
    TAX_MISMATCH = "tax_mismatch"
    COMPLIANCE_VIOLATION = "compliance_violation"
    INCONSISTENCY = "inconsistency"
    DUPLICATE = "duplicate"


class GroundTruthIssue(BaseModel):
    invoice_id: str
    field: str
    category: IssueCategory
    severity: IssueSeverity
    description: str
    expected_value: str = ""
    actual_value: str = ""
    # Optional citation to the actual Indian GST legal source (CGST/IGST Act
    # section, CGST Rule, or CBIC notification). Populated by the rules engine
    # in data/gst_rules.py so that every ground-truth issue is traceable back
    # to the specific statute that makes it a violation. Agents and users can
    # surface this in UIs or eval reports.
    legal_reference: str = ""


# ── OpenEnv API Models ──────────────────────────────────────────────────────

class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TaskInfo(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: TaskDifficulty
    max_steps: int
    num_invoices: int


class EnvironmentInfo(BaseModel):
    name: str = "GST Invoice Compliance Checker"
    version: str = "1.0.0"
    description: str = (
        "An OpenEnv environment where an AI agent audits GST invoices "
        "for compliance violations. Tasks range from basic field validation "
        "to complex multi-invoice audits with reverse charge and ITC rules."
    )
    tasks: list[TaskInfo]


class ActionType(str, Enum):
    FLAG_ISSUE = "flag_issue"
    APPROVE = "approve"
    SUBMIT_REPORT = "submit_report"


class AgentAction(BaseModel):
    action: ActionType
    invoice_id: str = ""
    field: str = ""
    category: str = ""
    severity: str = ""
    description: str = ""


class EnvState(BaseModel):
    task_id: str
    task_description: str
    difficulty: str
    invoices: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    step_count: int
    max_steps: int
    done: bool
    score: float | None = None


class ResetRequest(BaseModel):
    task_id: str = "easy_1"


class StepRequest(BaseModel):
    action: AgentAction


class StepResponse(BaseModel):
    state: EnvState
    reward: float
    done: bool
    info: dict[str, Any] = {}


class GraderResult(BaseModel):
    task_id: str
    # OpenEnv validator requires task scores strictly in (0, 1) — never exactly
    # 0.0 or 1.0 — so the grader must clamp into the open interval before
    # constructing this model.
    score: float = Field(gt=0.0, lt=1.0)
    details: dict[str, Any] = {}
