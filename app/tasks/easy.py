"""Easy difficulty tasks: basic field validation, GSTIN format, tax rate checks."""

from app.models import (
    GroundTruthIssue,
    Invoice,
    InvoiceLineItem,
    TaskDifficulty,
    TaskInfo,
    TaxType,
)
from data.gst_rules import run_easy_validation

# ── Task E1: Missing Mandatory Fields ──────────────────────────────────────

TASK_E1_INFO = TaskInfo(
    task_id="easy_1",
    name="Missing Field Detection",
    description=(
        "Audit a B2B GST invoice for missing mandatory fields. "
        "The invoice has several fields left blank or empty that are "
        "required under GST law. Identify all missing fields."
    ),
    difficulty=TaskDifficulty.EASY,
    max_steps=15,
    num_invoices=1,
)

TASK_E1_INVOICES = [
    Invoice(
        invoice_id="INV-E1-001",
        invoice_number="",  # MISSING
        invoice_date="2025-03-15",
        supplier_name="TechCorp Solutions Pvt Ltd",
        supplier_gstin="27AABCT1234F1Z5",
        supplier_state_code="27",
        recipient_name="",  # MISSING
        recipient_gstin="",  # MISSING (B2B requires this)
        recipient_state_code="27",
        place_of_supply="27",
        supply_type="B2B",
        line_items=[
            InvoiceLineItem(
                description="Laptop Computer",
                hsn_code="8471",
                quantity=5,
                unit_price=45000.0,
                taxable_value=225000.0,
                tax_rate=18.0,
                tax_type=TaxType.CGST_SGST,
                cgst_amount=20250.0,
                sgst_amount=20250.0,
                total_amount=265500.0,
            ),
        ],
        total_taxable_value=225000.0,
        total_tax=40500.0,
        total_invoice_value=265500.0,
    ),
]


# ── Task E2: Invalid GSTIN Format & State Codes ────────────────────────────

TASK_E2_INFO = TaskInfo(
    task_id="easy_2",
    name="GSTIN Format & Code Validation",
    description=(
        "Audit a GST invoice for format errors in GSTIN numbers, "
        "state codes, and date formats. Check if all identifiers "
        "follow the prescribed formats."
    ),
    difficulty=TaskDifficulty.EASY,
    max_steps=15,
    num_invoices=1,
)

TASK_E2_INVOICES = [
    Invoice(
        invoice_id="INV-E2-001",
        invoice_number="GST/2025/0042",
        invoice_date="15-03-2025",  # WRONG FORMAT (should be YYYY-MM-DD)
        supplier_name="Global Pharma Ltd",
        supplier_gstin="29BBDPG5678K1Z",  # INVALID: only 14 chars
        supplier_state_code="29",
        recipient_name="City Hospital",
        recipient_gstin="33AADCH9012M1Z8",  # INVALID: 16 chars
        recipient_state_code="33",
        place_of_supply="99",  # INVALID state code
        supply_type="B2B",
        line_items=[
            InvoiceLineItem(
                description="Paracetamol tablets",
                hsn_code="3004",
                quantity=100,
                unit_price=50.0,
                taxable_value=5000.0,
                tax_rate=5.0,
                tax_type=TaxType.IGST,
                igst_amount=250.0,
                total_amount=5250.0,
            ),
            InvoiceLineItem(
                description="Surgical bandages",
                hsn_code="3005",
                quantity=200,
                unit_price=25.0,
                taxable_value=5000.0,
                tax_rate=12.0,
                tax_type=TaxType.IGST,
                igst_amount=600.0,
                total_amount=5600.0,
            ),
        ],
        total_taxable_value=10000.0,
        total_tax=850.0,
        total_invoice_value=10850.0,
    ),
]


# ── Task E3: Wrong Tax Rates for HSN Codes ─────────────────────────────────

TASK_E3_INFO = TaskInfo(
    task_id="easy_3",
    name="Tax Rate vs HSN Code Mismatch",
    description=(
        "Audit a GST invoice where tax rates applied to line items "
        "don't match the prescribed rates for their HSN/SAC codes. "
        "Identify which items have incorrect tax rates."
    ),
    difficulty=TaskDifficulty.EASY,
    max_steps=15,
    num_invoices=1,
)

TASK_E3_INVOICES = [
    Invoice(
        invoice_id="INV-E3-001",
        invoice_number="INV/2025/0108",
        invoice_date="2025-02-20",
        supplier_name="FoodMart Wholesale",
        supplier_gstin="07AABCF5678D1Z3",
        supplier_state_code="07",
        recipient_name="Delhi Grocers",
        recipient_gstin="07BCDEG9012H1Z7",
        recipient_state_code="07",
        place_of_supply="07",
        supply_type="B2B",
        line_items=[
            InvoiceLineItem(
                description="Wheat flour",
                hsn_code="1101",
                quantity=500,
                unit_price=40.0,
                taxable_value=20000.0,
                tax_rate=18.0,  # WRONG: should be 5%
                tax_type=TaxType.CGST_SGST,
                cgst_amount=1800.0,
                sgst_amount=1800.0,
                total_amount=23600.0,
            ),
            InvoiceLineItem(
                description="Mineral water bottles",
                hsn_code="2201",
                quantity=100,
                unit_price=20.0,
                taxable_value=2000.0,
                tax_rate=18.0,  # CORRECT for 2201
                tax_type=TaxType.CGST_SGST,
                cgst_amount=180.0,
                sgst_amount=180.0,
                total_amount=2360.0,
            ),
            InvoiceLineItem(
                description="Aerated cola drinks",
                hsn_code="2202",
                quantity=50,
                unit_price=30.0,
                taxable_value=1500.0,
                tax_rate=18.0,  # WRONG: should be 28%
                tax_type=TaxType.CGST_SGST,
                cgst_amount=135.0,
                sgst_amount=135.0,
                total_amount=1770.0,
            ),
            InvoiceLineItem(
                description="Rice bags",
                hsn_code="1006",
                quantity=200,
                unit_price=60.0,
                taxable_value=12000.0,
                tax_rate=12.0,  # WRONG: should be 5%
                tax_type=TaxType.CGST_SGST,
                cgst_amount=720.0,
                sgst_amount=720.0,
                total_amount=13440.0,
            ),
        ],
        total_taxable_value=35500.0,
        total_tax=5670.0,
        total_invoice_value=41170.0,
    ),
]


# ── Ground Truth Generator ─────────────────────────────────────────────────

def get_easy_task(task_id: str) -> tuple[TaskInfo, list[Invoice], list[GroundTruthIssue]]:
    """Return task info, invoices, and ground truth issues for an easy task."""
    tasks = {
        "easy_1": (TASK_E1_INFO, TASK_E1_INVOICES),
        "easy_2": (TASK_E2_INFO, TASK_E2_INVOICES),
        "easy_3": (TASK_E3_INFO, TASK_E3_INVOICES),
    }

    if task_id not in tasks:
        raise ValueError(f"Unknown easy task: {task_id}")

    info, invoices = tasks[task_id]
    ground_truth = []
    for inv in invoices:
        ground_truth.extend(run_easy_validation(inv))

    return info, invoices, ground_truth
