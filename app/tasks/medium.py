"""Medium difficulty tasks: inter/intra-state tax, arithmetic, e-way bill, ITC."""

from app.models import (
    GroundTruthIssue,
    Invoice,
    InvoiceLineItem,
    TaskDifficulty,
    TaskInfo,
    TaxType,
)
from data.gst_rules import run_medium_validation

# ── Task M1: Inter-state vs Intra-state Tax Type Errors ────────────────────

TASK_M1_INFO = TaskInfo(
    task_id="medium_1",
    name="Inter/Intra-State Tax Logic",
    description=(
        "Audit a GST invoice for incorrect tax type application. "
        "Inter-state supplies must use IGST while intra-state supplies "
        "must use CGST+SGST. Check if the tax type matches the supply "
        "direction and if amounts are correctly split."
    ),
    difficulty=TaskDifficulty.MEDIUM,
    max_steps=20,
    num_invoices=2,
)

TASK_M1_INVOICES = [
    # Invoice 1: Inter-state (MH→KA) but using CGST+SGST (WRONG)
    Invoice(
        invoice_id="INV-M1-001",
        invoice_number="INV/2025/0201",
        invoice_date="2025-03-01",
        supplier_name="Mumbai Electronics Hub",
        supplier_gstin="27AABCE1234F1Z5",
        supplier_state_code="27",
        recipient_name="Bangalore Tech Store",
        recipient_gstin="29BCDFT5678G1Z2",
        recipient_state_code="29",
        place_of_supply="29",  # Karnataka — inter-state
        supply_type="B2B",
        line_items=[
            InvoiceLineItem(
                description="Desktop Computers",
                hsn_code="8471",
                quantity=10,
                unit_price=35000.0,
                taxable_value=350000.0,
                tax_rate=18.0,
                tax_type=TaxType.CGST_SGST,  # WRONG: should be IGST
                cgst_amount=31500.0,
                sgst_amount=31500.0,
                igst_amount=0.0,
                total_amount=413000.0,
            ),
            InvoiceLineItem(
                description="Laser Printers",
                hsn_code="8443",
                quantity=5,
                unit_price=12000.0,
                taxable_value=60000.0,
                tax_rate=18.0,
                tax_type=TaxType.CGST_SGST,  # WRONG: should be IGST
                cgst_amount=5400.0,
                sgst_amount=5400.0,
                igst_amount=0.0,
                total_amount=70800.0,
            ),
        ],
        total_taxable_value=410000.0,
        total_tax=73800.0,
        total_invoice_value=483800.0,
        eway_bill_number="",  # MISSING: > 50K inter-state
    ),
    # Invoice 2: Intra-state (MH→MH) but using IGST (WRONG)
    Invoice(
        invoice_id="INV-M1-002",
        invoice_number="INV/2025/0202",
        invoice_date="2025-03-02",
        supplier_name="Mumbai Electronics Hub",
        supplier_gstin="27AABCE1234F1Z5",
        supplier_state_code="27",
        recipient_name="Pune Office Supplies",
        recipient_gstin="27DEFGH5678K1Z9",
        recipient_state_code="27",
        place_of_supply="27",  # Maharashtra — intra-state
        supply_type="B2B",
        line_items=[
            InvoiceLineItem(
                description="Office Chairs",
                hsn_code="9401",
                quantity=20,
                unit_price=8000.0,
                taxable_value=160000.0,
                tax_rate=18.0,
                tax_type=TaxType.IGST,  # WRONG: should be CGST+SGST
                cgst_amount=0.0,
                sgst_amount=0.0,
                igst_amount=28800.0,
                total_amount=188800.0,
            ),
        ],
        total_taxable_value=160000.0,
        total_tax=28800.0,
        total_invoice_value=188800.0,
    ),
]


# ── Task M2: Arithmetic Errors & GSTIN-State Mismatch ──────────────────────

TASK_M2_INFO = TaskInfo(
    task_id="medium_2",
    name="Arithmetic & GSTIN Consistency",
    description=(
        "Audit invoices for calculation errors and data inconsistencies. "
        "Check if line item totals, tax amounts, and invoice totals are "
        "arithmetically correct. Also verify GSTIN state codes match "
        "declared state codes."
    ),
    difficulty=TaskDifficulty.MEDIUM,
    max_steps=20,
    num_invoices=2,
)

TASK_M2_INVOICES = [
    # Invoice 1: Arithmetic errors in line items and totals
    Invoice(
        invoice_id="INV-M2-001",
        invoice_number="INV/2025/0301",
        invoice_date="2025-03-10",
        supplier_name="Steel Works India",
        supplier_gstin="24AABCS5678D1Z6",
        supplier_state_code="24",  # Gujarat
        recipient_name="Construction Corp",
        recipient_gstin="27BCDEC9012F1Z3",  # MH GSTIN
        recipient_state_code="24",  # WRONG: says Gujarat but GSTIN says MH(27)
        place_of_supply="24",
        supply_type="B2B",
        line_items=[
            InvoiceLineItem(
                description="Steel bars",
                hsn_code="7213",
                quantity=100,
                unit_price=450.0,
                taxable_value=44000.0,  # WRONG: 100*450=45000
                tax_rate=18.0,
                tax_type=TaxType.CGST_SGST,
                cgst_amount=4050.0,
                sgst_amount=4050.0,
                total_amount=52100.0,  # WRONG
            ),
            InvoiceLineItem(
                description="Cement bags",
                hsn_code="2523",
                quantity=200,
                unit_price=350.0,
                taxable_value=70000.0,
                tax_rate=28.0,
                tax_type=TaxType.CGST_SGST,
                cgst_amount=9800.0,
                sgst_amount=9800.0,
                total_amount=89600.0,
            ),
        ],
        total_taxable_value=115000.0,  # WRONG: should be 44000+70000=114000
        total_tax=27700.0,
        total_invoice_value=142700.0,
    ),
    # Invoice 2: Tax amount calculation errors
    Invoice(
        invoice_id="INV-M2-002",
        invoice_number="INV/2025/0302",
        invoice_date="2025-03-12",
        supplier_name="Rajasthan Textiles",
        supplier_gstin="08AABCR3456E1Z1",
        supplier_state_code="08",
        recipient_name="Delhi Fashion House",
        recipient_gstin="07BCDFD7890G1Z4",
        recipient_state_code="07",
        place_of_supply="07",  # Delhi — inter-state from Rajasthan
        supply_type="B2B",
        line_items=[
            InvoiceLineItem(
                description="Cotton fabric rolls",
                hsn_code="5208",
                quantity=50,
                unit_price=2000.0,
                taxable_value=100000.0,
                tax_rate=5.0,
                tax_type=TaxType.IGST,
                igst_amount=7500.0,  # WRONG: 5% of 100000 = 5000
                total_amount=107500.0,  # WRONG
            ),
            InvoiceLineItem(
                description="Men's trousers",
                hsn_code="6203",
                quantity=100,
                unit_price=800.0,
                taxable_value=80000.0,
                tax_rate=12.0,
                tax_type=TaxType.IGST,
                igst_amount=9600.0,
                total_amount=89600.0,
            ),
        ],
        total_taxable_value=180000.0,
        total_tax=17100.0,
        total_invoice_value=197100.0,
        eway_bill_number="",  # MISSING: > 50K inter-state
    ),
]


# ── Task M3: E-way Bill & Input Tax Credit Issues ──────────────────────────

TASK_M3_INFO = TaskInfo(
    task_id="medium_3",
    name="E-way Bill & Compliance Checks",
    description=(
        "Audit invoices for e-way bill compliance and other regulatory "
        "requirements. Inter-state supplies exceeding INR 50,000 require "
        "a valid 12-digit e-way bill number. Also check for state code "
        "validity and other compliance issues."
    ),
    difficulty=TaskDifficulty.MEDIUM,
    max_steps=20,
    num_invoices=2,
)

TASK_M3_INVOICES = [
    # Invoice 1: Missing e-way bill for high-value inter-state supply
    Invoice(
        invoice_id="INV-M3-001",
        invoice_number="INV/2025/0401",
        invoice_date="2025-03-20",
        supplier_name="Chennai Auto Parts",
        supplier_gstin="33AABCA1234B1Z8",
        supplier_state_code="33",
        recipient_name="Hyderabad Motors",
        recipient_gstin="36BCDEH5678C1Z5",
        recipient_state_code="36",
        place_of_supply="36",  # Telangana — inter-state
        supply_type="B2B",
        line_items=[
            InvoiceLineItem(
                description="Motorcycle tyres",
                hsn_code="4011",
                quantity=100,
                unit_price=2500.0,
                taxable_value=250000.0,
                tax_rate=28.0,
                tax_type=TaxType.IGST,
                igst_amount=70000.0,
                total_amount=320000.0,
            ),
            InvoiceLineItem(
                description="Motorcycle parts",
                hsn_code="8714",
                quantity=50,
                unit_price=1500.0,
                taxable_value=75000.0,
                tax_rate=18.0,
                tax_type=TaxType.IGST,
                igst_amount=13500.0,
                total_amount=88500.0,
            ),
        ],
        total_taxable_value=325000.0,
        total_tax=83500.0,
        total_invoice_value=408500.0,
        eway_bill_number="12345678",  # INVALID: only 8 digits, needs 12
    ),
    # Invoice 2: Invalid e-way bill + wrong HSN code
    Invoice(
        invoice_id="INV-M3-002",
        invoice_number="INV/2025/0402",
        invoice_date="2025-03-22",
        supplier_name="Gujarat Appliances",
        supplier_gstin="24AABCG5678D1Z3",
        supplier_state_code="24",
        recipient_name="MP Home Store",
        recipient_gstin="23BCDEM9012E1Z7",
        recipient_state_code="23",
        place_of_supply="23",  # MP — inter-state
        supply_type="B2B",
        line_items=[
            InvoiceLineItem(
                description="Air conditioners",
                hsn_code="8415",
                quantity=20,
                unit_price=32000.0,
                taxable_value=640000.0,
                tax_rate=28.0,
                tax_type=TaxType.IGST,
                igst_amount=179200.0,
                total_amount=819200.0,
            ),
            InvoiceLineItem(
                description="Washing machines",
                hsn_code="8450",
                quantity=15,
                unit_price=18000.0,
                taxable_value=270000.0,
                tax_rate=28.0,  # WRONG: 8450 is 18%
                tax_type=TaxType.IGST,
                igst_amount=75600.0,
                total_amount=345600.0,
            ),
        ],
        total_taxable_value=910000.0,
        total_tax=254800.0,
        total_invoice_value=1164800.0,
        eway_bill_number="",  # MISSING
    ),
]


def get_medium_task(task_id: str) -> tuple[TaskInfo, list[Invoice], list[GroundTruthIssue]]:
    """Return task info, invoices, and ground truth issues for a medium task."""
    tasks = {
        "medium_1": (TASK_M1_INFO, TASK_M1_INVOICES),
        "medium_2": (TASK_M2_INFO, TASK_M2_INVOICES),
        "medium_3": (TASK_M3_INFO, TASK_M3_INVOICES),
    }

    if task_id not in tasks:
        raise ValueError(f"Unknown medium task: {task_id}")

    info, invoices = tasks[task_id]
    ground_truth = []
    for inv in invoices:
        ground_truth.extend(run_medium_validation(inv))

    return info, invoices, ground_truth
