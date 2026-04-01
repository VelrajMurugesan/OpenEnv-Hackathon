"""Hard difficulty tasks: batch audits, reverse charge, composition scheme."""

from app.models import (
    GroundTruthIssue,
    Invoice,
    InvoiceLineItem,
    TaskDifficulty,
    TaskInfo,
    TaxType,
)
from data.gst_rules import run_batch_validation

# ── Task H1: Reverse Charge & Composition Scheme Violations ────────────────

TASK_H1_INFO = TaskInfo(
    task_id="hard_1",
    name="Reverse Charge & Composition Scheme",
    description=(
        "Audit invoices for Reverse Charge Mechanism (RCM) violations and "
        "composition scheme compliance. RCM applies to specific services "
        "from unregistered suppliers. Composition scheme dealers face "
        "restrictions on inter-state supply and tax rates."
    ),
    difficulty=TaskDifficulty.HARD,
    max_steps=25,
    num_invoices=3,
)

TASK_H1_INVOICES = [
    # Invoice 1: Should be RCM but not marked
    Invoice(
        invoice_id="INV-H1-001",
        invoice_number="INV/2025/0501",
        invoice_date="2025-03-05",
        supplier_name="Local Transport Co",
        supplier_gstin="27AABCL1234M1Z6",
        supplier_state_code="27",
        recipient_name="Acme Manufacturing",
        recipient_gstin="27BCDEA5678N1Z3",
        recipient_state_code="27",
        place_of_supply="27",
        supply_type="B2B",
        reverse_charge=False,  # WRONG: GTA service (9961) requires RCM
        line_items=[
            InvoiceLineItem(
                description="Goods transport service",
                hsn_code="9961",  # GTA — reverse charge applicable
                quantity=1,
                unit_price=45000.0,
                taxable_value=45000.0,
                tax_rate=5.0,
                tax_type=TaxType.CGST_SGST,
                cgst_amount=1125.0,
                sgst_amount=1125.0,
                total_amount=47250.0,
            ),
        ],
        total_taxable_value=45000.0,
        total_tax=2250.0,
        total_invoice_value=47250.0,
    ),
    # Invoice 2: Composition scheme dealer making inter-state supply (WRONG)
    Invoice(
        invoice_id="INV-H1-002",
        invoice_number="INV/2025/0502",
        invoice_date="2025-03-08",
        supplier_name="Small Trader",
        supplier_gstin="08AABCS9012P1Z7",
        supplier_state_code="08",  # Rajasthan
        recipient_name="Delhi Retailer",
        recipient_gstin="07BCDED3456Q1Z4",
        recipient_state_code="07",
        place_of_supply="07",  # Delhi — inter-state (VIOLATION for composition)
        supply_type="B2B",
        is_composition_scheme=True,
        line_items=[
            InvoiceLineItem(
                description="Handicraft items",
                hsn_code="6802",
                quantity=50,
                unit_price=500.0,
                taxable_value=25000.0,
                tax_rate=12.0,  # WRONG: composition scheme max 5%
                tax_type=TaxType.IGST,
                igst_amount=3000.0,
                total_amount=28000.0,
            ),
        ],
        total_taxable_value=25000.0,
        total_tax=3000.0,
        total_invoice_value=28000.0,
    ),
    # Invoice 3: Composition scheme + reverse charge (double violation)
    Invoice(
        invoice_id="INV-H1-003",
        invoice_number="INV/2025/0503",
        invoice_date="2025-03-10",
        supplier_name="Village Services",
        supplier_gstin="09AABCV7890R1Z1",
        supplier_state_code="09",
        recipient_name="UP Factory",
        recipient_gstin="09BCDEF1234S1Z8",
        recipient_state_code="09",
        place_of_supply="09",
        supply_type="B2B",
        is_composition_scheme=True,
        reverse_charge=True,  # VIOLATION: composition + RCM
        line_items=[
            InvoiceLineItem(
                description="Financial advisory service",
                hsn_code="9971",  # Financial services — RCM applicable
                quantity=1,
                unit_price=30000.0,
                taxable_value=30000.0,
                tax_rate=18.0,  # WRONG: composition max 5%
                tax_type=TaxType.CGST_SGST,
                cgst_amount=2700.0,
                sgst_amount=2700.0,
                total_amount=35400.0,
            ),
        ],
        total_taxable_value=30000.0,
        total_tax=5400.0,
        total_invoice_value=35400.0,
    ),
]


# ── Task H2: Multi-Invoice Batch Audit with Duplicates ─────────────────────

TASK_H2_INFO = TaskInfo(
    task_id="hard_2",
    name="Batch Audit with Duplicates",
    description=(
        "Perform a comprehensive audit on a batch of 4 invoices. "
        "Check for all types of violations including duplicate invoice "
        "numbers from the same supplier, cross-invoice inconsistencies, "
        "and individual invoice compliance issues."
    ),
    difficulty=TaskDifficulty.HARD,
    max_steps=30,
    num_invoices=4,
)

TASK_H2_INVOICES = [
    # Invoice 1: Clean invoice (no issues — tests false positive avoidance)
    Invoice(
        invoice_id="INV-H2-001",
        invoice_number="INV/2025/0601",
        invoice_date="2025-03-15",
        supplier_name="Quality Furniture",
        supplier_gstin="29AABCQ1234T1Z5",
        supplier_state_code="29",
        recipient_name="Bangalore Office Hub",
        recipient_gstin="29BCDEF5678U1Z2",
        recipient_state_code="29",
        place_of_supply="29",
        supply_type="B2B",
        line_items=[
            InvoiceLineItem(
                description="Office desks",
                hsn_code="9403",
                quantity=10,
                unit_price=15000.0,
                taxable_value=150000.0,
                tax_rate=18.0,
                tax_type=TaxType.CGST_SGST,
                cgst_amount=13500.0,
                sgst_amount=13500.0,
                total_amount=177000.0,
            ),
        ],
        total_taxable_value=150000.0,
        total_tax=27000.0,
        total_invoice_value=177000.0,
    ),
    # Invoice 2: Multiple issues — wrong tax type + missing e-way bill
    Invoice(
        invoice_id="INV-H2-002",
        invoice_number="INV/2025/0602",
        invoice_date="2025-03-16",
        supplier_name="Kerala Spices Export",
        supplier_gstin="32AABCK5678V1Z9",
        supplier_state_code="32",
        recipient_name="Mumbai Traders",
        recipient_gstin="27BCDEM9012W1Z6",
        recipient_state_code="27",
        place_of_supply="27",  # MH — inter-state from KL
        supply_type="B2B",
        line_items=[
            InvoiceLineItem(
                description="Coffee premium blend",
                hsn_code="0901",
                quantity=200,
                unit_price=500.0,
                taxable_value=100000.0,
                tax_rate=5.0,
                tax_type=TaxType.CGST_SGST,  # WRONG: inter-state → IGST
                cgst_amount=2500.0,
                sgst_amount=2500.0,
                igst_amount=0.0,
                total_amount=105000.0,
            ),
            InvoiceLineItem(
                description="Tea assortment",
                hsn_code="0902",
                quantity=300,
                unit_price=300.0,
                taxable_value=90000.0,
                tax_rate=5.0,
                tax_type=TaxType.CGST_SGST,  # WRONG: inter-state → IGST
                cgst_amount=2250.0,
                sgst_amount=2250.0,
                igst_amount=0.0,
                total_amount=94500.0,
            ),
        ],
        total_taxable_value=190000.0,
        total_tax=9500.0,
        total_invoice_value=199500.0,
        eway_bill_number="",  # MISSING
    ),
    # Invoice 3: DUPLICATE invoice number of Invoice 4 (same supplier)
    Invoice(
        invoice_id="INV-H2-003",
        invoice_number="INV/2025/0699",  # DUPLICATE with INV-H2-004
        invoice_date="2025-03-18",
        supplier_name="Tamil IT Services",
        supplier_gstin="33AABCT3456X1Z3",
        supplier_state_code="33",
        recipient_name="Chennai Client Corp",
        recipient_gstin="33BCDEF7890Y1Z7",
        recipient_state_code="33",
        place_of_supply="33",
        supply_type="B2B",
        line_items=[
            InvoiceLineItem(
                description="IT consulting services",
                hsn_code="9983",
                quantity=1,
                unit_price=200000.0,
                taxable_value=200000.0,
                tax_rate=18.0,
                tax_type=TaxType.CGST_SGST,
                cgst_amount=18000.0,
                sgst_amount=18000.0,
                total_amount=236000.0,
            ),
        ],
        total_taxable_value=200000.0,
        total_tax=36000.0,
        total_invoice_value=236000.0,
    ),
    # Invoice 4: DUPLICATE invoice number + wrong tax rate
    Invoice(
        invoice_id="INV-H2-004",
        invoice_number="INV/2025/0699",  # DUPLICATE with INV-H2-003
        invoice_date="2025-03-19",
        supplier_name="Tamil IT Services",
        supplier_gstin="33AABCT3456X1Z3",  # Same supplier
        supplier_state_code="33",
        recipient_name="Madurai Software",
        recipient_gstin="33BCDEG1234Z1Z4",
        recipient_state_code="33",
        place_of_supply="33",
        supply_type="B2B",
        line_items=[
            InvoiceLineItem(
                description="Software development",
                hsn_code="9983",
                quantity=1,
                unit_price=350000.0,
                taxable_value=350000.0,
                tax_rate=12.0,  # WRONG: 9983 should be 18%
                tax_type=TaxType.CGST_SGST,
                cgst_amount=21000.0,
                sgst_amount=21000.0,
                total_amount=392000.0,
            ),
        ],
        total_taxable_value=350000.0,
        total_tax=42000.0,
        total_invoice_value=392000.0,
    ),
]


# ── Task H3: Full Compliance Audit — Everything Combined ───────────────────

TASK_H3_INFO = TaskInfo(
    task_id="hard_3",
    name="Full Compliance Audit",
    description=(
        "Perform a comprehensive GST compliance audit on a batch of 5 invoices. "
        "This combines ALL validation types: mandatory fields, GSTIN formats, "
        "tax rates, inter/intra-state logic, arithmetic, e-way bills, "
        "reverse charge, composition scheme, and duplicate detection. "
        "Some invoices may be fully compliant — avoid false positives."
    ),
    difficulty=TaskDifficulty.HARD,
    max_steps=35,
    num_invoices=5,
)

TASK_H3_INVOICES = [
    # Invoice 1: Clean — no issues
    Invoice(
        invoice_id="INV-H3-001",
        invoice_number="INV/2025/0701",
        invoice_date="2025-03-25",
        supplier_name="Delhi IT Solutions",
        supplier_gstin="07AABCD1234E1Z8",
        supplier_state_code="07",
        recipient_name="Noida Tech Park",
        recipient_gstin="09BCDEF5678G1Z5",
        recipient_state_code="09",
        place_of_supply="09",  # UP — inter-state
        supply_type="B2B",
        line_items=[
            InvoiceLineItem(
                description="Server maintenance",
                hsn_code="9987",
                quantity=1,
                unit_price=80000.0,
                taxable_value=80000.0,
                tax_rate=18.0,
                tax_type=TaxType.IGST,
                igst_amount=14400.0,
                total_amount=94400.0,
            ),
        ],
        total_taxable_value=80000.0,
        total_tax=14400.0,
        total_invoice_value=94400.0,
        eway_bill_number="123456789012",
    ),
    # Invoice 2: Missing fields + invalid GSTIN + wrong date format
    Invoice(
        invoice_id="INV-H3-002",
        invoice_number="",  # MISSING
        invoice_date="25/03/2025",  # WRONG format
        supplier_name="Anonymous Supplier",
        supplier_gstin="INVALIDGSTIN",  # INVALID
        supplier_state_code="27",
        recipient_name="",  # MISSING
        recipient_gstin="27BCDEF1234H1Z2",
        recipient_state_code="27",
        place_of_supply="27",
        supply_type="B2B",
        line_items=[
            InvoiceLineItem(
                description="Office supplies",
                hsn_code="4820",
                quantity=100,
                unit_price=50.0,
                taxable_value=5000.0,
                tax_rate=12.0,
                tax_type=TaxType.CGST_SGST,
                cgst_amount=300.0,
                sgst_amount=300.0,
                total_amount=5600.0,
            ),
        ],
        total_taxable_value=5000.0,
        total_tax=600.0,
        total_invoice_value=5600.0,
    ),
    # Invoice 3: Inter-state using CGST+SGST + arithmetic error + missing eway
    Invoice(
        invoice_id="INV-H3-003",
        invoice_number="INV/2025/0703",
        invoice_date="2025-03-26",
        supplier_name="Gujarat Chemicals",
        supplier_gstin="24AABCG5678I1Z9",
        supplier_state_code="24",
        recipient_name="MP Industries",
        recipient_gstin="23BCDEM9012J1Z6",
        recipient_state_code="23",
        place_of_supply="23",  # MP — inter-state
        supply_type="B2B",
        line_items=[
            InvoiceLineItem(
                description="Industrial solvents",
                hsn_code="2201",
                quantity=500,
                unit_price=200.0,
                taxable_value=100000.0,
                tax_rate=18.0,
                tax_type=TaxType.CGST_SGST,  # WRONG: inter-state → IGST
                cgst_amount=9000.0,
                sgst_amount=9000.0,
                igst_amount=0.0,
                total_amount=118000.0,
            ),
            InvoiceLineItem(
                description="Lab equipment",
                hsn_code="9018",
                quantity=10,
                unit_price=25000.0,
                taxable_value=250000.0,
                tax_rate=12.0,
                tax_type=TaxType.CGST_SGST,  # WRONG: inter-state → IGST
                cgst_amount=15000.0,
                sgst_amount=15000.0,
                igst_amount=0.0,
                total_amount=275000.0,  # WRONG: 250000+30000=280000
            ),
        ],
        total_taxable_value=350000.0,
        total_tax=48000.0,
        total_invoice_value=398000.0,
        eway_bill_number="",  # MISSING
    ),
    # Invoice 4: RCM violation + composition scheme inter-state
    Invoice(
        invoice_id="INV-H3-004",
        invoice_number="INV/2025/0704",
        invoice_date="2025-03-27",
        supplier_name="Rural Services Co",
        supplier_gstin="33AABCR1234K1Z3",
        supplier_state_code="33",  # TN
        recipient_name="Kerala Enterprises",
        recipient_gstin="32BCDEF5678L1Z7",
        recipient_state_code="32",
        place_of_supply="32",  # KL — inter-state (VIOLATION for composition)
        supply_type="B2B",
        is_composition_scheme=True,
        reverse_charge=True,  # VIOLATION: composition + RCM
        line_items=[
            InvoiceLineItem(
                description="Rental service",
                hsn_code="9973",  # RCM applicable
                quantity=1,
                unit_price=60000.0,
                taxable_value=60000.0,
                tax_rate=18.0,  # WRONG: composition max 5%
                tax_type=TaxType.IGST,
                igst_amount=10800.0,
                total_amount=70800.0,
            ),
        ],
        total_taxable_value=60000.0,
        total_tax=10800.0,
        total_invoice_value=70800.0,
        eway_bill_number="",  # MISSING for inter-state > 50K
    ),
    # Invoice 5: Duplicate of Invoice 1's number from same "supplier"
    # but actually different supplier — NOT a duplicate (tests false positive)
    Invoice(
        invoice_id="INV-H3-005",
        invoice_number="INV/2025/0701",  # Same number as INV-H3-001 but DIFFERENT supplier
        invoice_date="2025-03-28",
        supplier_name="Kolkata Distributors",
        supplier_gstin="19AABCK9012M1Z1",  # Different supplier
        supplier_state_code="19",
        recipient_name="Bihar Retailers",
        recipient_gstin="10BCDEF3456N1Z8",
        recipient_state_code="10",
        place_of_supply="10",
        supply_type="B2B",
        line_items=[
            InvoiceLineItem(
                description="Notebooks and registers",
                hsn_code="4820",
                quantity=1000,
                unit_price=30.0,
                taxable_value=30000.0,
                tax_rate=12.0,
                tax_type=TaxType.IGST,
                igst_amount=3600.0,
                total_amount=33600.0,
            ),
        ],
        total_taxable_value=30000.0,
        total_tax=3600.0,
        total_invoice_value=33600.0,
        eway_bill_number="987654321012",
    ),
]


def get_hard_task(task_id: str) -> tuple[TaskInfo, list[Invoice], list[GroundTruthIssue]]:
    """Return task info, invoices, and ground truth issues for a hard task."""
    tasks = {
        "hard_1": (TASK_H1_INFO, TASK_H1_INVOICES),
        "hard_2": (TASK_H2_INFO, TASK_H2_INVOICES),
        "hard_3": (TASK_H3_INFO, TASK_H3_INVOICES),
    }

    if task_id not in tasks:
        raise ValueError(f"Unknown hard task: {task_id}")

    info, invoices = tasks[task_id]
    ground_truth = run_batch_validation(invoices)

    return info, invoices, ground_truth
