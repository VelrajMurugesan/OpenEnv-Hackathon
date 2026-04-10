"""GST validation rules engine.

Every ground-truth issue emitted by this module carries a `legal_reference`
string pointing to the actual Indian GST statute that makes the finding a
violation — the CGST/IGST Act section, the CGST Rule, or the CBIC
notification. This makes the benchmark auditable by real Chartered
Accountants and GST practitioners, and it gives downstream LLMs a
structured signal they can cite in agent outputs.

References used throughout:

  - **CGST Act 2017**, Sections 9, 10, 25, 31
  - **IGST Act 2017**, Sections 7, 8
  - **CGST Rules 2017**, Rules 10, 46, 138
  - **Notification 1/2017-Central Tax (Rate)** — HSN code → tax rate mapping for goods
  - **Notification 11/2017-Central Tax (Rate)** — SAC code → tax rate mapping for services
  - **Notification 13/2017-Central Tax (Rate)** — services liable to Reverse Charge
  - **Notification 12/2018-Central Tax** — E-way bill threshold (INR 50,000)
"""

from __future__ import annotations

import re

from data.hsn_codes import (
    EWAY_BILL_THRESHOLD,
    HSN_DATABASE,
    REVERSE_CHARGE_SERVICES,
    STATE_CODES,
    VALID_TAX_RATES,
)
from app.models import (
    GroundTruthIssue,
    Invoice,
    IssueCategory,
    IssueSeverity,
)


# ── Legal reference lookup ──────────────────────────────────────────────────
#
# Maps (category, field) to the specific Indian GST statute citation. This
# is used by every validator so that judges, auditors, and downstream
# systems can trace any ground-truth finding back to the law that defines
# it. Fallback order: (category, field_prefix) → category alone → generic.

_LEGAL_BY_FIELD: dict[tuple[str, str], str] = {
    # --- mandatory fields (CGST Rule 46 lists required fields on a tax invoice) ---
    (IssueCategory.MISSING_FIELD.value, "invoice_number"):    "CGST Rule 46(b); CGST Act s.31",
    (IssueCategory.MISSING_FIELD.value, "invoice_date"):      "CGST Rule 46(c); CGST Act s.31",
    (IssueCategory.MISSING_FIELD.value, "supplier_name"):     "CGST Rule 46(a)",
    (IssueCategory.MISSING_FIELD.value, "supplier_gstin"):    "CGST Rule 46(a); CGST Act s.25",
    (IssueCategory.MISSING_FIELD.value, "recipient_name"):    "CGST Rule 46(e)",
    (IssueCategory.MISSING_FIELD.value, "recipient_gstin"):   "CGST Rule 46(f); CGST Act s.25",
    (IssueCategory.MISSING_FIELD.value, "place_of_supply"):   "CGST Rule 46(h); IGST Act s.10-13",
    (IssueCategory.MISSING_FIELD.value, "line_items"):        "CGST Rule 46(g)",
    (IssueCategory.MISSING_FIELD.value, "hsn_code"):          "CGST Rule 46(g); Notification 78/2020",

    # --- format violations ---
    (IssueCategory.INVALID_FORMAT.value, "supplier_gstin"):   "CGST Rule 10; CGST Act s.25(6A)",
    (IssueCategory.INVALID_FORMAT.value, "recipient_gstin"):  "CGST Rule 10; CGST Act s.25(6A)",
    (IssueCategory.INVALID_FORMAT.value, "invoice_date"):     "CGST Rule 46(c)",
    (IssueCategory.INVALID_FORMAT.value, "supplier_state_code"):  "CGST Rule 10; state code schedule",
    (IssueCategory.INVALID_FORMAT.value, "recipient_state_code"): "CGST Rule 10; state code schedule",
    (IssueCategory.INVALID_FORMAT.value, "place_of_supply"):  "CGST Rule 10; state code schedule",
    (IssueCategory.INVALID_FORMAT.value, "hsn_code"):         "CGST Rule 46(g); Customs Tariff Act",
    (IssueCategory.INVALID_FORMAT.value, "eway_bill_number"): "CGST Rule 138(1); 12-digit EBN format",

    # --- inconsistencies ---
    (IssueCategory.INCONSISTENCY.value, "supplier_state_code"):  "CGST Rule 10 — GSTIN prefix must equal declared state code",
    (IssueCategory.INCONSISTENCY.value, "recipient_state_code"): "CGST Rule 10 — GSTIN prefix must equal declared state code",

    # --- compliance violations ---
    (IssueCategory.COMPLIANCE_VIOLATION.value, "eway_bill_number"):    "CGST Rule 138(1); Notification 12/2018 — E-way bill mandatory above INR 50,000",
    (IssueCategory.COMPLIANCE_VIOLATION.value, "reverse_charge"):      "CGST Act s.9(3); Notification 13/2017 — RCM for notified services",
    (IssueCategory.COMPLIANCE_VIOLATION.value, "is_composition_scheme"): "CGST Act s.10(4) — Composition dealer cannot pay tax under RCM",
    (IssueCategory.COMPLIANCE_VIOLATION.value, "supply_type"):         "CGST Act s.10(2)(c) — Composition dealer restricted to intra-state supply",
}

_LEGAL_BY_CATEGORY: dict[str, str] = {
    IssueCategory.MISSING_FIELD.value:        "CGST Rule 46",
    IssueCategory.INVALID_FORMAT.value:       "CGST Rule 46",
    IssueCategory.WRONG_VALUE.value:          "CGST Rule 46; Notification 1/2017 & 11/2017 (Rate)",
    IssueCategory.TAX_MISMATCH.value:         "IGST Act s.7-8; CGST Act s.12-13 — place of supply rules",
    IssueCategory.COMPLIANCE_VIOLATION.value: "CGST Act s.9-10; CGST Rule 138",
    IssueCategory.INCONSISTENCY.value:        "CGST Rule 10",
    IssueCategory.DUPLICATE.value:            "CGST Rule 46(b) — consecutive serial number unique per financial year",
}


def get_legal_reference(category: IssueCategory | str, field: str = "") -> str:
    """Resolve the best-matching legal citation for an issue.

    Tries (category, exact_field) first, then (category, field_tail) for
    nested paths like `line_items[0].cgst_amount`, then falls back to the
    category-level default.
    """
    cat_value = category.value if isinstance(category, IssueCategory) else str(category)
    if field:
        if (cat_value, field) in _LEGAL_BY_FIELD:
            return _LEGAL_BY_FIELD[(cat_value, field)]
        # Handle nested paths: "line_items[0].tax_rate" -> "tax_rate"
        tail = field.split(".")[-1] if "." in field else field
        if (cat_value, tail) in _LEGAL_BY_FIELD:
            return _LEGAL_BY_FIELD[(cat_value, tail)]
    return _LEGAL_BY_CATEGORY.get(cat_value, "Indian GST Act 2017")


def validate_gstin_format(gstin: str) -> bool:
    """Validate GSTIN format: 2-digit state + 10-char PAN + entity + Z + checksum."""
    if not gstin or len(gstin) != 15:
        return False
    pattern = r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$"
    return bool(re.match(pattern, gstin))


def extract_state_code_from_gstin(gstin: str) -> str:
    """Extract the 2-digit state code from a GSTIN."""
    if gstin and len(gstin) >= 2:
        return gstin[:2]
    return ""


def validate_hsn_code(hsn: str) -> bool:
    """Check if HSN/SAC code exists in the database."""
    return hsn in HSN_DATABASE


def get_valid_tax_rates_for_hsn(hsn: str) -> list[float]:
    """Return valid tax rates for a given HSN code."""
    if hsn in HSN_DATABASE:
        return HSN_DATABASE[hsn][1]
    return []


def is_interstate(supplier_state: str, place_of_supply: str) -> bool:
    """Determine if the supply is inter-state."""
    return supplier_state != place_of_supply


def validate_invoice_mandatory_fields(invoice: Invoice) -> list[GroundTruthIssue]:
    """Check for missing or empty mandatory fields."""
    issues = []
    mandatory = {
        "invoice_number": "Invoice Number",
        "invoice_date": "Invoice Date",
        "supplier_name": "Supplier Name",
        "supplier_gstin": "Supplier GSTIN",
        "recipient_name": "Recipient Name",
        "place_of_supply": "Place of Supply",
    }

    if invoice.supply_type == "B2B":
        mandatory["recipient_gstin"] = "Recipient GSTIN"

    for field, label in mandatory.items():
        value = getattr(invoice, field, "")
        if not value or str(value).strip() == "":
            issues.append(GroundTruthIssue(
                invoice_id=invoice.invoice_id,
                field=field,
                category=IssueCategory.MISSING_FIELD,
                severity=IssueSeverity.CRITICAL,
                description=f"{label} is missing",
                expected_value="non-empty value",
                actual_value="",
            ))

    if not invoice.line_items:
        issues.append(GroundTruthIssue(
            invoice_id=invoice.invoice_id,
            field="line_items",
            category=IssueCategory.MISSING_FIELD,
            severity=IssueSeverity.CRITICAL,
            description="Invoice has no line items",
            expected_value="at least one line item",
            actual_value="0 items",
        ))

    return issues


def validate_gstin_formats(invoice: Invoice) -> list[GroundTruthIssue]:
    """Validate GSTIN format for supplier and recipient."""
    issues = []

    if invoice.supplier_gstin and not validate_gstin_format(invoice.supplier_gstin):
        issues.append(GroundTruthIssue(
            invoice_id=invoice.invoice_id,
            field="supplier_gstin",
            category=IssueCategory.INVALID_FORMAT,
            severity=IssueSeverity.CRITICAL,
            description=f"Supplier GSTIN '{invoice.supplier_gstin}' has invalid format",
            expected_value="15-char GSTIN: SS PPPPP NNNN X NZC",
            actual_value=invoice.supplier_gstin,
        ))

    if invoice.recipient_gstin and not validate_gstin_format(invoice.recipient_gstin):
        issues.append(GroundTruthIssue(
            invoice_id=invoice.invoice_id,
            field="recipient_gstin",
            category=IssueCategory.INVALID_FORMAT,
            severity=IssueSeverity.CRITICAL,
            description=f"Recipient GSTIN '{invoice.recipient_gstin}' has invalid format",
            expected_value="15-char GSTIN: SS PPPPP NNNN X NZC",
            actual_value=invoice.recipient_gstin,
        ))

    return issues


def validate_gstin_state_consistency(invoice: Invoice) -> list[GroundTruthIssue]:
    """Check if GSTIN state codes match the declared state codes."""
    issues = []

    if invoice.supplier_gstin and len(invoice.supplier_gstin) >= 2:
        gstin_state = extract_state_code_from_gstin(invoice.supplier_gstin)
        if gstin_state != invoice.supplier_state_code:
            issues.append(GroundTruthIssue(
                invoice_id=invoice.invoice_id,
                field="supplier_state_code",
                category=IssueCategory.INCONSISTENCY,
                severity=IssueSeverity.MAJOR,
                description=(
                    f"Supplier GSTIN state code ({gstin_state}) doesn't match "
                    f"declared state code ({invoice.supplier_state_code})"
                ),
                expected_value=gstin_state,
                actual_value=invoice.supplier_state_code,
            ))

    if invoice.recipient_gstin and len(invoice.recipient_gstin) >= 2:
        gstin_state = extract_state_code_from_gstin(invoice.recipient_gstin)
        if gstin_state != invoice.recipient_state_code:
            issues.append(GroundTruthIssue(
                invoice_id=invoice.invoice_id,
                field="recipient_state_code",
                category=IssueCategory.INCONSISTENCY,
                severity=IssueSeverity.MAJOR,
                description=(
                    f"Recipient GSTIN state code ({gstin_state}) doesn't match "
                    f"declared state code ({invoice.recipient_state_code})"
                ),
                expected_value=gstin_state,
                actual_value=invoice.recipient_state_code,
            ))

    return issues


def validate_tax_rates(invoice: Invoice) -> list[GroundTruthIssue]:
    """Validate that tax rates match HSN codes and are valid GST rates."""
    issues = []
    for idx, item in enumerate(invoice.line_items):
        if item.tax_rate not in VALID_TAX_RATES:
            issues.append(GroundTruthIssue(
                invoice_id=invoice.invoice_id,
                field=f"line_items[{idx}].tax_rate",
                category=IssueCategory.WRONG_VALUE,
                severity=IssueSeverity.CRITICAL,
                description=f"Tax rate {item.tax_rate}% is not a valid GST rate",
                expected_value=f"One of {VALID_TAX_RATES}",
                actual_value=str(item.tax_rate),
            ))

        valid_rates = get_valid_tax_rates_for_hsn(item.hsn_code)
        if valid_rates and item.tax_rate not in valid_rates:
            issues.append(GroundTruthIssue(
                invoice_id=invoice.invoice_id,
                field=f"line_items[{idx}].tax_rate",
                category=IssueCategory.TAX_MISMATCH,
                severity=IssueSeverity.CRITICAL,
                description=(
                    f"Tax rate {item.tax_rate}% doesn't match HSN {item.hsn_code} "
                    f"(valid: {valid_rates})"
                ),
                expected_value=str(valid_rates),
                actual_value=str(item.tax_rate),
            ))

    return issues


def validate_tax_type_and_amounts(invoice: Invoice) -> list[GroundTruthIssue]:
    """Validate IGST vs CGST+SGST based on inter/intra state supply."""
    issues = []
    interstate = is_interstate(invoice.supplier_state_code, invoice.place_of_supply)

    for idx, item in enumerate(invoice.line_items):
        tax_amount = round(item.taxable_value * item.tax_rate / 100, 2)

        if interstate:
            # Should be IGST
            if item.tax_type.value != "IGST":
                issues.append(GroundTruthIssue(
                    invoice_id=invoice.invoice_id,
                    field=f"line_items[{idx}].tax_type",
                    category=IssueCategory.TAX_MISMATCH,
                    severity=IssueSeverity.CRITICAL,
                    description=(
                        f"Inter-state supply should use IGST, not {item.tax_type.value}"
                    ),
                    expected_value="IGST",
                    actual_value=item.tax_type.value,
                ))

            if abs(item.igst_amount - tax_amount) > 0.01:
                issues.append(GroundTruthIssue(
                    invoice_id=invoice.invoice_id,
                    field=f"line_items[{idx}].igst_amount",
                    category=IssueCategory.WRONG_VALUE,
                    severity=IssueSeverity.MAJOR,
                    description=(
                        f"IGST amount {item.igst_amount} doesn't match "
                        f"expected {tax_amount}"
                    ),
                    expected_value=str(tax_amount),
                    actual_value=str(item.igst_amount),
                ))
        else:
            # Should be CGST + SGST
            if item.tax_type.value != "CGST+SGST":
                issues.append(GroundTruthIssue(
                    invoice_id=invoice.invoice_id,
                    field=f"line_items[{idx}].tax_type",
                    category=IssueCategory.TAX_MISMATCH,
                    severity=IssueSeverity.CRITICAL,
                    description=(
                        f"Intra-state supply should use CGST+SGST, not {item.tax_type.value}"
                    ),
                    expected_value="CGST+SGST",
                    actual_value=item.tax_type.value,
                ))

            half_tax = round(tax_amount / 2, 2)
            if abs(item.cgst_amount - half_tax) > 0.01:
                issues.append(GroundTruthIssue(
                    invoice_id=invoice.invoice_id,
                    field=f"line_items[{idx}].cgst_amount",
                    category=IssueCategory.WRONG_VALUE,
                    severity=IssueSeverity.MAJOR,
                    description=(
                        f"CGST amount {item.cgst_amount} doesn't match "
                        f"expected {half_tax} (half of {tax_amount})"
                    ),
                    expected_value=str(half_tax),
                    actual_value=str(item.cgst_amount),
                ))
            if abs(item.sgst_amount - half_tax) > 0.01:
                issues.append(GroundTruthIssue(
                    invoice_id=invoice.invoice_id,
                    field=f"line_items[{idx}].sgst_amount",
                    category=IssueCategory.WRONG_VALUE,
                    severity=IssueSeverity.MAJOR,
                    description=(
                        f"SGST amount {item.sgst_amount} doesn't match "
                        f"expected {half_tax} (half of {tax_amount})"
                    ),
                    expected_value=str(half_tax),
                    actual_value=str(item.sgst_amount),
                ))

    return issues


def validate_arithmetic(invoice: Invoice) -> list[GroundTruthIssue]:
    """Validate all arithmetic: line totals, tax totals, invoice total."""
    issues = []

    computed_taxable = 0.0
    computed_tax = 0.0

    for idx, item in enumerate(invoice.line_items):
        expected_taxable = round(item.quantity * item.unit_price, 2)
        if abs(item.taxable_value - expected_taxable) > 0.01:
            issues.append(GroundTruthIssue(
                invoice_id=invoice.invoice_id,
                field=f"line_items[{idx}].taxable_value",
                category=IssueCategory.WRONG_VALUE,
                severity=IssueSeverity.MAJOR,
                description=(
                    f"Taxable value {item.taxable_value} doesn't match "
                    f"qty({item.quantity}) x price({item.unit_price}) = {expected_taxable}"
                ),
                expected_value=str(expected_taxable),
                actual_value=str(item.taxable_value),
            ))

        expected_tax = round(item.taxable_value * item.tax_rate / 100, 2)
        actual_tax = item.igst_amount + item.cgst_amount + item.sgst_amount
        if abs(actual_tax - expected_tax) > 0.01:
            issues.append(GroundTruthIssue(
                invoice_id=invoice.invoice_id,
                field=f"line_items[{idx}].tax_amounts",
                category=IssueCategory.WRONG_VALUE,
                severity=IssueSeverity.MAJOR,
                description=(
                    f"Total tax amount {actual_tax} doesn't match "
                    f"expected {expected_tax} ({item.tax_rate}% of {item.taxable_value})"
                ),
                expected_value=str(expected_tax),
                actual_value=str(actual_tax),
            ))

        expected_total = round(item.taxable_value + expected_tax, 2)
        if abs(item.total_amount - expected_total) > 0.01:
            issues.append(GroundTruthIssue(
                invoice_id=invoice.invoice_id,
                field=f"line_items[{idx}].total_amount",
                category=IssueCategory.WRONG_VALUE,
                severity=IssueSeverity.MAJOR,
                description=(
                    f"Line total {item.total_amount} doesn't match "
                    f"taxable({item.taxable_value}) + tax({expected_tax}) = {expected_total}"
                ),
                expected_value=str(expected_total),
                actual_value=str(item.total_amount),
            ))

        computed_taxable += item.taxable_value
        computed_tax += actual_tax

    computed_taxable = round(computed_taxable, 2)
    computed_tax = round(computed_tax, 2)

    if abs(invoice.total_taxable_value - computed_taxable) > 0.01:
        issues.append(GroundTruthIssue(
            invoice_id=invoice.invoice_id,
            field="total_taxable_value",
            category=IssueCategory.WRONG_VALUE,
            severity=IssueSeverity.MAJOR,
            description=(
                f"Invoice total taxable {invoice.total_taxable_value} doesn't match "
                f"sum of line items {computed_taxable}"
            ),
            expected_value=str(computed_taxable),
            actual_value=str(invoice.total_taxable_value),
        ))

    if abs(invoice.total_tax - computed_tax) > 0.01:
        issues.append(GroundTruthIssue(
            invoice_id=invoice.invoice_id,
            field="total_tax",
            category=IssueCategory.WRONG_VALUE,
            severity=IssueSeverity.MAJOR,
            description=(
                f"Invoice total tax {invoice.total_tax} doesn't match "
                f"sum of line taxes {computed_tax}"
            ),
            expected_value=str(computed_tax),
            actual_value=str(invoice.total_tax),
        ))

    expected_invoice_total = round(computed_taxable + computed_tax, 2)
    if abs(invoice.total_invoice_value - expected_invoice_total) > 0.01:
        issues.append(GroundTruthIssue(
            invoice_id=invoice.invoice_id,
            field="total_invoice_value",
            category=IssueCategory.WRONG_VALUE,
            severity=IssueSeverity.MAJOR,
            description=(
                f"Invoice total {invoice.total_invoice_value} doesn't match "
                f"taxable({computed_taxable}) + tax({computed_tax}) = {expected_invoice_total}"
            ),
            expected_value=str(expected_invoice_total),
            actual_value=str(invoice.total_invoice_value),
        ))

    return issues


def validate_eway_bill(invoice: Invoice) -> list[GroundTruthIssue]:
    """Check e-way bill requirement for inter-state supplies above threshold."""
    issues = []
    interstate = is_interstate(invoice.supplier_state_code, invoice.place_of_supply)

    if interstate and invoice.total_invoice_value > EWAY_BILL_THRESHOLD:
        if not invoice.eway_bill_number or invoice.eway_bill_number.strip() == "":
            issues.append(GroundTruthIssue(
                invoice_id=invoice.invoice_id,
                field="eway_bill_number",
                category=IssueCategory.COMPLIANCE_VIOLATION,
                severity=IssueSeverity.CRITICAL,
                description=(
                    f"E-way bill required for inter-state supply above "
                    f"INR {EWAY_BILL_THRESHOLD} (invoice value: {invoice.total_invoice_value})"
                ),
                expected_value="valid e-way bill number",
                actual_value="missing",
            ))
        elif len(invoice.eway_bill_number) != 12 or not invoice.eway_bill_number.isdigit():
            issues.append(GroundTruthIssue(
                invoice_id=invoice.invoice_id,
                field="eway_bill_number",
                category=IssueCategory.INVALID_FORMAT,
                severity=IssueSeverity.MAJOR,
                description="E-way bill number must be 12 digits",
                expected_value="12-digit number",
                actual_value=invoice.eway_bill_number,
            ))

    return issues


def validate_reverse_charge(invoice: Invoice) -> list[GroundTruthIssue]:
    """Validate reverse charge mechanism applicability."""
    issues = []

    for idx, item in enumerate(invoice.line_items):
        is_rcm_service = item.hsn_code in REVERSE_CHARGE_SERVICES
        if is_rcm_service and not invoice.reverse_charge:
            issues.append(GroundTruthIssue(
                invoice_id=invoice.invoice_id,
                field="reverse_charge",
                category=IssueCategory.COMPLIANCE_VIOLATION,
                severity=IssueSeverity.CRITICAL,
                description=(
                    f"HSN/SAC {item.hsn_code} ({item.description}) is subject to "
                    f"Reverse Charge Mechanism but invoice not marked as RCM"
                ),
                expected_value="true",
                actual_value="false",
            ))

    if invoice.reverse_charge and invoice.is_composition_scheme:
        issues.append(GroundTruthIssue(
            invoice_id=invoice.invoice_id,
            field="is_composition_scheme",
            category=IssueCategory.COMPLIANCE_VIOLATION,
            severity=IssueSeverity.CRITICAL,
            description="Composition scheme dealers cannot issue reverse charge invoices",
            expected_value="false",
            actual_value="true",
        ))

    return issues


def validate_composition_scheme(invoice: Invoice) -> list[GroundTruthIssue]:
    """Validate composition scheme restrictions."""
    issues = []

    if invoice.is_composition_scheme:
        interstate = is_interstate(invoice.supplier_state_code, invoice.place_of_supply)
        if interstate:
            issues.append(GroundTruthIssue(
                invoice_id=invoice.invoice_id,
                field="supply_type",
                category=IssueCategory.COMPLIANCE_VIOLATION,
                severity=IssueSeverity.CRITICAL,
                description="Composition scheme dealers cannot make inter-state supplies",
                expected_value="intra-state only",
                actual_value="inter-state",
            ))

        for idx, item in enumerate(invoice.line_items):
            if item.tax_rate > 5.0:
                issues.append(GroundTruthIssue(
                    invoice_id=invoice.invoice_id,
                    field=f"line_items[{idx}].tax_rate",
                    category=IssueCategory.COMPLIANCE_VIOLATION,
                    severity=IssueSeverity.MAJOR,
                    description=(
                        f"Composition scheme tax rate cannot exceed 5%, "
                        f"found {item.tax_rate}%"
                    ),
                    expected_value="<= 5.0",
                    actual_value=str(item.tax_rate),
                ))

    return issues


def validate_hsn_codes(invoice: Invoice) -> list[GroundTruthIssue]:
    """Validate HSN/SAC codes exist."""
    issues = []
    for idx, item in enumerate(invoice.line_items):
        if not item.hsn_code or item.hsn_code.strip() == "":
            issues.append(GroundTruthIssue(
                invoice_id=invoice.invoice_id,
                field=f"line_items[{idx}].hsn_code",
                category=IssueCategory.MISSING_FIELD,
                severity=IssueSeverity.MAJOR,
                description="HSN/SAC code is missing",
                expected_value="valid HSN/SAC code",
                actual_value="",
            ))
        elif not validate_hsn_code(item.hsn_code):
            issues.append(GroundTruthIssue(
                invoice_id=invoice.invoice_id,
                field=f"line_items[{idx}].hsn_code",
                category=IssueCategory.INVALID_FORMAT,
                severity=IssueSeverity.MAJOR,
                description=f"HSN/SAC code '{item.hsn_code}' is not recognized",
                expected_value="valid HSN/SAC from database",
                actual_value=item.hsn_code,
            ))

    return issues


def validate_state_codes(invoice: Invoice) -> list[GroundTruthIssue]:
    """Validate state codes are valid Indian state codes."""
    issues = []
    for field_name, label in [
        ("supplier_state_code", "Supplier state code"),
        ("recipient_state_code", "Recipient state code"),
        ("place_of_supply", "Place of supply"),
    ]:
        code = getattr(invoice, field_name, "")
        if code and code not in STATE_CODES:
            issues.append(GroundTruthIssue(
                invoice_id=invoice.invoice_id,
                field=field_name,
                category=IssueCategory.INVALID_FORMAT,
                severity=IssueSeverity.MAJOR,
                description=f"{label} '{code}' is not a valid Indian state code",
                expected_value=f"valid state code (01-38)",
                actual_value=code,
            ))

    return issues


def validate_invoice_date(invoice: Invoice) -> list[GroundTruthIssue]:
    """Validate invoice date format."""
    issues = []
    date_pattern = r"^\d{4}-\d{2}-\d{2}$"
    if not re.match(date_pattern, invoice.invoice_date):
        issues.append(GroundTruthIssue(
            invoice_id=invoice.invoice_id,
            field="invoice_date",
            category=IssueCategory.INVALID_FORMAT,
            severity=IssueSeverity.MINOR,
            description=f"Invoice date '{invoice.invoice_date}' is not in YYYY-MM-DD format",
            expected_value="YYYY-MM-DD",
            actual_value=invoice.invoice_date,
        ))
    return issues


def detect_duplicate_invoices(invoices: list[Invoice]) -> list[GroundTruthIssue]:
    """Detect duplicate invoice numbers from the same supplier."""
    issues = []
    seen: dict[str, str] = {}  # (supplier_gstin, invoice_number) → invoice_id

    for inv in invoices:
        key = f"{inv.supplier_gstin}|{inv.invoice_number}"
        if key in seen:
            issues.append(GroundTruthIssue(
                invoice_id=inv.invoice_id,
                field="invoice_number",
                category=IssueCategory.DUPLICATE,
                severity=IssueSeverity.CRITICAL,
                description=(
                    f"Duplicate invoice number '{inv.invoice_number}' from "
                    f"supplier {inv.supplier_gstin} (also in {seen[key]})"
                ),
                expected_value="unique invoice number per supplier",
                actual_value=inv.invoice_number,
            ))
        else:
            seen[key] = inv.invoice_id

    return issues


# ── Aggregated Validators ───────────────────────────────────────────────────


def _annotate_with_legal_refs(issues: list[GroundTruthIssue]) -> list[GroundTruthIssue]:
    """Populate the `legal_reference` field on any issue that doesn't already
    carry one. This lets the individual validators stay concise while every
    ground-truth issue still carries an auditable citation."""
    for issue in issues:
        if not issue.legal_reference:
            issue.legal_reference = get_legal_reference(issue.category, issue.field)
    return issues


def run_easy_validation(invoice: Invoice) -> list[GroundTruthIssue]:
    """Easy: mandatory fields + GSTIN format + HSN + basic tax rate checks."""
    issues = []
    issues.extend(validate_invoice_mandatory_fields(invoice))
    issues.extend(validate_gstin_formats(invoice))
    issues.extend(validate_hsn_codes(invoice))
    issues.extend(validate_tax_rates(invoice))
    issues.extend(validate_invoice_date(invoice))
    issues.extend(validate_state_codes(invoice))
    return _annotate_with_legal_refs(issues)


def run_medium_validation(invoice: Invoice) -> list[GroundTruthIssue]:
    """Medium: easy + tax type, arithmetic, GSTIN state consistency, e-way bill."""
    issues = run_easy_validation(invoice)
    issues.extend(validate_tax_type_and_amounts(invoice))
    issues.extend(validate_arithmetic(invoice))
    issues.extend(validate_gstin_state_consistency(invoice))
    issues.extend(validate_eway_bill(invoice))
    return _annotate_with_legal_refs(issues)


def run_hard_validation(invoice: Invoice) -> list[GroundTruthIssue]:
    """Hard: medium + reverse charge, composition scheme."""
    issues = run_medium_validation(invoice)
    issues.extend(validate_reverse_charge(invoice))
    issues.extend(validate_composition_scheme(invoice))
    return _annotate_with_legal_refs(issues)


def run_batch_validation(invoices: list[Invoice]) -> list[GroundTruthIssue]:
    """Full batch audit: hard validation on each + cross-invoice duplicate check."""
    issues = []
    for inv in invoices:
        issues.extend(run_hard_validation(inv))
    issues.extend(detect_duplicate_invoices(invoices))
    return _annotate_with_legal_refs(issues)
