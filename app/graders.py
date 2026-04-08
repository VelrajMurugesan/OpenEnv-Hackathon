"""Grading engine — scores agent findings against ground truth."""

from __future__ import annotations

from app.models import GraderResult, GroundTruthIssue

# OpenEnv validator requires task scores to lie strictly within (0, 1) — never 0.0
# and never 1.0. We clamp every grader output into this open interval. SCORE_MAX
# is set to 0.9998 (not 0.9999) so that downstream `:.4f` formatting in the
# inference log can never round up to "1.0000".
SCORE_MIN = 0.0001
SCORE_MAX = 0.9998


def _clamp_score(value: float) -> float:
    """Clamp a raw score into the strict open interval (0, 1)."""
    if value <= SCORE_MIN:
        return SCORE_MIN
    if value >= SCORE_MAX:
        return SCORE_MAX
    return round(value, 4)


def _normalize(text: str) -> str:
    """Lowercase and strip for fuzzy matching."""
    return text.lower().strip()


def _match_finding_to_issue(
    finding: dict,
    issue: GroundTruthIssue,
) -> bool:
    """Check if an agent finding matches a ground truth issue.

    Matching criteria (flexible to allow natural language variation):
    1. invoice_id must match exactly
    2. field must partially match (agent may use slightly different names)
    3. category or description must have keyword overlap
    """
    # Invoice ID must match
    if finding.get("invoice_id", "") != issue.invoice_id:
        return False

    # Field matching — partial/fuzzy
    agent_field = _normalize(finding.get("field", ""))
    truth_field = _normalize(issue.field)

    # Direct match or substring match
    field_match = (
        agent_field == truth_field
        or agent_field in truth_field
        or truth_field in agent_field
        # Handle indexed fields: agent says "tax_rate" matches "line_items[0].tax_rate"
        or agent_field.split(".")[-1] in truth_field
        or truth_field.split(".")[-1] in agent_field
        # Handle common aliases
        or _fields_are_equivalent(agent_field, truth_field)
    )

    if not field_match:
        return False

    # Category or description keyword overlap
    agent_cat = _normalize(finding.get("category", ""))
    agent_desc = _normalize(finding.get("description", ""))
    truth_cat = _normalize(issue.category.value)
    truth_desc = _normalize(issue.description)

    # Category match
    if agent_cat and (agent_cat == truth_cat or agent_cat in truth_cat or truth_cat in agent_cat):
        return True

    # Description keyword overlap — at least 2 significant words must match
    agent_words = set(agent_desc.split()) - {"the", "a", "an", "is", "of", "for", "in", "to", "and", "or"}
    truth_words = set(truth_desc.split()) - {"the", "a", "an", "is", "of", "for", "in", "to", "and", "or"}
    common = agent_words & truth_words
    if len(common) >= 2:
        return True

    # If field matched well and any category/severity info given, accept
    if field_match and (agent_cat or finding.get("severity", "")):
        return True

    return False


def _fields_are_equivalent(f1: str, f2: str) -> bool:
    """Check if two field names refer to the same thing."""
    equivalences = [
        {"supplier_gstin", "gstin", "supplier_gst"},
        {"recipient_gstin", "recipient_gst", "buyer_gstin"},
        {"invoice_number", "inv_number", "invoice_no"},
        {"invoice_date", "date", "inv_date"},
        {"tax_rate", "rate", "gst_rate"},
        {"tax_type", "igst", "cgst", "sgst", "tax_category"},
        {"eway_bill_number", "eway_bill", "e_way_bill", "eway"},
        {"total_invoice_value", "invoice_total", "total_value", "grand_total"},
        {"total_taxable_value", "taxable_total", "subtotal"},
        {"total_tax", "tax_total", "total_gst"},
        {"reverse_charge", "rcm", "reverse_charge_mechanism"},
        {"is_composition_scheme", "composition_scheme", "composition"},
        {"hsn_code", "hsn", "sac_code", "sac"},
        {"place_of_supply", "pos", "supply_place"},
    ]
    for group in equivalences:
        if f1 in group and f2 in group:
            return True
    return False


def grade_findings(
    findings: list[dict],
    ground_truth: list[GroundTruthIssue],
    task_id: str,
) -> GraderResult:
    """Score agent findings against ground truth issues.

    Scoring formula:
    - precision = matched_findings / total_findings (penalizes false positives)
    - recall = matched_issues / total_issues (penalizes missed issues)
    - score = F1 = 2 * (precision * recall) / (precision + recall)

    Severity weighting:
    - critical issues are worth 3x
    - major issues are worth 2x
    - minor issues are worth 1x
    """
    if not ground_truth:
        # No issues to find — score based on false positives
        if not findings:
            return GraderResult(
                task_id=task_id,
                score=_clamp_score(1.0),
                details={
                    "message": "Correctly identified no issues",
                    "precision": 1.0,
                    "recall": 1.0,
                    "true_positives": 0,
                    "false_positives": 0,
                    "missed_issues": 0,
                },
            )
        else:
            # False positives only
            penalty = min(len(findings) * 0.2, 1.0)
            return GraderResult(
                task_id=task_id,
                score=_clamp_score(1.0 - penalty),
                details={
                    "message": f"No real issues but agent flagged {len(findings)} false positives",
                    "precision": 0.0,
                    "recall": 1.0,
                    "true_positives": 0,
                    "false_positives": len(findings),
                    "missed_issues": 0,
                },
            )

    if not findings:
        return GraderResult(
            task_id=task_id,
            score=_clamp_score(0.0),
            details={
                "message": f"Agent found no issues but {len(ground_truth)} exist",
                "precision": 0.0,
                "recall": 0.0,
                "true_positives": 0,
                "false_positives": 0,
                "missed_issues": len(ground_truth),
            },
        )

    # Severity weights
    severity_weight = {"critical": 3.0, "major": 2.0, "minor": 1.0}

    # Match findings to ground truth
    matched_truth_indices: set[int] = set()
    matched_finding_indices: set[int] = set()

    # First pass: try to match each finding to unmatched ground truth
    for f_idx, finding in enumerate(findings):
        for t_idx, issue in enumerate(ground_truth):
            if t_idx in matched_truth_indices:
                continue
            if _match_finding_to_issue(finding, issue):
                matched_truth_indices.add(t_idx)
                matched_finding_indices.add(f_idx)
                break

    # Weighted precision
    tp_weight = sum(
        severity_weight.get(ground_truth[t].severity.value, 1.0)
        for t in matched_truth_indices
    )
    total_finding_weight = sum(
        severity_weight.get(findings[f].get("severity", "minor"), 1.0)
        for f in range(len(findings))
    )
    precision = tp_weight / total_finding_weight if total_finding_weight > 0 else 0.0

    # Weighted recall
    total_truth_weight = sum(
        severity_weight.get(issue.severity.value, 1.0)
        for issue in ground_truth
    )
    recall = tp_weight / total_truth_weight if total_truth_weight > 0 else 0.0

    # F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    score = _clamp_score(f1)

    missed = [
        {
            "invoice_id": ground_truth[i].invoice_id,
            "field": ground_truth[i].field,
            "description": ground_truth[i].description,
            "severity": ground_truth[i].severity.value,
        }
        for i in range(len(ground_truth))
        if i not in matched_truth_indices
    ]

    false_positives = [
        findings[i]
        for i in range(len(findings))
        if i not in matched_finding_indices
    ]

    return GraderResult(
        task_id=task_id,
        score=score,
        details={
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": score,
            "true_positives": len(matched_truth_indices),
            "false_positives": len(false_positives),
            "missed_issues": len(missed),
            "total_ground_truth": len(ground_truth),
            "total_findings": len(findings),
            "missed_details": missed[:10],  # Cap for readability
            "false_positive_details": false_positives[:10],
        },
    )
