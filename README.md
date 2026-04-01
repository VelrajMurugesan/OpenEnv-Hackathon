# GST Invoice Compliance Checker — OpenEnv

An **OpenEnv** environment where AI agents audit Indian GST (Goods and Services Tax) invoices for compliance violations.

## Overview

The agent receives GST invoices and must identify compliance issues — missing fields, invalid formats, tax calculation errors, inter/intra-state tax logic violations, e-way bill non-compliance, reverse charge mechanism errors, and composition scheme violations.

**9 tasks** across 3 difficulty levels test progressively complex audit capabilities.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Environment

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### 3. Run Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export OPENAI_API_KEY="your-key"
export ENV_URL="http://localhost:7860"
python inference.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`      | Health check |
| `GET`  | `/info`  | Environment metadata and task list |
| `GET`  | `/tasks` | List all 9 available tasks |
| `POST` | `/reset` | Initialize a session with `{"task_id": "easy_1"}` |
| `GET`  | `/state` | Get current state (invoices, findings, step count) |
| `POST` | `/step`  | Submit an action (flag_issue / approve / submit_report) |
| `GET`  | `/grade` | Get the grader result after submission |

## Action Space

The agent can take three types of actions:

### `flag_issue` — Report a compliance violation
```json
{
  "action": {
    "action": "flag_issue",
    "invoice_id": "INV-E1-001",
    "field": "supplier_gstin",
    "category": "invalid_format",
    "severity": "critical",
    "description": "GSTIN format is invalid"
  }
}
```

### `approve` — Mark an invoice as compliant
```json
{
  "action": {
    "action": "approve",
    "invoice_id": "INV-H2-001"
  }
}
```

### `submit_report` — Finalize the audit
```json
{
  "action": {
    "action": "submit_report"
  }
}
```

## Observation Space

The state returned by `/state` and `/reset`:

```json
{
  "task_id": "easy_1",
  "task_description": "Audit a B2B GST invoice for missing mandatory fields",
  "difficulty": "easy",
  "invoices": [ ... ],
  "findings": [ ... ],
  "step_count": 0,
  "max_steps": 15,
  "done": false,
  "score": null
}
```

## Reward & Scoring

- **Metric**: Weighted F1 score (precision × recall) on a 0.0–1.0 scale
- **Severity weights**: critical = 3×, major = 2×, minor = 1×
- **Partial credit**: Intermediate rewards for each correct finding (+0.05) and penalties for false positives (−0.02) and approving bad invoices (−0.1)

## Tasks

### Easy (3 tasks)
| ID | Name | Invoices | Focus |
|----|------|----------|-------|
| `easy_1` | Missing Field Detection | 1 | Mandatory GST fields |
| `easy_2` | GSTIN Format & Code Validation | 1 | Format rules |
| `easy_3` | Tax Rate vs HSN Code Mismatch | 1 | HSN-rate mapping |

### Medium (3 tasks)
| ID | Name | Invoices | Focus |
|----|------|----------|-------|
| `medium_1` | Inter/Intra-State Tax Logic | 2 | IGST vs CGST+SGST |
| `medium_2` | Arithmetic & GSTIN Consistency | 2 | Math + state codes |
| `medium_3` | E-way Bill & Compliance | 2 | E-way bill rules |

### Hard (3 tasks)
| ID | Name | Invoices | Focus |
|----|------|----------|-------|
| `hard_1` | Reverse Charge & Composition | 3 | RCM + composition scheme |
| `hard_2` | Batch Audit with Duplicates | 4 | Cross-invoice checks |
| `hard_3` | Full Compliance Audit | 5 | All rules combined |

## GST Rules Implemented

1. **Mandatory Fields** — invoice number, date, supplier/recipient info, GSTIN (B2B), place of supply
2. **GSTIN Format** — 15-char format: 2-digit state + PAN + entity + Z + checksum
3. **HSN/SAC Codes** — Validated against 60+ codes with prescribed tax rates
4. **Tax Rates** — Must be 0%, 5%, 12%, 18%, or 28% and match HSN code
5. **Inter/Intra-State Logic** — Different state → IGST; Same state → CGST+SGST
6. **Arithmetic Verification** — Line totals, tax amounts, invoice totals
7. **E-way Bill** — Required for inter-state supply > INR 50,000 (12-digit number)
8. **Reverse Charge** — Specific service codes require RCM marking
9. **Composition Scheme** — No inter-state supply, max 5% tax, no RCM
10. **Duplicate Detection** — Same supplier + same invoice number across batch

## Configuration Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM endpoint URL |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face authentication token |
| `ENV_URL` | Environment server URL (default: `http://localhost:7860`) |

## Deployment

### Docker

```bash
docker build -t gst-compliance-checker .
docker run -p 7860:7860 gst-compliance-checker
```

### Hugging Face Spaces

Push the repo to a Hugging Face Space with Docker SDK. The `Dockerfile` is pre-configured for port 7860.

## Tech Stack

- **Python 3.11** — Runtime
- **FastAPI** — API framework
- **Pydantic v2** — Typed models and validation
- **OpenAI Client** — LLM inference
- **Uvicorn** — ASGI server
- **Docker** — Containerization
