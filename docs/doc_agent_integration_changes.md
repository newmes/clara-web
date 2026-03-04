# Doc Agent Integration — Change Log

**Date**: 2026-02-18  
**Author**: Integration with ClinicalTrialEngine  
**For**: Hyena (original doc_agent developer)

---

## Overview

`sim/doc_agent/` has been integrated into the ClinicalTrialEngine system. The original functionality (MedWatch 3500A PDF + E2B(R3) XML generation from CRF data) is fully preserved. Changes were made only to adapt the package to work within our project structure.

**Our code (`sim/crf_mapper.py`, `sim/orchestrator.py`, etc.) was NOT modified.** All changes are in `sim/doc_agent/` files and new integration files.

---

## 1. Import Path Changes (all files)

All `from app.*` imports were changed to relative imports within the `src.doc_agent` package.

| File | Before | After |
|------|--------|-------|
| `agent.py` | `from app.agents.doc.code_maps import ...` | `from .code_maps import ...` |
| `agent.py` | `from app.config import Settings` | `from .config import Settings` |
| `agent.py` | `from app.schemas.crf import CRFData` | `from .schemas.crf import CRFData` |
| `e2b_converter.py` | `from app.agents.doc.code_maps import ...` | `from .code_maps import ...` |
| `e2b_converter.py` | `from app.config import Settings` | `from .config import Settings` |
| `medwatch_mapper.py` | `from app.config import Settings` | `from .config import Settings` |
| `medwatch_mapper.py` | `from app.schemas.medwatch import ...` | `from .schemas.medwatch import ...` |
| `medwatch_pdf.py` | `from app.schemas.medwatch import ...` | `from .schemas.medwatch import ...` |
| `meddra_coder.py` | `from app.schemas.agent_output import ...` | `from .schemas.agent_output import ...` |
| `scripts/test_doc_agent.py` | `from app.*` imports | `from src.doc_agent.*` imports |

---

## 2. `__init__.py` Files Added

Created to make `sim/doc_agent/` a proper Python package:
- `sim/doc_agent/__init__.py`
- `sim/doc_agent/schemas/__init__.py`

---

## 3. `config.py` — Settings Updated

**Original** (hardcoded for T-DXd / Enhertu):
```python
DRUG_NAME = "Trastuzumab deruxtecan (T-DXd, Enhertu)"
DRUG_MANUFACTURER = "Daiichi Sankyo / AstraZeneca"
INDICATION = "HER2-positive metastatic breast cancer"
PROTOCOL_NUMBER = "PGM-ADC-2025-001"
SPONSOR_NAME = "PharmaGemma Research Inc."
env_prefix = "PGM_"
```

**After** (aligned with our Padcev+Pembrolizumab simulation):
```python
DRUG_NAME = "Enfortumab vedotin (Padcev)"
DRUG_MANUFACTURER = "Astellas / Seagen"
INDICATION = "Locally advanced or metastatic urothelial carcinoma"
PROTOCOL_NUMBER = "CTE-SIM-2026-001"
SPONSOR_NAME = "ClinicalTrialEngine Research"
env_prefix = "CTE_"
```

Added `GOOGLE_API_KEY` field for Gemini API fallback.

Added `Settings.from_simulation()` class method for creating per-run settings:
```python
settings = Settings.from_simulation(
    drug_name="Custom Drug Name",
    indication="Custom Indication",
    manufacturer="Mfr Name",
)
```

vLLM settings (VLLM_BASE_URL, VLLM_MODEL_ID) are unchanged — MedGemma via vLLM still works.

---

## 4. `meddra_coder.py` — Lookup Path Fixed

**Before**: `Path(__file__).resolve().parents[4] / "data" / "meddra_lookup.json"`  
This pointed outside the project (assumed the original AlphaRaven directory structure).

**After**: `Path(__file__).resolve().parent / "data" / "meddra_lookup.json"`  
Now correctly resolves to `sim/doc_agent/data/meddra_lookup.json`.

---

## 5. New Files Added

### `sim/doc_agent/sim_to_crf_adapter.py`

Bridge between our simulation output and the doc_agent's `CRFData` Pydantic model.

**Key conversions handled:**
| Our System | Doc Agent | Conversion |
|------------|-----------|------------|
| Day numbers (int) | `datetime.date` | `sim_start_date + timedelta(days=day-1)` |
| `AESER: bool` | `AESER: "Y"/"N"` | `"Y" if val else "N"` |
| `AESEV: "LIFE-THREATENING"/"FATAL"` | `AESEV: "MILD"/"MODERATE"/"SEVERE"` | Grade 3+ → "SEVERE" |
| `AESDTH/AESLIFE/...: bool` | Same fields as `"Y"/"N"` | bool → "Y"/"N" |
| Labs as nested dict `{name: {LBORRES, LBORRESU}}` | `list[LBRecord]` | Flatten to records |
| VS as `{SYSBP_VSORRES: val}` | `list[VSRecord]` | Restructure |

**Public API:**
```python
from src.doc_agent.sim_to_crf_adapter import build_crf_for_sae, find_serious_aes

# Find all grade≥3 AEs
saes = find_serious_aes(day_records)

# Build CRFData for a specific SAE
crf = build_crf_for_sae(
    patient_profile=profile,
    day_records=records,
    target_ae_term="neutropenia",
    sim_start_date=date(2026, 1, 6),
)
```

### `sim/doc_agent/service.py`

High-level service API wrapping the full pipeline:

```python
from src.doc_agent.service import generate_documents, generate_all_sae_documents

# Single SAE
result = generate_documents(
    patient_profile=profile_dict,
    day_records=day_record_list,
    target_ae_term="fatigue",
    run_id="run_id_string",
    sim_start_date=date(2026, 1, 6),
    drug_name="Padcev",
    indication="Urothelial carcinoma",
    use_ai=False,  # True enables MedGemma narrative
)
# Returns: {success, patient_id, ae_term, medwatch_pdf_path, e2b_xml_path, meddra, ...}

# All SAEs for a patient
results = generate_all_sae_documents(profile, records, run_id=..., ...)
```

Documents are saved to `data/documents/{run_id}/{patient_id}/`.

---

## 6. Django Integration

### New API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| POST | `/api/doc/generate` | Generate MedWatch 3500A + E2B XML for one SAE |
| GET | `/api/doc/saes/{run_id}/{patient_id}/` | List all SAEs for a patient |
| GET | `/api/doc/download/{run_id}/{patient_id}/{filename}` | Download PDF or XML |
| GET | `/api/doc/list/{run_id}/` | List all generated documents for a run |

**POST `/api/doc/generate` body:**
```json
{
    "run_id": "20260218_...",
    "patient_id": "PT-001",
    "ae_term": "fatigue",
    "ae_day": 65,
    "mode": "natural",
    "use_ai": false
}
```

### UI Integration

The Patient State page (`patient_state.html`) now shows a "📋 MedWatch" button next to each serious AE (grade ≥ 3). Clicking generates the documents and shows download links in a toast notification.

---

## 7. Dependencies Added

```
agno          (Agno workflow framework)
lxml          (E2B XML generation + XSD validation)
reportlab     (PDF overlay generation)
pypdf         (PDF template merging)
pydantic-settings  (env var config)
openai        (OpenAI-compatible client for vLLM)
```

---

## 8. Files NOT Changed

These doc_agent files were NOT modified (no import changes needed):
- `schemas/crf.py` — CRF domain Pydantic models
- `schemas/medwatch.py` — MedWatch section models
- `schemas/agent_output.py` — AI output schemas
- `prompts.py` — LLM prompt templates
- `code_maps.py` — E2B code mapping tables
- `data/meddra_lookup.json` — MedDRA lookup table
- `data/patients/*.json` — Test patient data
- `templates/fda_3500a_template.pdf` — FDA form template
- `templates/ich_icsr_v2.1.xsd` — E2B XSD schema

---

## 9. Test Results

E2E test across 10 patients from a 50-patient Padcev+Pembrolizumab simulation:
- 6 SAEs detected (grade ≥ 3)
- 6 PDF + 6 XML documents generated successfully
- MedDRA coding: lookup table matched 3/6 (Fatigue, Stomatitis), remaining 3 fallback to raw term
- All deterministic pipeline components verified: CRF mapping → MedWatch → E2B XML → PDF overlay
