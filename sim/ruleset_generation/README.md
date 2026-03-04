# Ruleset Generation

LLM-based clinical trial simulation rule generation system. An AI agent gathers evidence from **10 biomedical databases** and synthesizes structured JSON rule sets that parameterize clinical trial simulations for any drug-indication pair.

Each generated rule set is a complete simulation model: regimen dosing, patient demographics, multi-system adverse events with severity distributions and cascade triggers, comorbidity risk modifiers, and efficacy endpoints (ORR, PFS, OS).

## Quick Start

```bash
cd ruleset_generation

# Full setup: Python env + download all data (~3.5 GB)
bash setup.sh

# Or quick setup if data/ is already on disk
bash setup.sh --no-data
```

Set your Gemini API key, then generate:

```bash
export RULE_ENGINE_LLM_API_KEY="your-gemini-api-key"

# Generate a rule set for any drug-indication pair
python -m rule_engine generate-single "Etoposide+Cisplatin" "Small Cell Lung Cancer" \
    -o output --multi-stage

# Combination therapy (join drugs with +)
python -m rule_engine generate-single "Paclitaxel+Carboplatin+Bevacizumab" \
    "Non-Small Cell Lung Cancer" -o output --multi-stage
```

Each drug takes ~2-3 minutes with Gemini 2.0 Flash.

## Output Format

Output is written as a split-file pair per drug:

```
output/
  etoposide+cisplatin_small_cell_lung_cancer/
    base.json                  # Full simulation model (~35 KB)
    iv_combination.json        # Route-specific overlay (~0.5 KB)
```

`base.json` contains all simulation parameters:

| Section | Description |
|---------|-------------|
| `drug_name`, `indication` | Drug(s) and target disease |
| `trial_design` | Cycle length |
| `demographics` | Age (normal dist), sex ratio, race, ECOG PS distributions |
| `comorbidities` | Conditions with base probabilities and AE risk modifiers |
| `disease_baseline` | Tumor sites, response distribution, baseline lab values |
| `ae_profile` | Adverse events: incidence (0-1), grade distribution, onset/duration (lognormal), cascade triggers |
| `efficacy` | ORR, complete response rate, PFS/OS (exponential distributions) |
| `administration_schedule` | Per-drug: dose, route, cycle days |
| `dose_modification_rules` | Grade-based dose reductions and holds |
| `ae_cascade_rules` | Cross-AE trigger relationships |
| `supportive_care_rules` | G-CSF, antiemetics, etc. |
| `lab_reference_ranges` | Normal ranges in SI units |
| `mortality_model` | Death channel proportions |
| `ecog_model` | PS transition probabilities |
| `disposition_model` | Withdrawal/discontinuation hazards |

The type overlay (e.g., `iv_combination.json`) adds route-specific fields like `infusion_duration_minutes` (IV), `injection_volume_ml` (SC), or `daily_dosing_schedule` (oral).

## Data Sources

The pipeline queries **10 biomedical databases** to ground every clinical parameter in real evidence:

| # | Source | Type | What It Provides |
|---|--------|------|------------------|
| 1 | **DailyMed** | API | FDA-approved drug labels — AE incidence tables, boxed warnings, dosage |
| 2 | **ClinicalTrials.gov** | API | Trial demographics, eligibility, endpoints, enrollment, reported AEs |
| 3 | **OnSIDES** | Local DB | 7.1M validated drug-ADE pairs extracted from 51,460 FDA labels (PubMedBERT, F1=0.90) |
| 4 | **OpenFDA / FAERS** | API | Post-market adverse event reports with time-to-onset data |
| 5 | **PrimeKG** | Local CSV | Knowledge graph: drug-disease, drug-target, protein-protein relationships (Harvard Dataverse) |
| 6 | **DrugBank** | Local CSV | Drug-target binding, pharmacological properties (extracted from PrimeKG) |
| 7 | **PubChem** | API | Chemical structure, molecular properties, Lipinski violations |
| 8 | **ChEMBL** | API | Bioactivity data, mechanisms of action |
| 9 | **PubMed** | API | Literature co-occurrence scoring for drug-indication pairs |
| 10 | **Project Data Sphere** | Remote (credentials) | Patient-level data from 255+ oncology trials (250K+ patients) |

### Local Data (downloaded by `setup.sh`)

| Dataset | Size | Source |
|---------|------|--------|
| PrimeKG (raw + processed) | ~925 MB | Harvard Dataverse (public) |
| DrugBank CSVs | ~170 MB | Extracted from PrimeKG |
| OnSIDES SQLite DB | ~2 GB | [Tatonetti Lab, v3.1.0](https://github.com/tatonetti-lab/onsides) (public) |
| PDS trial data | ~400 MB | Project Data Sphere (account required) |

### Project Data Sphere (Optional)

[Project Data Sphere](https://projectdatasphere.org) provides patient-level clinical trial data from 255+ oncology trials. Anyone with a PDS account can use it — register for free at the link above.

To include PDS data during setup, set your credentials before running:

```bash
export RULE_ENGINE_PDS_USERNAME="your-email@example.com"
export RULE_ENGINE_PDS_PASSWORD="your-password"
bash setup.sh
```

PDS data improves accuracy for SCLC/NSCLC drugs by providing real patient demographics, AE profiles, and efficacy data at the individual patient level (vs. aggregate summaries from ClinicalTrials.gov). The pipeline works without PDS — it falls back to the other 9 sources.

## Validation & Evaluation

```bash
# Validate output against JSON schemas (base + route-type)
python scripts/validate_target_schema.py output/

# Check for hallucination patterns (6 detectors)
python scripts/analyze_hallucinations.py output/

# Compare against ground truth (9 dimensions, 7 drugs)
python scripts/compare_all_gt.py
```

### Ground Truth

Hand-curated rule sets for 7 SCLC/NSCLC drugs are in `ground_truth/`:

| # | Drug(s) | Indication | Schema Type |
|---|---------|------------|-------------|
| 1 | Darbepoetin alfa | SCLC | SC monotherapy |
| 2 | Etoposide + Cisplatin | SCLC | IV combination |
| 3 | Paclitaxel + Cisplatin + Etoposide | SCLC | IV combination |
| 4 | Carboplatin + Etoposide | SCLC | IV combination |
| 6 | Paclitaxel + Carboplatin + Bevacizumab | NSCLC | IV combination |
| 7 | Paclitaxel + Carboplatin | NSCLC | IV combination |
| 8 | Gemcitabine + Cisplatin | Squamous NSCLC | IV combination |

Current pipeline accuracy: **80.5%** across 9 scoring dimensions (Doses, ORR, Age, Sex, ECOG, AE Count, AE Frequency, Top AE Overlap, PFS/OS).

## Schema Types

The system auto-detects schema type from drug count and administration routes:

| Schema | When |
|--------|------|
| `iv_monotherapy` | 1 drug, IV route |
| `iv_combination` | 2+ drugs, all IV |
| `oral_monotherapy` | 1 drug, oral route |
| `oral_iv_combination` | 2+ drugs, oral + IV mix |
| `subcutaneous_monotherapy` | 1 drug, SC route |

Extension schemas (`biomarker_targeted`, `maintenance_therapy`) can overlay any base type.

## Pipeline Architecture

The pipeline runs a **3-stage multi-stage LLM process** with programmatic overrides:

1. **Stage 1** — 5 parallel extraction calls (AE frequency, severity, onset, triggers, demographics)
2. **Stage 2** — Grounding verification (validates Stage 1 against raw evidence)
3. **Stage 3** — Final JSON synthesis combining all extracted data

After LLM synthesis, **5 programmatic overrides** correct common LLM errors using the collected evidence directly:
- Dose correction (DailyMed + CT.gov parsing)
- Efficacy extraction (ORR/PFS/OS from trial outcomes)
- Demographics alignment (age, sex, ECOG from largest trial)
- AE frequency correction (conservative merge: min(DailyMed x0.5, CT.gov))
- AE pruning (removes unevidenced AEs, caps at 30)

A post-generation **validator** detects and auto-corrects hallucination signatures (mechanical severity patterns, implausible onsets, monotonic triggers).

## Requirements

- Python 3.10+
- Gemini API key (Gemini 2.0 Flash via OpenAI-compatible endpoint)
- ~4 GB disk for data files (downloaded by `setup.sh`)
- Internet access for API-based evidence sources

See `requirements.txt` for Python dependencies.

## Project Structure

```
ruleset_generation/
  setup.sh                  # One-click setup (venv + data downloads)
  requirements.txt          # Python dependencies
  rule_engine/              # Pipeline code
    agent_multistage.py     #   3-stage LLM pipeline + overrides
    evidence/               #   10 evidence source modules
    converter.py            #   Internal format -> target schema
    validator.py            #   Auto-correction + hallucination detection
    pipeline.py             #   Orchestration + output writing
    schema.py               #   Pydantic models (internal format)
    config.py               #   Environment-based configuration
  schema/                   # Target JSON schemas (base + 5 types + 2 extensions)
  output/                   # Generated rule sets (7 drugs)
  ground_truth/             # Hand-curated reference rule sets (7 drugs)
  scripts/                  # Data processing, validation, comparison tools
```
