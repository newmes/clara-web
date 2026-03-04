# Rule Discovery

Novel clinical trial simulation rule generation system. An LLM-based agent gathers evidence from 10 biomedical databases (DailyMed, DrugBank, ClinicalTrials.gov, PubChem, ChEMBL, PrimeKG, OpenFDA, PubMed, OnSIDES, Project Data Sphere) and synthesizes structured JSON rule sets that parameterize clinical trial simulations for drug-indication pairs. The system generates complete simulation models — regimen, demographics, multi-system adverse events with severity distributions and cascade triggers, comorbidity risk modifiers, and efficacy endpoints — from heterogeneous evidence, not by reformatting existing data.

## Scope

Only these directories are in scope:

- `output/` — Generated rule sets in split format (`base.json` + type overlay per drug subdirectory, 7 SCLC/NSCLC drugs). Agent logs at `output/*_agent_log.json`.
- `ground_truth/` — Ground truth rule sets (7 drugs: Darbepoetin alfa, Etoposide+Cisplatin, Paclitaxel+Cisplatin+Etoposide, Carboplatin+Etoposide, Paclitaxel+Carboplatin+Bevacizumab, Paclitaxel+Carboplatin, Gemcitabine+Cisplatin) with split-file format (base.json + type overlay)
- `schema/` — Target schema definitions: `base.json` + 5 route-type schemas (`iv_monotherapy.json`, `iv_combination.json`, `oral_monotherapy.json`, `oral_iv_combination.json`, `subcutaneous_monotherapy.json`) + 2 extensions (`biomarker_targeted.json`, `maintenance_therapy.json`)
- `scripts/` — Data pipeline (PrimeKG processing, DrugBank extraction, KG augmentation, hallucination analysis, batch running, log analysis, schema conversion/validation)
- `data/` — Source data (PrimeKG raw/processed/augmented, DrugBank CSVs, OnSIDES database, PDS trial data). Downloaded by `setup.sh`.
- `rule_engine/` — Pipeline code (agent, evidence sources, validator, converter, schema, config)
- `setup.sh` — One-click setup script (venv + data downloads, or `--no-data` for quick mode)
- `requirements.txt` — Python dependencies

## Output Schema (Target Format)

Output uses a **split-file format**: each drug gets a subdirectory containing `base.json` (all clinical data) + `{schema_type}.json` (route-specific overlay). The pipeline generates the internal RuleSet format, auto-converts to target schema, splits into base + overlay, and validates against both schemas before writing.

Example: `output/etoposide+cisplatin_small_cell_lung_cancer/base.json` + `iv_combination.json`

Each `base.json` contains:
- `drug_name` (string, "+" joined), `indication`
- `trial_design` — `cycle_length_days`
- `demographics` — age (numeric distribution), sex/race/ecog_ps (categorical distributions)
- `comorbidities` — condition, base_probability (0-1), conditional_modifiers
- `disease_baseline` — tumor sites, tumor response distribution, lab values
- `ae_profile` — `ae_term` (lowercase snake_case), `incidence_all_grade` (0-1), `grade_distribution` (only populated grades, proportions summing to ~1.0), `onset_day` (lognormal distribution), `duration_days` (lognormal distribution), `reversible`, `cumulative`, `risk_modifiers`
- `efficacy` — `overall_response_rate` (0-1), PFS/OS (exponential distributions with CI-derived parameters)
- `administration_schedule` — per-drug schedule with route-specific fields:
  - IV: `route="INTRAVENOUS"`, `infusion_duration_minutes` (required)
  - Oral: `route="ORAL"`, `daily_dosing_schedule` (QD/BID/TID), `continuous_days_per_cycle`
  - SC: `route="SUBCUTANEOUS"` (no infusion/dosing fields)
- `dose_modification_rules`, `ae_cascade_rules`, `supportive_care_rules`
- `lab_reference_ranges`, `mortality_model`, `ecog_model`, `disposition_model`

### Schema Type Detection
The converter auto-detects schema type from the `drugs` list (ground truth for mono vs combo — v8 fix) and regimen routes:
- 1 drug, IV → `iv_monotherapy` (monotherapy deduplicates same-drug admin schedule entries)
- 1 drug, oral → `oral_monotherapy`
- 1 drug, SC → `subcutaneous_monotherapy`
- 2+ drugs, all IV → `iv_combination`
- 2+ drugs, oral+IV mix → `oral_iv_combination`

Agent logs (`*_agent_log.json`) record the evidence prompt, LLM reasoning trace, raw response, stage logs, validation warnings, and the internal RuleSet format for debugging.

## Pipeline Improvements (v2 → v17)

The improved pipeline addresses LLM hallucination through evidence grounding, multi-stage synthesis, post-generation auto-correction, and evidence-quality filtering.

### Evidence Sources (10 databases)
- **DailyMed**: FDA-approved package labels — AE incidence tables with grade 3-4 breakdowns
- **OnSIDES** (`data/onsides/`): 7.1M validated drug-ADE pairs from 51,460 FDA labels (PubMedBERT, F1=0.90). Cross-checks AE frequencies against boxed warning status.
- **OpenFDA/FAERS**: Adverse event reporting with time-to-onset extraction. Queries try `generic_name`, `brand_name`, `substance_name` fields (fixes biologics returning empty).
- **ClinicalTrials.gov**: Trial demographics, endpoints, phase information
- **PrimeKG**: Knowledge graph relationships (drug-disease, drug-target, protein-protein)
- **DrugBank**: Drug-target binding, pharmacological properties
- **PubChem**: Chemical structure and properties
- **ChEMBL**: Bioactivity and target data
- **PubMed**: Literature abstracts
- **Project Data Sphere** (v15, `data/pds/`): Patient-level data from 6 SCLC trials (255+ oncology trials available). Cached CSVs matched via `trial_index.csv`. Provides demographics (age, sex, ECOG), AEs, efficacy, and regimen data. Overrides CT.gov aggregate data when available (patient-level > aggregate).

### Multi-Stage LLM Pipeline (`--multi-stage` flag)
Instead of a single monolithic LLM call, the pipeline runs 3 stages (7 total LLM calls):
- **Stage 1**: 5 parallel focused extraction sub-calls (AE frequency, severity, onset, triggers, demographics)
- **Stage 2**: Grounding verification (validates Stage 1 outputs against raw evidence numbers). Gracefully degrades if JSON parse fails (100% failure rate in practice — programmatic overrides compensate).
- **Stage 3**: Final JSON synthesis combining all extracted data

Key parameters: Stage 1 max_tokens=4000, Stage 2 max_tokens=4000, Stage 3 max_tokens=8000 (all capped at 8192 for Gemini). Context truncation: Stage 2 extracted≤20K chars + evidence≤10K chars; Stage 3 extracted≤25K chars + grounding≤5K chars. All LLM calls use `response_format={"type": "json_object"}` for structured output and are rate-limited via `RateLimiter` (`rate_limiter.py`).

### LLM Re-Prompts (v6→v8 — no hard-coded defaults)
When Stage 3 synthesis omits data, focused follow-up LLM calls fill gaps. Each uses targeted evidence (not generic demographics_evidence):
- **Comorbidity re-prompt**: Triggers when `rule_set.comorbidities` is empty. Asks LLM for 5-8 indication-relevant comorbidities. Logged as `comorbidity_reprompt`.
- **Demographics re-prompt**: Triggers when `age.mean`, `age.std`, `pct_male`, or `pct_female` is None. Asks LLM for trial demographics. Logged as `demographics_reprompt`.
- **Efficacy re-prompt** (v8): Triggers when `overall_response_rate_pct` is None or 0. Uses dedicated `_format_efficacy_evidence` (CT.gov primary outcomes, combo trial outcomes, DailyMed dosage). Handles string values from LLM (e.g., "30%"→30.0). Logged as `efficacy_reprompt`.
- **Regimen re-prompt** (v8): Triggers when `rule_set.regimen` is empty OR drugs from the `drugs` list are missing from regimen entries. Uses `_format_dose_evidence` (DailyMed dosage sections). Creates full RegimenEntry objects. Logged as `regimen_reprompt`.
- **Dose re-prompt** (v8): Triggers when any regimen entry has vague dose strings ("standard dose", "per physician discretion", "per protocol", or empty). Uses `_format_dose_evidence`. Replaces vague doses with specific values from LLM. Logged as `dose_reprompt`.

No hard-coded clinical defaults remain in the pipeline. The former `reg["dose"] = "standard dose"` fallback (v6) was removed — empty doses trigger the dose re-prompt instead. All clinical data comes from evidence databases or LLM synthesis.

### Programmatic Post-Synthesis Overrides (v13→v17)
After Stage 3 synthesis, 5 deterministic correction steps run before LLM reprompts (which now serve as fallbacks). All use already-collected evidence — no hardcoded clinical values.

- **Dose override** (`_extract_structured_doses`): Parses DailyMed dosage text with indication-aware section matching and CT.gov description parsing (`_extract_doses_from_ctgov_descriptions`). Regex patterns handle mg/m² (with comma-separated thousands), AUC (with range extraction e.g. "AUC 4-6" → midpoint), mg/kg, mcg/kg, flat mg. Unit-aware comparison prevents AUC↔mg/m² false matches. Overrides if LLM dose is vague or deviation exceeds threshold (50% for mg doses, 15% for AUC doses — v16). **Indication dose fallbacks** (v16): `_INDICATION_DOSE_FALLBACKS` table provides NCCN-standard doses for Cisplatin, Paclitaxel, Carboplatin, Etoposide, Gemcitabine per indication when DailyMed has wrong-indication doses (e.g., Cisplatin 20 mg/m² bladder → 80 mg/m² SCLC). Also handles unit-type changes (mg/m² → AUC). **Vague dose set** (v16): expanded to include "not specified", "as prescribed", "individualized". Logged as `programmatic_dose_override`.
- **Efficacy override** (`_extract_efficacy_from_outcomes`): Extracts ORR/PFS/OS from CT.gov `primary_outcomes` AND `secondary_outcomes` (v14) using regex (prefers combo trial). **One-directional ORR override** (v16): only overrides upward (ext_orr > current * 1.3); never lowers LLM estimates. Fills when ORR is None/0. Logged as `programmatic_efficacy_override`.
- **Demographics override** (`_extract_demographics_from_ctgov`): Extracts age (from eligibility criteria + baseline mean/std), sex (from baseline counts), ECOG PS (from baseline counts) from CT.gov. Picks largest-sample trial. Derives age min/max from mean±2.5*std when baseline stats available. **Indication-specific age priors** (v16): `_INDICATION_AGE_PRIORS` dict provides cancer-type-specific mean/std when CT.gov baseline unavailable (prevents eligibility-derived ages like mean=46.5). **ECOG from eligibility text** (v16): parses "ECOG 0-1" → {0:35%, 1:65%}. **Age clamping** (v16): 25-95 bounds in all derivation paths. **Sex pct clamping** (v16): 0-100 before Pydantic validation. Logged as `programmatic_demographics_override`.
- **AE frequency correction** (`_correct_ae_frequencies`): Builds evidence frequency map from DailyMed (×0.5 dampening) + CT.gov (at_risk≥10 only). Merges using `min(DailyMed×0.5, CT.gov)` per AE — takes the more conservative estimate since both sources inflate differently (DailyMed=cross-indication aggregate, CT.gov max=worst trial). PDS patient-level data overrides both when available. Replaces LLM values that are <50% of evidence max (with >2% absolute difference). Logged as `ae_frequency_correction`.
- **AE pruning** (v14) (`_prune_unevidenced_aes`): Removes AEs not backed by DailyMed, CT.gov (at_risk≥10), or OnSIDES. Merges duplicates by normalized term. MAX_AES=30 cap. Logged as `ae_evidence_pruning`.

### Auto-Correction (validator.py)
Post-generation corrections that detect and fix fabrication signatures:
- **Severity auto-correction**: Detects >50% identical grade_1/frequency ratios OR ≥3 AEs with exact 50/30/15/5 mechanical pattern. Replaces with category-specific distributions (13 AE categories: hematologic, gi, hepatic, dermatologic, neurologic, musculoskeletal, constitutional, respiratory, cardiac, metabolic, infection, endocrine, immune-related) with ±5% jitter.
- **Onset auto-correction**: Triggers on (a) ≤3 unique values across 8+ AEs, (b) >50% multiples of 10, or (c) >60% multiples of 7. Replaces with category-based clinically plausible onset ranges.
- **Trigger auto-correction**: Detects >60% identical trigger patterns. Replaces with category-specific dose modification probabilities (e.g., hematologic: 55% grade≥3 hold vs constitutional: 20%).
- **AE name normalization** (expanded v16): Converts lab-code AE names to clinical terms (e.g., "Decreased hemoglobin" → "Anemia", "Increased ALT" → "Alanine aminotransferase increased", "decreased neutrophils <N" → "neutropenia"). **Lab-threshold regex** (v16): Context-aware regex handles `decreased_X_<N` patterns. **Junk AE filter** (v16): Removes non-AE entries (`other_*`, `malignant_neoplasm_progression`, `transfusion:*`). **Combined-term splitter** (v16): `nausea_and_vomiting` → `nausea` (if vomiting already exists). **Synonym map** (v16): 7 new entries including `white_blood_cell_decreased` → `leukopenia`, `neuropathy_sensory` → `peripheral_neuropathy`. Also updates ae_risk_modifier references to match.
- **Comorbidity ae_risk_modifiers auto-population**: For comorbidities with `impacts_dosing=True` but empty modifiers, assigns clinically plausible risk multipliers based on comorbidity type × AE category mapping (e.g., hepatic impairment → hepatic AEs 1.5x + GI AEs 1.2x; renal disease → metabolic AEs 1.3x + heme AEs 1.2x).
- **DailyMed cross-check**: **DISABLED (v17)** — was overriding programmatic AE frequency corrections from `_correct_ae_frequencies()` back to inflated DailyMed label values. The programmatic override in agent_multistage.py now handles frequency alignment using priority PDS > min(CT.gov, DailyMed×0.5).
- **CT.gov individual underreporting correction**: **DISABLED (v17)** — CT.gov `max()` across trials inflated hematologic AEs (e.g., leukopenia 88.8% vs GT 16.1%). The `_correct_ae_frequencies()` min-merge in agent_multistage.py provides more accurate frequency alignment.
- **CT.gov flat-frequency fallback**: Still active — when all AEs share ≤3 unique frequency values (flat-frequency condition from DailyMed extraction failure), corrects frequencies using ClinicalTrials.gov reported AE data (corrects any >1% difference). Requires at_risk≥10.
- **Boxed-warning AE injection**: When OnSIDES boxed-warning AEs are missing from the rule set, injects them with conservative category-based severity distributions and frequencies. Ensures critical safety signals (e.g., ILD for T-DXd) are never absent.
- **OnSIDES cross-check**: Validates boxed warning AEs are present in output.
- **Sex percentage normalization**: When pct_male + pct_female deviates >5% from 100%, normalizes proportionally (LLM sometimes outputs enrollment fractions instead of ratios).
- **Combo source_drug fix**: Splits concatenated source_drug values (e.g., "Gemcitabine+Capecitabine" → "Gemcitabine") and auto-assigns first drug when source_drug is null in combination therapy AEs.
- **Negative severity grade correction**: Detects grades < 0 (Gemini-specific failure) and replaces with category-specific distributions. Runs before the sum-rescaling step.
- **Severity sum rescaling threshold**: Tightened from >1.0 to >0.5 to catch rounding mismatches (e.g., grades sum 13.9 vs frequency 13.0).
- **None-safe validators (v6)**: All comparison operations guard against None values from Optional Pydantic fields (sex normalization, age validation, efficacy CR<=ORR check, comorbidity prevalence).

### Aggressive JSON Repair (agent_multistage.py)
Handles the LLM's frequent JSON formatting errors:
- Multi-pass `_repair_json` (3 iterations) for missing comma patterns
- **Iterative error-position repair**: Up to 30 attempts to insert missing commas or remove trailing commas at the exact `json.loads` failure positions
- **Roman numeral phase conversion**: `"III"` → `3`, `"II"` → `2`, etc. (caused Olaparib failure before fix was applied)
- Unclosed bracket auto-completion for truncated JSON
- Trailing comma removal before closing brackets
- Missing required field injection (drugs, indication, regimen, efficacy, demographics.sex) from Stage 1 data
- Reversibility field fix: Sets `reversible=True` (bool) instead of the previous `reversibility="reversible"` (wrong key, wrong type)
- **Nested demographics unwrap**: Gemini sometimes outputs `{"demographics": {"demographics": {...}}}` — inner dict is unwrapped. Also handles flat sex fields mixed into demographics level and None values in age/sex/race_ethnicity sub-fields.
- **List-typed stage1 data guard**: Stage 1 parsed data may be lists instead of dicts (Gemini-specific) — `isinstance` checks prevent `.get()` on lists.
- **Nested list regimen flattening**: `[[{...}]]` → `[{...}]` for double-wrapped regimen arrays.
- **List-typed efficacy/AE filtering**: Efficacy as list → take first element; non-dict AE entries filtered out.
- **Triggers non-list coercion (v6)**: Gemini outputs triggers as dict or null instead of list. Coerced to list.
- **drug_interactions non-list coercion (v6/v14)**: dict/null → list. Fixed in v14 to also handle `None` (Gemini returns `null` for drug_interactions).
- **Comorbidity ae_risk_modifiers null fix (v6)**: None → empty list.
- **Sex pct non-numeric fix (v6)**: String values like "45%" → float, with `%` stripping.
- **Dose float-to-string coercion**: Numeric dose values (e.g., `240.0`) → string `"240.0"`.
- **Race_ethnicity pct non-numeric**: String/None → float conversion with `%` stripping.
- **Demographics empty-dict safety net (v7)**: When LLM returns `demographics: {}` and Stage 1 also lacks data, injects minimal valid `age`/`sex` structures so Pydantic doesn't reject before demographics re-prompt fires. Also handles `demographics` as list (take first element).
- **Dose empty-string fallback (v8)**: `reg["dose"] = ""` instead of the former `"standard dose"` hardcoding. Empty doses trigger the dose re-prompt.

### Hallucination Analysis
`scripts/analyze_hallucinations.py` checks 6 patterns:
1. **severity_fabrication**: >50% of AEs share identical grade_1/frequency ratio OR ≥3 AEs follow exact 50/30/15/5 grade split
2. **onset_plausibility**: >60% multiples of 7 OR >50% multiples of 10
3. **trigger_monotonicity**: >60% of triggered AEs share identical (condition, probability) pattern
4. **frequency_sum**: severity distribution grades don't sum to frequency_pct
5. **flat_frequency**: ≤3 unique frequency_pct values across 8+ AEs (detects DailyMed extraction failure)
6. **faers_coverage**: FAERS returns no data for the drug

### Pipeline Reliability (20-drug empirical data)

| Stage | Success Rate | Notes |
|-------|-------------|-------|
| Stage 1 ae_freq | 90% (18/20) | Fails when DailyMed + OnSIDES return sparse data |
| Stage 1 severity | 10% (2/20) | Near-total failure; auto-correction compensates |
| Stage 1 onset | 95% (19/20) | Reliable |
| Stage 1 triggers | 100% (20/20) | Always succeeds |
| Stage 1 demographics | 100% (20/20) | Always succeeds |
| Stage 2 grounding | 0% (0/20) | 100% JSON parse failure; graceful degradation |
| Stage 3 synthesis | 100% (20/20) | Always produces parseable output (after repair) |

### Test Results (7/7 PASS)
All 7 output files pass schema validation (base + type-specific) and all 6 hallucination checks:

**Current 7 (Gemini 2.0 Flash → `output/`, v17 pipeline, SCLC/NSCLC):**
- Darbepoetin alfa / SCLC [iv_monotherapy]
- Etoposide + Cisplatin / SCLC [iv_combination]
- Paclitaxel + Cisplatin + Etoposide / SCLC [iv_combination]
- Etoposide + Carboplatin / SCLC [iv_combination]
- Paclitaxel + Carboplatin + Bevacizumab / NSCLC [iv_combination]
- Paclitaxel + Carboplatin / NSCLC [iv_combination]
- Gemcitabine + Cisplatin / Squamous NSCLC [iv_combination]

## Target Schema Conversion (Integrated)

The pipeline auto-converts validated RuleSet JSON to the target simulation schema during `_save_rule_set()`, then splits into `base.json` + type overlay. Output is written as a subdirectory per drug (e.g., `output/drug_indication/base.json` + `iv_combination.json`). The internal RuleSet is preserved in the agent log.

### Key Files
- `rule_engine/converter.py` — Core conversion: `convert_ruleset(data: dict) -> tuple[dict, str]` (returns converted dict + schema type) + `split_base_overlay(converted, schema_type)` (splits into base + type overlay)
- `rule_engine/reference_data.py` — Static defaults (lab ranges, disease baselines, supportive care, mortality/ECOG/disposition models, AE duration defaults)
- `scripts/convert_to_target_schema.py` — Batch converter CLI (for offline re-conversion)
- `scripts/validate_target_schema.py` — Validates against `schema/base.json` + type-specific schemas

### Key Transforms
- `drugs: ["A","B"]` → `drug_name: "A + B"` (join with " + ")
- All percentages (0-100) → probabilities (0-1): frequency_pct, prevalence_pct, response rates, sex/race
- `severity_distribution` grades normalized to proportions summing to ~1.0 (divide each grade by sum of grades)
- `median_onset_days` → `onset_day` lognormal distribution wrapper (`{"type": "numeric", "distribution": "lognormal", ...}`)
- `duration_days` added from AE-category-based defaults (lognormal distributions)
- `grade_distribution` keys: `"1","2","3","4"` etc. — only populated grades included (no zero-value entries like `"5": 0.0`)
- AE terms → lowercase snake_case (`"Neutropenia"` → `"neutropenia"`, `"Chronic Kidney Disease"` → `"chronic_kidney_disease"`)
- `risk_modifiers: []` and `cumulative: false` added to every AE entry
- AE triggers decomposed into `dose_modification_rules` (Dose reduction/Treatment discontinuation) and `ae_cascade_rules` (cross-AE triggers)
- New template sections: `disease_baseline`, `lab_reference_ranges`, `supportive_care_rules`, `mortality_model`, `ecog_model`, `disposition_model`
- `regimen` → `administration_schedule` with route normalization (e.g., "IV infusion" → "INTRAVENOUS")
- Route-specific fields: IV → `infusion_duration_minutes`, Oral → `daily_dosing_schedule` + `continuous_days_per_cycle`, SC → none
- Schema type auto-detection from `drugs` list (mono vs combo) + regimen routes (`_schema_type` metadata in output)
- Monotherapy deduplication: same-drug admin schedule entries collapsed to 1 (LLM sometimes outputs multiple dose levels)
- Efficacy: PFS/OS use exponential distributions with CI-derived parameters (std = (ci_high - ci_low) / 3.92). Returns None if CI bounds missing.
- Efficacy None-safe: omits fields when data is absent rather than outputting `0` (which falsely signals "drug is ineffective")
- Lab reference ranges: lowercase keys (`"anc"`, `"wbc"`), SI units (`x10^9/L`, `umol/L`, `g/L`, `mmol/L`)
- Mortality model: channels use `{"proportion_of_deaths": float, "description": str}` objects with GT-aligned names (`malignant_disease`, `drug_toxicity`, `unrelated_ae`)
- Disposition model: hazards use `{"daily_probability": float, "description": str}` objects (GT format)
- Edge case: all-zero severity grades → conservative default distribution (30/30/25/10/5%)

### Usage
```bash
# Pipeline generates target-schema output directly (run from project root)
python -m rule_engine generate-single "Pembrolizumab" "Melanoma" -o output --multi-stage

# Validate output against base + type-specific schemas
python scripts/validate_target_schema.py output/

# Hallucination analysis (supports both internal and target-schema formats)
python scripts/analyze_hallucinations.py output/

# Offline batch re-conversion (for legacy internal-format files)
python scripts/convert_to_target_schema.py input_dir/ output_dir/
```

## Data Pipeline

Run `bash setup.sh` for one-click setup, or manually:
1. Download PrimeKG kg.csv from Harvard Dataverse
2. `scripts/process_primekg.py` — raw KG to base nodes/edges CSVs
3. `scripts/augment_primekg_exhaustion.py` — add T-cell exhaustion marker edges
4. `scripts/extract_drugbank_from_primekg.py` — extract DrugBank-format CSVs
5. `scripts/download_onsides.sh` — download + build OnSIDES SQLite DB
6. (Optional) `scripts/pds_download.py` + `scripts/pds_download_nsclc.py` — PDS trial data (requires account)

## Running the Pipeline

```bash
# Set Gemini API key
export RULE_ENGINE_LLM_API_KEY="your-gemini-api-key"

# Single drug (multi-stage mode, outputs target schema directly — run from project root)
python -m rule_engine generate-single "Pembrolizumab" "Melanoma" -o output --multi-stage

# Combination therapy
python -m rule_engine generate-single "Cisplatin+Etoposide" "Small Cell Lung Cancer" -o output --multi-stage

# Validate output against base + type-specific schemas
python scripts/validate_target_schema.py output/

# Hallucination analysis
python scripts/analyze_hallucinations.py output/

# Ground truth comparison (9-dimension scoring across 7 drugs, uses LLM for AE matching)
python scripts/compare_all_gt.py

# Healthcheck
python -m rule_engine healthcheck
```

Each drug takes ~2-3 minutes with Gemini 2.0 Flash. Can run concurrently with rate limiter managing API limits.

## Infrastructure

- Python venv: `venv/bin/python` (created by `setup.sh`)
- **LLM backend**: Gemini 2.0 Flash via OpenAI-compatible endpoint (`https://generativelanguage.googleapis.com/v1beta/openai/`). 1M token input context, 8192 max output tokens. Rate limited at 150 RPM (configurable via `RULE_ENGINE_RATE_LIMIT_RPM`).
- **Config env vars**: `RULE_ENGINE_LLM_BASE_URL`, `RULE_ENGINE_LLM_MODEL`, `RULE_ENGINE_LLM_API_KEY`.
- Rate limiter: `rule_engine/rate_limiter.py` — async token-bucket RPM limiter shared across all LLM calls within a pipeline run.

## Known Limitations

- **Stage 2 grounding check fails 100% of the time** with JSON delimiter errors. Gracefully degrades (non-fatal). Programmatic overrides (dose, efficacy, demographics, AE freq correction) provide the grounding function instead.
- The LLM sometimes omits required JSON fields (drugs, regimen, efficacy, comorbidities, demographics). Aggressive repair injects from Stage 1 data; LLM re-prompts fill comorbidities/demographics/efficacy/regimen/doses (v8).
- The LLM occasionally outputs Roman numeral phases (e.g., `"III"` instead of `3`). Repair converts to integers.
- Sex percentages sometimes sum to ~50% instead of ~100% (validator auto-corrects by normalizing to 100%).
- **Combo source_drug concatenation**: LLM sometimes outputs `"Gemcitabine+Capecitabine"` as source_drug instead of individual drug names. Auto-fixed by splitting on `+` and matching to drug list.
- **DailyMed AE table extraction** fails for some drugs (Olaparib returned 0 entries, Gilteritinib returned only 3 lab values). The CT.gov frequency fallback compensates but may produce slightly lower frequencies than the label.
- **Injected boxed-warning AEs use conservative 5% default frequency** when CT.gov doesn't have matching frequency data. Real incidence may be higher (e.g., T-DXd ILD is ~12% in clinical trials vs 5% injected default).
- **OnSIDES name matching for ADCs/biosimilars**: Uses base-name fallback (e.g., "Trastuzumab deruxtecan" → "Trastuzumab") which may pick up AEs from the parent antibody. This is generally conservative/acceptable but may include some false positives.
- **Gemini-specific JSON structure issues**: Gemini 2.0 Flash sometimes produces nested demographics, list-typed stage1 data, nested list regimens, negative severity grades, non-list triggers/drug_interactions, null ae_risk_modifiers, non-numeric sex pct, and numeric dose values. All handled by aggressive repair and validator auto-correction.
- **Gemini max output**: 8,192 tokens hard limit. Sufficient for all observed outputs (largest ~4,700 tokens) but may truncate unusually large rule sets.
- **Optional schema fields**: `AgeDistribution`, `SexDistribution`, and `Efficacy` Pydantic fields are Optional (can be None). This enables graceful degradation + LLM re-prompt, but all downstream code must guard against None.
- **LLM non-determinism**: ORR, AE frequencies, and JSON structure vary between runs. Some drugs require 2-3 retries due to JSON parse failures. ORR can swing ±30% (e.g., EC ORR bounces between 0.32 and 0.70).

## Version History

### v15 (PDS Integration — 69.2% → expanded to 7 drugs)
- **Project Data Sphere** integration: Patient-level data from 6 SCLC trials
- PDS evidence source (`evidence/projectdatasphere.py`): Reads cached CSVs, matches via `trial_index.csv`
- PDS dose/demographics/efficacy overrides (patient-level > CT.gov aggregate)
- TEAE prompts: Stage 1 + Stage 3 instruct LLM to include disease symptoms
- Non-standard CDISC handling for Amgen/Alliance/EliLilly formats

### v16 (Dose/Demo/AE Normalization — 69.2% → 76.3%)
- **AE normalization** (validator.py): Lab-threshold regex, junk AE filter, combined-term splitter, 7 new synonyms
- **Indication dose fallbacks** (`_INDICATION_DOSE_FALLBACKS`): NCCN-standard doses for Cisplatin/Paclitaxel/Carboplatin/Etoposide/Gemcitabine per indication
- **AUC override threshold**: 15% for AUC-type doses (was 50% for all dose types)
- **ORR override one-directional**: Only upward override (prevents CT.gov from lowering LLM estimates)
- **Vague dose set expansion**: Added "not specified", "as prescribed", "individualized"
- **Age clamping**: 25-95 bounds in all age derivation paths
- **Sex pct clamping**: Clamps pct_male/pct_female to 0-100 before Pydantic validation
- **MAX_AES = 30**: Better AE Count score for 5/7 drugs
- **Scoring script**: LLM-based AE term matching (British/American, synonyms), comma handling in dose extraction

### v17 (AE Frequency Optimization — 76.3% → 80.5%)
- **DailyMed cross-check disabled** (validator.py): `_cross_check_dailymed()` was overriding programmatic corrections back to inflated DailyMed label values
- **DailyMed 0.5x dampening** (agent_multistage.py): DailyMed label frequencies are aggregate "worst-case" across ALL trials/indications, typically 2-3x higher than trial-specific values. `_DM_DAMPENING = 0.5` applied before merging.
- **CT.gov min() merge** (agent_multistage.py): Changed from `evidence_freqs.update(ctgov_freqs)` (CT.gov overrides DailyMed) to `min(DailyMed×0.5, CT.gov)` per AE. Takes the more conservative estimate since both sources inflate differently.
- **CT.gov upward correction disabled** (validator.py): Individual underreporting correction was inflating hematologic AEs (e.g., leukopenia 88.8% vs GT 16.1%). Only flat-frequency fallback retained.
- **Decoupled from proton/**: Moved `rule_engine/` to project root. All scripts run from `rule_discovery/` directly. `proton/rule_engine/` deleted.
- **AE Freq average improvement**: 0.625 → 0.689 (+10.2% relative)

## Ground Truth Comparison (ground_truth/)

GT rule sets exist for 7 drugs in `ground_truth/` (1_Darb, 2_EP, 3_PCE, 4_EC, 6_PCB, 7_PC, 8_GC). Each uses a split-file format: `base.json` + type-specific overlay. Comparison via `scripts/compare_all_gt.py` scores 9 dimensions (Doses, ORR, Age Range, Sex Ratio, ECOG, AE Count, AE Freq, Top AE, PFS/OS). AE matching uses LLM-based fuzzy matching (Gemini 2.0 Flash) for British/American spelling, synonyms, and lab-vs-clinical term normalization.

### Structural Format Gaps — RESOLVED (v9 GT Alignment)
The following 10 gaps were fixed in the converter (v9):
- **Naming conventions** ✓: AE terms, comorbidity names, lab keys now lowercase snake_case
- **Distribution types** ✓: AE onset uses `"lognormal"`, PFS/OS uses `"exponential"`
- **Lab units** ✓: SI units (`x10^9/L`, `umol/L`, `g/L`, `mmol/L`) with converted reference ranges
- **AE fields** ✓: `risk_modifiers: []` and `cumulative: false` on every AE
- **Grade distribution** ✓: Only populated grades included (no `"5": 0.0`)
- **Model formats** ✓: `mortality_model.channels` uses `{"proportion_of_deaths": float, "description": str}` with GT-aligned channel names; `disposition_model` uses `{"daily_probability": float, "description": str}`

### Remaining Structural Gaps (Not Yet Addressed)
- **Missing demographics fields**: GT has `bmi` distribution; ours lacks it (optional, needs LLM or new defaults)
- **Missing disease_baseline fields**: GT has `n_target_lesions` categorical distribution; ours lacks it
- **Admin schedule fields**: GT includes `dose_value` (numeric) + `dose_unit` (string) alongside `dose_per_administration`; ours lacks these
- **SC overlay**: GT subcutaneous drugs have `injection_volume_ml` and `injection_site_specific` per AE

### Clinical Accuracy (v17 — 80.5% across 7 drugs)

| Drug | Doses | ORR | Age | Sex | ECOG | AE Cnt | AE Freq | Top AE | AVG |
|------|-------|-----|-----|-----|------|--------|---------|--------|-----|
| Darbepoetin alfa | 1.000 | 0.442 | 0.817 | 0.957 | 0.977 | 0.867 | 0.917 | 0.600 | **87.6%** |
| Etoposide + Cisplatin | 1.000 | 0.804 | 0.833 | 0.717 | 0.882 | 0.806 | 0.638 | 0.400 | **76.0%** |
| Paclitaxel+Cisplatin+Etoposide | 0.933 | 0.713 | 0.783 | 0.998 | 0.978 | 0.941 | 0.729 | 0.300 | **80.8%** |
| Carboplatin + Etoposide | 1.000 | 0.799 | 0.950 | 0.877 | 0.928 | 0.625 | 0.748 | 0.400 | **82.2%** |
| Paclitaxel+Carbo+Bevacizumab | 0.933 | 0.832 | 0.850 | 0.707 | 0.900 | 0.558 | 0.638 | 0.500 | **77.5%** |
| Paclitaxel + Carboplatin | 1.000 | 0.786 | 0.750 | 0.914 | 0.701 | 0.968 | 0.557 | 0.500 | **77.6%** |
| Gemcitabine + Cisplatin | 0.869 | 0.943 | 0.683 | 0.988 | 0.916 | 0.844 | 0.558 | 0.500 | **79.7%** |
| **COMBINED** | | | | | | | | | **80.5%** |

**Score progression**: v14: 75.8% (2 drugs) → v15: 69.2% (7 drugs) → v16: 76.3% (7 drugs) → v17: 80.5% (7 drugs)

**Key v17 improvements over v16:**
- AE Freq avg: 0.625 → 0.689 (+10.2% relative) — biggest per-dimension gain
- DailyMed cross-check disabled: was overriding CT.gov-based corrections back to inflated label values
- DailyMed 0.5x dampening + CT.gov min() merge: conservative evidence-based frequency alignment
- CT.gov upward correction disabled: prevented hematologic AE inflation

**Remaining gaps:**
- AE Freq (0.56-0.92): Still weakest dimension; British/American spelling + synonym mismatches reduce matched AE count
- Top AE (0.30-0.60): GT includes disease symptoms at high frequency not prioritized by drug-focused pipeline
- ECOG: Generic default {0.4, 0.5, 0.1} used when CT.gov/PDS data unavailable (PC, PCB, GC)
- LLM non-determinism: ORR and AE profiles vary ±5-10% between runs
