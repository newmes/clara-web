"""Prompt templates for multi-indication rule set merging.

The MULTIDRUG_MERGE_PROMPT guides the LLM to combine 3 individual rule sets
(each for a separate drug-regimen/indication pair) into one unified JSON rule set
for clinical trial simulation of a patient treated across all indications.
"""

MULTIDRUG_MERGE_PROMPT = """\
You are a clinical pharmacologist merging multiple drug regimen rule sets into one \
unified multi-indication clinical trial simulation rule set.

## Input

### Individual Rule Sets (one per indication)
{individual_rule_sets}

### Drug-Drug Interaction Evidence
{ddi_evidence}

### Overlapping Adverse Events (pre-computed)
{overlapping_aes}

## Task

Merge these individual rule sets into ONE unified JSON rule set for a patient being \
treated for all indications simultaneously: {all_indications}.

All drugs across all regimens: {all_drugs}

## Merging Rules

### drugs
Union of all drugs across all regimens. No duplicates.

### indication
Comma-separated: e.g. "SCLC, NSCLC, Squamous NSCLC"

### regimen
Include every drug's regimen entry from the individual rule sets. If a drug appears \
in multiple regimens (e.g. Cisplatin in two regimens), include both entries.

### demographics
Weighted average across indication populations. Use the individual rule sets' demographics \
to compute blended age (mean/std), sex ratios, and race/ethnicity.

### comorbidities
Union of all comorbidities. Deduplicate by condition name (keep the entry with more \
ae_risk_modifiers). Merge ae_risk_modifiers from all sources.

### adverse_events — CRITICAL
For each unique AE across all regimens:
1. **Non-overlapping AEs** (only in one regimen): Copy directly with source_drug.
2. **Overlapping AEs** (same AE in multiple regimens): Use the probabilistic independence \
model to combine frequencies:
   P(combined) = 1 - (1 - Pa/100) * (1 - Pb/100) * ... (for each regimen's frequency)
   Convert back to percentage. Cap at 95%.
3. **Severity**: For overlapping AEs, weight toward the more severe distribution \
(higher grade_3 + grade_4 fraction).
4. **Onset**: For overlapping AEs, use the EARLIER onset (lower median_onset_days).
5. **source_drug**: For overlapping AEs, assign the drug with the highest individual frequency.
6. **Triggers**: Merge triggers from all sources, dedup by (target_ae, condition).
7. Each AE's severity_distribution grades must sum to frequency_pct.
8. Do NOT exceed the sum of individual AE counts (no fabrication).

### efficacy
Use the PRIMARY indication's efficacy (first regimen) as the top-level efficacy block.

### per_indication_efficacy
One entry per indication with: indication name, regimen_drugs list, efficacy data, phase, \
and treatment_duration_days from the corresponding individual rule set.

### drug_interactions
Populate from the DDI evidence. For each interaction pair:
- interaction_type: "pharmacokinetic", "pharmacodynamic", or "synergistic"
- severity: "mild", "moderate", or "severe" based on clinical significance
- ae_impact: list AE names that may be affected
- frequency_modifier: 1.0-1.5 for mild, 1.5-2.0 for moderate, 2.0-3.0 for severe
- monitoring_recommendation: brief clinical guidance

### overlapping_ae_notes
For each overlapping AE, document:
- event name
- contributing_drugs list
- unadjusted_frequency_sum (naive sum)
- adjusted_frequency_pct (after probabilistic model)
- rationale (brief explanation)

### Multi-indication flags
Set: is_multi_indication=true, indications=[list of all indications], \
source_rule_sets=[list of source filenames if known]

### phase and treatment_duration_days
Use the maximum phase and maximum treatment_duration_days across all indications.

## Output Format

Respond ONLY with valid JSON matching this schema (no markdown fences, no commentary):

{schema}

Remember:
- All severity_distribution grades must sum to frequency_pct for each AE
- Use probabilistic independence model for overlapping AE frequencies, NOT naive sums
- Include per_indication_efficacy for EACH indication
- Include drug_interactions for cross-regimen drug pairs with DDI evidence
- Set is_multi_indication to true
"""
