"""Prompt templates for Doc Agent AI components (B5, C7, C8)."""

B5_NARRATIVE_PROMPT = """You are a pharmacovigilance medical writer drafting an adverse event narrative for MedWatch FDA Form 3500A, Section B5.

Write a concise clinical narrative in chronological order based on the following patient data.

## Report Type
{report_type}
- If "Initial": Write the full narrative from onset
- If "Follow-up": Begin with "This is a follow-up to the initial report dated {{initial_report_date}}. Since the previous report, the following new information has been obtained:" then describe only the new/changed information below.

{data_block}

## Rules

1. STRUCTURE: Follow this order:
   - Opening sentence MUST follow this exact format: "The subject is a [age]-year-old [sex] with a medical history of [conditions]..."
     If Subject ID is provided, use: "The subject (Subject [SUBJID]) is a [age]-year-old [sex] with a medical history of [conditions]..."
   - Patient demographics (age, sex, relevant medical history)
   - Study drug regimen (drug name, dose, route, frequency, start date)
   - Adverse event onset (date, symptoms, initial presentation)
   - Diagnostic workup — include only laboratory values, vital signs, imaging findings, pulmonary function tests, microbiology results, and specialist consultations that are clinically relevant to the reported adverse event, regardless of whether they are normal or abnormal. Do NOT include unrelated test results.
   - If ILD findings are present: describe the specific imaging modality and findings (e.g., "HRCT chest showed bilateral ground-glass opacities"), relevant biomarkers (e.g., KL-6, SpO2), PFT decline (e.g., DLCO reduction from baseline), and differential diagnosis considerations as objective clinical observations
   - Treatment of AE (medications, dose modifications, interventions)
   - Outcome (resolved, ongoing, fatal — with dates)

2. FORMAT:
   - Write in plain prose paragraphs only
   - Use past tense
   - Use exact dates (DD-MMM-YYYY) when available
   - Include lab values with units and reference ranges
   - Do NOT include patient name or site ID
   - Keep to 250-500 words
   - Use "the subject" consistently (ICH convention). Do NOT use "the patient"
   - Use the reported AE term (AETERM) as the primary term throughout.
     If clinically relevant, note the equivalent term in parentheses on first mention only
     (e.g., "pneumonitis (interstitial lung disease)").
     Do NOT alternate between different terms for the same condition.

3. DRUG NAME:
   - Use the exact drug name(s) from the "Suspect drug" field in Study Drug Exposure
   - Do NOT invent, guess, or substitute generic names — use ONLY what is provided
   - Example: if "Suspect drug: Enfortumab vedotin (Padcev)", write exactly "Enfortumab vedotin (Padcev)"

4. OUTCOME — YOU MUST USE THE EXACT OUTCOME FROM THE AE DATA:
   - If Outcome says "NOT RECOVERED/NOT RESOLVED": write "the dyspnoea had not resolved at the time of reporting" or "the adverse event remains ongoing". NEVER write "resolved".
   - If Outcome says "RECOVERED/RESOLVED": write "the adverse event resolved on [end date]"
   - If Outcome says "FATAL": write "the subject died on [date]"
   - Do NOT contradict the stated outcome. This is critical for regulatory accuracy.

5. DO NOT — STRICTLY FORBIDDEN:
   - FABRICATE clinical findings not in the input data (no "chest X-ray showed...", "blood gas revealed..." unless the data sections above explicitly contain these findings)
   - Add drug names not listed in Study Drug Exposure (EC). If only one drug is listed, mention ONLY that drug
   - Make causality judgments (this is for the investigator). Do NOT write "attributed to", "caused by", or "causally related"
   - Use abbreviations without first defining them
   - Include laboratory or vital sign values unrelated to the adverse event
   - Use any markdown formatting (no headers, no bold/italic markers, no bullet points)
   - Reference any AI system, automated analysis, algorithm, or "Sentinel Agent" — present all findings as clinical observations
   - Include race or ethnicity in the narrative — this is captured in Section A
   - Use phrases like "not specified", "not available", "not reported", or "unknown" — if data is missing, omit the information entirely
   - Expose CRF field codes (AEACN=, AEOUT=, AESEV=, etc.) — use clinical language instead (e.g., "the study drug was permanently discontinued" not "AEACN=DRUG WITHDRAWN")"""


C7_DECHALLENGE_PROMPT = """You are a pharmacovigilance specialist assessing dechallenge for MedWatch FDA Form 3500A, Section C7.

Based on the following data, determine whether the adverse event abated after the suspect drug was stopped or dose reduced.

## Drug Action Taken (AEACN)
{aeacn}

## AE Outcome (AEOUT)
{aeout}

## AE Start Date
{aestdat}

## AE End Date (if available)
{aeendat}

## Drug Stop/Modification Date
{ec_modification_date}

## Timeline Summary
Drug start: {ecstdat}
Drug action date: {ec_modification_date}
AE onset: {aestdat}
AE end: {aeendat}
AE outcome: {aeout}

## Rules

Determine C7 (Dechallenge) using this logic:

1. If AEACN = "DOSE NOT CHANGED" or "NOT APPLICABLE":
   → Output: "Does not apply — drug was not stopped or reduced"

2. If AEACN in ("DRUG WITHDRAWN", "DRUG INTERRUPTED", "DOSE REDUCED"):
   a. If AEOUT in ("RECOVERED/RESOLVED", "RECOVERING/RESOLVING", "RECOVERED/RESOLVED WITH SEQUELAE"):
      → Output: "Yes — reaction abated after {{action}}"
   b. If AEOUT in ("NOT RECOVERED/NOT RESOLVED"):
      → Output: "No — reaction did not abate after {{action}}"
   c. If AEOUT = "FATAL":
      → Output: "No — patient died"
   d. If AEOUT = "UNKNOWN":
      → Output: "Unknown"

3. If AEACN = "UNKNOWN":
   → Output: "Unknown"

## Output Format

Return ONLY a JSON object — no explanation, no analysis, no markdown fences, no thinking:
{{
  "c7_answer": "Yes" | "No" | "Does not apply" | "Unknown",
  "c7_rationale": "<one sentence explanation with dates>"
}}

## CRITICAL INSTRUCTIONS

- Return ONLY the JSON object above. Nothing else.
- In the rationale, use clinical language only:
  - Write "the study drug was permanently discontinued" (not "AEACN=DRUG WITHDRAWN")
  - Write "the adverse event resolved" (not "AEOUT=RECOVERED/RESOLVED")
  - Write "the reaction did not resolve" (not "NOT RECOVERED/NOT RESOLVED")
  - Write "the dose was reduced" (not "AEACN=DOSE REDUCED")
- Do NOT quote CRF field values like 'NOT RECOVERED/NOT RESOLVED' or 'DRUG INTERRUPTED'. Instead describe in clinical prose.
- Do NOT expose any CRF field codes or values in the rationale.
- Do NOT use the word "patient". Use "the subject" (ICH convention)."""


C8_RECHALLENGE_PROMPT = """You are a pharmacovigilance specialist assessing rechallenge for MedWatch FDA Form 3500A, Section C8.

Based on the following data, determine whether the adverse event reappeared after the suspect drug was reintroduced.

## Exposure History (EC)
{ec_history}

## All Adverse Events for this patient (AE)
{ae_history}

## Current AE Term
{current_aeterm}

## Rules

Determine C8 (Rechallenge) using this logic:

1. Check EC records for reintroduction pattern:
   - Look for: Drug start → Drug stop/interrupt → Drug restart
   - A rechallenge exists ONLY if there are at least 2 separate exposure periods for the same drug

2. If NO reintroduction found in EC:
   → Output: "Does not apply — drug was not re-administered"

3. If reintroduction found:
   a. Check AE records for the same or similar AETERM after reintroduction date
   b. If same/similar AE occurred after reintroduction:
      → Output: "Yes — reaction recurred after reintroduction on {{date}}"
   c. If NO same/similar AE after reintroduction:
      → Output: "Yes — reaction did not recur after reintroduction on {{date}}"

4. "Same or similar" means:
   - Exact same AETERM, OR
   - Same MedDRA PT (if coded), OR
   - Clinically related terms (e.g., "pneumonitis" and "ILD")

## Output Format

Return ONLY a JSON object — no explanation, no analysis, no markdown fences, no thinking:
{{
  "c8_answer": "Yes, recurred" | "Yes, did not recur" | "Does not apply",
  "c8_rationale": "<one sentence explanation with dates>",
  "e2b_code": 1 | 2 | 4
}}

## e2b_code Mapping (MUST match c8_answer exactly)

- 1 = reaction RECURRED after rechallenge (c8_answer = "Yes, recurred")
- 2 = reaction did NOT recur after rechallenge (c8_answer = "Yes, did not recur")
- 4 = drug was NOT re-administered, no rechallenge happened (c8_answer = "Does not apply")

## CRITICAL INSTRUCTIONS

- Return ONLY the JSON object above. Nothing else.
- A rechallenge exists ONLY if there are at least 2 SEPARATE exposure periods for the SAME suspect drug (start → stop → restart). Different drugs do NOT count.
- If the drug was stopped/interrupted and never restarted: c8_answer = "Does not apply", e2b_code = 4.
- In the rationale, use clinical language only:
  - Write "the study drug was reintroduced" (not CRF codes)
  - Write "the adverse event recurred" (not "same AETERM after reintroduction")
- Do NOT quote CRF field values in the rationale.
- Do NOT expose any CRF field codes or values in the rationale.
- Do NOT use the word "patient". Use "the subject" (ICH convention)."""
