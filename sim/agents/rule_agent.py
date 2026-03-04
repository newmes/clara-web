"""Rule Agent — 약물별 시뮬레이션 규칙 발견 + 확률 테이블 생성

Phase 0: drug_name + indication → LLM이 규칙 카테고리를 결정 → 확률 테이블을 채움.

이 Agent는 "무엇을 질문할지"부터 LLM이 결정한다.
항암제가 아닌 다른 약물에도 적용 가능하도록 drug-agnostic 설계.

출력:
  rule_set.json = {
    "drug_name": "...",
    "indication": "...",
    "trial_design": {...},
    "demographics": {...},
    "comorbidities": [...],
    "disease_baseline": {...},
    "ae_profile": [...],
    "efficacy": {...},
    "lab_reference_ranges": {...},
    "administration_schedule": [...],
    "dose_modification_rules": [...],
    "supportive_care_rules": [...],
  }
"""

import json
from pathlib import Path
from sim.agents.llm_client import generate_json, set_caller, DEFAULT_MODEL
from sim.logger import get_logger

_logger = get_logger("rule_agent")

RULE_DISCOVERY_SYSTEM = (
    "You are a clinical trial simulation architect. \n"
    "Given a drug and indication, you design the probabilistic simulation rules "
    "needed to generate realistic patient cohorts and outcomes.\n\n"
    "Your expertise spans all therapeutic areas — oncology, cardiology, neurology, "
    "endocrinology, immunology, etc.\n\n"
    "You will be asked to:\n"
    "1. Identify what probabilistic decisions are needed for this specific drug/indication\n"
    "2. Fill in probability distributions based on published clinical data\n\n"
    "CRITICAL RULES:\n"
    "- All probabilities must be decimal (0-1), NOT percentages\n"
    "- All probability distributions must sum to ~1.0 for categorical options\n"
    "- Base estimates on pivotal trial data (FDA labels, published papers)\n"
    "- Include conditional probabilities where patient factors modify risk\n"
    "- Be specific about distributions: mean, std, min, max for numerics\n"
    "- Output ONLY valid JSON. No explanations outside the JSON."
)

RULE_SET_SCHEMA = {
    "drug_name": "string",
    "indication": "string",
    "trial_design": {
        "cycle_length_days": "integer (CRITICAL: used by simulation engine)",
    },
    "demographics": {
        "age": {
            "type": "numeric",
            "distribution": "normal",
            "params": {"mean": "float", "std": "float", "min": "float", "max": "float"},
        },
        "sex": {"type": "categorical", "options": {"M": "float (probability)", "F": "float"}},
        "race": {"type": "categorical", "options": {"race_name": "float (probability)", "...": "..."}},
        "smoking": {"type": "categorical", "options": {"never": "float", "former": "float", "current": "float"}},
        "ecog_ps": {"type": "categorical", "options": {"0": "float", "1": "float", "2": "float"}},
        "bmi": {
            "type": "numeric",
            "distribution": "normal",
            "params": {"mean": "float", "std": "float", "min": "float", "max": "float"},
        },
    },
    "comorbidities": [
        {
            "condition": "string (e.g., hypertension)",
            "base_probability": "float (0-1)",
            "conditional_modifiers": [
                {"if_condition": "string (e.g., 'age > 70')", "multiplier": "float (e.g., 1.3)"}
            ],
        }
    ],
    "disease_baseline": {
        "_description": "Disease-specific baseline measurements. Structure depends on disease type.",
        "_examples": {
            "oncology": {
                "tumor_sites": {"site_name": "float (probability of metastasis to this site)"},
                "n_target_lesions": {"type": "categorical", "options": {"2": "float", "3": "float", "4": "float", "5": "float"}},
                "sum_of_diameters_mm": {"type": "numeric", "distribution": "lognormal", "params": {}},
                "tumor_response_distribution": {"CR": "float", "PR": "float", "SD": "float", "PD": "float"},
            },
            "diabetes": {
                "baseline_hba1c": {"type": "numeric", "distribution": "normal", "params": {}},
            },
        },
        "YOUR_DISEASE_FIELDS_HERE": "Fill based on the actual indication",
    },
    "ae_profile": [
        {
            "ae_term": "string (snake_case)",
            "incidence_all_grade": "float (0-1, CRITICAL: used by hazard engine)",
            "grade_distribution": {"1": "float", "2": "float", "3": "float", "4": "float", "5": "float"},
            "onset_day": {"distribution": "normal | lognormal | uniform", "params": {"mean": "float", "std": "float", "min": "float", "max": "float"}},
            "duration_days": {"distribution": "normal", "params": {"mean": "float", "std": "float", "min": "float"}},
            "risk_modifiers": [{"condition": "string", "incidence_multiplier": "float"}],
            "cumulative": "boolean (CRITICAL: affects grade worsening over time)",
            "reversible": "boolean (CRITICAL: if false, AE never resolves)",
        }
    ],
    "efficacy": {
        "_description": "Disease-specific efficacy measures.",
        "_oncology_required": {
            "overall_response_rate": "float (CRITICAL: fallback for tumor response sampling)",
            "complete_response_rate": "float (CRITICAL: used with ORR)",
        },
        "YOUR_EFFICACY_FIELDS_HERE": "Fill based on the actual indication",
    },
    "lab_reference_ranges": {
        "lab_name": {
            "unit": "string",
            "normal_range": {"min": "float", "max": "float"},
            "ULN": "float (upper limit of normal)",
            "LLN": "float (lower limit of normal, if relevant)",
        }
    },
}


def discover_rules(drug_name: str, indication: str, model: str = DEFAULT_MODEL, save_path=None) -> dict:
    """약물 + 적응증으로부터 시뮬레이션 규칙 전체를 생성한다."""
    set_caller("rule_agent")
    _logger.info(f"[Rule Agent] Discovering simulation rules for {drug_name} ({indication})...")
    print(f"[Rule Agent] Discovering simulation rules for {drug_name} ({indication})...")

    user_prompt = (
        f"Drug: {drug_name}\nIndication: {indication}\n\n"
        "Generate a complete set of probabilistic simulation rules for a clinical trial of this drug.\n\n"
        "The rule set must enable:\n"
        "1. Generating realistic patient cohorts (demographics, comorbidities, baseline values)\n"
        "2. Determining what adverse events each patient will experience (and when)\n"
        "3. Determining treatment efficacy for each patient\n"
        "4. Day-by-day simulation of patient status\n\n"
        "Use data from the pivotal clinical trial(s) for this drug/indication.\n"
        "If this is a combination therapy, consider AEs from all components.\n\n"
        "CRITICAL REQUIREMENTS (the simulation engine reads these fields directly):\n"
        "- disease_baseline: Fill with fields appropriate for THIS disease.\n"
        '  For oncology: MUST include "tumor_response_distribution" '
        'with {"CR": float, "PR": float, "SD": float, "PD": float}.\n'
        "  Also include tumor_sites, n_target_lesions, sum_of_diameters_mm.\n"
        "  For non-oncology: include disease-specific measures.\n"
        '- efficacy: For oncology, MUST include "overall_response_rate" and "complete_response_rate".\n'
        "- ae_profile: Include at least 12-15 AEs ordered by incidence. Each AE MUST have:\n"
        "  ae_term, incidence_all_grade, grade_distribution, onset_day (with distribution+params),\n"
        "  duration_days (with distribution+params), cumulative, reversible.\n"
        "- comorbidities: Include 6-10 common comorbidities with base_probability and conditional_modifiers.\n"
        "- lab_reference_ranges: Include all labs that could be affected by the drug or disease.\n"
        "- Do NOT include administration_schedule, dose_modification_rules, or supportive_care_rules.\n"
        "- Do NOT include \"reasoning\" fields.\n\n"
        f"Output the following JSON structure:\n\n{json.dumps(RULE_SET_SCHEMA, indent=2, ensure_ascii=False)}"
    )

    rule_set = generate_json(RULE_DISCOVERY_SYSTEM, user_prompt, model=model, max_tokens=16384)

    # CRF Supplement
    _logger.info("[Rule Agent] Supplementing CRF fields (admin schedule, dose mods, supportive care)...")
    print("  -> Supplementing CRF fields (admin schedule, dose mods, supportive care)...")

    supplement_keys = ("administration_schedule", "dose_modification_rules", "supportive_care_rules")
    max_supplement_retries = 2

    for attempt in range(max_supplement_retries + 1):
        try:
            supplement = _generate_crf_supplement(drug_name, indication, rule_set, model)
            for k in supplement_keys:
                if k in supplement:
                    rule_set[k] = supplement[k]
            missing = [k for k in supplement_keys if k not in rule_set]
            if missing:
                _logger.warning(f"[Rule Agent] CRF supplement attempt {attempt + 1}: missing {missing}")
                continue
            else:
                break
        except Exception as e:
            _logger.warning(f"[Rule Agent] CRF supplement attempt {attempt + 1} failed: {e}")
            import time
            time.sleep(1)
    else:
        missing = [k for k in supplement_keys if k not in rule_set]
        if missing:
            raise RuntimeError(
                f"CRF supplement failed after {max_supplement_retries + 1} attempts. Missing: {missing}"
            )

    # Composite Model
    _logger.info("[Rule Agent] Generating composite risk model...")
    print("  -> Generating composite risk model (mortality, ECOG, AE cascade, disposition)...")
    composite = _generate_composite_model_supplement(drug_name, indication, rule_set, model)
    for k, v in composite.items():
        rule_set[k] = v

    rule_set["drug_name"] = drug_name
    rule_set["indication"] = indication

    _logger.info(f"[Rule Agent] Rule discovery complete. Keys: {list(rule_set.keys())}")
    return rule_set


CRF_SUPPLEMENT_SCHEMA = {
    "administration_schedule": [
        {
            "drug_name": "string",
            "dose_per_administration": "string (e.g., '1.25 mg/kg')",
            "dose_value": "float",
            "dose_unit": "mg | mg/kg | mg/m2",
            "route": "INTRAVENOUS | ORAL | SUBCUTANEOUS",
            "cycle_days": "[int] (days within a cycle)",
            "infusion_duration_minutes": "integer | null",
        }
    ],
    "dose_modification_rules": [
        {
            "ae_term": "string (specific AE or 'default')",
            "grade_actions": {
                "1": "DOSE NOT CHANGED | DOSE REDUCED | DRUG INTERRUPTED | DRUG WITHDRAWN",
                "2": "action",
                "3": "action",
                "4": "action",
            },
            "dose_reduction_levels": [1.0, 0.75, 0.5],
            "rechallenge_criteria": "string",
        }
    ],
    "supportive_care_rules": [
        {
            "ae_term": "string",
            "treatments": [
                {
                    "drug": "string",
                    "dose": "string",
                    "unit": "mg | ug | mL",
                    "route": "ORAL | INTRAVENOUS | SUBCUTANEOUS | TOPICAL",
                    "frequency": "QD | BID | TID | PRN",
                    "probability": "float (0-1)",
                }
            ],
        }
    ],
}


def _generate_crf_supplement(drug_name, indication, rule_set, model=DEFAULT_MODEL):
    """기존 rule_set에 없는 CRF 필드를 별도 LLM 호출로 생성."""
    ae_terms = [ae["ae_term"] for ae in rule_set.get("ae_profile", [])][:10]
    system = (
        "You are a clinical trial protocol expert.\n"
        "Given a drug/indication and a list of known AEs, generate:\n"
        "1. Administration schedule (exact dosing for each drug in the regimen)\n"
        "2. Dose modification rules (CTCAE-based, for each significant AE)\n"
        "3. Supportive care rules (standard medications for each AE)\n"
        "Output ONLY valid JSON. No explanations."
    )
    user = (
        f"Drug: {drug_name}\nIndication: {indication}\n"
        f"Cycle length: {rule_set.get('trial_design', {}).get('cycle_length_days', 21)} days\n\n"
        f"Known AEs from this drug: {ae_terms}\n\n"
        "IMPORTANT:\n"
        "- administration_schedule: List each drug separately if combination therapy.\n"
        "  Include exact cycle_days (e.g., [1, 8] for Day 1 and Day 8).\n"
        "  CRITICAL: Use the COMBINATION THERAPY schedule, not the monotherapy schedule.\n"
        "  For example, Padcev (enfortumab vedotin) + Pembrolizumab (EV-302/KEYNOTE-A39):\n"
        "    - Padcev: Day 1,8 Q21D (NOT Day 1,8,15 — Day 15 is omitted in combination)\n"
        "    - Pembrolizumab: Day 1 Q21D\n"
        "  Always verify the approved schedule for the specific combination regimen.\n"
        "- dose_modification_rules: Include 'default' rule AND specific rules for top 5 AEs.\n"
        "  grade_actions values must be exactly: DOSE NOT CHANGED | DOSE REDUCED | DRUG INTERRUPTED | DRUG WITHDRAWN\n"
        "  CLINICAL PRINCIPLE for grade_actions (follow real prescribing information):\n"
        "    * Grade 1: Almost always DOSE NOT CHANGED.\n"
        "    * Grade 2: Usually DOSE NOT CHANGED (managed with conmed/supportive care first).\n"
        "      Only use DOSE REDUCED for cumulative toxicities (e.g., neuropathy).\n"
        "    * Grade 3: DRUG INTERRUPTED (standard). DRUG WITHDRAWN only for life-threatening.\n"
        "    * Grade 4: DRUG WITHDRAWN for most AEs.\n"
        "  The default rule should be: G1=NOT CHANGED, G2=NOT CHANGED, G3=INTERRUPTED, G4=WITHDRAWN.\n"
        "- supportive_care_rules: For each AE, 1-3 medications with probabilities summing to 1.\n\n"
        f"Output the following JSON structure:\n\n{json.dumps(CRF_SUPPLEMENT_SCHEMA, indent=2, ensure_ascii=False)}"
    )
    return generate_json(system, user, model=model, max_tokens=8192)


COMPOSITE_MODEL_SCHEMA = {
    "mortality_model": {
        "baseline_annual_mortality": "float (0-1, age/stage adjusted)",
        "channels": {
            "disease_progression": {
                "pd_multiplier": "float (e.g., 4.0)",
                "response_lag_days": "integer (e.g., 21)",
                "response_reduction": "float (0-1, e.g., 0.3)",
            },
            "treatment_toxicity": {
                "ae_grade_multipliers": {"3": "float", "4": "float"},
                "concurrent_ae_threshold": "integer (e.g., 3)",
                "concurrent_ae_multiplier": "float (e.g., 2.0)",
            },
        },
    },
    "ecog_model": {
        "ae_burden_weight": "float (e.g., 0.15)",
        "disease_weight": "float (e.g., 0.3)",
        "response_lag_days": "integer (e.g., 21)",
        "response_benefit": "float (negative, e.g., -0.3)",
        "comorbidity_penalty": "float (e.g., 0.1)",
    },
    "ae_cascade_rules": [
        {
            "trigger_ae": "string",
            "grade_threshold": "integer",
            "target_ae": "string",
            "multiplier": "float",
        }
    ],
    "disposition_model": {
        "independent_hazards": {
            "consent_withdrawal": {
                "base_daily_rate": "float",
                "split": {"WITHDREW CONSENT": "float", "WITHDRAWAL BY SUBJECT": "float"},
                "risk_factors": {
                    "active_ae_grade_3_plus": "float",
                    "ecog_worsened": "float",
                    "treatment_weeks_gt_12": "float",
                    "poor_response": "float",
                },
            },
            "physician_decision": {
                "base_daily_rate": "float",
                "risk_factors": {
                    "ecog_ge_3": "float",
                    "multiple_dose_reductions": "float",
                    "poor_tumor_response": "float",
                    "severe_ae": "float",
                },
            },
        }
    },
}


def _generate_composite_model_supplement(drug_name, indication, rule_set, model=DEFAULT_MODEL):
    """복합 위험 모델 파라미터를 별도 LLM 호출로 생성."""
    ae_terms = [ae["ae_term"] for ae in rule_set.get("ae_profile", [])][:10]
    system = (
        "You are a clinical trial statistician and oncologist.\n"
        "Given a drug/indication and known AEs, generate composite risk model parameters.\n"
        "These determine: daily mortality, ECOG trajectory, AE cascade, and discontinuation.\n\n"
        "Key principles:\n"
        "- All rates should be medically realistic for this specific drug/indication\n"
        "- AE cascades: only genuinely causal relationships\n"
        "- Disposition rates: based on typical Phase 3 oncology trial dropout patterns\n"
        "Output ONLY valid JSON. No explanations."
    )
    user = (
        f"Drug: {drug_name}\nIndication: {indication}\nKnown AEs: {ae_terms}\n\n"
        "Generate these composite model parameters:\n\n"
        "1. mortality_model: Annual baseline mortality + 2 channels:\n"
        "   - disease_progression: PD multiplier, response lag/reduction\n"
        "   - treatment_toxicity: AE grade multipliers, concurrent threshold\n\n"
        "2. ecog_model: Weights for AE burden, disease progression, comorbidity on ECOG\n\n"
        "3. ae_cascade_rules: 3-6 genuinely causal AE cascades\n\n"
        "4. disposition_model: 2 channels only:\n"
        "   - consent_withdrawal: base rate + risk factors\n"
        "   - physician_decision: base rate + risk factors\n\n"
        f"Output schema:\n{json.dumps(COMPOSITE_MODEL_SCHEMA, indent=2, ensure_ascii=False)}"
    )
    return generate_json(system, user, model=model, max_tokens=4096)


def load_rules(path: str) -> dict:
    """기존 rule_set을 파일에서 로드."""
    rule_set = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"[Rule Agent] Loaded rules from {path}")
    return rule_set
