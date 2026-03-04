"""Patient Agent — LLM → rand → LLM 방식 환자 생성

기존 God Agent를 대체한다.

흐름:
  1. rule_set의 demographics 확률 → Sampler가 인구통계 추출
  2. LLM이 (인구통계 + 질환) 보고 동반질환 확률 조정 → Sampler가 동반질환 추출
  3. LLM이 (인구통계 + 동반질환) 보고 기저 Lab/Vitals/질환 기저값 생성
  4. LLM이 페르소나 생성

단계 1은 LLM 호출 없이 rule_set에서 바로 샘플링.
단계 2-4만 LLM 호출.
"""
import json
from pathlib import Path
from sim.agents.llm_client import generate_json, set_caller, DEFAULT_MODEL
from sim.engine.sampler import Sampler
from sim.engine.prob_engine import estimate_probabilities, generate_details
from sim.logger import get_logger
from sim.config.defaults import DEFAULT_COMORBIDITY_MEDICATIONS

_logger = get_logger('patient_agent')


def _sample_demographics(rule_set: dict, sampler: Sampler) -> dict:
    """rule_set.demographics에서 인구통계를 샘플링한다. 순수 코드."""
    demo_rules = rule_set.get('demographics') or {}
    result = {}
    for field, spec in demo_rules.items():
        if isinstance(spec, dict) and 'type' in spec:
            result[field] = sampler.sample_from_spec(spec)
            continue
        if isinstance(spec, dict) and 'options' in spec:
            result[field] = sampler.categorical(spec['options'])
            continue
        if isinstance(spec, dict) and 'distribution' in spec:
            params = {k: v for k, v in spec.items() if k != 'distribution'}
            if 'params' in spec:
                params = spec['params']
            result[field] = sampler.numeric(spec['distribution'], params)
    if 'age' in result:
        result['age'] = round(result['age'])
    if 'ecog_ps' in result:
        result['ecog_ps'] = int(result['ecog_ps'])
    if 'bmi' in result:
        result['bmi'] = round(result['bmi'], 1)
    return result


def _sample_comorbidities(
    rule_set: dict,
    demographics: dict,
    sampler: Sampler,
    model: str = DEFAULT_MODEL,
) -> list[dict]:
    """LLM이 이 환자의 동반질환 확률을 조정 → Sampler가 추출.

    rule_set.comorbidities를 기본 확률로 사용하되,
    환자의 인구통계에 따라 LLM이 조건부 확률을 조정한다.
    """
    comorbidity_rules = rule_set.get('comorbidities') or []
    if not comorbidity_rules:
        return []
    drug_name = rule_set.get('drug_name', '')
    indication = rule_set.get('indication', '')
    context = f"""Drug: {drug_name}
Indication: {indication}
Patient demographics: {json.dumps(demographics, ensure_ascii=False)}

Base comorbidity rates from clinical trial data:
{json.dumps([{'condition': c['condition'], 'base_probability': c['base_probability']} for c in comorbidity_rules], indent=2, ensure_ascii=False)}"""
    question = f"""Given this patient's demographics (age={demographics.get('age')}, \nsex={demographics.get('sex')}, smoking={demographics.get('smoking', 'unknown')}), \nadjust the comorbidity probabilities.

For each comorbidity, output the adjusted probability considering:
- Age (older → higher rates of hypertension, diabetes, CKD)
- Sex (some conditions are sex-dependent)
- Smoking (COPD, cardiovascular disease correlations)
- Obesity/BMI if available

Also add conditional dependencies between comorbidities:
- If diabetes is present → CKD probability increases
- If hypertension is present → cardiovascular disease probability increases
"""
    prob_schema = {
        'comorbidities': [
            {
                'condition': 'string',
                'adjusted_probability': 'float (0-1)',
                'reasoning': 'string (brief)',
            }
        ]
    }
    adjusted = estimate_probabilities(context, question, prob_schema, model)
    selected = []
    adjusted_list = adjusted.get('comorbidities', comorbidity_rules)
    for item in adjusted_list:
        condition = item.get('condition', '')
        prob = item.get('adjusted_probability', item.get('base_probability', 0))
        try:
            prob = float(prob)
        except (TypeError, ValueError):
            prob = 0.0
        if sampler.boolean(prob):
            original = next(
                (c for c in comorbidity_rules if c['condition'] == condition), {}
            )
            med_from_rule = original.get('associated_medications') or []
            if med_from_rule:
                medication = med_from_rule[0]
            else:
                cond_key = condition.lower().replace(' ', '_')
                default_meds = DEFAULT_COMORBIDITY_MEDICATIONS.get(cond_key, [])
                if default_meds:
                    idx = int(sampler.numeric('uniform', {'min': 0, 'max': len(default_meds) - 0.01}))
                    medication = default_meds[min(idx, len(default_meds) - 1)]
                else:
                    medication = None
            selected.append({
                'condition': condition,
                'ongoing': True,
                'medication': medication,
                'lab_impacts': original.get('lab_impacts') or {},
            })

    # Conditional modifiers: 한 질환이 다른 질환의 확률을 올리는 경우
    conditions_present = {c['condition'].lower() for c in selected}
    for item in adjusted_list:
        condition = item.get('condition', '')
        if condition.lower() in conditions_present:
            continue
        for cm in comorbidity_rules:
            if cm['condition'] == condition:
                for mod in (cm.get('conditional_modifiers') or []):
                    trigger = mod.get('if_condition', '').lower()
                    for present in conditions_present:
                        if present in trigger:
                            try:
                                _ap = float(item.get('adjusted_probability', 0))
                            except (TypeError, ValueError):
                                _ap = 0.0
                            extra_prob = _ap * float(mod.get('multiplier', 1))
                            if sampler.boolean(min(extra_prob, 0.95)):
                                original = next(
                                    (c for c in comorbidity_rules if c['condition'] == condition), {}
                                )
                                selected.append({
                                    'condition': condition,
                                    'ongoing': True,
                                'medication': (original.get('associated_medications') or [None])[0],
                                'lab_impacts': original.get('lab_impacts') or {},
                                })
    return selected


def _generate_baseline(
    rule_set: dict,
    demographics: dict,
    comorbidities: list[dict],
    model: str = DEFAULT_MODEL,
    pre_disease: dict | None = None,
) -> dict:
    """인구통계 + 동반질환이 결정된 상태에서 LLM이 기저값을 생성.

    Labs, Vitals, 질환 기저값(종양 등)을 의학적으로 일관되게 채운다.
    pre_disease: 코드가 사전 샘플링한 질환 기저값 (tumor_sites, n_target_lesions 등)
    """
    drug_name = rule_set.get('drug_name', '')
    indication = rule_set.get('indication', '')
    disease_baseline = rule_set.get('disease_baseline') or {}
    lab_ranges = rule_set.get('lab_reference_ranges') or {}

    pre_disease_text = ''
    if pre_disease:
        pre_disease_text = f"""
Disease values (PREDETERMINED by random sampling - do not change):
{json.dumps(pre_disease, indent=2, ensure_ascii=False)}
Use EXACTLY these tumor sites and number of target lesions."""

    context = f"""Drug: {drug_name}
Indication: {indication}

Patient demographics (PREDETERMINED - do not change):
{json.dumps(demographics, indent=2, ensure_ascii=False)}

Comorbidities (PREDETERMINED - do not change):
{json.dumps(comorbidities, indent=2, ensure_ascii=False)}

Lab reference ranges:
{json.dumps(lab_ranges, indent=2, ensure_ascii=False)}

Disease baseline context:
{json.dumps(disease_baseline, indent=2, ensure_ascii=False)}
{pre_disease_text}"""

    schema = {
        'baseline_labs': {
            '_description': 'Each lab: {value: float, unit: string}. Must be consistent with comorbidities.',
            '_required_labs': 'ANC, hemoglobin, platelets, creatinine, eGFR, ALT, AST, total_bilirubin, glucose_fasting, HbA1c, TSH, LDH',
            'lab_name': {
                'value': 'float',
                'unit': 'string',
            },
        },
        'baseline_vitals': {
            '_units': 'SBP/DBP: mmHg, HR: bpm, BT: °C (Celsius, NOT Fahrenheit — e.g. 36.8, NOT 98.6), RR: breaths/min, SpO2: %, weight_kg: kg',
            'SBP': 'float (mmHg)',
            'DBP': 'float (mmHg)',
            'HR': 'float (bpm)',
            'BT': 'float (°C, Celsius — normal ~36.5-37.2, NEVER use Fahrenheit)',
            'RR': 'float (breaths/min)',
            'SpO2': 'float (%)',
            'weight_kg': 'float (kg)',
        },
        'disease_specific_baseline': {
            '_description': 'Disease-specific measurements. For oncology: stage, target_lesions, sum_of_diameters_mm. For others: disease-specific measures.',
            'stage': "string (AJCC stage, e.g., 'IV', 'IIIB'. For metastatic disease, always 'IV')",
        },
        'baseline_ecog': 'integer (must match the ECOG from demographics)',
        'consistency_notes': 'string (brief explanation of how labs reflect comorbidities)',
    }

    system_prompt = """You are a clinical data specialist generating baseline patient data.

CRITICAL RULES:
- All values must be medically consistent with the patient's comorbidities:
  * CKD → creatinine elevated (≥1.5), eGFR reduced (<60)
  * Diabetes → glucose_fasting elevated (≥126), HbA1c elevated (≥6.5)
  * Hypertension → SBP elevated (≥140) OR on medication (then may be controlled)
  * Anemia → hemoglobin reduced
  * COPD → SpO2 may be slightly reduced (92-96%)
- Values must meet typical clinical trial eligibility:
  * ANC ≥ 1.5 x10^9/L
  * Platelets ≥ 100 x10^9/L
  * Adequate organ function (unless specific comorbidity)
- For oncology: include target_lesions with realistic sizes. Use EXACTLY the predetermined tumor_sites and n_target_lesions if provided
- BMI should be consistent with weight_kg and the demographics
- ALL vital signs must use METRIC / SI units:
  * Body temperature (BT) in °C (Celsius) — e.g. 36.8, NOT 98.6°F
  * Weight in kg, Blood pressure in mmHg, Heart rate in bpm
- Output ONLY valid JSON."""

    return generate_details(
        context=context,
        predetermined={'demographics': demographics, 'comorbidities': comorbidities},
        output_schema=schema,
        system_prompt=system_prompt,
        model=model,
    )


# 성중립 페르소나 타입 (성별/나이 정보는 demographics에서 가져옴)
PERSONA_TYPES = [
    "stoic_minimizer",          # 증상 축소, 묻기 전 말 안 함
    "anxious_reporter",         # 사소한 변화도 즉시 보고
    "shame_avoidant",           # 비뇨/피부/정서 증상 회피
    "confused_elderly",         # 인지 저하, 증상 구분 어려움
    "health_literate",          # 의학 지식 있음, 자가 해석
    "minimizer",                # 전반적 경시, "괜찮다" 반복
    "catastrophizer",           # 미미한 증상도 심각하게 해석
    "caregiver_dependent",      # 보호자 통해 간접 소통
    "language_barrier",         # 언어 장벽, 의사소통 제한
    "compliant_but_forgetful",  # 순응적이나 기억력 약함
]


def _generate_persona(
    demographics: dict,
    comorbidities: list[dict],
    sampler: Sampler,
    model: str = DEFAULT_MODEL,
) -> dict:
    """인구통계에 기반한 페르소나를 LLM이 생성한다."""
    persona_type = sampler.categorical(
        {pt: 1.0 / len(PERSONA_TYPES) for pt in PERSONA_TYPES}
    )

    sex = demographics.get('sex', 'Unknown')
    age = demographics.get('age', 'Unknown')

    context = f"""Patient demographics: {json.dumps(demographics, ensure_ascii=False)}
Comorbidities: {json.dumps([c['condition'] for c in comorbidities], ensure_ascii=False)}
Assigned persona type: {persona_type}

IMPORTANT: This patient is {age} years old, sex={sex}. 
The description MUST match this patient's sex and age. Use appropriate pronouns."""

    schema = {
        'type': 'string (the persona type name)',
        'description': "string (2-3 sentences: personality, communication style, life context. Must match patient's sex and age.)",
        'disclosure_tendencies': {
            'pain': 'minimizes | reports_normally | exaggerates | denies_until_severe',
            'fatigue': 'minimizes | reports_normally | exaggerates | attributes_to_age',
            'nausea': 'minimizes | reports_normally | exaggerates | denies_until_severe',
            'skin_changes': 'minimizes | reports_normally | exaggerates | avoids_topic',
            'emotional_distress': 'minimizes | reports_normally | exaggerates | avoids_topic',
            'urinary_symptoms': 'minimizes | reports_normally | avoids_topic | shame_avoidant',
        },
    }

    return generate_details(
        context=context,
        predetermined={'type': persona_type},
        output_schema=schema,
        model=model,
    )


def generate_patient(
    rule_set: dict,
    patient_number: int,
    total_patients: int,
    sampler: Sampler,
    model: str = DEFAULT_MODEL,
) -> dict:
    """LLM → rand → LLM 패턴으로 환자 1명을 생성한다.

    Args:
        rule_set: Rule Agent 출력
        patient_number: 환자 번호 (1-based)
        total_patients: 전체 환자 수
        sampler: 난수 생성기
        model: LLM 모델 ID

    Returns:
        환자 데이터 dict
    """
    pid = f"PT-{patient_number:03d}"
    set_caller(f'patient_agent.{pid}')
    _logger.info(f'[Patient Agent] Generating {pid} ({patient_number}/{total_patients})...')
    print(f'[Patient Agent] Generating {pid} ({patient_number}/{total_patients})...')

    # Step 1: 인구통계 (코드만 — LLM 호출 없음)
    demographics = _sample_demographics(rule_set, sampler)
    print(f"  Demographics: age={demographics.get('age')}, sex={demographics.get('sex')}, race={demographics.get('race', '?')}")

    # Step 2: 동반질환 (LLM 확률 조정 + rand 샘플링)
    comorbidities = _sample_comorbidities(rule_set, demographics, sampler, model)
    cond_names = [c['condition'] for c in comorbidities] if comorbidities else ['none']
    print(f"  Comorbidities: {cond_names}")

    # Step 2.5: 질환 기저값 사전 샘플링 (코드 — tumor sites, n_target_lesions)
    disease_baseline = rule_set.get('disease_baseline') or {}
    pre_disease = {}

    tumor_sites_prob = disease_baseline.get('tumor_sites') or {}
    if tumor_sites_prob and isinstance(tumor_sites_prob, dict):
        sampled_sites = sampler.multi_boolean(tumor_sites_prob)
        selected_sites = [s for s, hit in sampled_sites.items() if hit]
        if not selected_sites:
            selected_sites = [max(tumor_sites_prob, key=tumor_sites_prob.get)]
        pre_disease['tumor_sites'] = selected_sites

    n_lesions_spec = disease_baseline.get('n_target_lesions') or {}
    if n_lesions_spec:
        raw = sampler.sample_from_spec(n_lesions_spec)
        import re as _re
        digits = _re.sub(r'[^\d]', '', str(raw))
        n_lesions = int(digits) if digits else 3
        pre_disease['n_target_lesions'] = max(1, n_lesions)

    # Step 3: 기저값 생성 (LLM — 사전 샘플링된 질환값 반영)
    baseline = _generate_baseline(rule_set, demographics, comorbidities, model,
                                  pre_disease=pre_disease)
    print(f"  Baseline generated (labs + vitals + disease-specific)")

    # Step 4: 페르소나 생성 (LLM)
    persona = _generate_persona(demographics, comorbidities, sampler, model)
    print(f"  Persona: {persona.get('type')}")

    # 환자 데이터 조립
    diagnosis = baseline.get('disease_specific_baseline', {})
    if not isinstance(diagnosis, dict):
        diagnosis = {}
    diagnosis.setdefault('primary', rule_set.get('indication', ''))
    diagnosis.setdefault('stage', '')

    # Ensure target_lesions is populated from pre-sampled tumor sites
    if not diagnosis.get('target_lesions') and pre_disease.get('tumor_sites'):
        import random as _rand
        sod_spec = disease_baseline.get('sum_of_diameters_mm', {}).get('params', {})
        sod_mean = sod_spec.get('mean', 80)
        sod_std = sod_spec.get('std', 40)
        sites = pre_disease['tumor_sites']
        n_lesions = pre_disease.get('n_target_lesions', len(sites))
        n_lesions = max(n_lesions, len(sites))
        lesions = []
        diameters = []
        for i in range(n_lesions):
            site = sites[i % len(sites)]
            d = max(10, _rand.gauss(sod_mean / n_lesions, sod_std / n_lesions))
            diameters.append(d)
            lesions.append({'tumor_site': site, 'diameter_mm': round(d, 1)})
        diagnosis['target_lesions'] = lesions
        diagnosis['sum_of_diameters_mm'] = round(sum(diameters), 1)

    patient = {
        'patient_id': pid,
        'emr': {
            'demographics': demographics,
            'diagnosis': diagnosis,
            'medical_history': comorbidities,
            'baseline_ecog': baseline.get('baseline_ecog', demographics.get('ecog_ps')),
            'baseline_labs': baseline.get('baseline_labs', {}),
            'baseline_vitals': baseline.get('baseline_vitals', {}),
            'baseline_tumor': diagnosis,
        },
        'persona': persona,
        'initial_state': {
            'location': 'HOME',
            'treatment_status': 'screening',
            'overall_awareness': 'UNAWARE',
        },
    }

    return patient


def generate_patients(
    rule_set: dict,
    n: int,
    model: str = DEFAULT_MODEL,
    save_dir: str | None = None,
    seed: int = 0,
) -> list[dict]:
    """N명의 환자를 순차 생성한다.

    각 환자마다 다른 seed를 사용하여 다양성을 보장한다.
    """
    patients = []
    for i in range(1, n + 1):
        patient_seed = seed + i
        sampler = Sampler(seed=patient_seed)
        patient = generate_patient(rule_set, i, n, sampler, model)
        patients.append(patient)
        if save_dir:
            out_path = Path(save_dir) / f"{patient['patient_id']}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(patient, indent=2, ensure_ascii=False), encoding='utf-8')
            print(f"  -> Saved to {out_path}")
    return patients


# CDASH 변환 (map_patient_record, _normalize_race)

def map_patient_record(patient: dict) -> dict:
    """환자 데이터를 CDASH DM + MH 형식으로 변환한다."""
    emr = patient.get('emr', {})
    demo = emr.get('demographics', {})
    mh_list = emr.get('medical_history') or []

    dm = {
        'AGE': demo.get('age'),
        'AGEU': 'YEARS',
        'SEX': demo.get('sex'),
        'RACE': _normalize_race(demo.get('race', '')),
        'ETHNIC': 'NOT REPORTED',
    }

    mh_records = []
    for item in mh_list:
        mh_rec = {
            'MHTERM': item.get('condition', ''),
            'MHSTDAT': None,
            'MHONGO': item.get('ongoing', True),
            'MHENDAT': None,
        }
        mh_records.append(mh_rec)

    return {
        'patient_id': patient.get('patient_id'),
        'DM': dm,
        'MH': mh_records,
        'emr': emr,
        'persona': patient.get('persona'),
        'initial_state': patient.get('initial_state'),
    }


def _normalize_race(race: str) -> str:
    """race 값을 CDASH codelist로 정규화."""
    r = race.upper().strip()
    if 'WHITE' in r or 'CAUCASIAN' in r:
        return 'WHITE'
    elif 'BLACK' in r or 'AFRICAN' in r:
        return 'BLACK OR AFRICAN AMERICAN'
    elif 'ASIAN' in r:
        return 'ASIAN'
    elif 'NATIVE' in r and 'HAWAIIAN' in r:
        return 'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER'
    elif 'INDIAN' in r or 'ALASKA' in r:
        return 'AMERICAN INDIAN OR ALASKA NATIVE'
    elif r in ('NOT REPORTED', 'UNKNOWN', 'OTHER', ''):
        return r or 'NOT REPORTED'
    return 'OTHER'
