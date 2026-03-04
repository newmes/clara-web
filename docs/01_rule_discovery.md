# 01. Phase 0: Rule Discovery (Rule Agent)

> **파일:** `sim/agents/rule_agent.py` (539 lines)
> **역할:** 약물명 + 적응증을 입력받아, 시뮬레이션에 필요한 모든 확률 규칙을 LLM으로 생성
> **LLM 호출:** 약물당 3회 (Main + CRF Supplement + Composite Model)
> **난수 사용:** 없음 (순수 LLM 기반)

---

## 1. 목적

Rule Agent는 시뮬레이션의 **의학적 기반(medical foundation)**을 구축한다. 임상시험의 pivotal trial 데이터(FDA 라벨, 논문)를 기반으로, LLM이 다음을 결정한다:

- 환자 인구통계 분포 (연령, 성별, 인종, 흡연, ECOG, BMI)
- 동반질환 확률 및 조건부 의존성
- AE 프로파일 (발생률, onset 분포, grade 분포, 지속기간, 누적성, 가역성)
- 치료 효능 (반응률, 종양 반응 분포)
- 검사 기준치 (정상 범위, ULN, LLN)
- 투약 스케줄 (약물별 용량, 경로, 주기)
- 용량 조절 규칙 (CTCAE grade별 action)
- 보조약 처방 규칙 (AE별 표준 치료)
- 사망/ECOG/중도탈락 모델 파라미터

이 모든 것이 하나의 `rule_set.json` 파일로 저장되며, 이후 Phase 1, 2에서 참조된다.

---

## 2. 아키텍처

```
discover_rules(drug_name, indication)
    │
    ├─ Call 1: Main Rule Set ──────────── [LLM, 16K tokens]
    │   RULE_DISCOVERY_SYSTEM prompt
    │   + RULE_SET_SCHEMA template
    │   → demographics, comorbidities, ae_profile,
    │     efficacy, lab_reference_ranges, disease_baseline
    │
    ├─ Call 2: CRF Supplement ─────────── [LLM, 8K tokens]
    │   "clinical trial protocol expert" prompt
    │   + CRF_SUPPLEMENT_SCHEMA
    │   → administration_schedule, dose_modification_rules,
    │     supportive_care_rules
    │   [최대 3회 재시도, 누락 키 검증]
    │
    ├─ Call 3: Composite Model ────────── [LLM, 4K tokens]
    │   "clinical trial statistician" prompt
    │   + COMPOSITE_MODEL_SCHEMA
    │   → mortality_model, ecog_model,
    │     ae_cascade_rules, disposition_model
    │
    └─ Save: rule_set.json
```

---

## 3. LLM 프롬프트 상세

### 3.1 Main Call — System Prompt

```
You are a clinical trial simulation architect.
Given a drug and indication, you design the probabilistic simulation rules
needed to generate realistic patient cohorts and outcomes.

CRITICAL RULES:
- All probabilities must be decimal (0-1), NOT percentages
- All probability distributions must sum to ~1.0 for categorical options
- Base estimates on pivotal trial data (FDA labels, published papers)
- Include conditional probabilities where patient factors modify risk
- Be specific about distributions: mean, std, min, max for numerics
- Output ONLY valid JSON. No explanations outside the JSON.
```

**설계 의도:**
- LLM을 "simulation architect"로 위치시켜 확률 설계자 역할 부여
- `decimal (0-1)` 강제 — 100%, 50% 등의 표기가 들어오면 하위 파이프라인에서 10000%, 5000%로 해석됨
- `sum to ~1.0` — 카테고리별 분포의 수학적 유효성 보장
- `pivotal trial data` — LLM이 실제 임상시험 결과에 근거하도록 유도
- `Output ONLY valid JSON` — `generate_json()`의 JSON mode와 이중 보장

### 3.2 Main Call — User Prompt 핵심 지시사항

```
CRITICAL REQUIREMENTS (the simulation engine reads these fields directly):
- ae_profile: Include at least 12-15 AEs ordered by incidence. Each AE MUST have:
  ae_term, incidence_all_grade, grade_distribution, onset_day (with distribution+params),
  duration_days (with distribution+params), cumulative, reversible.
- For oncology: MUST include "tumor_response_distribution"
  with {"CR": float, "PR": float, "SD": float, "PD": float}
```

**12-15개 AE 요구 이유:** 너무 적으면 시뮬레이션이 단조로움, 너무 많으면 희귀 AE까지 포함돼 노이즈 증가.

---

## 4. rule_set.json 스키마 상세

### 4.1 `trial_design`

```json
{
  "cycle_length_days": 21  // Padcev+Pembro: 21일 사이클
}
```

이 값은 시뮬레이션의 시간 구조를 결정:
- 사이클 경계 계산
- 병원 방문일 판단
- 주입일 스케줄
- RECIST 스캔 스케줄 (`2 × cycle_length + 7`일마다)

### 4.2 `demographics`

```json
{
  "age": {"type": "numeric", "distribution": "normal", "params": {"mean": 68, "std": 9, "min": 40, "max": 90}},
  "sex": {"type": "categorical", "options": {"M": 0.75, "F": 0.25}},
  "race": {"type": "categorical", "options": {"White": 0.55, "Asian": 0.25, "Black": 0.12, ...}},
  "smoking": {"type": "categorical", "options": {"never": 0.30, "former": 0.50, "current": 0.20}},
  "ecog_ps": {"type": "categorical", "options": {"0": 0.40, "1": 0.55, "2": 0.05}},
  "bmi": {"type": "numeric", "distribution": "normal", "params": {"mean": 27, "std": 5, "min": 16, "max": 45}}
}
```

각 필드는 `Sampler.sample_from_spec()`이 직접 해석 가능한 구조화 스펙.

### 4.3 `comorbidities`

```json
[
  {
    "condition": "hypertension",
    "base_probability": 0.45,
    "conditional_modifiers": [
      {"if_condition": "age > 65", "multiplier": 1.3},
      {"if_condition": "diabetes", "multiplier": 1.2}
    ],
    "associated_medications": [
      {"name": "Amlodipine", "dose": "5mg", "route": "PO", "frequency": "QD"}
    ],
    "lab_impacts": {"SBP": "may_normalize", "creatinine": "stable"}
  }
]
```

- `conditional_modifiers`: 환자별 확률 조정 (Phase 1에서 LLM이 추가 보정)
- `associated_medications`: Phase 1에서 선택된 동반질환에 약물 배정
- `lab_impacts`: 현재 코드에서는 informational (향후 lab 초기화에 활용 가능)

### 4.4 `ae_profile` (핵심 데이터 구조)

```json
[
  {
    "ae_term": "peripheral_neuropathy",
    "incidence_all_grade": 0.56,
    "grade_distribution": {"1": 0.45, "2": 0.35, "3": 0.15, "4": 0.04, "5": 0.01},
    "onset_day": {
      "distribution": "normal",
      "params": {"mean": 63, "std": 21, "min": 7, "max": 180}
    },
    "duration_days": {
      "distribution": "normal",
      "params": {"mean": 30, "std": 10, "min": 7, "max": 120}
    },
    "risk_modifiers": [
      {"condition": "diabetes", "incidence_multiplier": 1.4},
      {"condition": "age > 70", "incidence_multiplier": 1.2}
    ],
    "cumulative": true,
    "reversible": true
  }
]
```

| 필드 | 사용처 | 설명 |
|------|--------|------|
| `ae_term` | 전체 파이프라인 | 고유 AE 식별자 (snake_case) |
| `incidence_all_grade` | `hazard.daily_onset_hazard()` | 혼합 모델의 I (발생 확률) |
| `grade_distribution` | `daily_agent._check_new_ae_onsets()` | 초기 grade 결정 |
| `onset_day` | `hazard.daily_onset_hazard()` | 혼합 모델의 F(t) 분포 |
| `duration_days` | `hazard.daily_resolution_hazard()` | 해소 확률 계산 |
| `risk_modifiers` | `hazard.adjust_incidence_by_risk_modifiers()` | 환자별 보정 |
| `cumulative` | `hazard.grade_transition_probs()` | 시간에 따른 악화 경향 |
| `reversible` | `hazard.daily_resolution_hazard()` | `false`면 해소 확률 = 0 |

### 4.5 `administration_schedule`

```json
[
  {
    "drug_name": "Enfortumab vedotin",
    "dose_per_administration": "1.25 mg/kg",
    "dose_value": 1.25,
    "dose_unit": "mg/kg",
    "route": "IV",
    "cycle_days": [1, 8],
    "infusion_duration_minutes": 30
  },
  {
    "drug_name": "Pembrolizumab",
    "dose_per_administration": "200mg",
    "dose_value": 200,
    "dose_unit": "mg",
    "route": "IV",
    "cycle_days": [1],
    "infusion_duration_minutes": 30
  }
]
```

- `cycle_days` → 오케스트레이터의 `_is_hospital_day()` 함수에서 사용
- `dose_value` + `dose_unit` → DailySimulator에서 누적 용량 계산
- `"mg/kg"` → 체중 기반 용량 계산 (환자의 `weight_kg` 참조)

### 4.6 `dose_modification_rules`

```json
[
  {
    "ae_term": "peripheral_neuropathy",
    "grade_actions": {
      "1": "DOSE NOT CHANGED",
      "2": "DOSE REDUCED",
      "3": "DRUG INTERRUPTED",
      "4": "DRUG WITHDRAWN"
    },
    "dose_reduction_levels": [1.0, 0.75, 0.5],
    "rechallenge_criteria": "Resume when improved to Grade ≤1"
  },
  {
    "ae_term": "default",
    "grade_actions": {
      "1": "DOSE NOT CHANGED",
      "2": "DOSE NOT CHANGED",
      "3": "DRUG INTERRUPTED",
      "4": "DRUG WITHDRAWN"
    },
    "dose_reduction_levels": [1.0, 0.75, 0.5],
    "rechallenge_criteria": "Resume when improved to Grade ≤2"
  }
]
```

- `"default"` 엔트리: rule_set에 명시되지 않은 AE에 대한 fallback 규칙
- `grade_actions`의 4가지 action은 정확히 CDASH EC 도메인의 표준 코드와 대응
- `dose_reduction_levels`: [1.0 → 0.75 → 0.5] = 2단계 감량. 3번째 감량 시 WITHDRAWN

### 4.7 `mortality_model` (Composite Model)

```json
{
  "baseline_annual_mortality": 0.25,
  "disease_progression": {
    "pd_multiplier": 4.0,
    "response_lag_days": 30,
    "response_reduction": 0.4
  },
  "treatment_toxicity": {
    "grade_multipliers": {"3": 1.5, "4": 3.0},
    "concurrent_ae_threshold": 3,
    "concurrent_ae_multiplier": 1.5
  }
}
```

→ `hazard.compute_daily_mortality()`의 4-channel 모델에서 직접 사용.
상세 수학은 [03_hazard_engine.md](03_hazard_engine.md) 참조.

### 4.8 `ae_cascade_rules`

```json
[
  {"trigger_ae": "neutropenia", "grade_threshold": 3, "target_ae": "febrile_neutropenia", "multiplier": 3.0},
  {"trigger_ae": "hepatotoxicity", "grade_threshold": 2, "target_ae": "alt_increased", "multiplier": 2.0}
]
```

인과 관계 AE 체인 — 호중구감소증 Grade 3 이상이면 발열성 호중구감소증 위험 3배 증가.

---

## 5. 재시도(Retry) 및 에러 처리

### CRF Supplement (3회 재시도)

```python
for attempt in range(max_supplement_retries + 1):  # 0, 1, 2
    try:
        supplement = _generate_crf_supplement(...)
        missing = [k for k in supplement_keys if k not in supplement]
        if missing:
            # 이미 얻은 키는 보존, 누락된 키만 재시도
            continue
        else:
            break  # 성공
    except Exception:
        time.sleep(1)  # 재시도 전 1초 대기 (rate limit 회피)
```

- 부분 성공 시: 이미 얻은 키는 보존하고 재시도
- 모든 시도 실패 시: `RuntimeError` raise — **silent fallback 금지**

### Composite Model (1회 시도)

재시도 없음. 실패 시 즉시 `RuntimeError`.

**설계 철학:** 불완전한 rule_set으로 시뮬레이션을 진행하면 의학적으로 무의미한 결과가 생성되므로, fail-fast 전략 채택.

---

## 6. `load_rules()` — Rule Set 재사용

```python
def load_rules(path: str) -> dict:
    rule_set = json.loads(Path(path).read_text(encoding="utf-8"))
    return rule_set
```

CLI의 `--skip-rules` 플래그와 함께 사용. 약물이 동일하면 기존 rule_set을 재사용하여 LLM 비용 절감.

---

## 7. 출력 예시 (Padcev + Pembrolizumab)

```
[Rule Agent] Main rule set generated
[Rule Agent] CRF supplement (attempt 1/3)...
[Rule Agent] ✓ CRF supplement complete
[Rule Agent] Composite risk model...
[Rule Agent] ✓ Composite model complete
[Rule Agent] Summary:
  AE profile: 15 adverse events
  Comorbidities: 8 conditions
  Admin schedule: 2 drugs
  Composite models: mortality ✓, ECOG ✓, cascade ✓, disposition ✓
[Rule Agent] Saved to data/runs/.../rule_set.json
```