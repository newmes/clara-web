# 02. Phase 1: Patient Generation (Patient Agent)

> **파일:** `src/agents/patient_agent.py` (455 lines)
> **역할:** rule_set의 확률 분포에서 개별 환자를 생성 (LLM→rand→LLM 패턴)
> **LLM 호출:** 환자당 3회
> **난수 사용:** 환자당 독립 Sampler(seed=base+patient_num)
>
> 🔗 **웹에서 확인:** [Patient Detail (PT-001)](http://49.254.130.90:9000/patient/20260216_062601_Padcev___Pembrolizumab_10pt_84d/PT-001/) — 생성된 환자의 demographics, comorbidities, baseline 데이터 확인

---

## 1. 핵심 패턴: LLM→rand→LLM

Patient Agent는 `prob_engine.py`의 핵심 패턴을 구현한다:

```
Step 1: LLM이 확률을 추정
  "이 환자(68세, 남, 흡연력)의 고혈압 확률은 0.52"

Step 2: 코드가 주사위를 굴림
  sampler.boolean(0.52) → True (고혈압 있음)

Step 3: LLM이 결정된 사실에 맞춰 상세 생성
  "고혈압 환자이므로 creatinine 1.4, SBP 148..."
```

**왜 이 패턴인가?**
- LLM만 쓰면 → mode collapse (대부분 비슷한 환자 생성)
- 코드만 쓰면 → 의학적 일관성 부재 (고혈압인데 정상 혈압)
- LLM→rand→LLM → 다양성(코드) + 일관성(LLM) 동시 확보

---

## 2. 4-Step 환자 생성 파이프라인

```
generate_patient(rule_set, patient_number, total_patients, sampler, model)
    │
    ├── Step 1: _sample_demographics()           [코드만, LLM 0회]
    │    rule_set.demographics에서 직접 샘플링
    │    → {age: 68, sex: "M", race: "White", smoking: "former", ecog_ps: 1, bmi: 27.3}
    │
    ├── Step 2: _sample_comorbidities()          [LLM 1회]
    │    LLM이 환자 특성에 맞게 확률 조정 → 코드가 각각 coin flip
    │    → [{condition: "hypertension", medication: {...}}, ...]
    │
    ├── Step 2.5: Pre-sample disease baseline    [코드만]
    │    종양 부위, 표적 병변 수 → multi_boolean + sample_from_spec
    │    → {tumor_sites: ["bladder", "lymph_node"], n_target_lesions: 3}
    │
    ├── Step 3: _generate_baseline()             [LLM 1회]
    │    demographics + comorbidities + pre_disease → LLM이 일관된 baseline 생성
    │    → {baseline_labs, baseline_vitals, disease_specific_baseline}
    │
    └── Step 4: _generate_persona()              [LLM 1회]
         10종 페르소나 중 코드가 1개 균등 선택 → LLM이 상세 생성
         → {type: "stoic_minimizer", description: "...", disclosure_tendencies: {...}}
```

---

## 3. Step 1: Demographics 샘플링 (코드만)

```python
def _sample_demographics(rule_set: dict, sampler: Sampler) -> dict:
```

**LLM 호출 없음.** rule_set의 demographics 스펙을 Sampler에 직접 전달.

| 필드 | 분포 | 후처리 | 예시 |
|------|------|--------|------|
| `age` | Normal(68, 9, 40, 90) | `round()` → 정수 | 72 |
| `sex` | Categorical(M:0.75, F:0.25) | — | "M" |
| `race` | Categorical(White:0.55, ...) | — | "White" |
| `smoking` | Categorical(never:0.30, ...) | — | "former" |
| `ecog_ps` | Categorical(0:0.40, 1:0.55, 2:0.05) | `int()` | 1 |
| `bmi` | Normal(27, 5, 16, 45) | `round(x, 1)` | 27.3 |

**스펙 해석 방법 (3가지 fallback):**
1. `spec["type"]` 존재 → `sampler.sample_from_spec(spec)` (가장 구조적)
2. `spec["options"]` 존재 → `sampler.categorical(options)` (카테고리 shorthand)
3. `spec["distribution"]` 존재 → `sampler.numeric(distribution, params)` (수치 shorthand)

---

## 4. Step 2: 동반질환 샘플링 (LLM→rand)

```python
def _sample_comorbidities(rule_set, demographics, sampler, model) -> list[dict]:
```

### 4.1 LLM 확률 조정 (1회 호출)

**Context 구성:**
```
Drug: Padcev + Pembrolizumab
Indication: metastatic urothelial carcinoma
Patient: 72-year-old Male, former smoker, ECOG 1, BMI 27.3
Race: White

Base comorbidity rates:
- hypertension: 0.45
- diabetes: 0.25
- CKD: 0.15
- COPD: 0.12
...
```

**Question:**
```
Based on this patient's demographics, adjust each comorbidity probability.
Consider: age (older → higher HTN, DM, CKD), sex, smoking history (COPD, cardiovascular),
BMI/obesity.
```

**LLM 응답 (예시):**
```json
{
  "comorbidities": [
    {"condition": "hypertension", "adjusted_probability": 0.52, "reasoning": "72yo + male + former smoker"},
    {"condition": "diabetes", "adjusted_probability": 0.30, "reasoning": "age factor, normal BMI"},
    {"condition": "CKD", "adjusted_probability": 0.20, "reasoning": "age 72 + former smoker"}
  ]
}
```

### 4.2 코드 샘플링

각 동반질환에 대해:
```python
prob = adjusted_probability  # LLM이 보정한 확률 (fallback: base_probability)
selected = sampler.boolean(prob)  # coin flip
```

### 4.3 약물 배정

선택된 동반질환에 약물 배정:
1. `rule_set.associated_medications` → 우선 사용
2. `DEFAULT_COMORBIDITY_MEDICATIONS` (config/defaults.py) → fallback
3. 여러 약물 중 1개 선택: `sampler.numeric("uniform", {min:0, max:len-0.01})`로 다양성 확보

### 4.4 조건부 의존성 (Conditional Dependencies)

초기 샘플링 후, 미선택 동반질환에 대해 재평가:

```python
# 예: 당뇨가 선택되었고, CKD에 "if_condition: diabetes, multiplier: 1.5"가 있으면
for unselected in unselected_conditions:
    for modifier in unselected.conditional_modifiers:
        if any(selected_cond in modifier.if_condition for selected_cond in selected):
            extra_prob = adjusted_prob * modifier.multiplier
            if sampler.boolean(min(extra_prob, 0.95)):  # 상한 95%
                selected.append(unselected)
```

**텍스트 매칭:** `"diabetes" in "diabetes"` 또는 `"diabetes" in "type 2 diabetes"` 등 양방향 substring 매칭.

---

## 5. Step 2.5: Disease Baseline 사전 샘플링 (코드만)

Phase 1 중간에서 종양 관련 값을 코드로 미리 결정:

```python
# 종양 부위: 각각 독립적 Bernoulli 시행
tumor_sites_spec = rule_set["disease_baseline"]["tumor_sites"]
selected_sites = sampler.multi_boolean(tumor_sites_spec)
# 최소 1개 보장: 모두 False면 최고 확률 부위 강제 선택

# 표적 병변 수
n_lesions = int(sampler.sample_from_spec(n_lesions_spec))  # 최소 1
```

**왜 미리 샘플링하나?**
LLM이 baseline을 생성할 때, 종양 부위와 병변 수를 "PREDETERMINED"로 넘겨 LLM이 임의로 바꾸지 못하게 한다. 이렇게 해야 환자 간 종양 분포가 rule_set의 확률을 정확히 따른다.

---

## 6. Step 3: Baseline 생성 (LLM)

```python
def _generate_baseline(rule_set, demographics, comorbidities, model, pre_disease) -> dict:
```

### 6.1 System Prompt 핵심

```
You are a clinical data specialist generating baseline patient data.

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
- ALL vital signs must use METRIC / SI units:
  * Body temperature (BT) in °C — NOT °F
```

**의학적 일관성 규칙:**
- CKD가 있으면 크레아티닌 ↑ — LLM이 이 규칙을 무시하면 시뮬레이션의 의학적 신뢰성 저하
- 동시에 임상시험 적격 기준 충족 필요 — 실제 환자는 screening에서 적격 기준을 통과한 사람만 등록

### 6.2 출력 스키마

```json
{
  "baseline_labs": {
    "ANC": {"value": 4.2, "unit": "x10^9/L"},
    "hemoglobin": {"value": 13.1, "unit": "g/dL"},
    "platelets": {"value": 185, "unit": "x10^9/L"},
    "creatinine": {"value": 1.4, "unit": "mg/dL"},
    "eGFR": {"value": 52, "unit": "mL/min/1.73m2"},
    "ALT": {"value": 28, "unit": "U/L"},
    "AST": {"value": 25, "unit": "U/L"},
    "total_bilirubin": {"value": 0.8, "unit": "mg/dL"},
    "glucose_fasting": {"value": 132, "unit": "mg/dL"},
    "HbA1c": {"value": 6.8, "unit": "%"},
    "TSH": {"value": 2.1, "unit": "mIU/L"},
    "LDH": {"value": 210, "unit": "U/L"}
  },
  "baseline_vitals": {
    "SBP": 148, "DBP": 88, "HR": 76,
    "BT": 36.7, "RR": 16, "SpO2": 96, "weight_kg": 78
  },
  "disease_specific_baseline": {
    "stage": "IV",
    "target_lesions": [
      {"site": "bladder", "diameter_mm": 35},
      {"site": "lymph_node", "diameter_mm": 28},
      {"site": "lymph_node", "diameter_mm": 22}
    ],
    "sum_of_diameters_mm": 85
  },
  "baseline_ecog": 1,
  "consistency_notes": "CKD reflected in elevated creatinine/reduced eGFR. Diabetes reflected in glucose/HbA1c."
}
```

---

## 7. Step 4: Persona 생성 (rand→LLM)

### 7.1 10종 Persona Type

```python
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
```

### 7.2 코드가 균등 선택

```python
persona_type = sampler.categorical(
    {pt: 1.0 / len(PERSONA_TYPES) for pt in PERSONA_TYPES}
)
```

**각 타입 10% 확률.** LLM에게 선택을 맡기면 "anxious_reporter"나 "confused_elderly" 같은 "흥미로운" 유형에 편중 → mode collapse 발생.

### 7.3 LLM이 상세 생성

`generate_details()`로 LLM에 전달:
- **predetermined:** `{"type": "stoic_minimizer"}` — 변경 불가
- **생성 대상:** description, disclosure_tendencies

**disclosure_tendencies 스키마:**
```json
{
  "pain": "minimizes | reports_normally | exaggerates | denies_until_severe",
  "fatigue": "minimizes | reports_normally | exaggerates | attributes_to_age",
  "nausea": "minimizes | reports_normally | exaggerates | denies_until_severe",
  "skin_changes": "minimizes | reports_normally | exaggerates | avoids_topic",
  "emotional_distress": "minimizes | reports_normally | exaggerates | avoids_topic",
  "urinary_symptoms": "minimizes | reports_normally | avoids_topic | shame_avoidant"
}
```

이 `disclosure_tendencies`는 `mood.py`의 7차원 모델, `observation.py`의 감지 확률과 함께 Care AI의 환자 소통 난이도를 결정한다.

---

## 8. 최종 환자 데이터 구조

```json
{
  "patient_id": "PT-001",
  "emr": {
    "demographics": {"age": 72, "sex": "M", "race": "White", "smoking": "former", "ecog_ps": 1, "bmi": 27.3},
    "diagnosis": {"primary": "metastatic urothelial carcinoma", "stage": "IV"},
    "medical_history": [
      {"condition": "hypertension", "ongoing": true, "medication": {"name": "Amlodipine", ...}},
      {"condition": "CKD", "ongoing": true, "medication": null}
    ],
    "baseline_ecog": 1,
    "baseline_labs": {"ANC": {"value": 4.2, "unit": "x10^9/L"}, ...},
    "baseline_vitals": {"SBP": 148, "DBP": 88, "HR": 76, "BT": 36.7, ...},
    "baseline_tumor": {"stage": "IV", "target_lesions": [...], "sum_of_diameters_mm": 85}
  },
  "persona": {
    "type": "stoic_minimizer",
    "description": "72세 전직 기계공. 평생 '괜찮다'를 입에 달고 살아온 ...",
    "disclosure_tendencies": {"pain": "denies_until_severe", "fatigue": "attributes_to_age", ...}
  },
  "initial_state": {
    "location": "HOME",
    "treatment_status": "screening",
    "overall_awareness": "UNAWARE"
  }
}
```

---

## 9. 병렬 생성 & 재현성

### Seed 전략

```python
# generate_patients()에서:
patient_seed = (seed or 0) + i
sampler = Sampler(seed=patient_seed)
```

- 환자 1: seed = base + 1
- 환자 2: seed = base + 2
- ...
- **환자별 독립 Sampler 인스턴스** → 이전 환자의 LLM 호출 수에 관계없이 동일 난수열

### 병렬 실행

`orchestrator.py`의 `create_patients_parallel(n, max_workers=10)`:
- `ThreadPoolExecutor`로 환자를 병렬 생성
- 각 스레드에서 독립 `Sampler(seed=base+i)` 생성
- 결과는 `patients[idx]`에 인덱스 기반 배치 → 순서 보장
- `_print_lock`으로 출력 동기화

---

## 10. 에러 처리

| 단계 | 실패 시 행동 |
|------|------------|
| Demographics | `ValueError` (알 수 없는 분포 타입) |
| Comorbidities (LLM) | `adjusted_probability` 누락 시 `base_probability` fallback |
| Baseline (LLM) | `generate_json` 내부 retry (2회) → 실패 시 raise |
| Persona (LLM) | `generate_json` 내부 retry (2회) → 실패 시 raise |

**원칙:** Demographics는 코드만이므로 사실상 실패 불가. Comorbidities만 soft fallback (base rate 사용). 나머지는 hard failure.