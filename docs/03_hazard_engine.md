# 03. Hazard Engine — 확률 수학 모듈

> **파일:** `sim/engine/hazard.py` (1,020 lines), `sim/engine/sampler.py` (145 lines), `sim/engine/prob_engine.py` (202 lines)
> **역할:** Fate table 없이 매일의 이벤트를 동적으로 결정하는 수학 엔진
> **LLM 호출:** 없음 (hazard.py, sampler.py는 순수 수학. prob_engine.py는 LLM↔Sampler 브릿지)

---

## 1. Sampler — 확률의 주사위

### 1.1 설계 원칙

`Sampler`는 Python 표준 라이브러리의 `random.Random(seed)` 인스턴스를 래핑한다. 핵심은:

- **인스턴스 격리:** `random.seed()`와 독립적. 각 Sampler가 자체 Mersenne Twister PRNG 보유
- **재현 가능:** 동일 seed → 동일 난수열 → 동일 시뮬레이션 결과
- **환자별 독립:** 환자마다 새 Sampler 인스턴스 → LLM 호출 수 변동에 영향 받지 않음

### 1.2 메서드 요약

| 메서드 | 입력 | 내부 구현 | 용도 |
|--------|------|----------|------|
| `categorical(options)` | `{name: weight}` | `random.choices(k=1)` | AE 타입, 인종, 흡연 등 |
| `boolean(prob)` | float 0-1 | `random.random() < prob` | AE 발생, 동반질환 선택, 사망 등 |
| `multi_boolean(options)` | `{name: prob}` | 각각 독립 `boolean()` | 종양 부위 선택 |
| `numeric(dist, params)` | 분포명+파라미터 | `gauss/uniform/lognormvariate/triangular` | 연령, BMI, lab 변동 등 |
| `integer(dist, params)` | 분포명+파라미터 | `round(numeric())` | 병변 수, 기간 |
| `sample_from_spec(spec)` | 구조화 스펙 | type 기반 dispatch | 범용 (rule_set 필드 직접 해석) |

### 1.3 Numeric 분포별 구현

| 분포 | Python 함수 | 파라미터 | 클램핑 |
|------|-----------|---------|--------|
| `normal` | `rng.gauss(mean, std)` | mean, std (default 1.0) | min/max 존재 시 |
| `uniform` | `rng.uniform(min, max)` | min (default 0), max (default 1) | 내재적 |
| `lognormal` | `rng.lognormvariate(mu, sigma)` | mu, sigma (log-space) | min/max 존재 시 |
| `triangular` | `rng.triangular(min, max, mode)` | min, max, mode | 내재적 |

**주의:** Normal + 클램핑은 진정한 truncated normal이 아니라 boundary에서 질량 축적(clamping). 좁은 범위에 넓은 std를 쓰면 경계값 편중 발생 가능.

---

## 2. Prob Engine — LLM↔Sampler 브릿지

### 2.1 Two-Phase 패턴

```
Phase A: estimate_probabilities(context, question, schema, model)
  → LLM에게 "확률을 추정해 줘" → JSON 확률 맵

Phase B: generate_details(context, predetermined, schema, model)
  → LLM에게 "이미 결정된 사실에 맞춰 상세를 채워 줘" → JSON 상세 데이터

중간: _auto_sample(prob_output, sampler)
  → LLM 출력을 자동 파싱하여 Sampler로 샘플링
```

### 2.2 System Prompts

**확률 추정용:**
```
You are a medical probability estimator.
- Base estimates on published clinical trial data, FDA labels, medical literature
- All probabilities in [0, 1]
- Categorical options must sum to ~1.0
- Include reasoning for traceability
```

**상세 생성용:**
```
You are a clinical data specialist generating patient data.
- Predetermined conditions are FIXED — cannot be changed
- Fill in remaining fields with medically plausible values
- Ensure internal consistency
```

**핵심:** "PREDETERMINED → FIXED" 지시가 LLM의 자의적 판단을 차단한다. 코드가 결정한 `sex=M, age=72, has_CKD=True`를 LLM이 "더 일반적인" 조합으로 바꾸지 못하게 함.

### 2.3 `_auto_sample()` — 자동 파싱

LLM 출력 형식이 일정하지 않은 현실에 대응:

| 패턴 감지 | Sampler 메서드 |
|----------|---------------|
| `"type"` 키 존재 | `sample_from_spec()` |
| `"options"` 키 (dict) | `categorical()` |
| `"probability"` 키 (수치) | `boolean()` |
| `"distribution"` 키 | `numeric()` |
| 중첩 dict | **재귀** `_auto_sample()` |
| 스칼라 (int/float/str) | pass-through |

`reasoning`, `_reasoning`, `_note` 키는 건너뜀.

---

## 3. Hazard Functions — 핵심 수학

### 3.1 기초: 분포 함수

#### Normal CDF

$$\Phi(x) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x - \mu}{\sigma\sqrt{2}}\right)\right]$$

Python의 `math.erf` (C-level 구현) 사용. `std ≤ 0`이면 Dirac delta (x ≥ mean → 1.0).

#### LogNormal CDF

$$F_{LN}(x) = \Phi\left(\frac{\ln x - \mu}{\sigma}\right)$$

주의: `mu`, `sigma`는 **log-space** 파라미터. `x ≤ 0` → 0.0.

#### Uniform CDF

$$F_U(x) = \text{clamp}\left(\frac{x - lo}{hi - lo}, 0, 1\right)$$

#### Truncated CDF

모든 분포에 적용되는 범위 제한:

$$F_T(x \mid lo \le X \le hi) = \frac{F(x) - F(lo)}{F(hi) - F(lo)}$$

- 분모가 ≤ 1e-12이면 (분포 질량이 범위 밖): 중점 step function
- 결과는 [0, 1]로 클램핑

#### 분포 dispatch

```python
def _distribution_cdf(x, distribution, params):
    if distribution == "normal":
        # Truncated Normal, default: mean=30, std=10, min=0, max=365
    elif distribution == "lognormal":
        # Truncated LogNormal, default: mu=ln(30), sigma=0.5, min=0, max=365
    elif distribution == "uniform":
        # Uniform(min, max), default: 0~365
    else:
        # Unknown → Uniform fallback
```

---

### 3.2 ★ AE Onset Hazard — Mixture Model (Daily)

이 함수가 시뮬레이션의 핵심이다. Fate table을 대체하는 수학 모델.

#### 개념: Mixture (Cure) Model

모든 환자가 AE를 겪는 것이 아니다:
- 확률 `I` = incidence로 AE를 겪을 "운명"
- 확률 `1 - I`로 영원히 겪지 않음
- 겪는다면, onset 시점은 분포 `F(t)`를 따름

#### 수학적 유도

**onset까지 AE가 없을 확률:**
$$P(\text{no onset by } t) = (1 - I) + I \cdot (1 - F(t)) = 1 - I \cdot F(t)$$

**이산 확률 질량 (discrete PMF):**
$$f(t) = F(t) - F(t-1)$$

**일별 조건부 hazard:**
$$h(t) = \frac{I \cdot f(t)}{1 - I \cdot F(t-1)}$$

이것은 "아직 AE가 발생하지 않았다"는 조건 하에서, "오늘 AE가 발생할" 확률이다.

#### 구현

```python
def daily_onset_hazard(day, incidence, onset_spec):
    if incidence <= 0 or day < 1:
        return 0.0

    dist = onset_spec.get("distribution", "normal")
    params = onset_spec.get("params", onset_spec)

    F_t = _distribution_cdf(day, dist, params)      # CDF at day t
    F_prev = _distribution_cdf(day - 1, dist, params)  # CDF at day t-1
    f_t = max(0.0, F_t - F_prev)                    # PMF

    p_no_onset_yet = 1.0 - incidence * F_prev
    if p_no_onset_yet <= 1e-12:
        return 0.0  # 이미 거의 확실히 발생했어야 할 시점

    hazard = incidence * f_t / p_no_onset_yet
    return min(max(hazard, 0.0), 1.0)
```

#### 수치 예시: Peripheral Neuropathy

```
incidence = 0.56, onset ~ Normal(mean=63, std=21, min=7, max=180)

Day   F(t-1)   F(t)    f(t)     1-I·F(t-1)   hazard
  1   0.0000   0.0000  0.0000   1.0000        0.0000
 30   0.0473   0.0544  0.0071   0.9735        0.0041  (0.4%)
 42   0.1449   0.1632  0.0183   0.9189        0.0112  (1.1%)
 63   0.4591   0.4778  0.0187   0.7429        0.0141  (1.4%) ← 피크 근처
 84   0.7845   0.7967  0.0122   0.5607        0.0122  (1.2%)
120   0.9837   0.9870  0.0033   0.4491        0.0041  (0.4%)
180   1.0000   1.0000  0.0000   0.4400        0.0000
```

**핵심 관찰:**
- 초기에는 hazard가 매우 낮음 (아직 AE 발생 시기가 아님)
- 평균(63일) 근처에서 최고치 ~1.4%
- 이후 감소하지만, 분모도 줄어들어 **"아직 안 겪었으면 이제 곧"** 효과
- Day 180 이후 f(t)=0이므로 hazard=0 (더 이상 onset 가능성 없음)

#### 사용 흐름

```python
# daily_agent._check_new_ae_onsets(day, day_results)에서:
for ae in rule_set.ae_profile:
    if ae.ae_term not in occurred_aes and ae.ae_term not in resolved_aes:
        adjusted_incidence = ae_risks[ae.ae_term]  # LLM-calibrated
        cascade_mult = cascade_multipliers.get(ae.ae_term, 1.0)
        hazard = daily_onset_hazard(day, adjusted_incidence * cascade_mult, ae.onset_day)
        if sampler.boolean(hazard):
            # → AE onset!
            initial_grade = sample_initial_grade(ae.grade_distribution)
```

---

### 3.3 AE Resolution Hazard

#### 공식

AE가 활성화된 후, 매일의 해소 확률:

$$h_r(d) = \frac{f(d)}{1 - F(d-1)}$$

동일한 discrete hazard 공식이지만, incidence 파라미터 없음 (활성 AE는 100% "해소될 운명"이므로).

#### 특수 케이스

| 조건 | 반환값 | 이유 |
|------|--------|------|
| `duration_spec is None` | 0.0 | 비가역적 AE (탈모 등) |
| `duration_spec`이 단순 숫자 | d ≥ spec → 1.0, else 0.0 | 고정 지속기간 |
| `p_still_active ≤ 1e-12` | **0.8** | 예상 기간을 크게 초과 → 80%/일로 강제 해소 |

**0.8 매직 넘버의 의미:** 해소 확률이 분모 근사 0으로 비정상적으로 커지는 것을 방지. 동시에 "기대 지속기간을 크게 초과한 AE는 대부분 자연 해소된다"는 의학적 직관 반영.

---

### 3.4 Grade Transition 확률

활성 AE의 일별 grade 변화 확률:

#### 기본 상수 (config/defaults.py)

| 상수 | 값 | 의미 |
|------|---|------|
| `BASE_WORSEN` | 0.020 | 일별 악화 기본 확률 2% |
| `BASE_IMPROVE` | 0.005 | 일별 호전 기본 확률 0.5% |
| `TIME_STABILIZE_DAY` | 21 | 비누적 AE: 21일 후 안정화 |
| `HIGH_WORSEN_DAMPING` | 0.7 | Grade 3+ 악화 감쇠 |
| `GRADE_4_TO_5_DAMPING` | 0.3 | Grade 4→5 추가 감쇠 |
| `HIGH_IMPROVE_BOOST` | 1.3 | Grade 3+ 호전 부스트 |
| `GRADE_4_IMPROVE_BOOST` | 1.5 | Grade 4 호전 추가 부스트 |
| `MAX_TRANSITION_PROB` | 0.5 | 단일 전이 확률 상한 |

#### 알고리즘

```
1. Grade 5 (사망) → {worsen: 0, improve: 0, stable: 1} (흡수 상태)

2. Time factor: tf = min(days_active / 60, 2.0)  [0~2.0 범위, 60일에 포화]

3-a. 누적성 AE (peripheral neuropathy 등):
     worsen *= (1 + tf)         → 시간에 따라 최대 3배
     improve *= max(0.3, 1 - tf*0.3)  → 시간에 따라 최소 30%로 감소

3-b. 비누적 AE (nausea 등):
     21일 이후: worsen *= 0.6, improve *= 1.3  → 안정화

4. 고등급 감쇠:
     Grade 3+: worsen *= 0.7, improve *= 1.3
     Grade 4+: worsen *= 0.3, improve *= 1.5  (누적 적용)

5. 각각 MAX_TRANSITION_PROB(0.5) 상한 적용
6. stable = 1 - worsen - improve
```

#### 수치 예시

**Grade 2, 누적성 AE, Day 60:**
```
tf = min(60/60, 2.0) = 1.0
worsen = 0.020 * (1 + 1.0) = 0.040 (4%)
improve = 0.005 * max(0.3, 1 - 0.3) = 0.005 * 0.7 = 0.0035 (0.35%)
stable = 0.9565
```

**Grade 4, 누적성 AE, Day 60:**
```
tf = 1.0
worsen = 0.020 * 2.0 * 0.7 * 0.3 = 0.0084 (0.84%)
improve = 0.005 * 0.70 * 1.3 * 1.5 = 0.0068 (0.68%)
stable = 0.9848
```

Grade 4 → 5로의 전이는 일별 0.84% — 시뮬레이션 84일 동안 누적 ~50% 확률.

---

### 3.5 Tumor Change Model — Sigmoid Response

#### 핵심 공식

모든 반응 카테고리에 공통된 sigmoid onset:

$$\text{blend}(t) = 1 - \exp\left(-\frac{t}{\tau}\right)$$

여기서 `t` = 주(weeks), `τ` = lag_weeks / patient_scale.

#### 반응별 수학

| 반응 | τ (weeks) | 공식 | 범위 |
|------|----------|------|------|
| **CR** | 4.5/ps | `blend × (-95% × ps)` | max -100% |
| **PR** | 5.0/ps | `blend × (-55% × ps)` | [-80%, -30%] |
| **SD** | 4.0 | `(-5%×ps)×blend + 4%×sin(0.4t)×blend` | [-29%, +19%] |
| **PD** | 2.0/ps | `blend × 3.5×ps × weeks` | max 200% |

**patient_scale (ps):** LogNormal(0, 0.30) → 범위 [0.4, 2.5]. 환자간 종양 반응 속도 차이 모델링.

#### SD의 Sinusoidal 변동

안정 병변(Stable Disease)은 실제로 약간의 크기 변동이 있다:

$$\Delta\% = (-5\% \times ps) \times \text{blend} + 4\% \times \sin(0.4t) \times \text{blend}$$

- 기본 추세: 약간의 축소 (-5%)
- 주기적 변동: ±4% 진폭, 주기 ≈ 15.7주
- 두 항 모두 blend로 곱셈 → 초기에는 변동 없음, 점진적으로 나타남

---

### 3.6 Daily Mortality — 4-Channel + CSF

가장 복잡한 함수. 일별 사망 확률을 4개 채널의 곱과 Clinical Stability Factor(CSF)의 곱으로 계산.

#### Step 1: 연간→일별 변환

$$p_{daily} = 1 - (1 - \min(p_{annual}, 0.99))^{1/365}$$

Padcev+Pembro 방광암: annual = 0.25 → daily ≈ 0.000789

#### Step 2: Channel Multipliers

| 채널 | 핵심 인자 | 범위 |
|------|----------|------|
| Disease progression | PD: ×4.0, CR/PR: ×response_reduction(0.4) with lag | 0.4~4.0 |
| Treatment toxicity | G3: ×1.5, G4: ×3.0, concurrent AE threshold | 1.0~4.5 |
| ECOG | ECOG 0-1: ×1, 2: ×1.5, 3: ×2.5, 4: ×5 | 1.0~5.0 |
| Treatment discontinued | ×1.5 with 30-day half-life decay | 1.0~1.5 |

**Coherence check:** Grade 4+ AE 없고, ECOG < 3이고, PD 아니면 → treatment_toxicity multiplier를 1.0으로 리셋. "경미한 AE 축적에 의한 사망" 방지.

#### Step 3: Clinical Stability Factor (CSF)

3개 서브팩터의 **가중 기하 평균:**

$$\text{CSF} = \text{csf}_{tumor}^{0.40} \times \text{csf}_{ecog}^{0.35} \times \text{csf}_{ae}^{0.25}$$

| 서브팩터 | 가중치 | 값 범위 |
|---------|--------|--------|
| csf_tumor | 0.40 | PD: 1.0, SD: 0.05, PR: 0.02, CR: 0.02, unknown: 0.15 |
| csf_ecog | 0.35 | ECOG 0: 0.10, 1: 0.25, 2: 0.60, 3+: 1.0 |
| csf_ae | 0.25 | G4+: 1.0, G3: 0.25, multi-G2: 0.20, G2: 0.10, G0-1: 0.03 |

#### 수치 예시

**안정 환자 (PR + ECOG 0 + AE 없음):**
```
CSF = 0.02^0.40 × 0.10^0.35 × 0.03^0.25
    = 0.105 × 0.224 × 0.416 = 0.0098

daily_risk = 0.000789 × 1.0 (channels) × 0.0098 = 0.0000077
84일 누적 사망 확률 ≈ 0.065%
```

**위험 환자 (PD + ECOG 3 + Grade 4 AE):**
```
CSF = 1.0^0.40 × 1.0^0.35 × 1.0^0.25 = 1.0
channels = 4.0 × 3.0 × 2.5 × 1.0 = 30.0

daily_risk = min(0.000789 × 30.0 × 1.0, 0.5) = 0.0237
84일 누적 사망 확률 ≈ 86%
```

**상한:** `MAX_DAILY_MORTALITY = 0.5` — 어떤 상황에서도 일별 사망 확률 50% 초과 불가.

---

### 3.7 Dynamic ECOG

#### 점수 구성 (가산 모델)

```
base = baseline_ecog
+ AE burden: Σ max(0, grade-1) × weight, capped at 2.0
+ High-grade floor: G3→base+1, G4→base+2, G5→4
+ Treatment fatigue: min(day/21 × 0.05, 0.5)
+ Disease status: PD→+disease_weight×2, CR/PR with lag→negative
+ Comorbidities: 1+→+penalty, 3+→+penalty×2
+ Discontinued: +0.2
```

**Rate limiting:** 일별 변화량 ±1로 제한. 급격한 ECOG 변화 방지 → 현실적 패턴.

**반올림:** `int(score + 0.5)` — Python의 banker's rounding 회피.

---

### 3.8 Causal Lab Targets

Ornstein-Uhlenbeck 과정의 **목표값**을 결정하는 3-layer 모델:

#### Layer 1: AE → Lab

```
hepatotoxicity G3 → ALT target = baseline × 10.0
neutropenia G2 → ANC target = baseline × 0.5
```

AE가 없으면 target = baseline.

#### Layer 2: Cumulative Dose → Lab

$$\text{target} \times= \text{per\_100mg\_multiplier}^{dose/100}$$

지수 모델 — 누적 용량에 따라 기하급수적 영향.

#### Layer 3: CM Correction (보조약)

$$\text{target} = \text{target} - \text{correction} \times (\text{target} - \text{baseline})$$

보조약이 Lab 값을 baseline 쪽으로 당김. 예: insulin의 glucose correction = 0.6 → 편차 60% 감소.

---

### 3.9 AE Cascade Multipliers

인과 관계 AE 체인:

```
neutropenia G3+ → febrile_neutropenia × 3.0
hepatotoxicity G2+ → alt_increased × 2.0
```

여러 trigger가 동시 활성화되면 multiplier가 **곱셈으로 누적.**
결과는 `MAX_AE_CASCADE_HAZARD = 0.8`로 상한 적용.

---

### 3.10 Discontinuation Risk — 2-Channel + Background

#### Channel 1: Patient Withdrawal

$$p_{patient} = \min(\text{base\_rate} \times \prod \text{risk\_factors}, 0.05)$$

Risk factors: G3+ AE (×2.5), ECOG 악화 (×2.0), 12주 초과 (×1.5), 부실 반응 (×1.3)

#### Channel 2: Physician Decision

$$p_{physician} = \min(\text{base\_rate} \times \prod \text{risk\_factors}, 0.03)$$

Risk factors: ECOG ≥ 3 (×3.0), 2+ 감량 (×2.0), 부실 반응 (×2.0), 중증 AE (×2.0)

#### Background

고정 일별 0.024% — 기록 누락, 전원, 기타 비의학적 이유.

치료가 이미 중단된 경우 모든 채널 0 반환.