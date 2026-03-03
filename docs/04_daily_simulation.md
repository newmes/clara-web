# 04. Phase 2: Daily Simulation (DailySimulator + Orchestrator)

> **파일:** `src/agents/daily_agent.py` (~3,050 lines), `src/orchestrator.py` (~1,100 lines)
> **역할:** 환자의 일별 상태를 hazard function 기반으로 동적 생성
> **LLM 호출:** 초기화 1회 + 이벤트 날만 (전체의 ~30-40%)
>
> 🔗 **웹에서 확인:** [Trial Viewer](http://49.254.130.90:9000/trial/20260216_062601_Padcev___Pembrolizumab_10pt_84d/) — Day-by-Day 시뮬레이션 결과를 Map + Dashboard로 탐색

---

## 1. DailySimulator 클래스 — 상태 관리

### 1.1 인스턴스 = 환자 1명의 전체 시뮬레이션

```python
simulator = DailySimulator(rule_set, patient, sampler, model, actual_duration)
```

생성자에서 다음이 실행됨:
1. 환자 데이터 파싱 (demographics, comorbidities, weight, baseline)
2. **LLM 호출: `_calibrate_ae_risks()`** — 환자별 AE 발생률 보정
3. **코드만: `_sample_tumor_response()`** — 종양 반응 카테고리 결정
4. 투약 스케줄 파싱 (약물명 매칭, 사이클일 추출)
5. RECIST 스캔 스케줄 생성
6. 약물 분류 (IO vs ADC/chemo)
7. 용량 조절 규칙 파싱
8. 보조약 규칙 매핑
9. Composite model 파라미터 추출

### 1.2 추적 상태 (13개 카테고리)

| 카테고리 | 변수 | 설명 |
|---------|------|------|
| AE 추적 | `occurred_aes`, `resolved_aes` | 발생/해소 이력 |
| 용량 추적 | `cumulative_doses`, `dose_levels` | 약물별 누적 용량, 현재 감량 수준 |
| 약물 상태 | `held_drugs`, `discontinued_drugs` | 약물별 보류/중단 |
| 전체 중단 | `discontinuation_day`, `hold_reason` | 전체 치료 중단 정보 |
| 보조약 | `active_cm` | 활성 concomitant medication 목록 |
| 사망 | `is_deceased`, `death_day`, `death_cause` | 사망 상태 |
| ECOG | `baseline_ecog`, `current_ecog` | 수행능력 |
| Baseline | `baseline_labs`, `baseline_vitals` | OU 과정의 mean-reversion target |
| RECIST | `recist_scan_days`, `recist_results`, `best_response_pct` | 종양 평가 |
| Disposition | `ds_record` | 중도탈락 CRF |
| Cycle | `frozen_cycle`, `frozen_cycle_day` | 중단 시 사이클 동결 |

---

## 2. AE Risk Calibration (LLM 1회)

### 2.1 Two-Stage 보정

```
Stage 1: 코드 기반 (hazard.adjust_incidence_by_risk_modifiers)
  rule_set.risk_modifiers 적용
  예: 당뇨 → peripheral_neuropathy × 1.4

Stage 2: LLM 기반 (fine-tuning)
  환자의 전체 프로필 (demographics + comorbidities + ECOG + BMI + 흡연)
  → LLM이 복합적 상호작용 고려하여 최종 incidence 조정
  예: "72세 + CKD + 당뇨 → 신독성 위험 시너지 효과" → nephrotoxicity ×1.8
```

### 2.2 LLM 호출 Context

```
Patient Profile:
- Age: 72, Sex: M, BMI: 27.3, Smoking: former, ECOG: 1
- Comorbidities: hypertension, CKD, diabetes
- Drug: Padcev + Pembrolizumab

Code-adjusted incidences:
  peripheral_neuropathy: 0.78 (base 0.56 × 1.4 diabetes)
  neutropenia: 0.35
  rash: 0.38
  ...

Fine-tune considering:
- Comorbidity synergies
- Age-related pharmacokinetic factors
- Drug-drug interactions
```

### 2.3 Fallback

LLM 실패 시: Stage 1 코드 기반 값만 사용 (경고 로그 출력). Hard failure 아님 — 시뮬레이션 진행 가능.

---

## 3. Tumor Response Sampling (코드만)

```python
def _sample_tumor_response(self):
    # 1. 반응 카테고리: rule_set.tumor_response_distribution에서 categorical 샘플링
    response = sampler.categorical({"CR": 0.10, "PR": 0.44, "SD": 0.30, "PD": 0.16})
    
    # 2. 반응 시작일: Normal(mean=3×cycle, std=cycle, min=2×cycle, max=8×cycle)
    onset_day = sampler.numeric("normal", {mean: 63, std: 21, min: 42, max: 168})
    
    # 3. 환자 개인차: LogNormal(mu=0, sigma=0.30) → [0.4, 2.5]
    patient_scale = sampler.numeric("lognormal", {mu: 0, sigma: 0.30, min: 0.4, max: 2.5})
    
    return response, onset_day, patient_scale
```

**patient_scale:** 약물 대사, 종양 생물학의 개인차를 모델링. ps=0.5인 환자는 평균보다 느리게 반응, ps=2.0인 환자는 빠르게 반응.

---

## 4. generate_day() — 10-Step Pipeline

이것이 시뮬레이션의 핵심 루프. 매일 1번 호출.

```
Step 0:  Cycle Freeze     치료 중단 시 사이클 동결
Step 1:  AE Onset         새 AE 발생 (hazard function)
Step 2:  AE Changes       활성 AE grade 변화/해소 (hazard function)
Step 3:  Tumor Change     종양 크기 변화 (sigmoid model)
Step 3b: RECIST Eval      스캔일이면 RECIST 1.1 평가
Step 4:  Event Aggregation 이벤트 수집
Step 5:  LLM or Code      Event Day → LLM / Quiet Day → 코드만
Step 6:  CRF Enrichment   EC, CM 등 CRF 필드 생성
Step 7:  Baseline Capture  Day 1에 baseline 저장
Step 8:  AE Cascade       AE 간 인과 관계 multiplier 갱신
Step 9:  Dynamic ECOG     수행능력 재계산
Step 10: Mortality        일별 사망 확률
Step 11: Discontinuation  중도탈락 확률
```

### 4.1 Step 0: Cycle Freeze

```python
if self.treatment_discontinued:
    cycle = self.frozen_cycle       # 중단 시점의 사이클
    cycle_day = self.frozen_cycle_day  # 중단 시점의 사이클일
```

치료 중단 후에도 시뮬레이션은 계속되지만, 사이클이 진행하지 않음. 이는 실제 임상시험의 "follow-up period"를 모델링.

### 4.2 Step 1: New AE Onset

```python
for ae in rule_set.ae_profile:
    if ae not in occurred_aes and ae not in resolved_aes:
        incidence = ae_risks[ae.ae_term]  # LLM-calibrated
        cascade_mult = cascade_multipliers.get(ae.ae_term, 1.0)
        
        hazard = daily_onset_hazard(day, incidence * cascade_mult, ae.onset_day)
        
        if sampler.boolean(hazard):
            initial_grade = sample_grade(ae.grade_distribution)
            occurred_aes[ae.ae_term] = {
                onset_day: day,
                grade: initial_grade,
                status: "active",
                days_active: 0
            }
```

### 4.3 Step 2: Active AE Changes

각 활성 AE에 대해:

```python
for ae_term, ae_state in occurred_aes.items():
    if ae_state.status != "active": continue
    
    # 해소 체크
    duration_spec = rule_set.get_ae_duration(ae_term)
    res_hazard = daily_resolution_hazard(ae_state.days_active, duration_spec)
    if sampler.boolean(res_hazard):
        ae_state.status = "resolved"
        resolved_aes.add(ae_term)
        continue
    
    # Grade 전이
    probs = grade_transition_probs(ae_state.grade, ae_state.days_active, ae.cumulative)
    transition = sampler.categorical({
        "worsen": probs.worsen,
        "improve": probs.improve,
        "stable": probs.stable
    })
    
    if transition == "worsen":
        ae_state.grade = min(ae_state.grade + 1, 5)
    elif transition == "improve":
        ae_state.grade = max(ae_state.grade - 1, 1)
```

### 4.4 Step 5: Event Day vs Quiet Day

**분기 조건:**
```python
is_event = bool(all_events) or is_hospital or len(day_results) == 0
```

| 조건 | Event Day | Quiet Day |
|------|-----------|-----------|
| 새 AE onset | O | |
| AE grade 변화 | O | |
| AE 해소 | O | |
| RECIST 스캔 | O | |
| 병원 방문일 | O | |
| Day 1 | O | |
| 아무 이벤트 없음 | | O |

**Event Day 처리:**
```
_generate_event_day(day, cycle, cycle_day, is_hospital, events, day_results)
  → LLM 호출: 전체 환자 상태 (labs, vitals, subjective, narrative) 생성
  → _normalize_llm_result(): LLM 출력 형식 보정
  → AE state sync: 코드 추적값으로 LLM 값 덮어쓰기
  → Tumor value sync: 코드 계산값으로 덮어쓰기
  → Location/Treatment status correction
```

**Quiet Day 처리:**
```
_generate_quiet_day(day, cycle, cycle_day, is_hospital, day_results)
  → LLM 호출 없음
  → Lab: Ornstein-Uhlenbeck 과정 (baseline 방향 mean-reversion + 노이즈)
  → Vitals: 마찬가지로 OU + 보조약 영향
  → Tumor: sigmoid 공식 적용
  → Location: 이전 값 유지 (HOME 또는 OUTPATIENT)
```

**LLM 출력 보정 (5단계):**

| 단계 | 무엇을 | 왜 |
|------|--------|---|
| Step 5-pre: AE State Sync | AE의 onset_day, grade, status를 코드값으로 강제 | LLM이 AE 상태를 "잊거나" 변조하는 경우 방지 |
| Step 5a: Tumor Sync | tumor change %를 코드 계산값으로 강제 | LLM이 RECIST 수학과 불일치하는 값 생성 방지 |
| Step 5b: Location Fix | Day 1은 OUTPATIENT, 투약일은 HOME→OUTPATIENT | LLM이 잘못된 location 배정 방지 |
| Step 5c: Treatment Status Sync | 내부 약물 추적 상태 반영 | LLM이 중단된 치료를 "계속"으로 표시 방지 |
| Step 5d: Days Active Sync | days_active = day - onset_day 계산 | LLM이 이 필드를 자주 누락 |

---

## 5. Lab Evolution — Ornstein-Uhlenbeck 과정

Quiet Day에서 lab 값의 변동을 모델링하는 확률 과정.

### 5.1 기본 OU 공식

$$x_{t+1} = x_t + \theta \cdot (target - x_t) + \sigma \cdot \mathcal{N}(0, 1)$$

- `θ` (mean-reversion speed): 0.05~0.15
- `target`: `compute_causal_lab_target()`의 3-layer 모델에서 결정
- `σ`: lab별 변동성 (ANC는 크고, 크레아티닌은 작음)

### 5.2 Target 결정 우선순위

```
1. 활성 AE → 해당 lab target (grade 의존적)
2. 누적 용량 → target 배수
3. 보조약 → target을 baseline 방향으로 보정
4. 위 모든 것 없으면 → target = baseline
```

### 5.3 CTCAE-guided Lab Convergence (Event Day)

Event Day에서 LLM이 생성한 lab 값이 활성 AE의 CTCAE grade와 불일치할 때:

```python
# AE가 G3 hepatotoxicity (ALT > 5×ULN)인데 LLM이 ALT = 80 (정상) 생성
# → CTCAE G3 midpoint으로 일별 delta 내에서 강제 수렴
target_midpoint = (5 * ULN + 10 * ULN) / 2
delta = min(abs(target_midpoint - current), MAX_DAILY_LAB_DELTA)
```

---

## 6. Dose Modification — HR 기반 의사 결정

### 6.1 핵심 원칙: 의사는 GT를 모른다

```python
def apply_hospital_dose_modifications(self, observed_aes, ...):
    # observed_aes는 Hospital Record에서 온 것 — Ground Truth가 아님!
    # HR에는 감지된 AE만 있고, grade가 distortion 되었을 수 있음
```

이것이 GT/HR 분리의 실질적 효과:
- **Natural mode:** 병원 방문일에만 AE를 감지 → 감지 지연 → 늦은 용량 조절
- **Care AI mode:** 매일 영상통화 → AE 조기 감지 → force_hospital_tomorrow → 빠른 용량 조절

### 6.2 약물별 타겟팅

```python
def _get_causative_drugs(self, ae_term):
    # IO-specific AEs (hepatitis, colitis, pneumonitis 등)
    #   → IO drugs만 대상 (Pembrolizumab)
    # ADC/chemo-specific AEs (neuropathy, neutropenia 등)
    #   → non-IO drugs만 대상 (Padcev)
    # Shared AEs (rash 등)
    #   → 모든 약물 대상
```

### 6.3 Action 실행

| Grade Action | 실행 |
|-------------|------|
| `DOSE NOT CHANGED` | 아무것도 하지 않음 |
| `DOSE REDUCED` | `dose_levels[drug] -= 1단계` (1.0→0.75→0.5) |
| `DRUG INTERRUPTED` | `held_drugs.add(drug)`, G3+면 미리 1단계 감량 |
| `DRUG WITHDRAWN` | `discontinued_drugs.add(drug)`, `held_drugs.discard(drug)` |

### 6.4 Hold Release

```python
# 보류 원인 AE가 더 이상 활성이 아니면 → 보류 해제
for drug in held_drugs:
    if hold_reason_ae not in active_ae_terms:
        held_drugs.discard(drug)
        # EC record에 RESUMED 이벤트 추가
```

---

## 7. Orchestrator — 시뮬레이션 루프

### 7.1 Natural Mode 루프

```
for day in range(1, total_days + 1):
    cycle, cycle_day = get_cycle_info(day, cycle_length)
    is_hospital = is_hospital_day(day, cycle_length, admin_cycle_days)
    
    # 1. GT 생성 (hazard 기반)
    day_result = simulator.generate_day(day_results, day, cycle, cycle_day, is_hospital)
    day_result["care_record"] = []  # Natural: Care AI 없음
    
    # 2. GT → HR 변환
    obs_result = observation_model.process_day(ground_truth, day, is_hospital, is_admin_day)
    
    # 3. 병원 방문일이면 용량 조절
    if is_hospital:
        observed_aes = hospital_record.objective.active_aes  # HR에서 관찰된 AE만!
        modifications = simulator.apply_hospital_dose_modifications(observed_aes)
    
    # 4. JSONL 저장
    save_day(day_result)
    
    # 5. 사망/중단 체크
    if is_deceased: break
    if is_discontinued: run_followup(remaining_days) ; break
```

### 7.2 Care AI Mode 추가 요소

```
for day in range(1, total_days + 1):
    # ... (1-2는 동일)
    
    # 2.5. Care AI 영상통화 (매일)
    care_result = care_agent.conduct_video_call(day_result, last_hospital_record)
    day_result["care_record"] = care_result
    
    # 2.6. 개입 적용
    apply_interventions(care_result, simulator)
    
    # 2.7. 에스컬레이션
    if care_result.actions.escalate_to_physician:
        if urgency == "emergency":
            same_day_er = True  # 오늘 병원 방문으로 변경
        else:
            force_hospital_tomorrow = True  # ★ 핵심: 내일 병원 방문 강제
    
    # 3. 병원 방문 (스케줄 + force_hospital)
    is_hospital = is_scheduled_hospital or force_hospital_tomorrow or same_day_er
    if is_hospital:
        modifications = simulator.apply_hospital_dose_modifications(observed_aes)
    
    force_hospital_tomorrow = False  # 리셋
```

### 7.3 force_hospital_tomorrow — Care AI 가치의 핵심 메커니즘

```
Natural:
  Day 42: AE onset (neuropathy G2) — 환자는 HOME, 병원 모름
  Day 43: HOME (quiet day) — 악화 위험
  Day 44-62: HOME — 여전히 모름 (다음 사이클까지 19일)
  Day 63: 병원 방문 — 21일 지연 감지, 이미 G3로 악화
  → 21일 지연

Care AI:
  Day 42: AE onset (neuropathy G2) — 영상통화에서 감지!
  Day 42: force_hospital_tomorrow = True
  Day 43: 강제 병원 방문 — 1일 지연 감지, 아직 G2
  → dose reduction 즉시 적용 → G3 악화 예방
```

### 7.4 A/B Seed 분리

```python
# Natural mode: seed base
natural_sampler = Sampler(seed=(base_seed or 200) + patient_num)

# Care AI mode: seed + 10000 (독립 난수열)
care_sampler = Sampler(seed=(base_seed or 200) + patient_num + 10000)
```

독립 seed → Care AI의 존재가 GT 생성에 영향을 미치지 않음. 비교의 공정성 보장.

추가 offset: mood (+20000), observation (+30000), care agent (+40000).

### 7.5 Follow-up Period

치료 중단 후 최대 30일의 추적관찰:

```python
followup_days = min(remaining_days, 30)
for d in range(day + 1, day + followup_days + 1):
    result = simulator.generate_day(...)  # 치료 없이 GT 계속 생성
    obs_result = observation_model.process_day(...)
    # Care AI mode: 영상통화도 계속
    if is_deceased: break
```

---

## 8. 병렬 실행

### ThreadPoolExecutor 기반

```python
def run_parallel(patients, total_days, mode, max_workers=10):
    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as pool:
        futures = {
            pool.submit(_run_single_patient, patient, ...): patient_id
            for patient in patients
        }
        for future in as_completed(futures):
            results[pid] = future.result()
```

### 스레드 안전성

- `_run_patient_silent()`: stdout 대신 `lines` 버퍼에 출력 → 스레드간 출력 섞임 방지
- 각 스레드가 독립 `Sampler`, `MoodState`, `ObservationModel` 인스턴스 보유
- `_print_lock`으로 최종 요약 출력 동기화

---

## 9. CLI 실행

```bash
# 기본: 1명, 21일, Natural
python src/run_simulation.py

# 10명, 84일, Natural + Care AI 비교
python src/run_simulation.py --patients 10 --days 84 --mode both --workers 5

# 규칙 재사용 + 시드 고정
python src/run_simulation.py --patients 5 --days 42 --seed 42 --skip-rules

# 다른 약물
python src/run_simulation.py --drug "Ozempic" --indication "type 2 diabetes" --patients 5
```

### 출력 디렉토리 구조

```
data/runs/20260216_143052_Padcev___Pembrolizumab_10pt_84d/
├── rule_set.json
├── patients/
│   ├── PT-001.json
│   ├── PT-002.json
│   └── ...
└── simulations/
    ├── PT-001_natural.jsonl      # Day 1~84, 한 줄당 1일
    ├── PT-001_care_ai.jsonl
    ├── PT-002_natural.jsonl
    └── ...
```

### _TeeWriter — 이중 출력

```python
class _TeeWriter:
    def write(self, text):
        self._terminal.write(text)    # 터미널 출력
        self._log_file.write(text)    # 파일 저장
        self._log_file.flush()        # 즉시 flush
```

모든 console 출력이 `logs/console_{run_name}.log`에도 저장.