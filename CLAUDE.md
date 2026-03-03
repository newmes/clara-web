# CLAUDE.md — Clinical Trial Simulation Engine v2.3 (Hospital Record 기반)

## 한 줄 요약
**약물명 하나를 입력하면, LLM이 확률 규칙을 생성하고, hazard function이 매일의 이벤트를 동적으로 결정하며, 치료 결정은 병원 기록(Hospital Record) 기반으로만 이루어지는 3-Phase 시뮬레이션 엔진.**

## 목적
- MedGemma Impact Challenge (마감: 2026-02-24, $100K)
- 핵심 데모: Padcev + Pembrolizumab 방광암 시험에서 AI 간호사의 AE 조기 감지 가치 입증
- Drug-agnostic: 항암제 외 다른 약물에도 확장 가능

---

## 아키텍처: 3-Phase (No Fate Table)

### 핵심 원리

```
LLM은 "확률 규칙을 정하는 역할"     → 의학 지식 활용 (Phase 0, 1)
코드는 "주사위를 굴리는 역할"        → hazard function + rand, 분포 보장
LLM은 "상세를 채우는 역할"          → 이벤트 날에만 호출 (Phase 2)
```

### ★ v2.3 핵심 변경: Ground Truth ↔ Hospital Record 분리

```
Ground Truth (GT):
  - hazard function이 매일 계산하는 "실제 환자 상태"
  - AE 발생, grade 변화, 종양 크기 등
  - 병원도 환자도 모르는 절대적 진실

Hospital Record (HR):
  - 병원이 "관찰 시점에 파악한" 환자 상태
  - 병원 방문일에만 전체 검사 (labs, vitals, physical exam)
  - 비방문일에는 환자 자발 보고, Care AI 영상통화 등으로만 일부 파악
  - ★ 모든 치료 결정(dose hold/reduce/withdraw)은 HR 기반으로만 발생

왜 이렇게 바꿨나?
  이전: GT에서 AE 발생 → 즉시 dose modification (신의 시점)
  현재: GT에서 AE 발생 → 다음 병원 방문까지 모름 → 방문 시 의사가 발견 → 조치
  → Care AI가 조기 감지 → 조기 내원 → 빠른 조치 — 이 가치가 드러남
```

### 파이프라인

```
입력: drug_name + indication (필수)

Phase 0: 규칙 발견 (약물당 1회)
═══════════════════════════════
  Rule Agent (LLM) ──→ rule_set.json
       확률 테이블: demographics, comorbidities,
       AE incidence + onset distributions, efficacy, ...

Phase 1: 환자 생성 (환자당)
═══════════════════════════════
  ┌─────────────────────────────────────────────────┐
  │ Step 1: demographics ← rule_set에서 rand() 샘플링 │
  │ Step 2: comorbidities ← LLM 확률 조정 → rand()   │
  │ Step 3: baseline labs/vitals ← LLM 생성           │
  │ Step 4: persona ← rand() + LLM 생성              │
  └─────────────────────────────────────────────────┘

Phase 2: 일별 시뮬레이션 (Day 1 ~ N)
═══════════════════════════════
  ┌─ Day K ─────────────────────────────────────────┐
  │                                                   │
  │ Step 1: GT 생성 (hazard → rand)                   │
  │   AE onset, grade 변화, 종양 변화                  │
  │   ★ dose modification 없음 (GT는 순수 생물학)      │
  │                                                   │
  │ Step 2: 이벤트 판정                               │
  │   이벤트 있음 → LLM이 전체 상태 생성 [Event Day]   │
  │   이벤트 없음 → 코드가 소폭 변동 적용 [Quiet Day]  │
  │                                                   │
  │ Step 3: Care AI 영상통화 [Care AI 모드만]          │
  │   T1: Patient → T2: Nurse → T3: Patient → T4: Nurse│
  │   시각/청각 AE 감지, 환자 보고 청취                 │
  │                                                   │
  │ Step 4: Observation Model                         │
  │   GT → Hospital Record 변환 (관찰 가능한 것만)     │
  │   병원 방문일: 전체 검사 (labs, vitals, exam)       │
  │   비방문일: patient_reported, video_detectable만   │
  │                                                   │
  │ Step 5: 치료 결정 [병원 방문일만]                   │
  │   ★ Hospital Record의 known_aes 기반              │
  │   simulator.apply_hospital_dose_modifications()   │
  │   → dose hold/reduce/withdraw/resume 결정         │
  │                                                   │
  │ Step 6: Mood 업데이트                             │
  │   7차원 심리 모델 → 보고 행동에 영향               │
  └─────────────────────────────────────────────────┘
```

---

## 핵심 모듈

### engine/ — 확률 엔진 (LLM 독립)
| 파일 | 역할 |
|------|------|
| `engine/sampler.py` | 난수 생성기. categorical, boolean, numeric, multi_boolean. 시드 기반 재현 가능. |
| `engine/prob_engine.py` | LLM→rand→LLM 패턴 구현. `estimate_probabilities()`, `generate_details()`. |
| `engine/hazard.py` | ★ **Hazard function 수학 모듈**. AE onset/resolution/grade 일별 확률 계산. 순수 코드. |
| `engine/observation.py` | ★ **GT ↔ Hospital Record 관찰 모델**. AE 감지 채널, 관찰 시점 관리, 화이트리스트 방식. |
| `engine/mood.py` | **7차원 심리 모델**. anxiety, depression, fatigue, irritability, hopefulness, defensiveness, trust. |

### agents/ — LLM Agent
| 파일 | 역할 | LLM 호출 |
|------|------|---------|
| `agents/rule_agent.py` | 약물별 규칙 발견 + 확률 테이블 생성 | 약물당 1회 |
| `agents/patient_agent.py` | 환자 생성 (LLM→rand→LLM) | 환자당 2-3회 |
| `agents/daily_agent.py` | ★ **일별 GT 시뮬레이션** (hazard 기반, dose mod는 외부) | 초기화 1회 + 이벤트 날만 |
| `agents/care_agent.py` | ★ **Care AI 간호사** (4-turn 영상통화, 감지+보고만) | 매일 2-4회 |
| `agents/llm_client.py` | Google Gemini API 공통 호출 | — |

---

## Observation Model (★ v2.3 핵심)

### 2-Layer 데이터 모델

```
┌─────────────────────────────────────────────────┐
│ Ground Truth (GT)                                │
│   hazard function이 매일 생성                    │
│   active_aes, labs, vitals, tumor — 절대값       │
│   ★ 치료 결정에 직접 사용하지 않음               │
├─────────────────────────────────────────────────┤
│ Hospital Record (HR)                             │
│   관찰 시점에만 GT에서 선택적으로 갱신           │
│   known_aes, known_labs, known_vitals — stale OK │
│   ★ 모든 치료 결정의 유일한 근거                 │
└─────────────────────────────────────────────────┘
```

### AE 감지 채널 (observation.py: AE_DETECTION_CHANNELS)

```
lab:               혈액 검사로만 감지 (neutropenia, thrombocytopenia 등)
patient_reported:  환자가 증상을 느끼고 보고 (nausea, pain 등)
video_detectable:  Care AI 영상통화로 시각/청각 감지 (rash, alopecia, cough 등)
physical_exam:     의사 신체검진 (neuropathy 등)
```

### 관찰 시점 (Observation Points)

```
scheduled_visit:   투약일 외래 → 전체 검사 (labs, vitals, exam, AE assessment)
scheduled_scan:    RECIST 영상 → 종양 정보
self_report:       환자 자발 보고 (mood 기반 확률) → patient_reported AE만
video_call:        Care AI 영상통화 → video_detectable + patient_reported AE
er_visit:          응급실 (Grade 4+ 또는 위험 vitals) → 전체 검사
```

### Hospital Record 화이트리스트 (GT fallback 없음)

```python
hospital_record = {
    "objective": {
        "location": ...,
        "treatment_status": known_treatment_status,  # ★ GT가 아닌 HR 자체 관리
        "labs": known_labs,          # 병원 방문 시에만 갱신
        "vitals": known_vitals,      # 병원 방문 시에만 갱신
        "active_aes": hr_aes,        # 감지된 AE만 (채널 기반)
        "ecog": known_ecog,          # 병원 방문 시에만 갱신
        "tumor": known_tumor,        # 스캔 시에만 갱신
    },
    "EC": ...,  # 투약 기록 — 항상 정확 (약사 관리)
    "CM": ...,  # 병용약 — 항상 정확
    "DS": ...,  # disposition — 항상 정확
    "AE": ...,  # 감지된 AE의 CRF 레코드만
}
# ★ GT의 다른 필드는 HR에 자동 복사되지 않음
```

---

## Dose Modification 흐름 (★ v2.3)

```
이전 (v2.2):
  generate_day() 내부에서 GT 이벤트 → 즉시 _apply_dose_modifications()
  → 신의 시점: AE 발생 즉시 조치 (비현실적)

현재 (v2.3):
  generate_day() → GT만 생성 (dose mod 없음)
  observation_model.process_day() → GT → HR 변환
  if 병원 방문일:
    simulator.apply_hospital_dose_modifications(HR의 known_aes)
    simulator.patch_day_treatment_status(day_result)
    observation_model.update_treatment_status(new_status)
  → 의사가 병원에서 관찰한 것만으로 판단 (현실적)
```

### DailySimulator 공개 API

```python
simulator = DailySimulator(rule_set, patient, sampler, model)

# 매일: GT 생성 (dose modification 없음)
result = simulator.generate_day(day_results, day, cycle, cycle_day, is_hospital)

# 병원 방문일만: HR 기반 치료 결정
dose_changes = simulator.apply_hospital_dose_modifications(
    observed_aes,  # hospital_record["objective"]["active_aes"]
    day, cycle, cycle_day,
)
simulator.patch_day_treatment_status(result)  # 치료 상태 갱신
```

---

## Care AI 가치 입증 (핵심 목표)

```
Run A (Natural): Care AI 없음
  → 병원 방문 간격 (10~13일) 동안 AE 사각지대
  → 다음 방문까지 AE 악화 가능 → 늦은 발견 → 늦은 조치
  → dose hold 적음 (발견 자체가 늦으므로)

Run B (Care AI): 매일 4-turn 영상통화
  → 시각/청각 AE 매일 감지 (rash, alopecia, cough 등)
  → 조기 감지 → recommend_early_visit → 빠른 병원 방문
  → 의사가 빨리 확인 → 빠른 dose modification → AE 악화 방지

Care AI의 역할 (명확):
  ✓ 감지(detect) + 보고(report) + 조기 내원 연결(refer)
  ✗ 치료 결정(dose hold/discontinue)은 의사만 가능

비교 방법: 대수의 법칙 (N ≥ 50 권장)
  같은 seed로 시작하나, 이후 확률은 독립적으로 전개
  통계적 유의성으로 Care AI 가치 입증
```

---

## Hazard Function (핵심 수학)

Fate table 대신, rule_set의 onset 분포에서 매일의 AE 발생 확률을 계산한다.

### 혼합 모델 (Mixture Model)

```
AE가 발생할 전체 확률: I = incidence_all_grade (환자별 조정됨)
발생한다면 시점 분포: F(t) = onset CDF (Normal, LogNormal, Uniform)

일별 onset hazard:
  P(onset day t | no onset before t) = I·f(t) / (1 − I·F(t−1))

  여기서 f(t) = F(t) − F(t−1) (이산 확률 질량)
```

### AE Grade 전이

```
활성 AE의 daily grade 전이 확률:
  base_worsen = 0.015 (1.5%)
  base_improve = 0.005 (0.5%)

  누적 독성 → 시간에 따라 worsen↑
  Grade 3+ → worsen↓, improve↑
  Care AI 개입 → worsen×0.3, improve×3.0
```

---

## 환자 심리 모델 (Mood — 7차원)

```
anxiety:       불안 (0~1). 높으면 과보고, 조기 ER 방문
depression:    우울 (0~1). 높으면 보고 동기 감소
fatigue:       피로 (0~1). 높으면 통화 참여도 감소
irritability:  짜증 (0~1). 높으면 조기 통화 종료
hopefulness:   희망 (0~1). 높으면 치료 순응도 증가
defensiveness: 방어적 (0~1). 높으면 축소 보고 (Grade 과소 평가)
trust:         신뢰 (0~1). 높으면 정확한 보고, 통화 참여도 증가

persona별 baseline 설정, 이벤트에 따라 daily 업데이트,
Care AI 통화 turn마다 미세 조정
```

---

## 파일 구조

```
ClinicalTrialEngine/
├── CLAUDE.md                   ← 지금 읽고 있는 파일
│
├── src/
│   ├── engine/                 ← ★ 확률 엔진 + 관찰 모델
│   │   ├── sampler.py          ← 난수 생성기
│   │   ├── prob_engine.py      ← LLM→rand→LLM 패턴
│   │   ├── hazard.py           ← hazard function
│   │   ├── observation.py      ← ★ GT ↔ HR 관찰 모델 (v2.3)
│   │   └── mood.py             ← 7차원 심리 모델
│   │
│   ├── agents/
│   │   ├── llm_client.py       ← Gemini API 공통 호출
│   │   ├── rule_agent.py       ← Phase 0: 규칙 발견
│   │   ├── patient_agent.py    ← Phase 1: 환자 생성
│   │   ├── daily_agent.py      ← Phase 2: GT 생성 + HR 기반 dose mod
│   │   └── care_agent.py       ← Care AI: 4-turn 영상통화
│   │
│   ├── multimodal/             ← Multimodal AE 감지 (SigLIP + HeAR)
│   ├── orchestrator.py         ← 3-Phase 시뮬레이션 루프
│   ├── run_simulation.py       ← CLI 실행
│   ├── logger.py               ← 로깅 (logs/ 디렉토리)
│   ├── context_manager.py      ← 컨텍스트 압축
│   └── validator.py            ← 스키마 검증
│
├── data/
│   ├── rule_set.json           ← Rule Agent 출력
│   └── runs/                   ← 실험별 폴더
│       └── {timestamp}_{drug}_{N}pt_{D}d/
│           ├── rule_set.json   ← 복사본
│           ├── patients/       ← 환자 JSON
│           └── simulations/    ← 일별 JSONL (natural/care_ai)
│
├── logs/                       ← ★ 모든 로그 저장 (logger.py 관리)
│   ├── sim_{ts}.log            ← 시뮬레이션 로그
│   └── sim_{ts}.stats.json     ← LLM 호출 통계
│
├── frontend/                   ← Django 웹 뷰어
│   └── viewer/views/           ← views 패키지 (기능별 분리)
│       ├── __init__.py         ← 전체 view re-export
│       ├── _helpers.py         ← 공통 데이터 유틸리티
│       ├── core.py             ← landing, simulation_list
│       ├── trial.py            ← trial viewer, patient state
│       ├── demo.py             ← 데모 페이지, antihallu API
│       ├── compare.py          ← A/B 비교 대시보드
│       ├── sim_api.py          ← 시뮬레이션 시작/중지/상태
│       ├── doc.py              ← doc agent, SAE, CRF, 통계
│       ├── care_api.py         ← care agent 데모 API
│       ├── medgemma.py         ← MedGemma API
│       ├── stats.py            ← 통계 분석 + 챗봇
│       ├── ruleset.py          ← ruleset 생성/비교 API
│       └── map.py              ← 맵 생성 API
│
├── schemas/                    ← JSON 스키마
└── prompts/                    ← 시스템 프롬프트
```

---

## 실행 방법

```bash
# 기본: Padcev+Pembro, 환자 1명, 21일
python src/run_simulation.py

# A/B 비교 (Natural + Care AI 동시 실행, 같은 환자)
python src/run_simulation.py --patients 10 --days 84 --seed 42 --mode both --skip-rules

# Natural만
python src/run_simulation.py --patients 10 --days 84 --seed 42 --mode natural --skip-rules

# Care AI만
python src/run_simulation.py --patients 10 --days 84 --seed 42 --mode care_ai --skip-rules

# 다른 약물 (drug-agnostic)
python src/run_simulation.py --drug "Ozempic" --indication "type 2 diabetes" --patients 5
```

---

## 설계 원칙

1. **LLM은 확률을 정하고, 코드가 주사위를 굴린다** — mode collapse 방지, 분포 보장
2. **GT와 HR을 명확히 분리** — 치료 결정은 HR 기반만, GT fallback 금지
3. **Drug-agnostic** — 약물명만 바꾸면 질환/약물 구조를 LLM이 자동 파악
4. **Fate table 없음** — hazard function으로 매일 동적 결정
5. **조용한 날 최적화** — 이벤트 없는 날은 LLM 호출 없이 코드만으로 처리
6. **재현 가능** — 시드 고정으로 동일 시뮬레이션 재현
7. **Fallback 금지** — 미등록 AE, 미정의 규칙은 에러/경고 (숨기지 않음)
8. **Care AI는 감지+보고만** — 치료 결정은 의사 (병원 방문 시) 전담

---

## 코딩 컨벤션

- Python 3.12+
- LLM: Google Gemini API (gemini-2.0-flash)
- 출력: JSON (structured output / JSON mode)
- 한글 주석 허용, 변수명/키 이름은 영어
- 확률은 항상 decimal (0.5, not 50%)
- 환자 데이터는 파일 저장 (data/runs/)
- 로그는 logs/ 디렉토리에 저장 (/tmp 사용 금지)