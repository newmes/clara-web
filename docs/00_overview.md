# 00. Project Overview — Clinical Trial Simulation Engine

> **Web Demo:** 아래 링크에서 실제 구동 중인 시뮬레이션을 확인할 수 있습니다.
> 서버 기본 주소: `http://49.254.130.90:9000`
>
> | 페이지 | URL | 설명 |
> |--------|-----|------|
> | **랜딩** | [localhost:9000/](http://49.254.130.90:9000/) | 모든 시뮬레이션 run 목록 |
> | **Trial Viewer** | [Day-by-Day 뷰어](http://49.254.130.90:9000/trial/20260216_062601_Padcev___Pembrolizumab_10pt_84d/) | 10명×84일 시뮬레이션 탐색 (Map + Dashboard) |
> | **Patient Detail** | [PT-001 상세](http://49.254.130.90:9000/patient/20260216_062601_Padcev___Pembrolizumab_10pt_84d/PT-001/) | 개별 환자 종단적 데이터 |
> | **Game Landing** | [게임 환자 선택](http://49.254.130.90:9000/game/20260216_062601_Padcev___Pembrolizumab_10pt_84d/) | 인터랙티브 게임 모드 시작 |
> | **Game Play** | [PT-001 게임](http://49.254.130.90:9000/game/20260216_062601_Padcev___Pembrolizumab_10pt_84d/PT-001/) | 간호사 역할 플레이 |

## 한 줄 요약

**약물명 하나를 입력하면, LLM이 확률 규칙을 생성하고, hazard function이 매일의 이벤트를 동적으로 결정하는 3-Phase in-silico 임상시험 시뮬레이션 엔진.**

## 목적

- **MedGemma Impact Challenge** (마감: 2026-02-24, $100K)
- 핵심 데모: Padcev + Pembrolizumab 방광암 시험에서 **AI Care Agent의 AE 조기 감지 가치** 입증
- Drug-agnostic: 약물명만 바꾸면 어떤 약물/질환이든 시뮬레이션 가능

---

## 핵심 가설: Small Model Swarm ≥ Large Model

> **"작은 MedGemma(4B)라도, 특수 목적으로 fine-tuning하여 군집(swarm)으로 엮으면,
> 범용 대형 모델(Gemini Pro 등) 하나가 하는 일을 동등 이상의 품질로 수행할 수 있다."**

이 프로젝트가 MedGemma Impact Challenge에서 보여주고자 하는 핵심 강점이다.

### 왜 Swarm인가?

임상시험 시뮬레이션은 **성격이 전혀 다른 여러 전문 작업**의 집합이다:

| 작업 | 요구 능력 | Fine-tuned MedGemma 역할 |
|------|----------|--------------------------|
| 약물 규칙 생성 (Phase 0) | 약리학 지식, AE 발생률/분포 추정 | **Rule MedGemma** — KG 보강 약물 프로파일링 |
| 환자 생성 (Phase 1) | 인구통계/동반질환/검사치 일관성 | **Patient MedGemma** — 의학적 정합성 특화 |
| 일별 GT 생성 (Phase 2) | CTCAE 등급, 검사값 변동, 임상 서사 | **Clinical MedGemma** — 임상 기록 생성 특화 |
| 환자 AI 대화 (Care Agent) | 자연스러운 증상 표현, 심리 반영 | **Patient-Voice MedGemma** — 환자 시뮬레이션 특화 |
| 간호사 AI 판단 (Care Agent) | AE triage, 긴급도 판단, 질문 전략 | **Nurse MedGemma** — 임상 의사결정 특화 |
| SAE 보고서 생성 | ICH E2B 양식, 규제 용어, 인과관계 서술 | **Regulatory MedGemma** — 규제 문서 특화 |

### 대형 모델 vs Small Model Swarm

```
┌─────────────────────────────────────────────────────────┐
│  접근 A: 범용 대형 모델 1개 (Gemini Pro / GPT-4)         │
│                                                          │
│  [모든 작업] ──→ 🤖 Large Model ──→ [결과]               │
│                                                          │
│  장점: 범용성, 단일 API                                   │
│  단점: 비용 높음, 의학 특화 부족, hallucination 위험,      │
│        하나의 모델이 모든 도메인에 균일하게 잘하긴 어려움     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  접근 B: 특화 소형 모델 군집 (MedGemma 4B × N)           │
│                                                          │
│  [약물 규칙] ──→ 🩺 Rule MedGemma ──→ rule_set.json      │
│  [환자 생성] ──→ 🧬 Patient MedGemma ──→ patient.json    │
│  [임상 기록] ──→ 📋 Clinical MedGemma ──→ daily GT       │
│  [환자 대화] ──→ 🗣️ Patient-Voice MedGemma ──→ 대화      │
│  [간호사 판단] ──→ 👩‍⚕️ Nurse MedGemma ──→ triage         │
│  [SAE 보고서] ──→ 📄 Regulatory MedGemma ──→ E2B report  │
│                                                          │
│  장점: 각 영역 전문성 극대화, 비용 효율, 의학 특화,         │
│        hallucination 감소 (좁은 범위 집중), 병렬 실행 가능  │
│  단점: fine-tuning 파이프라인 필요, 모델 간 일관성 관리     │
└─────────────────────────────────────────────────────────┘
```

### 시뮬레이션 엔진의 역할

```
시뮬레이션 엔진 = Small Model Swarm의 "오케스트레이터"

 ┌───────────────────────────────────────────────────┐
 │                Simulation Engine                    │
 │                                                     │
 │  Phase 0 ──→ Rule MedGemma        ─┐               │
 │  Phase 1 ──→ Patient MedGemma      │               │
 │  Phase 2 ──→ Clinical MedGemma     ├─→ 결과 통합   │
 │  Care AI ──→ Patient + Nurse MedGemma│               │
 │  SAE     ──→ Regulatory MedGemma   ─┘               │
 │                                                     │
 │  + hazard function (코드)                            │
 │  + observation model (코드)                          │
 │  + sampler (코드)                                    │
 │                                                     │
 │  "LLM은 각자의 전문 영역만 담당하고,                   │
 │   코드가 확률적 일관성과 파이프라인 흐름을 보장한다"      │
 └───────────────────────────────────────────────────┘
```

### Challenge에서 입증할 것

1. **품질**: Fine-tuned MedGemma 4B swarm이 생성한 임상 데이터의 의학적 정확도 ≥ Gemini Pro 단독
2. **비용**: 소형 모델 군집의 inference 비용이 대형 모델 대비 대폭 절감
3. **전문성**: 각 모델이 자기 영역에서 대형 범용 모델보다 더 정확한 결과 생성 (특히 CTCAE 등급, 규제 문서 양식)
4. **확장성**: 새로운 전문 작업 추가 시 해당 목적의 fine-tuned 모델만 추가하면 됨

---

## 핵심 설계 원칙

| # | 원칙 | 설명 |
|---|------|------|
| 1 | **LLM은 확률을 정하고, 코드가 주사위를 굴린다** | Mode collapse 방지, 분포 보장 |
| 2 | **Fate table 없음** | Hazard function으로 매일 동적 결정 → Care AI 개입이 진짜 예방 효과 |
| 3 | **GT와 HR의 엄격한 분리** | Ground Truth ≠ Hospital Record → 정보 비대칭이 현실적 |
| 4 | **치료 결정은 HR 기반** | 의사는 GT(신의 시점)가 아닌 관찰 가능한 데이터로만 판단 |
| 5 | **Care AI는 감지+보고만** | 치료 결정(dose hold 등)은 의사 권한, AI는 조기 경보 |
| 6 | **Fallback 금지** | 에러 발생 시 silent fallback 없이 hard failure → 데이터 신뢰성 |
| 7 | **재현 가능** | Seed 고정으로 동일 시뮬레이션 재현 |

---

## 전체 아키텍처: 3-Phase Pipeline

```
입력: drug_name + indication
 │
 ▼
Phase 0: 규칙 발견 ────────────────── [약물당 1회]
 │  Rule Agent (LLM × 3회)
 │  → rule_set.json
 │    ├─ demographics (인구통계 분포)
 │    ├─ comorbidities (동반질환 확률)
 │    ├─ ae_profile (12~15개 AE: 발생률, onset 분포, grade 분포, 지속기간)
 │    ├─ efficacy (반응률, PFS, OS)
 │    ├─ administration_schedule (투약 스케줄)
 │    ├─ dose_modification_rules (용량 조절 기준)
 │    ├─ supportive_care_rules (보조약 처방 기준)
 │    ├─ mortality_model (사망 위험 모델)
 │    ├─ ecog_model (수행능력 변화 모델)
 │    └─ disposition_model (중도탈락 모델)
 │
 ▼
Phase 1: 환자 생성 ────────────────── [환자당 3회 LLM 호출]
 │  Patient Agent (LLM→rand→LLM 패턴)
 │  → PT-001.json, PT-002.json, ...
 │    ├─ demographics (코드 샘플링, LLM 호출 0)
 │    ├─ comorbidities (LLM 확률 조정 → 코드 샘플링)
 │    ├─ baseline labs/vitals (LLM 생성)
 │    └─ persona (코드로 10종 중 1개 선택 → LLM 상세 생성)
 │
 ▼
Phase 2: 일별 시뮬레이션 ──────────── [Day 1 ~ N]
 │  Daily Agent + Hazard Engine + Observation Model
 │
 │  ┌─ 초기화 (환자당 1회) ──────────────────────────────┐
 │  │  • LLM이 환자별 AE incidence 보정 (1회 호출)       │
 │  │  • 종양 반응 카테고리 rand 샘플링 (코드만)          │
 │  └──────────────────────────────────────────────────┘
 │
 │  ┌─ 매일 반복 ──────────────────────────────────────┐
 │  │                                                    │
 │  │  1. GT 생성 (hazard.py)                           │
 │  │     AE onset/grade변화/해소, 종양변화, 사망, ECOG   │
 │  │                                                    │
 │  │  2. [Care AI 모드] 영상통화                        │
 │  │     4-Turn: 환자초기보고→간호사질문→환자응답→최종판정  │
 │  │                                                    │
 │  │  3. Observation: GT → Hospital Record 필터링       │
 │  │     AE 감지(채널별), grade distortion(mood 기반)    │
 │  │                                                    │
 │  │  4. [병원방문일] Dose Modification (HR 기반)       │
 │  │     관찰된 AE만으로 용량 조절 결정                   │
 │  │                                                    │
 │  │  5. 결과 저장 (JSONL)                             │
 │  └──────────────────────────────────────────────────┘
 │
 ▼
출력:
 ├─ data/runs/{timestamp}/
 │   ├─ rule_set.json
 │   ├─ patients/PT-001.json, ...
 │   └─ simulations/PT-001_natural.jsonl, PT-001_care_ai.jsonl, ...
 └─ logs/console_{run_name}.log, sim_{timestamp}.log
```

---

## 모듈 구조

```
ClinicalTrialEngine/
│
├── src/
│   ├── engine/                          ★ 확률 엔진 (LLM 독립, 순수 수학)
│   │   ├── sampler.py                   난수 생성기 (Mersenne Twister 기반)
│   │   ├── prob_engine.py               LLM→rand→LLM 패턴 오케스트레이터
│   │   ├── hazard.py                    ★ Hazard function (onset/grade/resolution/mortality/ECOG)
│   │   ├── observation.py               ★ GT→HR 필터링 + AE 감지 모델 (40개 AE 채널 정의)
│   │   └── mood.py                      환자 심리 7차원 모델 (10개 페르소나)
│   │
│   ├── agents/                          LLM Agent 집합
│   │   ├── llm_client.py                Gemini API 공통 호출 (JSON mode, retry, 토큰 제어)
│   │   ├── rule_agent.py                Phase 0: 약물별 규칙 발견 (LLM ×3)
│   │   ├── patient_agent.py             Phase 1: 환자 생성 (LLM ×3/환자)
│   │   ├── daily_agent.py               Phase 2: DailySimulator (3000+ lines)
│   │   └── care_agent.py                Care AI 4-Turn 영상통화 에이전트
│   │
│   ├── orchestrator.py               시뮬레이션 루프 (Natural/Care AI/Both)
│   ├── run_simulation.py             CLI 실행 + 로깅
│   ├── game_session.py                  인터랙티브 게임 모드 세션 관리
│   ├── crf_mapper.py                    CDASH CRF 포맷 매핑
│   ├── context_manager.py               컨텍스트 압축
│   ├── validator.py                     스키마 검증
│   └── logger.py                        구조화 로깅
│
├── config/
│   └── defaults.py                      전역 상수 (grade transition, tumor rate, mortality 등)
│
├── frontend/                            Django 웹 UI
│   ├── trial_server/                    Django 설정/URL
│   ├── viewer/views.py                  View 함수 (페이지 + JSON API + Game API)
│   ├── templates/                       HTML 템플릿
│   │   ├── base/base.html               공통 레이아웃 + 글로벌 네비게이션
│   │   ├── landing/landing.html         랜딩 페이지 (run 선택)
│   │   ├── trial/trial.html             Day-by-day 뷰어 (Generative Agents 스타일)
│   │   ├── patient_state/               환자 상세 페이지
│   │   └── game/                        인터랙티브 게임 모드 (환자 선택 + 플레이)
│   └── static_dirs/css/style.css        다크모드 UI 스타일
│
├── data/
│   ├── rule_set.json                    현재 활성 rule set
│   └── runs/                            시뮬레이션 실행 결과
│
├── logs/                                시뮬레이션 로그
├── docs/                                이 문서들
└── CLAUDE.md                            AI 어시스턴트용 프로젝트 요약
```

---

## LLM 호출 비용 분석

| Phase | 호출 대상 | 호출 횟수 | 토큰 규모 |
|-------|----------|----------|----------|
| 0: Rule Discovery | Rule Agent | 약물당 3회 | ~30K output |
| 1: Patient Gen | Patient Agent | 환자당 3회 | ~4K output/환자 |
| 2: Daily Init | AE Calibration | 환자당 1회 | ~2K output |
| 2: Event Day | Daily Agent | 이벤트 날만 (~30-40%) | ~3K output/일 |
| 2: Care AI | Care Agent | 매일 4턴 (조기종료 시 2턴) | ~2K output/일 |

**10명 × 84일 시뮬레이션 예상 비용:**
- Natural: ~300 event days × 1 LLM = 300 호출
- Care AI: ~840일 × 4턴 = 3,360 호출 + ~300 event days
- 합계: ~3,700 호출 (Gemini Flash 기준 약 $1-2)

---

## 핵심 데이터 흐름: A/B 비교

```
같은 rule_set + 같은 환자 프로필
    │
    ├─── Natural Mode (Care AI 없음)
    │    seed: base + patient_num
    │    병원 방문일에만 AE 감지
    │    감지 지연 → Grade 악화 → 늦은 용량 조절
    │
    └─── Care AI Mode
         seed: base + patient_num + 10000 (독립 난수열)
         매일 영상통화 → AE 조기 감지
         force_hospital_tomorrow → 조기 병원 방문
         빠른 용량 조절 → Grade 악화 예방

비교 지표:
 - AE 감지 지연일 (mean_delay_days)
 - Grade 3+ 악화 건수
 - 치료 지속 기간
 - 용량 조절 적시성
```

---

## 문서 목차

| 문서 | 내용 | 관련 웹 페이지 |
|------|------|---------------|
| [00_overview.md](00_overview.md) | 이 문서 — 전체 개요 | [랜딩 페이지](http://49.254.130.90:9000/) |
| [01_rule_discovery.md](01_rule_discovery.md) | Phase 0: Rule Agent 상세 | — |
| [02_patient_generation.md](02_patient_generation.md) | Phase 1: Patient Agent 상세 | [Patient Detail](http://49.254.130.90:9000/patient/20260216_062601_Padcev___Pembrolizumab_10pt_84d/PT-001/) |
| [03_hazard_engine.md](03_hazard_engine.md) | Hazard function 수학 + 구현 | — |
| [04_daily_simulation.md](04_daily_simulation.md) | Phase 2: Daily Simulation 10-step pipeline | [Trial Viewer](http://49.254.130.90:9000/trial/20260216_062601_Padcev___Pembrolizumab_10pt_84d/) |
| [05_observation_model.md](05_observation_model.md) | GT/HR 분리 + AE 감지 모델 | [Trial Viewer (GT/HR 토글)](http://49.254.130.90:9000/trial/20260216_062601_Padcev___Pembrolizumab_10pt_84d/) |
| [06_care_agent.md](06_care_agent.md) | Care AI 4-Turn 대화 시스템 | [Trial Viewer Care AI 모드](http://49.254.130.90:9000/trial/20260216_062601_Padcev___Pembrolizumab_10pt_84d/?mode=care_ai) |
| [07_interactive_game.md](07_interactive_game.md) | 인터랙티브 게임 모드 | [게임 플레이](http://49.254.130.90:9000/game/20260216_062601_Padcev___Pembrolizumab_10pt_84d/PT-001/) |
| [08_web_ui.md](08_web_ui.md) | 웹 뷰어 + API | [랜딩 페이지](http://49.254.130.90:9000/) |
| [09_future_work.md](09_future_work.md) | Future Work + MedGemma Challenge 로드맵 | — |