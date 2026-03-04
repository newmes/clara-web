# 06. Care AI Agent — 4-Turn 영상통화 시뮬레이션

> **파일:** `sim/agents/care_agent.py` (~920 lines)
> **역할:** 매일 AI 간호사가 환자와 영상통화를 수행하여 AE를 조기 감지
> **LLM 호출:** 매일 최대 4회 (조기 종료 시 2회)
> **정보 비대칭:** Patient LLM은 GT 접근, Nurse LLM은 HR만 접근
>
> 🔗 **웹에서 확인:** [Trial Viewer (Care AI 대화)](http://49.254.130.90:9000/trial/20260216_062601_Padcev___Pembrolizumab_10pt_84d/) — 시뮬레이션 뷰어에서 Care AI 대화 내역과 개입 효과 확인

---

## 1. 핵심 가치: 관찰 공백 해소

```
Natural Mode:
  병원 방문(3주 간격) 사이에 환자의 상태 변화를 알 수 없음
  → AE 감지 평균 지연 ~10일

Care AI Mode:
  매일 영상통화 → 관찰 공백 1일 이내
  → AE 감지 평균 지연 ~2일
```

---

## 2. CareAgent 클래스

### 2.1 초기화

```python
class CareAgent:
    def __init__(self, patient, rule_set, mood, sampler, model):
        self.known_aes = rule_set.ae_profile[:12]  # 상위 12개 AE만 참조
        self.call_history = []                       # 최근 7일 통화 이력
        self.max_history = 7
        self._last_hospital_record = None            # 마지막 병원 기록 (Nurse 참조용)
```

**`known_aes`를 12개로 제한하는 이유:**
- 간호사 프롬프트의 토큰 절약
- 실제 간호사도 모든 가능한 AE를 외우고 있지 않음
- 가장 빈도 높은 AE에 집중 (우선순위 기반)

**`call_history`를 7일로 제한하는 이유:**
- 종단적(longitudinal) 컨텍스트 제공 (추세 파악)
- 토큰 제한 내 유지
- 7일 = 기억의 심리적 유효 범위

---

## 3. 정보 비대칭 설계

```
┌──────────────────────────────┐
│ Patient LLM (환자 역할)       │
│ ✓ Ground Truth (GT)          │
│ ✓ 자신의 mood state          │
│ ✓ 이전 통화 기억 (3일분)     │
│ ✗ Nurse의 내부 판단          │
│ ✗ 병원 기록                  │
└──────────────────────────────┘

┌──────────────────────────────┐
│ Nurse LLM (AI 간호사 역할)    │
│ ✓ Hospital Record (HR)       │
│ ✓ 이전 통화 이력 (5일분)     │
│ ✓ 약물 AE 프로파일           │
│ ✓ 환자 quality 지표          │
│ ✗ Ground Truth               │  ← 절대 접근 불가!
│ ✗ 환자의 mood state          │
│ ✗ 환자가 숨긴 증상           │
└──────────────────────────────┘
```

---

## 4. 4-Turn 대화 흐름

### Turn 1: 환자 초기 보고 (`_patient_initial_report`)

**Patient LLM에 전달되는 정보:**
- 인구통계, 페르소나, 약물/적응증
- Mood 기반 행동 파라미터: engagement, under_report, over_report, grade_distortion
- GT: 활성 AE (ae_term, grade, days_active, video_visible), 주관적 인식, 체온, 체중
- 최근 3일 통화 기억

**행동 규칙 (프롬프트 내):**
```
If under_report_prob is HIGH (> 0.4):
  → say "I'm fine" / minimize symptoms
  → omit some AEs from reporting

If over_report_prob is HIGH (> 0.3):
  → emphasize every sensation
  → use alarming language

If grade_distortion == -1:
  → describe Grade 2 nausea as "just a little queasy"

If grade_distortion == +1:
  → describe Grade 1 fatigue as "I can barely get out of bed"
```

**출력 스키마:**
```json
{
  "greeting": "안녕하세요 간호사님, 오늘은 좀...",
  "reported_symptoms": [
    {"symptom": "좀 속이 메스꺼워요", "severity_perception": "mild",
     "duration": "3일째", "is_new": false}
  ],
  "omitted_symptoms": ["발에 저림이 있는데 말하지 않음"],
  "general_wellbeing": "좀 피곤한 것 같아요",
  "mood_expression": "resigned",
  "video_visible": ["slight pallor", "fatigue_visible"]
}
```

**`omitted_symptoms`:** GT에는 존재하지만 환자가 의도적으로 보고하지 않은 증상. Nurse LLM은 이 필드를 볼 수 없음 — 오직 GT 평가용.

### Early Termination Check

T1 직후:
```python
if sampler.boolean(quality["early_termination_prob"]):
    # 환자가 전화를 끊음! T2, T3 생략
    mood.update_turn({"irritability": +0.05, "trust_in_ai": -0.03})
    # → T4만 실행 (제한된 정보로 판단)
```

조기 종료 확률은 `irritability × 0.35 + (1-energy) × 0.25 + (1-trust_in_ai) × 0.15`로 계산.

---

### Turn 2: 간호사 추가 질문 (`_nurse_followup_questions`)

**Nurse LLM에 전달되는 정보:**
- T1 환자 보고 전문
- 이전 5일 통화 이력 요약 (severity, action, 조기종료 여부)
- 마지막 병원 기록 요약 (AE grades, 비정상 lab, vitals, ECOG)
- 약물 AE 프로파일 (12개)
- 환자 quality 지표: under_report_prob, video_cooperation, engagement

**전략 가이드라인 (프롬프트 내):**
```
If under_report_prob > 0.4: HIGH
  → Probe specifically for common AEs (rash, nausea, neuropathy)
  → Patient may be hiding symptoms

If video_cooperation > 0.5: HIGH
  → Request visual inspection (skin, mouth, hands)

If engagement > 0.5: HIGH
  → Can ask detailed questions
  Else: Keep brief, 2 questions max

Maximum 3 targeted questions (prevent overwhelming)
```

**출력 스키마:**
```json
{
  "approach_style": "empathetic",
  "acknowledgment": "I understand you're feeling tired...",
  "questions": [
    {"question": "피부에 변화가 있으신가요?", "target_ae": "rash",
     "requires_visual": true, "rationale": "Padcev known skin toxicity"}
  ],
  "visual_request": {"requested": true, "body_area": "arms and hands", "reason": "check for rash/PPE"},
  "preliminary_concerns": ["possible Grade 2 fatigue worsening"]
}
```

### Nurse Approach → Mood 효과

`approach_style`이 T3 전에 환자의 mood를 업데이트:

| Approach | trust_in_ai | defensiveness | anxiety | irritability | energy |
|----------|------------|--------------|---------|-------------|--------|
| `empathetic` | **+0.05** | **-0.04** | -0.02 | -0.02 | — |
| `neutral` | +0.01 | -0.01 | — | — | — |
| `concerned` | +0.03 | -0.03 | +0.04 | — | — |
| `urgent` | +0.02 | **-0.06** | **+0.08** | — | +0.03 |

**트레이드오프:**
- `empathetic` → 신뢰 ↑, 방어 ↓, 불안 ↓ (최적의 일반 전략)
- `urgent` → 방어 가장 많이 ↓ (숨겨진 증상 드러남), 하지만 불안 크게 ↑

---

### Turn 3: 환자 응답 (`_patient_followup_response`)

**Visual Cooperation 메커니즘:**
```python
will_cooperate = sampler.boolean(quality["video_cooperation"])
# video_cooperation = energy × 0.30 + (1-defensiveness) × 0.35 + trust_in_ai × 0.20 + (1-irritability) × 0.15
```

**부분 드러남(Partial Revelation) 규칙:**
```
If nurse asks about a symptom the patient HAS but was hiding:
  → Reveal PARTIALLY: "음... 사실 좀 그런 게 있긴 해요"
  → honesty_level: "partial" or "evasive"

If nurse asks about something the patient DOESN'T have:
  → Deny: "아뇨, 그런 건 없어요"
  → honesty_level: "full" (truthful denial)

If visual cooperation == WILLING and nurse requested visual:
  → Show the area on camera: "여기 좀 보여드릴게요"
  → what_visible_to_nurse: "redness on forearms"

If visual cooperation == RELUCTANT:
  → Make excuses: "잘 안 보일 것 같은데요..."
```

**출력 스키마:**
```json
{
  "responses": [
    {"to_question": "피부에 변화가 있으신가요?",
     "answer": "음... 사실 팔에 좀 빨간 게 있긴 해요",
     "revealed_symptom": "rash_maculopapular",
     "honesty_level": "partial"}
  ],
  "visual_response": {
    "cooperated": true,
    "what_shown": "forearms",
    "what_visible_to_nurse": "erythematous maculopapular rash on both forearms"
  },
  "new_info_revealed": true,
  "emotional_reaction": "somewhat relieved to have shared"
}
```

**T3 후 mood 업데이트:**
```python
if new_info_revealed:
    mood.update_turn({"defensiveness": -0.05, "anxiety": +0.03})
# 정보를 드러냈으므로 방어가 낮아지고, 약간의 불안 증가
```

---

### Turn 4: 간호사 최종 판정 (`_nurse_final_assessment`)

**Severity Triage 가이드라인:**

| Level | 색상 | 행동 | 기준 |
|-------|------|------|------|
| `GREEN` | 녹색 | `no_action` | 안정, 특이 소견 없음. 대부분의 통화가 GREEN이어야 함 |
| `YELLOW` | 노란 | `monitor_closely` | 경미한 증상, 추적 관찰. Grade 1 피로/오심/식욕저하는 예상됨 → 에스컬레이션 금지 |
| `ORANGE` | 주황 | `recommend_conmed`, `recommend_early_visit` | 보조약 추천, 조기 방문 권유 |
| `RED` | 빨강 | `recommend_hospital_visit`, `escalate_to_physician` | 당일 평가 필요, SAE 의심 시 긴급 의사 연락 |

**핵심 규칙:**
```
1. 감지는 환자 보고 + 영상 관찰만으로 판단 (Lab 참조 불가)
2. dose_hold나 치료 변경은 절대 권한 외 (의사 전용)
3. Grade 1 AE는 예상 범위 → 과잉 반응 금지
4. 조기 종료된 통화: 제한적 정보 인정, 최악 가정 금지
5. 환자의 인식 vs 실제 심각도 구분 필요
```

**출력 스키마:**
```json
{
  "assessment": {
    "severity_level": "orange",
    "summary": "New maculopapular rash detected via video, Grade 2 estimated",
    "detected_issues": [
      {"issue": "Rash on forearms", "suspected_ae": "rash_maculopapular",
       "estimated_grade": 2, "confidence": "medium",
       "detection_source": "visual_observation"}
    ]
  },
  "detection": {
    "aes_detected": ["rash_maculopapular"],
    "concerns_for_followup": ["monitor rash evolution"],
    "early_warning_signs": ["possible PPE developing"]
  },
  "actions": [
    {"action": "recommend_conmed", "reason": "Grade 2 rash management",
     "detail": "Topical diphenhydramine + oral antihistamine", "urgency": "routine"},
    {"action": "recommend_early_visit", "reason": "Rash evaluation by dermatology",
     "detail": "Within 2-3 days", "urgency": "urgent"}
  ]
}
```

**T4 후 mood 업데이트:**
```python
# 정상 완료
mood.update_turn({"trust_in_ai": +0.04, "anxiety": -0.02})
```

---

## 5. 개입(Intervention) 엔진

### 5.1 `apply_interventions()` — 독립 함수

```python
def apply_interventions(care_record, simulator, day) -> list[str]:
```

Care AI의 action을 시뮬레이션 상태에 반영:

| Action | 시뮬레이션 효과 | 비고 |
|--------|---------------|------|
| `no_action` | 없음 | Skip |
| `monitor_closely` | 로깅만 | 상태 변화 없음 |
| `recommend_conmed` | **CM 추가** | `_create_conmed_from_recommendation()` |
| `recommend_early_visit` | 로깅 (orchestrator가 `force_hospital_tomorrow` 설정) | Care AI 직접 불가 |
| `recommend_hospital_visit` | 로깅 (orchestrator가 처리) | Care AI 직접 불가 |
| `escalate_to_physician` | 로깅 + urgency 전달 | emergency → same_day_er |

**핵심 원칙:** Care AI는 보조약(CM) 추가만 직접 가능. 용량 조절, 치료 중단은 의사 권한.

### 5.2 `_create_conmed_from_recommendation()` — 키워드→약물 매핑

| 키워드 | 약물 | 용량 | 경로 | 빈도 |
|--------|------|------|------|------|
| nausea, vomit | Ondansetron | 8mg | PO | PRN |
| diarrhea | Loperamide | 4mg→2mg | PO | PRN |
| pain, headache | Acetaminophen | 500mg | PO | Q6H PRN |
| fever | Acetaminophen | 650mg | PO | Q6H PRN |
| rash, itch, prurit | Diphenhydramine | 25mg | PO | Q8H PRN |
| mucosit, stomatit | Lidocaine mouthwash | 15mL | PO | QID |
| constipat | Docusate | 100mg | PO | BID |
| neuropath, tingl, numb | Gabapentin | 300mg | PO | TID |
| anorexia, appetite | Megestrol | 400mg | PO | QD |
| fatigue | Activity modification | — | NON-DRUG | — |

**중복 방지:** 같은 약물이 이미 `simulator.active_cm`에 존재하면 추가하지 않음.

---

## 6. Orchestrator에서의 Care AI 통합

```python
# orchestrator.py의 run_care_ai() 내부:

# 1. 매일 영상통화
care_result = care_agent.conduct_video_call(day, day_result, day_results, last_hospital_record)
day_result["care_record"] = [care_result]

# 2. 개입 적용
applied = apply_interventions(care_result, simulator, day)

# 3. 에스컬레이션 처리
for action in care_result.get("actions", []):
    act = action.get("action", "")
    urgency = action.get("urgency", "routine")
    
    if act == "escalate_to_physician" and urgency == "emergency":
        same_day_er = True   # 오늘 당장 병원
    elif act in ("recommend_hospital_visit", "recommend_early_visit", "escalate_to_physician"):
        force_hospital_tomorrow = True  # 내일 병원
```

---

## 7. Mood 변화 요약 (완전한 통화 1회)

```
통화 시작 전:
  G3+ AE 존재 → defensiveness override (-0.15~-0.30)

T1 (환자 보고):
  → 조기 종료 시: irritability +0.05, trust_in_ai -0.03 → T4로 직행

T2 (간호사 질문):
  approach 스타일에 따른 mood 업데이트
  empathetic: trust +0.05, defense -0.04, anxiety -0.02, irritability -0.02

T3 (환자 응답):
  new_info_revealed: defense -0.05, anxiety +0.03

T4 (최종 판정):
  정상 완료: trust +0.04, anxiety -0.02

총 효과 (empathetic + 정보 드러남):
  trust: +0.05 +0.04 = +0.09
  defense: -0.04 -0.05 = -0.09
  anxiety: -0.02 +0.03 -0.02 = -0.01
```

**장기 효과:** 매일 empathetic 접근을 유지하면, 7일간 trust ≈ +0.63, defensiveness ≈ -0.63 → 처음에 비협조적이던 stoic_minimizer도 점차 개방적으로 변화.

---

## 8. 에러 처리 및 Fallback

| Turn | 실패 시 | Fallback 내용 |
|------|---------|-------------|
| T1 | `_patient_fallback()` | 모든 활성 AE를 "moderate" severity로 보고 |
| T2 | 기본 empathetic | "피부 변화 있으신가요?" 1개 질문 + 시각 검사 요청 |
| T3 | 단순 응답 | "특별히 더 말씀드릴 건 없어요" + 시각 협조 거부 |
| T4 | 증상 기반 판단 | reported_symptoms 존재 → yellow, 없으면 → green |

**원칙:** LLM 실패 시에도 시뮬레이션은 계속. 최소한의 의미 있는 결과를 보장.