"""Care Agent — 매일 환자와 AI 4-Turn 영상통화를 시뮬레이션하는 에이전트

Care AI는 환자와 매일 영상통화를 수행하여:
  1. 환자의 현재 상태를 파악 (Patient LLM)
  2. 이상 징후를 조기 감지 (Nurse LLM)
  3. 필요시 보조약(conmed) 권고 또는 조기 병원방문/의사 의뢰

핵심: Care AI는 치료 결정(dose hold/discontinue)을 하지 않음.
      조기 감지 → 의료진에게 빠르게 연결하는 것이 가치.

4-Turn 대화 구조:
  T1: Patient → 초기 보고 (mood 기반 축소/과대)
  T2: Nurse  → 탐색 질문 (quality 계수 기반 전략)
  T3: Patient → 추가 정보 공개 (updated mood → 방어 약화)
  T4: Nurse  → 최종 평가 + 개입 결정

7-Dimension Mood 통합:
  ① anxiety → 과보고 ② depression → 무응답 ③ irritability → 조기종료
  ④ energy → 참여도  ⑤ cognitive_clarity → 정확도
  ⑥ trust_in_ai → 이행률 ⑦ defensiveness → 축소보고

핵심 원칙:
  - Care AI는 질병 진행 자체를 바꾸지 않음
  - 정보 비대칭 해소: 더 빨리 감지 → 더 빨리 개입 → 더 나은 결과
"""

import json
from typing import Any

from sim.agents.llm_client import generate_json, set_caller, DEFAULT_MODEL
from sim.engine.mood import (
    MoodState,
    compute_interaction_quality,
    compute_grade_distortion,
)
from sim.engine.sampler import Sampler
from sim.logger import get_logger

_logger = get_logger("care_agent")


# ══════════════════════════════════════════════════════
# Care Agent 핵심 클래스
# ══════════════════════════════════════════════════════

class CareAgent:
    """매일 환자와 4-Turn AI 영상통화를 시뮬레이션하는 에이전트."""

    def __init__(
        self,
        patient: dict,
        rule_set: dict,
        mood: MoodState,
        sampler: Sampler,
        model: str = DEFAULT_MODEL,
    ):
        self.patient = patient
        self.rule_set = rule_set
        self.mood = mood
        self.sampler = sampler
        self.model = model
        self.pid = patient.get("patient_id", "?")
        self.persona = patient.get("persona", {})
        self.demographics = patient.get("emr", {}).get("demographics", {})

        # 약물 AE 프로파일 (Nurse가 참고)
        self.known_aes = [
            ae["ae_term"] for ae in rule_set.get("ae_profile", [])[:12]
        ]

        # 누적 통화 이력 (최근 N개만 유지)
        self.call_history: list[dict] = []
        self.max_history = 7

        # Nurse가 참조하는 마지막 병원 기록 (conduct_video_call에서 갱신)
        self._last_hospital_record: dict | None = None

        _logger.info(f"[CareAgent] Initialized for {self.pid} (4-turn mode)")

    # ── 메인 진입점 ─────────────────────────────────────

    def conduct_video_call(
        self,
        day: int,
        day_result: dict,
        day_results: list[dict],
        last_hospital_record: dict | None = None,
    ) -> dict:
        """하루의 4-Turn 영상통화를 시뮬레이션한다.

        정보 접근 원칙:
          - Patient LLM: ground truth (day_result) → 자기 몸 상태를 알고 있음
          - Nurse LLM: hospital_record + call_history + 대화 내용만
          - Nurse는 ground truth에 직접 접근 불가

        Flow:
          T1: Patient → 초기 보고 (mood 기반 행동)
          [early_termination 확률 체크]
          T2: Nurse  → 탐색 질문 (hospital_record + T1 기반)
          T3: Patient → 추가 정보
          T4: Nurse  → 최종 판정 + 개입 결정 (hospital_record + 전체 대화 기반)

        Args:
            day: 시뮬레이션 일차
            day_result: ground truth (Patient LLM만 사용)
            day_results: 이전 일차 결과 (Patient LLM만 사용)
            last_hospital_record: 마지막 병원 내원 시 기록 (Nurse LLM 사용)

        Returns:
            care_record dict with turns, assessment, actions, detection
        """
        set_caller(f"care_agent.{self.pid}")

        # ── ground truth에서 mood 관련 정보만 추출 (시뮬레이션 엔진용) ──
        obj = day_result.get("objective", {})
        active_aes = obj.get("active_aes", [])
        max_ae_grade = max((ae.get("grade", 0) for ae in active_aes), default=0)

        # Nurse에게 전달할 병원 기록 요약
        self._last_hospital_record = last_hospital_record

        # ── mood 기반 행동 계수 산출 ──
        quality = compute_interaction_quality(self.mood)
        grade_distortion = compute_grade_distortion(self.mood)

        # G3+ AE가 있으면 방어벽 약화 (환자 심리 모델 — GT 사용 정당)
        if max_ae_grade >= 3:
            self.mood.apply_defensiveness_override(max_ae_grade)
            quality = compute_interaction_quality(self.mood)  # 재계산

        turns: list[dict] = []
        terminated_early = False

        # ══════════════════════════════════════════════════
        # T1: Patient initial report
        # ══════════════════════════════════════════════════
        t1 = self._patient_initial_report(day, day_result, quality, grade_distortion)
        turns.append({"turn": 1, "role": "patient", "content": t1})

        # ── Early termination check ──
        # irritability 높고 energy 낮으면 "됐어요" 하고 끊음
        if self.sampler.boolean(quality["early_termination_prob"]):
            _logger.info(
                f"[{self.pid}] Day {day}: early termination "
                f"(prob={quality['early_termination_prob']:.2f})"
            )
            terminated_early = True
            # Nurse는 T1만으로 판단 (GT 접근 없음)
            t4 = self._nurse_final_assessment(
                day, turns, quality,
            )
            turns.append({"turn": 4, "role": "nurse", "content": t4})
            self.mood.update_turn({
                "irritability": +0.05, "trust_in_ai": -0.03,
            })
        else:
            # ══════════════════════════════════════════════
            # T2: Nurse follow-up questions
            # ══════════════════════════════════════════════
            t2 = self._nurse_followup_questions(
                day, t1, quality,
            )
            turns.append({"turn": 2, "role": "nurse", "content": t2})

            # Nurse의 질문에 따른 mood 변화
            nurse_approach = t2.get("approach_style", "neutral")
            self.mood.update_turn(
                _nurse_approach_mood_effect(nurse_approach),
            )

            # ══════════════════════════════════════════════
            # T3: Patient responds to questions
            # ══════════════════════════════════════════════
            quality_t3 = compute_interaction_quality(self.mood)
            t3 = self._patient_followup_response(
                day, day_result, t2, quality_t3, grade_distortion,
            )
            turns.append({"turn": 3, "role": "patient", "content": t3})

            # 환자가 추가 정보를 공개하면 defensiveness 약화
            if t3.get("new_info_revealed"):
                self.mood.update_turn({
                    "defensiveness": -0.05, "anxiety": +0.03,
                })

            # ══════════════════════════════════════════════
            # T4: Nurse final assessment + actions
            # ══════════════════════════════════════════════
            quality_t4 = compute_interaction_quality(self.mood)
            t4 = self._nurse_final_assessment(
                day, turns, quality_t4,
            )
            turns.append({"turn": 4, "role": "nurse", "content": t4})

            # 적절한 대응 → 신뢰 상승
            self.mood.update_turn({
                "trust_in_ai": +0.04, "anxiety": -0.02,
            })

        # ── care_record 조립 ──
        assessment = t4.get("assessment", {})
        actions = t4.get("actions", [])
        detection = t4.get("detection", {})

        care_record = {
            "day": day,
            "turns": turns,
            "terminated_early": terminated_early,
            "interaction_quality": quality,
            "grade_distortion": grade_distortion,
            "mood_snapshot": self.mood.to_dict(),
            "nurse_assessment": assessment,
            "actions": actions,
            "detection": detection,
        }

        # 통화 이력 유지
        self.call_history.append({
            "day": day,
            "severity": assessment.get("severity_level", "?"),
            "summary": assessment.get("summary", ""),
            "actions": [a.get("action", "") for a in actions],
            "terminated_early": terminated_early,
        })
        if len(self.call_history) > self.max_history:
            self.call_history = self.call_history[-self.max_history:]

        action_names = [a.get("action", "?") for a in actions]
        n_turns = len(turns)
        _logger.info(
            f"[{self.pid}] Day {day} call: {n_turns} turns, "
            f"early_stop={terminated_early}, actions={action_names}"
        )

        return care_record

    # ══════════════════════════════════════════════════════
    # T1: Patient Initial Report
    # ══════════════════════════════════════════════════════

    def _patient_initial_report(
        self,
        day: int,
        day_result: dict,
        quality: dict,
        grade_distortion: int,
    ) -> dict:
        """T1: 환자의 초기 보고. mood 기반 축소/과대 보고."""
        obj = day_result.get("objective", {})
        subj = day_result.get("subjective", {})
        active_aes = obj.get("active_aes", [])
        vitals = obj.get("vitals", {})

        system_prompt = f"""You are roleplaying as a clinical trial patient in a daily video call with an AI nurse.

PATIENT PROFILE:
- Age: {self.demographics.get('age', '?')}, Sex: {self.demographics.get('sex', '?')}
- Personality: {json.dumps(self.persona, ensure_ascii=False)}
- Drug: {self.rule_set.get('drug_name', '?')} for {self.rule_set.get('indication', '?')}

BEHAVIORAL PARAMETERS (from your current mood state — follow these strictly):
- Engagement level: {quality['engagement']:.2f} (0=silent, 1=very talkative)
- Under-report probability: {quality['under_report_prob']:.2f} (higher = more likely to hide symptoms)
- Over-report probability: {quality['over_report_prob']:.2f} (higher = exaggerate)
- Grade distortion: {grade_distortion:+d} (negative=downplay severity, positive=exaggerate)

RULES:
- This is your INITIAL greeting and report — keep it natural and conversational
- Report what YOU feel/see — you don't know lab values or medical terms
- If under-report is high: omit mild symptoms, say "I'm fine" more
- If over-report is high: emphasize every sensation, worry about everything
- If engagement is low: short answers, less detail
- If grade_distortion is negative: describe symptoms as less severe than they are

Output JSON only."""

        ae_summary = json.dumps(
            [{"ae": ae.get("ae", ""), "grade": ae.get("grade", 0),
              "days_active": ae.get("days_active", 0), "visual": ae.get("visual")}
             for ae in active_aes], ensure_ascii=False,
        )
        symptoms_json = json.dumps(subj.get("symptoms_patient_perceives", []), ensure_ascii=False)
        history_json = json.dumps(self.call_history[-3:], ensure_ascii=False)

        user_prompt = f"""Day {day}. Location: {obj.get('location', 'HOME')}

GROUND TRUTH (what you're actually experiencing — filter through your personality):
- Active side effects: {ae_summary}
- Subjective awareness: {subj.get("overall_awareness", "UNAWARE")}
- Symptoms you perceive: {symptoms_json}
- Body temp: {vitals.get("BT", "?")}°C, Weight: {vitals.get("weight_kg", "?")}kg

Previous calls:
{history_json}

OUTPUT:
{{
    "greeting": "string",
    "reported_symptoms": [
        {{"symptom": "string (your own words)", "severity_perception": "none|mild|moderate|severe",
          "duration": "string", "is_new": true/false}}
    ],
    "omitted_symptoms": ["string (symptoms you're hiding or unaware of)"],
    "general_wellbeing": "string",
    "mood_expression": "string (emotional state shown in call)",
    "video_visible": ["string (what nurse can SEE on camera — skin, posture, etc.)"]
}}"""

        try:
            result = generate_json(system_prompt, user_prompt, model=self.model, max_tokens=2048)
        except Exception as e:
            _logger.error(f"[{self.pid}] T1 Patient LLM failed: {e}")
            result = self._patient_fallback(active_aes, subj)
        result["_turn"] = 1
        return result

    # ══════════════════════════════════════════════════════
    # T2: Nurse Follow-up Questions
    # ══════════════════════════════════════════════════════

    def _nurse_followup_questions(
        self,
        day: int,
        patient_t1: dict,
        quality: dict,
    ) -> dict:
        """T2: Nurse가 T1을 분석하고 탐색 질문을 생성한다.

        Nurse가 접근 가능한 정보:
        - patient_t1: 오늘 환자가 보고한 내용
        - call_history: 이전 통화 기록
        - last_hospital_record: 마지막 병원 내원 시 기록
        - known AEs: 이 약물의 알려진 부작용 목록
        """
        recent_summary = self._summarize_recent()
        hospital_summary = self._summarize_hospital_record()

        system_prompt = f"""You are an AI nurse conducting Turn 2 of a daily video call.
You've just heard the patient's initial report. Now you need to probe deeper.

CLINICAL CONTEXT:
- Drug: {self.rule_set.get('drug_name', '?')}
- Indication: {self.rule_set.get('indication', '?')}
- Known AEs for this drug: {json.dumps(self.known_aes, ensure_ascii=False)}

LAST HOSPITAL RECORD (from most recent clinic visit):
{hospital_summary}

PATIENT INTERACTION PROFILE (from mood model):
- Under-report probability: {quality['under_report_prob']:.2f} — {'HIGH: patient likely hiding symptoms, probe carefully' if quality['under_report_prob'] > 0.4 else 'moderate' if quality['under_report_prob'] > 0.2 else 'low: patient is forthcoming'}
- Video cooperation: {quality['video_cooperation']:.2f} — {'HIGH: can ask to show skin/body' if quality['video_cooperation'] > 0.5 else 'LOW: patient unlikely to cooperate with visual requests'}
- Engagement: {quality['engagement']:.2f} — {'HIGH: can ask detailed questions' if quality['engagement'] > 0.5 else 'LOW: keep questions brief and focused'}
- Report accuracy: {quality['report_accuracy']:.2f}

YOUR STRATEGY:
1. Compare patient's report with last hospital record — any discrepancies?
2. If under-report is high: ask specifically about common AEs (rash, nausea, neuropathy)
3. If video cooperation is high: request visual inspection (show skin, mouth, hands)
4. Ask about changes since yesterday in specific terms
5. Maximum 3 targeted questions (don't overwhelm the patient)

Output JSON only."""

        user_prompt = f"""DAY {day} — TURN 2

PATIENT'S INITIAL REPORT (T1):
{json.dumps(patient_t1, indent=2, ensure_ascii=False)}

RECENT CALL HISTORY:
{recent_summary}

OUTPUT:
{{
    "approach_style": "empathetic|neutral|concerned|urgent",
    "acknowledgment": "string (brief response to patient's T1)",
    "questions": [
        {{
            "question": "string (what you ask the patient)",
            "target_ae": "string|null (which AE you're probing for)",
            "requires_visual": true/false,
            "rationale": "string (why you're asking this)"
        }}
    ],
    "visual_request": {{
        "requested": true/false,
        "body_area": "string|null (skin, mouth, hands, feet, etc.)",
        "reason": "string|null"
    }},
    "preliminary_concerns": ["string (initial suspicions from T1)"]
}}"""

        try:
            result = generate_json(system_prompt, user_prompt, model=self.model, max_tokens=2048)
        except Exception as e:
            _logger.error(f"[{self.pid}] T2 Nurse LLM failed: {e}")
            result = {
                "approach_style": "empathetic",
                "acknowledgment": "Thank you for sharing.",
                "questions": [
                    {"question": "Have you noticed any skin changes?",
                     "target_ae": "rash", "requires_visual": True,
                     "rationale": "Common AE for this drug"}
                ],
                "visual_request": {"requested": True, "body_area": "skin", "reason": "routine check"},
                "preliminary_concerns": [],
                "_fallback": True,
            }
        result["_turn"] = 2
        return result

    # ══════════════════════════════════════════════════════
    # T3: Patient Follow-up Response
    # ══════════════════════════════════════════════════════

    def _patient_followup_response(
        self,
        day: int,
        day_result: dict,
        nurse_t2: dict,
        quality: dict,
        grade_distortion: int,
    ) -> dict:
        """T3: Nurse의 질문에 환자가 응답한다. updated mood 반영."""
        obj = day_result.get("objective", {})
        active_aes = obj.get("active_aes", [])
        subj = day_result.get("subjective", {})

        # Nurse가 시각 관찰을 요청했는가?
        visual_req = nurse_t2.get("visual_request", {})
        visual_requested = visual_req.get("requested", False)
        # 환자가 협조할 확률
        will_cooperate_visual = (
            visual_requested and self.sampler.boolean(quality["video_cooperation"])
        )

        system_prompt = f"""You are the same clinical trial patient, responding to the nurse's follow-up questions.

PATIENT PROFILE:
- Age: {self.demographics.get('age', '?')}, Sex: {self.demographics.get('sex', '?')}
- Personality: {json.dumps(self.persona, ensure_ascii=False)}

CURRENT MOOD PARAMETERS:
- Under-report probability: {quality['under_report_prob']:.2f}
- Video cooperation: {'WILLING to show' if will_cooperate_visual else 'RELUCTANT to show on camera'}
- Grade distortion: {grade_distortion:+d}
- Engagement: {quality['engagement']:.2f}

RULES:
- The nurse asked specific questions — answer them based on your ACTUAL symptoms
- Your defensiveness may have lowered slightly because the nurse was caring
- If nurse asks about a symptom you HAVE but were hiding: reveal PARTIALLY
  (e.g., "Well, maybe a little..." instead of full description)
- If nurse asks about something you DON'T have: say you don't have it
- If visual cooperation is WILLING: describe what's visible or show it
- If visual cooperation is RELUCTANT: make excuses ("it's hard to show on camera")

Output JSON only."""

        questions = nurse_t2.get("questions", [])
        ae_t3 = json.dumps(
            [{"ae": ae.get("ae", ""), "grade": ae.get("grade", 0), "visual": ae.get("visual")}
             for ae in active_aes], ensure_ascii=False,
        )
        symptoms_t3 = json.dumps(subj.get("symptoms_patient_perceives", []), ensure_ascii=False)

        user_prompt = f"""Day {day} — TURN 3: Responding to nurse's questions

NURSE'S QUESTIONS:
{json.dumps(questions, indent=2, ensure_ascii=False)}

VISUAL REQUEST: {json.dumps(visual_req, ensure_ascii=False)}
YOUR COOPERATION: {'WILLING' if will_cooperate_visual else 'RELUCTANT'}

GROUND TRUTH (your actual state):
- Active AEs: {ae_t3}
- Symptoms perceived: {symptoms_t3}

OUTPUT:
{{
    "responses": [
        {{
            "to_question": "string (nurse's question)",
            "answer": "string (your response in your own words)",
            "revealed_symptom": "string|null (AE term if newly revealed)",
            "honesty_level": "full|partial|evasive|denied"
        }}
    ],
    "visual_response": {{
        "cooperated": true/false,
        "what_shown": "string|null",
        "what_visible_to_nurse": "string|null (what nurse can observe)"
    }},
    "new_info_revealed": true/false,
    "emotional_reaction": "string (patient's emotional response to being asked)"
}}"""

        try:
            result = generate_json(system_prompt, user_prompt, model=self.model, max_tokens=2048)
        except Exception as e:
            _logger.error(f"[{self.pid}] T3 Patient LLM failed: {e}")
            result = {
                "responses": [],
                "visual_response": {"cooperated": will_cooperate_visual,
                                    "what_shown": None, "what_visible_to_nurse": None},
                "new_info_revealed": False,
                "emotional_reaction": "neutral",
                "_fallback": True,
            }
        result["_turn"] = 3
        return result

    # ══════════════════════════════════════════════════════
    # T4: Nurse Final Assessment
    # ══════════════════════════════════════════════════════

    def _nurse_final_assessment(
        self,
        day: int,
        turns: list[dict],
        quality: dict,
    ) -> dict:
        """T4: 모든 턴의 정보를 종합하여 최종 판정 + 개입 결정.

        Nurse가 접근 가능한 정보:
        - turns: 오늘 대화 전체 (T1~T3)
        - call_history: 이전 통화 기록
        - last_hospital_record: 마지막 병원 내원 시 기록
        - known AEs: 이 약물의 알려진 부작용 목록
        - quality: 환자 보고 신뢰도 계수 (mood 모델 기반)

        Ground truth (day_result)는 전달하지 않음.
        """
        recent_summary = self._summarize_recent()
        hospital_summary = self._summarize_hospital_record()

        # 대화 전사록 구성
        conversation_parts = []
        for t in turns:
            role = t["role"].upper()
            turn_n = t["turn"]
            content_summary = json.dumps(t["content"], ensure_ascii=False)[:800]
            conversation_parts.append(f"[T{turn_n} {role}]: {content_summary}")
        conversation_transcript = "\n".join(conversation_parts)

        n_turns = len(turns)
        early = n_turns < 4

        system_prompt = f"""You are an AI nurse making your FINAL assessment after a {'truncated (patient ended call early)' if early else 'complete 4-turn'} video call.

CLINICAL CONTEXT:
- Drug: {self.rule_set.get('drug_name', '?')}
- Indication: {self.rule_set.get('indication', '?')}
- Known AEs: {json.dumps(self.known_aes, ensure_ascii=False)}

LAST HOSPITAL RECORD (from most recent clinic visit):
{hospital_summary}

PATIENT QUALITY METRICS:
- Report accuracy: {quality['report_accuracy']:.2f}
- Compliance rate: {quality['compliance_rate']:.2f}
{'- ⚠️ CALL ENDED EARLY — limited information available' if early else ''}

YOUR ROLE: You are a monitoring AI nurse. You DETECT and REPORT — you do NOT make treatment decisions.
- You CANNOT hold, reduce, or discontinue any medication. Only physicians can do that.
- Your value is EARLY DETECTION and TIMELY REFERRAL to the medical team.

DECISION GUIDELINES (follow this escalation ladder strictly — start low, escalate only when needed):

LEVEL 1 — GREEN (no_action): Patient stable, no new/worsening symptoms. ~40-50% of calls.
LEVEL 2 — YELLOW (monitor_closely): Mild or expected symptoms, track tomorrow. ~30-40% of calls.
  * Grade 1 fatigue, nausea, appetite loss, mild rash, mild tingling are EXPECTED on this regimen.
  * Stable Grade 2 that is already being managed → monitor_closely (NOT early visit).
  * Only escalate beyond this if symptoms are NEWLY WORSENING or FUNCTIONALLY LIMITING.
LEVEL 3 — ORANGE/recommend_conmed: NEW Grade 1-2 symptom that can be managed with supportive medication. ~10-15% of calls.
  * Examples: new nausea → ondansetron, new rash → topical steroid, new itching → antihistamine
  * ALWAYS try recommend_conmed BEFORE recommend_early_visit for Grade 1-2 AEs.
  * This is your PRIMARY intervention tool — use it proactively.
LEVEL 4 — ORANGE/recommend_early_visit: Only when conmed alone is insufficient. ~3-5% of calls.
  * Worsening despite ongoing conmed treatment (not new symptoms)
  * Functional impairment (can't eat, can't walk, can't sleep for >2 days)
  * Multiple Grade 2 AEs simultaneously worsening
  * NEVER use this for isolated Grade 1 or stable Grade 2 AEs.
LEVEL 5 — RED (recommend_hospital_visit): Suspected Grade 3+ symptoms. ~1-2% of calls.
  * Severe pain, high fever, bleeding, severe breathing difficulty
  * Rapid deterioration within the call
LEVEL 6 — RED (escalate_to_physician): Life-threatening emergency only. <1% of calls.
  * Suspected SAE requiring immediate physician decision

CRITICAL RULES:
1. Base detection ONLY on what patient reported + what you observed on video.
2. NEVER recommend dose_hold or treatment changes — that is the physician's decision.
3. Grade 1-2 AEs are EXPECTED during cancer treatment. Manage with conmed, do NOT send to hospital.
4. If the call ended early, note it as a limitation but do NOT assume the worst.
5. Distinguish between patient's PERCEPTION of severity and likely ACTUAL severity.
   A confused/anxious patient saying "terrible" may be Grade 1-2 in reality.
6. COST OF OVER-ESCALATION: Every unnecessary hospital visit → potential dose hold → treatment interruption → worse tumor outcome. Be judicious.
7. recommend_conmed is your most valuable tool. Use it early and often for Grade 1-2 AEs.

Output JSON only."""

        user_prompt = f"""DAY {day} — FINAL ASSESSMENT (Turn 4)

CONVERSATION TRANSCRIPT:
{conversation_transcript}

RECENT CALL HISTORY:
{recent_summary}

OUTPUT:
{{
    "assessment": {{
        "severity_level": "green|yellow|orange|red",
        "summary": "string (1-2 sentence assessment)",
        "detected_issues": [
            {{"issue": "string", "suspected_ae": "string|null",
              "estimated_grade": "int|null", "confidence": "low|medium|high",
              "detection_source": "patient_report|visual_observation|probing_question|clinical_inference"}}
        ]
    }},
    "detection": {{
        "aes_detected": ["string (AEs identified from patient report + video observation)"],
        "concerns_for_followup": ["string (things you couldn't fully assess today — e.g., need labs, need physical exam)"],
        "early_warning_signs": ["string (subtle signs that may indicate emerging AEs)"]
    }},
    "actions": [
        {{
            "action": "no_action|monitor_closely|recommend_conmed|recommend_early_visit|recommend_hospital_visit|escalate_to_physician",
            "reason": "string",
            "detail": "string",
            "urgency": "routine|urgent|emergency"
        }}
    ],
    "call_effectiveness": {{
        "information_gained": "string (what was learned from this call)",
        "limitations": "string (what couldn't be assessed)"
    }}
}}"""

        try:
            result = generate_json(system_prompt, user_prompt, model=self.model, max_tokens=3072)
        except Exception as e:
            _logger.error(f"[{self.pid}] T4 Nurse LLM failed: {e}")
            # Fallback: GT 없이 대화 내용만으로 판단
            has_reported_symptoms = any(
                t.get("content", {}).get("reported_symptoms")
                for t in turns if t.get("role") == "patient"
            )
            result = {
                "assessment": {
                    "severity_level": "yellow" if has_reported_symptoms else "green",
                    "summary": "Assessment unavailable due to system error. Monitoring recommended.",
                    "detected_issues": [],
                },
                "detection": {
                    "aes_detected": [],
                    "concerns_for_followup": ["Unable to assess — LLM unavailable"],
                    "early_warning_signs": [],
                },
                "actions": [{"action": "monitor_closely", "reason": "LLM unavailable",
                             "detail": "Fallback — monitor patient", "urgency": "routine"}],
                "call_effectiveness": {"information_gained": "None (LLM error)",
                                       "limitations": "Full assessment unavailable"},
                "_fallback": True,
            }
        result["_turn"] = 4
        return result

    # ── 유틸리티 ─────────────────────────────────────────

    def _patient_fallback(self, active_aes: list, subj: dict) -> dict:
        """Patient LLM 실패 시 최소한의 보고 생성."""
        return {
            "greeting": "Hi",
            "reported_symptoms": [
                {"symptom": ae.get("ae", ""), "severity_perception": "moderate",
                 "duration": f"{ae.get('days_active', 0)} days",
                 "is_new": ae.get("days_active", 0) <= 1}
                for ae in active_aes
            ],
            "omitted_symptoms": [],
            "general_wellbeing": subj.get("overall_awareness", "OK"),
            "mood_expression": "neutral",
            "video_visible": [],
            "_fallback": True,
        }

    def _summarize_hospital_record(self) -> str:
        """마지막 병원 내원 시 기록을 Nurse가 읽을 수 있는 형태로 요약한다."""
        hr = self._last_hospital_record
        if not hr:
            return "No hospital record available yet (patient has not visited the clinic)."

        visit_day = hr.get("day", "?")
        obj = hr.get("objective", {})
        lines = [f"Last visit: Day {visit_day}"]

        # 치료 상태
        tx = obj.get("treatment_status", "unknown")
        lines.append(f"Treatment status: {tx}")

        # 기록된 AE
        known_aes = obj.get("active_aes", [])
        if known_aes:
            ae_strs = [
                f"  - {ae.get('ae', '?')} Grade {ae.get('grade', '?')} "
                f"(detected day {ae.get('detected_day', ae.get('onset_day', '?'))})"
                for ae in known_aes
            ]
            lines.append("Documented AEs:\n" + "\n".join(ae_strs))
        else:
            lines.append("Documented AEs: none")

        # 마지막 검사 결과 (labs)
        labs = obj.get("labs", {})
        if labs:
            abnormal = {k: v for k, v in labs.items()
                        if isinstance(v, dict) and v.get("flag") in ("H", "L", "HH", "LL")}
            if abnormal:
                lab_strs = [f"  - {k}: {v.get('value', '?')} {v.get('unit', '')} [{v.get('flag', '')}]"
                            for k, v in abnormal.items()]
                lines.append("Abnormal labs:\n" + "\n".join(lab_strs))
            else:
                lines.append("Labs: all within normal range")

        # 마지막 활력징후
        vitals = obj.get("vitals", {})
        if vitals:
            v_items = []
            for k in ("BT", "HR", "SBP", "DBP", "SpO2", "weight_kg"):
                if k in vitals:
                    v_items.append(f"{k}={vitals[k]}")
            if v_items:
                lines.append(f"Vitals: {', '.join(v_items)}")

        # ECOG
        ecog = obj.get("ecog")
        if ecog is not None:
            lines.append(f"ECOG PS: {ecog}")

        return "\n".join(lines)

    def _summarize_recent(self) -> str:
        """최근 N일 통화 이력을 요약한다.

        핵심: Nurse가 알 수 있는 정보만 포함한다.
        - 이전 통화에서 환자가 보고한 증상
        - 이전 Nurse 판정 결과
        - Ground truth (objective.active_aes)는 절대 포함하지 않음
        """
        if not self.call_history:
            return "No prior calls."
        lines = []
        for entry in self.call_history[-5:]:
            d = entry.get("day", "?")
            severity = entry.get("severity", "?")
            summary = entry.get("summary", "")[:100]
            actions = entry.get("actions", [])
            early = " (early termination)" if entry.get("terminated_early") else ""
            lines.append(
                f"Day {d}: {severity}{early} — {summary}"
                + (f" → Actions: {', '.join(actions)}" if actions else "")
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════
# Mood Effect Helpers
# ══════════════════════════════════════════════════════

def _nurse_approach_mood_effect(approach: str) -> dict[str, float]:
    """Nurse의 접근 방식이 환자 mood에 미치는 효과."""
    effects = {
        "empathetic": {
            "trust_in_ai": +0.05, "defensiveness": -0.04,
            "anxiety": -0.02, "irritability": -0.02,
        },
        "neutral": {
            "trust_in_ai": +0.01, "defensiveness": -0.01,
        },
        "concerned": {
            "anxiety": +0.04, "trust_in_ai": +0.03,
            "defensiveness": -0.03,
        },
        "urgent": {
            "anxiety": +0.08, "trust_in_ai": +0.02,
            "defensiveness": -0.06, "energy": +0.03,
        },
    }
    return effects.get(approach, effects["neutral"])


# ══════════════════════════════════════════════════════
# Intervention Engine — Care Agent의 개입을 시뮬레이션에 반영
# ══════════════════════════════════════════════════════

def apply_interventions(
    care_record: dict,
    simulator: Any,
    day: int,
) -> list[str]:
    """care_record의 actions를 DailySimulator 상태에 반영한다.

    핵심 원칙: Care AI는 감지+보고만 한다. 치료 결정은 하지 않는다.
    - dose_hold는 Care AI가 직접 적용하지 않음
    - 대신 hospital_visit/early_visit를 통해 의사가 판단하도록 연결
    - conmed(보조약)만 직접 추가 가능 (간호사 권한 범위)

    Args:
        care_record: CareAgent.conduct_video_call()의 출력
        simulator: DailySimulator 인스턴스
        day: 현재 날짜

    Returns:
        적용된 개입 목록 (로깅용)
    """
    applied: list[str] = []
    actions = care_record.get("actions", [])

    for action_record in actions:
        action = action_record.get("action", "no_action")
        detail = action_record.get("detail", "")
        reason = action_record.get("reason", "")

        if action == "no_action":
            continue

        elif action == "monitor_closely":
            applied.append(f"monitor_closely: {reason}")

        elif action == "recommend_conmed":
            cm_record = _create_conmed_from_recommendation(detail, reason, day)
            if cm_record:
                existing = [
                    cm for cm in simulator.active_cm
                    if cm.get("CMTRT") == cm_record.get("CMTRT") and cm.get("CMONGO", False)
                ]
                if not existing:
                    simulator.active_cm.append(cm_record)
                    applied.append(f"conmed_added: {cm_record.get('CMTRT', '?')}")
                    _logger.info(
                        f"  [CareIntervention] CM added: {cm_record.get('CMTRT')} for {reason}"
                    )

        elif action == "recommend_dose_hold":
            # Care AI는 dose_hold를 직접 적용하지 않음
            # 대신 hospital_visit로 격상하여 의사가 판단하도록 함
            applied.append(f"referred_to_physician: {reason}")
            _logger.info(
                f"  [CareIntervention] Dose hold requested → referred to physician: {reason}"
            )
            # force_hospital_tomorrow는 orchestrator에서 처리

        elif action == "recommend_early_visit":
            # 예정보다 빨리 클리닉 방문 권고 → 다음 날 병원 방문
            applied.append(f"early_visit: {reason}")
            _logger.info(f"  [CareIntervention] Early clinic visit recommended: {reason}")

        elif action == "recommend_hospital_visit":
            applied.append(f"hospital_visit: {reason}")
            _logger.info(f"  [CareIntervention] Hospital visit recommended: {reason}")

        elif action == "escalate_to_physician":
            urgency = action_record.get("urgency", "urgent")
            applied.append(f"escalation_{urgency}: {reason}")
            _logger.info(
                f"  [CareIntervention] Escalated to physician ({urgency}): {reason}"
            )
            # emergency 상황에서도 Care AI가 직접 hold하지 않음
            # orchestrator에서 hospital visit을 강제하고, 의사가 판단

    return applied


def _create_conmed_from_recommendation(detail: str, reason: str, day: int) -> dict | None:
    """Nurse의 추천에서 CM 레코드를 생성한다."""
    detail_lower = detail.lower()
    reason_lower = reason.lower()
    combined = detail_lower + " " + reason_lower

    med_map = {
        "nausea": ("Ondansetron", "8mg", "ORAL", "PRN"),
        "vomit": ("Ondansetron", "8mg", "ORAL", "PRN"),
        "emesis": ("Ondansetron", "8mg", "ORAL", "PRN"),
        "diarrhea": ("Loperamide", "4mg then 2mg", "ORAL", "PRN"),
        "pain": ("Acetaminophen", "500mg", "ORAL", "Q6H PRN"),
        "headache": ("Acetaminophen", "500mg", "ORAL", "Q6H PRN"),
        "fever": ("Acetaminophen", "650mg", "ORAL", "Q6H PRN"),
        "rash": ("Diphenhydramine", "25mg", "ORAL", "Q8H PRN"),
        "itch": ("Diphenhydramine", "25mg", "ORAL", "Q8H PRN"),
        "prurit": ("Diphenhydramine", "25mg", "ORAL", "Q8H PRN"),
        "mucosit": ("Mouthwash (lidocaine-based)", "15mL", "ORAL", "QID"),
        "stomatit": ("Mouthwash (lidocaine-based)", "15mL", "ORAL", "QID"),
        "constipat": ("Docusate", "100mg", "ORAL", "BID"),
        "fatigue": ("Activity modification", "", "NON-DRUG", ""),
        "anorexia": ("Megestrol acetate", "400mg", "ORAL", "QD"),
        "appetite": ("Megestrol acetate", "400mg", "ORAL", "QD"),
        "neuropath": ("Gabapentin", "300mg", "ORAL", "TID"),
        "tingl": ("Gabapentin", "300mg", "ORAL", "TID"),
        "numb": ("Gabapentin", "300mg", "ORAL", "TID"),
    }

    for keyword, (drug, dose, route, freq) in med_map.items():
        if keyword in combined:
            return {
                "CMTRT": drug,
                "CMINDC": reason,
                "CMDSTXT": dose,
                "CMDOSU": "mg",
                "CMDOSFRQ": freq,
                "CMROUTE": route,
                "CMSTDAT": day,
                "CMONGO": True,
                "CMENDAT": None,
                "_source": "care_ai_recommendation",
            }

    return {
        "CMTRT": "Supportive care (per physician)",
        "CMINDC": reason,
        "CMDSTXT": "",
        "CMDOSU": "",
        "CMDOSFRQ": "PRN",
        "CMROUTE": "ORAL",
        "CMSTDAT": day,
        "CMONGO": True,
        "CMENDAT": None,
        "_source": "care_ai_recommendation",
    }