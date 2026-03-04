"""observation.py — Ground Truth ↔ Hospital Record 관찰 모델

환자의 "실제 상태" (ground truth)와 "병원이 아는 상태" (hospital record)를 분리한다.
이 분리가 핵심 가치: Care AI가 없으면 병원은 내원/보고 시점에만 상태를 파악.
Care AI가 있으면 매일 영상통화를 통해 시각적 AE를 추가 감지.

의학적 근거:
  - CTCAE 기반 AE는 의사 평가(clinician-reported) vs 환자 보고(PRO) 간 불일치
  - Basch et al. (2006): 환자-의사 간 AE grade 일치율 ~50%
  - Di Maio et al. (2015): 의사는 57% AE를 저평가
  - 환자는 내원 시에만 검사 → 사이 기간의 상태는 알 수 없음

관찰 지점 (Observation Point):
  1. 예정 방문 (scheduled_visit) — 투약일 외래 → 전체 검사
  2. RECIST 스캔 (scheduled_scan) — 영상 검사 → 종양 정보 갱신
  3. 환자 자발 보고 (self_report) — 전화/ER 방문 → mood 의존
  4. Care AI 영상통화 (video_call) — 시각/청각 AE 감지 → 매일
  5. 응급실 (er_visit) — 의학적 기준 초과 시 → 전체 검사
"""

from __future__ import annotations

import copy
from typing import Any

from sim.engine.mood import (
    MoodState,
    compute_self_report_probability,
    compute_grade_distortion,
    compute_interaction_quality,
    should_visit_er,
)
from sim.engine.sampler import Sampler

# ══════════════════════════════════════════════════════
# A. AE 감지 채널 정의
# ══════════════════════════════════════════════════════

# 각 AE가 어떤 경로로 감지 가능한지 분류
# "lab": 검사로만 감지 (환자는 무증상일 수 있음)
# "patient_reported": 환자가 증상을 느끼고 보고
# "video_detectable": Care AI 영상통화로 시각/청각적 감지 가능
# "physical_exam": 의사의 신체검진으로 감지

AE_DETECTION_CHANNELS: dict[str, dict[str, Any]] = {
    # ── 혈액학적 (Lab-detected: 환자는 모름) ──
    "neutropenia": {
        "channels": ["lab"],
        "patient_aware_threshold": 3,  # Grade 3+: 열/감염으로 증상
        "detection_requires_lab": True,
    },
    "thrombocytopenia": {
        "channels": ["lab", "video_detectable"],
        "patient_aware_threshold": 2,  # petechiae → 시각적
        "video_signs": ["bruising", "petechiae"],
        "detection_requires_lab": True,
    },
    "anemia": {
        "channels": ["lab", "video_detectable", "patient_reported"],
        "patient_aware_threshold": 2,
        "video_signs": ["pallor", "conjunctival_pallor"],
    },
    "leukopenia": {
        "channels": ["lab"],
        "patient_aware_threshold": 3,
        "detection_requires_lab": True,
    },

    # ── 피부 (Video-detectable: Care AI가 볼 수 있음) ──
    "rash": {
        "channels": ["patient_reported", "video_detectable", "physical_exam"],
        "patient_aware_threshold": 1,
        "video_signs": ["visible_rash", "erythema", "papules"],
    },
    "rash_maculopapular": {
        "channels": ["patient_reported", "video_detectable", "physical_exam"],
        "patient_aware_threshold": 1,
        "video_signs": ["visible_rash", "erythema", "maculopapular_lesions"],
    },
    "pruritus": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },
    "skin_eruption": {
        "channels": ["patient_reported", "video_detectable", "physical_exam"],
        "patient_aware_threshold": 1,
        "video_signs": ["skin_lesions", "erythema"],
    },
    "palmar_plantar": {
        "channels": ["patient_reported", "video_detectable"],
        "patient_aware_threshold": 1,
        "video_signs": ["hand_redness", "peeling_skin"],
    },
    "alopecia": {
        "channels": ["video_detectable", "patient_reported"],
        "patient_aware_threshold": 1,
        "video_signs": ["visible_hair_loss", "thinning_hair"],
    },
    "nail_changes": {
        "channels": ["video_detectable", "patient_reported"],
        "patient_aware_threshold": 1,
        "video_signs": ["nail_discoloration", "nail_dystrophy"],
    },
    "nail_loss": {
        "channels": ["video_detectable", "patient_reported"],
        "patient_aware_threshold": 1,
        "video_signs": ["nail_separation", "nail_absent"],
    },
    "stomatitis": {
        "channels": ["patient_reported", "video_detectable"],
        "patient_aware_threshold": 1,
        "video_signs": ["oral_lesions", "lip_swelling"],
    },
    "mucositis": {
        "channels": ["patient_reported", "video_detectable"],
        "patient_aware_threshold": 1,
        "video_signs": ["oral_erythema", "lip_dryness"],
    },

    # ── 전신 (환자 보고 중심) ──
    "fatigue": {
        "channels": ["patient_reported", "video_detectable"],
        "patient_aware_threshold": 1,
        "video_signs": ["visible_fatigue", "slow_movements"],
    },
    "nausea": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },
    "vomiting": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },
    "diarrhea": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },
    "constipation": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },
    "decreased_appetite": {
        "channels": ["patient_reported", "video_detectable"],
        "patient_aware_threshold": 1,
        "video_signs": ["weight_loss_visible", "cachexia"],
    },

    # ── 신경학적 ──
    "peripheral_neuropathy": {
        "channels": ["patient_reported", "physical_exam"],
        "patient_aware_threshold": 1,  # 저림/통증 환자가 인지
    },
    "neuropathy": {
        "channels": ["patient_reported", "physical_exam"],
        "patient_aware_threshold": 1,
    },
    "dysgeusia": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },

    # ── 호흡기 ──
    "dyspnea": {
        "channels": ["patient_reported", "video_detectable"],
        "patient_aware_threshold": 1,
        "video_signs": ["labored_breathing", "tachypnea"],
    },
    "cough": {
        "channels": ["patient_reported", "video_detectable"],
        "patient_aware_threshold": 1,
        "video_signs": ["audible_cough"],
    },
    "pneumonitis": {
        "channels": ["patient_reported", "lab", "physical_exam"],
        "patient_aware_threshold": 2,
    },

    # ── 간 (Lab-detected) ──
    "hepatotoxicity": {
        "channels": ["lab", "video_detectable"],
        "patient_aware_threshold": 3,
        "video_signs": ["jaundice", "scleral_icterus"],
        "detection_requires_lab": True,
    },
    "hepatitis": {
        "channels": ["lab", "video_detectable"],
        "patient_aware_threshold": 3,
        "video_signs": ["jaundice"],
        "detection_requires_lab": True,
    },
    "alt_increased": {
        "channels": ["lab"],
        "patient_aware_threshold": 3,
        "detection_requires_lab": True,
    },
    "ast_increased": {
        "channels": ["lab"],
        "patient_aware_threshold": 3,
        "detection_requires_lab": True,
    },

    # ── 신장 ──
    "nephrotoxicity": {
        "channels": ["lab"],
        "patient_aware_threshold": 3,
        "detection_requires_lab": True,
    },
    "nephritis": {
        "channels": ["lab", "patient_reported"],
        "patient_aware_threshold": 2,
        "detection_requires_lab": True,
    },
    "proteinuria": {
        "channels": ["lab"],
        "patient_aware_threshold": 3,
        "detection_requires_lab": True,
    },

    # ── 내분비 ──
    "hypothyroidism": {
        "channels": ["lab", "patient_reported"],
        "patient_aware_threshold": 2,
        "detection_requires_lab": True,
    },
    "hyperthyroidism": {
        "channels": ["lab", "patient_reported"],
        "patient_aware_threshold": 2,
        "detection_requires_lab": True,
    },
    "hyperglycemia": {
        "channels": ["lab", "patient_reported"],
        "patient_aware_threshold": 2,
    },
    "adrenal_insufficiency": {
        "channels": ["lab", "patient_reported", "video_detectable"],
        "patient_aware_threshold": 2,
        "video_signs": ["hyperpigmentation", "visible_weakness"],
    },

    # ── 통증 ──
    "arthralgia": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },
    "myalgia": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },
    "headache": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },

    # ── 비뇨기 ──
    "urinary_tract_infection": {
        "channels": ["patient_reported", "lab"],
        "patient_aware_threshold": 1,
    },
    "hematuria": {
        "channels": ["patient_reported", "lab"],
        "patient_aware_threshold": 1,
    },

    # ── 심장 ──
    "myocarditis": {
        "channels": ["lab", "patient_reported", "physical_exam"],
        "patient_aware_threshold": 2,
        "detection_requires_lab": True,
    },

    # ── 면역 관련 ──
    "colitis": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },
    "infusion_related_reaction": {
        "channels": ["physical_exam", "video_detectable", "patient_reported"],
        "patient_aware_threshold": 1,
        "video_signs": ["flushing", "urticaria", "angioedema"],
    },

    # ── 기타/감염 ──
    "febrile_neutropenia": {
        "channels": ["patient_reported", "lab", "physical_exam"],
        "patient_aware_threshold": 1,  # 열은 환자가 인지
    },
    "sepsis": {
        "channels": ["patient_reported", "lab", "physical_exam"],
        "patient_aware_threshold": 1,
    },
}

def get_ae_channels(ae_term: str) -> dict[str, Any]:
    """AE의 감지 채널 정보를 반환한다.

    미등록 AE는 경고 로그를 남기고 patient_reported로 처리.
    AE_DETECTION_CHANNELS에 등록되지 않은 AE가 나오면 즉시 확인 필요.
    """
    # 정확한 매칭 먼저
    if ae_term in AE_DETECTION_CHANNELS:
        return AE_DETECTION_CHANNELS[ae_term]
    # 부분 매칭 (e.g., "peripheral_neuropathy_sensory" → "peripheral_neuropathy")
    for key in AE_DETECTION_CHANNELS:
        if key in ae_term or ae_term in key:
            return AE_DETECTION_CHANNELS[key]
    import logging
    logging.getLogger(__name__).warning(
        f"AE '{ae_term}' not in AE_DETECTION_CHANNELS — "
        f"defaulting to patient_reported. Add this AE to the channel registry."
    )
    return {"channels": ["patient_reported"], "patient_aware_threshold": 1}


# ══════════════════════════════════════════════════════
# B. 관찰 지점에서 캡처되는 데이터
# ══════════════════════════════════════════════════════

# 각 관찰 지점에서 hospital_record에 기록되는 도메인
OBSERVATION_CAPTURE: dict[str, dict[str, bool]] = {
    "scheduled_visit": {
        # 투약일 외래: 전체 검사
        "labs": True,
        "vitals": True,
        "physical_exam": True,
        "ae_assessment": True,
        "ecog": True,
        "weight": True,
        "patient_interview": True,
    },
    "scheduled_scan": {
        # RECIST 영상 스캔
        "tumor": True,
    },
    "self_report": {
        # 환자가 전화 — patient_reported AE만
        "ae_patient_reported": True,
    },
    "video_call": {
        # Care AI 영상통화 — 시각/청각 AE + 환자 보고
        "ae_video_detectable": True,
        "ae_patient_reported": True,
        "vitals_patient_reported": True,  # "열이 있어요" 등
    },
    "er_visit": {
        # 응급실 — 전체 검사
        "labs": True,
        "vitals": True,
        "physical_exam": True,
        "ae_assessment": True,
        "ecog": True,
    },
}


# ══════════════════════════════════════════════════════
# C. ObservationModel — Ground Truth → Hospital Record 변환
# ══════════════════════════════════════════════════════

class ObservationModel:
    """환자 1명의 관찰 모델.

    매일 ground_truth를 받아서:
    1. 관찰 지점을 결정
    2. 관찰 가능한 정보만 hospital_record에 반영
    3. 관찰 불가능한 정보는 이전 hospital_record 유지 (stale)
    """

    def __init__(
        self,
        mood: MoodState,
        sampler: Sampler,
        care_ai_enabled: bool = False,
    ):
        self.mood = mood
        self.sampler = sampler
        self.care_ai_enabled = care_ai_enabled

        # 병원이 "마지막으로 아는" 상태
        self.last_hospital_record: dict | None = None
        self.last_visit_day: int = 0
        self.known_aes: dict[str, dict] = {}  # ae_term → {grade, detected_day, channel}
        self.known_labs: dict = {}
        self.known_vitals: dict = {}
        self.known_ecog: int | None = None
        self.known_tumor: dict | None = None
        self.known_treatment_status: str = "on_treatment"  # 병원이 아는 치료 상태

        # 감지 기록 (분석용)
        self.detection_log: list[dict] = []

        # ── 전날 상태 추적 (transition vs ongoing 구분용) ──
        self._prev_treatment_status: str = "on_treatment"
        self._prev_has_grade3: bool = False
        self._prev_ae_terms: set[str] = set()

    def process_day(
        self,
        ground_truth: dict | None = None,
        day: int = 0,
        is_hospital: bool = False,
        is_admin_day: bool = False,
        care_record: list[dict] | None = None,
        *,
        day_result: dict | None = None,
        simulator=None,
    ) -> tuple[bool, dict]:
        """하루의 ground truth를 받아 (is_visit, observation_result)를 반환.

        Args:
            ground_truth: DailySimulator.generate_day()의 raw 출력 (positional 호환)
            day: 시뮬레이션 일수
            is_hospital: 병원 방문일 여부
            is_admin_day: 투약일 여부
            care_record: Care AI 통화 기록
            day_result: ground_truth의 keyword alias (orchestrator 호환)
            simulator: DailySimulator 인스턴스 (현재 미사용, 호환용)

        Returns:
            (is_visit, observation_result) 튜플
            - is_visit: 이 날 병원/클리닉 방문이 있었는가
            - observation_result: {ground_truth, hospital_record, observation_events, ...}
        """
        # day_result kwarg 호환: orchestrator는 day_result= 으로 호출
        if ground_truth is None and day_result is not None:
            ground_truth = day_result
        if ground_truth is None:
            raise ValueError("process_day requires ground_truth or day_result")
        obj = ground_truth.get("objective", {})
        active_aes = obj.get("active_aes", [])

        # 현재 최대 AE grade (mood, ER 판정용)
        max_grade = max((ae.get("grade", 0) for ae in active_aes), default=0)
        vitals = obj.get("vitals", {})
        labs = obj.get("labs", {})

        # ── Mood 이벤트 목록 생성 ──
        mood_events = self._collect_mood_events(ground_truth, day, is_admin_day)
        mood_delta = self.mood.update_daily(mood_events)

        # Grade 3+ 방어 무너짐
        self.mood.apply_defensiveness_override(max_grade)

        # ── 관찰 지점 결정 ──
        observation_events: list[dict] = []

        # (1) 예정 방문 (투약일 or 병원일)
        if is_hospital or is_admin_day:
            observation_events.append({
                "type": "scheduled_visit",
                "day": day,
                "captures": OBSERVATION_CAPTURE["scheduled_visit"],
            })
            self.last_visit_day = day

        # (2) RECIST 스캔 (ground truth에 recist 결과가 있으면)
        recist = ground_truth.get("recist_scan") or ground_truth.get("_recist_event")
        if recist:
            observation_events.append({
                "type": "scheduled_scan",
                "day": day,
                "captures": OBSERVATION_CAPTURE["scheduled_scan"],
            })

        # (3) ER 방문 (의학적 기준)
        if should_visit_er(max_grade, vitals, labs):
            observation_events.append({
                "type": "er_visit",
                "day": day,
                "captures": OBSERVATION_CAPTURE["er_visit"],
            })
            self.last_visit_day = day

        # (4) 환자 자발 보고 (mood 기반 확률)
        days_since_visit = day - self.last_visit_day
        self_report_prob = compute_self_report_probability(
            self.mood, max_grade, days_since_visit
        )
        if not is_hospital and self.sampler.boolean(self_report_prob):
            observation_events.append({
                "type": "self_report",
                "day": day,
                "captures": OBSERVATION_CAPTURE["self_report"],
                "probability": round(self_report_prob, 3),
            })

        # (5) Care AI 영상통화 (매일, Care AI 활성화 시)
        if self.care_ai_enabled and care_record:
            interaction_quality = compute_interaction_quality(self.mood)
            observation_events.append({
                "type": "video_call",
                "day": day,
                "captures": OBSERVATION_CAPTURE["video_call"],
                "interaction_quality": interaction_quality,
            })

        # ── Day 1 초기화: 첫 방문이므로 전체 정보 캡처 (HR 빌드 전 수행) ──
        if day == 1:
            self.known_labs = copy.deepcopy(labs)
            self.known_vitals = copy.deepcopy(vitals)
            self.known_ecog = obj.get("ecog")
            self.known_tumor = copy.deepcopy(obj.get("tumor"))
            self.last_visit_day = 1

        # ── Hospital Record 갱신 ──
        hospital_record = self._update_hospital_record(
            ground_truth, day, observation_events, active_aes
        )

        # is_visit: 병원/클리닉 방문 여부 (dose modification 판단용)
        obs_types_final = {e["type"] for e in observation_events}
        is_visit = (
            "scheduled_visit" in obs_types_final
            or "er_visit" in obs_types_final
        )

        result = {
            "ground_truth": ground_truth,
            "hospital_record": hospital_record,
            "observation_events": [
                {k: v for k, v in e.items() if k != "captures"}
                for e in observation_events
            ],
            "mood_state": self.mood.to_dict(),
            "mood_events": mood_events,
            "detection_log": list(self.detection_log[-10:]),
            # orchestrator 호환: observed.get('objective', {}).get('active_aes', [])
            "objective": hospital_record.get("objective", {}),
        }
        return is_visit, result

    # ── 내부: Hospital Record 갱신 ──────────────────────

    def _update_hospital_record(
        self,
        ground_truth: dict,
        day: int,
        observation_events: list[dict],
        active_aes: list[dict],
    ) -> dict:
        """관찰 이벤트에 따라 hospital_record를 갱신한다."""

        obj = ground_truth.get("objective", {})
        obs_types = {e["type"] for e in observation_events}

        # 전체 검사 (scheduled_visit, er_visit)
        full_exam = "scheduled_visit" in obs_types or "er_visit" in obs_types

        # ── Labs ──
        if full_exam:
            self.known_labs = copy.deepcopy(obj.get("labs", {}))

        # ── Vitals ──
        if full_exam:
            self.known_vitals = copy.deepcopy(obj.get("vitals", {}))
        elif "video_call" in obs_types:
            # Care AI: 환자가 말해주는 vitals만 (체온 등)
            # 정확한 혈압/심박수는 알 수 없음
            patient_temp = obj.get("vitals", {}).get("body_temperature")
            if patient_temp:
                if "body_temperature" not in self.known_vitals:
                    self.known_vitals["body_temperature"] = {}
                self.known_vitals["body_temperature"] = patient_temp

        # ── ECOG ──
        if full_exam:
            self.known_ecog = obj.get("ecog")

        # ── Tumor ──
        if "scheduled_scan" in obs_types:
            self.known_tumor = copy.deepcopy(obj.get("tumor"))

        # ── AE 감지 ──
        grade_distortion = compute_grade_distortion(self.mood)

        for ae in active_aes:
            ae_term = ae.get("ae", ae.get("ae_term", "unknown"))
            ae_grade = ae.get("grade", 1)
            channels = get_ae_channels(ae_term)
            ae_channels_available = channels.get("channels", ["patient_reported"])
            patient_aware_threshold = channels.get("patient_aware_threshold", 1)
            requires_lab = channels.get("detection_requires_lab", False)

            detected = False
            detection_channel = None

            # (A) 전체 검사에서 감지 (lab, physical_exam 포함)
            if full_exam:
                detected = True
                detection_channel = "clinical_assessment"

            # (B) Lab 필수인데 검사 안 한 경우 — 환자가 인지하는 threshold 이상이면 보고 가능
            elif requires_lab and ae_grade < patient_aware_threshold:
                # 환자도 모르고, lab도 안 했으므로 감지 불가
                detected = False

            # (C) 환자 보고
            elif "patient_reported" in ae_channels_available:
                if ae_grade >= patient_aware_threshold:
                    # 환자가 인지 → 보고 여부는 mood에 따라
                    if "self_report" in obs_types or full_exam:
                        detected = True
                        detection_channel = "patient_reported"
                    elif "video_call" in obs_types:
                        # Care AI 통화 중 환자가 말함 (engagement 기반)
                        iq = next(
                            (e.get("interaction_quality", {})
                             for e in observation_events if e["type"] == "video_call"),
                            {},
                        )
                        engagement = iq.get("engagement", 0.5)
                        under_report = iq.get("under_report_prob", 0.3)
                        # 참여도 높고 축소보고 안 하면 보고
                        if self.sampler.boolean(engagement * (1 - under_report)):
                            detected = True
                            detection_channel = "patient_reported_via_video"

            # (D) Care AI 영상통화: 시각/청각 감지
            if not detected and "video_call" in obs_types and "video_detectable" in ae_channels_available:
                iq = next(
                    (e.get("interaction_quality", {})
                     for e in observation_events if e["type"] == "video_call"),
                    {},
                )
                video_coop = iq.get("video_cooperation", 0.5)
                # 시각적 AE 감지 확률: 협조도 × 기본 감지율(grade 비례)
                base_detect = min(0.3 + ae_grade * 0.15, 0.90)
                if self.sampler.boolean(base_detect * video_coop):
                    detected = True
                    detection_channel = "video_detected"

            # ── 감지된 AE를 known_aes에 등록/갱신 ──
            if detected:
                # 병원이 기록하는 grade (distortion 적용)
                if detection_channel in ("clinical_assessment",):
                    # 의사 평가: 정확함
                    reported_grade = ae_grade
                else:
                    # 환자 보고/영상 감지: distortion 적용
                    reported_grade = max(1, min(5, ae_grade + grade_distortion))

                if ae_term not in self.known_aes:
                    # 새 감지
                    self.known_aes[ae_term] = {
                        "grade": reported_grade,
                        "detected_day": day,
                        "actual_onset_day": ae.get("onset_day"),
                        "channel": detection_channel,
                        "status": ae.get("status", "active_stable"),
                    }
                    self.detection_log.append({
                        "ae_term": ae_term,
                        "day": day,
                        "actual_onset_day": ae.get("onset_day"),
                        "detection_delay": day - ae.get("onset_day", day),
                        "channel": detection_channel,
                        "actual_grade": ae_grade,
                        "reported_grade": reported_grade,
                    })
                else:
                    # 이미 알고 있는 AE: grade 갱신
                    prev = self.known_aes[ae_term]
                    if detection_channel == "clinical_assessment":
                        prev["grade"] = ae_grade  # 의사 평가로 정확히 갱신
                    else:
                        prev["grade"] = reported_grade
                    prev["status"] = ae.get("status", "active_stable")
                    prev["last_updated_day"] = day

        # ── 해소된 AE 처리 ──
        active_ae_terms = {ae.get("ae", ae.get("ae_term", "")) for ae in active_aes}
        for ae_term in list(self.known_aes.keys()):
            if ae_term not in active_ae_terms:
                # Ground truth에서는 해소됐지만, 병원은 전체 검사 시에만 파악
                if full_exam:
                    self.known_aes[ae_term]["status"] = "resolved"
                    self.known_aes[ae_term]["resolved_day"] = day
                # 그렇지 않으면 병원은 여전히 active로 알고 있음

        # ── Hospital Record 조립 ──
        hr_aes = []
        for ae_term, ae_info in self.known_aes.items():
            if ae_info.get("status") == "resolved" and ae_info.get("resolved_day", day) < day - 7:
                continue  # 해소된 지 7일 넘으면 목록에서 제거
            hr_aes.append({
                "ae": ae_term,
                "grade": ae_info["grade"],
                "onset_day": ae_info.get("actual_onset_day"),
                "detected_day": ae_info["detected_day"],
                "detection_delay": ae_info["detected_day"] - ae_info.get("actual_onset_day", ae_info["detected_day"]),
                "channel": ae_info["channel"],
                "status": ae_info.get("status", "active_stable"),
            })

        # ── Hospital Record 조립 (화이트리스트 방식) ──
        # GT에서 자동 복사하지 않음. 병원이 아는 정보만 명시적으로 구성.
        hospital_record = {
            "patient_id": ground_truth.get("patient_id"),
            "day": day,
            "observation_types": sorted(obs_types),
            "objective": {
                "location": ground_truth.get("objective", {}).get("location", "HOME"),
                "treatment_status": self.known_treatment_status,
                "labs": copy.deepcopy(self.known_labs),
                "vitals": copy.deepcopy(self.known_vitals),
                "active_aes": hr_aes,
                "ecog": self.known_ecog,
                "tumor": copy.deepcopy(self.known_tumor),
                "labs_stale_days": day - self.last_visit_day if not full_exam else 0,
                "vitals_stale_days": day - self.last_visit_day if not full_exam else 0,
            },
        }

        # 투약(EC), 병용약(CM), disposition(DS)은 병원 시스템이 관리 → 항상 정확
        for domain in ("EC", "CM", "DS"):
            if domain in ground_truth:
                hospital_record[domain] = ground_truth[domain]

        # AE CRF 레코드: 병원이 아는 AE만 포함
        if "AE" in ground_truth:
            hospital_record["AE"] = self._filter_ae_crf(ground_truth.get("AE", []))

        # subjective는 관찰 시에만 기록
        if observation_events:
            hospital_record["subjective"] = ground_truth.get("subjective")
        else:
            hospital_record["subjective"] = None

        self.last_hospital_record = hospital_record
        return hospital_record

    def update_treatment_status(self, new_status: str) -> None:
        """병원이 아는 치료 상태를 갱신한다.

        orchestrator에서 dose modification 결정 후 호출.
        hospital_record의 treatment_status도 함께 갱신.
        """
        self.known_treatment_status = new_status
        if self.last_hospital_record:
            self.last_hospital_record["objective"]["treatment_status"] = new_status

    def _filter_ae_crf(self, ae_records: list[dict]) -> list[dict]:
        """CRF AE 레코드를 병원이 아는 AE로 필터링."""
        if not ae_records:
            return []
        filtered = []
        for rec in ae_records:
            ae_term = rec.get("AETERM", rec.get("ae_term", ""))
            ae_lower = ae_term.lower().replace(" ", "_")
            if ae_lower in self.known_aes or any(k in ae_lower or ae_lower in k for k in self.known_aes):
                filtered.append(rec)
        return filtered

    def _collect_mood_events(
        self,
        ground_truth: dict,
        day: int,
        is_admin_day: bool,
    ) -> list[str]:
        """오늘의 이벤트에서 mood에 영향주는 이벤트 목록 추출.

        핵심: transition(첫 날) vs ongoing(지속) 구분.
        - transition: full effect (e.g., 치료 중단된 날)
        - ongoing: "_ongoing:" prefix → MoodState에서 ONGOING_EVENT_DAMPING 적용
        이렇게 해야 장기 지속 상태에서 mood가 1.0에 포화되지 않는다.
        """
        events: list[str] = []
        obj = ground_truth.get("objective", {})
        active_aes = obj.get("active_aes", [])

        # 신규 AE (항상 transition — 한 번만 발생)
        for ae in active_aes:
            if ae.get("status") == "new_onset":
                events.append("new_ae_onset")

        # AE 악화/개선 (항상 transition)
        raw_events = ground_truth.get("_events", [])
        if isinstance(raw_events, list):
            for ev_str in raw_events:
                if isinstance(ev_str, str):
                    if "worsened" in ev_str.lower():
                        events.append("ae_grade_worsened")
                    elif "improved" in ev_str.lower():
                        events.append("ae_grade_improved")
                    elif "resolved" in ev_str.lower():
                        events.append("ae_resolved")

        # Grade 3+ 활성 AE: transition vs ongoing
        has_grade3 = any(ae.get("grade", 0) >= 3 for ae in active_aes)
        if has_grade3:
            if not self._prev_has_grade3:
                events.append("grade3_or_higher")  # 첫 날: full effect
            else:
                events.append("_ongoing:grade3_or_higher")  # 이후: damped

        # 투약일 (항상 transition — 간헐적 이벤트)
        if is_admin_day:
            events.append("infusion_day")

        # 치료 상태: transition vs ongoing
        ts = obj.get("treatment_status", "")
        if ts == "discontinued":
            if self._prev_treatment_status != "discontinued":
                events.append("treatment_discontinued")  # 중단된 날: full
            else:
                events.append("_ongoing:treatment_discontinued")  # 이후: damped
        elif ts in ("held", "partially_held"):
            if self._prev_treatment_status not in ("held", "partially_held"):
                events.append("treatment_held")  # hold 된 날: full
            else:
                events.append("_ongoing:treatment_held")  # 이후: damped

        # ── 전날 상태 업데이트 ──
        self._prev_treatment_status = ts
        self._prev_has_grade3 = has_grade3
        self._prev_ae_terms = {
            ae.get("ae", ae.get("ae_term", ""))
            for ae in active_aes
        }

        return events


# ══════════════════════════════════════════════════════
# D. 분석 유틸리티
# ══════════════════════════════════════════════════════

def compute_detection_delay_summary(detection_log: list[dict]) -> dict:
    """감지 지연 요약 통계."""
    if not detection_log:
        return {"count": 0}

    delays = [d["detection_delay"] for d in detection_log]
    channels = {}
    for d in detection_log:
        ch = d["channel"]
        channels[ch] = channels.get(ch, 0) + 1

    return {
        "count": len(delays),
        "mean_delay_days": round(sum(delays) / len(delays), 1),
        "max_delay_days": max(delays),
        "zero_delay_count": sum(1 for d in delays if d == 0),
        "by_channel": channels,
    }