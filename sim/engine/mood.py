"""mood.py — 환자 심리/컨디션 7-Dimension 모델

매일 환자의 심리 상태를 7차원 벡터로 표현한다.
이 상태가 자기 보고 확률, 보고 정확도, Care AI 대화 품질에 영향을 미친다.

차원:
  ① anxiety          불안 → 과보고 방향
  ② depression       우울 → 무응답 방향
  ③ irritability     짜증/귀찮음 → 조기종료 방향
  ④ energy           에너지 → 참여도
  ⑤ cognitive_clarity 인지 명료도 → 보고 정확도
  ⑥ trust_in_ai      AI 신뢰도 → 권고 이행률
  ⑦ defensiveness    방어적 경향 → 축소보고 방향

의학적 근거:
  - Basch et al. (2017): 환자 자가보고와 의사 기록 간 grade 불일치
  - Miller (1987): Monitoring vs Blunting 대처 유형
  - Greer et al.: 암 환자 대처 유형 (Fighting Spirit / Denial / Fatalism)
  - Lazarus & Folkman (1984): 스트레스 대처 이론
"""

from __future__ import annotations

import math
import random
from typing import Any

# ══════════════════════════════════════════════════════
# A. 7-Dimension 상태 구조
# ══════════════════════════════════════════════════════

MOOD_DIMENSIONS = (
    "anxiety",
    "depression",
    "irritability",
    "energy",
    "cognitive_clarity",
    "trust_in_ai",
    "defensiveness",
)

# 관성 계수: 새 값 = old * INERTIA + target * (1 - INERTIA)
MOOD_INERTIA = 0.75

# 지속 상태(ongoing) 이벤트의 감쇠 계수
# 첫 날(transition)은 full effect, 이후 매일은 × DAMPING
# 의학적 근거: 심리적 적응(habituation) — 같은 스트레서에 대한 반응이 시간에 따라 감소
ONGOING_EVENT_DAMPING = 0.15  # 지속 상태는 첫날의 15%만 영향

# 자연 감쇠 기본 비율 (baseline으로 회귀)
BASE_DECAY_RATE = 0.05  # 매일 5%

# 극단값 추가 감쇠 비율 (비선형 회귀)
# baseline에서 먼 값일수록 빠르게 회귀 (항상성)
# 의학적 근거: allostatic load — 만성 스트레스는 새 baseline을 형성하지만,
# 급성 스트레스 반응은 항상성에 의해 빠르게 감쇠
EXTREME_DECAY_BONUS = 0.15  # dist_from_baseline × 이 값이 추가 감쇠

# ══════════════════════════════════════════════════════
# B. 페르소나별 기본값 (baseline)
# ══════════════════════════════════════════════════════

PERSONA_MOOD_BASELINES: dict[str, dict[str, float]] = {
    "stoic_minimizer": {
        "anxiety": 0.15,
        "depression": 0.20,
        "irritability": 0.30,
        "energy": 0.50,
        "cognitive_clarity": 0.70,
        "trust_in_ai": 0.30,
        "defensiveness": 0.65,
    },
    "anxious_reporter": {
        "anxiety": 0.65,
        "depression": 0.30,
        "irritability": 0.15,
        "energy": 0.65,
        "cognitive_clarity": 0.75,
        "trust_in_ai": 0.55,
        "defensiveness": 0.15,
    },
    "shame_avoidant": {
        "anxiety": 0.45,
        "depression": 0.40,
        "irritability": 0.25,
        "energy": 0.45,
        "cognitive_clarity": 0.65,
        "trust_in_ai": 0.35,
        "defensiveness": 0.60,
    },
    "confused_elderly": {
        "anxiety": 0.40,
        "depression": 0.35,
        "irritability": 0.20,
        "energy": 0.30,
        "cognitive_clarity": 0.40,
        "trust_in_ai": 0.25,
        "defensiveness": 0.35,
    },
    "health_literate": {
        "anxiety": 0.35,
        "depression": 0.20,
        "irritability": 0.25,
        "energy": 0.55,
        "cognitive_clarity": 0.90,
        "trust_in_ai": 0.65,
        "defensiveness": 0.20,
    },
    "minimizer": {
        "anxiety": 0.15,
        "depression": 0.25,
        "irritability": 0.35,
        "energy": 0.50,
        "cognitive_clarity": 0.65,
        "trust_in_ai": 0.35,
        "defensiveness": 0.60,
    },
    "catastrophizer": {
        "anxiety": 0.75,
        "depression": 0.35,
        "irritability": 0.20,
        "energy": 0.55,
        "cognitive_clarity": 0.60,
        "trust_in_ai": 0.50,
        "defensiveness": 0.10,
    },
    "caregiver_dependent": {
        "anxiety": 0.50,
        "depression": 0.30,
        "irritability": 0.15,
        "energy": 0.35,
        "cognitive_clarity": 0.50,
        "trust_in_ai": 0.40,
        "defensiveness": 0.40,
    },
    "language_barrier": {
        "anxiety": 0.50,
        "depression": 0.30,
        "irritability": 0.30,
        "energy": 0.45,
        "cognitive_clarity": 0.35,
        "trust_in_ai": 0.30,
        "defensiveness": 0.45,
    },
    "compliant_but_forgetful": {
        "anxiety": 0.25,
        "depression": 0.25,
        "irritability": 0.15,
        "energy": 0.40,
        "cognitive_clarity": 0.45,
        "trust_in_ai": 0.55,
        "defensiveness": 0.20,
    },
}

# 매칭 안 되는 페르소나용 fallback
_DEFAULT_BASELINE = {
    "anxiety": 0.35,
    "depression": 0.25,
    "irritability": 0.25,
    "energy": 0.50,
    "cognitive_clarity": 0.65,
    "trust_in_ai": 0.40,
    "defensiveness": 0.35,
}

# ══════════════════════════════════════════════════════
# C. 페르소나 고정 속성 (시뮬레이션 내내 불변)
# ══════════════════════════════════════════════════════

PERSONA_FIXED_TRAITS: dict[str, dict[str, Any]] = {
    "stoic_minimizer":        {"health_literacy": 0.5, "social_support": 0.4, "digital_literacy": 0.5},
    "anxious_reporter":       {"health_literacy": 0.6, "social_support": 0.6, "digital_literacy": 0.7},
    "shame_avoidant":         {"health_literacy": 0.5, "social_support": 0.3, "digital_literacy": 0.5},
    "confused_elderly":       {"health_literacy": 0.3, "social_support": 0.5, "digital_literacy": 0.2},
    "health_literate":        {"health_literacy": 0.9, "social_support": 0.6, "digital_literacy": 0.8},
    "minimizer":              {"health_literacy": 0.5, "social_support": 0.5, "digital_literacy": 0.5},
    "catastrophizer":         {"health_literacy": 0.5, "social_support": 0.5, "digital_literacy": 0.6},
    "caregiver_dependent":    {"health_literacy": 0.4, "social_support": 0.8, "digital_literacy": 0.3},
    "language_barrier":       {"health_literacy": 0.2, "social_support": 0.4, "digital_literacy": 0.3},
    "compliant_but_forgetful": {"health_literacy": 0.5, "social_support": 0.5, "digital_literacy": 0.5},
}

_DEFAULT_TRAITS = {"health_literacy": 0.5, "social_support": 0.5, "digital_literacy": 0.5}


# ══════════════════════════════════════════════════════
# D. 이벤트 기반 Mood 변화 규칙
# ══════════════════════════════════════════════════════

# 일별 이벤트에 의한 mood delta
# 여러 이벤트가 있으면 additive, 최종 clamp(0, 1)
EVENT_MOOD_DELTAS: dict[str, dict[str, float]] = {
    # ── AE 관련 ──
    "new_ae_onset": {
        "anxiety": +0.12,
        "depression": +0.03,
        "energy": -0.05,
    },
    "ae_grade_worsened": {
        "anxiety": +0.10,
        "depression": +0.05,
        "energy": -0.08,
        "defensiveness": +0.05,  # 악화 인정하기 싫음
    },
    "ae_grade_improved": {
        "anxiety": -0.05,
        "depression": -0.05,
        "energy": +0.03,
        "defensiveness": -0.03,
    },
    "ae_resolved": {
        "anxiety": -0.08,
        "depression": -0.08,
        "energy": +0.05,
        "defensiveness": -0.05,
    },
    "grade3_or_higher": {
        # Grade 3+ AE가 하나라도 활성일 때 (매일 적용)
        "anxiety": +0.03,
        "depression": +0.02,
        "energy": -0.05,
        "defensiveness": -0.08,  # 심각하면 방어 무너짐
    },

    # ── 치료 관련 ──
    "infusion_day": {
        # 투약일 — 스테로이드 전처치 등으로 에너지 변동
        "anxiety": +0.05,
        "energy": -0.10,
        "irritability": +0.03,
        "cognitive_clarity": -0.05,
    },
    "post_infusion_day2_3": {
        # 투약 후 2-3일 — 스테로이드 crash
        "energy": -0.15,
        "cognitive_clarity": -0.08,
        "depression": +0.05,
        "irritability": +0.05,
    },
    "dose_reduction": {
        "anxiety": +0.10,
        "defensiveness": +0.12,  # "용량 줄이면 효과 줄까봐"
    },
    "treatment_held": {
        "anxiety": +0.15,
        "defensiveness": +0.10,
        "depression": +0.05,
    },
    "treatment_discontinued": {
        "anxiety": +0.12,
        "depression": +0.15,
        "defensiveness": -0.10,  # 이미 중단됐으니 숨길 이유 없음
    },

    # ── 스캔/결과 관련 ──
    "scan_approaching": {
        # RECIST 스캔 3일 전부터
        "anxiety": +0.15,
        "irritability": +0.05,
    },
    "scan_good_result": {
        # PR/CR
        "anxiety": -0.20,
        "depression": -0.15,
        "energy": +0.08,
        "trust_in_ai": +0.03,
    },
    "scan_bad_result": {
        # PD
        "anxiety": +0.15,
        "depression": +0.20,
        "energy": -0.10,
        "defensiveness": -0.10,
    },

    # ── 자연 감쇠 (매일 적용) ──
    "daily_decay": {
        # 극단값에서 중앙으로 회귀 (항상성)
        # → update_daily_mood에서 별도 처리
    },
}


# ══════════════════════════════════════════════════════
# E. MoodState 클래스
# ══════════════════════════════════════════════════════

class MoodState:
    """환자 1명의 심리/컨디션 상태를 관리한다.

    초기화 시 페르소나 기반으로 baseline을 설정하고,
    매일 이벤트에 따라 값이 변한다.
    """

    def __init__(self, persona_type: str, seed: int | None = None):
        self.persona_type = persona_type
        self.baseline = dict(
            PERSONA_MOOD_BASELINES.get(persona_type, _DEFAULT_BASELINE)
        )
        self.fixed_traits = dict(
            PERSONA_FIXED_TRAITS.get(persona_type, _DEFAULT_TRAITS)
        )
        self.state = dict(self.baseline)
        self._rng = random.Random(seed)

    def to_dict(self) -> dict[str, float]:
        """현재 mood state를 dict로 반환."""
        return dict(self.state)

    def get(self, dim: str) -> float:
        return self.state.get(dim, 0.5)

    # ── 매일 업데이트 ─────────────────────────────────

    def update_daily(self, events: list[str]) -> dict[str, float]:
        """하루의 이벤트 목록을 받아 mood를 업데이트한다.

        Args:
            events: EVENT_MOOD_DELTAS 키 목록
                    e.g. ["infusion_day", "new_ae_onset"]
                    "_ongoing:" prefix가 붙은 이벤트는 지속 상태 (magnitude 감소)

        Returns:
            변화량 delta dict
        """
        delta = {d: 0.0 for d in MOOD_DIMENSIONS}

        # (1) 이벤트 기반 delta 누적
        for event in events:
            # 지속 상태 이벤트: "_ongoing:" prefix → magnitude × ONGOING_DAMPING
            if event.startswith("_ongoing:"):
                base_event = event[len("_ongoing:"):]
                effects = EVENT_MOOD_DELTAS.get(base_event, {})
                for dim, val in effects.items():
                    if dim in delta:
                        delta[dim] += val * ONGOING_EVENT_DAMPING
            else:
                effects = EVENT_MOOD_DELTAS.get(event, {})
                for dim, val in effects.items():
                    if dim in delta:
                        delta[dim] += val

        # (2) 적응적 감쇠: baseline으로 회귀 (OU process)
        #     극단값일수록 빠르게 회귀 (비선형 감쇠)
        for dim in MOOD_DIMENSIONS:
            diff = self.baseline[dim] - self.state[dim]
            dist_from_baseline = abs(diff)
            # 기본 5% + 극단값에서 추가 15% (총 최대 20%)
            adaptive_decay = BASE_DECAY_RATE + dist_from_baseline * EXTREME_DECAY_BONUS
            delta[dim] += diff * adaptive_decay

        # (3) 노이즈 (일상적 변동)
        for dim in MOOD_DIMENSIONS:
            delta[dim] += self._rng.gauss(0, 0.02)

        # (4) 관성 적용 + clamp
        for dim in MOOD_DIMENSIONS:
            raw_target = self.state[dim] + delta[dim]
            self.state[dim] = _clamp(
                self.state[dim] * MOOD_INERTIA + raw_target * (1 - MOOD_INERTIA),
                0.0, 1.0,
            )

        return delta

    # ── 턴 단위 업데이트 (Care AI 대화 중) ─────────────

    def update_turn(self, effects: dict[str, float]) -> None:
        """한 턴의 효과를 적용한다. 대화 중 실시간 변화.

        관성이 약간 낮음 (대화 중에는 감정 변화가 더 빠름).
        """
        turn_inertia = 0.60  # 대화 중에는 관성 약함
        for dim, val in effects.items():
            if dim in self.state:
                raw = self.state[dim] + val
                self.state[dim] = _clamp(
                    self.state[dim] * turn_inertia + raw * (1 - turn_inertia),
                    0.0, 1.0,
                )

    # ── 방어 무너짐: 심각한 AE ──────────────────────

    def apply_defensiveness_override(self, max_ae_grade: int) -> None:
        """Grade 3+ AE는 방어벽을 약화시킨다."""
        if max_ae_grade >= 4:
            self.state["defensiveness"] = max(self.state["defensiveness"] - 0.30, 0.05)
        elif max_ae_grade >= 3:
            self.state["defensiveness"] = max(self.state["defensiveness"] - 0.15, 0.10)


# ══════════════════════════════════════════════════════
# F. Mood → Action 변환 함수
# ══════════════════════════════════════════════════════

def compute_self_report_probability(mood: MoodState, max_ae_grade: int,
                                    days_since_visit: int) -> float:
    """환자가 오늘 병원에 자발적으로 전화할 확률.

    Returns:
        0.0 ~ 1.0
    """
    s = mood.state

    # AE severity factor
    grade_factor = {0: 0.0, 1: 0.02, 2: 0.08, 3: 0.40, 4: 0.85}.get(
        max_ae_grade, 0.0
    )

    # 성격 factor: 불안이 높으면 전화, 방어/우울이 높으면 안 함
    personality_factor = (
        s["anxiety"] * 0.4
        - s["defensiveness"] * 0.25
        - s["depression"] * 0.15
        + (1.0 - s["irritability"]) * 0.1
    )

    # 시간 factor: 내원 직후엔 안 전화, 오래되면 불안 증가
    time_factor = 1.0
    if days_since_visit > 7:
        time_factor = min(1.0 + (days_since_visit - 7) * 0.03, 1.5)

    prob = (grade_factor + max(personality_factor, 0)) * time_factor
    return _clamp(prob, 0.0, 0.95)


def compute_grade_distortion(mood: MoodState) -> int:
    """환자가 AE grade를 몇 단계 왜곡해서 보고하는가.

    Returns:
        -2 ~ +2 (양수: 과보고, 음수: 축소보고)
    """
    s = mood.state

    # 과보고 점수
    over_score = s["anxiety"] * 0.6 + (1 - s["cognitive_clarity"]) * 0.2

    # 축소보고 점수
    under_score = (
        s["defensiveness"] * 0.5
        + s["depression"] * 0.2
        + (1 - s["energy"]) * 0.1
    )

    net = over_score - under_score

    if net > 0.35:
        return +1   # 1단계 과보고
    elif net > 0.60:
        return +2   # 2단계 과보고 (rare)
    elif net < -0.30:
        return -1   # 1단계 축소보고
    elif net < -0.55:
        return -2   # 2단계 축소보고 (rare)
    return 0         # 정확한 보고


def compute_interaction_quality(mood: MoodState) -> dict[str, float]:
    """Care AI 화상통화의 감지 정확도를 결정하는 상호작용 품질 계수.

    Returns:
        engagement:             대화 참여도 (0~1)
        report_accuracy:        보고 정확도 (0~1)
        over_report_prob:       과보고 확률 (0~0.5)
        under_report_prob:      과소보고 확률 (0~0.7)
        early_termination_prob: 통화 조기종료 확률 (0~0.6)
        compliance_rate:        권고 이행률 (0~1)
        video_cooperation:      시각 관찰 협조도 (0~1) — 피부 보여주기 등
    """
    s = mood.state

    # 대화 참여도: energy × (1 - irritability) × (1 - depression*0.5)
    engagement = (
        s["energy"] * (1 - s["irritability"] * 0.6) * (1 - s["depression"] * 0.4)
    )

    # 보고 정확도: 인지 명료도 기반, 불안이 적절(0.3)하면 최적
    anxiety_distortion = abs(s["anxiety"] - 0.3) * 0.4
    report_accuracy = s["cognitive_clarity"] * (1 - anxiety_distortion)

    # 과보고 확률
    over_report_prob = s["anxiety"] * 0.25 + (1 - s["cognitive_clarity"]) * 0.10

    # 과소보고 확률
    under_report_prob = (
        s["defensiveness"] * 0.30
        + s["depression"] * 0.20
        + s["irritability"] * 0.10
        + (1 - s["energy"]) * 0.10
    )

    # 통화 조기종료 확률
    early_termination_prob = (
        s["irritability"] * 0.35
        + (1 - s["energy"]) * 0.25
        + (1 - s["trust_in_ai"]) * 0.15
    )

    # 권고 이행률
    compliance_rate = (
        s["trust_in_ai"] * 0.45
        + s["energy"] * 0.25
        + (1 - s["depression"]) * 0.15
        + (1 - s["defensiveness"]) * 0.15
    )

    # 시각 관찰 협조도 (피부, 손, 입 보여주기 등)
    video_cooperation = (
        s["energy"] * 0.3
        + (1 - s["defensiveness"]) * 0.35
        + s["trust_in_ai"] * 0.20
        + (1 - s["irritability"]) * 0.15
    )

    return {
        "engagement": _clamp(engagement, 0.05, 1.0),
        "report_accuracy": _clamp(report_accuracy, 0.1, 1.0),
        "over_report_prob": _clamp(over_report_prob, 0.0, 0.50),
        "under_report_prob": _clamp(under_report_prob, 0.0, 0.70),
        "early_termination_prob": _clamp(early_termination_prob, 0.0, 0.60),
        "compliance_rate": _clamp(compliance_rate, 0.05, 1.0),
        "video_cooperation": _clamp(video_cooperation, 0.05, 1.0),
    }


def should_visit_er(max_ae_grade: int, vitals: dict, labs: dict) -> bool:
    """응급실 방문 필요 여부 (확정적 — mood 무관, 의학적 기준).

    Grade 4 AE, febrile neutropenia, SpO2 < 90% 등.
    """
    if max_ae_grade >= 4:
        return True
    temp = vitals.get("TEMP_VSORRES", vitals.get("body_temperature", 36.5))
    anc = 9999.0
    lab_results = labs.get("results", labs)
    if isinstance(lab_results, dict):
        anc_data = lab_results.get("absolute_neutrophil_count", {})
        if isinstance(anc_data, dict):
            anc = anc_data.get("LBORRES", anc_data.get("value", 9999.0))
        elif isinstance(anc_data, (int, float)):
            anc = anc_data
    # Febrile neutropenia: 열 38.3°C + ANC < 1000
    if temp >= 38.3 and anc < 1000:
        return True
    spo2 = vitals.get("_SpO2", vitals.get("SpO2", 100))
    if spo2 < 90:
        return True
    return False


# ── 유틸리티 ──────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))