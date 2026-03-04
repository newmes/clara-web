"""defaults.py — 약물/적응증에 무관한 시뮬레이션 엔진 상수

이 파일에 모든 하드코딩 상수를 모은다.
"이 시뮬레이션에서 하드코딩된 값이 뭐가 있지?" → 이 파일 하나만 보면 된다.

분류:
  A. AE grade 전이 — 역학 모델 기저 파라미터
  B. 종양 변화 — RECIST 범위 내 변화 속도
  C. Mortality — ECOG-사망률 매핑, 상한
  D. ECOG — 치료 중단 패널티
  E. OU process — vitals/labs mean-reversion
  F. CTCAE, 확률 상한, 보고 단위
  G. Causal Lab Fallback
  H. CM / 기저약물
  I. AE Onset / Drug Attribution / Intervention Effects
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════
# A. AE Grade 전이 확률 (역학 모델)
# ══════════════════════════════════════════════════════
GRADE_TRANSITION_BASE_WORSEN = 0.02       # 기본 악화 확률 (~2%/day)
GRADE_TRANSITION_BASE_IMPROVE = 0.005     # 기본 개선 확률 (~0.5%/day)
GRADE_TIME_STABILIZE_DAY = 21             # 비누적 AE — 이 날짜 이후 안정화
GRADE_HIGH_WORSEN_DAMPING = 0.7           # Grade ≥3 → 악화 확률 ×0.7 (감쇄)
GRADE_4_TO_5_DAMPING = 0.3               # Grade 4→5 추가 감쇄 (치명적 전환 억제)
GRADE_HIGH_IMPROVE_BOOST = 1.3            # Grade ≥3 → 개선 확률 ×1.3
GRADE_4_IMPROVE_BOOST = 1.5               # Grade 4 → 추가 개선 부스트
GRADE_CUMULATIVE_MAX_TIME_FACTOR = 2      # 누적 독성 시간 계수 상한

# ══════════════════════════════════════════════════════
# B. 종양 변화 — 시그모이드 모델 파라미터
# ══════════════════════════════════════════════════════
TUMOR_RATE = {
    "CR_plateau": -95,       # CR: 최종 -95%
    "CR_rate": 6,            # (레거시)
    "CR_lag_weeks": 4.5,     # 시그모이드 time constant (주)
    "PR_plateau": -55,       # PR: 최종 -55%
    "PR_rate": 4,
    "PR_lag_weeks": 5,
    "SD_plateau": -5,        # SD: 미세 변화
    "SD_amplitude": 4,       # SD: 오실레이션 진폭
    "SD_lag_weeks": 4,
    "PD_plateau": 80,        # (레거시)
    "PD_rate": 3.5,          # PD: 주당 증가율
    "PD_lag_weeks": 2,       # PD: lag
    "PD_max": 200,           # PD: 최대 증가 %
    "patient_scale_mean": 0, # lognormal patient scale
    "patient_scale_std": 0.3,
}

# Dose hold → 종양 반응에 미치는 effective treatment rate
TUMOR_DAILY_RATE_FULL = 1.0           # 모든 약물 투여 중 → 100% 효과
TUMOR_DAILY_RATE_PARTIAL_HOLD = 0.5   # 일부 약물 hold → 50% (나머지 약 효과)
TUMOR_DAILY_RATE_ALL_HELD = 0.0       # 모든 약물 hold → 0% (시계 정지)
TUMOR_DAILY_RATE_DISCONTINUED = 0.0   # 영구 중단 → 0% (별도 PD 메커니즘)

# ══════════════════════════════════════════════════════
# C. Mortality — ECOG-사망률 매핑
# ══════════════════════════════════════════════════════
ECOG_MORTALITY_MAP = {0: 1, 1: 1, 2: 1.5, 3: 2.5, 4: 5}
TREATMENT_DISCONTINUED_MORTALITY_MULT = 1.5
MAX_DAILY_MORTALITY = 0.5  # 일별 사망률 상한

# ══════════════════════════════════════════════════════
# D. ECOG — 동적 변화 파라미터
# ══════════════════════════════════════════════════════
TREATMENT_DISCONTINUED_ECOG_PENALTY = 0.2
ECOG_AE_PENALTY_CAP = 2                 # AE burden → ECOG 악화 상한
ECOG_MAX_DAILY_CHANGE = 1               # 일별 ECOG 변화 상한 (±1)
MIN_AE_BURDEN_WEIGHT = 0.25             # AE burden weight 하한
ECOG_TREATMENT_FATIGUE_PER_CYCLE = 0.05 # 사이클당 피로도

# ══════════════════════════════════════════════════════
# E. OU Process — vitals/labs mean-reversion
# ══════════════════════════════════════════════════════
OU_THETA_VITALS = 0.15  # vitals mean-reversion 속도
OU_THETA_LABS = 0.10     # labs mean-reversion 속도

VITALS_NOISE = {
    "SBP": 3, "DBP": 2, "HR": 2, "BT": 0.1,
    "RR": 0.5, "SpO2": 0.3, "weight_kg": 0.1,
}

# Lab별 noise fraction (값의 N%를 noise std로 사용)
# 생리적 일간 변동성이 큰 lab (ANC, WBC, platelets)은 큰 값,
# 안정적인 lab (hemoglobin, creatinine, HbA1c)은 작은 값.
# 의학적 근거: ANC는 일간 변동 CV 15-25%, hemoglobin은 CV 2-3%
LABS_NOISE_FRACTION_MAP: dict[str, float] = {
    "ANC": 0.06,               # 호중구: 높은 일간 변동 (6%)
    "WBC": 0.05,               # 백혈구: 높은 일간 변동 (5%)
    "platelets": 0.04,         # 혈소판: 중간-높은 변동 (4%)
    "hemoglobin": 0.015,       # 혈색소: 낮은 변동 (1.5%)
    "glucose_fasting": 0.05,   # 공복혈당: 높은 변동 (5%) — 식이, 스트레스 영향
    "creatinine": 0.02,        # 크레아티닌: 낮은 변동 (2%)
    "ALT": 0.04,               # 간효소: 중간 변동 (4%)
    "AST": 0.04,               # 간효소: 중간 변동 (4%)
    "total_bilirubin": 0.03,   # 빌리루빈: 중간 변동 (3%)
    "potassium": 0.03,         # 전해질: 중간 변동 (3%)
    "sodium": 0.01,            # 나트륨: 낮은 변동 (1%)
    "HbA1c": 0.005,            # 당화혈색소: 매우 낮은 변동 (3개월 평균)
    "TSH": 0.02,               # 갑상선: 낮은 변동 (2%)
    "LDH": 0.03,               # 젖산탈수소효소: 중간 변동 (3%)
    "albumin": 0.015,          # 알부민: 낮은 변동 (1.5%)
    "uric_acid": 0.03,         # 요산: 중간 변동 (3%)
}
LABS_NOISE_FRACTION = 0.025  # 매핑에 없는 lab의 기본값 (기존 0.01 → 0.025)

# Slow markers: 수주~수개월 단위로만 변동하는 검사. 인과적 이유 없이 daily drift 없음.
SLOW_MARKER_LABS = frozenset({
    "HbA1c",       # 3개월 평균 혈당 반영
    "TSH",         # 갑상선 기능 — 수주 단위
    "LDH",         # 종양 부담 마커 — 수주 단위
    "albumin",     # 영양 상태 — 수주 단위
})

# 임상 보고 단위 정밀도 (계측기 해상도 기준)
# 0 = 정수, 1 = 소수1자리, 2 = 소수2자리
LAB_ROUNDING = {
    "hemoglobin": 1,       # g/dL — 소수1자리
    "ANC": 1,              # x10^9/L — 소수1자리
    "platelets": 0,        # x10^9/L — 정수
    "creatinine": 2,       # mg/dL — 소수2자리
    "eGFR": 0,             # mL/min — 정수
    "ALT": 0,              # U/L — 정수
    "AST": 0,              # U/L — 정수
    "total_bilirubin": 1,  # mg/dL — 소수1자리
    "glucose_fasting": 0,  # mg/dL — 정수
    "HbA1c": 2,            # % — 소수2자리 (daily delta 0.005 반영)
    "TSH": 2,              # mIU/L — 소수2자리
    "LDH": 0,              # U/L — 정수
    "albumin": 2,          # g/dL — 소수2자리 (daily delta 0.05 반영)
    "sodium": 0,           # mmol/L — 정수
    "potassium": 1,        # mmol/L — 소수1자리
}

VITAL_ROUNDING = {
    "SBP": 0, "DBP": 0, "HR": 0,   # mmHg, bpm — 정수
    "BT": 1,                         # °C — 소수1자리
    "RR": 0,                         # breaths/min — 정수
    "SpO2": 0,                       # % — 정수
    "weight_kg": 1,                  # kg — 소수1자리
}

# Lab 일일 최대 변화량 (생리적 한계)
MAX_DAILY_LAB_DELTA: dict[str, float] = {
    "hemoglobin": 0.5,      # g/dL per day
    "ANC": 0.5,             # ×10³/μL per day
    "glucose_fasting": 50,  # mg/dL per day
    "creatinine": 0.3,      # mg/dL per day
    "ALT": 30,              # U/L per day
    "AST": 30,              # U/L per day
    "TSH": 1.0,             # mIU/L per day
    "HbA1c": 0.005,         # % per day — RBC 수명 120일 기반, 84일간 최대 ~0.4%
    "LDH": 20,              # U/L per day
    "albumin": 0.05,        # g/dL per day
    "total_bilirubin": 0.3, # mg/dL per day
    "sodium": 2.0,          # mmol/L per day
    "potassium": 0.3,       # mmol/L per day
    "platelets": 15,        # ×10³/μL per day
}

# ══════════════════════════════════════════════════════
# F-1. CTCAE v5.0 최대 등급 (Grade 5 미만인 AE만 기재)
# ══════════════════════════════════════════════════════
# CTCAE v5.0에서 대부분의 AE는 Grade 1-5까지 존재하지만,
# 일부 AE는 최대 등급이 제한되어 있다.
# 여기 없는 AE의 기본 상한은 5로 간주한다.
# key는 소문자 정규화, 부분 매칭으로 사용.

CTCAE_MAX_GRADE: dict[str, int] = {
    # ── 피부/모발/외형 (Grade 2 상한) ──
    "alopecia": 2,
    "hair_color_changes": 2,
    "nail_discoloration": 2,
    "nail_ridging": 2,
    "skin_hyperpigmentation": 2,
    "skin_hypopigmentation": 2,
    "stretch_marks": 2,
    "telangiectasia": 2,
    "bruising": 2,
    "hirsutism": 2,
    # ── 피부/점막 (Grade 3 상한) ──
    "dry_skin": 3,
    "pruritus": 3,
    "nail_loss": 3,
    "nail_changes": 3,
    # ── 내분비/대사 (Grade 2 상한) ──
    "cushingoid": 2,
    "gynecomastia": 2,
    "hot_flashes": 2,
    "menstrual_irregularity": 2,
    "virilization": 2,
    "delayed_puberty": 2,
    # ── 위장 (Grade 2 상한) ──
    "flatulence": 2,
    "hiccups": 2,
    "abdominal_distension": 3,
    # ── 근골격 (Grade 3 상한) ──
    "arthralgia": 3,
    "myalgia": 3,
    # ── 신경 (Grade 3 상한) ──
    "dysgeusia": 3,
    "paresthesia": 3,
    # ── 안과 (Grade 3 상한) ──
    "dry_eye": 3,
    "tearing": 3,
    "blurred_vision": 3,
    # ── 기타 ──
    "insomnia": 3,
    "fatigue": 3,  # CTCAE fatigue는 G4 있으나 드묾. 실제 G3 상한 적용.
}


def ctcae_max_grade(ae_term: str) -> int:
    """CTCAE v5.0 기준 해당 AE의 최대 허용 등급 반환.

    정확한 매칭 우선, 부분 매칭 fallback.
    테이블에 없으면 기본 5 (사망 포함) 반환.
    """
    normalized = ae_term.lower().replace(" ", "_").replace("-", "_")
    if normalized in CTCAE_MAX_GRADE:
        return CTCAE_MAX_GRADE[normalized]
    for key, max_g in CTCAE_MAX_GRADE.items():
        if key in normalized or normalized in key:
            return max_g
    return 5


# ══════════════════════════════════════════════════════
# F-2. Lab-defined AE: lab값에서 CTCAE grade를 역산
# ══════════════════════════════════════════════════════
#
# 의학적 근거: CTCAE v5.0에서 아래 AE들은 lab 값으로 grade가 **정의**된다.
# mode:
#   "absolute" — 절대값 기준 (Hgb g/dL, ANC x10^9/L 등)
#   "uln_multiple" — ULN 배수 기준 (ALT ×ULN 등). default_uln 명시.
# direction:
#   "decrease" — grade 높을수록 값 감소 (anemia, neutropenia 등)
#   "increase" — grade 높을수록 값 증가 (hyperglycemia, ALT 등)

CTCAE_LAB_RANGES: dict[str, dict] = {
    # ── 혈액학 (절대값, 감소형) ──
    "anemia": {
        "lab": "hemoglobin",
        "mode": "absolute",
        "direction": "decrease",
        "grades": {
            1: {"min": 10.0, "max": 12.0},
            2: {"min": 8.0,  "max": 10.0},
            3: {"min": 6.5,  "max": 8.0},
            4: {"min": 4.0,  "max": 6.5},
        },
    },
    "neutropenia": {
        "lab": "ANC",
        "mode": "absolute",
        "direction": "decrease",
        "grades": {
            1: {"min": 1.5, "max": 2.0},
            2: {"min": 1.0, "max": 1.5},
            3: {"min": 0.5, "max": 1.0},
            4: {"min": 0.0, "max": 0.5},
        },
    },
    "thrombocytopenia": {
        "lab": "platelets",
        "mode": "absolute",
        "direction": "decrease",
        "grades": {
            1: {"min": 75.0,  "max": 150.0},
            2: {"min": 50.0,  "max": 75.0},
            3: {"min": 25.0,  "max": 50.0},
            4: {"min": 0.0,   "max": 25.0},
        },
    },
    # ── 대사/내분비 (절대값, 증가형) ──
    "hyperglycemia": {
        "lab": "glucose_fasting",
        "mode": "absolute",
        "direction": "increase",
        "grades": {
            1: {"min": 100, "max": 160},
            2: {"min": 160, "max": 250},
            3: {"min": 250, "max": 500},
            4: {"min": 500, "max": 800},
        },
    },
    # ── 간기능 (ULN 배수, 증가형) ──
    "hepatotoxicity": {
        "lab": "ALT",
        "mode": "uln_multiple",
        "direction": "increase",
        "default_uln": 40,
        "grades": {
            1: {"min": 1.0, "max": 3.0},
            2: {"min": 3.0, "max": 5.0},
            3: {"min": 5.0, "max": 20.0},
            4: {"min": 20.0, "max": 50.0},
        },
    },
    "hepatotoxicity__AST": {
        "lab": "AST",
        "mode": "uln_multiple",
        "direction": "increase",
        "default_uln": 35,
        "grades": {
            1: {"min": 1.0, "max": 3.0},
            2: {"min": 3.0, "max": 5.0},
            3: {"min": 5.0, "max": 20.0},
            4: {"min": 20.0, "max": 50.0},
        },
    },
    "hepatotoxicity__bilirubin": {
        "lab": "total_bilirubin",
        "mode": "uln_multiple",
        "direction": "increase",
        "default_uln": 1.2,
        "grades": {
            1: {"min": 1.0, "max": 1.5},
            2: {"min": 1.5, "max": 3.0},
            3: {"min": 3.0, "max": 10.0},
            4: {"min": 10.0, "max": 20.0},
        },
    },
    # ── 신기능 (ULN 배수, 증가형) ──
    "nephrotoxicity": {
        "lab": "creatinine",
        "mode": "uln_multiple",
        "direction": "increase",
        "default_uln": 1.3,
        "grades": {
            1: {"min": 1.0, "max": 1.5},
            2: {"min": 1.5, "max": 3.0},
            3: {"min": 3.0, "max": 6.0},
            4: {"min": 6.0, "max": 12.0},
        },
    },
    # ── 갑상선 (절대값) ──
    "hypothyroidism": {
        "lab": "TSH",
        "mode": "absolute",
        "direction": "increase",
        "grades": {
            1: {"min": 4.5,  "max": 10.0},
            2: {"min": 10.0, "max": 20.0},
            3: {"min": 20.0, "max": 50.0},
            4: {"min": 50.0, "max": 100.0},
        },
    },
    "hyperthyroidism": {
        "lab": "TSH",
        "mode": "absolute",
        "direction": "decrease",
        "grades": {
            1: {"min": 0.1,  "max": 0.4},
            2: {"min": 0.01, "max": 0.1},
            3: {"min": 0.0,  "max": 0.01},
        },
    },
}


def ctcae_lab_range(ae_term: str, grade: int) -> list[tuple[str, float, float]] | None:
    """CTCAE v5.0 기준 AE grade에 대응하는 lab 값 범위를 반환한다.

    Args:
        ae_term: AE 용어 (소문자 정규화됨)
        grade: AE grade (1-4)

    Returns:
        [(lab_name, min_val, max_val), ...] 또는 None (매핑 없음).
        mode="uln_multiple"인 경우 min/max는 이미 절대값으로 변환되어 반환.
    """
    normalized = ae_term.lower().replace(" ", "_").replace("-", "_")
    results = []

    for key, spec in CTCAE_LAB_RANGES.items():
        base_key = key.split("__")[0]
        if base_key != normalized and base_key not in normalized and normalized not in base_key:
            continue

        grade_ranges = spec.get("grades", {})
        if grade not in grade_ranges:
            continue

        r = grade_ranges[grade]
        lab_name = spec["lab"]
        mode = spec.get("mode", "absolute")

        if mode == "uln_multiple":
            uln = spec.get("default_uln", 1.0)
            min_val = r["min"] * uln
            max_val = r["max"] * uln
        else:
            min_val = r["min"]
            max_val = r["max"]

        results.append((lab_name, min_val, max_val))

    return results if results else None


# ══════════════════════════════════════════════════════
# F-3. 확률 상한 (수치 안정성)
# ══════════════════════════════════════════════════════
MAX_GRADE_TRANSITION_PROB = 0.40   # grade 전이 확률 상한 (worsen 또는 improve 각각)
MAX_AE_CASCADE_HAZARD = 0.80       # AE cascade로 증폭된 hazard 상한
MAX_DISCONTINUATION_PATIENT = 0.02 # 환자 동의 철회 일별 확률 상한
MAX_DISCONTINUATION_PHYSICIAN = 0.01  # 의사 결정 중도탈락 일별 확률 상한

# ══════════════════════════════════════════════════════
# G. Causal Lab Fallback (빈 ae_lab_links일 때 기본 매핑)
# ══════════════════════════════════════════════════════
# rule_set.lab_causality.ae_lab_links가 비어있을 때 사용하는 기본 매핑
# grade_effects: baseline에 대한 배수 (multiplier)

DEFAULT_AE_LAB_LINKS: list[dict] = [
    {"ae_term": "anemia", "lab": "hemoglobin",
     "grade_effects": {"1": 0.9, "2": 0.75, "3": 0.6, "4": 0.45}},
    {"ae_term": "neutropenia", "lab": "ANC",
     "grade_effects": {"1": 0.75, "2": 0.5, "3": 0.25, "4": 0.1}},
    {"ae_term": "thrombocytopenia", "lab": "platelets",
     "grade_effects": {"1": 0.75, "2": 0.5, "3": 0.25, "4": 0.1}},
    {"ae_term": "hyperglycemia", "lab": "glucose_fasting",
     "grade_effects": {"1": 1.3, "2": 1.8, "3": 2.5, "4": 4.0}},
    {"ae_term": "hepatotoxicity", "lab": "ALT",
     "grade_effects": {"1": 2.0, "2": 4.0, "3": 10.0, "4": 25.0}},
    {"ae_term": "hepatotoxicity", "lab": "AST",
     "grade_effects": {"1": 2.0, "2": 4.0, "3": 10.0, "4": 25.0}},
    {"ae_term": "hepatotoxicity", "lab": "total_bilirubin",
     "grade_effects": {"1": 1.3, "2": 2.0, "3": 5.0, "4": 12.0}},
    {"ae_term": "nephrotoxicity", "lab": "creatinine",
     "grade_effects": {"1": 1.3, "2": 2.0, "3": 4.0, "4": 8.0}},
    {"ae_term": "hypothyroidism", "lab": "TSH",
     "grade_effects": {"1": 2.0, "2": 4.0, "3": 8.0, "4": 15.0}},
    {"ae_term": "hyperthyroidism", "lab": "TSH",
     "grade_effects": {"1": 0.5, "2": 0.15, "3": 0.02}},
    {"ae_term": "diarrhea", "lab": "potassium",
     "grade_effects": {"2": 0.95, "3": 0.85, "4": 0.75}},
    {"ae_term": "febrile_neutropenia", "lab": "ANC",
     "grade_effects": {"3": 0.15, "4": 0.05}},
]

# ══════════════════════════════════════════════════════
# H-1a. CM → Lab 교정 효과 (보조약물이 AE-driven lab 이탈을 교정)
# ══════════════════════════════════════════════════════
DEFAULT_CM_LAB_EFFECTS: list[dict] = [
    {"indication": "hyperglycemia", "lab": "glucose_fasting", "correction_factor": 0.6},
    {"indication": "hyperglycemia", "lab": "HbA1c", "correction_factor": 0.4},
    {"indication": "hypothyroidism", "lab": "TSH", "correction_factor": 0.5},
    {"indication": "hyperthyroidism", "lab": "TSH", "correction_factor": 0.5},
    {"indication": "hepatotoxicity", "lab": "ALT", "correction_factor": 0.3},
    {"indication": "hepatotoxicity", "lab": "AST", "correction_factor": 0.3},
    {"indication": "neutropenia", "lab": "ANC", "correction_factor": 0.4},
    {"indication": "anemia", "lab": "hemoglobin", "correction_factor": 0.3},
]

# ══════════════════════════════════════════════════════
# H-1b. CM → Lab Side Effects (보조약물 자체의 부작용)
# ══════════════════════════════════════════════════════
# 기존 CM_LAB_EFFECTS는 CM이 AE-driven lab 이탈을 "교정"하는 모델이다.
# 이 테이블은 반대로 CM 자체가 lab 이상을 "유발"하는 부작용을 모델링한다.
#
# target_multiplier: baseline 대비 목표 배수 (>1 상승, <1 하강)
# onset_days: 부작용 발현까지 지연일
# dose_dependent: True면 고용량일수록 효과 증폭
# high_dose_threshold: 이 dose 이상이면 고용량 배수 적용 (mg 단위)
# high_dose_target_multiplier: 고용량일 때의 목표 배수
#
# 의학적 근거:
#   Prednisone ≥0.5mg/kg:
#     - 고혈당: 인슐린 저항 → 공복혈당 130-200+, 거의 100% 발생 (Day 1-2)
#     - ANC 상승: 호중구 demargination → ANC 8-15+ (Day 2-3)
#     - HbA1c: 장기 투여 시 서서히 상승 (수 주)
#     - 체중 증가: Na/수분 저류 → 2-5kg (Week 1-2)
#     - 혈압 상승: mineralocorticoid 효과 → SBP +10-20 (Week 1-2)

DEFAULT_CM_SIDE_EFFECTS: list[dict] = [
    # ── Glucocorticoids (고용량 스테로이드) ──
    {
        "drug_keywords": ["prednisone", "prednisolone", "dexamethasone",
                          "methylprednisolone", "hydrocortisone"],
        "effects": [
            {
                "lab": "glucose_fasting",
                "target_multiplier": 1.5,           # baseline × 1.5 (e.g., 90 → 135)
                "high_dose_target_multiplier": 2.2,  # 고용량: baseline × 2.2 (e.g., 90 → 198)
                "high_dose_threshold_mg": 40,        # ≥40mg/day → 고용량
                "onset_days": 1,
            },
            {
                "lab": "HbA1c",
                "target_multiplier": 1.05,
                "high_dose_target_multiplier": 1.15,
                "high_dose_threshold_mg": 40,
                "onset_days": 21,
            },
            {
                "lab": "ANC",
                "target_multiplier": 1.5,            # demargination (4.5 → 6.75)
                "high_dose_target_multiplier": 2.5,
                "high_dose_threshold_mg": 40,
                "onset_days": 2,
            },
        ],
        "vitals_effects": [
            {
                "vital": "weight_kg",
                "daily_delta": 0.05,
                "high_dose_daily_delta": 0.15,
                "high_dose_threshold_mg": 40,
                "onset_days": 3,
                "max_total_delta": 5.0,
            },
            {
                "vital": "SBP",
                "daily_delta": 0.3,
                "high_dose_daily_delta": 0.8,
                "high_dose_threshold_mg": 40,
                "onset_days": 7,
                "max_total_delta": 20.0,
            },
        ],
    },
]

# ══════════════════════════════════════════════════════
# H-2. 기저질환 기본 약물 매핑 (Patient Agent fallback)
# ══════════════════════════════════════════════════════
DEFAULT_COMORBIDITY_MEDICATIONS: dict[str, list[dict]] = {
    "hypertension": [
        {"name": "Amlodipine", "dose": "5mg", "route": "PO", "frequency": "QD"},
        {"name": "Lisinopril", "dose": "10mg", "route": "PO", "frequency": "QD"},
        {"name": "Losartan", "dose": "50mg", "route": "PO", "frequency": "QD"},
    ],
    "diabetes": [
        {"name": "Metformin", "dose": "500mg", "route": "PO", "frequency": "BID"},
        {"name": "Glipizide", "dose": "5mg", "route": "PO", "frequency": "QD"},
        {"name": "Sitagliptin", "dose": "100mg", "route": "PO", "frequency": "QD"},
    ],
    "cardiovascular_disease": [
        {"name": "Aspirin", "dose": "81mg", "route": "PO", "frequency": "QD"},
        {"name": "Atorvastatin", "dose": "20mg", "route": "PO", "frequency": "QD"},
        {"name": "Metoprolol", "dose": "25mg", "route": "PO", "frequency": "BID"},
    ],
    "chronic_kidney_disease": [
        {"name": "Sodium Bicarbonate", "dose": "650mg", "route": "PO", "frequency": "TID"},
    ],
    "copd": [
        {"name": "Tiotropium", "dose": "18mcg", "route": "INH", "frequency": "QD"},
        {"name": "Albuterol", "dose": "90mcg", "route": "INH", "frequency": "PRN"},
    ],
    "atrial_fibrillation": [
        {"name": "Apixaban", "dose": "5mg", "route": "PO", "frequency": "BID"},
        {"name": "Metoprolol", "dose": "25mg", "route": "PO", "frequency": "BID"},
    ],
    "hypothyroidism": [
        {"name": "Levothyroxine", "dose": "50mcg", "route": "PO", "frequency": "QD"},
    ],
    "gastroesophageal_reflux": [
        {"name": "Omeprazole", "dose": "20mg", "route": "PO", "frequency": "QD"},
    ],
    "osteoarthritis": [
        {"name": "Acetaminophen", "dose": "500mg", "route": "PO", "frequency": "PRN"},
    ],
    "depression": [
        {"name": "Sertraline", "dose": "50mg", "route": "PO", "frequency": "QD"},
    ],
    "peripheral_neuropathy": [
        {"name": "Gabapentin", "dose": "300mg", "route": "PO", "frequency": "TID"},
    ],
}

# Discontinuation: 기본 background rate (protocol_violation + other 통합)
DISCONTINUATION_BACKGROUND_DAILY_RATE = 0.00024  # ~0.024%/day

# ══════════════════════════════════════════════════════
# I-0. AE Onset Grade 제한 (Prodrome 모델)
# ══════════════════════════════════════════════════════
# 대부분의 AE는 점진적으로 진행한다 (G1→G2→G3).
# 하지만 일부 AE는 급성 발현이 가능하다 (anaphylaxis, cardiac arrest 등).
# ACUTE_ONSET_AES: 급성 발현 가능 AE → onset 시 G3+ 허용
# 여기 없는 AE: onset 시 최대 G2에서 시작, 이후 grade_transition으로 진행

ACUTE_ONSET_AES: frozenset[str] = frozenset({
    "anaphylaxis",
    "infusion_related_reaction",
    "cardiac_arrest",
    "myocardial_infarction",
    "pulmonary_embolism",
    "cerebrovascular_accident",
    "stroke",
    "seizure",
    "tumor_lysis_syndrome",
    "stevens_johnson_syndrome",
    "toxic_epidermal_necrolysis",
    "disseminated_intravascular_coagulation",
    "sepsis",
    "septic_shock",
    "hemorrhage",
    "perforation",
    "bowel_perforation",
})

MAX_ONSET_GRADE_GRADUAL = 1  # 점진적 AE는 Grade 1로 시작 (Grade skip 방지)

# ══════════════════════════════════════════════════════
# I-1. IO/ADC Drug-AE Attribution
# ══════════════════════════════════════════════════════
IO_SPECIFIC_AES: frozenset[str] = frozenset({
    "pneumonitis",
    "colitis",
    "hepatitis",
    "thyroiditis",
    "hypothyroidism",
    "hyperthyroidism",
    "hyperglycemia",
    "myocarditis",
    "nephritis",
    "adrenal_insufficiency",
    "hypophysitis",
    "type_1_diabetes",
    "myositis",
    "encephalitis",
    "uveitis",
    "guillain_barre",
    "immune_thrombocytopenia",
    "autoimmune",
})

ADC_CHEMO_SPECIFIC_AES: frozenset[str] = frozenset({
    "peripheral_neuropathy",
    "neuropathy",
    "alopecia",
    "stomatitis",
    "mucositis",
    "myelosuppression",
    "neutropenia",
    "thrombocytopenia",
    "anemia",
    "febrile_neutropenia",
    "palmar_plantar",
    "nail_changes",
    "nail_loss",
    "skin_eruption",
})

# IO 약물 식별 키워드 (administration_schedule.drug_name 매칭)
IO_DRUG_KEYWORDS: frozenset[str] = frozenset({
    "pembrolizumab", "keytruda",
    "nivolumab", "opdivo",
    "atezolizumab", "tecentriq",
    "durvalumab", "imfinzi",
    "avelumab", "bavencio",
    "ipilimumab", "yervoy",
    "tremelimumab",
    "cemiplimab", "libtayo",
    "dostarlimab", "jemperli",
    "retifanlimab",
    "toripalimab",
    "tislelizumab",
})

# ══════════════════════════════════════════════════════
# I-2. Zero-AE Prevention (누적 hazard 부스트)
# ══════════════════════════════════════════════════════
ZERO_AE_BOOST_START_DAY = 21    # 1 cycle(21일) 이후에도 AE 없으면 부스트 시작
ZERO_AE_BOOST_PER_DAY = 0.008   # 하루당 +0.8%p 추가 (50일 후 ~23%p 부스트)
ZERO_AE_BOOST_MAX = 0.20        # 최대 20%p (원래 hazard에 가산)

# ══════════════════════════════════════════════════════
# I-3. Care AI / Natural 개입 효과 (Hazard Feedback)
# ══════════════════════════════════════════════════════
# Tier 1 — 약물 용량 조정 (dose hold, dose reduction)
#   가장 강력한 효과. 약물 노출 자체를 줄이므로 모든 확률에 영향.
#   근거: EV-302 dose modification guidelines, NCCN guidelines
#
# Tier 2 — 표적 보조약 (기전 일치)
#   예: antiemetic for nausea, topical steroid for IO-rash
#   근거: MASCC/ESMO antiemetic guideline, NCCN immune-related AE guideline
#   worsen ×0.6, improve ×1.5, resolution ×1.2
#
# Tier 3 — 대증 치료 (증상 완화 중심)
#   예: gabapentin for neuropathy, mouthwash for stomatitis
#   근거: CIPN prophylaxis RCTs (mixed/negative), symptomatic relief
#   worsen ×0.85, improve ×1.1, resolution ×1.0
#
# Tier 4 — 비약물적 개입 (활동 조절 등)
#   근거: exercise oncology, non-pharmacological fatigue management
#   worsen ×0.95, improve ×1.05, resolution ×1.0

INTERVENTION_EFFECTS = {
    "dose_hold": {
        "worsen_mult": 0.30,
        "improve_mult": 2.50,
        "resolution_mult": 1.50,
        "onset_hazard_mult": 0.40,
        "tier": 1,
    },
    "dose_reduction": {
        "worsen_mult": 0.50,
        "improve_mult": 1.80,
        "resolution_mult": 1.30,
        "onset_hazard_mult": 0.70,
        "tier": 1,
    },
    "conmed_tier2": {
        "worsen_mult": 0.60,
        "improve_mult": 1.50,
        "resolution_mult": 1.20,
        "onset_hazard_mult": 1.00,
        "tier": 2,
    },
    "conmed_tier3": {
        "worsen_mult": 0.85,
        "improve_mult": 1.10,
        "resolution_mult": 1.00,
        "onset_hazard_mult": 1.00,
        "tier": 3,
    },
    "conmed_tier4": {
        "worsen_mult": 0.95,
        "improve_mult": 1.05,
        "resolution_mult": 1.00,
        "onset_hazard_mult": 1.00,
        "tier": 4,
    },
}

# AE별 보조약 Tier 매핑 (supportive_care_rules에서 처방된 약이 해당 AE에 미치는 효과 Tier)
# 여기 없는 AE는 기본 Tier 3 (대증 치료)
CONMED_AE_TIER: dict[str, int] = {
    # Tier 2: 표적 보조약 (기전 일치, 근거 중간 이상)
    "rash": 2,
    "rash_maculopapular": 2,
    "nausea": 2,
    "vomiting": 2,
    "diarrhea": 2,
    "hyperglycemia": 2,
    "hypothyroidism": 2,
    "hyperthyroidism": 2,
    "anemia": 2,
    "colitis": 2,
    "hepatitis": 2,
    "pneumonitis": 2,
    # Tier 3: 대증 치료 (증상 완화, 진행 억제 효과 제한적)
    "peripheral_neuropathy": 3,
    "stomatitis": 3,
    "mucositis": 3,
    "fatigue": 4,
    "decreased_appetite": 3,
    "alopecia": 4,
    "dry_eye": 3,
    "pruritus": 3,
    "arthralgia": 3,
    "myalgia": 3,
}


# ══════════════════════════════════════════════════════
# J. FDA-Accurate Dose Modification Rules
# ══════════════════════════════════════════════════════

# J-1. Padcev 감량 단계 (FDA PI Section 2.3)
# 시작 용량 1.25 mg/kg → 1단계 1.0 → 2단계 0.75 → 3단계 0.5
# 코드에서는 시작을 1.0으로 정규화: [1.0, 0.8, 0.6, 0.4]
PADCEV_DOSE_REDUCTION_LEVELS: list[float] = [1.0, 0.8, 0.6, 0.4]

# J-2. AE별 FDA 정확 Dose Modification 오버라이드
# LLM 생성 규칙보다 우선 적용. 실제 처방정보(PI) 기반.
FDA_DOSE_MOD_OVERRIDES: dict[str, dict] = {
    # ── Padcev (ADC) AEs ──
    "peripheral_neuropathy": {
        "grade_actions": {
            "1": "DOSE NOT CHANGED",
            "2": "DRUG INTERRUPTED",   # Hold until ≤G1 (감량 아닌 보류)
            "3": "DRUG WITHDRAWN",     # 영구 중단 (FDA PI)
            "4": "DRUG WITHDRAWN",
            "5": "DRUG WITHDRAWN",
        },
    },
    "neuropathy": {
        "grade_actions": {
            "1": "DOSE NOT CHANGED",
            "2": "DRUG INTERRUPTED",
            "3": "DRUG WITHDRAWN",
            "4": "DRUG WITHDRAWN",
            "5": "DRUG WITHDRAWN",
        },
    },
    "rash_maculopapular": {
        "grade_actions": {
            "1": "DOSE NOT CHANGED",
            "2": "DOSE NOT CHANGED",   # 모니터링; 지속/재발 시 hold (recurrence 로직에서 처리)
            "3": "DRUG INTERRUPTED",   # Hold until ≤G1, 감량 후 재개
            "4": "DRUG WITHDRAWN",
        },
    },
    "skin_reaction": {
        "grade_actions": {
            "1": "DOSE NOT CHANGED",
            "2": "DOSE NOT CHANGED",
            "3": "DRUG INTERRUPTED",
            "4": "DRUG WITHDRAWN",
        },
    },
    # ── IO (Pembrolizumab) irAEs ──
    "pneumonitis": {
        "grade_actions": {
            "1": "DOSE NOT CHANGED",
            "2": "DRUG INTERRUPTED",   # Hold + 스테로이드 시작
            "3": "DRUG WITHDRAWN",     # 영구 중단
            "4": "DRUG WITHDRAWN",
        },
    },
    "colitis": {
        "grade_actions": {
            "1": "DOSE NOT CHANGED",
            "2": "DRUG INTERRUPTED",   # Hold + 스테로이드
            "3": "DRUG INTERRUPTED",   # Hold + 고용량 스테로이드
            "4": "DRUG WITHDRAWN",     # 영구 중단
        },
    },
    "hepatitis": {
        "grade_actions": {
            "1": "DOSE NOT CHANGED",
            "2": "DRUG INTERRUPTED",
            "3": "DRUG WITHDRAWN",
            "4": "DRUG WITHDRAWN",
        },
    },
    "myocarditis": {
        "grade_actions": {
            "1": "DOSE NOT CHANGED",
            "2": "DRUG WITHDRAWN",     # ≥G2 → 영구 중단 (FDA PI)
            "3": "DRUG WITHDRAWN",
            "4": "DRUG WITHDRAWN",
        },
    },
    "nephritis": {
        "grade_actions": {
            "1": "DOSE NOT CHANGED",
            "2": "DRUG INTERRUPTED",
            "3": "DRUG WITHDRAWN",
            "4": "DRUG WITHDRAWN",
        },
    },
    "encephalitis": {
        "grade_actions": {
            "1": "DOSE NOT CHANGED",
            "2": "DRUG WITHDRAWN",     # ≥G2 → 영구 중단 (FDA PI)
            "3": "DRUG WITHDRAWN",
            "4": "DRUG WITHDRAWN",
        },
    },
}

# J-3. 영구 중단 규칙 (Permanent Discontinuation Rules)
# 특정 AE가 특정 grade에 도달하면 해당 약물을 영구 중단.
# grade_actions 오버라이드와 중복되지만, 안전망으로 독립 체크.
#
# ae_pattern: AE 용어 부분 매칭
# grade_threshold: 이 grade 이상에서 영구 중단
# drug_scope: "non_io"=비IO약물, "io"=IO약물, "all"=전체
# recurrence_required: True면 재발 시에만 영구 중단 (첫 발생 시 hold)
PERMANENT_DC_RULES: list[dict] = [
    # ── Padcev (ADC/chemo) ──
    {"ae_pattern": "peripheral_neuropathy", "grade_threshold": 3, "drug_scope": "non_io",
     "recurrence_required": False, "note": "Padcev PI: ≥G3 PN → permanent d/c"},
    {"ae_pattern": "neuropathy", "grade_threshold": 3, "drug_scope": "non_io",
     "recurrence_required": False, "note": "Padcev PI: ≥G3 neuropathy → permanent d/c"},

    # ── IO (Pembrolizumab) — 즉시 영구 중단 ──
    {"ae_pattern": "pneumonitis", "grade_threshold": 3, "drug_scope": "io",
     "recurrence_required": False, "note": "Keytruda PI: G3+ pneumonitis → permanent d/c"},
    {"ae_pattern": "hepatitis", "grade_threshold": 3, "drug_scope": "io",
     "recurrence_required": False, "note": "Keytruda PI: G3+ hepatitis → permanent d/c"},
    {"ae_pattern": "nephritis", "grade_threshold": 3, "drug_scope": "io",
     "recurrence_required": False, "note": "Keytruda PI: G3+ nephritis → permanent d/c"},
    {"ae_pattern": "colitis", "grade_threshold": 4, "drug_scope": "io",
     "recurrence_required": False, "note": "Keytruda PI: G4 colitis → permanent d/c"},
    {"ae_pattern": "myocarditis", "grade_threshold": 2, "drug_scope": "io",
     "recurrence_required": False, "note": "Keytruda PI: ≥G2 myocarditis → permanent d/c"},
    {"ae_pattern": "encephalitis", "grade_threshold": 2, "drug_scope": "io",
     "recurrence_required": False, "note": "Keytruda PI: ≥G2 neurologic → permanent d/c"},

    # ── IO — 재발 시 영구 중단 ──
    {"ae_pattern": "pneumonitis", "grade_threshold": 2, "drug_scope": "io",
     "recurrence_required": True, "note": "Keytruda PI: recurrent G2 pneumonitis → permanent d/c"},
    {"ae_pattern": "colitis", "grade_threshold": 3, "drug_scope": "io",
     "recurrence_required": True, "note": "Keytruda PI: recurrent G3 colitis → permanent d/c"},

    # ── Universal (모든 약물 즉시 중단) ──
    {"ae_pattern": "stevens_johnson_syndrome", "grade_threshold": 1, "drug_scope": "all",
     "recurrence_required": False, "note": "SJS: immediate d/c all drugs"},
    {"ae_pattern": "toxic_epidermal_necrolysis", "grade_threshold": 1, "drug_scope": "all",
     "recurrence_required": False, "note": "TEN: immediate d/c all drugs"},
]

# J-4. IO 최대 투여 횟수 (Pembrolizumab ~2년 at Q3W ≈ 35 cycles)
IO_MAX_CYCLES: int = 35

# J-5. AE 재발 hazard 배수 (해소 후 재발 시 원래 hazard의 N%)
AE_RECURRENCE_HAZARD_MULT: float = 0.5


# ─── Normalize helpers (used by experiments) ─────────────────

_CANONICAL_LAB_NAMES: dict[str, str] = {
    "anc": "ANC", "wbc": "WBC", "alt": "ALT", "ast": "AST",
    "tsh": "TSH", "hba1c": "HbA1c", "ldh": "LDH", "egfr": "eGFR",
    "alanine_aminotransferase": "ALT", "alanine aminotransferase": "ALT",
    "aspartate_aminotransferase": "AST", "aspartate aminotransferase": "AST",
    "sgpt": "ALT", "sgot": "AST",
    "bilirubin": "total_bilirubin", "total bilirubin": "total_bilirubin",
    "tbil": "total_bilirubin", "direct_bilirubin": "total_bilirubin",
    "glucose": "glucose_fasting", "fasting_glucose": "glucose_fasting",
    "fasting glucose": "glucose_fasting", "blood_glucose": "glucose_fasting",
    "hgb": "hemoglobin", "hb": "hemoglobin", "haemoglobin": "hemoglobin",
    "hemoglobin a1c": "HbA1c", "hemoglobin_a1c": "HbA1c",
    "glycated_hemoglobin": "HbA1c", "glycated hemoglobin": "HbA1c", "a1c": "HbA1c",
    "absolute_neutrophil_count": "ANC", "white_blood_cell": "WBC",
    "white blood cell": "WBC", "plt": "platelets", "platelet": "platelets",
    "platelet_count": "platelets", "creat": "creatinine",
    "na": "sodium", "k": "potassium", "uric acid": "uric_acid",
}


def normalize_lab_key(key: str) -> str:
    low = key.strip().lower()
    return _CANONICAL_LAB_NAMES.get(low, key)


def normalize_ae_term(term: str) -> str:
    return term.strip().lower().replace(" ", "_").replace("-", "_")
