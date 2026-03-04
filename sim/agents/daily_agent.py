"""Daily Agent — fate table 없이 매일의 이벤트를 동적으로 결정하는 시뮬레이션 에이전트

핵심 차이점:
  - Fate table 없음: 어떤 AE가 언제 발생할지 미리 정하지 않는다.
  - Hazard function: rule_set의 onset 분포에서 매일의 발생 확률을 코드로 계산한다.
  - 환자 상태 반영: 현재 AE, care_record, 투약 상태에 따라 확률이 달라진다.
  - Care AI 친화적: 개입(care_record)이 실제로 이벤트를 "예방"할 수 있다.

흐름 (매일):
  1. 코드(hazard): 각 미발생 AE의 오늘 onset 확률 계산 → rand 샘플링
  2. 코드(hazard): 각 활성 AE의 grade 전이/해소 확률 계산 → rand 샘플링
  3. 코드: 종양/효능 변화 계산 (response category에 따라 deterministic)
  4. 이벤트 있으면 → LLM이 전체 상태 생성 [Event Day]
     이벤트 없으면 → 코드가 소폭 변동 적용 [Quiet Day]

초기화 시 1회:
  - LLM이 이 환자의 AE incidence를 조정 (환자 특성 반영)
  - 종양 반응 카테고리를 rule_set에서 rand 샘플링
"""

import json
import math
import re
from typing import Any

from sim.agents.llm_client import generate_json, set_caller, DEFAULT_MODEL
from sim.engine.sampler import Sampler
from sim.engine.prob_engine import estimate_probabilities, generate_details
from sim.engine.hazard import (
    daily_onset_hazard, daily_resolution_hazard, grade_transition_probs,
    tumor_change_pct, adjust_incidence_by_risk_modifiers,
    compute_daily_mortality, compute_dynamic_ecog, compute_causal_lab_target,
    compute_ae_cascade_multipliers, compute_discontinuation_risk,
)
from sim.config.defaults import (
    OU_THETA_VITALS, OU_THETA_LABS,
    VITALS_NOISE, LABS_NOISE_FRACTION, LABS_NOISE_FRACTION_MAP,
    SLOW_MARKER_LABS, LAB_ROUNDING, VITAL_ROUNDING,
    MAX_AE_CASCADE_HAZARD,
    DEFAULT_AE_LAB_LINKS, DEFAULT_CM_LAB_EFFECTS, DEFAULT_CM_SIDE_EFFECTS,
    ctcae_max_grade, ctcae_lab_range,
    ACUTE_ONSET_AES, MAX_ONSET_GRADE_GRADUAL,
    IO_SPECIFIC_AES, ADC_CHEMO_SPECIFIC_AES, IO_DRUG_KEYWORDS,
    MAX_DAILY_LAB_DELTA,
    ZERO_AE_BOOST_START_DAY, ZERO_AE_BOOST_PER_DAY, ZERO_AE_BOOST_MAX,
    CONMED_AE_TIER,
    TUMOR_DAILY_RATE_FULL, TUMOR_DAILY_RATE_PARTIAL_HOLD,
    TUMOR_DAILY_RATE_ALL_HELD, TUMOR_DAILY_RATE_DISCONTINUED,
    PADCEV_DOSE_REDUCTION_LEVELS, FDA_DOSE_MOD_OVERRIDES,
    PERMANENT_DC_RULES, IO_MAX_CYCLES, AE_RECURRENCE_HAZARD_MULT,
)
from sim.context_manager import compress_history
from sim.logger import get_logger, log_event, log_hazard

_logger = get_logger("daily_agent")


# ═══════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════

def create_simulator(rule_set, patient, sampler, model=DEFAULT_MODEL, actual_duration=None):
    """DailySimulator 인스턴스를 생성하는 팩토리 함수."""
    return DailySimulator(rule_set, patient, sampler, model, actual_duration)


# ═══════════════════════════════════════════════════════
# DailySimulator
# ═══════════════════════════════════════════════════════

class DailySimulator:
    """환자 1명의 일별 시뮬레이션을 수행하는 클래스.

    인스턴스 생성 시 1회 LLM 호출로 환자별 AE 위험도를 보정한다.
    이후 매일의 이벤트는 hazard function(코드) + rand로 결정한다.
    """

    def __init__(self, rule_set, patient, sampler: Sampler, model=DEFAULT_MODEL, actual_duration=None):
        self.rule_set = rule_set
        self.patient = patient
        self.sampler = sampler
        self.model = model
        self.actual_duration = actual_duration

        # ── EMR 파싱 ──
        emr = patient.get("emr")
        if not emr:
            raise ValueError(f"Patient {patient.get('patient_id', '?')}: 'emr' field is missing.")
        self.demographics = emr.get("demographics")
        if not self.demographics:
            raise ValueError(f"Patient {patient.get('patient_id', '?')}: 'emr.demographics' is missing.")
        self.conditions = {h.get("condition", "").lower() for h in (emr.get("medical_history") or [])}
        self.age = self.demographics.get("age")
        if self.age is None:
            raise ValueError(f"Patient {patient.get('patient_id', '?')}: 'demographics.age' is missing.")
        _raw_wt = (emr.get("baseline_vitals") or {}).get("weight_kg", 70)
        try:
            self.weight_kg = float(_raw_wt) if _raw_wt is not None else 70.0
        except (ValueError, TypeError):
            import re as _re
            _digits = _re.findall(r'[\d.]+', str(_raw_wt))
            self.weight_kg = float(_digits[0]) if _digits else 70.0

        # ── 상태 변수 초기화 ──
        self.occurred_aes: dict[str, dict] = {}
        self.resolved_aes: set[str] = set()
        self.cumulative_doses: dict[str, float] = {}
        self.dose_levels: dict[str, float] = {}
        self.last_admin_day: dict[str, int] = {}
        self.all_drug_names: set[str] = set()
        self.io_drugs: set[str] = set()
        self.non_io_drugs: set[str] = set()
        self.held_drugs: set[str] = set()
        self.discontinued_drugs: set[str] = set()
        self.hold_reasons: dict[str, str] = {}
        self.discontinuation_day: int | None = None
        self.hold_reason: str | None = None
        self.frozen_cycle: int | None = None
        self.frozen_cycle_day: int | None = None
        self.effective_treatment_days: int = 0
        self.dose_reduction_count: int = 0

        # ── Baseline CM (기저 약물) ──
        self.active_cm: list[dict] = []
        for history in (emr.get("medical_history") or []):
            med = history.get("medication")
            if med and history.get("ongoing", True):
                if isinstance(med, dict):
                    cm_record = {
                        "CMTRT": med.get("name", ""),
                        "CMINDC": history.get("condition", "comorbidity"),
                        "CMDOSE": med.get("dose", ""),
                        "CMROUTE": med.get("route", "PO"),
                        "CMDOSFRQ": med.get("frequency", "QD"),
                        "CMSTDAT": 0,
                        "CMENDAT": None,
                        "CMONGO": True,
                        "_baseline": True,
                    }
                elif isinstance(med, str) and med:
                    cm_record = {
                        "CMTRT": med,
                        "CMINDC": history.get("condition", "comorbidity"),
                        "CMDOSE": "",
                        "CMROUTE": "PO",
                        "CMDOSFRQ": "QD",
                        "CMSTDAT": 0,
                        "CMENDAT": None,
                        "CMONGO": True,
                        "_baseline": True,
                    }
                else:
                    continue
                self.active_cm.append(cm_record)

        # ── AE profile 검증 ──
        if "ae_profile" not in rule_set or not rule_set["ae_profile"]:
            raise ValueError("rule_set.ae_profile is missing or empty.")
        self._known_ae_terms = {ae["ae_term"] for ae in rule_set["ae_profile"]}

        # ── ECOG ──
        _raw_ecog = self.demographics.get("ecog_ps", patient.get("emr", {}).get("baseline_ecog"))
        if _raw_ecog is None:
            raise ValueError(
                f"Patient {patient.get('patient_id', '?')}: baseline ECOG not found."
            )
        self.baseline_ecog = int(_raw_ecog)
        self.current_ecog = self.baseline_ecog

        # ── 생존 상태 ──
        self.is_deceased = False
        self.death_day: int | None = None
        self.death_cause: str | None = None
        self.ds_record: dict | None = None

        # ── Baseline labs/vitals (pre-treatment, from EMR) ──
        self.baseline_labs: dict[str, float] = {}
        for _lab_name, _lab_data in (emr.get("baseline_labs") or {}).items():
            if isinstance(_lab_data, dict):
                _v = _lab_data.get("value")
            else:
                _v = _lab_data
            _parsed = DailySimulator._extract_number(_v)
            if _parsed is not None:
                self.baseline_labs[_lab_name] = _parsed
        self.baseline_vitals: dict[str, float] = DailySimulator._normalize_bt(
            dict(emr.get("baseline_vitals") or {})
        )

        # ── AE cascade ──
        self.ae_cascade_multipliers: dict[str, float] = {}

        # ── AE resolution tracking ──
        self._today_resolved: list[dict] = []  # 당일 해소된 AE (CRF에 포함용)

        # ── AEACN attribution: 어떤 AE가 어떤 약물 조치를 야기했는지 추적 ──
        self.ae_dose_actions: dict[str, str] = {}  # ae_term → action ("DRUG INTERRUPTED" 등)

        # ── AE 재발 추적 (rechallenge 시 에스컬레이션 판단용) ──
        self.ae_recurrence_count: dict[str, int] = {}  # ae_term → 재발 횟수

        # ── IO 약물 투여 횟수 (35-cycle 제한용) ──
        self.io_admin_count: dict[str, int] = {}  # drug_name → 누적 투여 횟수

        # ── Composite models (required) ──
        _required_model_keys = ["mortality_model", "ecog_model", "ae_cascade_rules", "disposition_model"]
        _missing_models = [k for k in _required_model_keys if k not in rule_set]
        if _missing_models:
            raise ValueError(
                f"rule_set is missing required composite model keys: {_missing_models}. "
                "Ensure Rule Agent completed composite model supplement."
            )
        self.mortality_config = rule_set["mortality_model"]
        self.ecog_config = rule_set["ecog_model"]
        self.lab_causality_config = rule_set.get("lab_causality") or {}
        self.ae_cascade_rules = rule_set["ae_cascade_rules"]
        _disp_raw = rule_set["disposition_model"]
        if "independent_hazards" in _disp_raw:
            self.disposition_config = _disp_raw["independent_hazards"]
        else:
            self.disposition_config = _disp_raw

        # ── 스케줄 파싱 ──
        self.admin_schedule = self._parse_administration_schedule()
        self.dose_mod_rules = self._parse_dose_modification_rules()
        self.support_care_map = self._parse_supportive_care_rules()

        # ── LLM 초기화: AE 위험도 보정 + 종양 반응 샘플링 ──
        set_caller("daily_agent.init")
        _logger.info(f"[DailySimulator] Initializing for {patient.get('patient_id', '?')}")
        self.ae_risks = self._calibrate_ae_risks()
        self.tumor_response, self.response_onset_day, self.patient_scale = self._sample_tumor_response()

        # ── 약물 분류 ──
        for drug in self.admin_schedule:
            dname = drug["drug_name"]
            self.dose_levels[dname] = 1.0
            self.cumulative_doses[dname] = 0.0
            self.all_drug_names.add(dname)
            if any(kw in dname.lower() for kw in IO_DRUG_KEYWORDS):
                self.io_drugs.add(dname)
            else:
                self.non_io_drugs.add(dname)
        _logger.info(f"  Drug classification: IO={self.io_drugs}, non-IO={self.non_io_drugs}")

        # ── RECIST 스캔 스케줄 ──
        cycle_len = (rule_set.get("trial_design") or {}).get("cycle_length_days", 21)
        first_scan_day = cycle_len * 2 + 7
        scan_interval = cycle_len * 2
        self.recist_scan_days: list[int] = []
        scan_day = first_scan_day
        effective_days = self.actual_duration or (rule_set.get("trial_design") or {}).get("planned_duration_days", 180)
        while scan_day <= effective_days:
            self.recist_scan_days.append(scan_day)
            scan_day += scan_interval
        eos_scan = effective_days - 7
        if eos_scan > first_scan_day and eos_scan not in self.recist_scan_days:
            too_close = any(abs(eos_scan - d) < 14 for d in self.recist_scan_days)
            if not too_close:
                self.recist_scan_days.append(eos_scan)
                self.recist_scan_days.sort()
        self.recist_results: list[dict] = []
        self.best_response_pct: float = 0.0

        _logger.debug(f"  AE risks: {json.dumps({k: round(v, 4) for k, v in self.ae_risks.items()}, ensure_ascii=False)}")
        _logger.debug(f"  Tumor: {self.tumor_response}, onset Day {self.response_onset_day}")

    # ═══════════════════════════════════════════════════════
    # Logging helpers
    # ═══════════════════════════════════════════════════════

    def _log_event(self, event_type: str, day: int, **kwargs) -> None:
        """log_event wrapper — patient_id 자동 주입."""
        pid = self.patient.get("patient_id", "?")
        log_event(pid, day, event_type, kwargs if kwargs else "")

    def _log_hazard(self, day: int, ae_term: str, hazard: float, triggered: bool) -> None:
        """log_hazard wrapper — patient_id 자동 주입."""
        pid = self.patient.get("patient_id", "?")
        log_hazard(pid, day, ae_term, hazard, triggered)

    # ═══════════════════════════════════════════════════════
    # Properties
    # ═══════════════════════════════════════════════════════

    @property
    def treatment_held(self) -> bool:
        """어느 약물이든 현재 홀드 상태인가?"""
        return bool(self.held_drugs - self.discontinued_drugs)

    @treatment_held.setter
    def treatment_held(self, value: bool):
        if value:
            self.held_drugs = set(self.all_drug_names)
        else:
            self.held_drugs.clear()

    @property
    def treatment_discontinued(self) -> bool:
        """모든 약물이 중단되었는가?"""
        return len(self.all_drug_names) > 0 and self.discontinued_drugs >= self.all_drug_names

    @treatment_discontinued.setter
    def treatment_discontinued(self, value: bool):
        if value:
            self.discontinued_drugs = set(self.all_drug_names)
            self.held_drugs -= self.discontinued_drugs

    def is_drug_held(self, drug_name: str) -> bool:
        return drug_name in self.held_drugs and drug_name not in self.discontinued_drugs

    def is_drug_discontinued(self, drug_name: str) -> bool:
        return drug_name in self.discontinued_drugs

    # ═══════════════════════════════════════════════════════
    # Drug attribution
    # ═══════════════════════════════════════════════════════

    def _get_causative_drugs(self, ae_term: str) -> set[str]:
        """AE에 대한 인과 약물(들)을 결정한다."""
        normalized = ae_term.lower().replace(" ", "_").replace("-", "_")
        for io_ae in IO_SPECIFIC_AES:
            if io_ae in normalized or normalized in io_ae:
                if self.io_drugs:
                    return set(self.io_drugs)
                break
        for chemo_ae in ADC_CHEMO_SPECIFIC_AES:
            if chemo_ae in normalized or normalized in chemo_ae:
                if self.non_io_drugs:
                    return set(self.non_io_drugs)
                break
        return set(self.all_drug_names)

    # ═══════════════════════════════════════════════════════
    # Parsing helpers
    # ═══════════════════════════════════════════════════════

    def _parse_administration_schedule(self) -> list[dict]:
        """rule_set에서 투약 스케줄을 파싱한다."""
        schedule = self.rule_set.get("administration_schedule") or []
        if not schedule or not isinstance(schedule, list):
            raise ValueError("rule_set.administration_schedule is missing or empty.")
        user_drug_name = self.rule_set.get("drug_name", "")
        user_drugs = [d.strip() for d in user_drug_name.replace("+", ",").split(",") if d.strip()]
        for drug in schedule:
            cd = drug.get("cycle_days", [1])
            if isinstance(cd, list):
                drug["cycle_days"] = [int(d) for d in cd]
            else:
                drug["cycle_days"] = [1]
            llm_name = drug.get("drug_name", "").lower()
            matched = False
            for user_name in user_drugs:
                if user_name.lower() == llm_name:
                    matched = True
                    break
                elif user_name.lower() in llm_name or llm_name in user_name.lower():
                    drug["drug_name"] = user_name
                    matched = True
                    break
            if not matched:
                for user_name in user_drugs:
                    if not any(d is not drug and d.get("drug_name") == user_name for d in schedule):
                        drug["drug_name"] = user_name
                        break
        return schedule

    def _parse_dose_modification_rules(self) -> dict[str, dict]:
        """dose_modification_rules를 ae_term으로 인덱싱된 dict로 변환.

        FDA PI 기반 오버라이드를 LLM 생성 규칙 위에 적용하고,
        감량 단계를 FDA 정확값 [1.0, 0.8, 0.6, 0.4]으로 교정한다.
        """
        _ACTION_ALIASES = {
            "hold_dose": "DRUG INTERRUPTED",
            "hold": "DRUG INTERRUPTED",
            "interrupt": "DRUG INTERRUPTED",
            "discontinue": "DRUG WITHDRAWN",
            "withdraw": "DRUG WITHDRAWN",
            "reduce": "DOSE REDUCED",
            "dose_reduce": "DOSE REDUCED",
            "no_change": "DOSE NOT CHANGED",
        }
        rules = self.rule_set.get("dose_modification_rules") or []
        result: dict[str, dict] = {}
        for rule in rules:
            ae_term = rule.get("ae_term", "default")
            ga = rule.get("grade_actions") or {}
            for g, a in ga.items():
                ga[g] = _ACTION_ALIASES.get(a, a)
            result[ae_term.lower()] = rule
        if "default" not in result:
            result["default"] = {
                "grade_actions": {"1": "DOSE NOT CHANGED", "2": "DOSE NOT CHANGED",
                                  "3": "DRUG INTERRUPTED", "4": "DRUG WITHDRAWN"},
                "dose_reduction_levels": list(PADCEV_DOSE_REDUCTION_LEVELS),
            }

        # FDA PI 오버라이드 적용 (LLM 규칙보다 우선)
        for ae_term, overrides in FDA_DOSE_MOD_OVERRIDES.items():
            key = ae_term.lower()
            if key in result:
                result[key].setdefault("grade_actions", {}).update(overrides["grade_actions"])
            else:
                result[key] = {
                    "grade_actions": dict(overrides["grade_actions"]),
                    "dose_reduction_levels": list(PADCEV_DOSE_REDUCTION_LEVELS),
                }

        # 모든 규칙의 감량 단계를 FDA 정확값으로 교정
        old_levels = [1.0, 0.75, 0.5]
        for _, rule in result.items():
            if rule.get("dose_reduction_levels") == old_levels:
                rule["dose_reduction_levels"] = list(PADCEV_DOSE_REDUCTION_LEVELS)

        return result

    def _parse_supportive_care_rules(self) -> dict[str, list[dict]]:
        """supportive_care_rules를 ae_term으로 인덱싱된 dict로 변환."""
        rules = self.rule_set.get("supportive_care_rules") or []
        result: dict[str, list[dict]] = {}
        for rule in rules:
            ae_term = rule.get("ae_term", "")
            if ae_term:
                result[ae_term.lower()] = rule.get("treatments") or []
        return result

    # ═══════════════════════════════════════════════════════
    # Initialization: AE Risk Calibration (LLM 1회)
    # ═══════════════════════════════════════════════════════

    def _calibrate_ae_risks(self) -> dict[str, float]:
        """환자별 AE 발생률을 보정한다. Stage 1: 코드, Stage 2: LLM."""
        ae_risks: dict[str, float] = {}
        for ae in self.rule_set["ae_profile"]:
            term = ae["ae_term"]
            base = float(ae.get("incidence_all_grade", 0.1))
            modifiers = ae.get("risk_modifiers") or []
            adjusted = adjust_incidence_by_risk_modifiers(
                base, modifiers, self.conditions, self.age
            )
            ae_risks[term] = adjusted

        # Stage 2: LLM fine-tuning
        try:
            pid = self.patient.get("patient_id", "?")
            demo = self.demographics
            conditions_str = ", ".join(sorted(self.conditions)) if self.conditions else "none"
            bmi = round(self.weight_kg / ((demo.get("height_cm", 170) / 100) ** 2), 1) if demo.get("height_cm") else "unknown"
            smoking = demo.get("smoking", "unknown")

            ae_summary = ", ".join(f"{k}: {v:.3f}" for k, v in ae_risks.items())
            prompt = (
                f"Patient Profile:\n"
                f"- Age: {self.age}, Sex: {demo.get('sex', '?')}, BMI: {bmi}, "
                f"Smoking: {smoking}, ECOG: {self.baseline_ecog}\n"
                f"- Comorbidities: {conditions_str}\n"
                f"- Drug: {self.rule_set.get('drug_name', '?')}\n\n"
                f"Code-adjusted AE incidences:\n  {ae_summary}\n\n"
                "Fine-tune considering comorbidity synergies, age-related PK, drug-drug interactions.\n"
                "Return JSON: {\"ae_term\": adjusted_float, ...}\n"
                "Only include AEs whose incidence you want to change."
            )
            system = (
                "You are a clinical pharmacologist. Adjust AE incidences for this specific patient. "
                "Values must be 0-1 decimal. Only adjust if patient factors warrant a change."
            )
            result = generate_json(system, prompt, model=self.model, max_tokens=2048, caller="daily_agent.calibrate")
            if isinstance(result, dict):
                for term, val in result.items():
                    if term in ae_risks and isinstance(val, (int, float)):
                        ae_risks[term] = min(max(float(val), 0.0), 0.99)
                _logger.info(f"  AE calibration: LLM adjusted {len(result)} AEs")
        except Exception as e:
            _logger.warning(f"  AE calibration LLM failed: {e}. Using code-only values.")

        return ae_risks

    # ═══════════════════════════════════════════════════════
    # Initialization: Tumor Response Sampling (코드만)
    # ═══════════════════════════════════════════════════════

    def _sample_tumor_response(self) -> tuple[str, int, float]:
        """종양 반응 카테고리와 onset을 코드로 샘플링한다."""
        disease_base = self.rule_set.get("disease_baseline") or {}
        tumor_dist = disease_base.get("tumor_response_distribution")
        if not tumor_dist:
            efficacy = self.rule_set.get("efficacy") or {}
            orr = float(efficacy.get("overall_response_rate", 0.50))
            cr_rate = float(efficacy.get("complete_response_rate", 0.10))
            pr_rate = orr - cr_rate
            tumor_dist = {"CR": cr_rate, "PR": pr_rate, "SD": 0.30, "PD": max(0.01, 1.0 - orr - 0.30)}

        # 확률 합 정규화
        total = sum(float(v) for v in tumor_dist.values())
        if total > 0:
            tumor_dist = {k: float(v) / total for k, v in tumor_dist.items()}

        response = self.sampler.categorical(tumor_dist)

        cycle_len = (self.rule_set.get("trial_design") or {}).get("cycle_length_days", 21)
        onset_day = self.sampler.numeric("normal", {
            "mean": 3 * cycle_len,
            "std": cycle_len,
            "min": 2 * cycle_len,
            "max": 8 * cycle_len,
        })
        onset_day = int(round(onset_day))

        patient_scale = self.sampler.numeric("lognormal", {
            "mu": 0, "sigma": 0.30, "min": 0.4, "max": 2.5,
        })

        _logger.info(f"  Tumor response: {response}, onset Day {onset_day}, scale {patient_scale:.2f}")
        return response, onset_day, patient_scale

    # ═══════════════════════════════════════════════════════
    # Static helpers
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _extract_number(val) -> float | None:
        """LLM이 '36.8 (°C, ...)' 같은 문자열을 반환할 때 숫자만 추출."""
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            m = re.search(r'-?\d+\.?\d*', val)
            if m:
                return float(m.group())
        return None

    @staticmethod
    def _normalize_bt(vitals: dict) -> dict:
        """체온(BT)을 섭씨로 정규화하고, 모든 수치형 vital을 float으로 변환한다."""
        for key in list(vitals.keys()):
            if key.startswith("_"):
                continue
            val = vitals[key]
            if isinstance(val, str):
                extracted = DailySimulator._extract_number(val)
                if extracted is not None:
                    vitals[key] = extracted
        bt = vitals.get("BT", vitals.get("body_temperature", vitals.get("temperature")))
        if bt is not None:
            bt = DailySimulator._extract_number(bt)
            if bt is not None:
                if bt > 45:  # Fahrenheit
                    bt = round((bt - 32) * 5 / 9, 1)
                vitals["BT"] = bt
        return vitals

    def is_administration_day(self, cycle_day: int) -> bool:
        """이 cycle_day가 투약일인가?"""
        for drug in self.admin_schedule:
            if cycle_day in drug.get("cycle_days", [1]):
                return True
        return False

    # ═══════════════════════════════════════════════════════
    # Main pipeline: generate_day()
    # ═══════════════════════════════════════════════════════

    def generate_day(self, day_results: list[dict], day: int, cycle: int,
                     cycle_day: int, is_hospital: bool) -> dict:
        """하루의 환자 상태를 생성하는 10-step 파이프라인."""
        set_caller("daily_agent")

        # 당일 해소 AE 리셋
        self._today_resolved.clear()

        if self.is_deceased:
            return self._deceased_record(day, cycle, cycle_day)

        # Step 0: Cycle freeze (치료 중단 시)
        if self.treatment_discontinued:
            if self.frozen_cycle is None:
                self.frozen_cycle = cycle
                self.frozen_cycle_day = cycle_day
            cycle = self.frozen_cycle
            cycle_day = self.frozen_cycle_day

        # Effective treatment tracking (fractional, using TUMOR_DAILY_RATE_*)
        # - Full treatment: all drugs active → +1.0 day
        # - Partial hold: some drugs held → +0.5 day (remaining drug still works)
        # - All held: all drugs held → +0.0 day (no effective treatment)
        # - Discontinued: treatment ended → +0.0 day
        if self.treatment_discontinued:
            self.effective_treatment_days += TUMOR_DAILY_RATE_DISCONTINUED
        else:
            n_held = len(self.held_drugs - self.discontinued_drugs)
            n_active = len(self.all_drug_names - self.discontinued_drugs)
            if n_active == 0:
                self.effective_treatment_days += TUMOR_DAILY_RATE_FULL
            elif n_held == 0:
                self.effective_treatment_days += TUMOR_DAILY_RATE_FULL
            elif n_held < n_active:
                self.effective_treatment_days += TUMOR_DAILY_RATE_PARTIAL_HOLD
            else:
                self.effective_treatment_days += TUMOR_DAILY_RATE_ALL_HELD

        # Step 1: New AE onset
        new_aes = self._check_new_ae_onsets(day, day_results)

        # Step 2: Active AE changes (grade transition / resolution)
        ae_changes = self._check_ae_changes(day)

        # Step 3: Tumor change
        tumor_pct = self._compute_tumor_change(day)

        # Step 3b: RECIST evaluation
        recist_result = None
        if day in self.recist_scan_days:
            recist_result = self._evaluate_recist(day, tumor_pct)

        # Step 4: Event aggregation
        events = []
        for ae_term, ae_data in new_aes.items():
            events.append(f"NEW_AE: {ae_term} G{ae_data['grade']}")
        for change in ae_changes:
            events.append(change["description"])
        if recist_result:
            events.append(f"RECIST: {recist_result.get('response', '?')}")

        # Step 4b: Drug administration
        # 병원일(is_hospital)에는 투약을 보류 — orchestrator가 dose modification 후 투약 결정
        # 비병원일(HOME)에는 스케줄대로 투약
        if not is_hospital:
            self._process_drug_administration(day, cycle_day)

        # Step 5: Event Day vs Quiet Day
        is_event = bool(events) or is_hospital or day == 1
        if is_event:
            result = self._generate_event_day(
                day, cycle, cycle_day, is_hospital, events, day_results
            )
        else:
            result = self._generate_quiet_day(
                day, cycle, cycle_day, is_hospital, day_results
            )

        # Step 6: CRF enrichment (EC records — 상태는 이미 Step 4b에서 업데이트됨)
        self._enrich_ec_records(result, day, cycle_day, is_hospital)

        # Step 6b: RECIST scan data
        if recist_result:
            result["recist_scan"] = {
                "recist_category": recist_result.get("response", "NE"),
                "tumor_change_pct": recist_result.get("change_pct", 0),
                "nadir_pct": recist_result.get("best_response_pct", 0),
                "day": day,
                "description": f"RECIST: {recist_result.get('response', 'NE')} "
                               f"({recist_result.get('change_pct', 0):+.1f}%)",
            }
            result["recist_history"] = list(self.recist_results)

        # Step 7: Baseline capture
        if day == 1:
            self._capture_baseline(result)

        # Step 8: AE cascade update
        active_aes_list = self._get_active_aes_list()
        self.ae_cascade_multipliers = compute_ae_cascade_multipliers(
            active_aes_list, self.ae_cascade_rules
        )

        # Step 9: Dynamic ECOG
        self.current_ecog = compute_dynamic_ecog(
            baseline_ecog=self.baseline_ecog,
            current_ecog=self.current_ecog,
            active_aes=active_aes_list,
            tumor_status=self.tumor_response if day > self.response_onset_day else "unknown",
            response_onset_day=self.response_onset_day,
            day=day,
            comorbidities=self.conditions,
            treatment_discontinued=self.treatment_discontinued,
            ecog_config=self.ecog_config,
        )
        result.setdefault("objective", {})["ecog"] = self.current_ecog

        # Step 10: Mortality
        mortality_risk, mortality_channels = compute_daily_mortality(
            day=day,
            active_aes=active_aes_list,
            tumor_status=self.tumor_response if day > self.response_onset_day else "unknown",
            ecog=self.current_ecog,
            treatment_discontinued=self.treatment_discontinued,
            response_onset_day=self.response_onset_day,
            risk_config=self.mortality_config,
            cumulative_doses=self.cumulative_doses,
            discontinuation_day=self.discontinuation_day,
        )
        result["_mortality_risk"] = round(mortality_risk, 6)
        result["_mortality_channels"] = mortality_channels

        if self.sampler.boolean(mortality_risk) and not self.is_deceased:
            self.is_deceased = True
            self.death_day = day
            # 사인 결정: mortality_channels에서 가장 높은 채널 사용
            top_channel = max(mortality_channels, key=mortality_channels.get) if mortality_channels else ""
            if "disease_progression" in top_channel:
                self.death_cause = "disease_progression"
            elif any(ae.get("grade", 0) >= 4 and ae.get("status") != "resolved" for ae in active_aes_list):
                self.death_cause = "treatment_toxicity"
            elif "disease" in top_channel or "tumor" in top_channel:
                self.death_cause = "disease_progression"
            else:
                self.death_cause = "clinical_deterioration"

            result["objective"]["location"] = "DECEASED"
            # 모든 약물 중단
            for dname in self.all_drug_names:
                self.discontinued_drugs.add(dname)
                self.held_drugs.discard(dname)
            self.treatment_discontinued = True
            self.discontinuation_day = day

            result["ds_record"] = {
                "DSDECOD": "DEATH",
                "DSTERM": self.death_cause,
                "DSSTDTC": day,
                "mortality_channels": mortality_channels,
            }
            self.ds_record = result["ds_record"]
            self._log_event("death", day=day, cause=self.death_cause)

        # Step 11: Discontinuation check (only if still on treatment)
        if not self.is_deceased and not self.treatment_discontinued:
            disc_risks = compute_discontinuation_risk(
                day=day,
                active_aes=active_aes_list,
                ecog=self.current_ecog,
                baseline_ecog=self.baseline_ecog,
                tumor_status=self.tumor_response if day > self.response_onset_day else "unknown",
                treatment_weeks=(day - 1) / 7,
                treatment_discontinued=self.treatment_discontinued,
                dose_reductions=self.dose_reduction_count,
                disposition_config=self.disposition_config,
            )
            result["_discontinuation_risks"] = disc_risks
            # Independent hazards: 1 - (1-p1)(1-p2)(1-p3)
            combined = 1.0
            for k, v in disc_risks.items():
                if k == "independent_hazards":
                    continue
                combined *= (1.0 - float(v))
            overall_disc_prob = 1.0 - combined

            if self.sampler.boolean(overall_disc_prob):
                # 어떤 채널에서 발생했는지 결정
                if self.sampler.boolean(disc_risks.get("patient_withdrawal", 0) / max(overall_disc_prob, 1e-9)):
                    reason = "WITHDREW CONSENT"
                elif self.sampler.boolean(disc_risks.get("physician_decision", 0) / max(overall_disc_prob, 1e-9)):
                    reason = "PHYSICIAN DECISION"
                else:
                    reason = "OTHER"
                self.treatment_discontinued = True
                self.discontinuation_day = day
                self.ds_record = {
                    "DSDECOD": reason,
                    "DSTERM": f"Discontinued: {reason}",
                    "DSSTDTC": day,
                }
                result["ds_record"] = self.ds_record
                result["objective"]["treatment_status"] = "discontinued"
                self._log_event("discontinuation", day=day, reason=reason)
        else:
            result["_discontinuation_risks"] = {
                "patient_withdrawal": 0.0,
                "physician_decision": 0.0,
                "background": 0.0,
            }

        return result

    # ═══════════════════════════════════════════════════════
    # Step 1: AE Onset
    # ═══════════════════════════════════════════════════════

    def _check_new_ae_onsets(self, day: int, day_results: list[dict]) -> dict[str, dict]:
        """각 미발생 AE의 오늘 onset 확률을 계산하고 샘플링한다."""
        new_aes: dict[str, dict] = {}

        # Zero-AE boost: 오랫동안 AE가 없으면 hazard 추가
        zero_ae_boost = 0.0
        if day > ZERO_AE_BOOST_START_DAY and not self.occurred_aes:
            boost_days = day - ZERO_AE_BOOST_START_DAY
            zero_ae_boost = min(boost_days * ZERO_AE_BOOST_PER_DAY, ZERO_AE_BOOST_MAX)

        for ae in self.rule_set["ae_profile"]:
            term = ae["ae_term"]
            # 이미 활성인 AE는 건너뜀
            if term in self.occurred_aes:
                continue

            # 재발 가능 여부: 해소된 AE는 원인 약물이 아직 투여 중이면 재발 가능
            is_recurrence = term in self.resolved_aes

            incidence = self.ae_risks.get(term, float(ae.get("incidence_all_grade", 0.1)))

            # AE cascade multiplier
            cascade_mult = self.ae_cascade_multipliers.get(term, 1.0)
            effective_incidence = min(incidence * cascade_mult, MAX_AE_CASCADE_HAZARD)

            onset_spec = ae.get("onset_day") or {"distribution": "normal", "params": {"mean": 42, "std": 14, "min": 1, "max": 180}}

            # 원인 약물의 상태 확인
            causative = self._get_causative_drugs(term)
            all_held = all(self.is_drug_held(d) for d in causative) if causative else False
            any_reduced = any(self.dose_levels.get(d, 1.0) < 1.0 for d in causative)
            all_discontinued = all(self.is_drug_discontinued(d) for d in causative) if causative else False

            if all_discontinued:
                continue  # 원인 약물 모두 중단 → onset 불가

            hazard = daily_onset_hazard(
                day=day,
                incidence=effective_incidence,
                onset_spec=onset_spec,
                is_drug_held=all_held,
                is_dose_reduced=any_reduced,
            )

            # 재발 시 hazard 감소 (원래의 50%)
            if is_recurrence:
                hazard *= AE_RECURRENCE_HAZARD_MULT

            hazard += zero_ae_boost

            triggered = self.sampler.boolean(min(hazard, 1.0))
            self._log_hazard(day, term, hazard, triggered)

            if triggered:
                # 재발 추적
                if is_recurrence:
                    self.ae_recurrence_count[term] = self.ae_recurrence_count.get(term, 0) + 1
                    self.resolved_aes.discard(term)
                    self._log_event("ae_recurrence", day=day, ae=term,
                                    count=self.ae_recurrence_count[term])

                # Grade 결정
                grade_dist = ae.get("grade_distribution") or {"1": 0.5, "2": 0.3, "3": 0.15, "4": 0.04, "5": 0.01}
                grade_dist_clean = {}
                for g, p in grade_dist.items():
                    g_int = int(g)
                    max_g = ctcae_max_grade(term)
                    if g_int <= max_g:
                        grade_dist_clean[str(g_int)] = float(p)
                if not grade_dist_clean:
                    grade_dist_clean = {"1": 1.0}
                total_p = sum(grade_dist_clean.values())
                if total_p > 0:
                    grade_dist_clean = {k: v / total_p for k, v in grade_dist_clean.items()}

                try:
                    initial_grade = int(self.sampler.categorical(grade_dist_clean))
                except (ValueError, TypeError):
                    initial_grade = 1

                # 점진적 AE는 G1에서 시작 (급성 제외)
                normalized_term = term.lower().replace(" ", "_").replace("-", "_")
                is_acute = any(a in normalized_term for a in ACUTE_ONSET_AES)
                if not is_acute and initial_grade > MAX_ONSET_GRADE_GRADUAL:
                    initial_grade = MAX_ONSET_GRADE_GRADUAL

                ae_data = {
                    "ae_term": term,
                    "grade": initial_grade,
                    "onset_day": day,
                    "status": "active",
                    "days_active": 0,
                    "peak_grade": initial_grade,
                    "days_held": 0,
                }
                self.occurred_aes[term] = ae_data
                new_aes[term] = ae_data
                self._log_event("ae_onset", day=day, ae=term, grade=initial_grade)

        return new_aes

    # ═══════════════════════════════════════════════════════
    # Step 2: AE Changes (grade transition / resolution)
    # ═══════════════════════════════════════════════════════

    def _check_ae_changes(self, day: int) -> list[dict]:
        """활성 AE의 grade 변화/해소를 계산한다."""
        changes: list[dict] = []

        for ae_term, ae_state in list(self.occurred_aes.items()):
            if ae_state["status"] != "active":
                continue

            ae_state["days_active"] = day - ae_state["onset_day"]

            # AE 프로파일에서 duration spec 가져오기
            ae_profile = next((a for a in self.rule_set["ae_profile"] if a["ae_term"] == ae_term), None)
            duration_spec = (ae_profile.get("duration_days") if ae_profile else None) or {"distribution": "normal", "params": {"mean": 21, "std": 10, "min": 3, "max": 90}}
            is_cumulative = ae_profile.get("cumulative", False) if ae_profile else False
            is_reversible = ae_profile.get("reversible", True) if ae_profile else True

            # 원인 약물 상태
            causative = self._get_causative_drugs(ae_term)
            any_held = any(self.is_drug_held(d) for d in causative) if causative else False
            any_reduced = any(self.dose_levels.get(d, 1.0) < 1.0 for d in causative)

            # conmed 상태
            has_conmed = any(
                cm.get("CMINDC", "").lower() == ae_term.lower() or ae_term.lower() in cm.get("CMINDC", "").lower()
                for cm in self.active_cm
            )
            conmed_tier = CONMED_AE_TIER.get(ae_term.lower(), 3)

            # 해소 체크
            if is_reversible and duration_spec is not None:
                res_hazard = daily_resolution_hazard(
                    days_active=ae_state["days_active"],
                    duration_spec=duration_spec,
                    is_drug_held=any_held,
                    is_dose_reduced=any_reduced,
                    has_active_conmed=has_conmed,
                    conmed_tier=conmed_tier,
                )
                if self.sampler.boolean(res_hazard):
                    # 점진적 해소: G2+ → 한 단계 낮춤, G1 → 완전 해소
                    if ae_state["grade"] > 1:
                        old_g = ae_state["grade"]
                        ae_state["grade"] -= 1
                        changes.append({
                            "ae_term": ae_term,
                            "type": "improving_toward_resolution",
                            "description": f"IMPROVING: {ae_term} G{old_g}→G{ae_state['grade']} (resolving)",
                        })
                        self._log_event("ae_improve", day=day, ae=ae_term, old=old_g, new=ae_state["grade"])
                        continue

                    # G1 → 완전 해소
                    ae_state["status"] = "resolved"
                    ae_state["resolved_day"] = day
                    self.resolved_aes.add(ae_term)
                    # 해소된 AE를 _today_resolved에 추가 (CRF 출력용)
                    self._today_resolved.append({
                        "ae": ae_term,
                        "ae_term": ae_term,
                        "grade": ae_state["grade"],
                        "onset_day": ae_state["onset_day"],
                        "status": "resolved",
                        "days_active": ae_state.get("days_active", 0),
                        "peak_grade": ae_state.get("peak_grade", ae_state["grade"]),
                        "resolved_day": day,
                    })
                    changes.append({
                        "ae_term": ae_term,
                        "type": "resolved",
                        "description": f"RESOLVED: {ae_term} (was G{ae_state['grade']}, {ae_state['days_active']}d)",
                    })
                    # Hold release는 병원 방문 시 apply_hospital_dose_modifications에서만 수행
                    # (자동 해제하면 비-방문일에도 hold가 풀리는 비현실적 상황 발생)
                    self._log_event("ae_resolved", day=day, ae=ae_term)
                    self.discontinue_conmed_for_resolved_ae(ae_term, day)
                    # AEACN 기록 정리
                    self.ae_dose_actions.pop(ae_term, None)
                    continue

            # Grade 전이 확률 계산
            probs = grade_transition_probs(
                current_grade=ae_state["grade"],
                days_active=ae_state["days_active"],
                is_cumulative=is_cumulative,
                has_active_conmed=has_conmed,
                conmed_tier=conmed_tier,
                is_drug_held=any_held,
                is_dose_reduced=any_reduced,
            )

            # ── 강제 개선 로직: dose hold 지속 시 + conmed 병용 시 ──
            # 약물 보류 7일 이상 + G3 이상 → 개선 확률 대폭 상승
            days_held = ae_state.get("days_held", 0)
            if any_held:
                ae_state["days_held"] = days_held + 1
            else:
                ae_state["days_held"] = 0
            days_held = ae_state["days_held"]

            # G3+ & dose held 7d+: 개선 확률 10-15%/day로 강제 상향
            if ae_state["grade"] >= 3 and days_held >= 7:
                forced_improve = 0.10 + (0.05 if has_conmed else 0)
                probs["improve"] = max(probs["improve"], forced_improve)
                probs["worsen"] = min(probs["worsen"], 0.005)
                probs["stable"] = 1.0 - probs["improve"] - probs["worsen"]

            # G2 & dose held/reduced + conmed → 개선 확률 boost
            elif ae_state["grade"] == 2 and (any_held or any_reduced) and has_conmed:
                if ae_state["days_active"] > 14:
                    probs["improve"] = max(probs["improve"], 0.05)
                    probs["stable"] = 1.0 - probs["improve"] - probs["worsen"]

            transition = self.sampler.categorical(probs)

            if transition == "worsen":
                max_g = ctcae_max_grade(ae_term)
                new_grade = min(ae_state["grade"] + 1, max_g)
                if new_grade != ae_state["grade"]:
                    old_g = ae_state["grade"]
                    ae_state["grade"] = new_grade
                    ae_state["peak_grade"] = max(ae_state["peak_grade"], new_grade)
                    changes.append({
                        "ae_term": ae_term,
                        "type": "worsened",
                        "description": f"WORSENED: {ae_term} G{old_g}→G{new_grade}",
                    })
                    self._log_event("ae_worsen", day=day, ae=ae_term, old=old_g, new=new_grade)
            elif transition == "improve":
                if ae_state["grade"] > 1:
                    old_g = ae_state["grade"]
                    ae_state["grade"] -= 1
                    changes.append({
                        "ae_term": ae_term,
                        "type": "improved",
                        "description": f"IMPROVED: {ae_term} G{old_g}→G{ae_state['grade']}",
                    })
                    self._log_event("ae_improve", day=day, ae=ae_term, old=old_g, new=ae_state["grade"])

        return changes

    # ═══════════════════════════════════════════════════════
    # Step 3: Tumor change
    # ═══════════════════════════════════════════════════════

    def _compute_tumor_change(self, day: int) -> float:
        """종양 크기 변화율을 계산한다."""
        eff_tx_weeks = self.effective_treatment_days / 7.0
        pct = tumor_change_pct(
            day=day,
            best_response=self.tumor_response,
            response_onset_day=self.response_onset_day,
            patient_scale=self.patient_scale,
            effective_treatment_weeks=eff_tx_weeks,
        )
        if pct is not None:
            self.best_response_pct = min(self.best_response_pct, pct) if pct < 0 else max(self.best_response_pct, pct)
        return pct or 0.0

    def _evaluate_recist(self, day: int, tumor_pct: float) -> dict | None:
        """RECIST 1.1 평가를 수행한다."""
        if tumor_pct <= -100:
            response = "CR"
        elif tumor_pct <= -30:
            response = "PR"
        elif tumor_pct >= 20:
            response = "PD"
        else:
            response = "SD"
        result = {
            "day": day,
            "response": response,
            "change_pct": round(tumor_pct, 1),
            "best_response_pct": round(self.best_response_pct, 1),
        }
        self.recist_results.append(result)
        return result

    # ═══════════════════════════════════════════════════════
    # Step 5: Event Day (LLM call)
    # ═══════════════════════════════════════════════════════

    def _generate_event_day(self, day: int, cycle: int, cycle_day: int,
                            is_hospital: bool, events: list[str],
                            day_results: list[dict]) -> dict:
        """이벤트 날: LLM이 상세 상태를 생성한다."""
        # LLM 프롬프트용: 활성 AE만 (해소된 것 제외)
        active_aes_for_prompt = self._get_active_aes_list(include_resolved_today=False)
        # CRF 출력용: 당일 해소된 AE도 포함
        active_aes_for_crf = self._get_active_aes_list(include_resolved_today=True)
        tumor_pct = self._compute_tumor_change(day)

        # 이전 데이터 요약 (컨텍스트)
        summary = compress_history(self.patient, day_results, recent_n=3)

        pid = self.patient.get("patient_id", "?")
        drug_name = self.rule_set.get("drug_name", "?")
        demo = self.demographics

        ae_desc = json.dumps(active_aes_for_prompt, ensure_ascii=False) if active_aes_for_prompt else "none"
        events_str = "; ".join(events) if events else "no events"
        cm_list = [{"drug": c.get("CMTRT", ""), "indication": c.get("CMINDC", "")} for c in self.active_cm[:10]]

        system_prompt = (
            "You are a clinical data specialist generating realistic patient simulation data.\n"
            "Generate labs, vitals, and subjective state for the given patient day.\n"
            "Rules:\n"
            "- Labs must be physiologically consistent with active AEs and treatment status\n"
            "- Vitals must reflect clinical state (fever if neutropenic, tachycardia if anemic, etc)\n"
            "- Day-to-day changes must be gradual (no sudden jumps without medical reason)\n"
            "- Output ONLY valid JSON"
        )

        user_prompt = (
            f"Patient {pid}, Day {day} (Cycle {cycle} Day {cycle_day})\n"
            f"Drug: {drug_name}\n"
            f"Demographics: age {demo.get('age')}, sex {demo.get('sex')}, weight {self.weight_kg}kg\n"
            f"ECOG: {self.current_ecog}\n"
            f"Treatment status: {'discontinued' if self.treatment_discontinued else 'held' if self.treatment_held else 'on_treatment'}\n"
            f"Hospital visit: {is_hospital}\n\n"
            f"Active AEs: {ae_desc}\n"
            f"Today's events: {events_str}\n"
            f"Tumor change: {tumor_pct:+.1f}%\n"
            f"Active medications: {json.dumps(cm_list, ensure_ascii=False)}\n\n"
            f"Previous summary: {json.dumps(summary, ensure_ascii=False)}\n\n"
            "Generate JSON with:\n"
            "{\n"
            '  "labs": {"lab_name": {"value": float, "unit": str, "trend": str}, ...},\n'
            '  "vitals": {"SBP": int, "DBP": int, "HR": int, "BT": float, "RR": int, "SpO2": int, "weight_kg": float},\n'
            '  "subjective": {"overall_awareness": "UNAWARE|NOTICED|CONCERNED|DISTRESSED|EMERGENCY",\n'
            '                  "symptoms_patient_perceives": [{"symptom": str, "severity": str}]},\n'
            '  "narrative": "Brief clinical note for the day"\n'
            "}"
        )

        try:
            llm_result = generate_json(system_prompt, user_prompt, model=self.model, max_tokens=4096)
        except Exception as e:
            _logger.warning(f"Event day LLM failed for Day {day}: {e}. Using code fallback.")
            llm_result = {}

        # Normalize and build result (pass day_results for lab clamping)
        result = self._build_day_result(day, cycle, cycle_day, is_hospital, llm_result, events, "event_day", day_results)

        # AE state sync: 코드 값으로 LLM 값 덮어쓰기 (당일 해소 AE 포함)
        result["objective"]["active_aes"] = active_aes_for_crf
        result["objective"]["tumor"] = {"estimated_change_pct": round(tumor_pct, 1)}

        return result

    # ═══════════════════════════════════════════════════════
    # Step 5: Quiet Day (code only)
    # ═══════════════════════════════════════════════════════

    def _generate_quiet_day(self, day: int, cycle: int, cycle_day: int,
                            is_hospital: bool, day_results: list[dict]) -> dict:
        """조용한 날: LLM 호출 없이 코드만으로 상태를 생성한다."""
        # OU process for labs
        labs = self._apply_ou_labs(day, day_results)
        vitals = self._apply_ou_vitals(day, day_results)
        tumor_pct = self._compute_tumor_change(day)
        # CRF 출력용: 당일 해소된 AE도 포함
        active_aes = self._get_active_aes_list(include_resolved_today=True)

        result = {
            "patient_id": self.patient.get("patient_id"),
            "day": day,
            "cycle": cycle,
            "cycle_day": cycle_day,
            "_generation_mode": "quiet",
            "_events_summary": "",
            "objective": {
                "location": self._determine_location(day, is_hospital),
                "treatment_status": self._get_treatment_status(),
                "ecog": self.current_ecog,
                "tumor": {"estimated_change_pct": round(tumor_pct, 1)},
                "active_aes": active_aes,
                "labs": labs,
                "vitals": vitals,
            },
            "subjective": {
                "overall_awareness": self._determine_awareness(active_aes),
                "symptoms_patient_perceives": [],
            },
        }

        # cm_records
        result["cm_records"] = self._get_active_cm_records(day)

        # 약물별 상태
        self._add_drug_status(result, day, cycle_day)

        return result

    # ═══════════════════════════════════════════════════════
    # Ornstein-Uhlenbeck (OU) processes
    # ═══════════════════════════════════════════════════════

    def _apply_ou_labs(self, day: int, day_results: list[dict]) -> dict:
        """OU process로 lab 값을 변동시킨다."""
        prev_labs = {}
        if day_results:
            prev = day_results[-1]
            # labs는 objective 안에 있음
            prev_labs = prev.get("objective", {}).get("labs", {})
            if not prev_labs:
                # 최상위에 있을 수도 (호환)
                prev_labs = prev.get("labs", {})
            if not prev_labs:
                # CDASH 형식에서 찾기
                lb = prev.get("LB", {})
                if lb and "results" in lb:
                    for name, data in lb["results"].items():
                        prev_labs[name] = {"value": data.get("LBORRES", data.get("value")), "unit": data.get("LBORRESU", data.get("unit", ""))}

        # Baseline labs가 없으면 이전 값에서 초기화
        if not self.baseline_labs and prev_labs:
            for name, data in prev_labs.items():
                val = data.get("value", data) if isinstance(data, dict) else data
                _parsed = DailySimulator._extract_number(val)
                if _parsed is not None:
                    self.baseline_labs[name] = _parsed
        elif not self.baseline_labs:
            bl = self.patient.get("emr", {}).get("baseline_labs", {})
            for name, val in bl.items():
                if isinstance(val, (int, float)):
                    self.baseline_labs[name] = float(val)
                elif isinstance(val, dict):
                    _parsed = DailySimulator._extract_number(val.get("value", val.get("LBORRES", 0)))
                    if _parsed is not None:
                        self.baseline_labs[name] = _parsed

        labs_result = {}
        active_aes = self._get_active_aes_list()

        # AE→Lab 링크
        ae_lab_links = self.lab_causality_config.get("ae_lab_links") or []
        if not ae_lab_links:
            ae_lab_links = DEFAULT_AE_LAB_LINKS
        cum_dose_effects = self.lab_causality_config.get("cumulative_dose_effects") or []
        cm_lab_effects = DEFAULT_CM_LAB_EFFECTS

        for lab_name, baseline in self.baseline_labs.items():
            if baseline is None or baseline == 0:
                continue

            # 이전 값
            prev_val = baseline
            if lab_name in prev_labs:
                pv = prev_labs[lab_name]
                if isinstance(pv, dict):
                    prev_val = float(pv.get("value", pv.get("LBORRES", baseline)))
                elif isinstance(pv, (int, float)):
                    prev_val = float(pv)

            # 인과적 목표값 계산 (3 layers: AE→Lab, CumDose→Lab, CM correction)
            target = compute_causal_lab_target(
                lab_name=lab_name,
                baseline_value=baseline,
                active_aes=active_aes,
                cumulative_doses=self.cumulative_doses,
                ae_lab_links=ae_lab_links,
                cumulative_dose_effects=cum_dose_effects,
                active_cms=self.active_cm,
                cm_lab_effects=cm_lab_effects,
            )

            # Layer 4: CM side effects (e.g., steroid → glucose↑, ANC↑)
            cm_side_mult = self._compute_cm_lab_side_effect(lab_name, day)
            if cm_side_mult != 1.0:
                target = target * cm_side_mult

            # OU step
            theta = OU_THETA_LABS
            if lab_name in SLOW_MARKER_LABS:
                theta *= 0.2  # 느린 마커는 천천히 이동

            noise_frac = LABS_NOISE_FRACTION_MAP.get(lab_name, LABS_NOISE_FRACTION)
            noise_std = abs(prev_val) * noise_frac
            noise = self.sampler.numeric("normal", {"mean": 0, "std": max(noise_std, 0.01)})

            new_val = prev_val + theta * (target - prev_val) + noise

            # Max daily delta 제한
            max_delta = MAX_DAILY_LAB_DELTA.get(lab_name, abs(prev_val) * 0.1)
            delta = new_val - prev_val
            if abs(delta) > max_delta:
                delta = max_delta if delta > 0 else -max_delta
                new_val = prev_val + delta

            # CTCAE 일관성: 활성 AE의 grade에 맞는 lab 범위로 수렴
            new_val = self._apply_ctcae_lab_convergence(lab_name, new_val, prev_val, active_aes)

            # 음수 방지 + 반올림
            new_val = max(0.0, new_val)
            decimals = LAB_ROUNDING.get(lab_name, 1)
            new_val = round(new_val, decimals)

            # 반올림 후 delta 재검증 (반올림이 delta 초과를 유발할 수 있음)
            if lab_name in MAX_DAILY_LAB_DELTA:
                post_round_delta = abs(new_val - round(prev_val, decimals))
                if post_round_delta > MAX_DAILY_LAB_DELTA[lab_name] * 1.01:
                    new_val = round(prev_val, decimals)

            # Trend 결정
            diff = new_val - prev_val
            if abs(diff) < abs(prev_val) * 0.02:
                trend = "stable"
            elif diff > 0:
                trend = "slight_increase" if diff < abs(prev_val) * 0.1 else "increase"
            else:
                trend = "slight_decrease" if abs(diff) < abs(prev_val) * 0.1 else "decrease"

            labs_result[lab_name] = {
                "value": new_val,
                "unit": self._get_lab_unit(lab_name),
                "trend": trend,
            }

        return labs_result

    def _apply_ou_vitals(self, day: int, day_results: list[dict]) -> dict:
        """OU process로 vitals 값을 변동시킨다.

        3-layer target model:
          1. Baseline (mean-reversion target)
          2. AE → Vitals shift (neutropenia G3+ → fever/tachycardia, pneumonitis → SpO2↓)
          3. CM side effects (steroid → weight↑, SBP↑)
        """
        prev_vitals = {}
        if day_results:
            prev = day_results[-1]
            prev_vitals = prev.get("objective", {}).get("vitals", {})
            if not prev_vitals:
                prev_vitals = prev.get("vitals", {})
            if not prev_vitals:
                vs = prev.get("VS", {})
                if vs:
                    prev_vitals = {
                        "SBP": vs.get("SYSBP_VSORRES"),
                        "DBP": vs.get("DIABP_VSORRES"),
                        "HR": vs.get("PULSE_VSORRES"),
                        "BT": vs.get("TEMP_VSORRES"),
                        "RR": vs.get("RESP_VSORRES"),
                        "SpO2": vs.get("_SpO2"),
                        "weight_kg": vs.get("WEIGHT_VSORRES"),
                    }

        baseline = self.baseline_vitals

        # ── AE-driven vitals targets ──
        ae_vitals_shift = self._compute_ae_vitals_shift()

        # ── CM side effects (daily deltas for weight, SBP) ──
        cm_vitals_delta = self._compute_cm_vitals_delta(day)

        result = {}
        for vital_name, noise_std in VITALS_NOISE.items():
            bl = float(baseline.get(vital_name, 0))
            if bl == 0:
                if vital_name == "SpO2":
                    bl = 97
                elif vital_name == "BT":
                    bl = 36.6
                else:
                    continue

            prev_val = prev_vitals.get(vital_name, bl) if prev_vitals else bl
            if prev_val is None:
                prev_val = bl
            prev_val = float(prev_val)

            # Target = baseline + AE shift
            target = bl + ae_vitals_shift.get(vital_name, 0.0)

            noise = self.sampler.numeric("normal", {"mean": 0, "std": noise_std})
            new_val = prev_val + OU_THETA_VITALS * (target - prev_val) + noise

            # Apply CM-driven daily delta (e.g., steroid weight gain)
            new_val += cm_vitals_delta.get(vital_name, 0.0)

            # 범위 제한
            if vital_name == "SpO2":
                new_val = max(85, min(100, new_val))
            elif vital_name == "BT":
                new_val = max(35.0, min(41.0, new_val))
            elif vital_name == "HR":
                new_val = max(40, min(180, new_val))
            elif vital_name in ("SBP", "DBP"):
                new_val = max(60, min(220, new_val))

            decimals = VITAL_ROUNDING.get(vital_name, 0)
            new_val = round(new_val, decimals)
            result[vital_name] = new_val

        return result

    def _compute_ae_vitals_shift(self) -> dict[str, float]:
        """활성 AE에 의한 vitals 변동 target shift를 계산한다.

        의학적 근거:
        - Febrile neutropenia (ANC < 0.5 + G3+) → BT↑ 38.5-40°C, HR↑ +20-40
        - Pneumonitis G2+ → SpO2↓ -3 to -10, RR↑ +4-8
        - Severe AE (any G4) → general tachycardia HR↑ +15
        - Diarrhea G3+ → HR↑ (dehydration), SBP↓ -10
        """
        shift: dict[str, float] = {}
        for ae_term, ae_state in self.occurred_aes.items():
            if ae_state.get("status") != "active":
                continue
            grade = ae_state.get("grade", 1)
            term_lower = ae_term.lower().replace(" ", "_").replace("-", "_")

            if "neutropenia" in term_lower and grade >= 3:
                shift["BT"] = shift.get("BT", 0) + (1.5 if grade == 3 else 2.5)
                shift["HR"] = shift.get("HR", 0) + (20 if grade == 3 else 35)
            if "febrile" in term_lower:
                shift["BT"] = shift.get("BT", 0) + 2.0
                shift["HR"] = shift.get("HR", 0) + 25

            if "pneumonitis" in term_lower and grade >= 2:
                spo2_drop = {2: -3, 3: -6, 4: -10}.get(grade, -3)
                shift["SpO2"] = shift.get("SpO2", 0) + spo2_drop
                shift["RR"] = shift.get("RR", 0) + (4 if grade == 2 else 8)

            if "diarrhea" in term_lower and grade >= 3:
                shift["HR"] = shift.get("HR", 0) + 10
                shift["SBP"] = shift.get("SBP", 0) - 10
                shift["DBP"] = shift.get("DBP", 0) - 5

            if grade >= 4:
                shift["HR"] = shift.get("HR", 0) + 15

        return shift

    def _compute_cm_vitals_delta(self, day: int) -> dict[str, float]:
        """보조약물(특히 스테로이드)의 vitals 부작용을 계산한다.

        DEFAULT_CM_SIDE_EFFECTS에 정의된 vitals_effects를 적용.
        예: prednisone → weight_kg +0.05~0.15/day, SBP +0.3~0.8/day
        """
        delta: dict[str, float] = {}
        for cm in self.active_cm:
            cm_name = (cm.get("CMTRT", "") or cm.get("name", "")).lower()
            cm_start = cm.get("_start_day", cm.get("CMSTDTC", 1))
            if isinstance(cm_start, str):
                try:
                    cm_start = int(cm_start)
                except (ValueError, TypeError):
                    cm_start = 1
            days_on_cm = max(0, day - cm_start)

            cm_dose_mg = 0.0
            dose_str = cm.get("CMDOSE", cm.get("dose", ""))
            if isinstance(dose_str, (int, float)):
                cm_dose_mg = float(dose_str)
            elif isinstance(dose_str, str):
                m = re.search(r"(\d+\.?\d*)", dose_str)
                if m:
                    cm_dose_mg = float(m.group(1))

            for cm_rule in DEFAULT_CM_SIDE_EFFECTS:
                keywords = cm_rule.get("drug_keywords", [])
                if not any(kw in cm_name for kw in keywords):
                    continue

                for vfx in cm_rule.get("vitals_effects", []):
                    vital = vfx["vital"]
                    onset = vfx.get("onset_days", 0)
                    if days_on_cm < onset:
                        continue

                    threshold = vfx.get("high_dose_threshold_mg", 999)
                    is_high_dose = cm_dose_mg >= threshold

                    daily_d = vfx.get("high_dose_daily_delta", vfx["daily_delta"]) if is_high_dose else vfx["daily_delta"]

                    # Cap cumulative effect
                    max_total = vfx.get("max_total_delta", 999)
                    total_so_far = daily_d * (days_on_cm - onset)
                    if abs(total_so_far) >= abs(max_total):
                        continue  # reached max effect

                    delta[vital] = delta.get(vital, 0) + daily_d

        return delta

    def _compute_cm_lab_side_effect(self, lab_name: str, day: int) -> float:
        """보조약물 자체의 lab 부작용 배수를 계산한다.

        DEFAULT_CM_SIDE_EFFECTS에 정의된 effects 적용.
        예: prednisone → glucose_fasting ×1.5~2.2, ANC ×1.5~2.5
        반환값은 baseline 대비 target multiplier.
        """
        combined_mult = 1.0
        for cm in self.active_cm:
            cm_name = (cm.get("CMTRT", "") or cm.get("name", "")).lower()
            cm_start = cm.get("_start_day", cm.get("CMSTDTC", 1))
            if isinstance(cm_start, str):
                try:
                    cm_start = int(cm_start)
                except (ValueError, TypeError):
                    cm_start = 1
            days_on_cm = max(0, day - cm_start)

            cm_dose_mg = 0.0
            dose_str = cm.get("CMDOSE", cm.get("dose", ""))
            if isinstance(dose_str, (int, float)):
                cm_dose_mg = float(dose_str)
            elif isinstance(dose_str, str):
                m = re.search(r"(\d+\.?\d*)", dose_str)
                if m:
                    cm_dose_mg = float(m.group(1))

            for cm_rule in DEFAULT_CM_SIDE_EFFECTS:
                keywords = cm_rule.get("drug_keywords", [])
                if not any(kw in cm_name for kw in keywords):
                    continue

                for lfx in cm_rule.get("effects", []):
                    if lfx["lab"] != lab_name:
                        continue
                    onset = lfx.get("onset_days", 0)
                    if days_on_cm < onset:
                        continue

                    threshold = lfx.get("high_dose_threshold_mg", 999)
                    is_high_dose = cm_dose_mg >= threshold

                    mult = lfx.get("high_dose_target_multiplier", lfx["target_multiplier"]) if is_high_dose else lfx["target_multiplier"]
                    combined_mult *= mult

        return combined_mult

    def _apply_ctcae_lab_convergence(self, lab_name: str, new_val: float,
                                      prev_val: float, active_aes: list[dict]) -> float:
        """활성 AE grade와 lab 값의 CTCAE 일관성을 확보한다."""
        for ae in active_aes:
            if ae.get("status") == "resolved":
                continue
            ae_term = ae.get("ae_term", ae.get("ae", ""))
            grade = ae.get("grade", 1)
            ranges = ctcae_lab_range(ae_term, grade)
            if not ranges:
                continue
            for rng_lab, rng_min, rng_max in ranges:
                if rng_lab != lab_name:
                    continue
                target_mid = (rng_min + rng_max) / 2
                max_delta = MAX_DAILY_LAB_DELTA.get(lab_name, abs(prev_val) * 0.1)
                delta = target_mid - new_val
                if abs(delta) > max_delta:
                    delta = max_delta if delta > 0 else -max_delta
                new_val = new_val + delta
                break
        return new_val

    # ═══════════════════════════════════════════════════════
    # Conmed Prescription (병원 방문 시 보조약 처방)
    # ═══════════════════════════════════════════════════════

    def prescribe_conmeds_for_aes(self, observed_aes: list[dict], day: int):
        """병원 방문 시 관찰된 AE에 대해 보조약을 처방한다.

        orchestrator에서 병원 방문(is_visit)일 때 호출됨.
        rule_set.supportive_care_rules + 기본 fallback 사용.
        """
        sc_rules = self.rule_set.get("supportive_care_rules") or []
        sc_map = {r["ae_term"].lower(): (r.get("treatments") or []) for r in sc_rules}

        for ae in observed_aes:
            ae_term = ae.get("ae", ae.get("ae_term", ""))
            grade = ae.get("grade", 1)
            if grade < 1:
                continue

            # 이미 이 AE에 보조약이 처방되어 있으면 건너뜀
            already_prescribed = any(
                cm.get("CMINDC", "").lower() == ae_term.lower()
                or ae_term.lower() in cm.get("CMINDC", "").lower()
                for cm in self.active_cm
                if not cm.get("_baseline")
            )
            if already_prescribed:
                continue

            # G1: skip unless rule_set defines SC for this AE, or it's nausea/pruritus
            has_sc_rule = ae_term.lower() in sc_map
            if grade < 2 and not has_sc_rule and ae_term.lower() not in ("nausea", "pruritus"):
                continue

            treatments = sc_map.get(ae_term.lower()) or []
            if not treatments:
                treatments = self._default_conmed_for_ae(ae_term)
            if not treatments:
                continue

            # 확률 기반 선택
            selected = None
            for tx in treatments:
                prob = float(tx.get("probability", 0.5))
                if self.sampler.boolean(prob):
                    selected = tx
                    break
            if selected is None and treatments:
                selected = treatments[0]
            if selected is None:
                continue

            cm_record = {
                "CMTRT": selected["drug"],
                "CMINDC": ae_term,
                "CMDSTXT": str(selected.get("dose", "")),
                "CMDOSU": selected.get("unit", "mg"),
                "CMDOSFRQ": selected.get("frequency", "QD"),
                "CMROUTE": selected.get("route", "ORAL"),
                "CMSTDAT": day,
                "CMONGO": True,
                "CMENDAT": None,
                "_baseline": False,
            }
            self.active_cm.append(cm_record)
            self._log_event("conmed_prescribed", day=day,
                            drug=selected["drug"], ae=ae_term, grade=grade)

    def discontinue_conmed_for_resolved_ae(self, ae_term: str, day: int):
        """해소된 AE에 대한 보조약을 종료한다."""
        for cm in self.active_cm:
            if cm.get("_baseline"):
                continue
            if (cm.get("CMINDC", "").lower() == ae_term.lower()
                    or ae_term.lower() in cm.get("CMINDC", "").lower()):
                if cm.get("CMONGO") and cm.get("CMENDAT") is None:
                    cm["CMENDAT"] = day
                    cm["CMONGO"] = False
                    self._log_event("conmed_stopped", day=day,
                                    drug=cm.get("CMTRT"), ae=ae_term)

    @staticmethod
    def _default_conmed_for_ae(ae_term: str) -> list[dict]:
        """rule_set에 없는 AE에 대한 기본 보조약 매핑."""
        _DEFAULTS = {
            "nausea": [{"drug": "Ondansetron", "dose": "8", "unit": "mg",
                        "route": "ORAL", "frequency": "Q8H", "probability": 0.8}],
            "vomiting": [{"drug": "Ondansetron", "dose": "8", "unit": "mg",
                          "route": "ORAL", "frequency": "Q8H", "probability": 0.8}],
            "diarrhea": [{"drug": "Loperamide", "dose": "4", "unit": "mg",
                          "route": "ORAL", "frequency": "Q6H PRN", "probability": 0.8}],
            "stomatitis": [{"drug": "Magic mouthwash", "dose": "15", "unit": "mL",
                            "route": "ORAL", "frequency": "QID", "probability": 0.7}],
            "peripheral_neuropathy": [
                {"drug": "Gabapentin", "dose": "300", "unit": "mg",
                 "route": "ORAL", "frequency": "TID", "probability": 0.5},
                {"drug": "Pregabalin", "dose": "75", "unit": "mg",
                 "route": "ORAL", "frequency": "BID", "probability": 0.5}],
            "hyperglycemia": [{"drug": "Insulin glargine", "dose": "10", "unit": "units",
                               "route": "SC", "frequency": "QD", "probability": 0.6}],
            "hypothyroidism": [{"drug": "Levothyroxine", "dose": "50", "unit": "mcg",
                                "route": "ORAL", "frequency": "QD", "probability": 0.9}],
            "hyperthyroidism": [{"drug": "Methimazole", "dose": "10", "unit": "mg",
                                 "route": "ORAL", "frequency": "QD", "probability": 0.8}],
            "anemia": [{"drug": "Epoetin alfa", "dose": "40000", "unit": "units",
                        "route": "SC", "frequency": "QW", "probability": 0.4}],
            "neutropenia": [{"drug": "Filgrastim", "dose": "5", "unit": "mcg/kg",
                             "route": "SC", "frequency": "QD", "probability": 0.5}],
            "hepatotoxicity": [{"drug": "Ursodiol", "dose": "300", "unit": "mg",
                                "route": "ORAL", "frequency": "BID", "probability": 0.5}],
            "pneumonitis": [{"drug": "Prednisone", "dose": "1", "unit": "mg/kg",
                             "route": "ORAL", "frequency": "QD", "probability": 0.9}],
            "colitis": [{"drug": "Prednisone", "dose": "1", "unit": "mg/kg",
                         "route": "ORAL", "frequency": "QD", "probability": 0.9}],
            "arthralgia": [{"drug": "Acetaminophen", "dose": "500", "unit": "mg",
                            "route": "ORAL", "frequency": "Q6H PRN", "probability": 0.6}],
            "myalgia": [{"drug": "Acetaminophen", "dose": "500", "unit": "mg",
                         "route": "ORAL", "frequency": "Q6H PRN", "probability": 0.6}],
            "dysgeusia": [],  # 효과적 치료 없음
            "alopecia": [],  # 치료 없음 (rule_set에 minoxidil이 있을 수 있음)
        }
        return _DEFAULTS.get(ae_term.lower(), [])

    def _get_active_cm_records(self, day: int) -> list[dict]:
        """현재 활성 상태인 CM 레코드를 반환한다 (CMENDAT이 없거나 >= day)."""
        return [
            cm for cm in self.active_cm
            if cm.get("CMONGO", True) or (cm.get("CMENDAT") is not None and cm["CMENDAT"] >= day)
        ]

    # ═══════════════════════════════════════════════════════
    # Result building helpers
    # ═══════════════════════════════════════════════════════

    def _clamp_llm_labs(self, llm_labs: dict, day_results: list[dict]) -> dict:
        """LLM이 생성한 lab 값에 MAX_DAILY_LAB_DELTA를 적용하여 비현실적 급변을 방지한다."""
        if not llm_labs:
            return llm_labs

        # 이전 labs 추출
        prev_labs: dict = {}
        if day_results:
            prev = day_results[-1] if day_results else {}
            prev_labs = prev.get("objective", {}).get("labs", {})
            if not prev_labs:
                prev_labs = prev.get("labs", {})
            if not prev_labs:
                lb = prev.get("LB", {})
                if lb and "results" in lb:
                    for name, data in lb["results"].items():
                        prev_labs[name] = {"value": data.get("LBORRES", data.get("value")),
                                           "unit": data.get("LBORRESU", "")}

        if not prev_labs:
            return llm_labs

        clamped = {}
        for lab_name, data in llm_labs.items():
            if isinstance(data, dict):
                val = data.get("value")
                if val is None:
                    clamped[lab_name] = data
                    continue
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    clamped[lab_name] = data
                    continue
            elif isinstance(data, (int, float)):
                val = float(data)
                data = {"value": val}
            else:
                clamped[lab_name] = data
                continue

            # 이전 값
            prev_data = prev_labs.get(lab_name)
            if prev_data is None:
                clamped[lab_name] = data
                continue
            prev_val = prev_data.get("value", prev_data) if isinstance(prev_data, dict) else prev_data
            try:
                prev_val = float(prev_val)
            except (ValueError, TypeError):
                clamped[lab_name] = data
                continue

            max_delta = MAX_DAILY_LAB_DELTA.get(lab_name, abs(prev_val) * 0.1 if prev_val else 1.0)
            delta = val - prev_val
            if abs(delta) > max_delta:
                val = prev_val + (max_delta if delta > 0 else -max_delta)

            # 반올림
            decimals = LAB_ROUNDING.get(lab_name, 2)
            val = round(val, decimals)

            # 반올림 후 delta 재검증
            if lab_name in MAX_DAILY_LAB_DELTA:
                post_round_delta = abs(val - round(prev_val, decimals))
                if post_round_delta > MAX_DAILY_LAB_DELTA[lab_name] * 1.01:
                    val = round(prev_val, decimals)

            new_data = dict(data)
            new_data["value"] = val
            clamped[lab_name] = new_data

        return clamped

    def _build_day_result(self, day: int, cycle: int, cycle_day: int,
                          is_hospital: bool, llm_result: dict,
                          events: list[str], mode: str,
                          day_results: list[dict] | None = None) -> dict:
        """LLM 결과 + 코드 상태를 합쳐 day result를 생성한다."""
        labs = llm_result.get("labs", {})
        vitals = llm_result.get("vitals", {})
        subjective = llm_result.get("subjective", {})

        # LLM labs가 비어있으면 OU fallback
        if not labs:
            labs = self._apply_ou_labs(day, day_results or [])
        else:
            # LLM이 생성한 lab에도 MAX_DAILY_LAB_DELTA 적용
            labs = self._clamp_llm_labs(labs, day_results or [])
        if not vitals:
            vitals = self._apply_ou_vitals(day, day_results or [])

        # CRF 출력용: 당일 해소된 AE도 포함
        active_aes = self._get_active_aes_list(include_resolved_today=True)

        result = {
            "patient_id": self.patient.get("patient_id"),
            "day": day,
            "cycle": cycle,
            "cycle_day": cycle_day,
            "_generation_mode": mode,
            "_events_summary": "; ".join(events) if events else "",
            "objective": {
                "location": self._determine_location(day, is_hospital),
                "treatment_status": self._get_treatment_status(),
                "ecog": self.current_ecog,
                "tumor": {"estimated_change_pct": 0.0},
                "active_aes": active_aes,
                "labs": labs,
                "vitals": vitals,
            },
            "subjective": subjective or {
                "overall_awareness": self._determine_awareness(active_aes),
                "symptoms_patient_perceives": [],
            },
        }

        # cm_records: 현재 활성 보조약 목록을 CRF mapper가 읽을 수 있도록 추가
        result["cm_records"] = self._get_active_cm_records(day)

        self._add_drug_status(result, day, cycle_day)
        return result

    def _get_active_aes_list(self, include_resolved_today: bool = False) -> list[dict]:
        """현재 활성 AE 목록을 리스트로 반환한다.

        Args:
            include_resolved_today: True면 당일 해소된 AE도 포함 (CRF 출력용)
        """
        aes = []
        for term, ae_state in self.occurred_aes.items():
            if ae_state["status"] == "active":
                ae_entry = {
                    "ae": term,
                    "ae_term": term,
                    "grade": ae_state["grade"],
                    "onset_day": ae_state["onset_day"],
                    "status": "active",
                    "days_active": ae_state.get("days_active", 0),
                    "peak_grade": ae_state.get("peak_grade", ae_state["grade"]),
                }
                # AEACN attribution: 이 AE가 약물 조치를 야기했으면 해당 action 기록
                if term in self.ae_dose_actions:
                    ae_entry["AEACN"] = self.ae_dose_actions[term]
                aes.append(ae_entry)

        # 당일 해소된 AE 포함 (CRF에 AEOUT/AEENDAT 기록 필요)
        if include_resolved_today and self._today_resolved:
            for resolved_ae in self._today_resolved:
                aes.append(resolved_ae)

        return aes

    def _determine_location(self, day: int, is_hospital: bool) -> str:
        """환자 위치 결정. ECOG 4 또는 Grade 4+ AE (AESLIFE)는 입원."""
        if self.is_deceased:
            return "DECEASED"
        # ECOG 4+ 또는 Grade 4+ AE → 입원
        max_ae_grade = max(
            (ae.get("grade", 0) for ae in self.occurred_aes.values()
             if ae.get("status") == "active"),
            default=0,
        )
        if self.current_ecog >= 4 or max_ae_grade >= 4:
            return "INPATIENT"
        # Grade 3 + AESHOSP → 입원
        if max_ae_grade >= 3:
            return "INPATIENT"
        if is_hospital or day == 1:
            return "OUTPATIENT"
        return "HOME"

    def _get_treatment_status(self) -> str:
        """현재 치료 상태 문자열."""
        if self.treatment_discontinued:
            return "discontinued"
        if self.treatment_held:
            return "held"
        return "on_treatment"

    def _determine_awareness(self, active_aes: list[dict]) -> str:
        """환자 인식 수준 결정."""
        if not active_aes:
            return "UNAWARE"
        max_g = max(ae.get("grade", 0) for ae in active_aes)
        if max_g >= 4:
            return "EMERGENCY"
        if max_g >= 3:
            return "DISTRESSED"
        if max_g >= 2:
            return "CONCERNED"
        return "NOTICED"

    def _add_drug_status(self, result: dict, day: int, cycle_day: int):
        """약물별 상태를 result에 추가한다."""
        for drug in self.admin_schedule:
            dname = drug["drug_name"]
            cycle_days = drug.get("cycle_days", [1])
            cycle_len = (self.rule_set.get("trial_design") or {}).get("cycle_length_days", 21)

            # 다음 투약 예정일 계산
            next_admin = None
            for cd in sorted(cycle_days):
                if cd > cycle_day:
                    days_until = cd - cycle_day
                    next_admin = day + days_until
                    break
            if next_admin is None and cycle_days:
                days_until = cycle_len - cycle_day + min(cycle_days)
                next_admin = day + days_until

            result["objective"][dname] = {
                "last_administered_day": self.last_admin_day.get(dname),
                "cumulative_dose_mg": round(self.cumulative_doses.get(dname, 0), 1),
                "dose_level": self.dose_levels.get(dname, 1.0),
                "next_scheduled_day": next_admin,
                "treatment_held": self.is_drug_held(dname),
                "treatment_discontinued": self.is_drug_discontinued(dname),
            }

    def _get_lab_unit(self, lab_name: str) -> str:
        """lab 단위를 반환한다."""
        ref = self.rule_set.get("lab_reference_ranges") or {}
        if lab_name in ref:
            return ref[lab_name].get("unit", "")
        defaults = {
            "hemoglobin": "g/dL", "ANC": "x10^9/L", "platelets": "x10^9/L",
            "creatinine": "mg/dL", "eGFR": "mL/min/1.73m^2",
            "ALT": "U/L", "AST": "U/L", "total_bilirubin": "mg/dL",
            "glucose_fasting": "mg/dL", "HbA1c": "%", "TSH": "mIU/L",
            "LDH": "U/L", "albumin": "g/dL", "sodium": "mmol/L",
            "potassium": "mmol/L",
        }
        return defaults.get(lab_name, "")

    def _process_drug_administration(self, day: int, cycle_day: int):
        """투약일에 약물 상태(cumulative dose, last admin day)를 업데이트한다.
        _add_drug_status 전에 호출되어야 result에 당일 투약이 반영된다."""
        if self.treatment_discontinued or self.is_deceased:
            return
        for drug in self.admin_schedule:
            dname = drug["drug_name"]
            if cycle_day not in drug.get("cycle_days", [1]):
                continue
            if self.is_drug_held(dname) or self.is_drug_discontinued(dname):
                continue

            dose_unit = drug.get("dose_unit", "mg")
            dose_value = float(drug.get("dose_value", 0))
            dose_level = self.dose_levels.get(dname, 1.0)

            if dose_unit == "mg/kg":
                actual_dose = dose_value * self.weight_kg * dose_level
            elif dose_unit == "mg/m2":
                bsa = 0.007184 * (self.weight_kg ** 0.425) * ((self.demographics.get("height_cm", 170)) ** 0.725)
                actual_dose = dose_value * bsa * dose_level
            else:
                actual_dose = dose_value * dose_level

            actual_dose = round(actual_dose, 1)
            self.cumulative_doses[dname] = self.cumulative_doses.get(dname, 0) + actual_dose
            self.last_admin_day[dname] = day
            # Note: effective_treatment_days is tracked in generate_day() Step 0
            # using TUMOR_DAILY_RATE_* constants. No double-counting here.

            # IO 약물 투여 횟수 추적 (Pembrolizumab 35-cycle 제한)
            if dname in self.io_drugs:
                self.io_admin_count[dname] = self.io_admin_count.get(dname, 0) + 1
                if self.io_admin_count[dname] >= IO_MAX_CYCLES:
                    self.discontinued_drugs.add(dname)
                    self._log_event("io_cycle_limit_reached", day=day, drug=dname,
                                    cycles=self.io_admin_count[dname])

    def _enrich_ec_records(self, result: dict, day: int, cycle_day: int, is_hospital: bool):
        """투약 실시 기록(EC)을 result에 추가한다. 약물 상태는 이미 업데이트 완료."""
        ec_records = []
        if self.treatment_discontinued or self.is_deceased:
            result["ec_records"] = ec_records
            return

        for drug in self.admin_schedule:
            dname = drug["drug_name"]
            if cycle_day not in drug.get("cycle_days", [1]):
                continue
            if self.is_drug_held(dname) or self.is_drug_discontinued(dname):
                continue

            dose_unit = drug.get("dose_unit", "mg")
            dose_value = float(drug.get("dose_value", 0))
            dose_level = self.dose_levels.get(dname, 1.0)

            if dose_unit == "mg/kg":
                actual_dose = dose_value * self.weight_kg * dose_level
            elif dose_unit == "mg/m2":
                h_cm = self.demographics.get("height_cm", 170)
                bsa = 0.007184 * (self.weight_kg ** 0.425) * (h_cm ** 0.725)
                actual_dose = dose_value * bsa * dose_level
            else:
                actual_dose = dose_value * dose_level

            actual_dose = round(actual_dose, 1)
            is_dose_adjusted = dose_level < 1.0
            ec_records.append({
                "drug_name": dname,
                "ECREFID": dname,
                "ECSTDAT": day,
                "ECDSTXT": str(actual_dose),
                "ECDOSU": "mg",
                "ECROUTE": drug.get("route", "IV"),
                "ECDOSFRQ": drug.get("frequency", ""),
                "dose_level": dose_level,
                "ECDOSADJ": is_dose_adjusted,
                "ECADJ": f"Dose reduced to {dose_level*100:.0f}%" if is_dose_adjusted else "",
            })

        result["ec_records"] = ec_records

    def _capture_baseline(self, result: dict):
        """Day 1: EMR baseline을 보충 (EMR에 없던 lab만 추가)."""
        obj = result.get("objective", {})
        labs = obj.get("labs", result.get("labs", {}))
        for lab_name, data in labs.items():
            if lab_name in self.baseline_labs:
                continue
            if isinstance(data, dict):
                val = data.get("value")
            else:
                val = data
            if isinstance(val, (int, float)):
                self.baseline_labs[lab_name] = float(val)

        vitals = obj.get("vitals", result.get("vitals", {}))
        for v_name, val in vitals.items():
            if v_name in self.baseline_vitals:
                continue
            if isinstance(val, (int, float)):
                self.baseline_vitals[v_name] = float(val)

        _wt = vitals.get("weight_kg", self.weight_kg)
        self.weight_kg = float(_wt) if _wt is not None else self.weight_kg

    def _deceased_record(self, day: int, cycle: int, cycle_day: int) -> dict:
        """사망한 환자에 대한 빈 기록."""
        return {
            "patient_id": self.patient.get("patient_id"),
            "day": day,
            "cycle": cycle,
            "cycle_day": cycle_day,
            "_generation_mode": "deceased",
            "_events_summary": "DECEASED",
            "objective": {
                "location": "DECEASED",
                "treatment_status": "discontinued",
                "ecog": 5,
                "tumor": {"estimated_change_pct": 0},
                "active_aes": [],
            },
            "subjective": {"overall_awareness": "N/A", "symptoms_patient_perceives": []},
            "labs": {},
            "vitals": {},
            "cm_records": self._get_active_cm_records(day),
            "_mortality_risk": 0,
            "_mortality_channels": {},
        }

    # ═══════════════════════════════════════════════════════
    # Dose Modification (병원 방문 시 HR 기반)
    # ═══════════════════════════════════════════════════════

    def apply_hospital_dose_modifications(self, observed_aes: list[dict],
                                           day: int, cycle: int, cycle_day: int) -> list[dict]:
        """병원 방문 시 관찰된 AE에 기반하여 용량 조절을 수행한다.

        Args:
            observed_aes: Hospital Record에서 관찰된 AE 목록
            day, cycle, cycle_day: 현재 날짜 정보

        Returns:
            실행된 변경 목록
        """
        if self.treatment_discontinued:
            return []

        changes: list[dict] = []

        # Hold release 체크: 보류 원인 AE가 더 이상 활성이 아니면 해제
        observed_terms = {ae.get("ae", ae.get("ae_term", "")).lower() for ae in observed_aes}
        # 관찰된 AE 중 G2+ (hold 사유가 될 수 있는 등급)
        observed_g2_plus = {
            ae.get("ae", ae.get("ae_term", "")).lower()
            for ae in observed_aes if ae.get("grade", 0) >= 2
        }

        for drug in list(self.held_drugs):
            if self.is_drug_discontinued(drug):
                continue
            reason = self.hold_reasons.get(drug, "")
            should_release = False

            if reason and reason.lower() not in observed_terms:
                # 원인 AE가 완전히 사라짐 → 해제
                should_release = True
            elif reason and reason.lower() in observed_terms and reason.lower() not in observed_g2_plus:
                # 원인 AE가 존재하지만 G1 이하로 개선됨 → 해제
                should_release = True
            elif not reason:
                # 원인이 기록되지 않았는데, 활성 G2+ AE가 없으면 해제
                if not observed_g2_plus:
                    should_release = True

            if should_release:
                # 추가 안전 체크: 이 약물의 인과 AE 중 G2+ 가 아직 관찰되면 해제 보류
                other_g2_blocking = False
                for ae in observed_aes:
                    ae_g = ae.get("grade", 0)
                    if ae_g < 2:
                        continue
                    ae_t = ae.get("ae", ae.get("ae_term", ""))
                    if drug in self._get_causative_drugs(ae_t):
                        other_g2_blocking = True
                        self.hold_reasons[drug] = ae_t  # hold 사유를 현재 G2+ AE로 갱신
                        break

                if other_g2_blocking:
                    continue  # G2+ AE 잔존 → hold 유지

                self.held_drugs.discard(drug)
                if drug in self.hold_reasons:
                    del self.hold_reasons[drug]
                changes.append({
                    "action": "RESUMED",
                    "drug": drug,
                    "reason": f"Hold AE resolved/improved: {reason or 'unknown'}",
                    "day": day,
                })
                self._log_event("hold_release", day=day, drug=drug, reason=reason)

        # 기존 ae_dose_actions 리셋 (이전 방문의 귀인 초기화)
        self.ae_dose_actions.clear()

        # 안전 규칙: G4+ AE(생명위협)가 있으면 모든 약물 hold
        max_observed_grade = max(
            (ae.get("grade", 0) for ae in observed_aes), default=0
        )
        if max_observed_grade >= 4:
            g4_ae = next(ae for ae in observed_aes if ae.get("grade", 0) >= 4)
            g4_term = g4_ae.get("ae", g4_ae.get("ae_term", ""))
            for dname in self.all_drug_names:
                if not self.is_drug_held(dname) and not self.is_drug_discontinued(dname):
                    self.held_drugs.add(dname)
                    self.hold_reasons[dname] = g4_term
                    self.ae_dose_actions[g4_term] = "DRUG INTERRUPTED"
                    changes.append({
                        "action": "DRUG INTERRUPTED",
                        "drug": dname,
                        "reason": f"Safety hold: {g4_term} G{g4_ae.get('grade')} (AESLIFE)",
                        "day": day,
                    })
                    self._log_event("safety_hold", day=day, drug=dname, ae=g4_term)

        # ── FDA PI 기반 영구 중단 규칙 (일반 규칙보다 우선) ──
        already_dc_by_perm_rule: set[str] = set()
        for ae in observed_aes:
            ae_term = ae.get("ae", ae.get("ae_term", ""))
            grade = ae.get("grade", 1)
            if grade < 1:
                continue
            normalized = ae_term.lower().replace(" ", "_").replace("-", "_")

            for pdc_rule in PERMANENT_DC_RULES:
                pattern = pdc_rule["ae_pattern"]
                if pattern not in normalized and normalized not in pattern:
                    continue
                if grade < pdc_rule["grade_threshold"]:
                    continue
                if pdc_rule.get("recurrence_required") and self.ae_recurrence_count.get(ae_term, 0) < 1:
                    continue

                scope = pdc_rule["drug_scope"]
                target_drugs: set[str] = set()
                if scope == "non_io":
                    target_drugs = set(self.non_io_drugs)
                elif scope == "io":
                    target_drugs = set(self.io_drugs)
                elif scope == "all":
                    target_drugs = set(self.all_drug_names)

                for dname in target_drugs:
                    if dname in already_dc_by_perm_rule or self.is_drug_discontinued(dname):
                        continue
                    self.discontinued_drugs.add(dname)
                    self.held_drugs.discard(dname)
                    self.ae_dose_actions[ae_term] = "DRUG WITHDRAWN"
                    already_dc_by_perm_rule.add(dname)
                    changes.append({
                        "action": "DRUG WITHDRAWN",
                        "drug": dname,
                        "reason": f"Permanent d/c: {ae_term} G{grade} ({pdc_rule['note']})",
                        "day": day,
                    })
                    self._log_event("permanent_dc", day=day, drug=dname, ae=ae_term, grade=grade)
                break

        # ── 각 관찰된 AE에 대해 dose modification 규칙 적용 ──
        for ae in observed_aes:
            ae_term = ae.get("ae", ae.get("ae_term", ""))
            grade = ae.get("grade", 1)
            if grade < 1:
                continue

            rule = self.dose_mod_rules.get(ae_term.lower()) or self.dose_mod_rules.get("default") or {}
            grade_actions = rule.get("grade_actions") or {}
            action = grade_actions.get(str(grade), "DOSE NOT CHANGED")

            if action == "DOSE NOT CHANGED":
                continue

            causative = self._get_causative_drugs(ae_term)

            for drug_name in causative:
                if self.is_drug_discontinued(drug_name):
                    continue

                is_io = drug_name in self.io_drugs
                effective_action = action
                if is_io and action == "DOSE REDUCED":
                    effective_action = "DRUG INTERRUPTED"

                if effective_action == "DOSE REDUCED":
                    levels = rule.get("dose_reduction_levels", list(PADCEV_DOSE_REDUCTION_LEVELS))
                    current_level = self.dose_levels.get(drug_name, 1.0)
                    next_level = None
                    for lvl in levels:
                        if lvl < current_level:
                            next_level = lvl
                            break
                    if next_level is not None:
                        self.dose_levels[drug_name] = next_level
                        self.dose_reduction_count += 1
                        self.ae_dose_actions[ae_term] = "DOSE REDUCED"
                        changes.append({
                            "action": "DOSE REDUCED",
                            "drug": drug_name,
                            "from_level": current_level,
                            "to_level": next_level,
                            "reason": f"{ae_term} G{grade}",
                            "day": day,
                        })
                        self._log_event("dose_reduction", day=day, drug=drug_name,
                                        level=next_level, ae=ae_term)
                    else:
                        self.discontinued_drugs.add(drug_name)
                        self.held_drugs.discard(drug_name)
                        self.ae_dose_actions[ae_term] = "DRUG WITHDRAWN"
                        changes.append({
                            "action": "DRUG WITHDRAWN",
                            "drug": drug_name,
                            "reason": f"Max dose reductions exceeded for {ae_term} G{grade}",
                            "day": day,
                        })

                elif effective_action == "DRUG INTERRUPTED":
                    # AEACN: 이 AE가 hold를 야기했음 (이미 hold 중이더라도 기록)
                    if ae_term not in self.ae_dose_actions:
                        self.ae_dose_actions[ae_term] = "DRUG INTERRUPTED"

                    if not self.is_drug_held(drug_name):
                        self.held_drugs.add(drug_name)
                        self.hold_reasons[drug_name] = ae_term

                        # 감량 결정: G3+ 또는 재발 AE → hold + 1단계 감량 (IO 제외)
                        recurrence = self.ae_recurrence_count.get(ae_term, 0)
                        should_reduce = (grade >= 3 or recurrence > 0) and not is_io

                        if should_reduce:
                            levels = rule.get("dose_reduction_levels",
                                              list(PADCEV_DOSE_REDUCTION_LEVELS))
                            current_level = self.dose_levels.get(drug_name, 1.0)
                            for lvl in levels:
                                if lvl < current_level:
                                    self.dose_levels[drug_name] = lvl
                                    self.dose_reduction_count += 1
                                    break

                        reason_detail = f"{ae_term} G{grade}"
                        if recurrence > 0:
                            reason_detail += f" (recurrence #{recurrence})"
                        changes.append({
                            "action": "DRUG INTERRUPTED",
                            "drug": drug_name,
                            "reason": reason_detail,
                            "day": day,
                        })
                        self._log_event("dose_hold", day=day, drug=drug_name, ae=ae_term)

                elif action == "DRUG WITHDRAWN":
                    self.discontinued_drugs.add(drug_name)
                    self.held_drugs.discard(drug_name)
                    self.ae_dose_actions[ae_term] = "DRUG WITHDRAWN"
                    changes.append({
                        "action": "DRUG WITHDRAWN",
                        "drug": drug_name,
                        "reason": f"{ae_term} G{grade}",
                        "day": day,
                    })
                    self._log_event("drug_withdrawn", day=day, drug=drug_name, ae=ae_term)

        # 전체 치료 중단 체크
        if self.all_drug_names and self.discontinued_drugs >= self.all_drug_names:
            if not self.discontinuation_day:
                self.discontinuation_day = day
                self.ds_record = {
                    "DSDECOD": "ADVERSE EVENT",
                    "DSTERM": "All study drugs withdrawn due to toxicity",
                    "DSSTDTC": day,
                }
                self._log_event("treatment_discontinued", day=day)

        return changes

    def patch_day_treatment_status(self, day_result: dict):
        """day_result의 treatment_status를 현재 상태로 갱신한다."""
        day_result.setdefault("objective", {})["treatment_status"] = self._get_treatment_status()
        for drug in self.admin_schedule:
            dname = drug["drug_name"]
            if dname in day_result.get("objective", {}):
                day_result["objective"][dname]["treatment_held"] = self.is_drug_held(dname)
                day_result["objective"][dname]["treatment_discontinued"] = self.is_drug_discontinued(dname)
                day_result["objective"][dname]["dose_level"] = self.dose_levels.get(dname, 1.0)
