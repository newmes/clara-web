"""Evaluator — Natural vs Care AI 비교 평가

같은 환자 프로필에 대한 두 시뮬레이션 결과를 비교하여
Care AI의 효과를 정량적으로 평가한다.

데이터 소스:
  GT (Ground Truth):
    d["AE"]  — CDASH 형식 AE 배열 (AETERM, _grade, _status, AESTDAT, ...)
    d["objective"]  — ecog, treatment_status, location, tumor, drug info
    d["DS"]  — 중도탈락 기록 (DSDECOD, DSSTDAT)

  HR (Hospital Record):
    d["hospital_record"]["objective"]["active_aes"]
      — {ae, grade, onset_day, detected_day, detection_delay, channel, status}

  Care AI:
    d["care_record"]  — [{day, turns, actions, detection, ...}]

평가 지표:
  - AE Detection Delay: GT onset → HR 최초 감지까지 일수
  - Grade 3+ Prevention: G3+ 도달 AE 건수
  - AE Burden: 총 (grade × days)
  - Treatment Duration: 치료 유지 일수
  - Survival / Discontinuation
  - ECOG Trajectory: 일별 ECOG 변화
  - Dose Modification Timeliness
  - Care AI Activity: 개입 횟수, 유형, 감지 수
  - 통계 검증: Wilcoxon signed-rank, Bootstrap CI
"""

import json
import math
import random
from pathlib import Path
from typing import Any

from sim.logger import get_logger

_logger = get_logger("evaluator")


# ══════════════════════════════════════════════════════
# 1. 데이터 로드
# ══════════════════════════════════════════════════════

def load_simulation_data(sim_dir: Path, mode: str) -> dict[str, list[dict]]:
    """시뮬레이션 결과를 로드한다.

    Args:
        sim_dir: simulations/ 디렉토리 경로
        mode: "natural" | "care_ai"

    Returns:
        {patient_id: [day_result, ...]}
    """
    patients: dict[str, list[dict]] = {}
    pattern = f"*_{mode}.jsonl"

    for f in sorted(sim_dir.glob(pattern)):
        if f.stat().st_size == 0:
            continue
        pid = f.stem.replace(f"_{mode}", "")
        days = []
        for line in f.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                try:
                    days.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if days:
            patients[pid] = days

    return patients


# ══════════════════════════════════════════════════════
# 2. 환자별 메트릭 계산
# ══════════════════════════════════════════════════════

def compute_patient_metrics(pid: str, days: list[dict], mode: str) -> dict:
    """환자 1명의 시뮬레이션 결과에서 모든 지표를 계산한다."""
    if not days:
        _logger.warning(f"Empty day list for {pid}/{mode}, returning minimal metrics")
        return {"patient_id": pid, "mode": mode, "total_days": 0, "skipped": True}

    metrics: dict[str, Any] = {
        "patient_id": pid,
        "mode": mode,
        "total_days": len(days),
    }

    # ── Treatment Duration ──
    treatment_days = 0
    for d in days:
        status = d["objective"]["treatment_status"]
        if status == "on_treatment":
            treatment_days += 1
    metrics["treatment_days"] = treatment_days

    # ── Survival ──
    final_day = days[-1]
    final_location = final_day["objective"]["location"]
    metrics["deceased"] = final_location == "DECEASED"
    metrics["death_day"] = final_day["day"] if metrics["deceased"] else None

    # ── Discontinuation ──
    ds = final_day.get("DS")
    metrics["discontinued"] = ds is not None
    metrics["discontinuation_reason"] = ds["DSDECOD"] if ds else ""
    metrics["discontinuation_day"] = ds["DSSTDAT"] if ds else None

    # ── ECOG Trajectory ──
    ecog_trajectory = [d["objective"]["ecog"] for d in days]
    metrics["ecog_start"] = ecog_trajectory[0]
    metrics["ecog_end"] = ecog_trajectory[-1]
    metrics["ecog_max"] = max(ecog_trajectory)
    metrics["ecog_mean"] = round(sum(ecog_trajectory) / len(ecog_trajectory), 2)
    metrics["ecog_trajectory"] = ecog_trajectory

    # ── AE Metrics (from GT: d["AE"]) ──
    gt_aes: dict[str, dict] = {}  # ae_term → tracker
    total_ae_days = 0
    total_ae_burden = 0
    grade3plus_ae_days = 0
    grade4plus_ae_days = 0
    sae_count = 0

    for d in days:
        day_num = d["day"]
        for ae in d.get("AE", []):
            ae_term = ae["AETERM"]
            grade = ae["_grade"]
            status = ae["_status"]

            if ae_term not in gt_aes:
                gt_aes[ae_term] = {
                    "onset_day": ae["AESTDAT"],
                    "max_grade": grade,
                    "last_day": day_num,
                    "resolved": False,
                    "reached_g3": grade >= 3,
                    "grade_history": [],
                }

            info = gt_aes[ae_term]
            info["max_grade"] = max(info["max_grade"], grade)
            info["last_day"] = day_num
            info["grade_history"].append({"day": day_num, "grade": grade})
            if grade >= 3:
                info["reached_g3"] = True
            if status == "resolved":
                info["resolved"] = True
                info["resolved_day"] = day_num

            if ae.get("AESER", False) or grade >= 3:
                sae_count += 1

            total_ae_days += 1
            total_ae_burden += grade
            if grade >= 3:
                grade3plus_ae_days += 1
            if grade >= 4:
                grade4plus_ae_days += 1

    metrics["unique_ae_count"] = len(gt_aes)
    metrics["total_ae_days"] = total_ae_days
    metrics["total_ae_burden"] = total_ae_burden
    metrics["grade3plus_ae_days"] = grade3plus_ae_days
    metrics["grade4plus_ae_days"] = grade4plus_ae_days
    metrics["sae_events"] = sae_count
    metrics["ae_details"] = {
        term: {
            "onset_day": info["onset_day"],
            "max_grade": info["max_grade"],
            "duration": info["last_day"] - info["onset_day"] + 1,
            "resolved": info["resolved"],
            "reached_g3": info["reached_g3"],
        }
        for term, info in gt_aes.items()
    }

    # ── AE Detection Delay (GT onset vs HR detection) ──
    hr_first_detected: dict[str, int] = {}  # ae_term → first detected_day in HR
    for d in days:
        hr = d.get("hospital_record", {})
        hr_obj = hr.get("objective", {})
        for hr_ae in hr_obj.get("active_aes", []):
            ae_term = hr_ae["ae"]
            detected_day = hr_ae.get("detected_day")
            if detected_day is not None and ae_term not in hr_first_detected:
                hr_first_detected[ae_term] = detected_day

    detection_delays: list[dict] = []
    for ae_term, gt_info in gt_aes.items():
        gt_onset = gt_info["onset_day"]
        hr_det = hr_first_detected.get(ae_term)
        delay = (hr_det - gt_onset) if hr_det is not None else None
        detection_delays.append({
            "ae": ae_term,
            "gt_onset_day": gt_onset,
            "hr_detected_day": hr_det,
            "delay_days": delay,
            "max_grade": gt_info["max_grade"],
            "undetected": hr_det is None,
        })

    metrics["detection_delays"] = detection_delays
    detected_delays = [dd["delay_days"] for dd in detection_delays if dd["delay_days"] is not None]
    metrics["mean_detection_delay"] = (
        round(sum(detected_delays) / len(detected_delays), 1) if detected_delays else None
    )
    metrics["undetected_ae_count"] = sum(1 for dd in detection_delays if dd["undetected"])
    metrics["zero_delay_count"] = sum(1 for d in detected_delays if d == 0)
    metrics["zero_delay_rate"] = (
        round(metrics["zero_delay_count"] / len(detected_delays), 3) if detected_delays else None
    )

    # ── Dose Modification Timeliness ──
    dose_mod_days: list[int] = []
    for d in days:
        for ec in d.get("EC", []):
            if ec.get("ECDOSADJ", False):
                dose_mod_days.append(d["day"])
                break
    metrics["dose_modification_days"] = dose_mod_days
    metrics["dose_modifications_count"] = len(dose_mod_days)

    # ── Care AI Specific Metrics ──
    if mode == "care_ai":
        total_interventions = 0
        intervention_types: dict[str, int] = {}
        detections = 0
        turn_counts: list[int] = []
        early_terminations = 0
        force_hospital_count = 0

        for d in days:
            care_records = d.get("care_record", [])
            if not isinstance(care_records, list):
                care_records = [care_records]
            for cr in care_records:
                if not isinstance(cr, dict):
                    continue
                actions = cr.get("actions", [])
                for action in actions:
                    action_type = action["action"]
                    if action_type != "no_action":
                        total_interventions += 1
                        intervention_types[action_type] = intervention_types.get(action_type, 0) + 1
                    if action_type == "recommend_early_visit":
                        force_hospital_count += 1

                detection = cr.get("detection", {})
                detected = detection.get("aes_detected", [])
                detections += len(detected)

                turns = cr.get("turns", [])
                turn_counts.append(len(turns))
                if cr.get("terminated_early", False):
                    early_terminations += 1

        metrics["care_ai_interventions"] = total_interventions
        metrics["care_ai_intervention_types"] = intervention_types
        metrics["care_ai_detections"] = detections
        metrics["care_ai_total_calls"] = len(turn_counts)
        metrics["care_ai_early_terminations"] = early_terminations
        metrics["care_ai_force_hospital"] = force_hospital_count
        metrics["care_ai_mean_turns_per_call"] = (
            round(sum(turn_counts) / len(turn_counts), 1) if turn_counts else 0
        )

    return metrics


# ══════════════════════════════════════════════════════
# 3. 통계 검증
# ══════════════════════════════════════════════════════

def _wilcoxon_signed_rank(x: list[float], y: list[float]) -> dict:
    """Wilcoxon signed-rank test (paired, non-parametric).

    Returns {"statistic": ..., "p_value": ..., "n": ...}
    scipy 없이 수동 구현.
    """
    assert len(x) == len(y), f"Length mismatch: {len(x)} vs {len(y)}"
    diffs = [xi - yi for xi, yi in zip(x, y)]
    # 0인 차이 제거
    diffs = [(i, d) for i, d in enumerate(diffs) if d != 0.0]
    n = len(diffs)
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n": 0}

    # 절대값 기준 순위
    abs_diffs = [(abs(d), i, d > 0) for i, d in diffs]
    abs_diffs.sort(key=lambda x: x[0])

    # 순위 부여 (tie 처리: 평균 순위)
    ranks: list[tuple[float, bool]] = []
    i = 0
    while i < len(abs_diffs):
        j = i
        while j < len(abs_diffs) and abs_diffs[j][0] == abs_diffs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks.append((avg_rank, abs_diffs[k][2]))
        i = j

    w_plus = sum(r for r, pos in ranks if pos)
    w_minus = sum(r for r, pos in ranks if not pos)
    w = min(w_plus, w_minus)

    # 정규 근사 (n >= 10)
    mean_w = n * (n + 1) / 4
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if std_w == 0:
        return {"statistic": w, "p_value": 1.0, "n": n}

    z = (w - mean_w) / std_w
    # 양측 검정 p-value (정규 근사)
    p_value = 2 * (1 - _normal_cdf(abs(z)))

    return {"statistic": round(w, 2), "p_value": round(p_value, 4), "n": n, "z": round(z, 3)}


def _normal_cdf(x: float) -> float:
    """표준 정규분포 CDF (수동 구현)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for the mean."""
    if not values:
        return {"mean": None, "ci_lower": None, "ci_upper": None}

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = 1 - confidence
    lo_idx = int(n_bootstrap * alpha / 2)
    hi_idx = int(n_bootstrap * (1 - alpha / 2))

    return {
        "mean": round(sum(values) / n, 2),
        "ci_lower": round(means[lo_idx], 2),
        "ci_upper": round(means[hi_idx], 2),
        "confidence": confidence,
    }


# ══════════════════════════════════════════════════════
# 4. 코호트 비교
# ══════════════════════════════════════════════════════

def compare_cohorts(
    natural_data: dict[str, list[dict]],
    care_ai_data: dict[str, list[dict]],
) -> dict:
    """두 코호트의 전체 결과를 비교한다."""
    assert natural_data, "No natural simulation data"
    assert care_ai_data, "No care_ai simulation data"

    natural_metrics = []
    care_ai_metrics = []

    for pid, days in sorted(natural_data.items()):
        natural_metrics.append(compute_patient_metrics(pid, days, "natural"))

    for pid, days in sorted(care_ai_data.items()):
        care_ai_metrics.append(compute_patient_metrics(pid, days, "care_ai"))

    n_natural = len(natural_metrics)
    n_care = len(care_ai_metrics)

    def _mean(values: list) -> float:
        assert values, "Cannot compute mean of empty list"
        return sum(values) / len(values)

    def _pct(count: int, total: int) -> float:
        return count / total * 100 if total > 0 else 0.0

    # ── Treatment Duration ──
    nat_tx = [m["treatment_days"] for m in natural_metrics]
    care_tx = [m["treatment_days"] for m in care_ai_metrics]

    # ── AE Burden ──
    nat_burden = [m["total_ae_burden"] for m in natural_metrics]
    care_burden = [m["total_ae_burden"] for m in care_ai_metrics]

    # ── Detection Delay ──
    nat_delays_per_patient = [
        m["mean_detection_delay"] for m in natural_metrics
        if m["mean_detection_delay"] is not None
    ]
    care_delays_per_patient = [
        m["mean_detection_delay"] for m in care_ai_metrics
        if m["mean_detection_delay"] is not None
    ]

    # ── Grade 3+ AE Days ──
    nat_g3 = [m["grade3plus_ae_days"] for m in natural_metrics]
    care_g3 = [m["grade3plus_ae_days"] for m in care_ai_metrics]

    # ── ECOG ──
    nat_ecog_end = [m["ecog_end"] for m in natural_metrics]
    care_ecog_end = [m["ecog_end"] for m in care_ai_metrics]
    nat_ecog_delta = [m["ecog_end"] - m["ecog_start"] for m in natural_metrics]
    care_ecog_delta = [m["ecog_end"] - m["ecog_start"] for m in care_ai_metrics]

    # ── Paired tests (for patients present in both) ──
    common_pids = sorted(set(p["patient_id"] for p in natural_metrics) &
                         set(p["patient_id"] for p in care_ai_metrics))
    nat_by_pid = {m["patient_id"]: m for m in natural_metrics}
    care_by_pid = {m["patient_id"]: m for m in care_ai_metrics}

    stats = {}
    if len(common_pids) >= 5:
        paired_nat_burden = [nat_by_pid[p]["total_ae_burden"] for p in common_pids]
        paired_care_burden = [care_by_pid[p]["total_ae_burden"] for p in common_pids]
        stats["ae_burden_wilcoxon"] = _wilcoxon_signed_rank(paired_nat_burden, paired_care_burden)

        paired_nat_g3 = [nat_by_pid[p]["grade3plus_ae_days"] for p in common_pids]
        paired_care_g3 = [care_by_pid[p]["grade3plus_ae_days"] for p in common_pids]
        stats["g3plus_wilcoxon"] = _wilcoxon_signed_rank(paired_nat_g3, paired_care_g3)

        paired_nat_tx = [nat_by_pid[p]["treatment_days"] for p in common_pids]
        paired_care_tx = [care_by_pid[p]["treatment_days"] for p in common_pids]
        stats["treatment_duration_wilcoxon"] = _wilcoxon_signed_rank(paired_care_tx, paired_nat_tx)

    # ── Bootstrap CIs ──
    stats["natural_delay_ci"] = _bootstrap_ci(nat_delays_per_patient)
    stats["care_delay_ci"] = _bootstrap_ci(care_delays_per_patient)
    stats["natural_burden_ci"] = _bootstrap_ci(nat_burden)
    stats["care_burden_ci"] = _bootstrap_ci(care_burden)

    # ── Per-patient AE timeline (for charts) ──
    ae_timelines = []
    for pid in common_pids:
        nat_m = nat_by_pid[pid]
        care_m = care_by_pid[pid]
        ae_timelines.append({
            "patient_id": pid,
            "natural": {
                "ae_details": nat_m["ae_details"],
                "detection_delays": nat_m["detection_delays"],
            },
            "care_ai": {
                "ae_details": care_m["ae_details"],
                "detection_delays": care_m["detection_delays"],
            },
        })

    comparison = {
        "cohort_sizes": {"natural": n_natural, "care_ai": n_care},

        "treatment_duration": {
            "natural_mean": round(_mean(nat_tx), 1),
            "care_ai_mean": round(_mean(care_tx), 1),
        },

        "mortality": {
            "natural_deaths": sum(1 for m in natural_metrics if m["deceased"]),
            "care_ai_deaths": sum(1 for m in care_ai_metrics if m["deceased"]),
            "natural_pct": round(_pct(sum(1 for m in natural_metrics if m["deceased"]), n_natural), 1),
            "care_ai_pct": round(_pct(sum(1 for m in care_ai_metrics if m["deceased"]), n_care), 1),
        },

        "discontinuation": {
            "natural_count": sum(1 for m in natural_metrics if m["discontinued"]),
            "care_ai_count": sum(1 for m in care_ai_metrics if m["discontinued"]),
            "natural_pct": round(_pct(sum(1 for m in natural_metrics if m["discontinued"]), n_natural), 1),
            "care_ai_pct": round(_pct(sum(1 for m in care_ai_metrics if m["discontinued"]), n_care), 1),
        },

        "ae_burden": {
            "natural_mean": round(_mean(nat_burden), 1),
            "care_ai_mean": round(_mean(care_burden), 1),
            "natural_unique_aes": round(_mean([m["unique_ae_count"] for m in natural_metrics]), 1),
            "care_ai_unique_aes": round(_mean([m["unique_ae_count"] for m in care_ai_metrics]), 1),
        },

        "detection_delay": {
            "natural_mean": round(_mean(nat_delays_per_patient), 1) if nat_delays_per_patient else None,
            "care_ai_mean": round(_mean(care_delays_per_patient), 1) if care_delays_per_patient else None,
            "natural_undetected": sum(m["undetected_ae_count"] for m in natural_metrics),
            "care_ai_undetected": sum(m["undetected_ae_count"] for m in care_ai_metrics),
        },

        "severe_aes": {
            "natural_g3plus_mean": round(_mean(nat_g3), 1),
            "care_ai_g3plus_mean": round(_mean(care_g3), 1),
            "natural_g4plus_mean": round(_mean([m["grade4plus_ae_days"] for m in natural_metrics]), 1),
            "care_ai_g4plus_mean": round(_mean([m["grade4plus_ae_days"] for m in care_ai_metrics]), 1),
        },

        "ecog": {
            "natural_mean_end": round(_mean(nat_ecog_end), 2),
            "care_ai_mean_end": round(_mean(care_ecog_end), 2),
            "natural_mean_delta": round(_mean(nat_ecog_delta), 2),
            "care_ai_mean_delta": round(_mean(care_ecog_delta), 2),
        },

        "care_ai_activity": {
            "mean_interventions": round(_mean(
                [m.get("care_ai_interventions", 0) for m in care_ai_metrics]), 1),
            "mean_detections": round(_mean(
                [m.get("care_ai_detections", 0) for m in care_ai_metrics]), 1),
            "intervention_type_totals": _aggregate_intervention_types(care_ai_metrics),
            "mean_turns_per_call": round(_mean(
                [m.get("care_ai_mean_turns_per_call", 0) for m in care_ai_metrics]), 1),
            "total_early_terminations": sum(
                m.get("care_ai_early_terminations", 0) for m in care_ai_metrics),
            "total_force_hospital": sum(
                m.get("care_ai_force_hospital", 0) for m in care_ai_metrics),
        },

        # ── Deltas ──
        "deltas": {
            "treatment_duration": round(_mean(care_tx) - _mean(nat_tx), 1),
            "ae_burden": round(_mean(care_burden) - _mean(nat_burden), 1),
            "g3plus_days": round(_mean(care_g3) - _mean(nat_g3), 1),
            "detection_delay": round(
                (_mean(care_delays_per_patient) - _mean(nat_delays_per_patient)), 1
            ) if nat_delays_per_patient and care_delays_per_patient else None,
            "ecog_delta": round(_mean(care_ecog_delta) - _mean(nat_ecog_delta), 2),
            "mortality_pct": round(
                _pct(sum(1 for m in care_ai_metrics if m["deceased"]), n_care)
                - _pct(sum(1 for m in natural_metrics if m["deceased"]), n_natural), 1),
        },

        # ── Statistical Tests ──
        "statistics": stats,

        # ── Per-patient data for charts ──
        "natural_patients": natural_metrics,
        "care_ai_patients": care_ai_metrics,
        "ae_timelines": ae_timelines,
    }

    return comparison


def _aggregate_intervention_types(care_metrics: list[dict]) -> dict[str, int]:
    """모든 환자의 개입 유형을 합산한다."""
    totals: dict[str, int] = {}
    for m in care_metrics:
        for action_type, count in m.get("care_ai_intervention_types", {}).items():
            totals[action_type] = totals.get(action_type, 0) + count
    return totals


# ══════════════════════════════════════════════════════
# 5. 리포트 출력
# ══════════════════════════════════════════════════════

def print_comparison_report(comparison: dict) -> str:
    """비교 결과를 보기 좋게 출력한다."""
    lines = []
    lines.append("=" * 70)
    lines.append("A/B COMPARISON REPORT: Natural vs Care AI")
    lines.append("=" * 70)

    sizes = comparison["cohort_sizes"]
    lines.append(f"Cohort: Natural={sizes['natural']}, Care AI={sizes['care_ai']}")
    lines.append("")

    # Detection Delay
    dd = comparison["detection_delay"]
    delta_dd = comparison["deltas"]["detection_delay"]
    lines.append("AE Detection Delay (days):")
    lines.append(f"  Natural: {dd['natural_mean']}d (mean)")
    lines.append(f"  Care AI: {dd['care_ai_mean']}d (mean)")
    if delta_dd is not None:
        lines.append(f"  Delta:   {delta_dd:+.1f}d {'(faster)' if delta_dd < 0 else '(slower)'}")
    lines.append(f"  Undetected AEs: Natural={dd['natural_undetected']}, Care AI={dd['care_ai_undetected']}")
    lines.append("")

    # Treatment Duration
    td = comparison["treatment_duration"]
    delta_td = comparison["deltas"]["treatment_duration"]
    lines.append("Treatment Duration:")
    lines.append(f"  Natural: {td['natural_mean']} days")
    lines.append(f"  Care AI: {td['care_ai_mean']} days")
    lines.append(f"  Delta:   {delta_td:+.1f} days {'(longer)' if delta_td > 0 else '(shorter)'}")
    lines.append("")

    # AE Burden
    ae = comparison["ae_burden"]
    delta_ae = comparison["deltas"]["ae_burden"]
    lines.append("AE Burden (grade × days):")
    lines.append(f"  Natural: {ae['natural_mean']} (mean)")
    lines.append(f"  Care AI: {ae['care_ai_mean']} (mean)")
    lines.append(f"  Delta:   {delta_ae:+.1f} {'(less)' if delta_ae < 0 else '(more)'}")
    lines.append("")

    # Severe AEs
    sev = comparison["severe_aes"]
    delta_g3 = comparison["deltas"]["g3plus_days"]
    lines.append("Severe AE Days (G3+):")
    lines.append(f"  Natural: {sev['natural_g3plus_mean']} (mean)")
    lines.append(f"  Care AI: {sev['care_ai_g3plus_mean']} (mean)")
    lines.append(f"  Delta:   {delta_g3:+.1f} {'(fewer)' if delta_g3 < 0 else '(more)'}")
    lines.append("")

    # ECOG
    ecog = comparison["ecog"]
    lines.append("ECOG Performance Status:")
    lines.append(f"  Natural: start→end delta = {ecog['natural_mean_delta']:+.2f}")
    lines.append(f"  Care AI: start→end delta = {ecog['care_ai_mean_delta']:+.2f}")
    lines.append("")

    # Mortality / Discontinuation
    mort = comparison["mortality"]
    disc = comparison["discontinuation"]
    lines.append(f"Mortality: Natural={mort['natural_deaths']}/{sizes['natural']} ({mort['natural_pct']}%), "
                 f"Care AI={mort['care_ai_deaths']}/{sizes['care_ai']} ({mort['care_ai_pct']}%)")
    lines.append(f"Discontinued: Natural={disc['natural_count']}/{sizes['natural']} ({disc['natural_pct']}%), "
                 f"Care AI={disc['care_ai_count']}/{sizes['care_ai']} ({disc['care_ai_pct']}%)")
    lines.append("")

    # Care AI Activity
    care = comparison["care_ai_activity"]
    lines.append("Care AI Activity:")
    lines.append(f"  Mean interventions/patient: {care['mean_interventions']}")
    lines.append(f"  Mean AE detections/patient: {care['mean_detections']}")
    lines.append(f"  Mean turns/call: {care['mean_turns_per_call']}")
    lines.append(f"  Early terminations: {care['total_early_terminations']}")
    lines.append(f"  Force-hospital triggers: {care['total_force_hospital']}")
    lines.append(f"  Intervention types: {json.dumps(care['intervention_type_totals'], indent=4)}")
    lines.append("")

    # Statistical Tests
    stats = comparison.get("statistics", {})
    if stats:
        lines.append("Statistical Tests:")
        for test_name, result in stats.items():
            if isinstance(result, dict) and "p_value" in result:
                sig = "***" if result["p_value"] < 0.001 else "**" if result["p_value"] < 0.01 else "*" if result["p_value"] < 0.05 else "ns"
                lines.append(f"  {test_name}: W={result['statistic']}, p={result['p_value']} {sig} (n={result['n']})")
            elif isinstance(result, dict) and "ci_lower" in result:
                lines.append(f"  {test_name}: mean={result['mean']} [{result['ci_lower']}, {result['ci_upper']}] {int(result.get('confidence',0.95)*100)}% CI")
        lines.append("")

    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)
    return report


# ══════════════════════════════════════════════════════
# 6. 실행 엔트리포인트
# ══════════════════════════════════════════════════════

def run_evaluation(run_dir: Path) -> dict:
    """실험 디렉토리에서 Natural vs Care AI 비교를 수행한다."""
    sim_dir = run_dir / "simulations"

    natural_data = load_simulation_data(sim_dir, "natural")
    care_ai_data = load_simulation_data(sim_dir, "care_ai")

    if not natural_data:
        raise ValueError(f"No natural simulation data in {sim_dir}")
    if not care_ai_data:
        raise ValueError(f"No care_ai simulation data in {sim_dir}")

    comparison = compare_cohorts(natural_data, care_ai_data)

    # JSON 저장
    report_path = run_dir / "comparison_report.json"
    try:
        report_path.write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        _logger.info(f"Comparison report saved to {report_path}")
    except PermissionError:
        alt_path = run_dir / "comparison_report_new.json"
        alt_path.write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        _logger.warning(f"PermissionError on {report_path}, saved to {alt_path}")

    # 텍스트 저장
    report_text = print_comparison_report(comparison)
    text_path = run_dir / "comparison_report.txt"
    try:
        text_path.write_text(report_text, encoding="utf-8")
    except PermissionError:
        alt_text = run_dir / "comparison_report_new.txt"
        alt_text.write_text(report_text, encoding="utf-8")
        _logger.warning(f"PermissionError on {text_path}, saved to {alt_text}")

    return comparison