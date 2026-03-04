"""
Statistical Comparator: Simulation vs Published Trial Data

Compares extracted simulation statistics against reference trial data
and produces a structured comparison with statistical tests.

Usage:
    python validation/compare.py <sim_stats.json> <reference.json> [-o report.json]
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats


# ─── Helpers ─────────────────────────────────────────────────

def _pct_diff(sim: float | None, ref: float | None) -> float | None:
    """Absolute percentage point difference."""
    if sim is None or ref is None:
        return None
    return round(sim - ref, 2)


def _relative_diff(sim: float | None, ref: float | None) -> float | None:
    """Relative percentage difference: (sim - ref) / ref * 100."""
    if sim is None or ref is None or ref == 0:
        return None
    return round((sim - ref) / ref * 100, 1)


def _proportion_ci(count: int, n: int,
                    confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = count / n
    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    lo = max(0, center - spread)
    hi = min(1, center + spread)
    return (round(lo * 100, 2), round(hi * 100, 2))


def _proportion_test(sim_pct: float, sim_n: int,
                     ref_pct: float, ref_n: int) -> dict:
    """Two-proportion z-test. Returns z-stat, p-value, and verdict."""
    p1 = sim_pct / 100
    p2 = ref_pct / 100
    n1 = sim_n
    n2 = ref_n

    if n1 == 0 or n2 == 0:
        return {"z": None, "p_value": None, "verdict": "insufficient_data"}

    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool == 0 or p_pool == 1:
        return {"z": 0, "p_value": 1.0, "verdict": "PASS"}

    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"z": 0, "p_value": 1.0, "verdict": "PASS"}

    z = (p1 - p2) / se
    p_val = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

    if p_val >= 0.05:
        verdict = "PASS"  # Not significantly different
    elif p_val >= 0.01:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"

    return {
        "z": round(z, 3),
        "p_value": round(p_val, 4),
        "verdict": verdict,
    }


def _grade_band(sim_pct: float, ref_pct: float,
                tolerance_abs: float = 10.0,
                tolerance_rel: float = 0.30) -> str:
    """Assign a concordance grade based on distance from reference.

    Grades:
        A: Within ±5 pp or ±15% relative
        B: Within ±10 pp or ±30% relative
        C: Within ±15 pp or ±50% relative
        D: Outside all bands
    """
    abs_diff = abs(sim_pct - ref_pct)
    rel_diff = abs(sim_pct - ref_pct) / max(ref_pct, 0.1) * 100

    if abs_diff <= 5 or rel_diff <= 15:
        return "A"
    elif abs_diff <= 10 or rel_diff <= 30:
        return "B"
    elif abs_diff <= 15 or rel_diff <= 50:
        return "C"
    else:
        return "D"


# ─── AE Name Mapping ────────────────────────────────────────

# Maps simulation AE terms to reference grouped terms
AE_TERM_MAP = {
    "rash_maculopapular": "rash",
    "rash": "rash",
    "dermatitis": "rash",
    "erythema": "rash",
    "skin_toxicity": "rash",
    "peripheral_neuropathy": "peripheral_neuropathy",
    "peripheral_sensory_neuropathy": "peripheral_neuropathy",
    "fatigue": "fatigue",
    "asthenia": "fatigue",
    "pruritus": "pruritus",
    "diarrhea": "diarrhea",
    "alopecia": "alopecia",
    "decreased_appetite": "decreased_appetite",
    "anorexia": "decreased_appetite",
    "nausea": "nausea",
    "hyperglycemia": "hyperglycemia",
    "pneumonitis": "pneumonitis",
    "stomatitis": "stomatitis",
    "anemia": "anemia",
    "colitis": "colitis",
    "infusion_related_reaction": "infusion_related_reaction",
    "neutropenia": "neutropenia",
    "constipation": "constipation",
}


def _map_ae_terms(sim_ae_stats: dict) -> dict:
    """Group simulation AE terms into reference categories."""
    grouped = {}
    for term, info in sim_ae_stats.items():
        mapped = AE_TERM_MAP.get(term, term)
        if mapped not in grouped:
            grouped[mapped] = {
                "n_patients": 0,
                "all_grade_pct": 0,
                "grade_gte3_pct": 0,
                "source_terms": [],
            }
        g = grouped[mapped]
        g["n_patients"] += info["n_patients"]
        # For grouped terms, we take the max of individual incidences
        # (since a patient can only be counted once)
        g["all_grade_pct"] = max(g["all_grade_pct"], info["all_grade_pct"])
        g["grade_gte3_pct"] = max(g["grade_gte3_pct"], info["grade_gte3_pct"])
        g["source_terms"].append(term)
    return grouped


# ─── Comparison Functions ────────────────────────────────────

def compare_demographics(sim: dict, ref: dict) -> dict:
    """Compare demographic distributions."""
    comparisons = []

    # Age
    sim_age = sim.get("age_median")
    ref_age = ref.get("age_median")
    comparisons.append({
        "metric": "Median Age",
        "sim_value": sim_age,
        "ref_value": ref_age,
        "diff": _pct_diff(sim_age, ref_age),
        "unit": "years",
        "grade": "A" if sim_age and ref_age and abs(sim_age - ref_age) <= 3
                 else "B" if sim_age and ref_age and abs(sim_age - ref_age) <= 5
                 else "C",
    })

    # Sex (male %)
    sim_male = sim.get("sex_male_pct")
    ref_male = ref.get("sex_male_pct")
    n_sim = sim.get("n_patients", 50)
    n_ref = 442  # EV-302
    test = _proportion_test(sim_male or 0, n_sim, ref_male or 0, n_ref)
    comparisons.append({
        "metric": "Male %",
        "sim_value": sim_male,
        "ref_value": ref_male,
        "diff_pp": _pct_diff(sim_male, ref_male),
        "grade": _grade_band(sim_male or 0, ref_male or 0),
        "test": test,
    })

    # ECOG PS distribution
    sim_ecog = sim.get("ecog_ps", {})
    ref_ecog = ref.get("ecog_ps", {})
    for ps in ["0", "1", "2"]:
        s = sim_ecog.get(ps, 0)
        r = ref_ecog.get(ps, 0)
        comparisons.append({
            "metric": f"ECOG PS {ps} %",
            "sim_value": s,
            "ref_value": r,
            "diff_pp": _pct_diff(s, r),
            "grade": _grade_band(s, r),
        })

    # Race distribution
    sim_race = sim.get("race", {})
    ref_race = ref.get("race", {})
    race_mappings = [
        ("White", ["WHITE", "White"]),
        ("Asian", ["ASIAN", "Asian"]),
        ("Black", ["BLACK", "Black", "BLACK OR AFRICAN AMERICAN"]),
    ]
    for label, sim_keys in race_mappings:
        s = 0
        for k in sim_keys:
            s += sim_race.get(k, 0)
        r = ref_race.get(label, 0)
        if r > 0 or s > 0:
            comparisons.append({
                "metric": f"Race {label} %",
                "sim_value": round(s, 1),
                "ref_value": r,
                "diff_pp": _pct_diff(s, r),
                "grade": _grade_band(s, r),
            })

    return {"comparisons": comparisons}


def compare_ae_rates(sim: dict, ref_safety: dict, n_sim: int, n_ref: int,
                     ref_ae_grouped: dict | None = None) -> dict:
    """Compare AE incidence rates.

    Args:
        sim: Simulation ae_statistics dict (has ae_by_term, any_ae_pct, etc.)
        ref_safety: Reference safety_overall dict
        ref_ae_grouped: Reference ae_rates_grouped dict (per-AE incidence)
    """

    # Overall safety
    overall = []
    overall_metrics = [
        ("Any AE %", "any_ae_pct", "any_teae_pct"),
        ("Grade ≥3 AE %", "grade_gte3_pct", "grade_gte3_teae_pct"),
        ("SAE %", "sae_pct", "sae_pct"),
        ("Fatal AE %", "fatal_ae_pct", "fatal_teae_pct"),
        ("AE → Discontinuation %", "ae_leading_to_discontinuation_pct",
         "ae_leading_to_discontinuation_pct"),
        ("AE → Dose Reduction %", "ae_leading_to_dose_reduction_pct",
         "ae_leading_to_dose_reduction_pct"),
    ]

    for label, sim_key, ref_key in overall_metrics:
        s = sim.get(sim_key)
        r = ref_safety.get(ref_key)
        if s is not None and r is not None:
            test = _proportion_test(s, n_sim, r, n_ref)
            overall.append({
                "metric": label,
                "sim_pct": s,
                "ref_pct": r,
                "diff_pp": _pct_diff(s, r),
                "grade": _grade_band(s, r),
                "test": test,
            })

    # Per-AE comparison
    sim_by_term = sim.get("ae_by_term", {})
    ref_grouped = ref_ae_grouped or {}

    # Map sim terms to grouped terms
    grouped_sim = _map_ae_terms(sim_by_term)

    per_ae = []
    matched_ref = set()
    for ref_term, ref_info in sorted(ref_grouped.items()):
        if ref_term.startswith("_"):
            continue
        ref_all = ref_info.get("all_grade_pct", 0)
        ref_g3 = ref_info.get("grade_gte3_pct", 0)

        sim_info = grouped_sim.get(ref_term)
        if sim_info:
            sim_all = sim_info["all_grade_pct"]
            sim_g3 = sim_info["grade_gte3_pct"]
            matched_ref.add(ref_term)
        else:
            sim_all = 0
            sim_g3 = 0

        test_all = _proportion_test(sim_all, n_sim, ref_all, n_ref)
        test_g3 = _proportion_test(sim_g3, n_sim, ref_g3, n_ref) if ref_g3 > 0 else None

        per_ae.append({
            "ae_term": ref_term,
            "sim_all_grade_pct": sim_all,
            "ref_all_grade_pct": ref_all,
            "diff_all_pp": _pct_diff(sim_all, ref_all),
            "grade_all": _grade_band(sim_all, ref_all),
            "test_all": test_all,
            "sim_grade3_pct": sim_g3,
            "ref_grade3_pct": ref_g3,
            "diff_g3_pp": _pct_diff(sim_g3, ref_g3),
            "grade_g3": _grade_band(sim_g3, ref_g3) if ref_g3 > 0 else "N/A",
            "test_g3": test_g3,
        })

    # Find sim AEs not in reference (unexpected)
    unexpected = []
    for term, info in grouped_sim.items():
        if term not in ref_grouped and info["all_grade_pct"] >= 5:
            unexpected.append({
                "ae_term": term,
                "sim_all_grade_pct": info["all_grade_pct"],
                "note": "Present in simulation but not in reference top AEs",
            })

    return {
        "overall_safety": overall,
        "per_ae": per_ae,
        "unexpected_aes": unexpected,
    }


def compare_efficacy(sim: dict, ref: dict, n_sim: int, n_ref: int) -> dict:
    """Compare efficacy endpoints."""
    comparisons = []

    metrics = [
        ("ORR %", "orr_pct", "orr_pct"),
        ("CR %", "cr_pct", "cr_pct"),
        ("PR %", "pr_pct", "pr_pct"),
    ]
    for label, sim_key, ref_key in metrics:
        s = sim.get(sim_key)
        r = ref.get(ref_key)
        if s is not None and r is not None:
            test = _proportion_test(s, n_sim, r, n_ref)
            comparisons.append({
                "metric": label,
                "sim_pct": s,
                "ref_pct": r,
                "diff_pp": _pct_diff(s, r),
                "grade": _grade_band(s, r),
                "sim_95ci": list(_proportion_ci(
                    round(s / 100 * n_sim), n_sim)),
                "ref_95ci": ref.get(f"{ref_key.replace('_pct', '_95ci')}"),
                "test": test,
            })

    return {"comparisons": comparisons}


def compare_treatment_exposure(sim: dict, ref: dict) -> dict:
    """Compare treatment exposure metrics."""
    comparisons = []

    sim_dur = sim.get("median_treatment_duration_days")
    ref_dur_months = ref.get("median_duration_any_drug_months")
    if sim_dur is not None and ref_dur_months is not None:
        ref_dur_days = ref_dur_months * 30.44  # approx days/month
        comparisons.append({
            "metric": "Median Treatment Duration",
            "sim_value_days": sim_dur,
            "ref_value_months": ref_dur_months,
            "ref_value_days_approx": round(ref_dur_days, 0),
            "diff_days": round(sim_dur - ref_dur_days, 0),
            "grade": _grade_band(
                sim_dur / 30.44, ref_dur_months,
                tolerance_abs=2, tolerance_rel=0.25),
        })

    expo_metrics = [
        ("Dose Reduction %", "dose_reduction_pct", "dose_reduction_pct"),
        ("Dose Interruption %", "dose_interruption_pct",
         "dose_interruption_pct"),
        ("Discontinuation %", "discontinuation_pct",
         "discontinuation_any_drug_pct"),
    ]
    for label, sim_key, ref_key in expo_metrics:
        s = sim.get(sim_key)
        r = ref.get(ref_key)
        if s is not None and r is not None:
            comparisons.append({
                "metric": label,
                "sim_pct": s,
                "ref_pct": r,
                "diff_pp": _pct_diff(s, r),
                "grade": _grade_band(s, r),
            })

    return {"comparisons": comparisons}


def compare_lab_abnormalities(sim: dict, ref: dict) -> dict:
    """Compare lab abnormality rates."""
    sim_labs = sim.get("lab_abnormalities", {})
    ref_labs = ref  # direct dict

    # Map sim lab names to ref lab names
    lab_map = {
        "AST": "AST_increased",
        "creatinine": "creatinine_increased",
        "glucose_fasting": "glucose_increased",
        "ALT": "ALT_increased",
        "hemoglobin": "hemoglobin_decreased",
        "sodium": "sodium_decreased",
        "albumin": "albumin_decreased",
        "ANC": "neutrophils_decreased",
        "potassium": "potassium_decreased",
    }

    comparisons = []
    for sim_name, ref_name in lab_map.items():
        s_info = sim_labs.get(sim_name)
        r_info = ref_labs.get(ref_name)
        if not s_info or not r_info:
            continue

        s_pct = s_info.get("any_abnormal_pct", 0)
        r_pct = r_info.get("all_grade_pct", 0)

        comparisons.append({
            "lab": ref_name,
            "sim_lab_name": sim_name,
            "sim_any_abnormal_pct": s_pct,
            "ref_all_grade_pct": r_pct,
            "diff_pp": _pct_diff(s_pct, r_pct),
            "grade": _grade_band(s_pct, r_pct),
            "sim_grade3_pct": s_info.get("grade_gte3_pct", 0),
            "ref_grade3_pct": r_info.get("grade_gte3_pct", 0),
        })

    return {"comparisons": comparisons}


# ─── Master Comparison ───────────────────────────────────────

def compute_overall_score(report: dict) -> dict:
    """Compute an overall concordance score from all comparisons."""
    grade_points = {"A": 4, "B": 3, "C": 2, "D": 1, "N/A": None}
    grades = []
    verdicts = []

    for section_name in ["demographics", "ae_rates", "efficacy",
                         "treatment_exposure", "lab_abnormalities"]:
        section = report.get(section_name, {})
        items = (section.get("comparisons", []) +
                 section.get("overall_safety", []) +
                 section.get("per_ae", []))
        for item in items:
            g = item.get("grade") or item.get("grade_all")
            if g and g != "N/A":
                grades.append(grade_points.get(g, 0))
            t = item.get("test", {})
            if isinstance(t, dict) and t.get("verdict"):
                verdicts.append(t["verdict"])

    grade_counts = {
        "A": sum(1 for g in grades if g == 4),
        "B": sum(1 for g in grades if g == 3),
        "C": sum(1 for g in grades if g == 2),
        "D": sum(1 for g in grades if g == 1),
    }
    mean_score = np.mean(grades) if grades else 0

    verdict_counts = {
        "PASS": sum(1 for v in verdicts if v == "PASS"),
        "MARGINAL": sum(1 for v in verdicts if v == "MARGINAL"),
        "FAIL": sum(1 for v in verdicts if v == "FAIL"),
    }

    # Overall rating
    if mean_score >= 3.5:
        overall = "EXCELLENT"
    elif mean_score >= 3.0:
        overall = "GOOD"
    elif mean_score >= 2.5:
        overall = "ACCEPTABLE"
    elif mean_score >= 2.0:
        overall = "POOR"
    else:
        overall = "FAIL"

    return {
        "overall_rating": overall,
        "mean_grade_score": round(mean_score, 2),
        "grade_distribution": grade_counts,
        "total_comparisons": len(grades),
        "statistical_tests": verdict_counts,
        "total_tests": len(verdicts),
    }


def run_comparison(sim_stats: dict, reference: dict) -> dict:
    """Run full comparison and return structured report."""
    meta = reference.get("_meta", {})
    sample_ref = reference.get("sample_size", {})
    n_sim = sim_stats.get("n_simulated", sim_stats.get("n_patients", 0))
    n_ref = sample_ref.get("safety_population",
                           sample_ref.get("randomized_itt", 442))

    report = {
        "meta": {
            "trial_id": meta.get("trial_id"),
            "trial_alias": meta.get("trial_alias"),
            "drug_name": meta.get("drug_name"),
            "sim_run_id": sim_stats.get("run_id"),
            "sim_mode": sim_stats.get("mode"),
            "n_sim_patients": n_sim,
            "n_ref_patients": n_ref,
        },
        "demographics": compare_demographics(
            sim_stats.get("demographics", {}),
            reference.get("demographics", {}),
        ),
        "ae_rates": compare_ae_rates(
            sim_stats.get("ae_statistics", {}),
            reference.get("safety_overall", {}),
            n_sim, n_ref,
            ref_ae_grouped=reference.get("ae_rates_grouped", {}),
        ),
        "efficacy": compare_efficacy(
            sim_stats.get("efficacy", {}),
            reference.get("efficacy", {}),
            n_sim, n_ref,
        ),
        "treatment_exposure": compare_treatment_exposure(
            sim_stats.get("treatment_exposure", {}),
            reference.get("treatment_exposure", {}),
        ),
        "lab_abnormalities": compare_lab_abnormalities(
            sim_stats.get("lab_abnormalities", {}),
            reference.get("lab_abnormalities", {}),
        ),
    }

    report["overall_score"] = compute_overall_score(report)

    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare simulation stats against reference trial data")
    parser.add_argument("sim_stats", help="Path to simulation stats JSON")
    parser.add_argument("reference", help="Path to reference trial JSON")
    parser.add_argument("--output", "-o", default=None,
                        help="Output comparison report JSON")
    args = parser.parse_args()

    with open(args.sim_stats) as f:
        sim = json.load(f)
    with open(args.reference) as f:
        ref = json.load(f)

    report = run_comparison(sim, ref)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.sim_stats).parent / "comparison_report.json"

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Comparison report written to {out_path}")


if __name__ == "__main__":
    main()
