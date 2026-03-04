"""
Simulation Statistics Extractor

Reads all patient profiles and simulation JSONL files from a run directory,
and computes aggregate statistics that are directly comparable to published
clinical trial data.

Usage:
    python validation/extract_sim_stats.py <run_dir> [--mode natural|care_ai]
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_patient_profiles(run_dir: Path) -> dict[str, dict]:
    """Load all patient profile JSONs from patients/ subdirectory."""
    profiles = {}
    patients_dir = run_dir / "patients"
    if not patients_dir.exists():
        raise FileNotFoundError(f"No patients/ directory in {run_dir}")
    for f in sorted(patients_dir.glob("*.json")):
        with open(f) as fh:
            p = json.load(fh)
        pid = p.get("patient_id", f.stem)
        profiles[pid] = p
    return profiles


def load_simulation(run_dir: Path, patient_id: str,
                    mode: str = "natural") -> list[dict]:
    """Load all daily records for a patient from JSONL."""
    fname = run_dir / "simulations" / f"{patient_id}_{mode}.jsonl"
    if not fname.exists():
        return []
    records = []
    with open(fname) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_demographics(profiles: dict[str, dict]) -> dict:
    """Extract demographic statistics from patient profiles."""
    ages = []
    sex_counts = Counter()
    race_counts = Counter()
    ecog_counts = Counter()
    bmis = []

    for pid, p in profiles.items():
        dm = p.get("DM", {})
        emr = p.get("emr", {})
        demo = emr.get("demographics", {})

        age = dm.get("AGE") or demo.get("age")
        if age is not None:
            ages.append(float(age))

        sex = dm.get("SEX") or demo.get("sex", "?")
        sex_counts[sex] += 1

        race = dm.get("RACE") or demo.get("race", "Unknown")
        race_counts[race.upper() if race else "UNKNOWN"] += 1

        ecog = demo.get("ecog_ps") or emr.get("baseline_ecog")
        if ecog is not None:
            ecog_counts[str(int(ecog))] += 1

        bmi = demo.get("bmi")
        if bmi is not None:
            bmis.append(float(bmi))

    n = len(profiles)
    return {
        "n_patients": n,
        "age_median": float(np.median(ages)) if ages else None,
        "age_mean": float(np.mean(ages)) if ages else None,
        "age_std": float(np.std(ages)) if ages else None,
        "age_range": [float(min(ages)), float(max(ages))] if ages else None,
        "sex_male_pct": round(sex_counts.get("M", 0) / n * 100, 1) if n else 0,
        "sex_female_pct": round(sex_counts.get("F", 0) / n * 100, 1) if n else 0,
        "race": {
            k: round(v / n * 100, 1) for k, v in race_counts.most_common()
        },
        "ecog_ps": {
            k: round(v / n * 100, 1) for k, v in sorted(ecog_counts.items())
        },
        "bmi_median": float(np.median(bmis)) if bmis else None,
        "bmi_mean": float(np.mean(bmis)) if bmis else None,
    }


def extract_ae_statistics(all_sims: dict[str, list[dict]]) -> dict:
    """Extract AE incidence, grade distribution, onset timing, etc."""
    n_patients = len(all_sims)

    # Per-patient AE tracking
    patient_aes: dict[str, dict[str, dict]] = {}  # pid -> {ae_term -> info}

    for pid, days in all_sims.items():
        seen_aes: dict[str, dict] = {}  # ae_term -> {max_grade, onset_day, ...}

        for day_data in days:
            ae_list = day_data.get("AE", [])
            for ae in ae_list:
                term = ae.get("AETERM", "unknown")
                grade = ae.get("_grade", 0)
                onset = ae.get("AESTDAT")
                is_serious = ae.get("AESER", False)
                is_related = ae.get("AEREL", True)
                caused_death = ae.get("AESDTH", False)
                action = ae.get("AEACN", "")

                if term not in seen_aes:
                    seen_aes[term] = {
                        "max_grade": grade,
                        "onset_day": onset,
                        "is_serious": is_serious,
                        "is_related": is_related,
                        "caused_death": caused_death,
                        "led_to_discontinuation": "WITHDRAWN" in str(action),
                        "led_to_interruption": "INTERRUPT" in str(action),
                        "led_to_reduction": "REDUCED" in str(action),
                    }
                else:
                    info = seen_aes[term]
                    if grade > info["max_grade"]:
                        info["max_grade"] = grade
                    if is_serious:
                        info["is_serious"] = True
                    if caused_death:
                        info["caused_death"] = True
                    if "WITHDRAWN" in str(action):
                        info["led_to_discontinuation"] = True
                    if "INTERRUPT" in str(action):
                        info["led_to_interruption"] = True
                    if "REDUCED" in str(action):
                        info["led_to_reduction"] = True

        patient_aes[pid] = seen_aes

    # Aggregate
    ae_term_counts = Counter()  # term -> n_patients with this AE
    ae_term_grade3plus = Counter()  # term -> n_patients with grade >=3
    ae_term_grades = defaultdict(lambda: Counter())  # term -> {grade -> count}
    ae_term_onsets = defaultdict(list)  # term -> [onset_days]
    any_ae_count = 0
    grade3_plus_count = 0
    sae_count = 0
    fatal_ae_count = 0
    ae_discontinuation_count = 0
    ae_interruption_count = 0
    ae_reduction_count = 0

    for pid, seen_aes in patient_aes.items():
        if seen_aes:
            any_ae_count += 1

        patient_max_grade = 0
        patient_has_sae = False
        patient_has_fatal = False
        patient_has_discont = False
        patient_has_interrupt = False
        patient_has_reduction = False

        for term, info in seen_aes.items():
            ae_term_counts[term] += 1
            grade = info["max_grade"]
            ae_term_grades[term][grade] += 1

            if grade >= 3:
                ae_term_grade3plus[term] += 1
            if grade > patient_max_grade:
                patient_max_grade = grade
            if info["is_serious"]:
                patient_has_sae = True
            if info["caused_death"]:
                patient_has_fatal = True
            if info["led_to_discontinuation"]:
                patient_has_discont = True
            if info["led_to_interruption"]:
                patient_has_interrupt = True
            if info["led_to_reduction"]:
                patient_has_reduction = True

            if info["onset_day"] is not None:
                ae_term_onsets[term].append(info["onset_day"])

        if patient_max_grade >= 3:
            grade3_plus_count += 1
        if patient_has_sae:
            sae_count += 1
        if patient_has_fatal:
            fatal_ae_count += 1
        if patient_has_discont:
            ae_discontinuation_count += 1
        if patient_has_interrupt:
            ae_interruption_count += 1
        if patient_has_reduction:
            ae_reduction_count += 1

    # Build per-AE summary
    ae_summary = {}
    for term in sorted(ae_term_counts, key=ae_term_counts.get, reverse=True):
        count = ae_term_counts[term]
        g3_count = ae_term_grade3plus.get(term, 0)
        onsets = ae_term_onsets.get(term, [])
        grades = ae_term_grades[term]
        grade_dist = {
            str(g): round(c / count * 100, 1)
            for g, c in sorted(grades.items())
        }
        ae_summary[term] = {
            "n_patients": count,
            "all_grade_pct": round(count / n_patients * 100, 1),
            "grade_gte3_pct": round(g3_count / n_patients * 100, 1),
            "grade_distribution_pct": grade_dist,
            "median_onset_day": float(np.median(onsets)) if onsets else None,
            "mean_onset_day": round(float(np.mean(onsets)), 1) if onsets else None,
        }

    return {
        "n_patients": n_patients,
        "any_ae_pct": round(any_ae_count / n_patients * 100, 1),
        "grade_gte3_pct": round(grade3_plus_count / n_patients * 100, 1),
        "sae_pct": round(sae_count / n_patients * 100, 1),
        "fatal_ae_pct": round(fatal_ae_count / n_patients * 100, 1),
        "ae_leading_to_discontinuation_pct": round(
            ae_discontinuation_count / n_patients * 100, 1),
        "ae_leading_to_interruption_pct": round(
            ae_interruption_count / n_patients * 100, 1),
        "ae_leading_to_dose_reduction_pct": round(
            ae_reduction_count / n_patients * 100, 1),
        "ae_by_term": ae_summary,
    }


def extract_treatment_exposure(all_sims: dict[str, list[dict]]) -> dict:
    """Extract treatment duration, cycles, dose modifications."""
    n_patients = len(all_sims)

    treatment_durations = []  # days on treatment
    total_days_tracked = []
    patients_with_dose_reduction = 0
    patients_with_dose_interruption = 0
    patients_with_discontinuation = 0
    drug_cycles: dict[str, list[int]] = defaultdict(list)
    drug_durations: dict[str, list[int]] = defaultdict(list)

    for pid, days in all_sims.items():
        if not days:
            continue

        last_day = days[-1].get("day", 0)
        total_days_tracked.append(last_day)

        # Track per-drug info
        drug_admin_days: dict[str, list[int]] = defaultdict(list)
        last_treatment_day = 0
        had_reduction = False
        had_interruption = False
        had_discontinuation = False

        for day_data in days:
            d = day_data.get("day", 0)
            obj = day_data.get("objective", {})

            # Check treatment status
            ts = obj.get("treatment_status", "")
            if "held" in ts:
                had_interruption = True
            if "discontinued" in ts:
                had_discontinuation = True

            # Check each drug in objective
            for key, val in obj.items():
                if isinstance(val, dict) and "last_administered_day" in val:
                    drug_name = key
                    admin_day = val.get("last_administered_day")
                    if admin_day is not None:
                        if admin_day not in drug_admin_days[drug_name]:
                            drug_admin_days[drug_name].append(admin_day)
                        last_treatment_day = max(
                            last_treatment_day, admin_day)
                    if val.get("dose_level") is not None:
                        if val["dose_level"] < 1.0:
                            had_reduction = True

            # Check EC records for administration
            for ec in day_data.get("EC", []):
                drug_name = ec.get("ECREFID") or ec.get("EXTRT", "")
                if not drug_name:
                    continue
                admin_day_ec = ec.get("ECSTDAT", d)
                if admin_day_ec not in drug_admin_days[drug_name]:
                    drug_admin_days[drug_name].append(admin_day_ec)
                    last_treatment_day = max(
                        last_treatment_day, admin_day_ec)

        treatment_durations.append(last_treatment_day)

        if had_reduction:
            patients_with_dose_reduction += 1
        if had_interruption:
            patients_with_dose_interruption += 1
        if had_discontinuation:
            patients_with_discontinuation += 1

        # Count cycles per drug (rough: number of administrations)
        for drug, admin_days in drug_admin_days.items():
            drug_cycles[drug].append(len(admin_days))
            if admin_days:
                drug_durations[drug].append(max(admin_days) - min(admin_days))

    result = {
        "n_patients": n_patients,
        "median_treatment_duration_days": float(np.median(treatment_durations))
            if treatment_durations else None,
        "mean_treatment_duration_days": round(
            float(np.mean(treatment_durations)), 1)
            if treatment_durations else None,
        "median_total_days_tracked": float(np.median(total_days_tracked))
            if total_days_tracked else None,
        "dose_reduction_pct": round(
            patients_with_dose_reduction / n_patients * 100, 1),
        "dose_interruption_pct": round(
            patients_with_dose_interruption / n_patients * 100, 1),
        "discontinuation_pct": round(
            patients_with_discontinuation / n_patients * 100, 1),
        "per_drug": {},
    }

    for drug in sorted(drug_cycles.keys()):
        cycles = drug_cycles[drug]
        durations = drug_durations.get(drug, [])
        result["per_drug"][drug] = {
            "n_administrations_median": float(np.median(cycles)) if cycles else None,
            "n_administrations_mean": round(
                float(np.mean(cycles)), 1) if cycles else None,
            "duration_days_median": float(np.median(durations))
                if durations else None,
        }

    return result


def extract_efficacy(all_sims: dict[str, list[dict]],
                     profiles: dict[str, dict]) -> dict:
    """Extract tumor response and survival statistics."""
    n_patients = len(all_sims)

    best_responses = []  # best tumor change pct per patient
    final_statuses = Counter()
    deaths = 0
    alive_at_end = 0
    tumor_response_cats = Counter()  # CR, PR, SD, PD

    for pid, days in all_sims.items():
        if not days:
            continue

        best_change = 0.0
        final_location = "HOME"
        final_status = "on_treatment"

        for day_data in days:
            obj = day_data.get("objective", {})
            tumor = obj.get("tumor", {})
            if tumor:
                change = tumor.get("estimated_change_pct", 0)
                if change is not None and change < best_change:
                    best_change = change

            final_location = obj.get("location", final_location)
            final_status = obj.get("treatment_status", final_status)

        best_responses.append(best_change)

        if final_location == "DECEASED":
            deaths += 1
        else:
            alive_at_end += 1

        final_statuses[final_status] += 1

        # RECIST-like categorization based on best change
        # CR: near-complete disappearance (≤-90% in simulation, since
        #     sigmoid model asymptotically approaches but rarely reaches -100%)
        # PR: ≤-30% (standard RECIST threshold)
        # PD: ≥+20% (standard RECIST threshold)
        # SD: between -30% and +20%
        if best_change <= -90:
            tumor_response_cats["CR"] += 1
        elif best_change <= -30:
            tumor_response_cats["PR"] += 1
        elif best_change <= 20:
            tumor_response_cats["SD"] += 1
        else:
            tumor_response_cats["PD"] += 1

    cr = tumor_response_cats.get("CR", 0)
    pr = tumor_response_cats.get("PR", 0)
    sd = tumor_response_cats.get("SD", 0)
    pd = tumor_response_cats.get("PD", 0)
    orr = cr + pr

    return {
        "n_patients": n_patients,
        "orr_pct": round(orr / n_patients * 100, 1) if n_patients else 0,
        "cr_pct": round(cr / n_patients * 100, 1) if n_patients else 0,
        "pr_pct": round(pr / n_patients * 100, 1) if n_patients else 0,
        "sd_pct": round(sd / n_patients * 100, 1) if n_patients else 0,
        "pd_pct": round(pd / n_patients * 100, 1) if n_patients else 0,
        "best_tumor_change_median": round(
            float(np.median(best_responses)), 1)
            if best_responses else None,
        "best_tumor_change_mean": round(
            float(np.mean(best_responses)), 1)
            if best_responses else None,
        "mortality_pct": round(deaths / n_patients * 100, 1) if n_patients else 0,
        "alive_at_simulation_end_pct": round(
            alive_at_end / n_patients * 100, 1) if n_patients else 0,
        "final_treatment_status": {
            k: round(v / n_patients * 100, 1) for k, v in final_statuses.items()
        },
    }


def extract_lab_abnormalities(all_sims: dict[str, list[dict]],
                              lab_ref_ranges: dict | None = None) -> dict:
    """Extract lab abnormality rates across simulation."""

    # Default reference ranges (CTCAE-aligned)
    default_ranges = {
        "ANC": {"LLN": 1.5, "ULN": 7.5, "unit": "x10^9/L",
                "grade_thresholds": {3: 1.0, 4: 0.5}},
        "hemoglobin": {"LLN": 13.5, "ULN": 17.5, "unit": "g/dL",
                       "grade_thresholds": {3: 8.0, 4: 6.5}},
        "platelets": {"LLN": 150, "ULN": 400, "unit": "x10^9/L",
                      "grade_thresholds": {3: 50, 4: 25}},
        "creatinine": {"LLN": 0.7, "ULN": 1.3, "unit": "mg/dL",
                       "grade_thresholds": {3: 3.5, 4: 6.0},
                       "direction": "high"},
        "ALT": {"LLN": 7, "ULN": 56, "unit": "U/L",
                "grade_thresholds": {3: 280, 4: 1120},
                "direction": "high"},
        "AST": {"LLN": 10, "ULN": 40, "unit": "U/L",
                "grade_thresholds": {3: 200, 4: 800},
                "direction": "high"},
        "glucose_fasting": {"LLN": 70, "ULN": 99, "unit": "mg/dL",
                            "grade_thresholds": {3: 250, 4: 500},
                            "direction": "high"},
        "total_bilirubin": {"LLN": 0.1, "ULN": 1.2, "unit": "mg/dL",
                            "grade_thresholds": {3: 3.6, 4: 12.0},
                            "direction": "high"},
        "albumin": {"LLN": 3.4, "ULN": 5.4, "unit": "g/dL",
                    "grade_thresholds": {3: 2.0}},
        "sodium": {"LLN": 135, "ULN": 145, "unit": "mmol/L",
                   "grade_thresholds": {3: 120, 4: 115}},
        "potassium": {"LLN": 3.5, "ULN": 5.0, "unit": "mmol/L",
                      "grade_thresholds_low": {3: 2.5},
                      "grade_thresholds_high": {3: 6.0, 4: 7.0}},
    }

    n_patients = len(all_sims)

    # Map simulation lab names to reference range names
    lab_name_map = {
        "ANC": "ANC", "anc": "ANC",
        "hemoglobin": "hemoglobin", "Hemoglobin": "hemoglobin",
        "platelets": "platelets", "Platelets": "platelets",
        "creatinine": "creatinine", "Creatinine": "creatinine",
        "ALT": "ALT", "alt": "ALT",
        "AST": "AST", "ast": "AST",
        "glucose_fasting": "glucose_fasting", "glucose": "glucose_fasting",
        "total_bilirubin": "total_bilirubin",
        "albumin": "albumin",
        "sodium": "sodium",
        "potassium": "potassium",
        "TSH": "TSH", "tsh": "TSH",
        "LDH": "LDH", "ldh": "LDH",
        "HbA1c": "HbA1c",
    }

    # Track per-patient worst lab values
    patient_worst: dict[str, dict[str, float]] = defaultdict(dict)

    for pid, days in all_sims.items():
        for day_data in days:
            lb = day_data.get("LB", {})
            results = lb.get("results", lb)
            if not isinstance(results, dict):
                continue
            for lab_name, lab_info in results.items():
                if lab_name.startswith("_"):
                    continue
                val = None
                if isinstance(lab_info, dict):
                    val = lab_info.get("LBORRES") or lab_info.get("value")
                elif isinstance(lab_info, (int, float)):
                    val = lab_info
                if val is None:
                    continue
                val = float(val)

                mapped = lab_name_map.get(lab_name, lab_name)
                if mapped not in patient_worst.get(pid, {}):
                    patient_worst[pid][mapped] = val
                else:
                    ref = default_ranges.get(mapped, {})
                    direction = ref.get("direction", "low")
                    if direction == "high":
                        if val > patient_worst[pid][mapped]:
                            patient_worst[pid][mapped] = val
                    else:
                        if val < patient_worst[pid][mapped]:
                            patient_worst[pid][mapped] = val

    # Compute abnormality rates
    lab_stats = {}
    for lab_name, ref in default_ranges.items():
        uln = ref.get("ULN")
        lln = ref.get("LLN")
        direction = ref.get("direction", "low")
        g3_thresh = ref.get("grade_thresholds", {}).get(3)

        any_abnormal = 0
        grade3_plus = 0
        values_collected = 0

        for pid in all_sims:
            val = patient_worst.get(pid, {}).get(lab_name)
            if val is None:
                continue
            values_collected += 1

            if direction == "high":
                if uln and val > uln:
                    any_abnormal += 1
                if g3_thresh and val > g3_thresh:
                    grade3_plus += 1
            else:
                if lln and val < lln:
                    any_abnormal += 1
                if g3_thresh and val < g3_thresh:
                    grade3_plus += 1

        if values_collected > 0:
            lab_stats[lab_name] = {
                "n_with_data": values_collected,
                "any_abnormal_pct": round(
                    any_abnormal / values_collected * 100, 1),
                "grade_gte3_pct": round(
                    grade3_plus / values_collected * 100, 1),
            }

    return {
        "n_patients": n_patients,
        "lab_abnormalities": lab_stats,
    }


def extract_all_stats(run_dir: str | Path,
                      mode: str = "natural") -> dict:
    """Master extraction function. Returns complete stats dict."""
    run_dir = Path(run_dir)

    print(f"Loading profiles from {run_dir / 'patients'}...")
    profiles = load_patient_profiles(run_dir)
    print(f"  → {len(profiles)} patients")

    print(f"Loading simulations (mode={mode})...")
    all_sims = {}
    for pid in sorted(profiles.keys()):
        days = load_simulation(run_dir, pid, mode)
        if days:
            all_sims[pid] = days
    print(f"  → {len(all_sims)} patients with simulation data")

    # Load rule_set for reference ranges if available
    rule_set_path = run_dir / "rule_set.json"
    lab_ref = None
    if rule_set_path.exists():
        with open(rule_set_path) as f:
            rs = json.load(f)
        lab_ref = rs.get("lab_reference_ranges")

    print("Extracting demographics...")
    demographics = extract_demographics(profiles)

    print("Extracting AE statistics...")
    ae_stats = extract_ae_statistics(all_sims)

    print("Extracting treatment exposure...")
    treatment = extract_treatment_exposure(all_sims)

    print("Extracting efficacy...")
    efficacy = extract_efficacy(all_sims, profiles)

    print("Extracting lab abnormalities...")
    labs = extract_lab_abnormalities(all_sims, lab_ref)

    return {
        "run_id": run_dir.name,
        "mode": mode,
        "n_patients": len(profiles),
        "n_simulated": len(all_sims),
        "demographics": demographics,
        "ae_statistics": ae_stats,
        "treatment_exposure": treatment,
        "efficacy": efficacy,
        "lab_abnormalities": labs,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract simulation statistics for validation")
    parser.add_argument("run_dir", help="Path to run directory")
    parser.add_argument("--mode", default="natural",
                        choices=["natural", "care_ai"],
                        help="Simulation mode")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    stats = extract_all_stats(args.run_dir, args.mode)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.run_dir) / f"validation_stats_{args.mode}.json"

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nStats written to {out_path}")


if __name__ == "__main__":
    main()
