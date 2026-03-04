#!/usr/bin/env python3
"""Compare Etoposide+Cisplatin pipeline output against Ground Truth (10 dimensions)."""

import json
import re
import sys

OUTPUT_PATH = "/home/ubuntu/samuel/rule_discovery/output/cisplatin+etoposide_small_cell_lung_cancer_rules.json"
GT_PATH = "/home/ubuntu/samuel/rule_discovery/ground_truth/2_Etoposide_Cisplatin/base.json"

V14_BASELINE = 0.824  # documented EP baseline

def load_json(path):
    with open(path) as f:
        return json.load(f)

def clamp01(x):
    return max(0.0, min(1.0, x))

def normalize_ae_term(term):
    """Normalize AE term: lowercase, replace spaces with _, strip trailing _s,
    handle British/US spelling, and common synonyms."""
    t = term.lower().strip().replace(" ", "_").replace("-", "_")
    # Remove trailing _s (but not if it's a real word ending in s)
    # Actually just strip trailing _s for normalization
    t = re.sub(r'_s$', '', t)
    # British -> US spelling
    synonyms = {
        "diarrhoea": "diarrhea",
        "dyspnoea": "dyspnea",
        "anaemia": "anemia",
        "haemoglobin_decreased": "lower_hemoglobin",
        "haemoglobin": "hemoglobin",
        "paraesthesia": "paresthesia",
        "hyponatraemia": "hyponatremia",
        "hypomagnesaemia": "hypomagnesemia",
        "oedema": "edema",
        "leucopenia": "leukopenia",
        "tumour": "tumor",
        "favourite": "favorite",
        "behaviour": "behavior",
        "pyrexia": "pyrexia",
        "fever": "fever",
    }
    if t in synonyms:
        t = synonyms[t]
    return t

def parse_dose_numeric(dose_str):
    """Extract numeric value from dose string like '80 mg/m^2' or '100 mg/m2'."""
    if not dose_str:
        return None
    m = re.search(r'([\d.]+)', str(dose_str))
    if m:
        return float(m.group(1))
    return None

def score_doses(out_data, gt_data):
    """Compare doses per drug. Score = avg per-drug match."""
    # Build drug -> dose maps from administration_schedule
    out_sched = out_data.get("administration_schedule", [])
    gt_sched = gt_data.get("administration_schedule", [])

    out_doses = {}
    for entry in out_sched:
        name = entry.get("drug_name", "").lower().strip()
        dose = parse_dose_numeric(entry.get("dose_per_administration", ""))
        if name and dose is not None:
            out_doses[name] = dose

    gt_doses = {}
    for entry in gt_sched:
        name = entry.get("drug_name", "").lower().strip()
        dose = parse_dose_numeric(entry.get("dose_per_administration", ""))
        if name and dose is not None:
            gt_doses[name] = dose

    print(f"  Output doses: {out_doses}")
    print(f"  GT doses:     {gt_doses}")

    if not gt_doses:
        return 0.0

    scores = []
    for drug, gt_dose in gt_doses.items():
        if drug in out_doses:
            out_dose = out_doses[drug]
            if gt_dose == 0 and out_dose == 0:
                scores.append(1.0)
            elif gt_dose == 0:
                scores.append(0.0)
            else:
                ratio = min(out_dose, gt_dose) / max(out_dose, gt_dose)
                if ratio >= 0.9:
                    scores.append(1.0)
                else:
                    scores.append(ratio)
        else:
            scores.append(0.0)
            print(f"  WARNING: Drug '{drug}' not found in output schedule")

    return sum(scores) / len(scores) if scores else 0.0

def score_orr(out_data, gt_data):
    """Compare ORR. Score = 1 - abs(diff)/gt."""
    out_orr = out_data.get("efficacy", {}).get("overall_response_rate")
    gt_orr = gt_data.get("efficacy", {}).get("overall_response_rate")

    print(f"  Output ORR: {out_orr}")
    print(f"  GT ORR:     {gt_orr}")

    if gt_orr is None or gt_orr == 0:
        return 0.0
    if out_orr is None:
        return 0.0
    return clamp01(1.0 - abs(out_orr - gt_orr) / gt_orr)

def score_age_range(out_data, gt_data):
    """Compare age min/max. Score = 1 - (|diff_min| + |diff_max|) / (gt_max - gt_min)."""
    out_age = out_data.get("demographics", {}).get("age", {}).get("params", {})
    gt_age = gt_data.get("demographics", {}).get("age", {}).get("params", {})

    out_min = out_age.get("min")
    out_max = out_age.get("max")
    gt_min = gt_age.get("min")
    gt_max = gt_age.get("max")

    print(f"  Output age: min={out_min}, max={out_max}")
    print(f"  GT age:     min={gt_min}, max={gt_max}")

    if gt_min is None or gt_max is None or gt_max == gt_min:
        return 0.0
    if out_min is None or out_max is None:
        return 0.0

    return clamp01(1.0 - (abs(out_min - gt_min) + abs(out_max - gt_max)) / (gt_max - gt_min))

def score_sex_ratio(out_data, gt_data):
    """Compare Male%. Both in 0-1 scale. GT uses options.Male directly."""
    out_sex = out_data.get("demographics", {}).get("sex", {}).get("options", {})
    gt_sex = gt_data.get("demographics", {}).get("sex", {}).get("options", {})

    out_male = out_sex.get("Male", 0)
    gt_male = gt_sex.get("Male", 0)

    # Handle if Male is a dict with "probability" key
    if isinstance(out_male, dict):
        out_male = out_male.get("probability", 0)
    if isinstance(gt_male, dict):
        gt_male = gt_male.get("probability", 0)

    print(f"  Output Male%: {out_male}")
    print(f"  GT Male%:     {gt_male}")

    return clamp01(1.0 - abs(out_male - gt_male))

def score_ecog(out_data, gt_data):
    """Compare ECOG distributions. Score = 1 - sum(|diff|)/2."""
    out_ecog = out_data.get("demographics", {}).get("ecog_ps", {}).get("options", {})
    gt_ecog = gt_data.get("demographics", {}).get("ecog_ps", {}).get("options", {})

    print(f"  Output ECOG: {out_ecog}")
    print(f"  GT ECOG:     {gt_ecog}")

    all_keys = set(list(out_ecog.keys()) + list(gt_ecog.keys()))
    total_diff = 0.0
    for k in all_keys:
        out_v = float(out_ecog.get(k, 0))
        gt_v = float(gt_ecog.get(k, 0))
        total_diff += abs(out_v - gt_v)

    return clamp01(1.0 - total_diff / 2.0)

def score_ae_count(out_data, gt_data):
    """Score = 1 - |n_output - n_gt| / max(n_output, n_gt)."""
    out_n = len(out_data.get("ae_profile", []))
    gt_n = len(gt_data.get("ae_profile", []))

    print(f"  Output AE count: {out_n}")
    print(f"  GT AE count:     {gt_n}")

    if max(out_n, gt_n) == 0:
        return 1.0
    return clamp01(1.0 - abs(out_n - gt_n) / max(out_n, gt_n))

def score_ae_freq(out_data, gt_data):
    """For AEs present in both (by normalized term), score = avg(1 - |diff|/max)."""
    out_aes = {}
    for ae in out_data.get("ae_profile", []):
        term = normalize_ae_term(ae.get("ae_term", ""))
        out_aes[term] = ae.get("incidence_all_grade", 0)

    gt_aes = {}
    for ae in gt_data.get("ae_profile", []):
        term = normalize_ae_term(ae.get("ae_term", ""))
        gt_aes[term] = ae.get("incidence_all_grade", 0)

    common = set(out_aes.keys()) & set(gt_aes.keys())
    print(f"  Common AEs ({len(common)}): {sorted(common)}")

    if not common:
        return 0.0

    scores = []
    for term in sorted(common):
        out_f = out_aes[term]
        gt_f = gt_aes[term]
        m = max(out_f, gt_f)
        if m == 0:
            s = 1.0
        else:
            s = 1.0 - abs(out_f - gt_f) / m
        scores.append(s)
        print(f"    {term:40s}: out={out_f:.3f} gt={gt_f:.3f} score={s:.3f}")

    return sum(scores) / len(scores)

def score_top_ae_overlap(out_data, gt_data):
    """GT's top 10 AEs by incidence. Count how many appear in output's top 10. Score = overlap/10."""
    out_aes = [(normalize_ae_term(ae.get("ae_term", "")), ae.get("incidence_all_grade", 0))
               for ae in out_data.get("ae_profile", [])]
    gt_aes = [(normalize_ae_term(ae.get("ae_term", "")), ae.get("incidence_all_grade", 0))
              for ae in gt_data.get("ae_profile", [])]

    out_aes.sort(key=lambda x: -x[1])
    gt_aes.sort(key=lambda x: -x[1])

    out_top10 = set(t for t, _ in out_aes[:10])
    gt_top10 = set(t for t, _ in gt_aes[:10])

    overlap = out_top10 & gt_top10

    print(f"  Output top 10: {[t for t,f in out_aes[:10]]}")
    print(f"  GT top 10:     {[t for t,f in gt_aes[:10]]}")
    print(f"  Overlap ({len(overlap)}): {sorted(overlap)}")

    return len(overlap) / 10.0

def score_pfs_os(out_data, gt_data):
    """Compare PFS mean and OS mean. Score = avg(1 - |diff|/gt) for both. Clamp 0-1."""
    out_eff = out_data.get("efficacy", {})
    gt_eff = gt_data.get("efficacy", {})

    scores = []

    # PFS
    out_pfs = out_eff.get("progression_free_survival_months", {}).get("params", {}).get("mean")
    gt_pfs = gt_eff.get("progression_free_survival_months", {}).get("params", {}).get("mean")
    print(f"  Output PFS mean: {out_pfs}")
    print(f"  GT PFS mean:     {gt_pfs}")
    if gt_pfs and gt_pfs > 0 and out_pfs is not None:
        scores.append(clamp01(1.0 - abs(out_pfs - gt_pfs) / gt_pfs))
    elif gt_pfs is None and out_pfs is None:
        scores.append(1.0)
    else:
        scores.append(0.0)

    # OS
    out_os = out_eff.get("overall_survival_months", {}).get("params", {}).get("mean")
    gt_os = gt_eff.get("overall_survival_months", {}).get("params", {}).get("mean")
    print(f"  Output OS mean:  {out_os}")
    print(f"  GT OS mean:      {gt_os}")
    if gt_os and gt_os > 0 and out_os is not None:
        scores.append(clamp01(1.0 - abs(out_os - gt_os) / gt_os))
    elif gt_os is None and out_os is None:
        scores.append(1.0)
    else:
        scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0

def main():
    out = load_json(OUTPUT_PATH)
    gt = load_json(GT_PATH)

    print("=" * 80)
    print("Etoposide+Cisplatin: Pipeline Output vs Ground Truth")
    print("=" * 80)
    print()

    dimensions = [
        ("1. Doses", score_doses),
        ("2. ORR", score_orr),
        ("3. Age Range", score_age_range),
        ("4. Sex Ratio", score_sex_ratio),
        ("5. ECOG", score_ecog),
        ("6. AE Count", score_ae_count),
        ("7. AE Freq", score_ae_freq),
        ("8. Top AE Overlap", score_top_ae_overlap),
        ("9. PFS/OS", score_pfs_os),
    ]

    results = {}
    for name, func in dimensions:
        print(f"\n--- {name} ---")
        score = func(out, gt)
        results[name] = score
        print(f"  >> Score: {score:.3f}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Dimension':<25s} {'Score':>8s}")
    print("-" * 35)
    for name, score in results.items():
        print(f"{name:<25s} {score:>8.3f}")

    avg = sum(results.values()) / len(results)
    print("-" * 35)
    print(f"{'Overall Average':<25s} {avg:>8.3f}")
    print(f"{'v14 Baseline (EP)':<25s} {V14_BASELINE:>8.3f}")
    delta = avg - V14_BASELINE
    sign = "+" if delta >= 0 else ""
    print(f"{'Delta':<25s} {sign}{delta:>7.3f}")
    print()

    # Print top 10 AEs from both for comparison
    print("=" * 80)
    print("TOP 10 AEs COMPARISON (by incidence)")
    print("=" * 80)

    out_aes_list = [(ae.get("ae_term", ""), ae.get("incidence_all_grade", 0))
                    for ae in out.get("ae_profile", [])]
    gt_aes_list = [(ae.get("ae_term", ""), ae.get("incidence_all_grade", 0))
                   for ae in gt.get("ae_profile", [])]

    out_aes_list.sort(key=lambda x: -x[1])
    gt_aes_list.sort(key=lambda x: -x[1])

    print(f"\n{'Rank':<6s} {'Output AE':<40s} {'Freq':>8s}   {'GT AE':<40s} {'Freq':>8s}")
    print("-" * 110)
    for i in range(10):
        out_term, out_freq = out_aes_list[i] if i < len(out_aes_list) else ("", 0)
        gt_term, gt_freq = gt_aes_list[i] if i < len(gt_aes_list) else ("", 0)
        print(f"{i+1:<6d} {out_term:<40s} {out_freq:>8.3f}   {gt_term:<40s} {gt_freq:>8.3f}")

    print()

if __name__ == "__main__":
    main()
