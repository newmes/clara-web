#!/usr/bin/env python3
"""Compare pipeline output against Ground Truth for both GT drugs.
Shows current (v15+PDS) vs v14 baseline vs GT."""

import json
import re
import sys

# Paths
OUTPUT_EP = "/home/ubuntu/samuel/rule_discovery/output/cisplatin+etoposide_small_cell_lung_cancer_rules.json"
GT_EP = "/home/ubuntu/samuel/rule_discovery/ground_truth/2_Etoposide_Cisplatin/base.json"

OUTPUT_DARB = "/home/ubuntu/samuel/rule_discovery/output/darbepoetin_alfa_small_cell_lung_cancer_rules.json"
GT_DARB = "/home/ubuntu/samuel/rule_discovery/ground_truth/1_Darbepoetin_alfa/base.json"

# v14 baselines (documented)
V14_EP = {
    "Doses": 1.000, "ORR": 0.804, "Age Range": 0.389, "Sex Ratio": 0.903,
    "ECOG": 1.000, "AE Count": 0.781, "AE Freq": 0.762, "Top AE": 0.600, "PFS/OS": 1.000,
}
V14_DARB = {
    "Doses": 0.001, "ORR": 0.442, "Age Range": 0.463, "Sex Ratio": 0.961,
    "ECOG": 1.000, "AE Count": 0.885, "AE Freq": 0.871, "Top AE": 0.300, "PFS/OS": 1.000,
}

# v15 baselines (documented in MEMORY.md)
V15_EP = {
    "Doses": 1.000, "ORR": 0.733, "Age Range": 0.486, "Sex Ratio": 0.838,
    "ECOG": 1.000, "AE Count": 0.781, "AE Freq": 0.761, "Top AE": 0.600, "PFS/OS": 1.000,
}
V15_DARB = {
    "Doses": 0.998, "ORR": 0.442, "Age Range": 0.771, "Sex Ratio": 0.961,
    "ECOG": 1.000, "AE Count": 0.833, "AE Freq": 0.814, "Top AE": 0.600, "PFS/OS": 1.000,
}


def load_json(path):
    with open(path) as f:
        return json.load(f)

def clamp01(x):
    return max(0.0, min(1.0, x))

def normalize_ae_term(term):
    t = term.lower().strip().replace(" ", "_").replace("-", "_")
    t = re.sub(r'_s$', '', t)
    synonyms = {
        "diarrhoea": "diarrhea", "dyspnoea": "dyspnea", "anaemia": "anemia",
        "haemoglobin_decreased": "lower_hemoglobin", "haemoglobin": "hemoglobin",
        "paraesthesia": "paresthesia", "hyponatraemia": "hyponatremia",
        "hypomagnesaemia": "hypomagnesemia", "oedema": "edema",
        "leucopenia": "leukopenia", "tumour": "tumor",
    }
    if t in synonyms:
        t = synonyms[t]
    return t

def parse_dose_numeric(dose_str):
    if not dose_str:
        return None
    m = re.search(r'([\d.]+)', str(dose_str))
    return float(m.group(1)) if m else None

def score_doses(out, gt):
    out_sched = out.get("administration_schedule", [])
    gt_sched = gt.get("administration_schedule", [])
    out_doses = {}
    for e in out_sched:
        name = e.get("drug_name", "").lower().strip()
        dose = parse_dose_numeric(e.get("dose_per_administration", ""))
        if name and dose is not None:
            out_doses[name] = dose
    gt_doses = {}
    for e in gt_sched:
        name = e.get("drug_name", "").lower().strip()
        dose = parse_dose_numeric(e.get("dose_per_administration", ""))
        if name and dose is not None:
            gt_doses[name] = dose
    if not gt_doses:
        return 0.0, f"out={out_doses} gt={gt_doses}"
    scores = []
    for drug, gt_dose in gt_doses.items():
        if drug in out_doses:
            out_dose = out_doses[drug]
            if gt_dose == 0:
                scores.append(1.0 if out_dose == 0 else 0.0)
            else:
                r = min(out_dose, gt_dose) / max(out_dose, gt_dose)
                scores.append(1.0 if r >= 0.9 else r)
        else:
            scores.append(0.0)
    return sum(scores)/len(scores), f"out={out_doses} gt={gt_doses}"

def score_orr(out, gt):
    out_orr = out.get("efficacy", {}).get("overall_response_rate")
    gt_orr = gt.get("efficacy", {}).get("overall_response_rate")
    if gt_orr is None or gt_orr == 0:
        return 0.0, f"out={out_orr} gt={gt_orr}"
    if out_orr is None:
        return 0.0, f"out={out_orr} gt={gt_orr}"
    return clamp01(1.0 - abs(out_orr - gt_orr) / gt_orr), f"out={out_orr:.3f} gt={gt_orr:.3f}"

def score_age_range(out, gt):
    out_age = out.get("demographics", {}).get("age", {}).get("params", {})
    gt_age = gt.get("demographics", {}).get("age", {}).get("params", {})
    out_min, out_max = out_age.get("min"), out_age.get("max")
    gt_min, gt_max = gt_age.get("min"), gt_age.get("max")
    if gt_min is None or gt_max is None or gt_max == gt_min:
        return 0.0, f"out={out_min}-{out_max} gt={gt_min}-{gt_max}"
    if out_min is None or out_max is None:
        return 0.0, f"out={out_min}-{out_max} gt={gt_min}-{gt_max}"
    return clamp01(1.0 - (abs(out_min - gt_min) + abs(out_max - gt_max)) / (gt_max - gt_min)), f"out={out_min}-{out_max} gt={gt_min}-{gt_max}"

def score_sex_ratio(out, gt):
    out_sex = out.get("demographics", {}).get("sex", {}).get("options", {})
    gt_sex = gt.get("demographics", {}).get("sex", {}).get("options", {})
    out_male = out_sex.get("Male", 0)
    gt_male = gt_sex.get("Male", 0)
    if isinstance(out_male, dict):
        out_male = out_male.get("probability", 0)
    if isinstance(gt_male, dict):
        gt_male = gt_male.get("probability", 0)
    return clamp01(1.0 - abs(out_male - gt_male)), f"out={out_male:.3f} gt={gt_male:.3f}"

def score_ecog(out, gt):
    out_ecog = out.get("demographics", {}).get("ecog_ps", {}).get("options", {})
    gt_ecog = gt.get("demographics", {}).get("ecog_ps", {}).get("options", {})
    all_keys = set(list(out_ecog.keys()) + list(gt_ecog.keys()))
    total_diff = sum(abs(float(out_ecog.get(k, 0)) - float(gt_ecog.get(k, 0))) for k in all_keys)
    return clamp01(1.0 - total_diff / 2.0), f"out={dict(out_ecog)} gt={dict(gt_ecog)}"

def score_ae_count(out, gt):
    out_n = len(out.get("ae_profile", []))
    gt_n = len(gt.get("ae_profile", []))
    if max(out_n, gt_n) == 0:
        return 1.0, f"out={out_n} gt={gt_n}"
    return clamp01(1.0 - abs(out_n - gt_n) / max(out_n, gt_n)), f"out={out_n} gt={gt_n}"

def score_ae_freq(out, gt):
    out_aes = {normalize_ae_term(ae.get("ae_term", "")): ae.get("incidence_all_grade", 0) for ae in out.get("ae_profile", [])}
    gt_aes = {normalize_ae_term(ae.get("ae_term", "")): ae.get("incidence_all_grade", 0) for ae in gt.get("ae_profile", [])}
    common = set(out_aes.keys()) & set(gt_aes.keys())
    if not common:
        return 0.0, "no common AEs"
    scores = []
    for t in sorted(common):
        m = max(out_aes[t], gt_aes[t])
        scores.append(1.0 if m == 0 else 1.0 - abs(out_aes[t] - gt_aes[t]) / m)
    return sum(scores)/len(scores), f"{len(common)} common AEs"

def score_top_ae(out, gt):
    out_aes = sorted([(normalize_ae_term(ae.get("ae_term", "")), ae.get("incidence_all_grade", 0)) for ae in out.get("ae_profile", [])], key=lambda x: -x[1])
    gt_aes = sorted([(normalize_ae_term(ae.get("ae_term", "")), ae.get("incidence_all_grade", 0)) for ae in gt.get("ae_profile", [])], key=lambda x: -x[1])
    out_top10 = set(t for t, _ in out_aes[:10])
    gt_top10 = set(t for t, _ in gt_aes[:10])
    overlap = out_top10 & gt_top10
    return len(overlap)/10.0, f"overlap={sorted(overlap)}"

def score_pfs_os(out, gt):
    out_eff, gt_eff = out.get("efficacy", {}), gt.get("efficacy", {})
    scores = []
    for key in ["progression_free_survival_months", "overall_survival_months"]:
        out_v = out_eff.get(key, {}).get("params", {}).get("mean")
        gt_v = gt_eff.get(key, {}).get("params", {}).get("mean")
        if gt_v and gt_v > 0 and out_v is not None:
            scores.append(clamp01(1.0 - abs(out_v - gt_v) / gt_v))
        elif gt_v is None and out_v is None:
            scores.append(1.0)
        else:
            scores.append(0.0)
    out_pfs = out_eff.get("progression_free_survival_months", {}).get("params", {}).get("mean")
    out_os = out_eff.get("overall_survival_months", {}).get("params", {}).get("mean")
    gt_pfs = gt_eff.get("progression_free_survival_months", {}).get("params", {}).get("mean")
    gt_os = gt_eff.get("overall_survival_months", {}).get("params", {}).get("mean")
    return sum(scores)/len(scores) if scores else 0.0, f"PFS: out={out_pfs} gt={gt_pfs}, OS: out={out_os} gt={gt_os}"

def run_comparison(name, out_path, gt_path, v14_baseline, v15_baseline):
    out = load_json(out_path)
    gt = load_json(gt_path)

    dims = [
        ("Doses", score_doses),
        ("ORR", score_orr),
        ("Age Range", score_age_range),
        ("Sex Ratio", score_sex_ratio),
        ("ECOG", score_ecog),
        ("AE Count", score_ae_count),
        ("AE Freq", score_ae_freq),
        ("Top AE", score_top_ae),
        ("PFS/OS", score_pfs_os),
    ]

    results = {}
    details = {}
    for dim_name, func in dims:
        score, detail = func(out, gt)
        results[dim_name] = score
        details[dim_name] = detail

    return results, details

def print_comparison_table(ep_results, ep_details, darb_results, darb_details):
    dims = ["Doses", "ORR", "Age Range", "Sex Ratio", "ECOG", "AE Count", "AE Freq", "Top AE", "PFS/OS"]

    print("=" * 120)
    print("GROUND TRUTH COMPARISON: v14 (no PDS) vs v15 (PDS) vs Current Run (PDS, latest)")
    print("=" * 120)

    # EP table
    print(f"\n{'─'*120}")
    print(f"  ETOPOSIDE + CISPLATIN / SCLC")
    print(f"{'─'*120}")
    print(f"{'Dimension':<15s} {'v14':>8s} {'v15':>8s} {'Current':>8s} {'Δ v14→Cur':>10s} {'Δ v15→Cur':>10s}   Details")
    print(f"{'─'*15} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10}   {'─'*40}")
    for d in dims:
        v14 = V14_EP.get(d, 0)
        v15 = V15_EP.get(d, 0)
        cur = ep_results.get(d, 0)
        d14 = cur - v14
        d15 = cur - v15
        s14 = f"+{d14:.3f}" if d14 >= 0 else f"{d14:.3f}"
        s15 = f"+{d15:.3f}" if d15 >= 0 else f"{d15:.3f}"
        print(f"{d:<15s} {v14:>8.3f} {v15:>8.3f} {cur:>8.3f} {s14:>10s} {s15:>10s}   {ep_details.get(d, '')}")

    ep_avg_v14 = sum(V14_EP.values()) / len(V14_EP)
    ep_avg_v15 = sum(V15_EP.values()) / len(V15_EP)
    ep_avg_cur = sum(ep_results.values()) / len(ep_results)
    d14 = ep_avg_cur - ep_avg_v14
    d15 = ep_avg_cur - ep_avg_v15
    print(f"{'─'*15} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")
    print(f"{'AVERAGE':<15s} {ep_avg_v14:>8.3f} {ep_avg_v15:>8.3f} {ep_avg_cur:>8.3f} {'+' if d14>=0 else ''}{d14:>9.3f} {'+' if d15>=0 else ''}{d15:>9.3f}")

    # Darb table
    print(f"\n{'─'*120}")
    print(f"  DARBEPOETIN ALFA / SCLC")
    print(f"{'─'*120}")
    print(f"{'Dimension':<15s} {'v14':>8s} {'v15':>8s} {'Current':>8s} {'Δ v14→Cur':>10s} {'Δ v15→Cur':>10s}   Details")
    print(f"{'─'*15} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10}   {'─'*40}")
    for d in dims:
        v14 = V14_DARB.get(d, 0)
        v15 = V15_DARB.get(d, 0)
        cur = darb_results.get(d, 0)
        d14 = cur - v14
        d15 = cur - v15
        s14 = f"+{d14:.3f}" if d14 >= 0 else f"{d14:.3f}"
        s15 = f"+{d15:.3f}" if d15 >= 0 else f"{d15:.3f}"
        print(f"{d:<15s} {v14:>8.3f} {v15:>8.3f} {cur:>8.3f} {s14:>10s} {s15:>10s}   {darb_details.get(d, '')}")

    darb_avg_v14 = sum(V14_DARB.values()) / len(V14_DARB)
    darb_avg_v15 = sum(V15_DARB.values()) / len(V15_DARB)
    darb_avg_cur = sum(darb_results.values()) / len(darb_results)
    d14 = darb_avg_cur - darb_avg_v14
    d15 = darb_avg_cur - darb_avg_v15
    print(f"{'─'*15} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")
    print(f"{'AVERAGE':<15s} {darb_avg_v14:>8.3f} {darb_avg_v15:>8.3f} {darb_avg_cur:>8.3f} {'+' if d14>=0 else ''}{d14:>9.3f} {'+' if d15>=0 else ''}{d15:>9.3f}")

    # Combined
    combined_v14 = (ep_avg_v14 + darb_avg_v14) / 2
    combined_v15 = (ep_avg_v15 + darb_avg_v15) / 2
    combined_cur = (ep_avg_cur + darb_avg_cur) / 2
    print(f"\n{'='*120}")
    print(f"  COMBINED SCORES")
    print(f"{'='*120}")
    print(f"{'Drug':<30s} {'v14':>8s} {'v15':>8s} {'Current':>8s} {'Δ v14→Cur':>10s} {'Δ v15→Cur':>10s}")
    print(f"{'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")
    print(f"{'Etoposide+Cisplatin':<30s} {ep_avg_v14:>8.1%} {ep_avg_v15:>8.1%} {ep_avg_cur:>8.1%} {'+' if ep_avg_cur-ep_avg_v14>=0 else ''}{(ep_avg_cur-ep_avg_v14):>9.1%} {'+' if ep_avg_cur-ep_avg_v15>=0 else ''}{(ep_avg_cur-ep_avg_v15):>9.1%}")
    print(f"{'Darbepoetin alfa':<30s} {darb_avg_v14:>8.1%} {darb_avg_v15:>8.1%} {darb_avg_cur:>8.1%} {'+' if darb_avg_cur-darb_avg_v14>=0 else ''}{(darb_avg_cur-darb_avg_v14):>9.1%} {'+' if darb_avg_cur-darb_avg_v15>=0 else ''}{(darb_avg_cur-darb_avg_v15):>9.1%}")
    print(f"{'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")
    print(f"{'COMBINED AVERAGE':<30s} {combined_v14:>8.1%} {combined_v15:>8.1%} {combined_cur:>8.1%} {'+' if combined_cur-combined_v14>=0 else ''}{(combined_cur-combined_v14):>9.1%} {'+' if combined_cur-combined_v15>=0 else ''}{(combined_cur-combined_v15):>9.1%}")
    print()

def main():
    ep_results, ep_details = run_comparison("Etoposide+Cisplatin", OUTPUT_EP, GT_EP, V14_EP, V15_EP)
    darb_results, darb_details = run_comparison("Darbepoetin alfa", OUTPUT_DARB, GT_DARB, V14_DARB, V15_DARB)
    print_comparison_table(ep_results, ep_details, darb_results, darb_details)

if __name__ == "__main__":
    main()
