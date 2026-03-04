"""
CSR (Clinical Study Report) Statistics Engine

Computes all statistical tables and chart data needed for ICH E3-compliant
clinical trial results reporting:
  - Subject disposition
  - Demographics & baseline characteristics
  - Efficacy (ORR with CI, waterfall, spider, Kaplan-Meier OS/PFS)
  - Safety (AE summary, by-term, grade distribution, timeline heatmap)
  - Lab abnormalities (shift tables, trends)
  - Treatment exposure
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta as beta_dist


# ─── Helpers ────────────────────────────────────────────────

def _pct(n: int, total: int) -> float:
    return round(n / total * 100, 1) if total else 0.0


def _desc_stats(values: list[float]) -> dict:
    """Descriptive statistics for continuous variables."""
    if not values:
        return {"n": 0}
    a = np.array(values, dtype=float)
    return {
        "n": len(a),
        "mean": round(float(np.mean(a)), 1),
        "std": round(float(np.std(a, ddof=1)), 1) if len(a) > 1 else 0.0,
        "median": round(float(np.median(a)), 1),
        "q1": round(float(np.percentile(a, 25)), 1),
        "q3": round(float(np.percentile(a, 75)), 1),
        "min": round(float(np.min(a)), 1),
        "max": round(float(np.max(a)), 1),
    }


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    lo = float(beta_dist.ppf(alpha / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(beta_dist.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0
    return (round(lo * 100, 1), round(hi * 100, 1))


def kaplan_meier(times: list[float], events: list[int],
                 ) -> dict:
    """
    Kaplan-Meier survival estimator.
    times:  observation durations
    events: 1=event occurred, 0=censored
    Returns curve points, median, 95% CI, risk table.
    """
    if not times:
        return {"curve": [], "median": None, "ci": [None, None], "risk_table": []}

    data = sorted(zip(times, events), key=lambda x: x[0])
    n_at_risk = len(data)
    surv = 1.0
    curve = [(0, 1.0)]
    event_times = []
    risk_table_entries = [(0, n_at_risk)]
    var_greenwood = 0.0

    i = 0
    median = None
    median_ci_lo = None
    median_ci_hi = None

    while i < len(data):
        t = data[i][0]
        d_i = 0  # events at this time
        c_i = 0  # censored at this time
        while i < len(data) and data[i][0] == t:
            if data[i][1] == 1:
                d_i += 1
            else:
                c_i += 1
            i += 1

        if d_i > 0:
            surv_prev = surv
            surv *= (n_at_risk - d_i) / n_at_risk
            curve.append((t, round(surv, 4)))
            event_times.append(t)

            if n_at_risk > d_i:
                var_greenwood += d_i / (n_at_risk * (n_at_risk - d_i))

            if median is None and surv <= 0.5:
                median = t
            se = surv * math.sqrt(var_greenwood) if var_greenwood > 0 else 0
            ci_lo_surv = max(0, surv - 1.96 * se)
            ci_hi_surv = min(1, surv + 1.96 * se)
            if median_ci_lo is None and ci_hi_surv <= 0.5:
                median_ci_lo = t
            if median_ci_hi is None and ci_lo_surv <= 0.5:
                median_ci_hi = t

        n_at_risk -= (d_i + c_i)
        if n_at_risk > 0:
            risk_table_entries.append((t, n_at_risk))

    milestone_days = [0, 30, 60, 90, 120, 150, 180]
    risk_table = []
    for md in milestone_days:
        n_r = len(times)
        for t, ev in data:
            if t < md:
                n_r -= 1
            else:
                break
        risk_at = sum(1 for t in times if t >= md)
        risk_table.append({"day": md, "n_at_risk": risk_at})

    return {
        "curve": curve,
        "median": median,
        "ci": [median_ci_lo, median_ci_hi],
        "risk_table": risk_table,
        "n_events": sum(events),
        "n_censored": len(events) - sum(events),
    }


# ─── Data loaders ───────────────────────────────────────────

def _load_profiles(run_dir: Path) -> dict[str, dict]:
    profiles = {}
    for f in sorted((run_dir / "patients").glob("*.json")):
        with open(f) as fh:
            p = json.load(fh)
        profiles[p.get("patient_id", f.stem)] = p
    return profiles


def _load_sims(run_dir: Path, pids: list[str],
               mode: str = "natural") -> dict[str, list[dict]]:
    sims = {}
    for pid in pids:
        fp = run_dir / "simulations" / f"{pid}_{mode}.jsonl"
        if not fp.exists():
            continue
        records = []
        with open(fp) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            sims[pid] = records
    return sims


def _load_rule_set(run_dir: Path) -> dict:
    rs_path = run_dir / "rule_set.json"
    if rs_path.exists():
        with open(rs_path) as f:
            return json.load(f)
    return {}


# ─── Section A: Disposition ─────────────────────────────────

def compute_disposition(sims: dict[str, list[dict]]) -> dict:
    n = len(sims)
    completed = 0
    deceased = 0
    discontinued = 0
    reasons: dict[str, int] = Counter()

    for pid, days in sims.items():
        if not days:
            continue
        last = days[-1]
        obj = last.get("objective", {})
        loc = obj.get("location", "HOME")
        ts = obj.get("treatment_status", "on_treatment")

        if loc == "DECEASED":
            deceased += 1
            reasons["Death"] += 1
        elif "discontinued" in ts:
            discontinued += 1
            ds_list = last.get("DS", [])
            reason = "Other"
            for ds in (ds_list if isinstance(ds_list, list) else [ds_list]):
                if not isinstance(ds, dict):
                    continue
                dec = ds.get("DSDECOD", "") or ""
                if "ADVERSE" in dec.upper() or "TOXICITY" in dec.upper():
                    reason = "Adverse Event"
                elif "PROGRESSIVE" in dec.upper() or "DISEASE" in dec.upper():
                    reason = "Disease Progression"
                elif "WITHDRAW" in dec.upper() or "CONSENT" in dec.upper():
                    reason = "Withdrawal by Subject"
                elif "DEATH" in dec.upper():
                    reason = "Death"
                elif dec:
                    reason = dec
            reasons[reason] += 1
        else:
            completed += 1

    return {
        "enrolled": n,
        "completed": {"n": completed, "pct": _pct(completed, n)},
        "discontinued": {"n": discontinued, "pct": _pct(discontinued, n)},
        "deaths": {"n": deceased, "pct": _pct(deceased, n)},
        "reasons": {k: v for k, v in reasons.most_common()},
    }


# ─── Section B: Demographics & Baseline ─────────────────────

def compute_demographics(profiles: dict[str, dict]) -> dict:
    ages, weights, heights, bmis, sods = [], [], [], [], []
    sex_counts: Counter = Counter()
    race_counts: Counter = Counter()
    ecog_counts: Counter = Counter()
    stage_counts: Counter = Counter()
    met_site_counts: Counter = Counter()
    mh_counts: Counter = Counter()

    for pid, p in profiles.items():
        dm = p.get("DM", {})
        emr = p.get("emr", {})
        demo = emr.get("demographics", {})
        ds = emr.get("disease_specific", {})

        age = dm.get("AGE") or demo.get("age")
        if age is not None:
            ages.append(float(age))

        sex = dm.get("SEX") or demo.get("sex", "?")
        sex_counts[sex] += 1

        race = dm.get("RACE") or demo.get("race", "Unknown")
        race_counts[race.upper() if race else "UNKNOWN"] += 1

        ecog = demo.get("ecog_ps")
        if ecog is None:
            ecog = emr.get("baseline_ecog")
        if ecog is not None:
            ecog_counts[str(int(ecog))] += 1

        w = demo.get("weight_kg")
        if w:
            weights.append(float(w))
        h = demo.get("height_cm")
        if h:
            heights.append(float(h))
        bmi = demo.get("bmi")
        if bmi:
            bmis.append(float(bmi))

        stage = ds.get("stage", "")
        if stage:
            stage_counts[stage] += 1

        for site in ds.get("sites_of_metastasis", []):
            if isinstance(site, str) and site:
                met_site_counts[site] += 1

        sod = ds.get("sum_of_diameters_mm")
        if sod is not None:
            sods.append(float(sod))

        for mh in p.get("MH", []):
            if isinstance(mh, dict):
                term = mh.get("MHTERM", "")
                if term:
                    mh_counts[term] += 1

    # Age bins for histogram
    age_bins = {"<50": 0, "50–59": 0, "60–69": 0, "70–79": 0, "≥80": 0}
    for a in ages:
        if a < 50:
            age_bins["<50"] += 1
        elif a < 60:
            age_bins["50–59"] += 1
        elif a < 70:
            age_bins["60–69"] += 1
        elif a < 80:
            age_bins["70–79"] += 1
        else:
            age_bins["≥80"] += 1

    n = len(profiles)
    return {
        "n": n,
        "age": _desc_stats(ages),
        "age_bins": age_bins,
        "sex": {k: {"n": v, "pct": _pct(v, n)} for k, v in sex_counts.most_common()},
        "race": {k: {"n": v, "pct": _pct(v, n)} for k, v in race_counts.most_common()},
        "ecog": {k: {"n": v, "pct": _pct(v, n)} for k, v in sorted(ecog_counts.items())},
        "weight": _desc_stats(weights),
        "height": _desc_stats(heights),
        "bmi": _desc_stats(bmis),
        "stage": {k: {"n": v, "pct": _pct(v, n)} for k, v in stage_counts.most_common()},
        "metastasis_sites": {k: {"n": v, "pct": _pct(v, n)} for k, v in met_site_counts.most_common()},
        "sum_of_diameters": _desc_stats(sods),
        "medical_history": {k: {"n": v, "pct": _pct(v, n)} for k, v in mh_counts.most_common(15)},
    }


# ─── Section C: Efficacy ────────────────────────────────────

def compute_efficacy(sims: dict[str, list[dict]]) -> dict:
    n = len(sims)
    best_changes: list[tuple[str, float, str]] = []  # (pid, best_change, category)
    os_times, os_events = [], []
    pfs_times, pfs_events = [], []
    time_to_response: list[float] = []

    dor_times: list[float] = []
    dor_events: list[int] = []

    for pid, days in sims.items():
        best_change = None
        worst_change = None
        nadir = 0.0
        has_tumor_data = False
        death_day = None
        pd_day = None
        first_response_day = None
        last_day = days[-1].get("day", 0) if days else 0

        for d in days:
            day_num = d.get("day", 0)
            obj = d.get("objective", {})
            tumor = obj.get("tumor") or {}

            change = tumor.get("estimated_change_pct")
            if change is not None:
                has_tumor_data = True
                if best_change is None or change < best_change:
                    best_change = change
                if worst_change is None or change > worst_change:
                    worst_change = change
                if change < nadir:
                    nadir = change

                if first_response_day is None and change <= -30:
                    first_response_day = day_num

                change_from_nadir = change - nadir
                if pd_day is None and change_from_nadir >= 20 and day_num > 21:
                    pd_day = day_num

            if obj.get("location") == "DECEASED":
                death_day = day_num

        if best_change is None:
            best_change = 0.0
        if worst_change is None:
            worst_change = 0.0

        if best_change <= -90:
            cat = "CR"
        elif best_change <= -30:
            cat = "PR"
        elif pd_day is not None:
            cat = "PD"
        elif worst_change >= 20:
            cat = "PD"
        else:
            cat = "SD"

        best_changes.append((pid, round(best_change, 1), cat))

        if first_response_day is not None:
            time_to_response.append(first_response_day)

        # DoR: for responders (CR/PR), duration from first response to PD/death/censor
        if cat in ("CR", "PR") and first_response_day is not None:
            end_event = min(x for x in [pd_day, death_day] if x is not None) \
                if any(x is not None for x in [pd_day, death_day]) else None
            if end_event and end_event > first_response_day:
                dor_times.append(end_event - first_response_day)
                dor_events.append(1)
            else:
                dor_times.append(last_day - first_response_day)
                dor_events.append(0)

        # OS
        os_times.append(death_day if death_day else last_day)
        os_events.append(1 if death_day else 0)

        # PFS
        pfs_event_day = min(x for x in [pd_day, death_day] if x is not None) \
            if any(x is not None for x in [pd_day, death_day]) else None
        pfs_times.append(pfs_event_day if pfs_event_day else last_day)
        pfs_events.append(1 if pfs_event_day else 0)

    # Response counts
    resp_counts = Counter(cat for _, _, cat in best_changes)
    cr = resp_counts.get("CR", 0)
    pr = resp_counts.get("PR", 0)
    sd = resp_counts.get("SD", 0)
    pd_n = resp_counts.get("PD", 0)
    orr_n = cr + pr
    dcr_n = cr + pr + sd
    orr_ci = clopper_pearson(orr_n, n)
    dcr_ci = clopper_pearson(dcr_n, n)

    # Waterfall: sorted by change
    waterfall = sorted(
        [{"pid": pid, "change": ch, "response": cat}
         for pid, ch, cat in best_changes],
        key=lambda x: x["change"]
    )

    km_os = kaplan_meier(os_times, os_events)
    km_pfs = kaplan_meier(pfs_times, pfs_events)
    km_dor = kaplan_meier(dor_times, dor_events) if dor_times else {
        "curve": [], "median": None, "ci": [None, None],
        "risk_table": [], "n_events": 0, "n_censored": 0,
    }

    return {
        "n": n,
        "orr": {"n": orr_n, "pct": _pct(orr_n, n), "ci": list(orr_ci)},
        "dcr": {"n": dcr_n, "pct": _pct(dcr_n, n), "ci": list(dcr_ci)},
        "best_response": {
            "CR": {"n": cr, "pct": _pct(cr, n)},
            "PR": {"n": pr, "pct": _pct(pr, n)},
            "SD": {"n": sd, "pct": _pct(sd, n)},
            "PD": {"n": pd_n, "pct": _pct(pd_n, n)},
        },
        "waterfall": waterfall,
        "km_os": km_os,
        "km_pfs": km_pfs,
        "km_dor": km_dor,
        "time_to_response": _desc_stats(time_to_response),
        "dor": _desc_stats(dor_times) if dor_times else {"n": 0},
    }


# ─── Section D: Safety — AE ─────────────────────────────────

def compute_safety(sims: dict[str, list[dict]]) -> dict:
    n = len(sims)
    patient_ae_info: dict[str, dict[str, dict]] = {}
    timeline_raw: dict[str, list[dict]] = defaultdict(list)

    for pid, days in sims.items():
        seen: dict[str, dict] = {}
        for d in days:
            day_num = d.get("day", 0)
            for ae in d.get("AE", []):
                term = ae.get("AETERM", "unknown")
                grade = ae.get("_grade", 0) or 0
                action = str(ae.get("AEACN", ""))
                onset = ae.get("AESTDAT")

                timeline_raw[term].append({
                    "pid": pid, "day": day_num, "grade": grade
                })

                if term not in seen:
                    seen[term] = {
                        "max_grade": grade, "onset": onset,
                        "serious": ae.get("AESER", False),
                        "fatal": ae.get("AESDTH", False),
                        "discont": "WITHDRAWN" in action,
                        "interrupt": "INTERRUPT" in action,
                        "reduction": "REDUCED" in action,
                    }
                else:
                    s = seen[term]
                    s["max_grade"] = max(s["max_grade"], grade)
                    if ae.get("AESER"):
                        s["serious"] = True
                    if ae.get("AESDTH"):
                        s["fatal"] = True
                    if "WITHDRAWN" in action:
                        s["discont"] = True
                    if "INTERRUPT" in action:
                        s["interrupt"] = True
                    if "REDUCED" in action:
                        s["reduction"] = True
        patient_ae_info[pid] = seen

    # Summary counts
    any_ae = sum(1 for v in patient_ae_info.values() if v)
    g3_plus = sum(1 for v in patient_ae_info.values()
                  if any(i["max_grade"] >= 3 for i in v.values()))
    sae = sum(1 for v in patient_ae_info.values()
              if any(i["serious"] for i in v.values()))
    fatal = sum(1 for v in patient_ae_info.values()
                if any(i["fatal"] for i in v.values()))
    discont = sum(1 for v in patient_ae_info.values()
                  if any(i["discont"] for i in v.values()))
    interrupt = sum(1 for v in patient_ae_info.values()
                    if any(i["interrupt"] for i in v.values()))
    reduction = sum(1 for v in patient_ae_info.values()
                    if any(i["reduction"] for i in v.values()))

    summary = {
        "any_ae": {"n": any_ae, "pct": _pct(any_ae, n)},
        "grade_gte3": {"n": g3_plus, "pct": _pct(g3_plus, n)},
        "sae": {"n": sae, "pct": _pct(sae, n)},
        "fatal": {"n": fatal, "pct": _pct(fatal, n)},
        "led_to_discont": {"n": discont, "pct": _pct(discont, n)},
        "led_to_interrupt": {"n": interrupt, "pct": _pct(interrupt, n)},
        "led_to_reduction": {"n": reduction, "pct": _pct(reduction, n)},
    }

    # Per-term table
    term_counts: Counter = Counter()
    term_g3: Counter = Counter()
    term_grades: dict[str, Counter] = defaultdict(Counter)
    term_onsets: dict[str, list] = defaultdict(list)
    for pid, seen in patient_ae_info.items():
        for term, info in seen.items():
            term_counts[term] += 1
            g = info["max_grade"]
            term_grades[term][g] += 1
            if g >= 3:
                term_g3[term] += 1
            if info["onset"] is not None:
                term_onsets[term].append(info["onset"])

    by_term = []
    for term, cnt in term_counts.most_common():
        g3c = term_g3.get(term, 0)
        onsets = term_onsets.get(term, [])
        grade_dist = {}
        for g in range(1, 6):
            grade_dist[str(g)] = term_grades[term].get(g, 0)
        by_term.append({
            "term": term,
            "all_grade": {"n": cnt, "pct": _pct(cnt, n)},
            "grade_gte3": {"n": g3c, "pct": _pct(g3c, n)},
            "grade_dist": grade_dist,
            "onset_median": round(float(np.median(onsets)), 0) if onsets else None,
            "onset_iqr": [round(float(np.percentile(onsets, 25)), 0),
                          round(float(np.percentile(onsets, 75)), 0)] if len(onsets) >= 2 else None,
        })

    # Timeline heatmap: per-term, deduplicate to one entry per pid-day (max grade)
    heatmap = []
    for term in [x["term"] for x in by_term[:20]]:
        entries = timeline_raw.get(term, [])
        day_grade: dict[int, int] = {}
        for e in entries:
            d = e["day"]
            if d not in day_grade or e["grade"] > day_grade[d]:
                day_grade[d] = e["grade"]
        heatmap.append({
            "term": term,
            "entries": [{"day": d, "grade": g} for d, g in sorted(day_grade.items())]
        })

    return {
        "n": n,
        "summary": summary,
        "by_term": by_term,
        "heatmap": heatmap,
    }


# ─── Section E: Lab Abnormalities ───────────────────────────

LAB_REFS = {
    "ANC": {"lln": 1.5, "uln": 7.5, "dir": "low",
            "g3": 1.0, "g4": 0.5, "unit": "×10⁹/L"},
    "hemoglobin": {"lln": 12.0, "uln": 17.5, "dir": "low",
                   "g3": 8.0, "g4": 6.5, "unit": "g/dL"},
    "platelets": {"lln": 150, "uln": 400, "dir": "low",
                  "g3": 50, "g4": 25, "unit": "×10⁹/L"},
    "creatinine": {"lln": 0.7, "uln": 1.3, "dir": "high",
                   "g3": 3.5, "g4": 6.0, "unit": "mg/dL"},
    "ALT": {"lln": 7, "uln": 56, "dir": "high",
            "g3": 280, "g4": 1120, "unit": "U/L"},
    "AST": {"lln": 10, "uln": 40, "dir": "high",
            "g3": 200, "g4": 800, "unit": "U/L"},
    "glucose_fasting": {"lln": 70, "uln": 99, "dir": "high",
                        "g3": 250, "g4": 500, "unit": "mg/dL"},
    "total_bilirubin": {"lln": 0.1, "uln": 1.2, "dir": "high",
                        "g3": 3.6, "g4": 12.0, "unit": "mg/dL"},
    "albumin": {"lln": 3.4, "uln": 5.4, "dir": "low",
                "g3": 2.0, "unit": "g/dL"},
    "sodium": {"lln": 135, "uln": 145, "dir": "low",
               "g3": 120, "g4": 115, "unit": "mmol/L"},
    "potassium": {"lln": 3.5, "uln": 5.0, "dir": "high",
                  "g3": 6.0, "g4": 7.0, "unit": "mmol/L"},
}

_LAB_NAME_MAP = {
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
}


def _lab_grade(test: str, value: float) -> int:
    """CTCAE-like grade for a lab value. 0=normal, 1=mild, 2=moderate, 3/4=severe."""
    ref = LAB_REFS.get(test)
    if not ref:
        return 0
    uln = ref["uln"]
    lln = ref["lln"]
    d = ref["dir"]
    g4 = ref.get("g4")
    g3 = ref.get("g3")

    if d == "high":
        if g4 and value >= g4:
            return 4
        if g3 and value >= g3:
            return 3
        if value > uln * 2.5:
            return 2
        if value > uln:
            return 1
        return 0
    else:
        if g4 and value <= g4:
            return 4
        if g3 and value <= g3:
            return 3
        if lln and value < lln * 0.75:
            return 2
        if lln and value < lln:
            return 1
        return 0


def compute_labs(sims: dict[str, list[dict]], profiles: dict[str, dict]) -> dict:
    n = len(sims)

    # Collect baseline & worst post-baseline per patient
    patient_baseline: dict[str, dict[str, float]] = defaultdict(dict)
    patient_worst: dict[str, dict[str, float]] = defaultdict(dict)
    # Trend data: test -> cycle -> [values]
    trend_data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for pid, days in sims.items():
        for d in days:
            day_num = d.get("day", 0)
            cycle = d.get("cycle", 1) or 1
            lb = d.get("LB", {})
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
                mapped = _LAB_NAME_MAP.get(lab_name, lab_name)
                if mapped not in LAB_REFS:
                    continue

                ref = LAB_REFS[mapped]
                trend_data[mapped][cycle].append(val)

                if day_num == 0:
                    patient_baseline[pid][mapped] = val
                elif day_num == 1 and mapped not in patient_baseline.get(pid, {}):
                    patient_baseline[pid][mapped] = val

                if day_num >= 1:
                    if mapped not in patient_worst[pid]:
                        patient_worst[pid][mapped] = val
                    elif ref["dir"] == "high":
                        patient_worst[pid][mapped] = max(patient_worst[pid][mapped], val)
                    else:
                        patient_worst[pid][mapped] = min(patient_worst[pid][mapped], val)

    # Abnormality summary
    abnormalities = {}
    for test, ref in LAB_REFS.items():
        total = sum(1 for pid in sims if test in patient_worst.get(pid, {}))
        if total == 0:
            continue
        any_abn = 0
        g3_plus = 0
        for pid in sims:
            val = patient_worst.get(pid, {}).get(test)
            if val is None:
                continue
            g = _lab_grade(test, val)
            if g >= 1:
                any_abn += 1
            if g >= 3:
                g3_plus += 1
        abnormalities[test] = {
            "n": total,
            "any_pct": _pct(any_abn, total),
            "g3_pct": _pct(g3_plus, total),
            "unit": ref.get("unit", ""),
        }

    # Shift tables: baseline grade → worst grade (4x4: 0,1,2,3+)
    shift_tables = {}
    for test in LAB_REFS:
        matrix = [[0] * 4 for _ in range(4)]
        for pid in sims:
            bv = patient_baseline.get(pid, {}).get(test)
            wv = patient_worst.get(pid, {}).get(test)
            if bv is None or wv is None:
                continue
            bg = min(_lab_grade(test, bv), 3)
            wg = min(_lab_grade(test, wv), 3)
            matrix[bg][wg] += 1
        if any(any(row) for row in matrix):
            shift_tables[test] = matrix

    # Trends: per-test, per-cycle mean ± SE
    trends = {}
    for test in LAB_REFS:
        if test not in trend_data:
            continue
        cycle_stats = []
        for cyc in sorted(trend_data[test].keys()):
            vals = trend_data[test][cyc]
            if not vals:
                continue
            a = np.array(vals)
            cycle_stats.append({
                "cycle": cyc,
                "mean": round(float(np.mean(a)), 2),
                "se": round(float(np.std(a, ddof=1) / math.sqrt(len(a))), 2) if len(a) > 1 else 0,
                "n": len(a),
            })
        if cycle_stats:
            trends[test] = {
                "points": cycle_stats,
                "ref_lo": LAB_REFS[test]["lln"],
                "ref_hi": LAB_REFS[test]["uln"],
                "unit": LAB_REFS[test].get("unit", ""),
            }

    return {
        "n": n,
        "abnormalities": abnormalities,
        "shift_tables": shift_tables,
        "trends": trends,
    }


# ─── Section E-bis: ECOG PS Shift ────────────────────────────

def compute_ecog_shift(sims: dict[str, list[dict]],
                       profiles: dict[str, dict]) -> dict:
    """Baseline ECOG → worst / last ECOG shift table."""
    n = len(sims)
    shift_worst = [[0] * 6 for _ in range(6)]
    shift_last = [[0] * 6 for _ in range(6)]
    patients = []

    for pid, days in sims.items():
        emr = profiles.get(pid, {}).get("emr", {})
        demo = emr.get("demographics", {})
        baseline = demo.get("ecog_ps")
        if baseline is None:
            baseline = emr.get("baseline_ecog")

        if days:
            first_obj = days[0].get("objective", {})
            if baseline is None:
                baseline = first_obj.get("ecog")

        if baseline is None:
            continue
        baseline = int(baseline)

        worst_ecog = baseline
        last_ecog = baseline
        for d in days:
            e = d.get("objective", {}).get("ecog")
            if e is not None:
                e = int(e)
                last_ecog = e
                if e > worst_ecog:
                    worst_ecog = e

        b = min(baseline, 5)
        w = min(worst_ecog, 5)
        l = min(last_ecog, 5)
        shift_worst[b][w] += 1
        shift_last[b][l] += 1
        patients.append({
            "pid": pid, "baseline": baseline,
            "worst": worst_ecog, "last": last_ecog,
        })

    improved = sum(1 for p in patients if p["last"] < p["baseline"])
    stable = sum(1 for p in patients if p["last"] == p["baseline"])
    worsened = sum(1 for p in patients if p["last"] > p["baseline"])
    total = len(patients)

    return {
        "n": total,
        "shift_worst": shift_worst,
        "shift_last": shift_last,
        "summary": {
            "improved": {"n": improved, "pct": _pct(improved, total)},
            "stable": {"n": stable, "pct": _pct(stable, total)},
            "worsened": {"n": worsened, "pct": _pct(worsened, total)},
        },
        "patients": patients,
    }


# ─── Section E-ter: Concomitant Medications ──────────────────

def compute_concomitant_meds(sims: dict[str, list[dict]]) -> dict:
    """Summarise concomitant medications across all patients."""
    n = len(sims)
    med_patients: dict[str, set] = defaultdict(set)
    med_indication: dict[str, Counter] = defaultdict(Counter)
    med_baseline: dict[str, set] = defaultdict(set)
    med_new: dict[str, set] = defaultdict(set)
    med_route: dict[str, str] = {}

    for pid, days in sims.items():
        seen_meds: set[str] = set()
        for d in days:
            for cm in d.get("CM", []):
                raw = cm.get("CMTRT", "")
                if isinstance(raw, dict):
                    name = str(raw.get("name", "")).strip()
                else:
                    name = str(raw).strip()
                if not name:
                    continue
                name_upper = name.upper()
                med_patients[name_upper].add(pid)
                raw_ind = cm.get("CMINDC", "")
                ind = str(raw_ind).strip() if not isinstance(raw_ind, dict) else str(raw_ind.get("name", "")).strip()
                if ind:
                    med_indication[name_upper][ind] += 1
                if cm.get("_baseline"):
                    med_baseline[name_upper].add(pid)
                elif name_upper not in seen_meds:
                    med_new[name_upper].add(pid)
                if name_upper not in med_route:
                    route = cm.get("CMROUTE", "")
                    if isinstance(raw, dict) and raw.get("route"):
                        route = raw["route"]
                    med_route[name_upper] = str(route)
                seen_meds.add(name_upper)

    table = []
    for med, pids in sorted(med_patients.items(),
                            key=lambda x: len(x[1]), reverse=True):
        cnt = len(pids)
        top_ind = med_indication[med].most_common(1)
        table.append({
            "medication": med.title(),
            "n": cnt,
            "pct": _pct(cnt, n),
            "baseline_n": len(med_baseline.get(med, set())),
            "new_n": len(med_new.get(med, set())),
            "indication": top_ind[0][0] if top_ind else "",
            "route": med_route.get(med, ""),
        })

    return {
        "n": n,
        "total_unique_meds": len(med_patients),
        "table": table,
    }


# ─── Section F: Treatment Exposure ──────────────────────────

def compute_treatment(sims: dict[str, list[dict]], rule_set: dict) -> dict:
    n = len(sims)
    durations, cycle_counts = [], []
    dose_red, dose_int, discont = 0, 0, 0
    ae_dose_red, ae_dose_int, ae_discont = 0, 0, 0
    per_drug_admin: dict[str, list[int]] = defaultdict(list)

    admin_sched = rule_set.get("administration_schedule", {})
    if isinstance(admin_sched, list):
        planned_cycle_length = admin_sched[0].get("cycle_length_days", 21) if admin_sched else 21
    else:
        planned_cycle_length = admin_sched.get("cycle_length_days", 21)

    for pid, days in sims.items():
        if not days:
            continue
        max_cycle = 1
        last_tx_day = 0
        had_red, had_int, had_disc = False, False, False
        had_ae_red, had_ae_int, had_ae_disc = False, False, False
        drug_admins: dict[str, int] = Counter()

        for d in days:
            day_num = d.get("day", 0)
            obj = d.get("objective", {})
            c = d.get("cycle", 1)
            if c and c > max_cycle:
                max_cycle = c

            ts = obj.get("treatment_status", "")
            if "held" in ts:
                had_int = True
            if "discontinued" in ts:
                had_disc = True

            for ae in d.get("AE", []):
                action = str(ae.get("AEACN", ""))
                if "WITHDRAWN" in action:
                    had_ae_disc = True
                if "INTERRUPT" in action:
                    had_ae_int = True
                if "REDUCED" in action:
                    had_ae_red = True

            for ec in d.get("EC", []):
                drug = ec.get("ECREFID") or ec.get("EXTRT", "Drug")
                drug_admins[drug] += 1
                ad = ec.get("ECSTDAT", day_num)
                last_tx_day = max(last_tx_day, ad)
                if ec.get("ECDOSADJ") or (ec.get("_dose_level") is not None and ec["_dose_level"] < 1.0):
                    had_red = True

        durations.append(last_tx_day)
        cycle_counts.append(max_cycle)
        if had_red:
            dose_red += 1
        if had_int:
            dose_int += 1
        if had_disc:
            discont += 1
        if had_ae_red:
            ae_dose_red += 1
        if had_ae_int:
            ae_dose_int += 1
        if had_ae_disc:
            ae_discont += 1

        for drug, cnt in drug_admins.items():
            per_drug_admin[drug].append(cnt)

    per_drug = {}
    for drug, admins in per_drug_admin.items():
        a = np.array(admins)
        planned_total = max(durations) / planned_cycle_length if planned_cycle_length and durations else None
        per_drug[drug] = {
            "n_admins": _desc_stats(admins),
            "rdi_median": round(float(np.median(a / planned_total * 100)), 1) if planned_total else None,
        }

    return {
        "n": n,
        "duration": _desc_stats(durations),
        "cycles": _desc_stats(cycle_counts),
        "dose_reduction_all": {"n": dose_red, "pct": _pct(dose_red, n)},
        "dose_interruption_all": {"n": dose_int, "pct": _pct(dose_int, n)},
        "discontinuation_all": {"n": discont, "pct": _pct(discont, n)},
        "ae_dose_reduction": {"n": ae_dose_red, "pct": _pct(ae_dose_red, n)},
        "ae_dose_interruption": {"n": ae_dose_int, "pct": _pct(ae_dose_int, n)},
        "ae_discontinuation": {"n": ae_discont, "pct": _pct(ae_discont, n)},
        "per_drug": per_drug,
    }


# ─── Master function ────────────────────────────────────────

def compute_csr_stats(run_dir: str | Path, mode: str = "natural") -> dict:
    """Compute all CSR statistics for a simulation run."""
    run_dir = Path(run_dir)
    profiles = _load_profiles(run_dir)
    sims = _load_sims(run_dir, sorted(profiles.keys()), mode)
    rule_set = _load_rule_set(run_dir)

    return {
        "run_id": run_dir.name,
        "mode": mode,
        "drug_name": rule_set.get("drug_name", "Unknown"),
        "indication": rule_set.get("indication", ""),
        "n_patients": len(profiles),
        "n_simulated": len(sims),
        "disposition": compute_disposition(sims),
        "demographics": compute_demographics(profiles),
        "efficacy": compute_efficacy(sims),
        "safety": compute_safety(sims),
        "labs": compute_labs(sims, profiles),
        "treatment": compute_treatment(sims, rule_set),
        "ecog_shift": compute_ecog_shift(sims, profiles),
        "concomitant_meds": compute_concomitant_meds(sims),
    }
