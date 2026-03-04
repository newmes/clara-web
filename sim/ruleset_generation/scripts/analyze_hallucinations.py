#!/usr/bin/env python3
"""Analyze rule set JSONs for hallucination patterns.

Checks:
1. Severity distribution: mechanical 50/30/15/5 ratios
2. Onset plausibility: suspicious round numbers, low diversity
3. Trigger monotonicity: identical trigger patterns across AEs
4. Frequency sum: severity grades should sum to frequency_pct
5. Flat frequency: ≤3 unique frequency values across 8+ AEs
6. FAERS coverage: empty FAERS data (needs agent log)

Outputs a per-file and aggregate report with PASS/FAIL per check.
Exit code 0 = all clean, 1 = hallucinations detected.
"""

import json
import sys
from collections import Counter
from pathlib import Path


def check_severity_fabrication(aes: list[dict]) -> list[str]:
    """Detect mechanical grade distribution ratios."""
    issues = []
    if len(aes) < 3:
        return issues

    grade1_ratios = []
    for ae in aes:
        sd = ae.get("severity_distribution", {})
        freq = ae.get("frequency_pct", 0)
        if freq <= 0 or not sd:
            continue
        g1 = sd.get("grade_1", 0)
        ratio = round(g1 / freq, 2)
        grade1_ratios.append(ratio)

    if not grade1_ratios:
        return issues

    # Check if >50% of AEs have identical grade_1/freq ratio
    counter = Counter(grade1_ratios)
    most_common_ratio, count = counter.most_common(1)[0]
    pct_identical = count / len(grade1_ratios) * 100
    if pct_identical > 50:
        issues.append(
            f"SEVERITY_FABRICATION: {pct_identical:.0f}% of AEs have identical grade_1/freq ratio={most_common_ratio} "
            f"({count}/{len(grade1_ratios)} AEs)"
        )

    # Check for exact 50/30/15/5 pattern
    n_mechanical = 0
    for ae in aes:
        sd = ae.get("severity_distribution", {})
        freq = ae.get("frequency_pct", 0)
        if freq <= 0 or not sd:
            continue
        g1 = sd.get("grade_1", 0)
        g2 = sd.get("grade_2", 0)
        g3 = sd.get("grade_3", 0)
        g4 = sd.get("grade_4", 0)
        # Check if ratios are ~50/30/15/5
        if freq > 0:
            r1 = round(g1 / freq * 100)
            r2 = round(g2 / freq * 100)
            r3 = round(g3 / freq * 100)
            if r1 == 50 and r2 == 30 and r3 == 15:
                n_mechanical += 1
    if n_mechanical >= 3:
        issues.append(
            f"MECHANICAL_5030155: {n_mechanical}/{len(aes)} AEs follow exact 50/30/15/5 grade split"
        )

    return issues


def check_onset_plausibility(aes: list[dict]) -> list[str]:
    """Detect suspicious onset patterns."""
    issues = []
    onsets = [ae.get("median_onset_days") for ae in aes if ae.get("median_onset_days") is not None]
    if len(onsets) < 5:
        return issues

    # Check round numbers (multiples of 7)
    mult7 = sum(1 for o in onsets if o % 7 == 0)
    pct_mult7 = mult7 / len(onsets) * 100
    if pct_mult7 > 60:
        issues.append(
            f"ONSET_ROUND_NUMBERS: {pct_mult7:.0f}% of onsets are multiples of 7 ({mult7}/{len(onsets)})"
        )

    # Check multiples of 10
    mult10 = sum(1 for o in onsets if o % 10 == 0)
    pct_mult10 = mult10 / len(onsets) * 100
    if pct_mult10 > 50:
        issues.append(
            f"ONSET_MULT_10: {pct_mult10:.0f}% of onsets are multiples of 10 ({mult10}/{len(onsets)})"
        )

    # Check diversity
    unique = len(set(onsets))
    if unique <= 3 and len(onsets) >= 8:
        issues.append(
            f"ONSET_LOW_DIVERSITY: only {unique} unique onset values across {len(onsets)} AEs: {sorted(set(onsets))}"
        )

    return issues


def check_trigger_monotonicity(aes: list[dict]) -> list[str]:
    """Detect identical trigger patterns across all AEs."""
    issues = []
    trigger_sigs = []
    for ae in aes:
        triggers = ae.get("triggers", [])
        if not triggers:
            continue
        sig = tuple(
            (t.get("condition", ""), t.get("probability_pct", 0))
            for t in sorted(triggers, key=lambda x: x.get("condition", ""))
        )
        trigger_sigs.append(sig)

    if len(trigger_sigs) < 3:
        return issues

    counter = Counter(trigger_sigs)
    most_common_sig, count = counter.most_common(1)[0]
    pct_identical = count / len(trigger_sigs) * 100
    if pct_identical > 60:
        # Format the signature
        sig_str = ", ".join(f"{c}:{p}%" for c, p in most_common_sig)
        issues.append(
            f"TRIGGER_MONOTONIC: {pct_identical:.0f}% of triggered AEs share identical pattern "
            f"[{sig_str}] ({count}/{len(trigger_sigs)})"
        )

    return issues


def check_faers_coverage(agent_log: dict | None) -> list[str]:
    """Check if FAERS returned data (from agent log)."""
    issues = []
    if agent_log is None:
        return issues
    prompt = agent_log.get("evidence_prompt", "") or agent_log.get("prompt", "")
    if "No data found" in prompt and "FAERS" in prompt:
        issues.append("FAERS_EMPTY: FAERS returned 'No data found' for this drug")
    return issues


def check_flat_frequency(aes: list[dict]) -> list[str]:
    """Detect flat-frequency fabrication (all AEs share ≤3 unique frequency values)."""
    issues = []
    freqs = [ae.get("frequency_pct", 0) for ae in aes if ae.get("frequency_pct", 0) > 0]
    if len(freqs) < 8:
        return issues
    unique = len(set(freqs))
    if unique <= 3:
        sorted_vals = sorted(set(freqs))
        issues.append(
            f"FLAT_FREQUENCY: only {unique} unique frequency_pct values across {len(freqs)} AEs: {sorted_vals}"
        )
    return issues


def check_frequency_sum(aes: list[dict]) -> list[str]:
    """Check that severity grades sum to frequency_pct."""
    issues = []
    for ae in aes:
        freq = ae.get("frequency_pct", 0)
        sd = ae.get("severity_distribution", {})
        if not sd or freq <= 0:
            continue
        grade_sum = sum(sd.values())
        if abs(grade_sum - freq) > 0.5:
            issues.append(
                f"GRADE_SUM_MISMATCH: {ae['event']}: grades sum to {grade_sum:.1f} but frequency_pct={freq}"
            )
    return issues


def _normalize_target_aes(ae_profile: list[dict]) -> list[dict]:
    """Convert target-schema ae_profile entries back to internal format for checks."""
    result = []
    for ae in ae_profile:
        incidence = ae.get("incidence_all_grade", 0)
        freq_pct = round(incidence * 100, 2)
        gd = ae.get("grade_distribution", {})
        # Target schema grades are proportions (sum ~1.0); convert back to absolute %
        sev = {}
        for g in ["1", "2", "3", "4", "5"]:
            sev[f"grade_{g}"] = round(gd.get(g, 0) * freq_pct, 2)
        onset_dist = ae.get("onset_day", {})
        onset_mean = onset_dist.get("params", {}).get("mean", 14) if isinstance(onset_dist, dict) else 14
        result.append({
            "event": ae.get("ae_term", "Unknown"),
            "frequency_pct": freq_pct,
            "severity_distribution": sev,
            "median_onset_days": onset_mean,
            "triggers": [],  # triggers are in dose_modification_rules / ae_cascade_rules
            "reversible": ae.get("reversible", True),
        })
    return result


def analyze_rule_set(path: Path, agent_log_path: Path | None = None) -> dict:
    """Analyze a single rule set file. Returns report dict."""
    data = json.loads(path.read_text())
    # Support both internal format (adverse_events) and target schema (ae_profile)
    if "adverse_events" in data:
        aes = data.get("adverse_events", [])
    elif "ae_profile" in data:
        aes = _normalize_target_aes(data.get("ae_profile", []))
    else:
        aes = []

    agent_log = None
    if agent_log_path and agent_log_path.exists():
        agent_log = json.loads(agent_log_path.read_text())

    # Support both internal format (drugs list) and target schema (drug_name string)
    drugs = data.get("drugs", [])
    if not drugs and "drug_name" in data:
        drugs = data["drug_name"].split(" + ")
    report = {
        "file": path.name,
        "drugs": drugs,
        "indication": data.get("indication", ""),
        "n_aes": len(aes),
        "checks": {},
        "all_pass": True,
    }

    checks = [
        ("severity_fabrication", check_severity_fabrication(aes)),
        ("onset_plausibility", check_onset_plausibility(aes)),
        ("trigger_monotonicity", check_trigger_monotonicity(aes)),
        ("frequency_sum", check_frequency_sum(aes)),
        ("flat_frequency", check_flat_frequency(aes)),
        ("faers_coverage", check_faers_coverage(agent_log)),
    ]

    for name, issues in checks:
        passed = len(issues) == 0
        report["checks"][name] = {"pass": passed, "issues": issues}
        if not passed:
            report["all_pass"] = False

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze rule sets for hallucination patterns")
    parser.add_argument("dir", type=Path, help="Directory containing *_rules.json files")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on any failure")
    args = parser.parse_args()

    # Find base.json files in subdirectories (split format)
    rule_files = sorted(args.dir.glob("*/base.json"))
    if not rule_files:
        print(f"No */base.json files found in {args.dir}")
        sys.exit(1)

    reports = []
    for rf in rule_files:
        # Agent log is flat in the parent dir, keyed by subdir name
        subdir_name = rf.parent.name
        log_path = rf.parent.parent / f"{subdir_name}_agent_log.json"
        report = analyze_rule_set(rf, log_path if log_path.exists() else None)
        reports.append(report)

    if args.json:
        print(json.dumps(reports, indent=2))
        sys.exit(0)

    # Pretty print
    total_pass = 0
    total_fail = 0
    for r in reports:
        drug_label = " + ".join(r["drugs"])
        status = "\033[32mPASS\033[0m" if r["all_pass"] else "\033[31mFAIL\033[0m"
        print(f"\n{'='*70}")
        print(f"{drug_label} / {r['indication']} [{status}] ({r['n_aes']} AEs)")
        print(f"  File: {r['file']}")

        for check_name, check_data in r["checks"].items():
            icon = "  PASS" if check_data["pass"] else "  FAIL"
            color = "\033[32m" if check_data["pass"] else "\033[31m"
            print(f"  {color}{icon}\033[0m {check_name}")
            for issue in check_data["issues"]:
                print(f"        {issue}")

        if r["all_pass"]:
            total_pass += 1
        else:
            total_fail += 1

    print(f"\n{'='*70}")
    print(f"SUMMARY: {total_pass} passed, {total_fail} failed out of {len(reports)} rule sets")

    if args.strict and total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
