#!/usr/bin/env python3
"""Compare before/after rule sets to measure pipeline improvements.

Usage:
    python scripts/compare_rule_sets.py --before rule_sets_before/ --after rule_sets_after/
    python scripts/compare_rule_sets.py --dir rule_sets/  # analyze a single directory

Metrics:
    1. FAERS coverage — % of drugs with non-empty OpenFDA FAERS data
    2. Onset diversity — % of AE onsets that are NOT multiples of 7
    3. Trigger diversity — unique trigger patterns / total AEs
    4. Severity diversity — unique grade_1 ratios / total AEs
    5. OnSIDES cross-check — % of rules with boxed warning AEs present
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load_rule_sets(directory: Path) -> dict[str, dict]:
    """Load all *_rules.json files from a directory."""
    rule_sets = {}
    for f in sorted(directory.glob("*_rules.json")):
        data = json.loads(f.read_text())
        key = f.stem.replace("_rules", "")
        rule_sets[key] = data
    return rule_sets


def analyze_onset_diversity(rule_set: dict) -> dict:
    """Analyze AE onset values for suspicious patterns."""
    aes = rule_set.get("adverse_events", [])
    if not aes:
        return {"total": 0, "non_round_pct": 0.0, "unique_count": 0}

    onsets = [ae.get("median_onset_days", 0) for ae in aes]
    onsets = [o for o in onsets if o > 0]
    if not onsets:
        return {"total": len(aes), "non_round_pct": 0.0, "unique_count": 0}

    multiples_of_7 = sum(1 for o in onsets if o % 7 == 0)
    non_round = len(onsets) - multiples_of_7
    non_round_pct = round(non_round / len(onsets) * 100, 1) if onsets else 0.0

    return {
        "total": len(onsets),
        "non_round_pct": non_round_pct,
        "unique_count": len(set(onsets)),
        "multiples_of_7": multiples_of_7,
        "values": sorted(set(onsets)),
    }


def analyze_trigger_diversity(rule_set: dict) -> dict:
    """Analyze trigger pattern diversity across AEs."""
    aes = rule_set.get("adverse_events", [])
    if not aes:
        return {"total": 0, "unique_patterns": 0, "diversity_pct": 0.0}

    patterns = []
    for ae in aes:
        triggers = ae.get("triggers", [])
        pattern = tuple(
            (t.get("target_ae"), t.get("condition"), t.get("probability_pct"))
            for t in triggers
        )
        patterns.append(pattern)

    unique = len(set(patterns))
    diversity_pct = round(unique / len(patterns) * 100, 1) if patterns else 0.0

    return {
        "total": len(patterns),
        "unique_patterns": unique,
        "diversity_pct": diversity_pct,
    }


def analyze_severity_diversity(rule_set: dict) -> dict:
    """Analyze severity distribution diversity across AEs."""
    aes = rule_set.get("adverse_events", [])
    if not aes:
        return {"total": 0, "unique_ratios": 0, "diversity_pct": 0.0}

    grade1_ratios = []
    for ae in aes:
        freq = ae.get("frequency_pct", 0)
        dist = ae.get("severity_distribution", {})
        g1 = dist.get("grade_1", 0)
        if freq > 0:
            ratio = round(g1 / freq, 2)
            grade1_ratios.append(ratio)

    if not grade1_ratios:
        return {"total": len(aes), "unique_ratios": 0, "diversity_pct": 0.0}

    ratio_counts = Counter(grade1_ratios)
    most_common_ratio, most_common_count = ratio_counts.most_common(1)[0]
    unique = len(set(grade1_ratios))
    diversity_pct = round(unique / len(grade1_ratios) * 100, 1)

    return {
        "total": len(grade1_ratios),
        "unique_ratios": unique,
        "diversity_pct": diversity_pct,
        "most_common_ratio": most_common_ratio,
        "most_common_count": most_common_count,
    }


def analyze_single(name: str, rule_set: dict) -> dict:
    """Produce a full analysis for one rule set."""
    return {
        "name": name,
        "drugs": rule_set.get("drugs", []),
        "indication": rule_set.get("indication", ""),
        "ae_count": len(rule_set.get("adverse_events", [])),
        "onset": analyze_onset_diversity(rule_set),
        "triggers": analyze_trigger_diversity(rule_set),
        "severity": analyze_severity_diversity(rule_set),
    }


def print_analysis(label: str, analyses: list[dict]) -> None:
    """Print summary metrics for a set of rule sets."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    for a in analyses:
        drug_str = " + ".join(a["drugs"])
        print(f"\n  {drug_str} / {a['indication']}")
        print(f"    AEs: {a['ae_count']}")

        onset = a["onset"]
        print(
            f"    Onset: {onset['non_round_pct']}% non-round, "
            f"{onset['unique_count']} unique values out of {onset['total']}"
        )
        if onset.get("values"):
            print(f"           values: {onset['values']}")

        trig = a["triggers"]
        print(
            f"    Triggers: {trig['unique_patterns']}/{trig['total']} unique patterns "
            f"({trig['diversity_pct']}% diverse)"
        )

        sev = a["severity"]
        print(
            f"    Severity: {sev['unique_ratios']}/{sev['total']} unique grade_1 ratios "
            f"({sev['diversity_pct']}% diverse)"
        )
        if sev.get("most_common_ratio") is not None:
            print(
                f"              most common ratio: {sev['most_common_ratio']} "
                f"(appears {sev['most_common_count']}x)"
            )

    # Aggregate metrics
    if analyses:
        avg_onset_div = sum(a["onset"]["non_round_pct"] for a in analyses) / len(analyses)
        avg_trigger_div = sum(a["triggers"]["diversity_pct"] for a in analyses) / len(analyses)
        avg_severity_div = sum(a["severity"]["diversity_pct"] for a in analyses) / len(analyses)

        print(f"\n  --- Aggregate for {label} ---")
        print(f"    Avg onset non-round %:    {avg_onset_div:.1f}%")
        print(f"    Avg trigger diversity %:  {avg_trigger_div:.1f}%")
        print(f"    Avg severity diversity %: {avg_severity_div:.1f}%")


def print_comparison(before: list[dict], after: list[dict]) -> None:
    """Print side-by-side comparison of before/after metrics."""
    before_map = {a["name"]: a for a in before}
    after_map = {a["name"]: a for a in after}

    common_keys = sorted(set(before_map) & set(after_map))
    if not common_keys:
        print("\nNo matching rule sets found for comparison.")
        return

    print(f"\n{'=' * 70}")
    print("  BEFORE vs AFTER COMPARISON")
    print(f"{'=' * 70}")

    for key in common_keys:
        b = before_map[key]
        a = after_map[key]
        drug_str = " + ".join(b["drugs"])
        print(f"\n  {drug_str} / {b['indication']}")

        # Onset
        b_onset = b["onset"]["non_round_pct"]
        a_onset = a["onset"]["non_round_pct"]
        delta_onset = a_onset - b_onset
        arrow = "+" if delta_onset > 0 else ""
        print(f"    Onset non-round: {b_onset}% -> {a_onset}% ({arrow}{delta_onset:.1f}%)")

        # Triggers
        b_trig = b["triggers"]["diversity_pct"]
        a_trig = a["triggers"]["diversity_pct"]
        delta_trig = a_trig - b_trig
        arrow = "+" if delta_trig > 0 else ""
        print(f"    Trigger diversity: {b_trig}% -> {a_trig}% ({arrow}{delta_trig:.1f}%)")

        # Severity
        b_sev = b["severity"]["diversity_pct"]
        a_sev = a["severity"]["diversity_pct"]
        delta_sev = a_sev - b_sev
        arrow = "+" if delta_sev > 0 else ""
        print(f"    Severity diversity: {b_sev}% -> {a_sev}% ({arrow}{delta_sev:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rule set quality metrics")
    parser.add_argument("--before", type=Path, help="Directory with baseline rule sets")
    parser.add_argument("--after", type=Path, help="Directory with improved rule sets")
    parser.add_argument("--dir", type=Path, help="Analyze a single directory")
    args = parser.parse_args()

    if args.dir:
        rule_sets = load_rule_sets(args.dir)
        if not rule_sets:
            print(f"No *_rules.json files found in {args.dir}")
            sys.exit(1)
        analyses = [analyze_single(name, rs) for name, rs in rule_sets.items()]
        print_analysis(str(args.dir), analyses)
    elif args.before and args.after:
        before_rs = load_rule_sets(args.before)
        after_rs = load_rule_sets(args.after)
        before_analyses = [analyze_single(name, rs) for name, rs in before_rs.items()]
        after_analyses = [analyze_single(name, rs) for name, rs in after_rs.items()]
        print_analysis("BEFORE", before_analyses)
        print_analysis("AFTER", after_analyses)
        print_comparison(before_analyses, after_analyses)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
