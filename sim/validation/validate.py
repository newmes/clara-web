"""
Master Validation Script

End-to-end: Extract stats → Compare → Generate report

Usage:
    python validation/validate.py <run_dir> <reference.json> [--mode natural] [-o output_dir]

Example:
    python validation/validate.py \
        data/runs/20260217_114656_Padcev___Pembrolizumab_50pt_126d \
        validation/reference_trials/ev302_padcev_pembro.json \
        --mode natural
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from extract_sim_stats import extract_all_stats
from compare import run_comparison
from report import generate_report


def validate(run_dir: str | Path,
             reference_path: str | Path,
             mode: str = "natural",
             output_dir: str | Path | None = None) -> dict:
    """Run the complete validation pipeline."""

    run_dir = Path(run_dir)
    reference_path = Path(reference_path)

    if output_dir is None:
        output_dir = run_dir / "validation"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference
    print(f"Loading reference data from {reference_path}...")
    with open(reference_path) as f:
        reference = json.load(f)
    trial_id = reference.get("_meta", {}).get("trial_id", "unknown")
    print(f"  → Trial: {trial_id}")

    # Step 1: Extract simulation statistics
    print("\n" + "=" * 60)
    print("STEP 1: Extracting simulation statistics")
    print("=" * 60)
    sim_stats = extract_all_stats(run_dir, mode)

    stats_path = output_dir / f"sim_stats_{mode}.json"
    with open(stats_path, "w") as f:
        json.dump(sim_stats, f, indent=2, ensure_ascii=False)
    print(f"  → Saved to {stats_path}")

    # Step 2: Compare
    print("\n" + "=" * 60)
    print("STEP 2: Comparing against reference")
    print("=" * 60)
    comparison = run_comparison(sim_stats, reference)

    comp_path = output_dir / f"comparison_{mode}.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"  → Saved to {comp_path}")

    # Step 3: Generate report
    print("\n" + "=" * 60)
    print("STEP 3: Generating report")
    print("=" * 60)
    md = generate_report(comparison)

    report_path = output_dir / f"validation_report_{mode}.md"
    with open(report_path, "w") as f:
        f.write(md)
    print(f"  → Report: {report_path}")

    # Summary
    score = comparison.get("overall_score", {})
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
    print(f"  Overall Rating: {score.get('overall_rating', '?')}")
    print(f"  Mean Score:     {score.get('mean_grade_score', 0):.2f}/4.0")
    gd = score.get("grade_distribution", {})
    print(f"  Grades:         A={gd.get('A',0)} B={gd.get('B',0)} "
          f"C={gd.get('C',0)} D={gd.get('D',0)}")
    st = score.get("statistical_tests", {})
    if st:
        print(f"  Stat Tests:     PASS={st.get('PASS',0)} "
              f"MARGINAL={st.get('MARGINAL',0)} FAIL={st.get('FAIL',0)}")
    print(f"\n  Full report: {report_path}")

    return comparison


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="End-to-end simulation validation")
    parser.add_argument("run_dir", help="Path to simulation run directory")
    parser.add_argument("reference", help="Path to reference trial JSON")
    parser.add_argument("--mode", default="natural",
                        choices=["natural", "care_ai"],
                        help="Simulation mode to validate")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    validate(args.run_dir, args.reference, args.mode, args.output)


if __name__ == "__main__":
    main()
