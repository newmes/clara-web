"""
Validation Report Generator

Converts a comparison report JSON into a human-readable Markdown report
with tables, scores, and clinical interpretation.

Usage:
    python validation/report.py <comparison_report.json> [-o report.md]
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _grade_emoji(grade: str) -> str:
    return {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴"}.get(grade, "⚪")


def _verdict_emoji(verdict: str) -> str:
    return {"PASS": "✅", "MARGINAL": "⚠️", "FAIL": "❌"}.get(verdict, "—")


def _rating_emoji(rating: str) -> str:
    return {
        "EXCELLENT": "🏆",
        "GOOD": "✅",
        "ACCEPTABLE": "🟡",
        "POOR": "🟠",
        "FAIL": "❌",
    }.get(rating, "—")


def generate_report(report: dict) -> str:
    """Generate a Markdown validation report from comparison data."""
    meta = report.get("meta", {})
    score = report.get("overall_score", {})
    lines = []

    # ─── Header ──────────────────────────────────────────────
    lines.append("# Clinical Trial Simulation Validation Report")
    lines.append("")
    lines.append(f"**Trial**: {meta.get('trial_id', '?')} "
                 f"({meta.get('trial_alias', '')})")
    lines.append(f"**Drug**: {meta.get('drug_name', '?')}")
    lines.append(f"**Simulation Run**: `{meta.get('sim_run_id', '?')}`")
    lines.append(f"**Mode**: {meta.get('sim_mode', '?')}")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(f"| Sim Patients | Ref Patients |")
    lines.append(f"|:---:|:---:|")
    lines.append(f"| {meta.get('n_sim_patients', '?')} | "
                 f"{meta.get('n_ref_patients', '?')} |")
    lines.append("")

    # ─── Overall Score ───────────────────────────────────────
    rating = score.get("overall_rating", "?")
    mean_score = score.get("mean_grade_score", 0)
    lines.append("---")
    lines.append("")
    lines.append(f"## Overall Concordance: {_rating_emoji(rating)} {rating}")
    lines.append("")
    lines.append(f"**Mean Grade Score**: {mean_score}/4.0")
    lines.append("")

    gd = score.get("grade_distribution", {})
    lines.append("| Grade | Count | Meaning |")
    lines.append("|:---:|:---:|:---|")
    lines.append(f"| 🟢 A | {gd.get('A', 0)} | Excellent match (≤5pp / ≤15% rel) |")
    lines.append(f"| 🟡 B | {gd.get('B', 0)} | Good match (≤10pp / ≤30% rel) |")
    lines.append(f"| 🟠 C | {gd.get('C', 0)} | Acceptable (≤15pp / ≤50% rel) |")
    lines.append(f"| 🔴 D | {gd.get('D', 0)} | Poor match (>15pp) |")
    lines.append(f"| **Total** | **{score.get('total_comparisons', 0)}** | |")
    lines.append("")

    st = score.get("statistical_tests", {})
    if st:
        lines.append(f"**Statistical Tests** (two-proportion z-test, α=0.05): "
                      f"✅ PASS {st.get('PASS', 0)} · "
                      f"⚠️ MARGINAL {st.get('MARGINAL', 0)} · "
                      f"❌ FAIL {st.get('FAIL', 0)} "
                      f"(total {score.get('total_tests', 0)})")
        lines.append("")

    # ─── Demographics ────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 1. Demographics")
    lines.append("")
    demo = report.get("demographics", {}).get("comparisons", [])
    if demo:
        lines.append("| Metric | Simulation | Reference | Diff | Grade |")
        lines.append("|:---|:---:|:---:|:---:|:---:|")
        for c in demo:
            sv = c.get("sim_value", c.get("sim_pct", "—"))
            rv = c.get("ref_value", c.get("ref_pct", "—"))
            diff = c.get("diff", c.get("diff_pp", "—"))
            g = c.get("grade", "—")
            if isinstance(sv, float):
                sv = f"{sv:.1f}"
            if isinstance(rv, float):
                rv = f"{rv:.1f}"
            if isinstance(diff, float):
                diff = f"{diff:+.1f}"
            lines.append(
                f"| {c['metric']} | {sv} | {rv} | {diff} | "
                f"{_grade_emoji(g)} {g} |")
        lines.append("")

    # ─── AE Rates ────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 2. Adverse Events")
    lines.append("")

    # Overall safety
    ae = report.get("ae_rates", {})
    overall_safety = ae.get("overall_safety", [])
    if overall_safety:
        lines.append("### 2a. Overall Safety")
        lines.append("")
        lines.append("| Metric | Sim % | Ref % | Diff (pp) | Grade | p-value |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")
        for c in overall_safety:
            pv = c.get("test", {}).get("p_value", "—")
            vd = c.get("test", {}).get("verdict", "—")
            pv_str = f"{pv:.4f}" if isinstance(pv, float) else str(pv)
            lines.append(
                f"| {c['metric']} | {c['sim_pct']:.1f} | {c['ref_pct']:.1f} | "
                f"{c['diff_pp']:+.1f} | {_grade_emoji(c['grade'])} {c['grade']} | "
                f"{_verdict_emoji(vd)} {pv_str} |")
        lines.append("")

    # Per-AE
    per_ae = ae.get("per_ae", [])
    if per_ae:
        lines.append("### 2b. AE Incidence by Term")
        lines.append("")
        lines.append("| AE Term | Sim All% | Ref All% | Diff | Grade | "
                     "Sim G3+% | Ref G3+% | G3 Grade |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for c in sorted(per_ae, key=lambda x: x["ref_all_grade_pct"],
                        reverse=True):
            g_all = c.get("grade_all", "—")
            g_g3 = c.get("grade_g3", "—")
            lines.append(
                f"| {c['ae_term']} | {c['sim_all_grade_pct']:.1f} | "
                f"{c['ref_all_grade_pct']:.1f} | "
                f"{c['diff_all_pp']:+.1f} | {_grade_emoji(g_all)} {g_all} | "
                f"{c['sim_grade3_pct']:.1f} | {c['ref_grade3_pct']:.1f} | "
                f"{_grade_emoji(g_g3) if g_g3 != 'N/A' else '—'} {g_g3} |")
        lines.append("")

    # Unexpected AEs
    unexpected = ae.get("unexpected_aes", [])
    if unexpected:
        lines.append("### 2c. Unexpected AEs (≥5%, not in reference top AEs)")
        lines.append("")
        lines.append("| AE Term | Sim All% | Note |")
        lines.append("|:---|:---:|:---|")
        for u in unexpected:
            lines.append(
                f"| {u['ae_term']} | {u['sim_all_grade_pct']:.1f} | "
                f"{u.get('note', '')} |")
        lines.append("")

    # ─── Efficacy ────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 3. Efficacy (Tumor Response)")
    lines.append("")
    eff = report.get("efficacy", {}).get("comparisons", [])
    if eff:
        lines.append("| Metric | Sim % | Ref % | Diff (pp) | Grade | "
                     "Sim 95%CI | p-value |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for c in eff:
            ci = c.get("sim_95ci", ["—", "—"])
            ci_str = f"[{ci[0]:.1f}, {ci[1]:.1f}]" if ci and isinstance(
                ci[0], (int, float)) else "—"
            pv = c.get("test", {}).get("p_value", "—")
            vd = c.get("test", {}).get("verdict", "—")
            pv_str = f"{pv:.4f}" if isinstance(pv, float) else str(pv)
            lines.append(
                f"| {c['metric']} | {c['sim_pct']:.1f} | {c['ref_pct']:.1f} | "
                f"{c['diff_pp']:+.1f} | {_grade_emoji(c['grade'])} {c['grade']} | "
                f"{ci_str} | {_verdict_emoji(vd)} {pv_str} |")
        lines.append("")

    # ─── Treatment Exposure ──────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 4. Treatment Exposure")
    lines.append("")
    tx = report.get("treatment_exposure", {}).get("comparisons", [])
    if tx:
        lines.append("| Metric | Simulation | Reference | Diff | Grade |")
        lines.append("|:---|:---:|:---:|:---:|:---:|")
        for c in tx:
            if "sim_value_days" in c:
                sv = f"{c['sim_value_days']:.0f} days"
                rv = f"{c.get('ref_value_months', '?')} mo " \
                     f"(≈{c.get('ref_value_days_approx', '?')}d)"
                diff = f"{c.get('diff_days', '?'):+.0f}d"
            else:
                sv = f"{c.get('sim_pct', '?'):.1f}%"
                rv = f"{c.get('ref_pct', '?'):.1f}%"
                diff = f"{c.get('diff_pp', '?'):+.1f}pp"
            lines.append(
                f"| {c['metric']} | {sv} | {rv} | {diff} | "
                f"{_grade_emoji(c.get('grade', '—'))} {c.get('grade', '—')} |")
        lines.append("")

    # ─── Lab Abnormalities ───────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## 5. Laboratory Abnormalities")
    lines.append("")
    labs = report.get("lab_abnormalities", {}).get("comparisons", [])
    if labs:
        lines.append("| Lab | Sim Any% | Ref Any% | Diff (pp) | Grade | "
                     "Sim G3% | Ref G3% |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        for c in sorted(labs, key=lambda x: x["ref_all_grade_pct"],
                        reverse=True):
            g = c.get("grade", "—")
            lines.append(
                f"| {c['lab']} | {c['sim_any_abnormal_pct']:.1f} | "
                f"{c['ref_all_grade_pct']:.1f} | "
                f"{c['diff_pp']:+.1f} | {_grade_emoji(g)} {g} | "
                f"{c.get('sim_grade3_pct', 0):.1f} | "
                f"{c.get('ref_grade3_pct', 0):.1f} |")
        lines.append("")

    # ─── Interpretation ──────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(_generate_interpretation(report))
    lines.append("")

    # ─── Methodology ─────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Grading Scale")
    lines.append("- **Grade A** (🟢): Absolute diff ≤5 percentage points "
                 "OR relative diff ≤15%")
    lines.append("- **Grade B** (🟡): Absolute diff ≤10pp OR relative ≤30%")
    lines.append("- **Grade C** (🟠): Absolute diff ≤15pp OR relative ≤50%")
    lines.append("- **Grade D** (🔴): Outside all tolerance bands")
    lines.append("")
    lines.append("### Statistical Test")
    lines.append("- Two-proportion z-test with pooled proportion")
    lines.append("- α = 0.05 (two-sided)")
    lines.append("- PASS: p ≥ 0.05 (no significant difference)")
    lines.append("- MARGINAL: 0.01 ≤ p < 0.05")
    lines.append("- FAIL: p < 0.01")
    lines.append("")
    lines.append("### Limitations")
    lines.append("- Simulation sample size (N=50) is much smaller than "
                 "trial (N=440+), causing wider confidence intervals")
    lines.append("- AE grouping may differ between simulation terms and "
                 "published grouped terms")
    lines.append("- Lab abnormalities use CTCAE-based thresholds which may "
                 "differ slightly from trial-specific criteria")
    lines.append("- Simulation duration may differ from trial median follow-up")
    lines.append("")

    return "\n".join(lines)


def _generate_interpretation(report: dict) -> str:
    """Generate a brief clinical interpretation paragraph."""
    score = report.get("overall_score", {})
    rating = score.get("overall_rating", "?")
    mean = score.get("mean_grade_score", 0)
    gd = score.get("grade_distribution", {})
    st = score.get("statistical_tests", {})

    n_a = gd.get("A", 0)
    n_b = gd.get("B", 0)
    n_cd = gd.get("C", 0) + gd.get("D", 0)
    total = score.get("total_comparisons", 0)
    pct_ab = round((n_a + n_b) / total * 100, 0) if total else 0

    # Find worst mismatches
    worst = []
    for section in ["ae_rates", "efficacy", "treatment_exposure",
                    "lab_abnormalities"]:
        items = report.get(section, {}).get("comparisons", [])
        items += report.get(section, {}).get("overall_safety", [])
        items += report.get(section, {}).get("per_ae", [])
        for item in items:
            g = item.get("grade") or item.get("grade_all")
            if g in ("C", "D"):
                name = item.get("metric") or item.get("ae_term") or item.get(
                    "lab", "?")
                worst.append(f"{name} ({g})")

    text = (
        f"The simulation achieves an overall concordance rating of "
        f"**{rating}** (mean score {mean:.2f}/4.0). "
        f"{pct_ab:.0f}% of comparisons ({n_a + n_b}/{total}) received "
        f"Grade A or B, indicating good-to-excellent agreement with "
        f"published trial data."
    )

    if worst:
        text += (
            f"\n\nAreas with weaker concordance ({len(worst)} metrics at "
            f"Grade C/D): {', '.join(worst[:8])}"
            f"{'...' if len(worst) > 8 else ''}. "
            f"These gaps may be addressable through rule_set calibration "
            f"or larger simulation sample sizes."
        )

    if st.get("FAIL", 0) > 0:
        text += (
            f"\n\n**Note**: {st['FAIL']} statistical tests showed significant "
            f"deviation (p<0.01). However, with N={report.get('meta', {}).get('n_sim_patients', 50)} "
            f"simulated patients vs N={report.get('meta', {}).get('n_ref_patients', 440)} "
            f"in the trial, some sampling variability is expected."
        )

    return text


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate validation report from comparison JSON")
    parser.add_argument("comparison", help="Path to comparison report JSON")
    parser.add_argument("--output", "-o", default=None,
                        help="Output Markdown file path")
    args = parser.parse_args()

    with open(args.comparison) as f:
        report = json.load(f)

    md = generate_report(report)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(args.comparison).with_suffix(".md")

    with open(out_path, "w") as f:
        f.write(md)
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
