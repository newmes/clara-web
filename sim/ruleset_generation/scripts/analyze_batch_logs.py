#!/usr/bin/env python3
"""Analyze agent logs and rule sets across all drugs.

Extracts:
  1. Stage 1 sub-call success rates (response_len > 0)
  2. Stage 2 grounding success (no error entry)
  3. Auto-correction trigger rates (warning message patterns)
  4. Evidence source coverage (from run_logs/ stdout)
  5. AE naming quality (lab codes vs clinical terms)
  6. Comorbidity completeness (impacts_dosing + ae_risk_modifiers)

Usage:
  python scripts/analyze_batch_logs.py [rule_sets_dir] [--run-logs run_logs/]
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ── AE naming quality: lab-code patterns that should be clinical terms ──
LAB_CODE_PATTERNS = [
    (re.compile(r"(?i)decreased\s+(lymphocytes|platelets|neutrophils|hemoglobin|white blood cells)"), "lab-code"),
    (re.compile(r"(?i)increased\s+(ALT|AST|bilirubin|creatinine|alkaline phosphatase)"), "lab-code"),
    (re.compile(r"(?i)<\d+.*cells/mm|>?\d+.*mg/dL|>?\d+.*U/L"), "lab-value"),
    (re.compile(r"(?i)(blood|serum)\s+\w+\s+(decreased|increased|elevated)"), "lab-code"),
]

# Original 10 drugs (for comparison grouping)
ORIGINAL_10 = {
    "pembrolizumab_melanoma",
    "osimertinib_non-small_cell_lung_cancer",
    "sotorasib_non-small_cell_lung_cancer",
    "lenalidomide_multiple_myeloma",
    "sacituzumab_govitecan_triple-negative_breast_cancer",
    "rituximab_diffuse_large_b-cell_lymphoma",
    "ibrutinib_chronic_lymphocytic_leukemia",
    "ipilimumab+nivolumab_melanoma",
    "docetaxel_prostate_cancer",
    "venetoclax+obinutuzumab_chronic_lymphocytic_leukemia",
}


def classify_drug(stem: str) -> str:
    """Return 'original' or 'new' based on the file stem."""
    # Normalize: remove _rules or _agent_log suffix
    base = stem.replace("_rules", "").replace("_agent_log", "")
    return "original" if base in ORIGINAL_10 else "new"


def analyze_agent_log(log_path: Path) -> dict:
    """Extract metrics from a single agent log."""
    with open(log_path) as f:
        log = json.load(f)

    result = {
        "file": log_path.name,
        "timestamp": log.get("timestamp", ""),
        "stage1_calls": {},
        "stage2_success": None,
        "stage2_error": None,
        "stage3_response_len": 0,
        "warning_counts": Counter(),
        "warnings_total": 0,
    }

    # Parse stage_logs
    for entry in log.get("stage_logs", []):
        stage = entry.get("stage", "")

        if stage.startswith("stage1_"):
            sub = stage.replace("stage1_", "")
            resp_len = entry.get("response_len", 0)
            result["stage1_calls"][sub] = {
                "success": resp_len > 0,
                "response_len": resp_len,
            }

        elif stage == "stage2_grounding":
            if "error" in entry:
                result["stage2_success"] = False
                result["stage2_error"] = entry["error"]
            elif result["stage2_success"] is None:
                result["stage2_success"] = True
                result["stage2_response_len"] = entry.get("response_len", 0)

        elif stage == "stage3_synthesis":
            result["stage3_response_len"] = entry.get("response_len", 0)

    # Parse warnings
    for w in log.get("warnings", []):
        result["warnings_total"] += 1
        if "severity" in w.lower() and ("auto-correct" in w.lower() or "identical" in w.lower()):
            result["warning_counts"]["severity_correction"] += 1
        elif "onset" in w.lower() and ("auto-correct" in w.lower() or "fabricated" in w.lower()):
            result["warning_counts"]["onset_correction"] += 1
        elif "trigger" in w.lower() and ("auto-correct" in w.lower() or "identical" in w.lower()):
            result["warning_counts"]["trigger_correction"] += 1
        elif "sex" in w.lower() and ("normaliz" in w.lower() or "auto-correct" in w.lower()):
            result["warning_counts"]["sex_normalization"] += 1
        elif "dailymed" in w.lower() and ("auto-correct" in w.lower() or "frequency" in w.lower() or "cap" in w.lower()):
            result["warning_counts"]["dailymed_correction"] += 1
        elif "onsides" in w.lower() or "boxed warning" in w.lower():
            result["warning_counts"]["onsides_check"] += 1
        elif "combo" in w.lower() and "source_drug" in w.lower():
            result["warning_counts"]["combo_source_fix"] += 1
        elif "irae" in w.lower() or "immune-related" in w.lower() or "missing" in w.lower():
            result["warning_counts"]["missing_aes"] += 1
        elif "severity cap" in w.lower():
            result["warning_counts"]["severity_cap"] += 1
        else:
            result["warning_counts"]["other"] += 1

    return result


def analyze_rule_set(rules_path: Path) -> dict:
    """Extract quality metrics from a rule set JSON."""
    with open(rules_path) as f:
        rules = json.load(f)

    result = {
        "file": rules_path.name,
        "drugs": rules.get("drugs", []),
        "indication": rules.get("indication", ""),
        "num_aes": 0,
        "lab_code_aes": [],
        "comorbidity_issues": [],
        "ae_names": [],
        "frequency_stats": {},
        "has_efficacy": bool(rules.get("efficacy")),
        "has_regimen": bool(rules.get("regimen")),
    }

    # AE analysis
    aes = rules.get("adverse_events", [])
    result["num_aes"] = len(aes)
    freqs = []
    for ae in aes:
        name = ae.get("event", "")
        result["ae_names"].append(name)
        freq = ae.get("frequency_pct", 0)
        if freq:
            freqs.append(freq)

        # Check for lab-code style names
        for pat, label in LAB_CODE_PATTERNS:
            if pat.search(name):
                result["lab_code_aes"].append(name)
                break

    if freqs:
        result["frequency_stats"] = {
            "min": min(freqs),
            "max": max(freqs),
            "mean": sum(freqs) / len(freqs),
            "count": len(freqs),
        }

    # Comorbidity analysis
    for comorb in rules.get("comorbidities", []):
        condition = comorb.get("condition", "")
        impacts = comorb.get("impacts_dosing", False)
        modifiers = comorb.get("ae_risk_modifiers", [])
        if impacts and not modifiers:
            result["comorbidity_issues"].append(
                f"{condition}: impacts_dosing=True but ae_risk_modifiers empty"
            )

    return result


def analyze_run_log(log_path: Path) -> dict:
    """Extract evidence source coverage from a run_logs/ stdout file."""
    result = {
        "file": log_path.name,
        "sources": {},
    }

    source_patterns = {
        "DailyMed": re.compile(r"(?i)dailymed.*(found|retrieved|fetched|success|label)", re.IGNORECASE),
        "DrugBank": re.compile(r"(?i)drugbank.*(found|success|loaded)", re.IGNORECASE),
        "ClinicalTrials": re.compile(r"(?i)clinicaltrials.*(found|retrieved|\d+\s+trials?)", re.IGNORECASE),
        "PubChem": re.compile(r"(?i)pubchem.*(found|success|cid)", re.IGNORECASE),
        "ChEMBL": re.compile(r"(?i)chembl.*(found|success|target)", re.IGNORECASE),
        "PrimeKG": re.compile(r"(?i)primekg.*(found|loaded|nodes?|edges?)", re.IGNORECASE),
        "OpenFDA": re.compile(r"(?i)(openfda|faers).*(found|retrieved|records?|events?)", re.IGNORECASE),
        "PubMed": re.compile(r"(?i)pubmed.*(found|retrieved|articles?|\d+\s+results?)", re.IGNORECASE),
        "OnSIDES": re.compile(r"(?i)onsides.*(found|loaded|matched|rows?)", re.IGNORECASE),
    }

    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return result

    for source, pat in source_patterns.items():
        matches = pat.findall(text)
        result["sources"][source] = len(matches) > 0

    # Also check for explicit "not found" / failures
    fail_patterns = {
        "PubChem": re.compile(r"(?i)pubchem.*found.*false|pubchem.*not found", re.IGNORECASE),
        "OpenFDA": re.compile(r"(?i)(openfda|faers).*(timeout|error|failed|no data)", re.IGNORECASE),
        "ChEMBL": re.compile(r"(?i)chembl.*(timeout|error|failed|not found)", re.IGNORECASE),
    }
    for source, pat in fail_patterns.items():
        if pat.search(text):
            result["sources"][f"{source}_failure"] = True

    return result


def print_summary(agent_results: list, rule_results: list, run_log_results: list):
    """Print formatted summary comparing original 10 vs new 10."""
    # Group by original/new
    groups = {"original": [], "new": []}
    for r in agent_results:
        stem = r["file"].replace("_agent_log.json", "")
        groups[classify_drug(stem)].append(r)

    rule_groups = {"original": [], "new": []}
    for r in rule_results:
        stem = r["file"].replace("_rules.json", "")
        rule_groups[classify_drug(stem)].append(r)

    print("=" * 80)
    print("  PIPELINE LOG ANALYSIS — ALL DRUGS")
    print("=" * 80)

    # ── 1. Stage 1 Sub-Call Success Rates ──
    print("\n┌─ 1. STAGE 1 SUB-CALL SUCCESS RATES ─────────────────────────────────────┐")
    subcalls = ["ae_freq", "severity", "onset", "triggers", "demographics"]
    for group_name in ["original", "new", "all"]:
        if group_name == "all":
            items = agent_results
        else:
            items = groups[group_name]
        if not items:
            continue

        print(f"\n  {group_name.upper()} ({len(items)} drugs):")
        for sc in subcalls:
            successes = sum(1 for r in items if r["stage1_calls"].get(sc, {}).get("success", False))
            total = len(items)
            pct = (successes / total * 100) if total else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    {sc:15s}  {bar}  {successes}/{total} ({pct:.0f}%)")

    print("└──────────────────────────────────────────────────────────────────────────┘")

    # ── 2. Stage 2 Grounding ──
    print("\n┌─ 2. STAGE 2 GROUNDING SUCCESS ───────────────────────────────────────────┐")
    for group_name in ["original", "new", "all"]:
        if group_name == "all":
            items = agent_results
        else:
            items = groups[group_name]
        if not items:
            continue

        s2_ok = sum(1 for r in items if r["stage2_success"])
        s2_fail = sum(1 for r in items if r["stage2_success"] is False)
        s2_none = sum(1 for r in items if r["stage2_success"] is None)
        total = len(items)
        print(f"  {group_name.upper():10s}  success={s2_ok}/{total}  parse_fail={s2_fail}/{total}  missing={s2_none}/{total}")
        if s2_fail > 0:
            errors = [r["stage2_error"][:60] for r in items if r.get("stage2_error")]
            err_counts = Counter(errors)
            for err, cnt in err_counts.most_common(3):
                print(f"    error ({cnt}x): {err}")

    print("└──────────────────────────────────────────────────────────────────────────┘")

    # ── 3. Auto-Correction Trigger Rates ──
    print("\n┌─ 3. AUTO-CORRECTION TRIGGER RATES ───────────────────────────────────────┐")
    correction_types = [
        "severity_correction", "severity_cap", "onset_correction",
        "trigger_correction", "sex_normalization", "dailymed_correction",
        "onsides_check", "combo_source_fix", "missing_aes", "other",
    ]
    for group_name in ["original", "new", "all"]:
        if group_name == "all":
            items = agent_results
        else:
            items = groups[group_name]
        if not items:
            continue

        print(f"\n  {group_name.upper()} ({len(items)} drugs):")
        for ct in correction_types:
            count = sum(r["warning_counts"].get(ct, 0) for r in items)
            drugs_affected = sum(1 for r in items if r["warning_counts"].get(ct, 0) > 0)
            if count > 0:
                print(f"    {ct:25s}  {count:3d} corrections across {drugs_affected}/{len(items)} drugs")

    print("└──────────────────────────────────────────────────────────────────────────┘")

    # ── 4. Evidence Source Coverage ──
    if run_log_results:
        print("\n┌─ 4. EVIDENCE SOURCE COVERAGE ────────────────────────────────────────────┐")
        sources = ["DailyMed", "DrugBank", "ClinicalTrials", "PubChem", "ChEMBL",
                    "PrimeKG", "OpenFDA", "PubMed", "OnSIDES"]
        for src in sources:
            found = sum(1 for r in run_log_results if r["sources"].get(src, False))
            total = len(run_log_results)
            fail_key = f"{src}_failure"
            failures = sum(1 for r in run_log_results if r["sources"].get(fail_key, False))
            status = f"  {found}/{total} found"
            if failures:
                status += f", {failures} with errors"
            print(f"    {src:15s}  {status}")
        print("└──────────────────────────────────────────────────────────────────────────┘")

    # ── 5. AE Naming Quality ──
    print("\n┌─ 5. AE NAMING QUALITY ───────────────────────────────────────────────────┐")
    for group_name in ["original", "new", "all"]:
        if group_name == "all":
            items = rule_results
        else:
            items = rule_groups[group_name]
        if not items:
            continue

        total_aes = sum(r["num_aes"] for r in items)
        total_lab = sum(len(r["lab_code_aes"]) for r in items)
        drugs_with_lab = sum(1 for r in items if r["lab_code_aes"])
        print(f"\n  {group_name.upper()} ({len(items)} drugs, {total_aes} total AEs):")
        print(f"    Lab-code style names: {total_lab}/{total_aes} ({total_lab/total_aes*100:.1f}%)" if total_aes else "    No AEs")
        print(f"    Drugs with lab-codes: {drugs_with_lab}/{len(items)}")
        if total_lab > 0:
            all_lab = []
            for r in items:
                all_lab.extend(r["lab_code_aes"])
            lab_counts = Counter(all_lab)
            print("    Most common lab-code AEs:")
            for name, cnt in lab_counts.most_common(10):
                print(f"      {name} ({cnt}x)")

    print("└──────────────────────────────────────────────────────────────────────────┘")

    # ── 6. Comorbidity Completeness ──
    print("\n┌─ 6. COMORBIDITY COMPLETENESS ────────────────────────────────────────────┐")
    for group_name in ["original", "new", "all"]:
        if group_name == "all":
            items = rule_results
        else:
            items = rule_groups[group_name]
        if not items:
            continue

        total_issues = sum(len(r["comorbidity_issues"]) for r in items)
        drugs_with_issues = sum(1 for r in items if r["comorbidity_issues"])
        print(f"\n  {group_name.upper()} ({len(items)} drugs):")
        print(f"    impacts_dosing=True with empty modifiers: {total_issues} across {drugs_with_issues}/{len(items)} drugs")
        if total_issues > 0:
            all_issues = []
            for r in items:
                all_issues.extend(r["comorbidity_issues"])
            conditions = Counter(issue.split(":")[0] for issue in all_issues)
            print("    Affected conditions:")
            for cond, cnt in conditions.most_common(10):
                print(f"      {cond} ({cnt}x)")

    print("└──────────────────────────────────────────────────────────────────────────┘")

    # ── Per-Drug Detail Table ──
    print("\n┌─ PER-DRUG DETAIL ────────────────────────────────────────────────────────┐")
    print(f"  {'Drug':<45s} {'AEs':>4s} {'S1ok':>4s} {'S2':>4s} {'Warns':>5s} {'Group':>8s}")
    print(f"  {'─'*45} {'─'*4} {'─'*4} {'─'*4} {'─'*5} {'─'*8}")

    # Pair agent logs with rule sets
    agent_by_stem = {}
    for r in agent_results:
        stem = r["file"].replace("_agent_log.json", "")
        agent_by_stem[stem] = r

    for r in sorted(rule_results, key=lambda x: x["file"]):
        stem = r["file"].replace("_rules.json", "")
        agent = agent_by_stem.get(stem, {})
        drug_label = " + ".join(r["drugs"]) if r["drugs"] else stem
        if len(drug_label) > 44:
            drug_label = drug_label[:41] + "..."
        s1_ok = sum(1 for v in agent.get("stage1_calls", {}).values() if v.get("success"))
        s1_total = len(agent.get("stage1_calls", {}))
        s2 = "OK" if agent.get("stage2_success") else "FAIL" if agent.get("stage2_success") is False else "N/A"
        warns = agent.get("warnings_total", 0)
        group = classify_drug(stem)
        s1_str = f"{s1_ok}/{s1_total}"
        print(f"  {drug_label:<45s} {r['num_aes']:>4d} {s1_str:<5s} {s2:>4s} {warns:>5d} {group:>8s}")

    print("└──────────────────────────────────────────────────────────────────────────┘")


def main():
    rule_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("rule_sets_improved")
    run_log_dir = None

    # Parse --run-logs flag
    for i, arg in enumerate(sys.argv):
        if arg == "--run-logs" and i + 1 < len(sys.argv):
            run_log_dir = Path(sys.argv[i + 1])

    if not rule_dir.exists():
        print(f"Error: {rule_dir} does not exist")
        sys.exit(1)

    # Collect agent logs
    agent_logs = sorted(rule_dir.glob("*_agent_log.json"))
    rule_files = sorted(rule_dir.glob("*_rules.json"))

    if not agent_logs:
        print(f"No agent logs found in {rule_dir}")
        sys.exit(1)

    print(f"Found {len(agent_logs)} agent logs, {len(rule_files)} rule sets")

    agent_results = []
    for log_path in agent_logs:
        try:
            agent_results.append(analyze_agent_log(log_path))
        except Exception as e:
            print(f"  Error parsing {log_path.name}: {e}")

    rule_results = []
    for rules_path in rule_files:
        try:
            rule_results.append(analyze_rule_set(rules_path))
        except Exception as e:
            print(f"  Error parsing {rules_path.name}: {e}")

    run_log_results = []
    if run_log_dir and run_log_dir.exists():
        for log_path in sorted(run_log_dir.glob("*.log")):
            try:
                run_log_results.append(analyze_run_log(log_path))
            except Exception as e:
                print(f"  Error parsing {log_path.name}: {e}")

    print_summary(agent_results, rule_results, run_log_results)


if __name__ == "__main__":
    main()
