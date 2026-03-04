#!/usr/bin/env python3
"""Compare all 7 pipeline outputs against their Ground Truth rule sets, field by field.

Uses Gemini 2.0 Flash for intelligent AE term matching (handles British/American
spelling, synonyms, partial name differences).
"""

import json
import os
import re
import sys
from pathlib import Path
from math import sqrt

GT_DIR = Path(__file__).resolve().parent.parent / "ground_truth"
OUT_DIR = Path(__file__).resolve().parent.parent / "output"

# Mapping: GT directory → output subdirectory name
GT_TO_OUTPUT = {
    "1_Darbepoetin_alfa": "darbepoetin_alfa_small_cell_lung_cancer",
    "2_Etoposide_Cisplatin": "etoposide+cisplatin_small_cell_lung_cancer",
    "3_CALGB9732_Paclitaxel_Cisplatin_Etoposide": "paclitaxel+cisplatin+etoposide_small_cell_lung_cancer",
    "4_Carboplatin_Etoposide": "etoposide+carboplatin_small_cell_lung_cancer",
    "6_Paclitaxel_Carboplatin_Bevacizumab": "paclitaxel+carboplatin+bevacizumab_non-small_cell_lung_cancer",
    "7_Paclitaxel_Carboplatin": "paclitaxel+carboplatin_non-small_cell_lung_cancer",
    "8_Gemcitabine_Cisplatin": "gemcitabine+cisplatin_squamous_non-small_cell_lung_cancer",
}

# --- LLM-based AE matching ---

_AE_MATCH_CACHE: dict[str, dict[str, str]] = {}  # cache per drug


def _match_ae_terms_llm(out_terms: list[str], gt_terms: list[str], drug_label: str) -> dict[str, str]:
    """Use Gemini 2.0 Flash to match output AE terms to GT AE terms.

    Returns mapping: gt_term -> out_term (for matched pairs only).
    """
    cache_key = f"{drug_label}:{','.join(sorted(out_terms)[:10])}:{','.join(sorted(gt_terms)[:10])}"
    if cache_key in _AE_MATCH_CACHE:
        return _AE_MATCH_CACHE[cache_key]

    try:
        from openai import OpenAI
    except ImportError:
        print("  [WARN] openai not installed, falling back to exact match", file=sys.stderr)
        return {}

    api_key = os.environ.get("RULE_ENGINE_LLM_API_KEY", "")
    if not api_key:
        print("  [WARN] RULE_ENGINE_LLM_API_KEY not set, falling back to exact match", file=sys.stderr)
        return {}

    client = OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    prompt = f"""You are a clinical terminology expert. Match adverse event (AE) terms between two lists.

List A (pipeline output):
{json.dumps(out_terms)}

List B (ground truth):
{json.dumps(gt_terms)}

For each term in List B, find the EQUIVALENT term in List A if one exists.
Two terms are equivalent if they refer to the same medical condition, even with:
- British vs American spelling (anaemia = anemia, diarrhoea = diarrhea, oedema = edema)
- Synonyms (decreased appetite = anorexia, pyrexia = fever)
- Partial name differences (blood alkaline phosphatase increased = alkaline phosphatase increased)
- Word order differences (appetite decreased = decreased appetite)
- Lab test vs clinical name (haemoglobin decreased = anemia, platelet count decreased = thrombocytopenia)

Return ONLY a JSON object mapping matched List B terms to their List A equivalents.
Only include pairs where you are confident they refer to the same condition.
Do NOT force-match unrelated terms.

Example output format:
{{"anaemia": "anemia", "diarrhoea": "diarrhea", "decreased appetite": "anorexia"}}"""

    try:
        resp = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=2000,
            temperature=0.0,
        )
        content = resp.choices[0].message.content
        mapping = json.loads(content)
        # Validate: all keys should be in gt_terms, all values in out_terms
        validated = {}
        out_set = set(out_terms)
        gt_set = set(gt_terms)
        for gt_t, out_t in mapping.items():
            if gt_t in gt_set and out_t in out_set:
                validated[gt_t] = out_t
        _AE_MATCH_CACHE[cache_key] = validated
        return validated
    except Exception as e:
        print(f"  [WARN] LLM AE matching failed: {e}", file=sys.stderr)
        return {}


def load_gt(gt_name: str) -> dict:
    """Load GT base.json (ignore type overlay — it can overwrite with empty values)."""
    base = GT_DIR / gt_name / "base.json"
    return json.loads(base.read_text())


def load_output(subdir_name: str) -> dict:
    """Load output base.json from subdirectory."""
    out = OUT_DIR / subdir_name / "base.json"
    return json.loads(out.read_text())


def safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def pct_match(a, b) -> float:
    """Symmetric percentage match: 1.0 = identical, 0.0 = maximally different."""
    if a is None or b is None:
        return 0.0
    fa, fb = safe_float(a), safe_float(b)
    if fa is None or fb is None:
        # Non-numeric: exact string match
        return 1.0 if str(a).strip().lower() == str(b).strip().lower() else 0.0
    if fa == 0 and fb == 0:
        return 1.0
    if fa == 0 or fb == 0:
        return 0.0
    ratio = min(fa, fb) / max(fa, fb)
    return round(ratio, 3)


def extract_doses(data: dict) -> dict:
    """Extract {drug_lower: dose_value} from admin schedule."""
    doses = {}
    for entry in data.get("administration_schedule", []):
        drug = (entry.get("drug_name") or "").lower().strip()
        dose = entry.get("dose_per_administration") or entry.get("dose_value")
        if drug and dose is not None:
            dose_str = str(dose).strip()
            # Try to extract a number from AUC-style doses like "AUC 5"
            # Handle comma-separated thousands (e.g., "1,000 mg/m^2")
            nums = re.findall(r"[\d,.]+", dose_str)
            if nums:
                try:
                    doses[drug] = float(nums[0].replace(",", ""))
                except ValueError:
                    doses[drug] = dose_str
            else:
                doses[drug] = dose_str
    return doses


def extract_age(data: dict) -> dict:
    """Extract age params from demographics."""
    demo = data.get("demographics", {})
    age = demo.get("age", {})
    params = age.get("params") or age.get("parameters") or age
    return {
        "min": safe_float(params.get("min")),
        "max": safe_float(params.get("max")),
        "mean": safe_float(params.get("mean")),
        "std": safe_float(params.get("std")),
    }


def extract_sex(data: dict) -> dict:
    """Extract sex ratio (Male probability 0-1)."""
    demo = data.get("demographics", {})
    sex = demo.get("sex", {})
    opts = sex.get("options", {})
    # GT format: options.Male = float directly
    male = opts.get("Male")
    if isinstance(male, dict):
        male = male.get("probability")
    return {"male": safe_float(male)}


def extract_ecog(data: dict) -> dict:
    """Extract ECOG PS distribution."""
    demo = data.get("demographics", {})
    ecog = demo.get("ecog_ps", {})
    opts = ecog.get("options") or ecog.get("categories") or ecog
    if isinstance(opts, list):
        return {str(e.get("score", e.get("ecog", i))): safe_float(e.get("probability", e.get("proportion"))) for i, e in enumerate(opts)}
    if isinstance(opts, dict):
        return {str(k): safe_float(v) if not isinstance(v, dict) else safe_float(v.get("probability")) for k, v in opts.items()}
    return {}


def extract_efficacy(data: dict) -> dict:
    eff = data.get("efficacy", {})
    orr = eff.get("overall_response_rate")
    if isinstance(orr, dict):
        orr = orr.get("value") or orr.get("rate")
    pfs = eff.get("progression_free_survival") or {}
    os_ = eff.get("overall_survival") or {}
    if isinstance(pfs, dict):
        pfs_params = pfs.get("params") or pfs.get("parameters") or pfs
        pfs_med = pfs_params.get("median")
    else:
        pfs_med = None
    if isinstance(os_, dict):
        os_params = os_.get("params") or os_.get("parameters") or os_
        os_med = os_params.get("median")
    else:
        os_med = None
    return {
        "orr": safe_float(orr),
        "pfs_median": safe_float(pfs_med),
        "os_median": safe_float(os_med),
    }


# Unified synonym map for normalizing AE terms in BOTH GT and output
# Maps variant terms to a single canonical form for comparison
_COMPARE_SYNONYMS: dict[str, str] = {
    # Direction normalization
    "decreased appetite": "appetite decreased",
    "appetite loss": "appetite decreased",
    "loss of appetite": "appetite decreased",
    "anorexia": "appetite decreased",
    # British → American
    "anaemia": "anemia",
    "diarrhoea": "diarrhea",
    "dyspnoea": "dyspnea",
    "oedema": "edema",
    "oedema peripheral": "peripheral edema",
    "haemoglobin decreased": "anemia",
    "haemoptysis": "hemoptysis",
    "hypokalaemia": "hypokalemia",
    "hyponatraemia": "hyponatremia",
    "hypomagnesaemia": "hypomagnesemia",
    "hypocalcaemia": "hypocalcemia",
    "hyperglycaemia": "hyperglycemia",
    "hyperkalaemia": "hyperkalemia",
    "leucopenia": "leukopenia",
    "paraesthesia": "paresthesia",
    "hypercreatininaemia": "blood creatinine increased",
    # Lab → clinical
    "neutrophil count decreased": "neutropenia",
    "white blood cell count decreased": "leukopenia",
    "white blood cell decreased": "leukopenia",
    "platelet count decreased": "thrombocytopenia",
    "blood alkaline phosphatase increased": "alkaline phosphatase increased",
    "blood creatinine increased": "creatinine increased",
    "blood lactate dehydrogenase increased": "lactate dehydrogenase increased",
    "decreased neutrophils": "neutropenia",
    "decreased hemoglobin": "anemia",
    "decreased platelets": "thrombocytopenia",
    "lower hemoglobin": "anemia",
    # Clinical synonyms
    "neuropathy peripheral": "peripheral neuropathy",
    "peripheral sensory neuropathy": "peripheral neuropathy",
    "neuropathy sensory": "peripheral neuropathy",
    "pyrexia": "fever",
    "mucosal inflammation": "stomatitis",
    "weight loss": "weight decreased",
}


def _normalize_compare_term(term: str) -> str:
    """Normalize an AE term for comparison (applies synonym map)."""
    t = term.lower().replace("_", " ").strip()
    return _COMPARE_SYNONYMS.get(t, t)


def extract_aes(data: dict) -> dict:
    """Extract {normalized_ae_term: incidence_all_grade} from ae_profile."""
    aes = {}
    for ae in data.get("ae_profile", []):
        raw_term = (ae.get("ae_term") or ae.get("event") or "").strip()
        term = _normalize_compare_term(raw_term)
        inc = safe_float(ae.get("incidence_all_grade") or ae.get("frequency_pct"))
        if term:
            # If duplicate after normalization, keep higher frequency
            if term in aes:
                if inc is not None and (aes[term] is None or inc > aes[term]):
                    aes[term] = inc
            else:
                aes[term] = inc
    return aes


def compare_drug(gt_name: str, out_name: str, drug_label: str = "") -> dict:
    gt = load_gt(gt_name)
    out = load_output(out_name)

    scores = {}
    details = {}

    # --- Doses ---
    gt_doses = extract_doses(gt)
    out_doses = extract_doses(out)
    if gt_doses:
        dose_matches = []
        dose_detail = []
        for drug, gt_val in gt_doses.items():
            # fuzzy match drug name
            out_val = None
            for od, ov in out_doses.items():
                if drug in od or od in drug:
                    out_val = ov
                    break
            # Handle GT "patient-specific" / "individualized" doses
            gt_str = str(gt_val).lower()
            if ("patient" in gt_str or "individualized" in gt_str) and out_val is not None:
                out_str = str(out_val).lower()
                if "auc" in out_str or isinstance(out_val, (int, float)):
                    m = 0.8
                else:
                    m = 0.5
            else:
                m = pct_match(gt_val, out_val)
            dose_matches.append(m)
            dose_detail.append(f"{drug}: out={out_val} gt={gt_val} ({m:.3f})")
        scores["Doses"] = sum(dose_matches) / len(dose_matches) if dose_matches else 0.0
        details["Doses"] = "; ".join(dose_detail)
    else:
        scores["Doses"] = None
        details["Doses"] = "no GT doses"

    # --- Age ---
    gt_age = extract_age(gt)
    out_age = extract_age(out)
    if gt_age["min"] is not None and out_age["min"] is not None:
        range_score = 1.0 - min(1.0, (abs(gt_age["min"] - out_age["min"]) + abs((gt_age["max"] or 0) - (out_age["max"] or 0))) / 60)
        scores["Age Range"] = max(0, round(range_score, 3))
    else:
        scores["Age Range"] = 0.0
    details["Age Range"] = f"out={out_age['min']}-{out_age['max']} gt={gt_age['min']}-{gt_age['max']}"

    # --- Sex ---
    gt_sex = extract_sex(gt)
    out_sex = extract_sex(out)
    scores["Sex Ratio"] = pct_match(gt_sex["male"], out_sex["male"])
    details["Sex Ratio"] = f"out={out_sex['male']} gt={gt_sex['male']}"

    # --- ECOG ---
    gt_ecog = extract_ecog(gt)
    out_ecog = extract_ecog(out)
    if gt_ecog and out_ecog:
        all_keys = set(gt_ecog.keys()) | set(out_ecog.keys())
        ecog_diffs = []
        for k in all_keys:
            g = gt_ecog.get(k, 0.0) or 0.0
            o = out_ecog.get(k, 0.0) or 0.0
            ecog_diffs.append(abs(float(g) - float(o)))
        ecog_score = max(0, 1.0 - sum(ecog_diffs) / 2)
        scores["ECOG"] = round(ecog_score, 3)
    else:
        scores["ECOG"] = 0.0 if gt_ecog else None
    details["ECOG"] = f"out={out_ecog} gt={gt_ecog}"

    # --- Efficacy ---
    gt_eff = extract_efficacy(gt)
    out_eff = extract_efficacy(out)
    eff_parts = []
    if gt_eff["orr"] is not None:
        eff_parts.append(("ORR", pct_match(gt_eff["orr"], out_eff["orr"])))
    if gt_eff["pfs_median"] is not None:
        eff_parts.append(("PFS", pct_match(gt_eff["pfs_median"], out_eff["pfs_median"])))
    if gt_eff["os_median"] is not None:
        eff_parts.append(("OS", pct_match(gt_eff["os_median"], out_eff["os_median"])))
    if eff_parts:
        scores["ORR"] = eff_parts[0][1] if eff_parts[0][0] == "ORR" else None
        pfs_os = [v for n, v in eff_parts if n in ("PFS", "OS")]
        scores["PFS/OS"] = round(sum(pfs_os) / len(pfs_os), 3) if pfs_os else None
    else:
        scores["ORR"] = None
        scores["PFS/OS"] = None
    details["ORR"] = f"out={out_eff['orr']} gt={gt_eff['orr']}"
    details["PFS/OS"] = f"PFS: out={out_eff['pfs_median']} gt={gt_eff['pfs_median']}, OS: out={out_eff['os_median']} gt={gt_eff['os_median']}"

    # --- AE matching (LLM-based) ---
    gt_aes = extract_aes(gt)
    out_aes = extract_aes(out)

    # Build unified mapping: gt_term -> out_term
    # Start with exact matches, then augment with LLM fuzzy matches
    exact_common = set(gt_aes.keys()) & set(out_aes.keys())
    gt_term_to_out: dict[str, str] = {t: t for t in exact_common}

    # LLM matching for unmatched terms
    unmatched_gt = [t for t in gt_aes if t not in gt_term_to_out]
    unmatched_out = [t for t in out_aes if t not in exact_common]
    if unmatched_gt and unmatched_out:
        llm_map = _match_ae_terms_llm(unmatched_out, unmatched_gt, drug_label)
        used_out = set()
        for gt_t, out_t in llm_map.items():
            if gt_t not in gt_term_to_out and out_t not in used_out:
                gt_term_to_out[gt_t] = out_t
                used_out.add(out_t)

    matched_count = len(gt_term_to_out)
    llm_matched = matched_count - len(exact_common)

    # --- AE Count (use matched + unmatched from both sides) ---
    if gt_aes:
        cnt_ratio = min(len(out_aes), len(gt_aes)) / max(len(out_aes), len(gt_aes)) if max(len(out_aes), len(gt_aes)) > 0 else 0
        scores["AE Count"] = round(cnt_ratio, 3)
    else:
        scores["AE Count"] = None
    details["AE Count"] = f"out={len(out_aes)} gt={len(gt_aes)}"

    # --- AE Frequency match (over all matched pairs) ---
    if gt_term_to_out:
        freq_matches = []
        for gt_t, out_t in gt_term_to_out.items():
            g = gt_aes[gt_t]
            o = out_aes[out_t]
            if g is not None and o is not None:
                freq_matches.append(pct_match(g, o))
        scores["AE Freq"] = round(sum(freq_matches) / len(freq_matches), 3) if freq_matches else 0.0
    else:
        scores["AE Freq"] = 0.0
    details["AE Freq"] = f"{matched_count} matched AEs (exact={len(exact_common)}, llm={llm_matched})"

    # --- Top AE overlap (using unified mapping) ---
    gt_top10 = list(sorted(gt_aes.keys(), key=lambda k: gt_aes[k] or 0, reverse=True))[:10]
    out_top10 = set(list(sorted(out_aes.keys(), key=lambda k: out_aes[k] or 0, reverse=True))[:10])
    overlap_count = 0
    overlap_names = []
    for gt_t in gt_top10:
        matched_out = gt_term_to_out.get(gt_t)
        if matched_out and matched_out in out_top10:
            overlap_count += 1
            if gt_t == matched_out:
                overlap_names.append(gt_t)
            else:
                overlap_names.append(f"{gt_t}={matched_out}")
    scores["Top AE"] = round(overlap_count / 10, 3) if gt_top10 else 0.0
    details["Top AE"] = f"overlap={sorted(overlap_names)}"

    return scores, details


def main():
    all_scores = {}
    dims = ["Doses", "ORR", "Age Range", "Sex Ratio", "ECOG", "AE Count", "AE Freq", "Top AE", "PFS/OS"]

    print("=" * 130)
    print("FIELD-BY-FIELD GROUND TRUTH COMPARISON — ALL 7 DRUGS (LLM AE matching)")
    print("=" * 130)

    for gt_name, out_name in GT_TO_OUTPUT.items():
        out_path = OUT_DIR / out_name / "base.json"
        if not out_path.exists():
            print(f"\n  SKIP {gt_name}: output {out_name}/base.json not found")
            continue

        gt = load_gt(gt_name)
        drug_label = gt.get("drug_name", gt_name)
        indication = gt.get("indication", "?")

        print(f"\n  Matching AEs for {drug_label}...", end="", flush=True)
        scores, details = compare_drug(gt_name, out_name, drug_label)
        all_scores[gt_name] = scores
        print(" done")

        avg = sum(v for v in scores.values() if v is not None) / sum(1 for v in scores.values() if v is not None)

        print(f"{'─' * 130}")
        print(f"  {drug_label} / {indication}  →  {out_name}")
        print(f"{'─' * 130}")
        print(f"{'Dimension':<18} {'Score':>8}   {'Details'}")
        print(f"{'─'*18} {'─'*8}   {'─'*80}")
        for dim in dims:
            s = scores.get(dim)
            d = details.get(dim, "")
            s_str = f"{s:.3f}" if s is not None else "  n/a"
            print(f"{dim:<18} {s_str:>8}   {d}")
        print(f"{'─'*18} {'─'*8}")
        print(f"{'AVERAGE':<18} {avg:>8.3f}")

    # Summary table
    print(f"\n{'=' * 130}")
    print("  SUMMARY — ALL DRUGS")
    print(f"{'=' * 130}")
    header = f"{'Drug':<50}"
    for dim in dims:
        header += f" {dim[:8]:>8}"
    header += f" {'AVG':>8}"
    print(header)
    print("─" * len(header))

    grand_scores = []
    for gt_name, out_name in GT_TO_OUTPUT.items():
        if gt_name not in all_scores:
            continue
        gt = load_gt(gt_name)
        drug_label = gt.get("drug_name", gt_name)
        scores = all_scores[gt_name]
        valid = [v for v in scores.values() if v is not None]
        avg = sum(valid) / len(valid) if valid else 0
        grand_scores.append(avg)

        row = f"{drug_label[:50]:<50}"
        for dim in dims:
            s = scores.get(dim)
            row += f" {s:>8.3f}" if s is not None else f" {'n/a':>8}"
        row += f" {avg:>8.1%}"
        print(row)

    print("─" * len(header))
    if grand_scores:
        overall = sum(grand_scores) / len(grand_scores)
        print(f"{'COMBINED AVERAGE':<50}" + " " * (8 * len(dims) + len(dims)) + f"{overall:>8.1%}")


if __name__ == "__main__":
    main()
