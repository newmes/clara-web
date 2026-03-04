"""Merge base.json + overlay (iv_combination/subcutaneous_monotherapy) → rule_set.json.

For each drug folder in data/new_drugs/:
  1. Load base.json (contains all core fields)
  2. Load overlay file (administration_schedule, optionally ae_profile)
  3. Deep merge: overlay fields replace/extend base fields
  4. Save as rule_set.json in the same folder

Usage:
    python -m src.tools.merge_rule_sets
    python -m src.tools.merge_rule_sets --drug 2_Etoposide_Cisplatin
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEW_DRUGS_DIR = PROJECT_ROOT / "data" / "new_drugs"


def deep_merge(base: dict, overlay: dict) -> dict:
    """Merge overlay into base. Lists are replaced, dicts are recursively merged."""
    result = base.copy()
    for key, val in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def merge_drug(drug_dir: Path) -> Path:
    base_path = drug_dir / "base.json"
    if not base_path.exists():
        raise FileNotFoundError(f"No base.json in {drug_dir}")

    with open(base_path) as f:
        base = json.load(f)

    overlay_files = [p for p in drug_dir.glob("*.json") if p.name != "base.json" and p.name != "rule_set.json"]
    if not overlay_files:
        raise FileNotFoundError(f"No overlay file found in {drug_dir}")

    merged = base
    for overlay_path in overlay_files:
        with open(overlay_path) as f:
            overlay = json.load(f)
        print(f"    + {overlay_path.name}: {list(overlay.keys())}")
        merged = deep_merge(merged, overlay)

    out_dir = PROJECT_ROOT / "data" / "rule_sets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rule_set_{drug_dir.name}.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug", help="Specific drug folder name (e.g. 2_Etoposide_Cisplatin)")
    args = parser.parse_args()

    if args.drug:
        dirs = [NEW_DRUGS_DIR / args.drug]
    else:
        dirs = sorted(d for d in NEW_DRUGS_DIR.iterdir() if d.is_dir())

    print(f"Merging rule sets for {len(dirs)} drugs...\n")

    for drug_dir in dirs:
        name = drug_dir.name
        print(f"  [{name}]")
        try:
            out = merge_drug(drug_dir)
            with open(out) as f:
                rs = json.load(f)
            n_aes = len(rs.get("ae_profile", []))
            print(f"    → {out.name} ({n_aes} AEs, {out.stat().st_size/1024:.1f}KB)")
        except Exception as e:
            print(f"    ERROR: {e}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
