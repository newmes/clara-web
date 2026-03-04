#!/usr/bin/env python3
"""Batch-convert rule set JSONs from current format to target simulation schema.

Usage:
    python scripts/convert_to_target_schema.py input_dir/ output_dir/
"""

import json
import sys
from pathlib import Path

# Add project root to path so rule_engine imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rule_engine.converter import convert_ruleset


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*_rules.json"))
    if not files:
        print(f"No *_rules.json files found in {input_dir}")
        sys.exit(1)

    print(f"Converting {len(files)} rule sets from {input_dir} -> {output_dir}")

    success = 0
    warnings = []
    for f in files:
        try:
            data = json.loads(f.read_text())
            converted, schema_type = convert_ruleset(data)
            out_path = output_dir / f.name
            out_path.write_text(json.dumps(converted, indent=2, ensure_ascii=False))
            n_aes = len(converted.get("ae_profile", []))
            n_dose_rules = len(converted.get("dose_modification_rules", []))
            n_support = len(converted.get("supportive_care_rules", []))
            print(f"  OK  {f.name} [{schema_type}] -> {n_aes} AEs, {n_dose_rules} dose rules, {n_support} supportive care")
            success += 1
        except Exception as e:
            warnings.append((f.name, str(e)))
            print(f"  FAIL {f.name}: {e}")

    print(f"\nDone: {success}/{len(files)} converted successfully")
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for name, msg in warnings:
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
