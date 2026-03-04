#!/usr/bin/env python3
"""Re-validate all rule sets using the latest validator code.

Loads each *_rules.json, re-runs validate_rule_set() (without EvidenceBundle —
DailyMed/OnSIDES corrections are already baked in from initial generation),
saves corrected versions in-place, and updates agent logs with new warnings.

Usage:
  python scripts/revalidate_all.py [rule_sets_dir]
"""

import json
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rule_engine.schema import RuleSet  # noqa: E402
from rule_engine.validator import validate_rule_set  # noqa: E402


def revalidate(rule_dir: Path):
    rule_files = sorted(rule_dir.glob("*_rules.json"))
    if not rule_files:
        print(f"No rule files found in {rule_dir}")
        sys.exit(1)

    print(f"Found {len(rule_files)} rule sets to revalidate\n")

    for rules_path in rule_files:
        stem = rules_path.stem.replace("_rules", "")
        log_path = rules_path.parent / f"{stem}_agent_log.json"

        print(f"  {stem}:")

        with open(rules_path) as f:
            raw = json.load(f)

        rule_set = RuleSet(**raw)
        warnings = validate_rule_set(rule_set, bundle=None)

        # Save corrected rule set
        corrected = rule_set.model_dump()
        corrected["drugs"] = raw.get("drugs", corrected.get("drugs", []))
        corrected["indication"] = raw.get("indication", corrected.get("indication", ""))

        with open(rules_path, "w") as f:
            json.dump(corrected, f, indent=2)

        # Update agent log warnings
        if log_path.exists():
            with open(log_path) as f:
                agent_log = json.load(f)
            agent_log["warnings"] = warnings
            agent_log["rule_set"] = corrected
            with open(log_path, "w") as f:
                json.dump(agent_log, f, indent=2)

        if warnings:
            print(f"    {len(warnings)} warnings (corrected in-place)")
            for w in warnings[:3]:
                print(f"      - {w[:90]}")
            if len(warnings) > 3:
                print(f"      ... and {len(warnings) - 3} more")
        else:
            print("    OK (no warnings)")

    print(f"\nDone. Revalidated {len(rule_files)} rule sets.")


if __name__ == "__main__":
    rule_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("rule_sets_improved")
    revalidate(rule_dir)
