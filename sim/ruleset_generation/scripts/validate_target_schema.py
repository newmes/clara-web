#!/usr/bin/env python3
"""Validate converted rule set JSONs against base.json AND type-specific schemas.

Usage:
    python scripts/validate_target_schema.py output/
"""

import json
import sys
from pathlib import Path

try:
    import jsonschema
    from jsonschema import RefResolver
except ImportError:
    print("Error: jsonschema not installed. Run: pip install jsonschema")
    sys.exit(1)


SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schema"


def load_schemas() -> tuple[dict, dict[str, dict]]:
    """Load base schema + all type-specific schemas."""
    base_path = SCHEMA_DIR / "base.json"
    if not base_path.exists():
        print(f"Error: base schema not found at {base_path}")
        sys.exit(1)
    base = json.loads(base_path.read_text())
    type_schemas: dict[str, dict] = {}
    for f in SCHEMA_DIR.glob("*.json"):
        if f.name != "base.json":
            type_schemas[f.stem] = json.loads(f.read_text())
    return base, type_schemas


def detect_schema_type_from_output(data: dict) -> str:
    """Detect schema type from converted output.

    Uses _schema_type metadata if present, otherwise infers from
    administration_schedule routes.
    """
    # Prefer embedded metadata
    if "_schema_type" in data:
        return data["_schema_type"]

    # Infer from administration_schedule
    schedule = data.get("administration_schedule", [])
    routes = {item.get("route") for item in schedule}
    is_combo = len(schedule) >= 2
    has_oral = "ORAL" in routes
    has_iv = "INTRAVENOUS" in routes
    has_sc = "SUBCUTANEOUS" in routes

    if is_combo:
        if has_oral and has_iv:
            return "oral_iv_combination"
        elif has_iv:
            return "iv_combination"
        else:
            return "oral_iv_combination"
    else:
        if has_sc:
            return "subcutaneous_monotherapy"
        elif has_oral:
            return "oral_monotherapy"
        else:
            return "iv_monotherapy"


def validate_file(filepath: Path, base_schema: dict, type_schemas: dict[str, dict],
                   base_only: bool = False) -> tuple[list[str], str]:
    """Validate a single JSON file against base + type-specific schema.

    When base_only=True, skip type-specific validation (for split-format base.json).
    Returns (list_of_errors, detected_schema_type).
    """
    try:
        data = json.loads(filepath.read_text())
    except json.JSONDecodeError as e:
        return [f"JSON parse error: {e}"], "unknown"

    schema_type = detect_schema_type_from_output(data)
    errors: list[str] = []

    # Build a resolver that maps $ref "base.json" to our loaded base schema
    schema_store = {base_schema.get("$id", "base.json"): base_schema}
    resolver = RefResolver.from_schema(base_schema, store=schema_store)

    # 1. Validate against base schema
    base_validator = jsonschema.Draft7Validator(base_schema, resolver=resolver)
    for error in sorted(base_validator.iter_errors(data), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in error.absolute_path)
        errors.append(f"[base] {path}: {error.message}")

    # 2. Validate against type-specific schema (skip for split base.json)
    if not base_only:
        if schema_type in type_schemas:
            type_schema = type_schemas[schema_type]
            type_validator = jsonschema.Draft7Validator(type_schema, resolver=resolver)
            for error in sorted(type_validator.iter_errors(data), key=lambda e: list(e.path)):
                path = ".".join(str(p) for p in error.absolute_path)
                errors.append(f"[{schema_type}] {path}: {error.message}")
        else:
            errors.append(f"No type-specific schema found for '{schema_type}'")

    return errors, schema_type


def spot_check(filepath: Path, is_split: bool = False) -> list[str]:
    """Run additional semantic checks beyond JSON schema validation.

    When is_split=True, skip route-specific field checks (they live in the overlay).
    """
    data = json.loads(filepath.read_text())
    issues = []

    # drug_name should be string
    if not isinstance(data.get("drug_name"), str):
        issues.append("drug_name is not a string")

    # All probabilities 0-1
    for ae in data.get("ae_profile", []):
        inc = ae.get("incidence_all_grade", 0)
        if inc > 1.0:
            issues.append(f"ae_profile '{ae.get('ae_term')}': incidence_all_grade={inc} > 1.0")
        gd = ae.get("grade_distribution", {})
        for g, v in gd.items():
            if v > 1.0:
                issues.append(f"ae_profile '{ae.get('ae_term')}': grade_distribution[{g}]={v} > 1.0")

    # Comorbidity probabilities
    for c in data.get("comorbidities", []):
        bp = c.get("base_probability", 0)
        if bp > 1.0:
            issues.append(f"comorbidity '{c.get('condition')}': base_probability={bp} > 1.0")

    # Efficacy
    orr = data.get("efficacy", {}).get("overall_response_rate", 0)
    if orr > 1.0:
        issues.append(f"efficacy: overall_response_rate={orr} > 1.0")

    # Required sections present
    for section in ["lab_reference_ranges", "mortality_model", "ecog_model", "disposition_model"]:
        if not data.get(section):
            issues.append(f"Missing or empty: {section}")

    # Type-specific field checks — skip for split base.json (fields are in overlay)
    if not is_split:
        schema_type = detect_schema_type_from_output(data)
        schedule = data.get("administration_schedule", [])

        if "iv" in schema_type:
            for i, item in enumerate(schedule):
                if item.get("route") == "INTRAVENOUS" and not item.get("infusion_duration_minutes"):
                    issues.append(f"admin_schedule[{i}] ({item.get('drug_name')}): missing infusion_duration_minutes for IV drug")

        if schema_type == "oral_monotherapy":
            for i, item in enumerate(schedule):
                if not item.get("daily_dosing_schedule"):
                    issues.append(f"admin_schedule[{i}] ({item.get('drug_name')}): missing daily_dosing_schedule for oral drug")
                if "infusion_duration_minutes" in item:
                    issues.append(f"admin_schedule[{i}] ({item.get('drug_name')}): oral drug should not have infusion_duration_minutes")

        if "combination" in schema_type and len(schedule) < 2:
            issues.append(f"combination schema type but only {len(schedule)} item(s) in administration_schedule")

    # Combination check works even for split (admin_schedule is still in base)
    if is_split:
        schema_type = detect_schema_type_from_output(data)
        schedule = data.get("administration_schedule", [])
        if "combination" in schema_type and len(schedule) < 2:
            issues.append(f"combination schema type but only {len(schedule)} item(s) in administration_schedule")

    return issues


def validate_overlay(overlay_path: Path, type_schemas: dict[str, dict]) -> list[str]:
    """Validate a type overlay JSON file exists and is valid JSON."""
    errors = []
    schema_type = overlay_path.stem  # e.g. "iv_combination"
    try:
        data = json.loads(overlay_path.read_text())
    except json.JSONDecodeError as e:
        return [f"JSON parse error in overlay: {e}"]

    if schema_type not in type_schemas:
        errors.append(f"No type-specific schema found for overlay '{schema_type}'")
    return errors


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <target_dir>")
        sys.exit(1)

    target_dir = Path(sys.argv[1])
    if not target_dir.is_dir():
        print(f"Error: directory not found: {target_dir}")
        sys.exit(1)

    base_schema, type_schemas = load_schemas()

    # Find base.json files in subdirectories (new split format)
    files = sorted(target_dir.glob("*/base.json"))

    if not files:
        print(f"No */base.json files found in {target_dir}")
        sys.exit(1)

    print(f"Validating {len(files)} rule sets against base schema (split format)")
    print(f"Available type schemas: {', '.join(sorted(type_schemas.keys()))}\n")

    pass_count = 0
    fail_count = 0

    for f in files:
        subdir = f.parent
        label = subdir.name

        # Validate base.json against base schema only (not type-specific,
        # since route-specific fields now live in the overlay file)
        schema_errors, schema_type = validate_file(f, base_schema, type_schemas, base_only=True)
        spot_issues = spot_check(f, is_split=True)

        # Validate overlay file exists and is valid JSON
        overlay_path = subdir / f"{schema_type}.json"
        if overlay_path.exists():
            overlay_errors = validate_overlay(overlay_path, type_schemas)
            schema_errors.extend(overlay_errors)
        else:
            schema_errors.append(f"Missing overlay file: {schema_type}.json")

        if not schema_errors and not spot_issues:
            print(f"  PASS  {label}/  [{schema_type}]")
            pass_count += 1
        else:
            print(f"  FAIL  {label}/  [{schema_type}]")
            fail_count += 1
            for e in schema_errors[:10]:
                print(f"    [schema] {e}")
            if len(schema_errors) > 10:
                print(f"    ... and {len(schema_errors) - 10} more schema errors")
            for i in spot_issues:
                print(f"    [check]  {i}")

    print(f"\nResults: {pass_count} PASS, {fail_count} FAIL out of {len(files)} rule sets")
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
