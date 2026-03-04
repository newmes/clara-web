#!/usr/bin/env python3
"""Build MedDRA lookup table from openFDA FAERS + optional AACT.

Strategy (doc_agent.md §6.6):
  - Primary: openFDA FAERS API → MedDRA PT terms with frequency counts
  - Secondary: AACT PostgreSQL → PT + SOC mappings (requires free account)
  - Merge: Deduplicate, normalize case, assign SOC, output JSON

Usage:
  # openFDA only (no account needed)
  python scripts/build_meddra_lookup.py

  # openFDA + AACT (requires AACT account)
  python scripts/build_meddra_lookup.py --aact-user YOUR_USER --aact-pass YOUR_PASS

  # Custom output path
  python scripts/build_meddra_lookup.py --output data/meddra_lookup.json

Output format (data/meddra_lookup.json):
  {
    "metadata": { "generated": "...", "sources": [...], "drug_count": 8, "term_count": 350 },
    "terms": {
      "nausea": { "pt": "Nausea", "soc": "Gastrointestinal disorders", "faers_count": 1234 },
      ...
    }
  }
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ADC drugs to query (brand name + generic variants)
# ---------------------------------------------------------------------------

ADC_DRUGS: list[dict[str, str | list[str]]] = [
    {
        "brand": "Enhertu",
        "generic": "trastuzumab deruxtecan",
        "aliases": ["T-DXd", "DS-8201", "fam-trastuzumab deruxtecan-nxki"],
    },
    {
        "brand": "Kadcyla",
        "generic": "ado-trastuzumab emtansine",
        "aliases": ["T-DM1"],
    },
    {
        "brand": "Padcev",
        "generic": "enfortumab vedotin",
        "aliases": ["enfortumab vedotin-ejfv"],
    },
    {
        "brand": "Trodelvy",
        "generic": "sacituzumab govitecan",
        "aliases": ["sacituzumab govitecan-hziy"],
    },
    {
        "brand": "Adcetris",
        "generic": "brentuximab vedotin",
        "aliases": [],
    },
    {
        "brand": "Polivy",
        "generic": "polatuzumab vedotin",
        "aliases": ["polatuzumab vedotin-piiq"],
    },
    {
        "brand": "Elahere",
        "generic": "mirvetuximab soravtansine",
        "aliases": ["mirvetuximab soravtansine-gynx"],
    },
    {
        "brand": "Tivdak",
        "generic": "tisotumab vedotin",
        "aliases": ["tisotumab vedotin-tftv"],
    },
]

# openFDA endpoint
OPENFDA_BASE = "https://api.fda.gov/drug/event.json"

# Known SOC mappings — used when AACT is unavailable.
# These are the 27 official MedDRA SOC classes.  We map common PT terms to SOC
# based on standard pharmacovigilance practice.  New terms from FAERS without a
# known SOC mapping will have soc=None (MedGemma fallback will fill it).
KNOWN_SOC_MAP: dict[str, str] = {
    # Respiratory
    "pneumonitis": "Respiratory, thoracic and mediastinal disorders",
    "interstitial lung disease": "Respiratory, thoracic and mediastinal disorders",
    "dyspnoea": "Respiratory, thoracic and mediastinal disorders",
    "cough": "Respiratory, thoracic and mediastinal disorders",
    "pulmonary fibrosis": "Respiratory, thoracic and mediastinal disorders",
    "pleural effusion": "Respiratory, thoracic and mediastinal disorders",
    "respiratory failure": "Respiratory, thoracic and mediastinal disorders",
    "hypoxia": "Respiratory, thoracic and mediastinal disorders",
    "pneumothorax": "Respiratory, thoracic and mediastinal disorders",
    "organising pneumonia": "Respiratory, thoracic and mediastinal disorders",
    "acute respiratory distress syndrome": "Respiratory, thoracic and mediastinal disorders",
    "productive cough": "Respiratory, thoracic and mediastinal disorders",
    "pulmonary embolism": "Respiratory, thoracic and mediastinal disorders",
    "bronchospasm": "Respiratory, thoracic and mediastinal disorders",

    # GI
    "nausea": "Gastrointestinal disorders",
    "vomiting": "Gastrointestinal disorders",
    "diarrhoea": "Gastrointestinal disorders",
    "constipation": "Gastrointestinal disorders",
    "stomatitis": "Gastrointestinal disorders",
    "abdominal pain": "Gastrointestinal disorders",
    "mucosal inflammation": "Gastrointestinal disorders",
    "colitis": "Gastrointestinal disorders",
    "dysphagia": "Gastrointestinal disorders",
    "gastrointestinal haemorrhage": "Gastrointestinal disorders",

    # Blood
    "neutropenia": "Blood and lymphatic system disorders",
    "febrile neutropenia": "Blood and lymphatic system disorders",
    "thrombocytopenia": "Blood and lymphatic system disorders",
    "anaemia": "Blood and lymphatic system disorders",
    "leukopenia": "Blood and lymphatic system disorders",
    "pancytopenia": "Blood and lymphatic system disorders",
    "lymphopenia": "Blood and lymphatic system disorders",
    "disseminated intravascular coagulation": "Blood and lymphatic system disorders",

    # General
    "fatigue": "General disorders and administration site conditions",
    "asthenia": "General disorders and administration site conditions",
    "pyrexia": "General disorders and administration site conditions",
    "oedema peripheral": "General disorders and administration site conditions",
    "malaise": "General disorders and administration site conditions",
    "death": "General disorders and administration site conditions",
    "multi-organ failure": "General disorders and administration site conditions",
    "infusion related reaction": "Injury, poisoning and procedural complications",

    # Skin
    "alopecia": "Skin and subcutaneous tissue disorders",
    "rash": "Skin and subcutaneous tissue disorders",
    "palmar-plantar erythrodysaesthesia syndrome": "Skin and subcutaneous tissue disorders",
    "pruritus": "Skin and subcutaneous tissue disorders",
    "dry skin": "Skin and subcutaneous tissue disorders",
    "dermatitis acneiform": "Skin and subcutaneous tissue disorders",
    "skin reaction": "Skin and subcutaneous tissue disorders",
    "erythema": "Skin and subcutaneous tissue disorders",
    "stevens-johnson syndrome": "Skin and subcutaneous tissue disorders",

    # Metabolism
    "decreased appetite": "Metabolism and nutrition disorders",
    "hypokalaemia": "Metabolism and nutrition disorders",
    "hyponatraemia": "Metabolism and nutrition disorders",
    "hyperglycaemia": "Metabolism and nutrition disorders",
    "dehydration": "Metabolism and nutrition disorders",
    "hypophosphataemia": "Metabolism and nutrition disorders",
    "hypomagnesaemia": "Metabolism and nutrition disorders",

    # Nervous system
    "peripheral sensory neuropathy": "Nervous system disorders",
    "headache": "Nervous system disorders",
    "dizziness": "Nervous system disorders",
    "neuropathy peripheral": "Nervous system disorders",
    "dysgeusia": "Nervous system disorders",
    "paraesthesia": "Nervous system disorders",
    "convulsion": "Nervous system disorders",

    # Eye
    "keratitis": "Eye disorders",
    "blurred vision": "Eye disorders",
    "dry eye": "Eye disorders",
    "conjunctivitis": "Eye disorders",

    # Hepatobiliary
    "hepatotoxicity": "Hepatobiliary disorders",
    "hepatic failure": "Hepatobiliary disorders",
    "hepatitis": "Hepatobiliary disorders",
    "jaundice": "Hepatobiliary disorders",

    # Investigations
    "alanine aminotransferase increased": "Investigations",
    "aspartate aminotransferase increased": "Investigations",
    "blood bilirubin increased": "Investigations",
    "weight decreased": "Investigations",
    "electrocardiogram qt prolonged": "Investigations",
    "ejection fraction decreased": "Investigations",
    "blood creatinine increased": "Investigations",
    "neutrophil count decreased": "Investigations",
    "platelet count decreased": "Investigations",
    "white blood cell count decreased": "Investigations",
    "blood alkaline phosphatase increased": "Investigations",
    "gamma-glutamyltransferase increased": "Investigations",
    "lipase increased": "Investigations",
    "amylase increased": "Investigations",
    "lymphocyte count decreased": "Investigations",
    "haemoglobin decreased": "Investigations",

    # Infections
    "sepsis": "Infections and infestations",
    "pneumonia": "Infections and infestations",
    "urinary tract infection": "Infections and infestations",
    "cellulitis": "Infections and infestations",
    "upper respiratory tract infection": "Infections and infestations",
    "covid-19": "Infections and infestations",

    # Vascular
    "hypertension": "Vascular disorders",
    "hypotension": "Vascular disorders",
    "deep vein thrombosis": "Vascular disorders",
    "haemorrhage": "Vascular disorders",

    # Cardiac
    "left ventricular dysfunction": "Cardiac disorders",
    "cardiac failure": "Cardiac disorders",
    "atrial fibrillation": "Cardiac disorders",
    "tachycardia": "Cardiac disorders",
    "cardiac arrest": "Cardiac disorders",

    # Renal
    "proteinuria": "Renal and urinary disorders",
    "acute kidney injury": "Renal and urinary disorders",
    "renal failure": "Renal and urinary disorders",

    # MSK
    "arthralgia": "Musculoskeletal and connective tissue disorders",
    "myalgia": "Musculoskeletal and connective tissue disorders",
    "back pain": "Musculoskeletal and connective tissue disorders",
    "pain in extremity": "Musculoskeletal and connective tissue disorders",
    "muscular weakness": "Musculoskeletal and connective tissue disorders",

    # Psychiatric
    "insomnia": "Psychiatric disorders",

    # Immune
    "anaphylactic reaction": "Immune system disorders",
    "hypersensitivity": "Immune system disorders",

    # Neoplasms
    "disease progression": "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
    "tumour lysis syndrome": "Metabolism and nutrition disorders",

    # Reproductive
    "infertility": "Reproductive system and breast disorders",

    # Injury
    "fall": "Injury, poisoning and procedural complications",
}


# ---------------------------------------------------------------------------
# openFDA FAERS extraction
# ---------------------------------------------------------------------------

def _openfda_count_ae(
    drug_name: str,
    field: str = "patient.drug.openfda.brand_name",
    limit: int = 200,
    api_key: Optional[str] = None,
) -> list[dict]:
    """Query openFDA for AE PT term counts for a specific drug."""
    params: dict[str, str | int] = {
        "search": f'{field}:"{drug_name}"',
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": limit,
    }
    if api_key:
        params["api_key"] = api_key

    try:
        resp = requests.get(OPENFDA_BASE, params=params, timeout=30)
        if resp.status_code == 404:
            logger.info("No FAERS data for %s (%s)", drug_name, field)
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except requests.RequestException as e:
        logger.warning("openFDA request failed for %s: %s", drug_name, e)
        return []


def extract_openfda(
    drugs: list[dict],
    api_key: Optional[str] = None,
) -> dict[str, dict]:
    """Extract AE terms from openFDA FAERS for all ADC drugs.

    Returns dict: { lowercase_pt: { "pt": str, "faers_count": int, "drugs": set } }
    """
    terms: dict[str, dict] = {}

    for drug_info in drugs:
        brand = drug_info["brand"]
        generic = drug_info["generic"]
        aliases = drug_info.get("aliases", [])

        # Try brand name first, then generic, then aliases
        queries = [
            (brand, "patient.drug.openfda.brand_name"),
            (generic, "patient.drug.medicinalproduct"),
        ]
        for alias in aliases:
            queries.append((alias, "patient.drug.medicinalproduct"))

        drug_terms_found = 0
        for query_name, field in queries:
            results = _openfda_count_ae(query_name, field=field, api_key=api_key)
            if not results:
                time.sleep(0.3)  # Rate limit courtesy
                continue

            for item in results:
                raw_term = item.get("term", "")
                count = item.get("count", 0)
                if not raw_term:
                    continue

                # Normalize: openFDA returns ALL CAPS for older data
                key = raw_term.strip().lower()
                # Title-case the PT for display
                pt_display = raw_term.strip().title()
                # Fix common MedDRA capitalizations
                pt_display = _fix_meddra_casing(pt_display)

                if key in terms:
                    terms[key]["faers_count"] = max(terms[key]["faers_count"], count)
                    terms[key]["drugs"].add(brand)
                else:
                    terms[key] = {
                        "pt": pt_display,
                        "faers_count": count,
                        "drugs": {brand},
                    }
                    drug_terms_found += 1

            logger.info(
                "openFDA: %s (%s) → %d AE terms",
                query_name, field, len(results),
            )
            time.sleep(0.3)  # Rate limit

            # If brand name worked, skip generic/aliases for this drug
            if results and field == "patient.drug.openfda.brand_name":
                break

        logger.info("  Drug %s: %d new terms added", brand, drug_terms_found)

    return terms


def _fix_meddra_casing(term: str) -> str:
    """Fix common MedDRA PT capitalizations from title-case conversion."""
    # MedDRA PTs are generally sentence-case (first word cap, rest lower)
    # except for acronyms and proper nouns
    if not term:
        return term

    # Convert to sentence case
    words = term.split()
    if not words:
        return term

    result = words[0]  # Keep first word as-is from title case
    for w in words[1:]:
        # Keep uppercase if it looks like an acronym (2-4 caps)
        if len(w) <= 4 and w.isupper():
            result += " " + w
        # Keep certain medical terms that start with lowercase
        elif w.lower() in ("qt", "hiv", "ecg", "mri", "ct"):
            result += " " + w.upper()
        else:
            result += " " + w.lower()

    # Special fixes
    replacements = {
        "Interstitial lung disease": "Interstitial lung disease",
        "Palmar-plantar erythrodysaesthesia syndrome": "Palmar-plantar erythrodysaesthesia syndrome",
        "Ejection fraction decreased": "Ejection fraction decreased",
        "Disseminated intravascular coagulation": "Disseminated intravascular coagulation",
        "Stevens-johnson syndrome": "Stevens-Johnson syndrome",
        "Covid-19": "COVID-19",
    }
    return replacements.get(result, result)


# ---------------------------------------------------------------------------
# AACT extraction (optional)
# ---------------------------------------------------------------------------

def extract_aact(
    drugs: list[dict],
    user: str,
    password: str,
    host: str = "aact-db.ctti-clinicaltrials.org",
    port: int = 5432,
    database: str = "aact",
) -> dict[str, str]:
    """Extract PT → SOC mappings from AACT for ADC drugs.

    Returns dict: { lowercase_pt: soc_name }
    """
    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 not available — skipping AACT extraction")
        return {}

    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    logger.info("Connecting to AACT: %s", host)

    try:
        conn = psycopg2.connect(conn_str, connect_timeout=15)
    except Exception as e:
        logger.warning("AACT connection failed: %s", e)
        return {}

    pt_to_soc: dict[str, str] = {}

    try:
        cur = conn.cursor()

        # Build drug name conditions
        drug_names = []
        for d in drugs:
            drug_names.append(d["brand"])
            drug_names.append(d["generic"])
            drug_names.extend(d.get("aliases", []))

        # Query: PT + SOC from reported_events joined with interventions
        name_conditions = " OR ".join(
            [f"i.name ILIKE %s" for _ in drug_names]
        )
        name_params = [f"%{n}%" for n in drug_names]

        query = f"""
            SELECT DISTINCT
                LOWER(TRIM(re.adverse_event_term)) AS pt_lower,
                re.organ_system AS soc,
                COUNT(DISTINCT re.nct_id) AS trial_count
            FROM ctgov.reported_events re
            JOIN ctgov.studies s ON re.nct_id = s.nct_id
            JOIN ctgov.interventions i ON s.nct_id = i.nct_id
            WHERE ({name_conditions})
              AND (re.vocab ILIKE '%%meddra%%' OR re.default_vocab ILIKE '%%meddra%%')
              AND re.adverse_event_term IS NOT NULL
              AND re.organ_system IS NOT NULL
            GROUP BY pt_lower, soc
            ORDER BY trial_count DESC
        """

        logger.info("AACT query: %d drug name variants", len(drug_names))
        cur.execute(query, name_params)
        rows = cur.fetchall()

        for pt_lower, soc, trial_count in rows:
            if pt_lower and soc:
                # Keep highest trial_count SOC for duplicate PTs
                if pt_lower not in pt_to_soc:
                    pt_to_soc[pt_lower] = soc

        logger.info("AACT: %d unique PT→SOC mappings extracted", len(pt_to_soc))

    except Exception as e:
        logger.warning("AACT query failed: %s", e)
    finally:
        conn.close()

    return pt_to_soc


# ---------------------------------------------------------------------------
# Merge + output
# ---------------------------------------------------------------------------

def merge_and_export(
    openfda_terms: dict[str, dict],
    aact_soc: dict[str, str],
    output_path: Path,
    min_count: int = 2,
) -> dict:
    """Merge openFDA terms with AACT SOC and export JSON lookup table."""
    lookup: dict[str, dict] = {}
    skipped = 0

    for key, info in sorted(openfda_terms.items()):
        # Skip very rare terms (likely noise)
        if info["faers_count"] < min_count:
            skipped += 1
            continue

        pt = info["pt"]

        # SOC priority: AACT > KNOWN_SOC_MAP > None
        soc = aact_soc.get(key) or KNOWN_SOC_MAP.get(key)

        entry: dict = {
            "pt": pt,
            "soc": soc,
            "llt": pt,  # Default LLT = PT
            "faers_count": info["faers_count"],
            "drug_count": len(info["drugs"]),
        }
        lookup[key] = entry

    # Also include any KNOWN_SOC_MAP terms not yet in lookup
    # (ensures our manually verified terms are always present)
    for key, soc in KNOWN_SOC_MAP.items():
        if key not in lookup:
            pt_display = key.title()
            pt_display = _fix_meddra_casing(pt_display)
            lookup[key] = {
                "pt": pt_display,
                "soc": soc,
                "llt": pt_display,
                "faers_count": 0,
                "drug_count": 0,
            }

    result = {
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "sources": ["openFDA FAERS"],
            "aact_used": bool(aact_soc),
            "drugs_queried": [d["brand"] for d in ADC_DRUGS],
            "total_terms": len(lookup),
            "skipped_rare": skipped,
            "min_count_threshold": min_count,
        },
        "terms": dict(sorted(lookup.items())),
    }

    if aact_soc:
        result["metadata"]["sources"].append("AACT (ClinicalTrials.gov)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(
        "Exported %d terms to %s (skipped %d rare terms)",
        len(lookup), output_path, skipped,
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build MedDRA lookup table from openFDA FAERS + AACT"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/meddra_lookup.json",
        help="Output JSON path (default: data/meddra_lookup.json)",
    )
    parser.add_argument(
        "--openfda-key",
        default=os.environ.get("OPENFDA_API_KEY"),
        help="openFDA API key (optional, increases rate limit)",
    )
    parser.add_argument(
        "--aact-user",
        default=os.environ.get("AACT_USER"),
        help="AACT PostgreSQL username (optional)",
    )
    parser.add_argument(
        "--aact-pass",
        default=os.environ.get("AACT_PASS"),
        help="AACT PostgreSQL password (optional)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum FAERS report count to include a term (default: 2)",
    )
    args = parser.parse_args()

    # Resolve output path relative to project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    # Step 1: openFDA FAERS
    logger.info("=== Step 1: Extracting AE terms from openFDA FAERS ===")
    openfda_terms = extract_openfda(ADC_DRUGS, api_key=args.openfda_key)
    logger.info("openFDA total: %d unique PT terms", len(openfda_terms))

    # Step 2: AACT (optional)
    aact_soc: dict[str, str] = {}
    if args.aact_user and args.aact_pass:
        logger.info("=== Step 2: Extracting SOC mappings from AACT ===")
        aact_soc = extract_aact(ADC_DRUGS, args.aact_user, args.aact_pass)
    else:
        logger.info("=== Step 2: AACT skipped (no credentials) — using built-in SOC map ===")

    # Step 3: Merge and export
    logger.info("=== Step 3: Merging and exporting ===")
    result = merge_and_export(
        openfda_terms, aact_soc, output_path, min_count=args.min_count,
    )

    # Summary
    meta = result["metadata"]
    terms_with_soc = sum(1 for t in result["terms"].values() if t.get("soc"))
    print(f"\n{'='*60}")
    print(f"MedDRA Lookup Table Built Successfully")
    print(f"{'='*60}")
    print(f"  Sources:        {', '.join(meta['sources'])}")
    print(f"  Drugs queried:  {meta['drugs_queried']}")
    print(f"  Total terms:    {meta['total_terms']}")
    print(f"  With SOC:       {terms_with_soc}")
    print(f"  Without SOC:    {meta['total_terms'] - terms_with_soc}")
    print(f"  Skipped (rare): {meta['skipped_rare']}")
    print(f"  Output:         {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
