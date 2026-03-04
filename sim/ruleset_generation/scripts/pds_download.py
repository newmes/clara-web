#!/usr/bin/env python3
"""Download selected trial datasets from Project Data Sphere CAS to local CSV.

Reads the catalog.json (from pds_discover.py) or accepts caslib names as CLI
arguments.  For each selected caslib, loads each recognized domain table into
CAS, fetches the data as a pandas DataFrame, and writes it as CSV to
data/pds/{caslib_name}/.

Also creates/updates data/pds/trial_index.csv mapping trial IDs to drug names,
indications, and patient counts.

Usage:
    # Download all relevant caslibs from catalog
    python scripts/pds_download.py

    # Download specific caslibs by name
    python scripts/pds_download.py Lung_na_2010_042 Lung_na_2012_087

    # Download from a custom catalog
    python scripts/pds_download.py --catalog data/pds/catalog.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rule_engine.config import RuleEngineConfig

log = logging.getLogger(__name__)

# Common SDTM/ADaM domain table names to attempt downloading
_DOMAIN_PRIORITIES = [
    # Core domains (SDTM)
    "dm", "ae", "ex", "rs", "tu", "tr", "ds", "lb", "vs", "cm", "mh",
    # ADaM analysis datasets
    "adsl", "adae", "adrs", "adtte", "adlb", "adeff",
]


def _find_table_file(tables: list[str], domain: str) -> str | None:
    """Find a table file matching a domain name (case-insensitive, any extension)."""
    domain_lower = domain.lower()
    for t in tables:
        base = t.lower().split(".")[0]
        if base == domain_lower:
            return t
    return None


def download_caslib(
    conn,
    caslib_name: str,
    tables: list[str],
    output_dir: Path,
) -> dict:
    """Download all recognized domain tables from a caslib to CSV files.

    Returns metadata dict for trial_index.
    """
    caslib_dir = output_dir / caslib_name
    caslib_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[str] = []
    n_patients = 0

    for domain in _DOMAIN_PRIORITIES:
        table_file = _find_table_file(tables, domain)
        if table_file is None:
            continue

        try:
            # Load table into CAS
            tmp_name = f"_tmp_{domain}"
            conn.loadtable(
                caslib=caslib_name,
                path=table_file,
                casout={"name": tmp_name, "replace": True},
            )

            # Get row count
            n_rows = int(conn.numrows(table=tmp_name).get("numrows", 0))
            if n_rows == 0:
                conn.droptable(name=tmp_name, quiet=True)
                continue

            # Fetch all rows (in chunks for large tables)
            chunk_size = 10000
            all_chunks = []
            for start in range(1, n_rows + 1, chunk_size):
                result = conn.fetch(
                    table=tmp_name,
                    from_=start,
                    to=min(start + chunk_size - 1, n_rows),
                )
                chunk_df = result.get("Fetch", None)
                if chunk_df is not None and len(chunk_df) > 0:
                    all_chunks.append(chunk_df)

            conn.droptable(name=tmp_name, quiet=True)

            if not all_chunks:
                continue

            import pandas as pd
            df = pd.concat(all_chunks, ignore_index=True)

            # Write to CSV
            csv_path = caslib_dir / f"{domain}.csv"
            df.to_csv(csv_path, index=False)
            downloaded.append(domain)

            if domain == "dm":
                n_patients = len(df)

            log.info(
                "  %s/%s: %d rows → %s",
                caslib_name, domain, len(df), csv_path,
            )

        except Exception as e:
            log.warning("  Failed to download %s/%s: %s", caslib_name, domain, e)

    return {
        "trial_id": caslib_name,
        "n_patients": n_patients,
        "domains_downloaded": ";".join(downloaded),
        "data_path": str(caslib_dir),
        "drugs": "",  # User fills in manually or auto-detected
        "indication": "",
    }


def _auto_detect_drugs(arms: list[str]) -> str:
    """Best-effort drug detection from trial arm names."""
    known_drugs = [
        "cisplatin", "etoposide", "carboplatin", "paclitaxel", "taxol",
        "bevacizumab", "darbepoetin", "gemcitabine", "docetaxel",
        "pemetrexed", "vinorelbine", "topotecan", "irinotecan",
    ]
    found = set()
    for arm in arms:
        arm_lower = arm.lower()
        for drug in known_drugs:
            if drug in arm_lower:
                found.add(drug.capitalize())
    return ";".join(sorted(found)) if found else ""


def download_selected(
    config: RuleEngineConfig,
    caslib_names: list[str] | None = None,
    catalog_path: Path | None = None,
) -> None:
    """Download selected trials from PDS CAS server.

    If caslib_names is provided, download those specific caslibs.
    Otherwise, download all 'relevant' caslibs from the catalog.
    """
    try:
        import swat
    except ImportError:
        log.error("SWAT package not installed. Run: pip install swat")
        sys.exit(1)

    catalog_path = catalog_path or config.pds_data_dir / "catalog.json"

    # Determine which caslibs to download
    catalog: dict = {}
    if catalog_path.exists():
        with open(catalog_path) as f:
            catalog = json.load(f)

    caslib_meta: dict[str, dict] = {}
    for entry in catalog.get("caslibs", []):
        caslib_meta[entry["name"]] = entry

    if caslib_names:
        targets = caslib_names
    else:
        targets = [
            e["name"] for e in catalog.get("caslibs", [])
            if e.get("relevant", False) and e.get("tables")
        ]

    if not targets:
        log.error("No caslibs to download. Run pds_discover.py first or specify names.")
        sys.exit(1)

    log.info("Will download %d caslibs: %s", len(targets), targets)

    # Connect
    username = config.pds_username
    password = config.pds_password
    if not username or not password:
        log.error("PDS credentials not set.")
        sys.exit(1)

    conn = swat.CAS(
        config.pds_cas_url,
        username=username,
        password=password,
    )
    log.info("Connected to PDS CAS.")

    output_dir = config.pds_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download each caslib
    index_rows: list[dict] = []
    for caslib_name in targets:
        meta = caslib_meta.get(caslib_name, {})
        tables = meta.get("tables", [])

        if not tables:
            # Probe tables dynamically if not in catalog
            try:
                file_info = conn.fileinfo(caslib=caslib_name)
                file_df = file_info.get("FileInfo", None)
                if file_df is not None:
                    tables = [str(r.get("Name", r.iloc[0])) for _, r in file_df.iterrows()]
            except Exception as e:
                log.warning("Could not list tables for %s: %s", caslib_name, e)
                continue

        log.info("Downloading %s (%d tables available)...", caslib_name, len(tables))
        row = download_caslib(conn, caslib_name, tables, output_dir)

        # Auto-detect drugs from arm names
        arms = meta.get("arms", [])
        row["drugs"] = _auto_detect_drugs(arms)
        if arms:
            row["arms"] = ";".join(arms)

        index_rows.append(row)

    conn.close()

    # Write/update trial_index.csv
    index_path = output_dir / "trial_index.csv"
    existing_rows: dict[str, dict] = {}
    if index_path.exists():
        with open(index_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows[row["trial_id"]] = row

    # Merge: new entries override existing
    for row in index_rows:
        tid = row["trial_id"]
        if tid in existing_rows:
            # Preserve user-filled fields
            for key in ["drugs", "indication"]:
                if not row.get(key) and existing_rows[tid].get(key):
                    row[key] = existing_rows[tid][key]
        existing_rows[tid] = row

    fieldnames = [
        "trial_id", "drugs", "indication", "n_patients",
        "domains_downloaded", "data_path", "arms",
    ]
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(existing_rows.values(), key=lambda r: r["trial_id"]):
            writer.writerow(row)

    log.info("Trial index written to %s (%d entries)", index_path, len(existing_rows))
    log.info("Download complete. Review trial_index.csv and fill in 'drugs' and 'indication' columns.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Download PDS trial datasets to CSV")
    parser.add_argument(
        "caslibs",
        nargs="*",
        help="Specific caslib names to download (default: all relevant from catalog)",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="Path to catalog.json (default: data/pds/catalog.json)",
    )
    args = parser.parse_args()

    config = RuleEngineConfig()
    download_selected(
        config,
        caslib_names=args.caslibs or None,
        catalog_path=args.catalog,
    )


if __name__ == "__main__":
    main()
