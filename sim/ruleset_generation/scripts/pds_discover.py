#!/usr/bin/env python3
"""Discover available datasets on Project Data Sphere CAS server.

Connects to the PDS SAS Viya CAS server, enumerates all caslibs (each
represents one clinical trial), and identifies lung cancer / SCLC / NSCLC
trials by name matching.  Outputs a catalog JSON to data/pds/catalog.json.

Usage:
    export RULE_ENGINE_PDS_USERNAME="your_username"
    export RULE_ENGINE_PDS_PASSWORD="your_password"
    python scripts/pds_discover.py [--output data/pds/catalog.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rule_engine.config import RuleEngineConfig

log = logging.getLogger(__name__)

# Keywords for identifying relevant trials (case-insensitive)
_RELEVANT_KEYWORDS = [
    "lung", "sclc", "nsclc",
    "cisplatin", "etoposide", "carboplatin", "paclitaxel", "taxol",
    "bevacizumab", "darbepoetin", "gemcitabine",
    "thoracic", "carcinoma",
]

# Common SDTM/ADaM domain table names to probe
_DOMAIN_NAMES = [
    "dm", "ae", "ex", "adsl", "adae", "adrs", "adtte", "adlb",
    "rs", "tu", "tr", "vs", "lb", "cm", "mh", "ds", "qs",
]


def _is_relevant(caslib_name: str) -> bool:
    """Check if a caslib name matches any lung cancer / drug keyword."""
    name_lower = caslib_name.lower()
    return any(kw in name_lower for kw in _RELEVANT_KEYWORDS)


def discover(config: RuleEngineConfig, output_path: Path) -> dict:
    """Connect to PDS CAS and enumerate available datasets.

    Returns the catalog dict and writes it to output_path.
    """
    try:
        import swat
    except ImportError:
        log.error("SWAT package not installed. Run: pip install swat")
        sys.exit(1)

    username = config.pds_username
    password = config.pds_password
    if not username or not password:
        log.error(
            "PDS credentials not set. Export RULE_ENGINE_PDS_USERNAME and "
            "RULE_ENGINE_PDS_PASSWORD environment variables."
        )
        sys.exit(1)

    # SSL config — SAS Viya may need custom CA handling
    ssl_ca = os.environ.get("CAS_CLIENT_SSL_CA_LIST", "")
    if ssl_ca:
        log.info("Using custom SSL CA list: %s", ssl_ca)

    log.info("Connecting to PDS CAS at %s ...", config.pds_cas_url)
    conn = swat.CAS(
        config.pds_cas_url,
        username=username,
        password=password,
    )
    log.info("Connected to CAS server: %s", conn.serverstatus().get("About", {}).get("CAS", "unknown"))

    # Enumerate caslibs
    caslib_info = conn.caslibinfo()
    caslib_df = caslib_info.get("CASLibInfo", None)
    if caslib_df is None or len(caslib_df) == 0:
        log.warning("No caslibs found on the server.")
        conn.close()
        return {"caslibs": []}

    catalog: dict = {"caslibs": [], "_server": config.pds_cas_url}
    total = len(caslib_df)
    log.info("Found %d caslibs. Scanning for relevant trials...", total)

    for idx, row in caslib_df.iterrows():
        caslib_name = row.get("Name", str(row.iloc[0]))
        caslib_desc = row.get("Description", "")
        caslib_type = row.get("Type", "")

        # Probe available tables in this caslib
        try:
            file_info = conn.fileinfo(caslib=caslib_name)
            file_df = file_info.get("FileInfo", None)
            tables = []
            if file_df is not None and len(file_df) > 0:
                tables = [
                    str(r.get("Name", r.iloc[0]))
                    for _, r in file_df.iterrows()
                ]
        except Exception:
            tables = []

        relevant = _is_relevant(caslib_name) or _is_relevant(caslib_desc)

        # For relevant caslibs, try to load DM table to get patient count
        n_patients = 0
        arms: list[str] = []
        if relevant and tables:
            dm_candidates = [t for t in tables if t.lower().startswith("dm")]
            if dm_candidates:
                try:
                    conn.loadtable(
                        caslib=caslib_name,
                        path=dm_candidates[0],
                        casout={"name": "_tmp_dm", "replace": True},
                    )
                    result = conn.numrows(table="_tmp_dm")
                    n_patients = int(result.get("numrows", 0))

                    # Try to get arm names
                    col_info = conn.columninfo(table="_tmp_dm")
                    col_df = col_info.get("ColumnInfo", None)
                    if col_df is not None:
                        col_names = [str(c).upper() for c in col_df["Column"].tolist()]
                        arm_col = None
                        for candidate in ["ARM", "ACTARM", "ARMCD", "TRT01A"]:
                            if candidate in col_names:
                                arm_col = candidate
                                break
                        if arm_col:
                            fetch_result = conn.fetch(
                                table="_tmp_dm",
                                to=min(n_patients, 5000),
                            )
                            fetch_df = fetch_result.get("Fetch", None)
                            if fetch_df is not None and arm_col in fetch_df.columns:
                                arms = sorted(fetch_df[arm_col].dropna().unique().tolist())

                    conn.droptable(name="_tmp_dm", quiet=True)
                except Exception as e:
                    log.debug("Could not probe DM for %s: %s", caslib_name, e)

        entry = {
            "name": caslib_name,
            "description": caslib_desc,
            "type": caslib_type,
            "tables": tables,
            "n_patients": n_patients,
            "arms": arms,
            "relevant": relevant,
        }
        catalog["caslibs"].append(entry)

        if relevant:
            log.info(
                "  [RELEVANT] %s: %d tables, %d patients, arms=%s",
                caslib_name, len(tables), n_patients, arms,
            )

    conn.close()

    # Summary
    relevant_count = sum(1 for c in catalog["caslibs"] if c["relevant"])
    log.info(
        "Discovery complete: %d total caslibs, %d relevant to lung cancer drugs.",
        total, relevant_count,
    )

    # Write catalog
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(catalog, f, indent=2, default=str)
    log.info("Catalog written to %s", output_path)

    return catalog


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Discover PDS CAS datasets")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output catalog path (default: data/pds/catalog.json)",
    )
    args = parser.parse_args()

    config = RuleEngineConfig()
    output_path = args.output or config.pds_data_dir / "catalog.json"

    discover(config, output_path)


if __name__ == "__main__":
    main()
