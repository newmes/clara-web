#!/usr/bin/env python3
"""Download NSCLC trial data from Project Data Sphere CAS.

Probes each LungNo_* caslib, discovers all available tables,
downloads ALL sas7bdat/csv files (not just standard domain names),
and uses pagination for large tables.

Usage:
    cd /home/ubuntu/samuel/rule_discovery
    python scripts/pds_download_nsclc.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rule_engine.config import RuleEngineConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# NSCLC caslibs from catalog
NSCLC_CASLIBS = [
    "LungNo_Celgene_2007_108",
    "LungNo_ELiLill_2006_116",
    "LungNo_EliLill_2008_141",
    "LungNo_EliLill_2008_148",
    "LungNo_EliLill_2009_438",
    "LungNo_EliLill_2010_272",
    "LungNo_MerckKG_2007_145",
    "LungNo_Pfizer_2007_115",
    "LungNo_SanofiU_2007_133",
]

# File extensions that contain actual data
DATA_EXTENSIONS = {".sas7bdat", ".csv", ".sashdat"}


def _is_data_file(filename: str) -> bool:
    """Check if a file is a loadable data file (not xlsx/xls/pdf)."""
    lower = filename.lower()
    for ext in DATA_EXTENSIONS:
        if lower.endswith(ext):
            return True
    # Also try files without extensions (dataFiles_XXX)
    if "." not in filename:
        return True
    return False


def _table_name(filename: str) -> str:
    """Extract a clean table name from a filename."""
    base = filename.rsplit(".", 1)[0] if "." in filename else filename
    return base.lower()


def fetch_all_rows(conn, table_name: str, chunk_size: int = 1000) -> pd.DataFrame | None:
    """Fetch all rows from a CAS table with pagination."""
    try:
        n_rows = int(conn.numrows(table=table_name).get("numrows", 0))
    except Exception:
        n_rows = 0

    if n_rows == 0:
        return None

    chunks = []
    for start in range(1, n_rows + 1, chunk_size):
        end = min(start + chunk_size - 1, n_rows)
        result = conn.fetch(table=table_name, from_=start, to=end)
        chunk = result.get("Fetch", None)
        if chunk is not None and len(chunk) > 0:
            chunks.append(chunk)

    if not chunks:
        return None

    return pd.concat(chunks, ignore_index=True)


def download_caslib_all(conn, caslib_name: str, output_dir: Path) -> dict:
    """Download ALL data files from a caslib."""
    caslib_dir = output_dir / caslib_name
    caslib_dir.mkdir(parents=True, exist_ok=True)

    # Probe available files
    try:
        file_info = conn.fileinfo(caslib=caslib_name)
        file_df = file_info.get("FileInfo", None)
        if file_df is None or len(file_df) == 0:
            log.warning("  %s: no files found", caslib_name)
            return {"trial_id": caslib_name, "n_files": 0, "files": []}
    except Exception as e:
        log.warning("  %s: fileinfo failed: %s", caslib_name, e)
        return {"trial_id": caslib_name, "n_files": 0, "files": []}

    # Get file names
    files = []
    for _, row in file_df.iterrows():
        name = str(row.get("Name", row.iloc[0]))
        files.append(name)

    log.info("  %s: %d files found: %s", caslib_name, len(files),
             [f for f in files[:10]])

    downloaded = []
    n_patients = 0

    for filename in files:
        if not _is_data_file(filename):
            log.info("    Skipping non-data file: %s", filename)
            continue

        tname = _table_name(filename)
        tmp_name = f"_tmp_{tname}"[:32]  # CAS table name limit

        try:
            conn.loadtable(
                caslib=caslib_name,
                path=filename,
                casout={"name": tmp_name, "replace": True},
            )
        except Exception as e:
            log.warning("    Failed to load %s: %s", filename, e)
            continue

        df = fetch_all_rows(conn, tmp_name)
        conn.droptable(name=tmp_name, quiet=True)

        if df is None or len(df) == 0:
            continue

        # Save as CSV
        csv_name = tname + ".csv"
        csv_path = caslib_dir / csv_name
        df.to_csv(csv_path, index=False)
        downloaded.append(csv_name)

        log.info("    %s: %d rows x %d cols", csv_name, len(df), len(df.columns))

        # Check if this is DM/ADSL (demographics)
        cols_lower = [c.lower() for c in df.columns]
        if tname in ("dm", "adsl", "demog", "demo") or "age" in cols_lower:
            n_patients = max(n_patients, len(df))

    return {
        "trial_id": caslib_name,
        "n_files": len(downloaded),
        "files": downloaded,
        "n_patients": n_patients,
    }


def main():
    config = RuleEngineConfig()
    output_dir = config.pds_data_dir

    import swat
    os.environ.setdefault("CAS_CLIENT_SSL_CA_LIST", "/etc/ssl/certs/ca-certificates.crt")

    log.info("Connecting to PDS CAS...")
    conn = swat.CAS(
        config.pds_cas_url,
        username=config.pds_username,
        password=config.pds_password,
    )
    log.info("Connected.")

    results = []
    for caslib in NSCLC_CASLIBS:
        log.info("Processing %s...", caslib)
        result = download_caslib_all(conn, caslib, output_dir)
        results.append(result)
        log.info("  → %d files downloaded, %d patients",
                 result["n_files"], result.get("n_patients", 0))

    conn.close()

    # Summary
    log.info("\n=== DOWNLOAD SUMMARY ===")
    for r in results:
        log.info("  %s: %d files, %d patients — %s",
                 r["trial_id"], r["n_files"], r.get("n_patients", 0),
                 r.get("files", [])[:5])


if __name__ == "__main__":
    main()
