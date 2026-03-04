"""Async evidence collector for OnSIDES — validated drug-ADE pairs from FDA labels.

OnSIDES (v3.1.0) contains 7.1M drug-ADE associations extracted from 51,460 FDA
drug labels using PubMedBERT (F1=0.90).  The local SQLite database
(built by ``scripts/download_onsides.sh``) stores a pre-materialized
``ingredient_ade_summary`` table for fast ingredient-level lookups.

Returns top ADE pairs sorted by label_count descending, with boxed-warning flags.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path

from rule_engine.config import RuleEngineConfig
from rule_engine.schema import OnSIDESEvidence

log = logging.getLogger(__name__)


def _fetch_onsides_sync(drug_name: str, db_path: Path, limit: int = 50) -> OnSIDESEvidence:
    """Synchronous OnSIDES lookup — runs inside ``asyncio.to_thread``."""
    if not db_path.exists():
        log.debug("OnSIDES DB not found at %s", db_path)
        return OnSIDESEvidence()

    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Look up ingredient by case-insensitive name match
        cur.execute(
            """
            SELECT ingredient_rxnorm_id, ingredient_name,
                   pt_meddra_id, pt_meddra_term,
                   label_count, mean_pred_score, max_pred_score, is_boxed_warning
            FROM ingredient_ade_summary
            WHERE ingredient_name = ? COLLATE NOCASE
            ORDER BY label_count DESC
            LIMIT ?
            """,
            (drug_name, limit),
        )
        rows = cur.fetchall()

        # Fallback: for multi-word names (e.g. ADCs like "Trastuzumab deruxtecan"),
        # try the first word as a base drug name lookup
        if not rows and " " in drug_name:
            base_name = drug_name.split()[0]
            cur.execute(
                """
                SELECT ingredient_rxnorm_id, ingredient_name,
                       pt_meddra_id, pt_meddra_term,
                       label_count, mean_pred_score, max_pred_score, is_boxed_warning
                FROM ingredient_ade_summary
                WHERE ingredient_name = ? COLLATE NOCASE
                ORDER BY label_count DESC
                LIMIT ?
                """,
                (base_name, limit),
            )
            rows = cur.fetchall()
            if rows:
                log.debug(
                    "OnSIDES: '%s' not found, using base name '%s' (%d pairs)",
                    drug_name, base_name, len(rows),
                )
        conn.close()

        if not rows:
            log.debug("OnSIDES: no ADE pairs found for '%s'", drug_name)
            return OnSIDESEvidence()

        ae_pairs = []
        boxed_warning_aes: list[str] = []
        seen_meddra_ids: set[int] = set()
        for row in rows:
            pair = {
                "pt_meddra_id": row["pt_meddra_id"],
                "pt_meddra_term": row["pt_meddra_term"],
                "label_count": row["label_count"],
                "mean_pred_score": round(row["mean_pred_score"], 3) if row["mean_pred_score"] else None,
                "max_pred_score": round(row["max_pred_score"], 3) if row["max_pred_score"] else None,
                "is_boxed_warning": bool(row["is_boxed_warning"]),
            }
            ae_pairs.append(pair)
            seen_meddra_ids.add(row["pt_meddra_id"])
            if pair["is_boxed_warning"]:
                boxed_warning_aes.append(row["pt_meddra_term"])

        # Ensure ALL boxed-warning AEs are included (may fall outside top-N limit)
        matched_name = rows[0]["ingredient_name"]
        conn_bw = sqlite3.connect(str(db_path), timeout=5)
        conn_bw.row_factory = sqlite3.Row
        bw_rows = conn_bw.execute(
            """
            SELECT ingredient_rxnorm_id, ingredient_name,
                   pt_meddra_id, pt_meddra_term,
                   label_count, mean_pred_score, max_pred_score, is_boxed_warning
            FROM ingredient_ade_summary
            WHERE ingredient_name = ? COLLATE NOCASE AND is_boxed_warning = 1
            """,
            (matched_name,),
        ).fetchall()
        conn_bw.close()
        for bw_row in bw_rows:
            if bw_row["pt_meddra_id"] not in seen_meddra_ids:
                pair = {
                    "pt_meddra_id": bw_row["pt_meddra_id"],
                    "pt_meddra_term": bw_row["pt_meddra_term"],
                    "label_count": bw_row["label_count"],
                    "mean_pred_score": round(bw_row["mean_pred_score"], 3) if bw_row["mean_pred_score"] else None,
                    "max_pred_score": round(bw_row["max_pred_score"], 3) if bw_row["max_pred_score"] else None,
                    "is_boxed_warning": True,
                }
                ae_pairs.append(pair)
                seen_meddra_ids.add(bw_row["pt_meddra_id"])
                boxed_warning_aes.append(bw_row["pt_meddra_term"])
                log.debug(
                    "OnSIDES: added boxed-warning AE '%s' (label_count=%d) that fell outside top-%d",
                    bw_row["pt_meddra_term"], bw_row["label_count"], limit,
                )

        # Get total count (may exceed limit) — use matched name from rows
        conn2 = sqlite3.connect(str(db_path), timeout=5)
        total = conn2.execute(
            "SELECT COUNT(*) FROM ingredient_ade_summary WHERE ingredient_name = ? COLLATE NOCASE",
            (matched_name,),
        ).fetchone()[0]
        conn2.close()

        evidence = OnSIDESEvidence(
            found=True,
            drug_concept_name=rows[0]["ingredient_name"],
            ae_pairs=ae_pairs,
            boxed_warning_aes=boxed_warning_aes,
            total_pairs=total,
        )
        log.debug(
            "OnSIDES: %s → %d ADE pairs (%d boxed warnings), total=%d",
            drug_name, len(ae_pairs), len(boxed_warning_aes), total,
        )
        return evidence

    except Exception as exc:
        log.warning("OnSIDES query failed for %s: %s", drug_name, exc)
        return OnSIDESEvidence()


async def fetch_onsides(drug_name: str, config: RuleEngineConfig) -> OnSIDESEvidence:
    """Fetch validated drug-ADE pairs from the local OnSIDES database.

    Args:
        drug_name: Generic drug name to search for.
        config: Pipeline config (provides ``onsides_db`` path).

    Returns:
        Populated OnSIDESEvidence; empty defaults if DB missing or no match.
    """
    try:
        return await asyncio.to_thread(_fetch_onsides_sync, drug_name, config.onsides_db)
    except Exception as exc:
        log.warning("OnSIDES request failed for %s: %s", drug_name, exc)
        return OnSIDESEvidence()
