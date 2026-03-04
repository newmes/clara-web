"""Local database evidence collectors — DrugBank and PrimeKG.

Reads CSV files directly from disk (paths from RuleEngineConfig) to avoid
importing anything from the agent/ package.  All blocking I/O is wrapped in
``asyncio.to_thread`` so callers can ``await`` safely.
"""

from __future__ import annotations

import asyncio
import csv
import logging
from collections import Counter
from pathlib import Path

from rule_engine.config import RuleEngineConfig
from rule_engine.schema import DrugBankEvidence, PrimeKGEvidence

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DrugBank
# ---------------------------------------------------------------------------

def _read_drugbank_sync(drug_name: str, config: RuleEngineConfig) -> DrugBankEvidence:
    """Synchronous DrugBank lookup — runs inside ``to_thread``."""
    drugbank_dir: Path = config.drugbank_dir

    # --- 1. Normalize drug name → DrugBank ID via vocabulary CSV ---
    vocab_path = drugbank_dir / "drugbank_vocabulary.csv"
    if not vocab_path.exists():
        log.warning("DrugBank vocabulary not found: %s", vocab_path)
        return DrugBankEvidence(found=False)

    drug_id: str | None = None
    name_lower = drug_name.lower()

    with open(vocab_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            common = row.get("Common name", "").strip()
            synonyms = row.get("Synonyms", "").strip()

            # Exact match on common name
            if common.lower() == name_lower:
                drug_id = row.get("DrugBank ID", "").strip()
                break

            # Check synonyms (pipe-separated)
            if synonyms:
                for syn in synonyms.split(" | "):
                    if syn.strip().lower() == name_lower:
                        drug_id = row.get("DrugBank ID", "").strip()
                        break
            if drug_id:
                break

    # Fallback: partial match (starts-with) on a second pass
    if not drug_id:
        with open(vocab_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                common = row.get("Common name", "").strip()
                if common.lower().startswith(name_lower) or name_lower.startswith(common.lower()):
                    drug_id = row.get("DrugBank ID", "").strip()
                    break

    if not drug_id:
        return DrugBankEvidence(found=False)

    # --- 2. Mechanism of Action ---
    moa: str | None = None
    moa_path = drugbank_dir / "moa.csv"
    if moa_path.exists():
        with open(moa_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("ID", "").strip() == drug_id:
                    moa = row.get("MOA", "").strip() or None
                    break

    # --- 3. Protein targets ---
    targets: list[dict] = []
    targets_path = drugbank_dir / "drug_protein.csv"
    if targets_path.exists():
        with open(targets_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("DrugBank", "").strip() == drug_id:
                    targets.append({
                        "uniprot_id": row.get("UniProtID", "").strip(),
                        "uniprot_name": row.get("UniProtName", "").strip(),
                        "gene_id": row.get("NCBIGeneID", "").strip(),
                        "relation": row.get("relation", "").strip(),
                    })

    # --- 4. Drug-drug interaction count ---
    ddi_count = 0
    ddi_path = drugbank_dir / "drug_drug.csv"
    if ddi_path.exists():
        with open(ddi_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("drug1", "").strip() == drug_id or row.get("drug2", "").strip() == drug_id:
                    ddi_count += 1

    return DrugBankEvidence(
        found=True,
        drugbank_id=drug_id,
        moa=moa,
        targets=targets,
        ddi_count=ddi_count,
    )


async def fetch_drugbank(drug_name: str, config: RuleEngineConfig) -> DrugBankEvidence:
    """Look up DrugBank evidence for *drug_name* from local CSV files.

    Args:
        drug_name: Drug name (case-insensitive).
        config: Pipeline configuration providing ``drugbank_dir``.

    Returns:
        Populated :class:`DrugBankEvidence`; ``found=False`` on any failure.
    """
    try:
        return await asyncio.to_thread(_read_drugbank_sync, drug_name, config)
    except Exception as exc:
        log.warning("DrugBank lookup failed for %s: %s", drug_name, exc)
        return DrugBankEvidence(found=False)


# ---------------------------------------------------------------------------
# PrimeKG
# ---------------------------------------------------------------------------

def _read_primekg_sync(
    drug_name: str,
    indication: str,
    config: RuleEngineConfig,
) -> PrimeKGEvidence:
    """Synchronous PrimeKG lookup — runs inside ``to_thread``."""
    nodes_path: Path = config.primekg_nodes
    edges_path: Path = config.primekg_edges

    if not nodes_path.exists() or not edges_path.exists():
        log.warning("PrimeKG files not found: nodes=%s edges=%s", nodes_path, edges_path)
        return PrimeKGEvidence(found=False)

    # --- 1. Find the drug node (case-insensitive partial match) ---
    drug_index: int | None = None
    name_lower = drug_name.lower()

    with open(nodes_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            node_name = row.get("node_name", "").strip()
            if name_lower in node_name.lower():
                node_type = row.get("node_type", "").strip()
                if node_type == "drug":
                    drug_index = int(row.get("node_index", -1))
                    break

    if drug_index is None or drug_index < 0:
        return PrimeKGEvidence(found=False)

    # --- 2. Collect all neighbor indices from edges ---
    neighbor_indices: set[int] = set()
    with open(edges_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            x_idx = int(row.get("x_index", -1))
            y_idx = int(row.get("y_index", -1))
            if x_idx == drug_index:
                neighbor_indices.add(y_idx)
            elif y_idx == drug_index:
                neighbor_indices.add(x_idx)

    if not neighbor_indices:
        return PrimeKGEvidence(found=True, neighbor_summary="0 neighbors")

    # --- 3. Resolve neighbor nodes ---
    # Maps node_index → {type, name}
    neighbor_info: dict[int, dict] = {}
    with open(nodes_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row.get("node_index", -1))
            if idx in neighbor_indices:
                neighbor_info[idx] = {
                    "type": row.get("node_type", "").strip(),
                    "name": row.get("node_name", "").strip(),
                }

    # --- 4. Categorize neighbors ---
    disease_associations: list[dict] = []
    gene_targets: list[dict] = []
    type_counts: Counter[str] = Counter()

    for info in neighbor_info.values():
        ntype = info["type"]
        type_counts[ntype] += 1

        if ntype == "disease":
            disease_associations.append({"name": info["name"]})
        elif ntype == "gene/protein":
            gene_targets.append({"name": info["name"]})

    # Build summary string, e.g. "12 gene targets, 5 diseases, 3 pathways"
    parts: list[str] = []
    label_map = {"gene/protein": "gene targets", "disease": "diseases", "pathway": "pathways"}
    for ntype, count in type_counts.most_common():
        label = label_map.get(ntype, ntype)
        parts.append(f"{count} {label}")
    neighbor_summary = ", ".join(parts) if parts else "0 neighbors"

    return PrimeKGEvidence(
        found=True,
        disease_associations=disease_associations,
        gene_targets=gene_targets,
        neighbor_summary=neighbor_summary,
    )


async def fetch_primekg(
    drug_name: str,
    indication: str,
    config: RuleEngineConfig,
) -> PrimeKGEvidence:
    """Look up PrimeKG evidence for *drug_name* from local CSV files.

    Args:
        drug_name: Drug name (case-insensitive partial match on node_name).
        indication: Indication term (unused currently, reserved for future filtering).
        config: Pipeline configuration providing ``primekg_nodes`` / ``primekg_edges``.

    Returns:
        Populated :class:`PrimeKGEvidence`; ``found=False`` on any failure.
    """
    try:
        return await asyncio.to_thread(_read_primekg_sync, drug_name, indication, config)
    except Exception as exc:
        log.warning("PrimeKG lookup failed for %s: %s", drug_name, exc)
        return PrimeKGEvidence(found=False)
