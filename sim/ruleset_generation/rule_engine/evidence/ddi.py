"""Cross-regimen drug-drug interaction (DDI) evidence collection.

Collects DDI data from two sources:
1. DrugBank drug_drug.csv — name-based matching for cross-regimen drug pairs
2. PrimeKG shared gene/protein targets — mechanistic interaction signals

Used by the multi-indication pipeline to enrich the LLM merge step.
"""

from __future__ import annotations

import asyncio
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

from rule_engine.config import RuleEngineConfig

log = logging.getLogger(__name__)


@dataclass
class DDIPair:
    """A single drug-drug interaction pair with evidence."""

    drug_a: str
    drug_b: str
    drugbank_relation: str = ""
    shared_targets: list[str] = field(default_factory=list)


@dataclass
class DDIEvidence:
    """Cross-regimen DDI evidence bundle."""

    pairs: list[DDIPair] = field(default_factory=list)
    total_drugbank_hits: int = 0
    total_shared_targets: int = 0


def _build_cross_regimen_pairs(regimens: list[tuple[list[str], str]]) -> set[tuple[str, str]]:
    """Build cross-regimen drug pairs (skip intra-regimen pairs).

    Returns set of (drug_a, drug_b) tuples where drug_a < drug_b (sorted).
    """
    # Group drugs by regimen index
    regimen_drugs: list[set[str]] = []
    for drugs, _ in regimens:
        regimen_drugs.append({d.lower() for d in drugs})

    pairs: set[tuple[str, str]] = set()
    for i in range(len(regimen_drugs)):
        for j in range(i + 1, len(regimen_drugs)):
            for da in regimen_drugs[i]:
                for db in regimen_drugs[j]:
                    if da != db:
                        pair = tuple(sorted([da, db]))
                        pairs.add(pair)
    return pairs


def _scan_drugbank_ddi_sync(
    target_pairs: set[tuple[str, str]],
    config: RuleEngineConfig,
) -> dict[tuple[str, str], str]:
    """Single-pass scan of drug_drug.csv matching by name1/name2 fields.

    Returns dict mapping (drug_a_lower, drug_b_lower) → relation string.
    """
    ddi_path: Path = config.drugbank_dir / "drug_drug.csv"
    if not ddi_path.exists():
        log.warning("DrugBank drug_drug.csv not found: %s", ddi_path)
        return {}

    # Build lookup set of target name pairs (both orderings for fast check)
    target_names: set[frozenset[str]] = {frozenset(p) for p in target_pairs}

    hits: dict[tuple[str, str], str] = {}
    with open(ddi_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n1 = row.get("name1", "").strip().lower()
            n2 = row.get("name2", "").strip().lower()
            if not n1 or not n2:
                continue
            pair_key = frozenset([n1, n2])
            if pair_key in target_names:
                sorted_pair = tuple(sorted([n1, n2]))
                relation = row.get("relation", "").strip()
                # Keep first hit (relations are almost always "synergistic interaction")
                if sorted_pair not in hits:
                    hits[sorted_pair] = relation

    return hits


def _scan_primekg_shared_targets_sync(
    target_pairs: set[tuple[str, str]],
    config: RuleEngineConfig,
) -> dict[tuple[str, str], list[str]]:
    """Find shared gene/protein neighbors between cross-regimen drugs in PrimeKG.

    Returns dict mapping (drug_a_lower, drug_b_lower) → list of shared target names.
    """
    nodes_path: Path = config.primekg_nodes
    edges_path: Path = config.primekg_edges

    if not nodes_path.exists() or not edges_path.exists():
        log.warning("PrimeKG files not found for DDI shared target scan")
        return {}

    # Collect all unique drug names we need to look up
    all_drugs: set[str] = set()
    for da, db in target_pairs:
        all_drugs.add(da)
        all_drugs.add(db)

    # Step 1: Find node indices for target drugs
    drug_indices: dict[str, int] = {}  # drug_name_lower → node_index
    with open(nodes_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            node_name = row.get("node_name", "").strip()
            node_type = row.get("node_type", "").strip()
            if node_type == "drug" and node_name.lower() in all_drugs:
                drug_indices[node_name.lower()] = int(row.get("node_index", -1))

    if len(drug_indices) < 2:
        return {}

    # Step 2: Collect gene/protein neighbors for each drug
    index_to_drug: dict[int, str] = {idx: name for name, idx in drug_indices.items()}
    drug_neighbors: dict[str, set[int]] = {name: set() for name in drug_indices}

    with open(edges_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            x_idx = int(row.get("x_index", -1))
            y_idx = int(row.get("y_index", -1))
            x_type = row.get("x_type", "").strip()
            y_type = row.get("y_type", "").strip()

            # Only care about drug → gene/protein edges
            if x_idx in index_to_drug and y_type == "gene/protein":
                drug_neighbors[index_to_drug[x_idx]].add(y_idx)
            elif y_idx in index_to_drug and x_type == "gene/protein":
                drug_neighbors[index_to_drug[y_idx]].add(x_idx)

    # Step 3: Find shared targets for each pair
    # Resolve target node indices to names
    all_target_indices: set[int] = set()
    for neighbors in drug_neighbors.values():
        all_target_indices.update(neighbors)

    target_names: dict[int, str] = {}
    if all_target_indices:
        with open(nodes_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                idx = int(row.get("node_index", -1))
                if idx in all_target_indices:
                    target_names[idx] = row.get("node_name", "").strip()

    result: dict[tuple[str, str], list[str]] = {}
    for da, db in target_pairs:
        if da not in drug_neighbors or db not in drug_neighbors:
            continue
        shared_indices = drug_neighbors[da] & drug_neighbors[db]
        if shared_indices:
            shared_names = [target_names.get(idx, f"node_{idx}") for idx in shared_indices]
            result[(da, db)] = sorted(shared_names)

    return result


async def collect_ddi_evidence(
    regimens: list[tuple[list[str], str]],
    config: RuleEngineConfig,
) -> DDIEvidence:
    """Collect cross-regimen DDI evidence from DrugBank and PrimeKG.

    Args:
        regimens: List of (drugs, indication) tuples — one per regimen.
        config: Pipeline configuration.

    Returns:
        DDIEvidence with interaction pairs and shared targets.
    """
    target_pairs = _build_cross_regimen_pairs(regimens)
    if not target_pairs:
        log.info("No cross-regimen drug pairs to check for DDIs")
        return DDIEvidence()

    log.info("Checking %d cross-regimen drug pairs for DDIs", len(target_pairs))

    # Run DrugBank and PrimeKG scans concurrently
    drugbank_task = asyncio.to_thread(_scan_drugbank_ddi_sync, target_pairs, config)
    primekg_task = asyncio.to_thread(_scan_primekg_shared_targets_sync, target_pairs, config)

    drugbank_hits, primekg_shared = await asyncio.gather(drugbank_task, primekg_task)

    # Merge results into DDIPair objects
    all_pair_keys = set(drugbank_hits.keys()) | set(primekg_shared.keys())
    pairs: list[DDIPair] = []
    for pair_key in sorted(all_pair_keys):
        da, db = pair_key
        pairs.append(DDIPair(
            drug_a=da,
            drug_b=db,
            drugbank_relation=drugbank_hits.get(pair_key, ""),
            shared_targets=primekg_shared.get(pair_key, []),
        ))

    evidence = DDIEvidence(
        pairs=pairs,
        total_drugbank_hits=len(drugbank_hits),
        total_shared_targets=sum(len(v) for v in primekg_shared.values()),
    )

    log.info(
        "DDI evidence: %d pairs found (%d DrugBank hits, %d shared targets)",
        len(pairs), evidence.total_drugbank_hits, evidence.total_shared_targets,
    )
    return evidence
