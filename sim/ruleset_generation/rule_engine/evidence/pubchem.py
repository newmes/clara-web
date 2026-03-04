"""Async evidence collector for PubChem PUG REST API.

Retrieves compound properties (MW, logP, TPSA, H-bond counts),
computes Lipinski rule-of-five violations, fetches bioassay summary
counts, and extracts pharmacological classification when available.
"""

from __future__ import annotations

import asyncio
import logging

import requests

from rule_engine.schema import PubChemEvidence

log = logging.getLogger(__name__)

PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


def _fetch_pubchem_sync(drug_name: str, timeout: int) -> PubChemEvidence:
    """Synchronous PubChem lookup — runs inside ``asyncio.to_thread``."""
    session = requests.Session()
    session.headers["User-Agent"] = "RuleEngine/1.0"

    # Step 1 — resolve compound CID
    try:
        resp = session.get(
            f"{PUG_REST}/compound/name/{requests.utils.quote(drug_name)}/cids/JSON",
            timeout=timeout,
        )
        resp.raise_for_status()
        cids = resp.json().get("IdentifierList", {}).get("CID", [])
        if not cids:
            return PubChemEvidence()
        cid = cids[0]
    except Exception as exc:
        log.debug("PubChem CID lookup failed for %s: %s", drug_name, exc)
        return PubChemEvidence()

    # Step 2 — compound properties
    mw = logp = tpsa = None
    hbd = hba = rotatable = 0
    try:
        props_resp = session.get(
            f"{PUG_REST}/compound/cid/{cid}/property/"
            "MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount,RotatableBondCount/JSON",
            timeout=timeout,
        )
        if props_resp.ok:
            props_list = props_resp.json().get("PropertyTable", {}).get("Properties", [])
            if props_list:
                p = props_list[0]
                mw = p.get("MolecularWeight")
                if mw is not None:
                    mw = float(mw)
                logp = p.get("XLogP")
                if logp is not None:
                    logp = float(logp)
                tpsa = p.get("TPSA")
                if tpsa is not None:
                    tpsa = float(tpsa)
                hbd = int(p.get("HBondDonorCount", 0))
                hba = int(p.get("HBondAcceptorCount", 0))
                rotatable = int(p.get("RotatableBondCount", 0))
    except Exception as exc:
        log.debug("PubChem properties failed for CID %d: %s", cid, exc)

    # Step 3 — Lipinski violations
    violations = 0
    if mw is not None and mw > 500:
        violations += 1
    if logp is not None and logp > 5:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1

    # Step 4 — bioassay summary counts
    active_count = total_count = 0
    try:
        assay_resp = session.get(
            f"{PUG_REST}/compound/cid/{cid}/assaysummary/JSON",
            timeout=timeout,
        )
        if assay_resp.ok:
            table = assay_resp.json().get("Table", {})
            rows = table.get("Row", [])
            total_count = len(rows)
            for row in rows:
                cells = row.get("Cell", [])
                # Activity Outcome is typically the 4th column
                if len(cells) > 3 and str(cells[3]).lower() == "active":
                    active_count += 1
    except Exception as exc:
        log.debug("PubChem bioassay failed for CID %d: %s", cid, exc)

    # Step 5 — pharmacological classification
    pharm_class: str | None = None
    try:
        class_resp = session.get(
            f"{PUG_REST}/compound/cid/{cid}/classification/JSON",
            timeout=timeout,
        )
        if class_resp.ok:
            hierarchies = class_resp.json().get("Hierarchies", {}).get("Hierarchy", [])
            for h in hierarchies:
                source = h.get("SourceName", "")
                if "MeSH" in source or "Pharmacological" in source.replace(" ", ""):
                    nodes = h.get("Node", [])
                    if nodes:
                        info = nodes[-1].get("Information", {})
                        raw_name = info.get("Name", "")
                        # Name can be a nested dict: {"StringWithMarkup": {"String": "..."}}
                        if isinstance(raw_name, dict):
                            raw_name = raw_name.get("StringWithMarkup", {}).get("String", "")
                        if raw_name:
                            pharm_class = raw_name
                            break
    except Exception as exc:
        log.debug("PubChem classification failed for CID %d: %s", cid, exc)

    evidence = PubChemEvidence(
        found=True,
        pubchem_cid=cid,
        molecular_weight=mw,
        logp=logp,
        tpsa=tpsa,
        hbd_count=hbd,
        hba_count=hba,
        rotatable_bonds=rotatable,
        lipinski_violations=violations,
        bioassay_active_count=active_count,
        bioassay_total_count=total_count,
        pharmacological_class=pharm_class,
    )
    log.debug(
        "PubChem: %s → CID %d, MW=%.1f, logP=%s, Lipinski=%d, assays=%d/%d",
        drug_name, cid, mw or 0, logp, violations, active_count, total_count,
    )
    return evidence


async def fetch_pubchem(
    drug_name: str,
    timeout: int = 10,
) -> PubChemEvidence:
    """Fetch compound properties and bioassay data from PubChem.

    Args:
        drug_name: Generic drug name to search for.
        timeout: HTTP request timeout in seconds.

    Returns:
        Populated PubChemEvidence; empty defaults on any failure.
    """
    try:
        return await asyncio.to_thread(_fetch_pubchem_sync, drug_name, timeout)
    except Exception as exc:
        log.warning("PubChem request failed for %s: %s", drug_name, exc)
        return PubChemEvidence()
