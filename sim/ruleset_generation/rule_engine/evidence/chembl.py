"""Async evidence collector for the ChEMBL REST API.

Performs a multi-step lookup:
  1. Search for the molecule by name.
  2. Retrieve activity count and known mechanisms of action.
  3. Package results into a ChEMBLEvidence model.

Mirrors the logic in agent/scoring/evidence.py but uses httpx (async)
and returns the rule_engine Pydantic schema directly.
"""

from __future__ import annotations

import logging

import httpx

from rule_engine.schema import ChEMBLEvidence

log = logging.getLogger(__name__)

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


async def fetch_chembl(
    drug_name: str,
    timeout: int = 10,
) -> ChEMBLEvidence:
    """Fetch molecule, activity, and mechanism data from ChEMBL.

    Args:
        drug_name: Generic drug name to search for.
        timeout: HTTP request timeout in seconds.

    Returns:
        Populated ChEMBLEvidence; empty defaults on any failure.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Step 1 — search molecule by name
            search_resp = await client.get(
                f"{CHEMBL_API}/molecule/search.json",
                params={"q": drug_name, "limit": 1},
            )
            search_resp.raise_for_status()
            molecules = search_resp.json().get("molecules", [])

            if not molecules:
                log.debug("ChEMBL: no molecule found for %s", drug_name)
                return ChEMBLEvidence()

            molecule = molecules[0]
            chembl_id: str = molecule.get("molecule_chembl_id", "")
            max_phase: int = int(float(molecule.get("max_phase", 0) or 0))
            molecule_type: str | None = molecule.get("molecule_type")

            # Step 2 — activity count
            activity_count = 0
            try:
                activity_resp = await client.get(
                    f"{CHEMBL_API}/activity.json",
                    params={"molecule_chembl_id": chembl_id, "limit": 1},
                )
                if activity_resp.is_success:
                    activity_count = (
                        activity_resp.json()
                        .get("page_meta", {})
                        .get("total_count", 0)
                    )
            except (httpx.HTTPError, Exception) as exc:
                log.warning("ChEMBL activity lookup failed for %s: %s", chembl_id, exc)

            # Step 3 — mechanisms of action
            mechanism_of_action: str | None = None
            target_count = 0
            try:
                mech_resp = await client.get(
                    f"{CHEMBL_API}/mechanism.json",
                    params={"molecule_chembl_id": chembl_id, "limit": 100},
                )
                if mech_resp.is_success:
                    mechanisms = mech_resp.json().get("mechanisms", [])
                    target_count = len(mechanisms)
                    descriptions = [
                        m.get("mechanism_of_action", "")
                        for m in mechanisms
                        if m.get("mechanism_of_action")
                    ]
                    if descriptions:
                        mechanism_of_action = "; ".join(descriptions)
            except (httpx.HTTPError, Exception) as exc:
                log.warning("ChEMBL mechanism lookup failed for %s: %s", chembl_id, exc)

            evidence = ChEMBLEvidence(
                has_data=True,
                max_phase=max_phase,
                mechanism_of_action=mechanism_of_action,
                activity_count=activity_count,
                target_count=target_count,
                molecule_type=molecule_type,
            )
            log.debug(
                "ChEMBL: %s (%s) → phase %d, %d activities, %d targets",
                drug_name, chembl_id, max_phase, activity_count, target_count,
            )
            return evidence

    except (httpx.HTTPError, Exception) as exc:
        log.warning("ChEMBL request failed for %s: %s", drug_name, exc)
        return ChEMBLEvidence()
