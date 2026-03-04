"""Async evidence collector for the NLM MeSH API.

Resolves disease terms to MeSH descriptors and retrieves the disease
hierarchy (tree numbers, parent/child terms, related concepts, and
applicable qualifiers) for improved disease context.
"""

from __future__ import annotations

import asyncio
import logging

import requests

from rule_engine.schema import MeSHEvidence

log = logging.getLogger(__name__)

MESH_LOOKUP = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
MESH_RESOURCE = "https://id.nlm.nih.gov/mesh"


def _extract_label(value: object) -> str:
    """Extract a plain string from a MeSH JSON-LD label value.

    Labels can be plain strings or language-tagged dicts like
    ``{'@language': 'en', '@value': 'Skin Neoplasms'}``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("@value", value.get("label", str(value)))
    return str(value)


def _resolve_label(session: requests.Session, mesh_id: str, timeout: int) -> str:
    """Resolve a MeSH ID (e.g., D012878 or Q000208) to its human-readable label."""
    try:
        resp = session.get(f"{MESH_RESOURCE}/{mesh_id}.json", timeout=timeout)
        if resp.ok:
            return _extract_label(resp.json().get("label", mesh_id))
    except Exception:
        pass
    return mesh_id


def _fetch_mesh_sync(indication: str, timeout: int) -> MeSHEvidence:
    """Synchronous MeSH lookup — runs inside ``asyncio.to_thread``."""
    session = requests.Session()
    session.headers["Accept"] = "application/json"
    session.headers["User-Agent"] = "RuleEngine/1.0"

    # Step 1 — search for descriptor matching the indication
    try:
        resp = session.get(
            MESH_LOOKUP,
            params={"label": indication, "match": "contains", "limit": 5},
            timeout=timeout,
        )
        resp.raise_for_status()
        results = resp.json()
    except Exception as exc:
        log.debug("MeSH lookup failed for %s: %s", indication, exc)
        return MeSHEvidence()

    if not results:
        return MeSHEvidence()

    # Pick the best match — prefer exact match, then first result
    best = results[0]
    for r in results:
        label = r.get("label", "")
        if label.lower() == indication.lower():
            best = r
            break

    resource_uri = best.get("resource", "")
    mesh_label = best.get("label", "")
    # Extract MeSH ID from URI like "http://id.nlm.nih.gov/mesh/D008545"
    mesh_id = resource_uri.rsplit("/", 1)[-1] if resource_uri else None

    if not mesh_id:
        return MeSHEvidence()

    # Step 2 — fetch full descriptor JSON
    tree_numbers: list[str] = []
    parent_terms: list[str] = []
    child_terms: list[str] = []
    related_terms: list[str] = []
    qualifiers: list[str] = []

    try:
        desc_resp = session.get(
            f"{MESH_RESOURCE}/{mesh_id}.json",
            timeout=timeout,
        )
        if desc_resp.ok:
            data = desc_resp.json()

            # Tree numbers — values are URI strings like "http://id.nlm.nih.gov/mesh/C04.557..."
            for tn in data.get("treeNumber", []):
                if isinstance(tn, str):
                    tree_numbers.append(tn.rsplit("/", 1)[-1])
                elif isinstance(tn, dict):
                    tree_numbers.append(tn.get("@id", "").rsplit("/", 1)[-1])

            # Broader (parent) descriptors — URI strings
            for bd in data.get("broaderDescriptor", []):
                if isinstance(bd, str):
                    parent_mesh_id = bd.rsplit("/", 1)[-1]
                    label = _resolve_label(session, parent_mesh_id, timeout)
                    parent_terms.append(label)
                elif isinstance(bd, dict):
                    parent_terms.append(bd.get("label", bd.get("@id", "").rsplit("/", 1)[-1]))

            # Narrower (child) descriptors — resolve up to 10
            for nd in data.get("narrowerDescriptor", [])[:10]:
                if isinstance(nd, str):
                    child_mesh_id = nd.rsplit("/", 1)[-1]
                    label = _resolve_label(session, child_mesh_id, timeout)
                    child_terms.append(label)
                elif isinstance(nd, dict):
                    child_terms.append(nd.get("label", nd.get("@id", "").rsplit("/", 1)[-1]))

            # See-also / related concepts — resolve up to 5
            for sa in data.get("seeAlso", [])[:5]:
                if isinstance(sa, str):
                    related_mesh_id = sa.rsplit("/", 1)[-1]
                    label = _resolve_label(session, related_mesh_id, timeout)
                    related_terms.append(label)
                elif isinstance(sa, dict):
                    related_terms.append(sa.get("label", ""))

            # Allowable qualifiers — resolve up to 10, use IDs for rest
            raw_qualifiers = data.get("allowableQualifier", [])
            for aq in raw_qualifiers[:10]:
                if isinstance(aq, str):
                    q_id = aq.rsplit("/", 1)[-1]
                    label = _resolve_label(session, q_id, timeout)
                    qualifiers.append(label)
                elif isinstance(aq, dict):
                    qualifiers.append(aq.get("label", aq.get("@id", "").rsplit("/", 1)[-1]))
            # Append remaining qualifier IDs without resolution
            for aq in raw_qualifiers[10:]:
                if isinstance(aq, str):
                    qualifiers.append(aq.rsplit("/", 1)[-1])
    except Exception as exc:
        log.debug("MeSH descriptor fetch failed for %s: %s", mesh_id, exc)

    # If we didn't get parent labels from broaderDescriptor, derive from tree numbers
    if not parent_terms and tree_numbers:
        _resolve_parents_from_trees(session, tree_numbers, parent_terms, timeout)

    evidence = MeSHEvidence(
        found=True,
        disease_mesh_id=mesh_id,
        disease_mesh_name=mesh_label,
        tree_numbers=tree_numbers,
        parent_terms=parent_terms[:20],
        child_terms=child_terms[:20],
        related_terms=related_terms[:10],
        qualifiers=qualifiers[:20],
    )
    log.debug(
        "MeSH: %s → %s (%s), %d tree numbers, %d parents, %d children",
        indication, mesh_id, mesh_label,
        len(tree_numbers), len(parent_terms), len(child_terms),
    )
    return evidence


def _resolve_parents_from_trees(
    session: requests.Session,
    tree_numbers: list[str],
    parent_terms: list[str],
    timeout: int,
) -> None:
    """Derive parent descriptor labels by walking tree numbers upward."""
    seen: set[str] = set()
    for tn in tree_numbers[:3]:  # limit to avoid too many requests
        parts = tn.split(".")
        if len(parts) > 1:
            parent_tree = ".".join(parts[:-1])
            if parent_tree not in seen:
                seen.add(parent_tree)
                try:
                    resp = session.get(
                        f"{MESH_LOOKUP}",
                        params={"treeNumber": parent_tree, "limit": 1},
                        timeout=timeout,
                    )
                    if resp.ok:
                        results = resp.json()
                        if results:
                            label = results[0].get("label", parent_tree)
                            parent_terms.append(label)
                except Exception:
                    parent_terms.append(parent_tree)


async def fetch_mesh(
    indication: str,
    timeout: int = 10,
) -> MeSHEvidence:
    """Fetch MeSH disease hierarchy for *indication*.

    Args:
        indication: Disease / indication term to look up.
        timeout: HTTP request timeout in seconds.

    Returns:
        Populated MeSHEvidence; empty defaults on any failure.
    """
    try:
        return await asyncio.to_thread(_fetch_mesh_sync, indication, timeout)
    except Exception as exc:
        log.warning("MeSH request failed for %s: %s", indication, exc)
        return MeSHEvidence()
