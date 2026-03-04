"""Async PubMed co-occurrence evidence collector.

Queries PubMed E-utilities (esearch) to count articles mentioning both a
drug and an indication in their title/abstract.  Uses ``httpx`` for async
HTTP and computes a log-scaled cooccurrence score in [0, 1].
"""

from __future__ import annotations

import logging
import math

import httpx

from rule_engine.schema import LiteratureEvidence

log = logging.getLogger(__name__)

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


async def fetch_literature(
    drug_name: str,
    indication: str,
    timeout: int = 10,
) -> LiteratureEvidence:
    """Fetch PubMed co-occurrence evidence for a drug-indication pair.

    Args:
        drug_name: Generic drug name.
        indication: Disease / indication term.
        timeout: HTTP request timeout in seconds.

    Returns:
        Populated :class:`LiteratureEvidence`; zero defaults on any failure.
    """
    query = f'"{drug_name}"[Title/Abstract] AND "{indication}"[Title/Abstract]'

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                PUBMED_ESEARCH_URL,
                params={
                    "db": "pubmed",
                    "term": query,
                    "rettype": "count",
                    "retmode": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, Exception) as exc:
        log.warning("PubMed request failed for %s + %s: %s", drug_name, indication, exc)
        return LiteratureEvidence()

    try:
        count = int(data.get("esearchresult", {}).get("count", 0))
    except (TypeError, ValueError):
        count = 0

    if count > 0:
        cooccurrence_score = min(math.log10(count + 1) / 3.0, 1.0)
    else:
        cooccurrence_score = 0.0

    log.debug(
        "PubMed: %s + %s → %d articles (score %.3f)",
        drug_name, indication, count, cooccurrence_score,
    )
    return LiteratureEvidence(
        cooccurrence_score=cooccurrence_score,
        article_count=count,
    )
