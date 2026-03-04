"""Async evidence collector for OpenFDA APIs.

Queries OpenFDA endpoints in parallel:
  1. FAERS adverse-event reports — top reactions and total report count.
  2. Drug labels — indications, warnings, and dosage text.
  3. FAERS time-to-onset — drug interval dosage timing data for top AEs.

Uses a fallback search strategy for biologics that may not match on
generic_name alone: generic_name → brand_name → substance_name → medicinalproduct.

Results are combined into a single OpenFDAEvidence model.
"""

from __future__ import annotations

import logging

import httpx

from rule_engine.schema import OpenFDAEvidence

log = logging.getLogger(__name__)

FAERS_URL = "https://api.fda.gov/drug/event.json"
LABEL_URL = "https://api.fda.gov/drug/label.json"

# Ordered fallback search fields for FAERS adverse-event queries.
# Each tuple is (search_field_template, description) where the template
# takes a single .format(drug_name=...) substitution.
_FAERS_SEARCH_FIELDS: list[tuple[str, str]] = [
    ('patient.drug.openfda.generic_name:"{drug_name}"', "generic_name"),
    ('patient.drug.openfda.brand_name:"{drug_name}"', "brand_name"),
    ('patient.drug.openfda.substance_name:"{drug_name}"', "substance_name"),
    ('patient.drug.medicinalproduct:"{drug_name}"', "medicinalproduct"),
]

# Ordered fallback search fields for drug label queries.
_LABEL_SEARCH_FIELDS: list[tuple[str, str]] = [
    ('openfda.generic_name:"{drug_name}"', "generic_name"),
    ('openfda.brand_name:"{drug_name}"', "brand_name"),
    ('openfda.substance_name:"{drug_name}"', "substance_name"),
]


async def _faers_query_with_fallback(
    client: httpx.AsyncClient,
    drug_name: str,
    extra_params: dict[str, str | int],
) -> dict | None:
    """Try FAERS search fields in order, return first non-empty JSON response."""
    for search_template, field_name in _FAERS_SEARCH_FIELDS:
        search_value = search_template.format(drug_name=drug_name)
        params = {"search": search_value, **extra_params}
        try:
            resp = await client.get(FAERS_URL, params=params)
            if resp.status_code == 404:
                log.debug("FAERS fallback %s: 404 for %s", field_name, drug_name)
                continue
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                log.debug("FAERS matched on %s for %s (%d results)", field_name, drug_name, len(results))
                return data
            log.debug("FAERS fallback %s: empty results for %s", field_name, drug_name)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                log.debug("FAERS fallback %s: 404 for %s", field_name, drug_name)
                continue
            log.warning("FAERS fallback %s failed for %s: %s", field_name, drug_name, exc)
        except httpx.HTTPError as exc:
            log.warning("FAERS fallback %s failed for %s: %s", field_name, drug_name, exc)
    return None


async def _label_query_with_fallback(
    client: httpx.AsyncClient,
    drug_name: str,
) -> dict | None:
    """Try label search fields in order, return first non-empty JSON response."""
    for search_template, field_name in _LABEL_SEARCH_FIELDS:
        search_value = search_template.format(drug_name=drug_name)
        params = {"search": search_value, "limit": 1}
        try:
            resp = await client.get(LABEL_URL, params=params)
            if resp.status_code == 404:
                log.debug("Label fallback %s: 404 for %s", field_name, drug_name)
                continue
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                log.debug("Label matched on %s for %s", field_name, drug_name)
                return data
            log.debug("Label fallback %s: empty results for %s", field_name, drug_name)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                log.debug("Label fallback %s: 404 for %s", field_name, drug_name)
                continue
            log.warning("Label fallback %s failed for %s: %s", field_name, drug_name, exc)
        except httpx.HTTPError as exc:
            log.warning("Label fallback %s failed for %s: %s", field_name, drug_name, exc)
    return None


async def _fetch_onset_timing(
    client: httpx.AsyncClient,
    drug_name: str,
) -> tuple[list[dict], bool]:
    """Fetch time-to-onset data from FAERS using drug interval dosage fields.

    Returns (time_to_onset_data, has_timing_data).
    """
    # Try to get onset timing by counting drug interval unit numbers.
    # This field captures the duration a patient took the drug before the
    # event was reported, giving a proxy for time-to-onset.
    for search_template, field_name in _FAERS_SEARCH_FIELDS:
        search_value = search_template.format(drug_name=drug_name)
        params = {
            "search": search_value,
            "count": "patient.drug.drugintervaldosageunitnumb",
        }
        try:
            resp = await client.get(FAERS_URL, params=params)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                # Map unit codes to labels.
                # FDA unit codes: 800=years, 801=months, 802=weeks, 803=days, 804=hours
                unit_labels = {
                    "800": "years", "801": "months", "802": "weeks",
                    "803": "days", "804": "hours",
                }
                onset_data = []
                total_reports = 0
                for entry in results:
                    term = str(entry.get("term", ""))
                    count = entry.get("count", 0)
                    onset_data.append({
                        "unit_code": term,
                        "unit_label": unit_labels.get(term, f"code_{term}"),
                        "count": count,
                    })
                    total_reports += count
                log.debug(
                    "FAERS onset timing matched on %s for %s: %d units, %d total reports",
                    field_name, drug_name, len(onset_data), total_reports,
                )
                return onset_data, bool(onset_data)
        except (httpx.HTTPError, Exception) as exc:
            log.debug("FAERS onset timing %s failed for %s: %s", field_name, drug_name, exc)
            continue
    return [], False


async def fetch_openfda_aes(
    drug_name: str,
    timeout: int = 30,
) -> OpenFDAEvidence:
    """Fetch adverse-event and label evidence from OpenFDA.

    Makes three parallel sub-queries — FAERS adverse events, drug labeling,
    and FAERS time-to-onset — then merges the results into one OpenFDAEvidence
    model.  Each sub-query uses a fallback search strategy to improve hit
    rates for biologics.

    Args:
        drug_name: Drug name (generic, brand, or substance).
        timeout: HTTP request timeout in seconds.

    Returns:
        Populated OpenFDAEvidence; empty defaults on any failure.
    """
    top_adverse_events: list[dict] = []
    total_ae_reports: int = 0
    label_indications: list[str] = []
    label_warnings: list[str] = []
    label_dosage: str | None = None
    time_to_onset_data: list[dict] = []
    has_timing_data: bool = False

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Fire all three sub-queries concurrently.
            # Each uses its own internal fallback chain.
            import asyncio

            ae_task = asyncio.create_task(
                _faers_query_with_fallback(
                    client, drug_name,
                    extra_params={"count": "patient.reaction.reactionmeddrapt.exact"},
                )
            )
            label_task = asyncio.create_task(
                _label_query_with_fallback(client, drug_name)
            )
            onset_task = asyncio.create_task(
                _fetch_onset_timing(client, drug_name)
            )

            ae_data, label_data, onset_result = await asyncio.gather(
                ae_task, label_task, onset_task,
            )

            # --- FAERS adverse events ---
            if ae_data is not None:
                for entry in ae_data.get("results", []):
                    top_adverse_events.append({
                        "term": entry.get("term"),
                        "count": entry.get("count", 0),
                    })
                    total_ae_reports += entry.get("count", 0)
                log.debug(
                    "OpenFDA FAERS: %s → %d AE terms, %d total reports",
                    drug_name, len(top_adverse_events), total_ae_reports,
                )
            else:
                log.warning("OpenFDA FAERS: no data found for %s after all fallbacks", drug_name)

            # --- Drug labels ---
            if label_data is not None:
                results = label_data.get("results", [])
                if results:
                    label = results[0]

                    raw_indications = label.get("indications_and_usage", [])
                    if raw_indications:
                        label_indications = [text[:200] for text in raw_indications]

                    raw_warnings = label.get("warnings_and_cautions") or label.get("warnings", [])
                    if raw_warnings:
                        label_warnings = [text[:200] for text in raw_warnings]

                    raw_dosage = label.get("dosage_and_administration", [])
                    if raw_dosage:
                        label_dosage = raw_dosage[0][:300]

                log.debug(
                    "OpenFDA Labels: %s → %d indications, %d warnings",
                    drug_name, len(label_indications), len(label_warnings),
                )
            else:
                log.warning("OpenFDA Labels: no data found for %s after all fallbacks", drug_name)

            # --- Time-to-onset ---
            time_to_onset_data, has_timing_data = onset_result

    except Exception as exc:
        log.warning("OpenFDA client setup failed for %s: %s", drug_name, exc)
        return OpenFDAEvidence()

    return OpenFDAEvidence(
        top_adverse_events=top_adverse_events,
        total_ae_reports=total_ae_reports,
        label_indications=label_indications,
        label_warnings=label_warnings,
        label_dosage=label_dosage,
        time_to_onset_data=time_to_onset_data,
        has_timing_data=has_timing_data,
    )
