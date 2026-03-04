"""Async evidence collector for ClinicalTrials.gov REST API v2.

Queries the public studies endpoint to gather trial metadata —
phases, eligibility criteria, endpoints, and enrollment sizes —
for a given drug-indication pair.
"""

from __future__ import annotations

import asyncio
import logging
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from rule_engine.schema import ClinicalTrialsEvidence

log = logging.getLogger(__name__)

CLINICALTRIALS_API = "https://clinicaltrials.gov/api/v2/studies"
CLINICALTRIALS_STUDY_API = "https://clinicaltrials.gov/api/v2/studies"

PHASE_MAP: dict[str, int] = {
    "EARLY_PHASE1": 1,
    "PHASE1": 1,
    "PHASE2": 2,
    "PHASE3": 3,
    "PHASE4": 4,
}


def _make_session() -> requests.Session:
    """Create a requests session with retry logic for transient failures.

    Retries 3 times with exponential backoff (1s, 2s, 4s) on connection
    errors, timeouts, and 5xx server errors — covers the SSL EOF errors
    from ClinicalTrials.gov.
    """
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,            # 1s, 2s, 4s between retries
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,        # let us handle status codes ourselves
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "RuleEngine/1.0 (clinical trial simulation; mailto:research@example.com)",
    })
    return session


def _parse_baseline(baseline_mod: dict, out: dict) -> None:
    """Extract age, sex, race distributions from baseline characteristics module."""
    for measure in baseline_mod.get("measures", []):
        title = (measure.get("title") or "").lower()
        classes = measure.get("classes", [])
        if not classes:
            continue
        categories = classes[0].get("categories", [])

        if "age" in title and "year" in title.lower():
            # Continuous age measure — look for mean/median
            for cat in categories:
                for m in cat.get("measurements", []):
                    if m.get("value"):
                        out.setdefault("age_mean", m["value"])
                        if m.get("spread"):
                            out.setdefault("age_std", m["spread"])
                        break
        elif title in ("sex: female, male", "sex", "gender"):
            for cat in categories:
                label = (cat.get("title") or "").lower()
                for m in cat.get("measurements", []):
                    val = m.get("value", "")
                    if val and label:
                        out.setdefault(f"sex_{label}", val)
        elif "race" in title or "ethnicity" in title:
            race_groups = {}
            for cat in categories:
                label = cat.get("title", "")
                for m in cat.get("measurements", []):
                    val = m.get("value", "")
                    if val and label:
                        race_groups[label] = val
            if race_groups:
                out.setdefault("race", race_groups)
        elif "ecog" in title or "performance" in title and "status" in title:
            ecog_dist = {}
            for cat in categories:
                label = (cat.get("title") or "").strip()
                for m in cat.get("measurements", []):
                    val = m.get("value", "")
                    if val and label:
                        ecog_dist[label] = val
            if ecog_dist:
                out.setdefault("ecog_ps", ecog_dist)


def _parse_result_aes(ae_mod: dict, out: list[dict]) -> None:
    """Extract AE terms with counts from the adverseEventsModule.

    Builds a serious-event lookup first so that grade 3-4 percentages
    (approximated from seriousEvents affected/at_risk) can be annotated
    onto every matching AE entry.
    """
    # First pass: collect serious events into a lookup
    serious_lookup: dict[str, dict] = {}  # term_lower -> {affected, at_risk, pct}
    for event_group in ae_mod.get("seriousEvents", []):
        term = event_group.get("term", "")
        if not term:
            continue
        stats = event_group.get("stats", [])
        for stat in stats:
            affected = stat.get("numAffected")
            at_risk = stat.get("numAtRisk")
            if affected is not None and at_risk and int(at_risk) > 0:
                serious_lookup[term.lower()] = {
                    "affected": int(affected),
                    "at_risk": int(at_risk),
                    "pct": round(int(affected) / int(at_risk) * 100, 1),
                }
                break

    # Second pass: process all events, annotating with grade34 data
    for event_type in ("otherEvents", "seriousEvents"):
        for event_group in ae_mod.get(event_type, []):
            term = event_group.get("term", "")
            if not term:
                continue
            stats = event_group.get("stats", [])
            for stat in stats:
                affected = stat.get("numAffected")
                at_risk = stat.get("numAtRisk")
                if affected is not None and at_risk and int(at_risk) > 0:
                    pct = round(int(affected) / int(at_risk) * 100, 1)
                    entry = {
                        "term": term,
                        "affected": int(affected),
                        "at_risk": int(at_risk),
                        "pct": pct,
                        "serious": event_type == "seriousEvents",
                    }
                    # Add grade34_pct from serious_lookup if available
                    serious_data = serious_lookup.get(term.lower())
                    if serious_data:
                        entry["grade34_pct"] = serious_data["pct"]
                    out.append(entry)
                    break  # one stat group per term is enough
    # Sort by frequency descending and limit
    out.sort(key=lambda x: x.get("pct", 0), reverse=True)
    del out[50:]  # keep top 50


_EFFICACY_PATTERNS = [
    re.compile(r"overall\s*response\s*rate|objective\s*response|\bORR\b", re.I),
    re.compile(r"complete\s*(?:response|remission)|\bCR\b", re.I),
    re.compile(r"progression[\s-]*free\s*survival|\bPFS\b", re.I),
    re.compile(r"overall\s*survival|\bOS\b", re.I),
]


def _parse_outcome_measures(outcome_mod: dict, out: list[dict]) -> None:
    """Extract outcome measures from outcomeMeasuresModule.

    PRIMARY outcomes are always included. SECONDARY outcomes are included
    only when their title matches known efficacy patterns (ORR, CR, PFS, OS),
    since ORR is almost always reported as a secondary outcome on CT.gov.
    """
    for measure in outcome_mod.get("outcomeMeasures", []):
        measure_type = measure.get("type", "").upper()
        if measure_type == "PRIMARY":
            pass  # always include
        elif measure_type == "SECONDARY":
            title = measure.get("title", "")
            if not any(p.search(title) for p in _EFFICACY_PATTERNS):
                continue
        else:
            continue
        title = measure.get("title", "")
        unit = measure.get("unitOfMeasure", "")
        description = measure.get("description", "")
        # Get the first group's value
        for group in measure.get("classes", []):
            for cat in group.get("categories", []):
                for m in cat.get("measurements", []):
                    value = m.get("value", "")
                    if value:
                        out.append({
                            "measure": title,
                            "value": value,
                            "unit": unit,
                            "description": description[:200],
                        })
                        break
                if out and out[-1].get("measure") == title:
                    break
            if out and out[-1].get("measure") == title:
                break
    del out[10:]  # limit


async def fetch_clinical_trials(
    drug_name: str,
    indication: str,
    timeout: int = 15,
) -> ClinicalTrialsEvidence:
    """Fetch clinical trial evidence for a drug-indication pair.

    Searches ClinicalTrials.gov v2 for studies matching both the drug name
    and the indication, then extracts phase, eligibility, endpoint, and
    enrollment information from each study's protocolSection.

    Uses requests (sync) via asyncio.to_thread because ClinicalTrials.gov
    WAF blocks the httpx library. Session includes retry logic for
    transient SSL/connection failures.

    Args:
        drug_name: Generic drug name to search for.
        indication: Disease / indication term.
        timeout: HTTP request timeout in seconds.

    Returns:
        Populated ClinicalTrialsEvidence; empty defaults on any failure.
    """

    def _fetch_sync() -> ClinicalTrialsEvidence:
        session = _make_session()
        headers = session.headers

        resp = session.get(
            CLINICALTRIALS_API,
            params={
                "query.term": drug_name,
                "query.cond": indication,
                "pageSize": 50,
                "format": "json",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        studies = data.get("studies", [])
        if not studies:
            return ClinicalTrialsEvidence()

        max_phase = 0
        primary_endpoints: list[str] = []
        sample_sizes: list[int] = []
        age_range: str | None = None
        sex_eligibility: str | None = None
        raw_studies: list[dict] = []

        for study in studies:
            protocol = study.get("protocolSection", {})
            design = protocol.get("designModule", {})
            eligibility = protocol.get("eligibilityModule", {})
            outcomes = protocol.get("outcomesModule", {})
            identification = protocol.get("identificationModule", {})

            # --- Phase ---
            for phase_str in design.get("phases", []):
                max_phase = max(max_phase, PHASE_MAP.get(phase_str, 0))

            # --- Eligibility ---
            if age_range is None:
                min_age = eligibility.get("minimumAge")
                max_age = eligibility.get("maximumAge")
                if min_age and max_age:
                    age_range = f"{min_age}-{max_age}"

            if sex_eligibility is None:
                sex = eligibility.get("sex")
                if sex:
                    sex_eligibility = sex

            # --- Primary endpoints ---
            for outcome in outcomes.get("primaryOutcomes", []):
                measure = outcome.get("measure")
                if measure and measure not in primary_endpoints:
                    primary_endpoints.append(measure)

            # --- Enrollment ---
            enrollment = design.get("enrollmentInfo", {}).get("count")
            if enrollment is not None:
                sample_sizes.append(int(enrollment))

            # --- Raw summary (up to 10) ---
            if len(raw_studies) < 10:
                raw_studies.append({
                    "nctId": identification.get("nctId"),
                    "briefTitle": identification.get("briefTitle"),
                    "phase": design.get("phases", []),
                    "enrollment": enrollment,
                })

        # --- Fetch results for top trials with posted results ---
        has_results = False
        baseline_demographics: dict = {}
        reported_aes: list[dict] = []
        primary_outcomes: list[dict] = []

        # Prefer trials with results AND meaningful enrollment (n >= 30)
        results_candidates = []
        for s in studies:
            if not s.get("hasResults", False):
                continue
            nct = s.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            enroll = s.get("protocolSection", {}).get("designModule", {}).get("enrollmentInfo", {}).get("count", 0)
            if nct:
                results_candidates.append((nct, int(enroll or 0)))
        # Sort by enrollment descending — larger trials give more representative data
        results_candidates.sort(key=lambda x: x[1], reverse=True)
        results_ncts = [nct for nct, _ in results_candidates[:5]]

        # Build a lookup of study metadata from the search results
        _study_meta: dict[str, dict] = {}
        for s in studies:
            proto = s.get("protocolSection", {})
            nid = proto.get("identificationModule", {}).get("nctId")
            if nid:
                phases = proto.get("designModule", {}).get("phases", [])
                _study_meta[nid] = {
                    "title": proto.get("identificationModule", {}).get("briefTitle", ""),
                    "phase": phases[0] if phases else "",
                }

        # Collect ALL qualifying demographics candidates, then pick best
        _demographics_candidates: list[dict] = []

        for nct_id in results_ncts:
            if not nct_id:
                continue
            try:
                r = session.get(
                    f"{CLINICALTRIALS_STUDY_API}/{nct_id}",
                    params={"fields": "resultsSection", "format": "json"},
                    headers=headers,
                    timeout=timeout,
                )
                if not r.ok:
                    continue
                results_section = r.json().get("resultsSection", {})
                if not results_section:
                    continue
                has_results = True

                # Baseline demographics — collect from every study with n >= 30
                baseline_mod = results_section.get("baselineCharacteristicsModule", {})
                denoms = baseline_mod.get("denoms", [])
                study_n = 0
                for d in denoms:
                    for c in d.get("counts", []):
                        try:
                            study_n = max(study_n, int(c.get("value", 0)))
                        except (ValueError, TypeError):
                            pass
                if baseline_mod and study_n >= 30:
                    candidate: dict = {}
                    _parse_baseline(baseline_mod, candidate)
                    candidate["_sample_size"] = study_n
                    candidate["_source_nct_id"] = nct_id
                    meta = _study_meta.get(nct_id, {})
                    candidate["_source_phase"] = meta.get("phase", "")
                    candidate["_source_title"] = meta.get("title", "")
                    _demographics_candidates.append(candidate)

                # Adverse events — merge across trials (max per term)
                ae_mod = results_section.get("adverseEventsModule", {})
                if ae_mod:
                    _trial_aes: list[dict] = []
                    _parse_result_aes(ae_mod, _trial_aes)
                    # Merge: keep the higher frequency for each AE term
                    _existing = {a["term"].lower(): a for a in reported_aes}
                    for ae in _trial_aes:
                        key = ae["term"].lower()
                        if key not in _existing or ae.get("pct", 0) > _existing[key].get("pct", 0):
                            _existing[key] = ae
                    reported_aes.clear()
                    reported_aes.extend(_existing.values())
                    reported_aes.sort(key=lambda x: x.get("pct", 0), reverse=True)

                # Primary outcomes — merge across trials
                outcome_mod = results_section.get("outcomeMeasuresModule", {})
                if outcome_mod:
                    _trial_outcomes: list[dict] = []
                    _parse_outcome_measures(outcome_mod, _trial_outcomes)
                    _existing_measures = {o["measure"] for o in primary_outcomes}
                    for o in _trial_outcomes:
                        if o["measure"] not in _existing_measures:
                            primary_outcomes.append(o)

            except Exception as exc:
                log.debug("Failed to fetch results for %s: %s", nct_id, exc)

        # Fallback: if best candidate is small, search specifically for Phase 3
        # trials with results — landmark trials often don't appear in the first
        # 50 generic search results for widely-studied drugs.
        best_n = max((c.get("_sample_size", 0) for c in _demographics_candidates), default=0)
        if best_n < 100:
            try:
                ph3_resp = session.get(
                    CLINICALTRIALS_API,
                    params={
                        "query.term": drug_name,
                        "query.cond": indication,
                        "filter.advanced": (
                            "AREA[ResultsFirstPostDate]RANGE[MIN,MAX] "
                            "AND AREA[Phase](PHASE3)"
                        ),
                        "pageSize": 5,
                        "format": "json",
                    },
                    headers=headers,
                    timeout=timeout,
                )
                if ph3_resp.ok:
                    ph3_studies = ph3_resp.json().get("studies", [])
                    for s in ph3_studies:
                        proto = s.get("protocolSection", {})
                        nid = proto.get("identificationModule", {}).get("nctId")
                        if not nid or nid in {c.get("_source_nct_id") for c in _demographics_candidates}:
                            continue
                        try:
                            r = session.get(
                                f"{CLINICALTRIALS_STUDY_API}/{nid}",
                                params={"fields": "resultsSection", "format": "json"},
                                headers=headers,
                                timeout=timeout,
                            )
                            if not r.ok:
                                continue
                            rs = r.json().get("resultsSection", {})
                            bl_mod = rs.get("baselineCharacteristicsModule", {})
                            dn = 0
                            for d in bl_mod.get("denoms", []):
                                for c in d.get("counts", []):
                                    try:
                                        dn = max(dn, int(c.get("value", 0)))
                                    except (ValueError, TypeError):
                                        pass
                            if bl_mod and dn >= 30:
                                cand: dict = {}
                                _parse_baseline(bl_mod, cand)
                                cand["_sample_size"] = dn
                                cand["_source_nct_id"] = nid
                                phases = proto.get("designModule", {}).get("phases", [])
                                cand["_source_phase"] = phases[0] if phases else ""
                                cand["_source_title"] = proto.get(
                                    "identificationModule", {}
                                ).get("briefTitle", "")
                                _demographics_candidates.append(cand)
                                has_results = True
                        except Exception:
                            pass
            except Exception as exc:
                log.debug("Phase 3 fallback search failed: %s", exc)

        # Pick demographics from the trial with the largest actual baseline N
        if _demographics_candidates:
            _demographics_candidates.sort(
                key=lambda c: c.get("_sample_size", 0), reverse=True,
            )
            baseline_demographics = _demographics_candidates[0]

        evidence = ClinicalTrialsEvidence(
            trial_count=len(studies),
            max_phase=max_phase,
            age_range=age_range,
            sex_eligibility=sex_eligibility,
            primary_endpoints=primary_endpoints,
            sample_sizes=sample_sizes,
            raw_studies=raw_studies,
            has_results=has_results,
            baseline_demographics=baseline_demographics,
            reported_aes=reported_aes,
            primary_outcomes=primary_outcomes,
        )
        log.debug(
            "ClinicalTrials: %s + %s → %d trials, max phase %d, has_results=%s",
            drug_name, indication, len(studies), max_phase, has_results,
        )
        return evidence

    try:
        return await asyncio.to_thread(_fetch_sync)
    except Exception as exc:
        log.warning("ClinicalTrials.gov request failed for %s + %s: %s", drug_name, indication, exc)
        return ClinicalTrialsEvidence()


async def fetch_clinical_trials_combo(
    drugs: list[str],
    indication: str,
    timeout: int = 15,
) -> ClinicalTrialsEvidence:
    """Search for combination trial data.

    Query: 'drug1 AND drug2 AND indication'. Uses the same parsing logic
    as fetch_clinical_trials but with a multi-drug query string. Session
    includes retry logic for transient SSL/connection failures.
    """
    if len(drugs) < 2:
        return ClinicalTrialsEvidence()

    combo_term = " AND ".join(drugs)

    def _fetch_combo_sync() -> ClinicalTrialsEvidence:
        session = _make_session()
        headers = session.headers

        resp = session.get(
            CLINICALTRIALS_API,
            params={
                "query.term": combo_term,
                "query.cond": indication,
                "pageSize": 50,
                "format": "json",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        studies = data.get("studies", [])
        if not studies:
            return ClinicalTrialsEvidence()

        max_phase = 0
        primary_endpoints: list[str] = []
        sample_sizes: list[int] = []
        age_range: str | None = None
        sex_eligibility: str | None = None
        raw_studies: list[dict] = []

        for study in studies:
            protocol = study.get("protocolSection", {})
            design = protocol.get("designModule", {})
            eligibility = protocol.get("eligibilityModule", {})
            outcomes = protocol.get("outcomesModule", {})
            identification = protocol.get("identificationModule", {})

            for phase_str in design.get("phases", []):
                max_phase = max(max_phase, PHASE_MAP.get(phase_str, 0))

            if age_range is None:
                min_age = eligibility.get("minimumAge")
                max_age = eligibility.get("maximumAge")
                if min_age and max_age:
                    age_range = f"{min_age}-{max_age}"

            if sex_eligibility is None:
                sex = eligibility.get("sex")
                if sex:
                    sex_eligibility = sex

            for outcome in outcomes.get("primaryOutcomes", []):
                measure = outcome.get("measure")
                if measure and measure not in primary_endpoints:
                    primary_endpoints.append(measure)

            enrollment = design.get("enrollmentInfo", {}).get("count")
            if enrollment is not None:
                sample_sizes.append(int(enrollment))

            if len(raw_studies) < 10:
                raw_studies.append({
                    "nctId": identification.get("nctId"),
                    "briefTitle": identification.get("briefTitle"),
                    "phase": design.get("phases", []),
                    "enrollment": enrollment,
                })

        # Fetch results for top trials with posted results
        has_results = False
        baseline_demographics: dict = {}
        reported_aes: list[dict] = []
        primary_outcomes: list[dict] = []

        results_candidates = []
        for s in studies:
            if not s.get("hasResults", False):
                continue
            nct = s.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
            enroll = s.get("protocolSection", {}).get("designModule", {}).get("enrollmentInfo", {}).get("count", 0)
            if nct:
                results_candidates.append((nct, int(enroll or 0)))
        results_candidates.sort(key=lambda x: x[1], reverse=True)
        results_ncts = [nct for nct, _ in results_candidates[:5]]

        # Build study metadata lookup
        _study_meta: dict[str, dict] = {}
        for s in studies:
            proto = s.get("protocolSection", {})
            nid = proto.get("identificationModule", {}).get("nctId")
            if nid:
                phases = proto.get("designModule", {}).get("phases", [])
                _study_meta[nid] = {
                    "title": proto.get("identificationModule", {}).get("briefTitle", ""),
                    "phase": phases[0] if phases else "",
                }

        # Collect demographics candidates
        _demographics_candidates: list[dict] = []

        for nct_id in results_ncts:
            if not nct_id:
                continue
            try:
                r = session.get(
                    f"{CLINICALTRIALS_STUDY_API}/{nct_id}",
                    params={"fields": "resultsSection", "format": "json"},
                    headers=headers,
                    timeout=timeout,
                )
                if not r.ok:
                    continue
                results_section = r.json().get("resultsSection", {})
                if not results_section:
                    continue
                has_results = True

                # Baseline demographics — collect from every study with n >= 30
                baseline_mod = results_section.get("baselineCharacteristicsModule", {})
                denoms = baseline_mod.get("denoms", [])
                study_n = 0
                for d in denoms:
                    for c in d.get("counts", []):
                        try:
                            study_n = max(study_n, int(c.get("value", 0)))
                        except (ValueError, TypeError):
                            pass
                if baseline_mod and study_n >= 30:
                    candidate: dict = {}
                    _parse_baseline(baseline_mod, candidate)
                    candidate["_sample_size"] = study_n
                    candidate["_source_nct_id"] = nct_id
                    meta = _study_meta.get(nct_id, {})
                    candidate["_source_phase"] = meta.get("phase", "")
                    candidate["_source_title"] = meta.get("title", "")
                    _demographics_candidates.append(candidate)

                # Adverse events — merge across trials (max per term)
                ae_mod = results_section.get("adverseEventsModule", {})
                if ae_mod:
                    _trial_aes2: list[dict] = []
                    _parse_result_aes(ae_mod, _trial_aes2)
                    _existing2 = {a["term"].lower(): a for a in reported_aes}
                    for ae in _trial_aes2:
                        key = ae["term"].lower()
                        if key not in _existing2 or ae.get("pct", 0) > _existing2[key].get("pct", 0):
                            _existing2[key] = ae
                    reported_aes.clear()
                    reported_aes.extend(_existing2.values())
                    reported_aes.sort(key=lambda x: x.get("pct", 0), reverse=True)

                # Primary outcomes — merge across trials
                outcome_mod = results_section.get("outcomeMeasuresModule", {})
                if outcome_mod:
                    _trial_outcomes2: list[dict] = []
                    _parse_outcome_measures(outcome_mod, _trial_outcomes2)
                    _existing_m2 = {o["measure"] for o in primary_outcomes}
                    for o in _trial_outcomes2:
                        if o["measure"] not in _existing_m2:
                            primary_outcomes.append(o)

            except Exception as exc:
                log.debug("Failed to fetch combo results for %s: %s", nct_id, exc)

        # Phase 3 fallback for combo — same pattern as main search
        best_n = max((c.get("_sample_size", 0) for c in _demographics_candidates), default=0)
        if best_n < 100:
            try:
                ph3_resp = session.get(
                    CLINICALTRIALS_API,
                    params={
                        "query.term": combo_term,
                        "query.cond": indication,
                        "filter.advanced": (
                            "AREA[ResultsFirstPostDate]RANGE[MIN,MAX] "
                            "AND AREA[Phase](PHASE3)"
                        ),
                        "pageSize": 10,
                        "format": "json",
                    },
                    headers=headers,
                    timeout=timeout,
                )
                if ph3_resp.ok:
                    ph3_studies = ph3_resp.json().get("studies", [])
                    for s in ph3_studies:
                        proto = s.get("protocolSection", {})
                        nid = proto.get("identificationModule", {}).get("nctId")
                        if not nid or nid in {c.get("_source_nct_id") for c in _demographics_candidates}:
                            continue
                        try:
                            r = session.get(
                                f"{CLINICALTRIALS_STUDY_API}/{nid}",
                                params={"fields": "resultsSection", "format": "json"},
                                headers=headers,
                                timeout=timeout,
                            )
                            if not r.ok:
                                continue
                            rs = r.json().get("resultsSection", {})
                            bl_mod = rs.get("baselineCharacteristicsModule", {})
                            dn = 0
                            for d in bl_mod.get("denoms", []):
                                for c in d.get("counts", []):
                                    try:
                                        dn = max(dn, int(c.get("value", 0)))
                                    except (ValueError, TypeError):
                                        pass
                            if bl_mod and dn >= 30:
                                cand: dict = {}
                                _parse_baseline(bl_mod, cand)
                                cand["_sample_size"] = dn
                                cand["_source_nct_id"] = nid
                                phases = proto.get("designModule", {}).get("phases", [])
                                cand["_source_phase"] = phases[0] if phases else ""
                                cand["_source_title"] = proto.get(
                                    "identificationModule", {}
                                ).get("briefTitle", "")
                                _demographics_candidates.append(cand)
                                has_results = True

                                # Also grab AEs/outcomes from Phase 3 if we don't have any
                                if not reported_aes:
                                    ae_mod = rs.get("adverseEventsModule", {})
                                    if ae_mod:
                                        _parse_result_aes(ae_mod, reported_aes)
                                if not primary_outcomes:
                                    outcome_mod = rs.get("outcomeMeasuresModule", {})
                                    if outcome_mod:
                                        _parse_outcome_measures(outcome_mod, primary_outcomes)
                        except Exception:
                            pass
            except Exception as exc:
                log.debug("Combo Phase 3 fallback search failed: %s", exc)

        # Pick demographics from the trial with the largest actual baseline N
        if _demographics_candidates:
            _demographics_candidates.sort(
                key=lambda c: c.get("_sample_size", 0), reverse=True,
            )
            baseline_demographics = _demographics_candidates[0]

        evidence = ClinicalTrialsEvidence(
            trial_count=len(studies),
            max_phase=max_phase,
            age_range=age_range,
            sex_eligibility=sex_eligibility,
            primary_endpoints=primary_endpoints,
            sample_sizes=sample_sizes,
            raw_studies=raw_studies,
            has_results=has_results,
            baseline_demographics=baseline_demographics,
            reported_aes=reported_aes,
            primary_outcomes=primary_outcomes,
        )
        log.debug(
            "ClinicalTrials combo: %s + %s → %d trials, max phase %d",
            " + ".join(drugs), indication, len(studies), max_phase,
        )
        return evidence

    try:
        return await asyncio.to_thread(_fetch_combo_sync)
    except Exception as exc:
        log.warning("ClinicalTrials.gov combo request failed for %s + %s: %s",
                     " + ".join(drugs), indication, exc)
        return ClinicalTrialsEvidence()
