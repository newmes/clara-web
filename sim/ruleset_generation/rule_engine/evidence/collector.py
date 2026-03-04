"""Parallel evidence orchestrator.

Runs all evidence collectors concurrently via ``asyncio.gather`` and
assembles the results into an :class:`EvidenceBundle`.  Any individual
failure is caught and replaced with the default (empty) evidence model.

Supports multi-drug (combo) collection: per-drug sources run in parallel
for each drug, shared indication-level sources (ClinicalTrials, PrimeKG,
PubMed) are searched for ALL drugs and merged, and combo-specific searches
run only when len(drugs) > 1.
"""

from __future__ import annotations

import asyncio
import logging

from rule_engine.config import RuleEngineConfig
from rule_engine.evidence.chembl import fetch_chembl
from rule_engine.evidence.clinical_trials import fetch_clinical_trials, fetch_clinical_trials_combo
from rule_engine.evidence.dailymed import fetch_dailymed
from rule_engine.evidence.literature import fetch_literature
from rule_engine.evidence.local_dbs import fetch_drugbank, fetch_primekg
from rule_engine.evidence.mesh import fetch_mesh
from rule_engine.evidence.onsides import fetch_onsides
from rule_engine.evidence.openfda import fetch_openfda_aes
from rule_engine.evidence.projectdatasphere import fetch_pds
from rule_engine.evidence.pubchem import fetch_pubchem
from rule_engine.schema import (
    ChEMBLEvidence,
    ClinicalTrialsEvidence,
    DailyMedEvidence,
    DrugBankEvidence,
    EvidenceBundle,
    LiteratureEvidence,
    MeSHEvidence,
    OnSIDESEvidence,
    OpenFDAEvidence,
    PDSEvidence,
    PrimeKGEvidence,
    PubChemEvidence,
    SingleDrugEvidence,
)

log = logging.getLogger(__name__)

# Per-drug collector slots: (field_name, default_factory)
_PER_DRUG_SLOTS: list[tuple[str, type]] = [
    ("dailymed", DailyMedEvidence),
    ("openfda", OpenFDAEvidence),
    ("chembl", ChEMBLEvidence),
    ("drugbank", DrugBankEvidence),
    ("pubchem", PubChemEvidence),
    ("onsides", OnSIDESEvidence),
]


async def _collect_single_drug(
    drug: str,
    config: RuleEngineConfig,
) -> SingleDrugEvidence:
    """Collect all per-drug evidence sources in parallel for one drug."""
    timeout = config.evidence_timeout

    results = await asyncio.gather(
        fetch_dailymed(drug, timeout),
        fetch_openfda_aes(drug, timeout),
        fetch_chembl(drug, timeout),
        fetch_drugbank(drug, config),
        fetch_pubchem(drug, timeout),
        fetch_onsides(drug, config),
        return_exceptions=True,
    )

    kwargs: dict[str, object] = {}
    for idx, (field, default_cls) in enumerate(_PER_DRUG_SLOTS):
        value = results[idx]
        if isinstance(value, BaseException):
            log.warning("Per-drug collector %s failed for %s: %s", field, drug, value)
            kwargs[field] = default_cls()
        else:
            kwargs[field] = value

    return SingleDrugEvidence(**kwargs)


def _merge_clinical_trials(results: list[ClinicalTrialsEvidence]) -> ClinicalTrialsEvidence:
    """Merge ClinicalTrialsEvidence from multiple drug searches.

    Picks the richest demographics (largest sample) and unions the rest.
    """
    valid = [r for r in results if r.trial_count > 0]
    if not valid:
        return ClinicalTrialsEvidence()
    if len(valid) == 1:
        return valid[0]

    # Pick best demographics — largest _sample_size
    best = max(valid, key=lambda r: r.baseline_demographics.get("_sample_size", 0))
    # Union trial counts, endpoints, AEs
    all_endpoints = []
    all_raw_studies = []
    all_sample_sizes = []
    all_reported_aes = best.reported_aes  # start with best trial's AEs
    all_primary_outcomes = best.primary_outcomes
    seen_ncts: set[str] = set()

    for r in valid:
        for ep in r.primary_endpoints:
            if ep not in all_endpoints:
                all_endpoints.append(ep)
        all_sample_sizes.extend(r.sample_sizes)
        for s in r.raw_studies:
            nct = s.get("nctId")
            if nct and nct not in seen_ncts:
                seen_ncts.add(nct)
                all_raw_studies.append(s)

    return ClinicalTrialsEvidence(
        trial_count=sum(r.trial_count for r in valid),
        max_phase=max(r.max_phase for r in valid),
        age_range=best.age_range,
        sex_eligibility=best.sex_eligibility,
        primary_endpoints=all_endpoints[:20],
        sample_sizes=all_sample_sizes,
        raw_studies=all_raw_studies[:10],
        has_results=best.has_results,
        baseline_demographics=best.baseline_demographics,
        reported_aes=all_reported_aes,
        primary_outcomes=all_primary_outcomes,
    )


def _merge_primekg(results: list[PrimeKGEvidence]) -> PrimeKGEvidence:
    """Merge PrimeKG results from multiple drug searches."""
    valid = [r for r in results if r.found]
    if not valid:
        return PrimeKGEvidence()
    if len(valid) == 1:
        return valid[0]
    # Combine: union disease/gene associations, concatenate summaries
    all_diseases: list[dict] = []
    all_genes: list[dict] = []
    summaries: list[str] = []
    seen_diseases: set[str] = set()
    seen_genes: set[str] = set()
    for r in valid:
        for d in r.disease_associations:
            name = d.get("name", "")
            if name not in seen_diseases:
                seen_diseases.add(name)
                all_diseases.append(d)
        for g in r.gene_targets:
            name = g.get("name", "")
            if name not in seen_genes:
                seen_genes.add(name)
                all_genes.append(g)
        if r.neighbor_summary:
            summaries.append(r.neighbor_summary)
    return PrimeKGEvidence(
        found=True,
        disease_associations=all_diseases[:20],
        gene_targets=all_genes[:20],
        neighbor_summary=" | ".join(summaries) if summaries else None,
    )


def _merge_literature(results: list[LiteratureEvidence]) -> LiteratureEvidence:
    """Merge literature results — take best score, sum articles."""
    valid = [r for r in results if r.article_count > 0]
    if not valid:
        return LiteratureEvidence()
    return LiteratureEvidence(
        cooccurrence_score=max(r.cooccurrence_score for r in valid),
        article_count=sum(r.article_count for r in valid),
    )


async def collect_evidence(
    drugs: list[str],
    indication: str,
    config: RuleEngineConfig | None = None,
) -> EvidenceBundle:
    """Gather evidence from all sources in parallel.

    Per-drug evidence (DailyMed, OpenFDA, ChEMBL, DrugBank, PubChem) is
    collected in parallel for each drug. Shared indication-level evidence
    (ClinicalTrials.gov, PrimeKG, PubMed) is searched for ALL drugs and
    merged. MeSH is searched once (indication-only). Combo trials are
    searched only when len(drugs) > 1.

    Args:
        drugs: List of generic drug names (1 or more).
        indication: Disease / indication term.
        config: Optional pipeline config; a default is created if *None*.

    Returns:
        Fully populated :class:`EvidenceBundle`.
    """
    if config is None:
        config = RuleEngineConfig()

    timeout = config.evidence_timeout

    # Build all tasks — per-drug + shared-per-drug + mesh + combo
    per_drug_tasks = [_collect_single_drug(drug, config) for drug in drugs]

    # Shared evidence — run for EVERY drug, then merge
    ct_tasks = [fetch_clinical_trials(drug, indication, timeout) for drug in drugs]
    kg_tasks = [fetch_primekg(drug, indication, config) for drug in drugs]
    lit_tasks = [fetch_literature(drug, indication, timeout) for drug in drugs]
    mesh_task = fetch_mesh(indication, timeout)

    combo_task = (
        fetch_clinical_trials_combo(drugs, indication, timeout)
        if len(drugs) > 1
        else _noop_clinical_trials()
    )

    pds_task = fetch_pds(drugs, indication, config)

    n_drugs = len(drugs)
    # Layout: [per_drug * N] [ct * N] [kg * N] [lit * N] [mesh] [combo] [pds]
    all_results = await asyncio.gather(
        *per_drug_tasks,
        *ct_tasks,
        *kg_tasks,
        *lit_tasks,
        mesh_task,
        combo_task,
        pds_task,
        return_exceptions=True,
    )

    # --- Unpack per-drug results ---
    per_drug: dict[str, SingleDrugEvidence] = {}
    for i, drug in enumerate(drugs):
        value = all_results[i]
        if isinstance(value, BaseException):
            log.warning("Per-drug collection failed for %s: %s", drug, value)
            per_drug[drug] = SingleDrugEvidence()
        else:
            per_drug[drug] = value

    # --- Unpack and merge shared results ---
    offset = n_drugs

    # Clinical trials — one per drug, merge
    ct_results: list[ClinicalTrialsEvidence] = []
    for j in range(n_drugs):
        val = all_results[offset + j]
        if isinstance(val, BaseException):
            log.warning("ClinicalTrials failed for drug %s: %s", drugs[j], val)
            ct_results.append(ClinicalTrialsEvidence())
        else:
            ct_results.append(val)
    clinical_trials = _merge_clinical_trials(ct_results)
    offset += n_drugs

    # PrimeKG — one per drug, merge
    kg_results: list[PrimeKGEvidence] = []
    for j in range(n_drugs):
        val = all_results[offset + j]
        if isinstance(val, BaseException):
            log.warning("PrimeKG failed for drug %s: %s", drugs[j], val)
            kg_results.append(PrimeKGEvidence())
        else:
            kg_results.append(val)
    primekg = _merge_primekg(kg_results)
    offset += n_drugs

    # Literature — one per drug, merge
    lit_results: list[LiteratureEvidence] = []
    for j in range(n_drugs):
        val = all_results[offset + j]
        if isinstance(val, BaseException):
            log.warning("Literature failed for drug %s: %s", drugs[j], val)
            lit_results.append(LiteratureEvidence())
        else:
            lit_results.append(val)
    literature = _merge_literature(lit_results)
    offset += n_drugs

    # MeSH — single (indication-level)
    mesh_val = all_results[offset]
    mesh = mesh_val if not isinstance(mesh_val, BaseException) else MeSHEvidence()
    if isinstance(mesh_val, BaseException):
        log.warning("MeSH failed: %s", mesh_val)
    offset += 1

    # Combo trials
    combo_val = all_results[offset]
    combo_trials = combo_val if not isinstance(combo_val, BaseException) else ClinicalTrialsEvidence()
    if isinstance(combo_val, BaseException):
        log.warning("Combo trials search failed: %s", combo_val)
    offset += 1

    # Project Data Sphere
    pds_val = all_results[offset]
    pds = pds_val if not isinstance(pds_val, BaseException) else PDSEvidence()
    if isinstance(pds_val, BaseException):
        log.warning("PDS evidence fetch failed: %s", pds_val)

    bundle = EvidenceBundle(
        drugs=drugs,
        indication=indication,
        per_drug=per_drug,
        clinical_trials=clinical_trials,
        combo_trials=combo_trials,
        primekg=primekg,
        literature=literature,
        mesh=mesh,
        pds=pds,
    )

    # Log summary
    drug_summaries = []
    for drug in drugs:
        sd = per_drug.get(drug, SingleDrugEvidence())
        drug_summaries.append(
            f"{drug}(dailymed={sd.dailymed.found}, fda_aes={sd.openfda.total_ae_reports}, "
            f"chembl={sd.chembl.has_data}, drugbank={sd.drugbank.found}, pubchem={sd.pubchem.found}, "
            f"onsides={sd.onsides.found})"
        )
    pds_tag = (
        f"pds={bundle.pds.matched_trial.trial_id}(n={bundle.pds.safety_population_n})"
        if bundle.pds.found and bundle.pds.matched_trial
        else "pds=none"
    )
    log.info(
        "Evidence collected for %s + %s: per_drug=[%s], trials=%d, "
        "combo_trials=%d, primekg=%s, literature=%d, mesh=%s, %s",
        " + ".join(drugs),
        indication,
        ", ".join(drug_summaries),
        bundle.clinical_trials.trial_count,
        bundle.combo_trials.trial_count,
        bundle.primekg.found,
        bundle.literature.article_count,
        bundle.mesh.found,
        pds_tag,
    )
    return bundle


async def _noop_clinical_trials() -> ClinicalTrialsEvidence:
    """Return empty evidence for single-drug (no combo search needed)."""
    return ClinicalTrialsEvidence()
