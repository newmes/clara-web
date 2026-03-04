"""Pydantic schemas for the Rule Discovery Pipeline.

Defines:
- RuleSet: The output schema — all fields parameterized for downstream patient/trial sampling.
- EvidenceBundle: Structured evidence from all sources, passed as context to the synthesis agent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# RuleSet output schema
# ---------------------------------------------------------------------------

class AgeDistribution(BaseModel):
    min: int | None = Field(None, description="Minimum eligible age (years)")
    max: int | None = Field(None, description="Maximum eligible age (years)")
    mean: float | None = Field(None, description="Expected mean age of enrolled population")
    std: float | None = Field(None, description="Standard deviation of age distribution")


class SexDistribution(BaseModel):
    pct_male: float | None = Field(None, ge=0, le=100, description="Percentage male")
    pct_female: float | None = Field(None, ge=0, le=100, description="Percentage female")


class RaceEthnicityGroup(BaseModel):
    group: str = Field(..., description="Race/ethnicity group name")
    pct: float = Field(..., ge=0, le=100, description="Percentage of trial population")


class Demographics(BaseModel):
    age: AgeDistribution
    sex: SexDistribution
    race_ethnicity: list[RaceEthnicityGroup] = Field(default_factory=list)
    ecog_ps: dict[str, float] | None = Field(None, description="ECOG PS distribution from trial baseline")


class AERiskModifier(BaseModel):
    ae: str = Field(..., description="Must reference an AE in adverse_events")
    risk_multiplier: float = Field(..., ge=1.0, description="1.0 = no change")


class Comorbidity(BaseModel):
    condition: str = Field(..., description="Comorbidity condition name")
    prevalence_pct: float = Field(..., ge=0, le=100, description="Prevalence in trial population (%)")
    impacts_dosing: bool = Field(False, description="Whether this comorbidity affects dosing decisions")
    ae_risk_modifiers: list[AERiskModifier] = Field(default_factory=list)


class AETrigger(BaseModel):
    target_ae: str = Field(..., description="AE name or 'Dose reduction' / 'Treatment discontinuation'")
    condition: str = Field(..., description="e.g. 'grade >= 3', 'any'")
    probability_pct: float = Field(..., ge=0, le=100)


class AdverseEvent(BaseModel):
    event: str = Field(..., description="Adverse event term (MedDRA preferred term)")
    frequency_pct: float = Field(..., ge=0, le=100, description="Frequency in treated population (%)")
    severity_distribution: dict[str, float] = Field(
        ..., description="Grade distribution e.g. {'grade_1': 25.0, 'grade_2': 15.0, ...}"
    )
    median_onset_days: int = Field(..., ge=0, description="Median onset from treatment start (days)")
    reversible: bool = Field(True, description="Whether the AE is typically reversible on treatment hold")
    source_drug: str | None = Field(None, description="For combo: which drug is primarily responsible")
    triggers: list[AETrigger] = Field(default_factory=list)


class Regimen(BaseModel):
    drug: str = Field(..., description="Drug name")
    dose: str = Field(..., description="Dose with units, e.g. '200 mg'")
    route: str = Field(..., description="Route of administration, e.g. 'IV'")
    cycle_days: int = Field(..., description="Cycle length in days")
    schedule: str = Field(..., description="e.g. 'D1, D8, D15'")


class Endpoint(BaseModel):
    endpoint: str = Field(..., description="Clinical endpoint name")
    expected_value: float = Field(..., description="Expected value for this endpoint")
    unit: str = Field(..., description="Unit of measurement")


class Efficacy(BaseModel):
    overall_response_rate_pct: float | None = Field(None, ge=0, le=100, description="Overall response rate (%)")
    complete_response_rate_pct: float | None = Field(None, ge=0, le=100, description="Complete response rate (%)")
    median_pfs_months: float | None = Field(None, ge=0, description="Median progression-free survival (months)")
    median_pfs_ci_low: float | None = Field(None, ge=0, description="PFS 95% CI lower bound (months)")
    median_pfs_ci_high: float | None = Field(None, ge=0, description="PFS 95% CI upper bound (months)")
    median_os_months: float | None = Field(None, ge=0, description="Median overall survival (months)")
    median_os_ci_low: float | None = Field(None, ge=0, description="OS 95% CI lower bound (months)")
    median_os_ci_high: float | None = Field(None, ge=0, description="OS 95% CI upper bound (months)")
    endpoints: list[Endpoint] = Field(default_factory=list, description="Additional clinical endpoints")


class DrugInteraction(BaseModel):
    """Drug-drug interaction between drugs from different regimens."""

    drug_a: str = Field(..., description="First drug name")
    drug_b: str = Field(..., description="Second drug name")
    interaction_type: str = Field(..., description="E.g. 'pharmacokinetic', 'pharmacodynamic', 'synergistic'")
    description: str = Field("", description="Clinical description of the interaction")
    severity: str = Field("moderate", description="mild / moderate / severe")
    ae_impact: list[str] = Field(default_factory=list, description="AE names affected by this interaction")
    frequency_modifier: float = Field(1.0, ge=1.0, le=3.0, description="Multiplier for affected AE frequencies")
    monitoring_recommendation: str = Field("", description="Recommended monitoring actions")
    drugbank_relation: str = Field("", description="DrugBank interaction relation type")


class IndicationEfficacy(BaseModel):
    """Efficacy data for one indication within a multi-indication rule set."""

    indication: str = Field(..., description="Indication name")
    regimen_drugs: list[str] = Field(..., description="Drugs used for this indication")
    efficacy: Efficacy
    phase: int = Field(3, ge=1, le=4, description="Clinical trial phase for this indication")
    treatment_duration_days: int = Field(365, ge=1, description="Treatment duration for this indication")


class OverlappingAENote(BaseModel):
    """Documents how overlapping AEs from multiple regimens were merged."""

    event: str = Field(..., description="AE event name")
    contributing_drugs: list[str] = Field(..., description="Drugs that can cause this AE")
    unadjusted_frequency_sum: float = Field(..., description="Naive sum of individual frequencies")
    adjusted_frequency_pct: float = Field(..., description="Final adjusted frequency using probabilistic model")
    rationale: str = Field("", description="Explanation of frequency adjustment")


class RuleSet(BaseModel):
    """Complete clinical trial simulation rule set for a drug-indication pair (or combo)."""

    drugs: list[str] = Field(..., description="Drug name(s) (generic)")
    indication: str = Field(..., description="Target indication / disease")
    phase: int = Field(..., ge=1, le=4, description="Clinical trial phase (1-4)")
    treatment_duration_days: int = Field(..., ge=1, description="Total treatment duration in days")
    regimen: list[Regimen] = Field(..., description="Per-drug regimen details")

    demographics: Demographics
    comorbidities: list[Comorbidity] = Field(default_factory=list)
    adverse_events: list[AdverseEvent] = Field(default_factory=list)
    efficacy: Efficacy

    # Multi-indication fields (all optional — backward compatible with single-indication rule sets)
    is_multi_indication: bool = Field(False, description="Whether this is a multi-indication unified rule set")
    indications: list[str] = Field(default_factory=list, description="All indications (multi-indication mode)")
    per_indication_efficacy: list[IndicationEfficacy] = Field(
        default_factory=list, description="Per-indication efficacy data"
    )
    drug_interactions: list[DrugInteraction] = Field(
        default_factory=list, description="Cross-regimen drug-drug interactions"
    )
    overlapping_ae_notes: list[OverlappingAENote] = Field(
        default_factory=list, description="Notes on how overlapping AEs were merged"
    )
    source_rule_sets: list[str] = Field(
        default_factory=list, description="Filenames of individual rule sets that were merged"
    )


# ---------------------------------------------------------------------------
# Evidence bundle — input to the synthesis agent
# ---------------------------------------------------------------------------

class ClinicalTrialsEvidence(BaseModel):
    trial_count: int = 0
    max_phase: int = 0
    age_range: str | None = None
    sex_eligibility: str | None = None
    primary_endpoints: list[str] = Field(default_factory=list)
    sample_sizes: list[int] = Field(default_factory=list)
    raw_studies: list[dict] = Field(default_factory=list, description="Up to 10 raw study summaries")
    # Results data from completed trials
    has_results: bool = False
    baseline_demographics: dict = Field(default_factory=dict)
    reported_aes: list[dict] = Field(default_factory=list)
    primary_outcomes: list[dict] = Field(default_factory=list)


class OpenFDAEvidence(BaseModel):
    top_adverse_events: list[dict] = Field(default_factory=list, description="Top AEs with report counts")
    total_ae_reports: int = 0
    label_indications: list[str] = Field(default_factory=list)
    label_warnings: list[str] = Field(default_factory=list)
    label_dosage: str | None = None
    time_to_onset_data: list[dict] = Field(default_factory=list, description="Time-to-onset data for top AEs from FAERS")
    has_timing_data: bool = False


class ChEMBLEvidence(BaseModel):
    has_data: bool = False
    max_phase: int = 0
    mechanism_of_action: str | None = None
    activity_count: int = 0
    target_count: int = 0
    molecule_type: str | None = None


class DrugBankEvidence(BaseModel):
    found: bool = False
    drugbank_id: str | None = None
    moa: str | None = None
    targets: list[dict] = Field(default_factory=list)
    ddi_count: int = 0


class PrimeKGEvidence(BaseModel):
    found: bool = False
    disease_associations: list[dict] = Field(default_factory=list)
    gene_targets: list[dict] = Field(default_factory=list)
    neighbor_summary: str | None = None


class LiteratureEvidence(BaseModel):
    cooccurrence_score: float = 0.0
    article_count: int = 0


class PubChemEvidence(BaseModel):
    found: bool = False
    pubchem_cid: int | None = None
    molecular_weight: float | None = None
    logp: float | None = None
    tpsa: float | None = None
    hbd_count: int = 0
    hba_count: int = 0
    rotatable_bonds: int = 0
    lipinski_violations: int = 0
    bioassay_active_count: int = 0
    bioassay_total_count: int = 0
    pharmacological_class: str | None = None


class OnSIDESEvidence(BaseModel):
    found: bool = False
    drug_concept_name: str | None = None
    ae_pairs: list[dict] = Field(default_factory=list, description="Validated drug-ADE pairs with label counts and prediction scores")
    boxed_warning_aes: list[str] = Field(default_factory=list, description="AEs with FDA boxed warning")
    total_pairs: int = 0


class DailyMedEvidence(BaseModel):
    found: bool = False
    set_id: str | None = None
    drug_name_label: str | None = None
    adverse_reactions_text: str | None = None
    ae_table: list[dict] = Field(default_factory=list)
    boxed_warning: str | None = None
    contraindications: str | None = None
    dosage_text: str | None = None
    special_populations: str | None = None


class MeSHEvidence(BaseModel):
    found: bool = False
    disease_mesh_id: str | None = None
    disease_mesh_name: str | None = None
    tree_numbers: list[str] = Field(default_factory=list)
    parent_terms: list[str] = Field(default_factory=list)
    child_terms: list[str] = Field(default_factory=list)
    related_terms: list[str] = Field(default_factory=list)
    qualifiers: list[str] = Field(default_factory=list)


class SingleDrugEvidence(BaseModel):
    """Evidence collected for one drug."""
    dailymed: DailyMedEvidence = Field(default_factory=DailyMedEvidence)
    openfda: OpenFDAEvidence = Field(default_factory=OpenFDAEvidence)
    chembl: ChEMBLEvidence = Field(default_factory=ChEMBLEvidence)
    drugbank: DrugBankEvidence = Field(default_factory=DrugBankEvidence)
    pubchem: PubChemEvidence = Field(default_factory=PubChemEvidence)
    onsides: OnSIDESEvidence = Field(default_factory=OnSIDESEvidence)


class PDSAEAggregate(BaseModel):
    """Aggregated AE data from Project Data Sphere patient-level records."""
    term: str
    n_patients_with_event: int = 0
    n_total_patients: int = 0
    frequency_pct: float = 0.0
    grade_distribution: dict[str, float] = Field(default_factory=dict)
    median_onset_day: float | None = None
    median_duration_days: float | None = None


class PDSDemographics(BaseModel):
    """Demographics aggregated from PDS patient-level data."""
    n_patients: int = 0
    age_mean: float | None = None
    age_std: float | None = None
    age_min: int | None = None
    age_max: int | None = None
    pct_male: float | None = None
    pct_female: float | None = None
    race_distribution: dict[str, float] = Field(default_factory=dict)
    ecog_distribution: dict[str, float] | None = None


class PDSEfficacy(BaseModel):
    """Efficacy endpoints aggregated from PDS patient-level data."""
    overall_response_rate_pct: float | None = None
    complete_response_rate_pct: float | None = None
    median_pfs_months: float | None = None
    median_os_months: float | None = None


class PDSRegimen(BaseModel):
    """Per-drug regimen data aggregated from PDS exposure records."""
    drug: str
    median_dose: float | None = None
    dose_unit: str | None = None
    route: str | None = None
    n_patients: int = 0


class PDSTrialMatch(BaseModel):
    """Matched PDS trial metadata."""
    trial_id: str
    drugs: list[str] = Field(default_factory=list)
    indication: str | None = None
    n_patients: int = 0
    match_score: float = 0.0


class PDSEvidence(BaseModel):
    """Evidence aggregated from Project Data Sphere patient-level clinical trial data."""
    found: bool = False
    matched_trial: PDSTrialMatch | None = None
    demographics: PDSDemographics | None = None
    ae_aggregates: list[PDSAEAggregate] = Field(default_factory=list)
    efficacy: PDSEfficacy | None = None
    regimen: list[PDSRegimen] = Field(default_factory=list)
    safety_population_n: int = 0


class EvidenceBundle(BaseModel):
    """All evidence gathered for a drug-indication pair (or combo), passed as context to the agent."""

    drugs: list[str]
    indication: str

    per_drug: dict[str, SingleDrugEvidence] = Field(default_factory=dict)

    # Shared (indication-level) evidence
    clinical_trials: ClinicalTrialsEvidence = Field(default_factory=ClinicalTrialsEvidence)
    combo_trials: ClinicalTrialsEvidence = Field(default_factory=ClinicalTrialsEvidence)
    primekg: PrimeKGEvidence = Field(default_factory=PrimeKGEvidence)
    literature: LiteratureEvidence = Field(default_factory=LiteratureEvidence)
    mesh: MeSHEvidence = Field(default_factory=MeSHEvidence)
    pds: PDSEvidence = Field(default_factory=PDSEvidence)
