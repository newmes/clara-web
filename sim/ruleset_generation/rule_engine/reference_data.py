"""Static reference data for target schema conversion.

Contains template defaults for fields that don't exist in the current RuleSet
output but are required by the target schema (lab ranges, disease baselines,
supportive care, mortality/ECOG/disposition models, AE duration defaults).
"""

# ---------------------------------------------------------------------------
# Lab reference ranges — standard oncology panel, same for all drugs
# ---------------------------------------------------------------------------
LAB_REFERENCE_RANGES = {
    "hemoglobin": {
        "unit": "g/dL",
        "LLN": 11.5,
        "ULN": 15.8,
        "normal_range": {"min": 11.5, "max": 15.8},
    },
    "wbc": {
        "unit": "x10^9/L",
        "LLN": 4.0,
        "ULN": 11.0,
        "normal_range": {"min": 4.0, "max": 11.0},
    },
    "platelets": {
        "unit": "x10^9/L",
        "LLN": 150,
        "ULN": 400,
        "normal_range": {"min": 150, "max": 400},
    },
    "anc": {
        "unit": "x10^9/L",
        "LLN": 1.5,
        "ULN": 8.0,
        "normal_range": {"min": 1.5, "max": 8.0},
    },
    "creatinine": {
        "unit": "umol/L",
        "LLN": 53,
        "ULN": 106,
        "normal_range": {"min": 53, "max": 106},
    },
    "ast": {
        "unit": "U/L",
        "LLN": 9,
        "ULN": 34,
        "normal_range": {"min": 9, "max": 34},
    },
    "alt": {
        "unit": "U/L",
        "LLN": 6,
        "ULN": 34,
        "normal_range": {"min": 6, "max": 34},
    },
    "ldh": {
        "unit": "U/L",
        "LLN": 53,
        "ULN": 234,
        "normal_range": {"min": 53, "max": 234},
    },
    "albumin": {
        "unit": "g/L",
        "LLN": 33,
        "ULN": 49,
        "normal_range": {"min": 33, "max": 49},
    },
    "sodium": {
        "unit": "mmol/L",
        "LLN": 135,
        "ULN": 145,
        "normal_range": {"min": 135, "max": 145},
    },
    "potassium": {
        "unit": "mmol/L",
        "LLN": 3.5,
        "ULN": 5.0,
        "normal_range": {"min": 3.5, "max": 5.0},
    },
    "glucose": {
        "unit": "mmol/L",
        "LLN": 3.9,
        "ULN": 6.1,
        "normal_range": {"min": 3.9, "max": 6.1},
    },
    "total_bilirubin": {
        "unit": "umol/L",
        "LLN": 3.4,
        "ULN": 20.5,
        "normal_range": {"min": 3.4, "max": 20.5},
    },
    "calcium": {
        "unit": "mmol/L",
        "LLN": 2.1,
        "ULN": 2.6,
        "normal_range": {"min": 2.1, "max": 2.6},
    },
    "magnesium": {
        "unit": "mmol/L",
        "LLN": 0.7,
        "ULN": 1.05,
        "normal_range": {"min": 0.7, "max": 1.05},
    },
    "alkaline_phosphatase": {
        "unit": "U/L",
        "LLN": 30,
        "ULN": 120,
        "normal_range": {"min": 30, "max": 120},
    },
}

# ---------------------------------------------------------------------------
# Disease baselines — keyed by indication keyword (case-insensitive match)
# ---------------------------------------------------------------------------
_DEFAULT_BASELINE = {
    "tumor_sites": {"primary": 0.4, "lymph_node": 0.3, "lung": 0.15, "liver": 0.10, "bone": 0.05},
    "tumor_response_distribution": {"CR": 0.05, "PR": 0.25, "SD": 0.40, "PD": 0.30},
    "sum_of_diameters_mm": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 50, "std": 30, "min": 10, "max": 200}},
    "ldh_uln_ratio": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 1.0, "std": 0.4, "min": 0.5, "max": 3.0}},
    "albumin_gdl": {"type": "numeric", "distribution": "normal", "params": {"mean": 3.8, "std": 0.5, "min": 2.0, "max": 5.5}},
}

DISEASE_BASELINES = {
    "lung": {
        "tumor_sites": {"lung": 0.98, "lymph_nodes": 0.85, "liver": 0.40, "adrenal": 0.35, "bone": 0.30, "brain": 0.25},
        "tumor_response_distribution": {"CR": 0.03, "PR": 0.25, "SD": 0.35, "PD": 0.37},
        "sum_of_diameters_mm": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 150, "std": 90, "min": 15, "max": 500}},
        "ldh_uln_ratio": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 1.9, "std": 1.2, "min": 0.5, "max": 8.0}},
        "albumin_gdl": {"type": "numeric", "distribution": "normal", "params": {"mean": 3.8, "std": 0.5, "min": 2.0, "max": 5.5}},
    },
    "melanoma": {
        "tumor_sites": {"skin": 0.30, "lymph_node": 0.25, "lung": 0.20, "liver": 0.10, "brain": 0.10, "bone": 0.05},
        "tumor_response_distribution": {"CR": 0.10, "PR": 0.30, "SD": 0.30, "PD": 0.30},
        "sum_of_diameters_mm": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 45, "std": 30, "min": 10, "max": 180}},
        "ldh_uln_ratio": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 1.2, "std": 0.6, "min": 0.5, "max": 4.0}},
        "albumin_gdl": {"type": "numeric", "distribution": "normal", "params": {"mean": 3.9, "std": 0.5, "min": 2.0, "max": 5.5}},
    },
    "breast": {
        "tumor_sites": {"breast": 0.25, "lymph_node": 0.25, "bone": 0.20, "lung": 0.15, "liver": 0.10, "brain": 0.05},
        "tumor_response_distribution": {"CR": 0.08, "PR": 0.30, "SD": 0.35, "PD": 0.27},
        "sum_of_diameters_mm": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 40, "std": 25, "min": 10, "max": 160}},
        "ldh_uln_ratio": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 1.0, "std": 0.4, "min": 0.5, "max": 3.0}},
        "albumin_gdl": {"type": "numeric", "distribution": "normal", "params": {"mean": 3.8, "std": 0.4, "min": 2.5, "max": 5.5}},
    },
    "myeloma": {
        "tumor_sites": {"bone_marrow": 0.50, "bone": 0.30, "kidney": 0.10, "soft_tissue": 0.10},
        "tumor_response_distribution": {"CR": 0.15, "PR": 0.35, "SD": 0.30, "PD": 0.20},
        "sum_of_diameters_mm": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 30, "std": 20, "min": 5, "max": 100}},
        "ldh_uln_ratio": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 1.0, "std": 0.3, "min": 0.5, "max": 2.5}},
        "albumin_gdl": {"type": "numeric", "distribution": "normal", "params": {"mean": 3.5, "std": 0.6, "min": 1.5, "max": 5.0}},
    },
    "leukemia": {
        "tumor_sites": {"bone_marrow": 0.60, "blood": 0.25, "spleen": 0.10, "lymph_node": 0.05},
        "tumor_response_distribution": {"CR": 0.20, "PR": 0.30, "SD": 0.25, "PD": 0.25},
        "sum_of_diameters_mm": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 25, "std": 15, "min": 5, "max": 80}},
        "ldh_uln_ratio": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 1.3, "std": 0.6, "min": 0.5, "max": 4.0}},
        "albumin_gdl": {"type": "numeric", "distribution": "normal", "params": {"mean": 3.6, "std": 0.5, "min": 2.0, "max": 5.0}},
    },
    "lymphoma": {
        "tumor_sites": {"lymph_node": 0.45, "spleen": 0.20, "bone_marrow": 0.15, "liver": 0.10, "lung": 0.10},
        "tumor_response_distribution": {"CR": 0.25, "PR": 0.30, "SD": 0.25, "PD": 0.20},
        "sum_of_diameters_mm": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 50, "std": 30, "min": 10, "max": 200}},
        "ldh_uln_ratio": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 1.2, "std": 0.5, "min": 0.5, "max": 3.5}},
        "albumin_gdl": {"type": "numeric", "distribution": "normal", "params": {"mean": 3.7, "std": 0.5, "min": 2.0, "max": 5.5}},
    },
    "prostate": {
        "tumor_sites": {"prostate": 0.30, "bone": 0.30, "lymph_node": 0.20, "lung": 0.10, "liver": 0.10},
        "tumor_response_distribution": {"CR": 0.05, "PR": 0.20, "SD": 0.40, "PD": 0.35},
        "sum_of_diameters_mm": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 45, "std": 25, "min": 10, "max": 150}},
        "ldh_uln_ratio": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 1.0, "std": 0.3, "min": 0.5, "max": 2.5}},
        "albumin_gdl": {"type": "numeric", "distribution": "normal", "params": {"mean": 3.8, "std": 0.5, "min": 2.0, "max": 5.5}},
    },
    "hepatocellular": {
        "tumor_sites": {"liver": 0.50, "lung": 0.20, "bone": 0.10, "lymph_node": 0.10, "adrenal": 0.10},
        "tumor_response_distribution": {"CR": 0.02, "PR": 0.10, "SD": 0.45, "PD": 0.43},
        "sum_of_diameters_mm": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 60, "std": 40, "min": 10, "max": 250}},
        "ldh_uln_ratio": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 1.3, "std": 0.5, "min": 0.5, "max": 4.0}},
        "albumin_gdl": {"type": "numeric", "distribution": "normal", "params": {"mean": 3.3, "std": 0.6, "min": 1.5, "max": 5.0}},
    },
    "renal": {
        "tumor_sites": {"kidney": 0.30, "lung": 0.25, "bone": 0.15, "lymph_node": 0.15, "liver": 0.10, "brain": 0.05},
        "tumor_response_distribution": {"CR": 0.05, "PR": 0.25, "SD": 0.35, "PD": 0.35},
        "sum_of_diameters_mm": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 55, "std": 35, "min": 10, "max": 200}},
        "ldh_uln_ratio": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 1.1, "std": 0.4, "min": 0.5, "max": 3.0}},
        "albumin_gdl": {"type": "numeric", "distribution": "normal", "params": {"mean": 3.7, "std": 0.5, "min": 2.0, "max": 5.5}},
    },
    "urothelial": {
        "tumor_sites": {"bladder": 0.30, "lymph_node": 0.25, "lung": 0.15, "liver": 0.15, "bone": 0.10, "soft_tissue": 0.05},
        "tumor_response_distribution": {"CR": 0.08, "PR": 0.25, "SD": 0.30, "PD": 0.37},
        "sum_of_diameters_mm": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 50, "std": 30, "min": 10, "max": 180}},
        "ldh_uln_ratio": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 1.1, "std": 0.4, "min": 0.5, "max": 3.0}},
        "albumin_gdl": {"type": "numeric", "distribution": "normal", "params": {"mean": 3.6, "std": 0.5, "min": 2.0, "max": 5.5}},
    },
    "ovarian": {
        "tumor_sites": {"ovary": 0.25, "peritoneum": 0.25, "lymph_node": 0.20, "omentum": 0.15, "liver": 0.10, "lung": 0.05},
        "tumor_response_distribution": {"CR": 0.15, "PR": 0.30, "SD": 0.30, "PD": 0.25},
        "sum_of_diameters_mm": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 45, "std": 30, "min": 10, "max": 180}},
        "ldh_uln_ratio": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 1.0, "std": 0.4, "min": 0.5, "max": 3.0}},
        "albumin_gdl": {"type": "numeric", "distribution": "normal", "params": {"mean": 3.5, "std": 0.5, "min": 2.0, "max": 5.0}},
    },
    "pancreatic": {
        "tumor_sites": {"pancreas": 0.30, "liver": 0.30, "peritoneum": 0.15, "lung": 0.10, "lymph_node": 0.10, "bone": 0.05},
        "tumor_response_distribution": {"CR": 0.01, "PR": 0.10, "SD": 0.35, "PD": 0.54},
        "sum_of_diameters_mm": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 55, "std": 30, "min": 10, "max": 200}},
        "ldh_uln_ratio": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 1.2, "std": 0.5, "min": 0.5, "max": 3.5}},
        "albumin_gdl": {"type": "numeric", "distribution": "normal", "params": {"mean": 3.3, "std": 0.6, "min": 1.5, "max": 5.0}},
    },
    "glioblastoma": {
        "tumor_sites": {"brain": 0.90, "spinal_cord": 0.10},
        "tumor_response_distribution": {"CR": 0.02, "PR": 0.08, "SD": 0.40, "PD": 0.50},
        "sum_of_diameters_mm": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 40, "std": 20, "min": 10, "max": 120}},
        "ldh_uln_ratio": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 1.0, "std": 0.3, "min": 0.5, "max": 2.5}},
        "albumin_gdl": {"type": "numeric", "distribution": "normal", "params": {"mean": 3.8, "std": 0.4, "min": 2.5, "max": 5.5}},
    },
    "cholangiocarcinoma": {
        "tumor_sites": {"bile_duct": 0.30, "liver": 0.30, "lymph_node": 0.15, "peritoneum": 0.10, "lung": 0.10, "bone": 0.05},
        "tumor_response_distribution": {"CR": 0.02, "PR": 0.15, "SD": 0.40, "PD": 0.43},
        "sum_of_diameters_mm": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 50, "std": 30, "min": 10, "max": 180}},
        "ldh_uln_ratio": {"type": "numeric", "distribution": "lognormal", "params": {"mean": 1.2, "std": 0.5, "min": 0.5, "max": 3.5}},
        "albumin_gdl": {"type": "numeric", "distribution": "normal", "params": {"mean": 3.4, "std": 0.5, "min": 2.0, "max": 5.0}},
    },
}

# ---------------------------------------------------------------------------
# Supportive care — keyed by AE keyword (case-insensitive substring match)
# ---------------------------------------------------------------------------
SUPPORTIVE_CARE_MAP = {
    "nausea": [
        {"drug": "Ondansetron", "dose": "8", "unit": "mg", "route": "ORAL", "frequency": "Q8H PRN", "probability": 0.8},
        {"drug": "Dexamethasone", "dose": "8", "unit": "mg", "route": "ORAL", "frequency": "QD x3 days", "probability": 0.4},
    ],
    "vomiting": [
        {"drug": "Ondansetron", "dose": "8", "unit": "mg", "route": "ORAL", "frequency": "Q8H PRN", "probability": 0.8},
    ],
    "diarrhea": [
        {"drug": "Loperamide", "dose": "4", "unit": "mg", "route": "ORAL", "frequency": "initial then 2mg Q2H PRN", "probability": 0.7},
    ],
    "neutropenia": [
        {"drug": "Filgrastim (G-CSF)", "dose": "5", "unit": "mcg/kg", "route": "SUBCUTANEOUS", "frequency": "QD until ANC recovery", "probability": 0.6},
    ],
    "febrile neutropenia": [
        {"drug": "Filgrastim (G-CSF)", "dose": "5", "unit": "mcg/kg", "route": "SUBCUTANEOUS", "frequency": "QD", "probability": 0.9},
        {"drug": "Broad-spectrum antibiotics", "dose": "per protocol", "unit": "", "route": "INTRAVENOUS", "frequency": "Q8H", "probability": 0.95},
    ],
    "anemia": [
        {"drug": "Epoetin alfa", "dose": "40000", "unit": "units", "route": "SUBCUTANEOUS", "frequency": "weekly", "probability": 0.4},
        {"drug": "Iron supplementation", "dose": "325", "unit": "mg", "route": "ORAL", "frequency": "TID", "probability": 0.5},
    ],
    "thrombocytopenia": [
        {"drug": "Platelet transfusion", "dose": "1", "unit": "unit", "route": "INTRAVENOUS", "frequency": "PRN (plt < 10K)", "probability": 0.3},
    ],
    "pain": [
        {"drug": "Acetaminophen", "dose": "500", "unit": "mg", "route": "ORAL", "frequency": "Q6H PRN", "probability": 0.7},
        {"drug": "Opioid analgesic", "dose": "per protocol", "unit": "", "route": "ORAL", "frequency": "Q4-6H PRN", "probability": 0.3},
    ],
    "rash": [
        {"drug": "Topical corticosteroid", "dose": "apply", "unit": "thin layer", "route": "TOPICAL", "frequency": "BID", "probability": 0.6},
        {"drug": "Diphenhydramine", "dose": "25", "unit": "mg", "route": "ORAL", "frequency": "Q6H PRN", "probability": 0.4},
    ],
    "pruritus": [
        {"drug": "Hydroxyzine", "dose": "25", "unit": "mg", "route": "ORAL", "frequency": "Q6H PRN", "probability": 0.5},
    ],
    "mucositis": [
        {"drug": "Magic mouthwash", "dose": "15", "unit": "mL", "route": "ORAL", "frequency": "QID swish & spit", "probability": 0.6},
    ],
    "stomatitis": [
        {"drug": "Magic mouthwash", "dose": "15", "unit": "mL", "route": "ORAL", "frequency": "QID swish & spit", "probability": 0.6},
    ],
    "constipation": [
        {"drug": "Docusate", "dose": "100", "unit": "mg", "route": "ORAL", "frequency": "BID", "probability": 0.5},
        {"drug": "Senna", "dose": "8.6", "unit": "mg", "route": "ORAL", "frequency": "QHS PRN", "probability": 0.4},
    ],
    "hypertension": [
        {"drug": "Amlodipine", "dose": "5", "unit": "mg", "route": "ORAL", "frequency": "QD", "probability": 0.5},
    ],
    "fatigue": [
        {"drug": "Methylphenidate", "dose": "5", "unit": "mg", "route": "ORAL", "frequency": "BID PRN", "probability": 0.2},
    ],
    "peripheral neuropathy": [
        {"drug": "Gabapentin", "dose": "300", "unit": "mg", "route": "ORAL", "frequency": "TID", "probability": 0.4},
        {"drug": "Duloxetine", "dose": "60", "unit": "mg", "route": "ORAL", "frequency": "QD", "probability": 0.3},
    ],
    "hypersensitivity": [
        {"drug": "Diphenhydramine", "dose": "50", "unit": "mg", "route": "INTRAVENOUS", "frequency": "once (pre-medication)", "probability": 0.8},
        {"drug": "Dexamethasone", "dose": "10", "unit": "mg", "route": "INTRAVENOUS", "frequency": "once (pre-medication)", "probability": 0.7},
    ],
    "edema": [
        {"drug": "Furosemide", "dose": "20", "unit": "mg", "route": "ORAL", "frequency": "QD PRN", "probability": 0.3},
    ],
}

# ---------------------------------------------------------------------------
# Mortality models — keyed by indication keyword
# Simulator-compatible format: disease_progression (multipliers) + treatment_toxicity (AE grade multipliers)
# ---------------------------------------------------------------------------
def _mortality_channels(
    pd_multiplier: float = 3.5,
    response_lag_days: int = 28,
    response_reduction: float = 0.25,
    ae_g3_mult: float = 1.5,
    ae_g4_mult: float = 3.0,
    concurrent_threshold: int = 3,
    concurrent_mult: float = 1.75,
) -> dict:
    return {
        "disease_progression": {
            "pd_multiplier": pd_multiplier,
            "response_lag_days": response_lag_days,
            "response_reduction": response_reduction,
        },
        "treatment_toxicity": {
            "ae_grade_multipliers": {"3": ae_g3_mult, "4": ae_g4_mult},
            "concurrent_ae_threshold": concurrent_threshold,
            "concurrent_ae_multiplier": concurrent_mult,
        },
    }


_DEFAULT_MORTALITY = {
    "baseline_annual_mortality": 0.30,
    "channels": _mortality_channels(),
}

MORTALITY_MODELS = {
    "lung": {"baseline_annual_mortality": 0.45, "channels": _mortality_channels(pd_multiplier=4.0, response_reduction=0.20)},
    "melanoma": {"baseline_annual_mortality": 0.25, "channels": _mortality_channels(pd_multiplier=3.0, ae_g3_mult=1.8, ae_g4_mult=2.5)},
    "breast": {"baseline_annual_mortality": 0.20, "channels": _mortality_channels(pd_multiplier=3.0, response_reduction=0.30)},
    "myeloma": {"baseline_annual_mortality": 0.25, "channels": _mortality_channels(pd_multiplier=3.5, ae_g3_mult=1.8, ae_g4_mult=3.5)},
    "leukemia": {"baseline_annual_mortality": 0.35, "channels": _mortality_channels(pd_multiplier=4.0, ae_g3_mult=2.0, ae_g4_mult=4.0)},
    "lymphoma": {"baseline_annual_mortality": 0.20, "channels": _mortality_channels(pd_multiplier=3.0, response_reduction=0.35)},
    "prostate": {"baseline_annual_mortality": 0.20, "channels": _mortality_channels(pd_multiplier=2.5, response_reduction=0.30)},
    "hepatocellular": {"baseline_annual_mortality": 0.55, "channels": _mortality_channels(pd_multiplier=4.5, ae_g3_mult=1.8, ae_g4_mult=3.5)},
    "renal": {"baseline_annual_mortality": 0.30, "channels": _mortality_channels(pd_multiplier=3.5, ae_g3_mult=1.5, ae_g4_mult=2.5)},
    "urothelial": {"baseline_annual_mortality": 0.40, "channels": _mortality_channels(pd_multiplier=3.5, ae_g3_mult=1.8, ae_g4_mult=2.5)},
    "ovarian": {"baseline_annual_mortality": 0.30, "channels": _mortality_channels(pd_multiplier=3.5, response_reduction=0.25)},
    "pancreatic": {"baseline_annual_mortality": 0.70, "channels": _mortality_channels(pd_multiplier=5.0, response_reduction=0.15)},
    "glioblastoma": {"baseline_annual_mortality": 0.60, "channels": _mortality_channels(pd_multiplier=5.0, response_reduction=0.10)},
    "cholangiocarcinoma": {"baseline_annual_mortality": 0.55, "channels": _mortality_channels(pd_multiplier=4.5, response_reduction=0.15)},
}

# ---------------------------------------------------------------------------
# ECOG model defaults — same for all drugs/indications
# ---------------------------------------------------------------------------
ECOG_MODEL_DEFAULTS = {
    "ae_burden_weight": 0.05,
    "disease_weight": 0.07,
    "response_lag_days": 42,
    "response_benefit": -0.4,
    "comorbidity_penalty": 0.03,
}

# ---------------------------------------------------------------------------
# ECOG PS distribution — default oncology trial enrollment
# ---------------------------------------------------------------------------
ECOG_PS_DEFAULTS = {"0": 0.40, "1": 0.50, "2": 0.10}

# ---------------------------------------------------------------------------
# Disposition model defaults — simulator-compatible format
# ---------------------------------------------------------------------------
DISPOSITION_DEFAULTS = {
    "independent_hazards": {
        "consent_withdrawal": {
            "base_daily_rate": 0.0003,
            "risk_factors": {
                "active_ae_grade_3_plus": 2.5,
                "ecog_worsened": 2.0,
                "treatment_weeks_gt_12": 1.5,
                "poor_response": 1.3,
            },
        },
        "physician_decision": {
            "base_daily_rate": 0.0001,
            "risk_factors": {
                "ecog_ge_3": 3.0,
                "multiple_dose_reductions": 2.0,
                "poor_tumor_response": 2.0,
                "severe_ae": 1.5,
            },
        },
    }
}

# ---------------------------------------------------------------------------
# Default dose modification rule — applied when LLM generates no AE-specific rules
# ---------------------------------------------------------------------------
DEFAULT_DOSE_MOD_RULE = {
    "ae_term": "default",
    "grade_actions": {
        "1": "DOSE NOT CHANGED",
        "2": "DOSE NOT CHANGED",
        "3": "DRUG INTERRUPTED",
        "4": "DRUG WITHDRAWN",
    },
    "dose_reduction_levels": [1.0, 0.75, 0.50],
    "rechallenge_criteria": "If toxicity resolves to Grade 1 or less, may resume at next lower dose level.",
}

# ---------------------------------------------------------------------------
# Default AE cascade rules — common oncology cascades
# Applied when LLM-extracted triggers yield no cross-AE cascades
# ---------------------------------------------------------------------------
DEFAULT_AE_CASCADE_RULES = [
    {"trigger_ae": "neutropenia", "grade_threshold": 3, "target_ae": "febrile_neutropenia", "multiplier": 3.0},
    {"trigger_ae": "nausea", "grade_threshold": 2, "target_ae": "decreased_appetite", "multiplier": 1.5},
    {"trigger_ae": "anemia", "grade_threshold": 3, "target_ae": "fatigue", "multiplier": 1.6},
    {"trigger_ae": "diarrhea", "grade_threshold": 3, "target_ae": "decreased_appetite", "multiplier": 1.4},
    {"trigger_ae": "nausea", "grade_threshold": 2, "target_ae": "vomiting", "multiplier": 2.0},
]

# ---------------------------------------------------------------------------
# AE duration defaults — keyed by AE category keyword (substring match)
# Values are (mean, std, min, max) for lognormal distribution in days
# ---------------------------------------------------------------------------
AE_DURATION_DEFAULTS = {
    "hematologic": {"mean": 10, "std": 5, "min": 3, "max": 28},
    "neutropenia": {"mean": 8, "std": 4, "min": 2, "max": 21},
    "anemia": {"mean": 14, "std": 7, "min": 5, "max": 42},
    "thrombocytopenia": {"mean": 10, "std": 5, "min": 3, "max": 28},
    "nausea": {"mean": 5, "std": 3, "min": 1, "max": 14},
    "vomiting": {"mean": 3, "std": 2, "min": 1, "max": 10},
    "diarrhea": {"mean": 5, "std": 3, "min": 1, "max": 21},
    "mucositis": {"mean": 10, "std": 5, "min": 3, "max": 28},
    "stomatitis": {"mean": 10, "std": 5, "min": 3, "max": 28},
    "fatigue": {"mean": 14, "std": 10, "min": 3, "max": 60},
    "rash": {"mean": 14, "std": 7, "min": 3, "max": 42},
    "pruritus": {"mean": 10, "std": 5, "min": 2, "max": 30},
    "dermatologic": {"mean": 14, "std": 7, "min": 3, "max": 42},
    "neuropathy": {"mean": 30, "std": 20, "min": 7, "max": 180},
    "hepatic": {"mean": 14, "std": 7, "min": 5, "max": 42},
    "hepatotoxicity": {"mean": 14, "std": 7, "min": 5, "max": 42},
    "cardiac": {"mean": 14, "std": 10, "min": 3, "max": 60},
    "hypertension": {"mean": 30, "std": 20, "min": 7, "max": 180},
    "respiratory": {"mean": 10, "std": 7, "min": 3, "max": 42},
    "pneumonitis": {"mean": 21, "std": 14, "min": 7, "max": 90},
    "interstitial lung disease": {"mean": 30, "std": 20, "min": 7, "max": 120},
    "infection": {"mean": 10, "std": 5, "min": 3, "max": 30},
    "edema": {"mean": 14, "std": 7, "min": 3, "max": 42},
    "pain": {"mean": 7, "std": 5, "min": 1, "max": 30},
    "arthralgia": {"mean": 10, "std": 7, "min": 2, "max": 42},
    "myalgia": {"mean": 7, "std": 5, "min": 2, "max": 30},
    "alopecia": {"mean": 90, "std": 30, "min": 30, "max": 365},
    "metabolic": {"mean": 14, "std": 7, "min": 3, "max": 42},
    "endocrine": {"mean": 30, "std": 20, "min": 7, "max": 365},
    "hypothyroidism": {"mean": 60, "std": 30, "min": 14, "max": 365},
    "hypersensitivity": {"mean": 3, "std": 2, "min": 1, "max": 7},
    "infusion": {"mean": 2, "std": 1, "min": 1, "max": 5},
}

# Default when no category matches
AE_DURATION_FALLBACK = {"mean": 10, "std": 7, "min": 2, "max": 42}

# ---------------------------------------------------------------------------
# Route normalization — maps current schema route strings to target enum
# ---------------------------------------------------------------------------
ROUTE_MAP = {
    "iv": "INTRAVENOUS",
    "iv infusion": "INTRAVENOUS",
    "intravenous": "INTRAVENOUS",
    "intravenous infusion": "INTRAVENOUS",
    "oral": "ORAL",
    "po": "ORAL",
    "subcutaneous": "SUBCUTANEOUS",
    "sc": "SUBCUTANEOUS",
    "sq": "SUBCUTANEOUS",
    "intramuscular": "INTRAMUSCULAR",
    "im": "INTRAMUSCULAR",
    "topical": "TOPICAL",
}


def get_default_baseline():
    """Return a copy of the generic disease baseline."""
    import copy
    return copy.deepcopy(_DEFAULT_BASELINE)


def get_default_mortality():
    """Return a copy of the generic mortality model."""
    import copy
    return copy.deepcopy(_DEFAULT_MORTALITY)
