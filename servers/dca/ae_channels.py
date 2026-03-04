"""AE Detection Channels — extracted from ClinicalTrialEngine observation.py.

Each AE's detection channel info: which observation methods can detect it.
"""

from __future__ import annotations

from typing import Any

AE_DETECTION_CHANNELS: dict[str, dict[str, Any]] = {
    # -- Hematologic (Lab-detected) --
    "neutropenia": {
        "channels": ["lab"],
        "patient_aware_threshold": 3,
        "detection_requires_lab": True,
    },
    "thrombocytopenia": {
        "channels": ["lab", "video_detectable"],
        "patient_aware_threshold": 2,
        "video_signs": ["bruising", "petechiae"],
        "detection_requires_lab": True,
    },
    "anemia": {
        "channels": ["lab", "video_detectable", "patient_reported"],
        "patient_aware_threshold": 2,
        "video_signs": ["pallor", "conjunctival_pallor"],
    },
    "leukopenia": {
        "channels": ["lab"],
        "patient_aware_threshold": 3,
        "detection_requires_lab": True,
    },

    # -- Skin (Video-detectable) --
    "rash": {
        "channels": ["patient_reported", "video_detectable", "physical_exam"],
        "patient_aware_threshold": 1,
        "video_signs": ["visible_rash", "erythema", "papules"],
    },
    "rash_maculopapular": {
        "channels": ["patient_reported", "video_detectable", "physical_exam"],
        "patient_aware_threshold": 1,
        "video_signs": ["visible_rash", "erythema", "maculopapular_lesions"],
    },
    "pruritus": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },
    "skin_eruption": {
        "channels": ["patient_reported", "video_detectable", "physical_exam"],
        "patient_aware_threshold": 1,
        "video_signs": ["skin_lesions", "erythema"],
    },
    "palmar_plantar": {
        "channels": ["patient_reported", "video_detectable"],
        "patient_aware_threshold": 1,
        "video_signs": ["hand_redness", "peeling_skin"],
    },
    "alopecia": {
        "channels": ["video_detectable", "patient_reported"],
        "patient_aware_threshold": 1,
        "video_signs": ["visible_hair_loss", "thinning_hair"],
    },
    "nail_changes": {
        "channels": ["video_detectable", "patient_reported"],
        "patient_aware_threshold": 1,
        "video_signs": ["nail_discoloration", "nail_dystrophy"],
    },
    "nail_loss": {
        "channels": ["video_detectable", "patient_reported"],
        "patient_aware_threshold": 1,
        "video_signs": ["nail_separation", "nail_absent"],
    },
    "stomatitis": {
        "channels": ["patient_reported", "video_detectable"],
        "patient_aware_threshold": 1,
        "video_signs": ["oral_lesions", "lip_swelling"],
    },
    "mucositis": {
        "channels": ["patient_reported", "video_detectable"],
        "patient_aware_threshold": 1,
        "video_signs": ["oral_erythema", "lip_dryness"],
    },

    # -- Systemic (Patient-reported) --
    "fatigue": {
        "channels": ["patient_reported", "video_detectable"],
        "patient_aware_threshold": 1,
        "video_signs": ["visible_fatigue", "slow_movements"],
    },
    "nausea": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },
    "vomiting": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },
    "diarrhea": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },
    "constipation": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },
    "decreased_appetite": {
        "channels": ["patient_reported", "video_detectable"],
        "patient_aware_threshold": 1,
        "video_signs": ["weight_loss_visible", "cachexia"],
    },

    # -- Neurological --
    "peripheral_neuropathy": {
        "channels": ["patient_reported", "physical_exam"],
        "patient_aware_threshold": 1,
    },
    "neuropathy": {
        "channels": ["patient_reported", "physical_exam"],
        "patient_aware_threshold": 1,
    },
    "dysgeusia": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },

    # -- Respiratory --
    "dyspnea": {
        "channels": ["patient_reported", "video_detectable"],
        "patient_aware_threshold": 1,
        "video_signs": ["labored_breathing", "tachypnea"],
    },
    "cough": {
        "channels": ["patient_reported", "video_detectable"],
        "patient_aware_threshold": 1,
        "video_signs": ["audible_cough"],
    },
    "pneumonitis": {
        "channels": ["patient_reported", "lab", "physical_exam"],
        "patient_aware_threshold": 2,
    },

    # -- Hepatic (Lab-detected) --
    "hepatotoxicity": {
        "channels": ["lab", "video_detectable"],
        "patient_aware_threshold": 3,
        "video_signs": ["jaundice", "scleral_icterus"],
        "detection_requires_lab": True,
    },
    "hepatitis": {
        "channels": ["lab", "video_detectable"],
        "patient_aware_threshold": 3,
        "video_signs": ["jaundice"],
        "detection_requires_lab": True,
    },
    "alt_increased": {
        "channels": ["lab"],
        "patient_aware_threshold": 3,
        "detection_requires_lab": True,
    },
    "ast_increased": {
        "channels": ["lab"],
        "patient_aware_threshold": 3,
        "detection_requires_lab": True,
    },

    # -- Renal --
    "nephrotoxicity": {
        "channels": ["lab"],
        "patient_aware_threshold": 3,
        "detection_requires_lab": True,
    },
    "nephritis": {
        "channels": ["lab", "patient_reported"],
        "patient_aware_threshold": 2,
        "detection_requires_lab": True,
    },
    "proteinuria": {
        "channels": ["lab"],
        "patient_aware_threshold": 3,
        "detection_requires_lab": True,
    },

    # -- Endocrine --
    "hypothyroidism": {
        "channels": ["lab", "patient_reported"],
        "patient_aware_threshold": 2,
        "detection_requires_lab": True,
    },
    "hyperthyroidism": {
        "channels": ["lab", "patient_reported"],
        "patient_aware_threshold": 2,
        "detection_requires_lab": True,
    },
    "hyperglycemia": {
        "channels": ["lab", "patient_reported"],
        "patient_aware_threshold": 2,
    },
    "adrenal_insufficiency": {
        "channels": ["lab", "patient_reported", "video_detectable"],
        "patient_aware_threshold": 2,
        "video_signs": ["hyperpigmentation", "visible_weakness"],
    },

    # -- Pain --
    "arthralgia": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },
    "myalgia": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },
    "headache": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },

    # -- Urological --
    "urinary_tract_infection": {
        "channels": ["patient_reported", "lab"],
        "patient_aware_threshold": 1,
    },
    "hematuria": {
        "channels": ["patient_reported", "lab"],
        "patient_aware_threshold": 1,
    },

    # -- Cardiac --
    "myocarditis": {
        "channels": ["lab", "patient_reported", "physical_exam"],
        "patient_aware_threshold": 2,
        "detection_requires_lab": True,
    },

    # -- Immune-related --
    "colitis": {
        "channels": ["patient_reported"],
        "patient_aware_threshold": 1,
    },
    "infusion_related_reaction": {
        "channels": ["physical_exam", "video_detectable", "patient_reported"],
        "patient_aware_threshold": 1,
        "video_signs": ["flushing", "urticaria", "angioedema"],
    },

    # -- Other/Infection --
    "febrile_neutropenia": {
        "channels": ["patient_reported", "lab", "physical_exam"],
        "patient_aware_threshold": 1,
    },
    "sepsis": {
        "channels": ["patient_reported", "lab", "physical_exam"],
        "patient_aware_threshold": 1,
    },
}
