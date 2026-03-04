"""E2B(R3) code mapping tables for regulatory reporting."""

# AE Outcome → E2B CL11
OUTCOME_CL11: dict[str, int] = {
    "RECOVERED/RESOLVED": 1,
    "RECOVERING/RESOLVING": 2,
    "NOT RECOVERED/NOT RESOLVED": 3,
    "RECOVERED/RESOLVED WITH SEQUELAE": 4,
    "FATAL": 5,
    "UNKNOWN": 6,
}

# Action Taken → E2B CL15
ACTION_CL15: dict[str, int] = {
    "DRUG WITHDRAWN": 1,
    "DRUG INTERRUPTED": 1,
    "DOSE REDUCED": 2,
    "DOSE INCREASED": 3,
    "DOSE NOT CHANGED": 4,
    "UNKNOWN": 5,
    "NOT APPLICABLE": 6,
}

# Rechallenge → E2B CL16
RECHALLENGE_CL16: dict[str, int] = {
    "Yes, recurred": 1,
    "Yes, did not recur": 2,
    "Does not apply": 4,
}

# Sex → E2B D.5
SEX_E2B: dict[str, int] = {
    "Male": 1,
    "Female": 2,
}
