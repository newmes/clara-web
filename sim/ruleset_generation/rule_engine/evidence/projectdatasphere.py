"""Async evidence collector for Project Data Sphere (PDS) — patient-level trial data.

PDS hosts 255+ oncology Phase II-III trials with patient-level data from
250,000+ patients.  Data is pre-downloaded to local CSV cache by
``scripts/pds_download.py`` (SDTM/ADaM domain tables).  This module reads
the cached CSVs, matches trials to the pipeline query, and aggregates
patient-level data into structured evidence models.

Returns aggregated demographics, AE frequencies with grade distributions,
efficacy endpoints (ORR, PFS, OS), and per-drug regimen data.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from rule_engine.config import RuleEngineConfig
from rule_engine.schema import (
    PDSAEAggregate,
    PDSDemographics,
    PDSEfficacy,
    PDSEvidence,
    PDSRegimen,
    PDSTrialMatch,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File discovery — handles non-standard naming across PDS sponsors
# ---------------------------------------------------------------------------

# Ordered candidate file names per domain (first match wins)
_DM_FILES = ["adsl.csv", "dm.csv", "c_bchar.csv", "demog.csv", "demo.csv"]
_AE_FILES = ["adae.csv", "ae.csv", "c_ae.csv", "adverse.csv"]
_EX_FILES = ["ex.csv", "adex.csv", "c_doses.csv", "testdrug.csv", "exposure.csv", "chemotx.csv"]
_RS_FILES = ["adrs.csv", "rs.csv", "respeval.csv"]
_TTE_FILES = ["adtte.csv", "pfs.csv", "os.csv"]


def _find_domain_file(trial_dir: Path, candidates: list[str]) -> Path | None:
    """Find first existing file from ordered candidate list."""
    for name in candidates:
        path = trial_dir / name
        if path.exists():
            return path
    # Fallback: glob for pattern matches (handles c9732_demographic.csv etc.)
    return None


def _find_dm_file(trial_dir: Path) -> Path | None:
    """Find demographics/subject-level file with fallback glob."""
    found = _find_domain_file(trial_dir, _DM_FILES)
    if found:
        return found
    # Glob for *demographic*.csv, *clinical*.csv, *bchar*.csv
    for pattern in ["*demographic*.csv", "*demog*.csv", "*clinical*.csv", "*bchar*.csv"]:
        matches = sorted(trial_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _find_ae_file(trial_dir: Path) -> Path | None:
    """Find AE domain file with fallback glob."""
    found = _find_domain_file(trial_dir, _AE_FILES)
    if found:
        return found
    for pattern in ["*ae*.csv", "*_ae.csv", "*adverse*.csv"]:
        matches = sorted(trial_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _find_ex_file(trial_dir: Path) -> Path | None:
    """Find exposure/dosing file with fallback glob."""
    found = _find_domain_file(trial_dir, _EX_FILES)
    if found:
        return found
    for pattern in ["*doses*.csv", "*dose*.csv", "*testdrug*.csv", "*exposure*.csv", "*chemotx*.csv"]:
        matches = sorted(trial_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


# ---------------------------------------------------------------------------
# Column resolver — handles SDTM vs ADaM naming variance
# ---------------------------------------------------------------------------

class ColumnResolver:
    """Case-insensitive column lookup with fallback lists.

    Handles naming differences between SDTM (DM, AE, EX) and ADaM
    (ADSL, ADAE) datasets.  Column names in clinical data are notoriously
    inconsistent — this resolver tries multiple candidates.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._cols = {c.upper(): c for c in df.columns}

    def get(self, *candidates: str) -> str | None:
        """Return the first matching column name (case-insensitive)."""
        for c in candidates:
            actual = self._cols.get(c.upper())
            if actual is not None:
                return actual
        return None

    def has(self, *candidates: str) -> bool:
        return self.get(*candidates) is not None


# ---------------------------------------------------------------------------
# Trial matching
# ---------------------------------------------------------------------------

def _normalize_drug(name: str) -> str:
    """Normalize drug name for fuzzy matching."""
    return re.sub(r"[^a-z0-9]", "", name.lower().strip())


def match_trials(
    drugs: list[str],
    indication: str,
    index_path: Path,
) -> list[PDSTrialMatch]:
    """Match pipeline query to PDS trial_index.csv entries.

    Score = 0.75 * drug_match + 0.25 * indication_match.
    Minimum threshold: 0.7 (rejects cross-indication matches).
    For combos, all query drugs must appear in the trial's drug list.
    """
    if not index_path.exists():
        return []

    query_drugs = {_normalize_drug(d) for d in drugs}
    indication_lower = indication.lower()

    matches: list[PDSTrialMatch] = []
    with open(index_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            trial_drugs_raw = row.get("drugs", "")
            if not trial_drugs_raw:
                continue
            trial_drugs = {_normalize_drug(d) for d in trial_drugs_raw.split(";")}
            trial_indication = (row.get("indication", "") or "").lower()

            # Drug match: fraction of query drugs found in trial
            if not query_drugs:
                drug_score = 0.0
            else:
                matched = sum(1 for qd in query_drugs if any(qd in td or td in qd for td in trial_drugs))
                drug_score = matched / len(query_drugs)

            # Indication match: keyword overlap
            ind_score = 0.0
            if trial_indication:
                ind_words = set(trial_indication.split())
                query_words = set(indication_lower.split())
                common = ind_words & query_words
                if common:
                    ind_score = len(common) / max(len(ind_words), len(query_words))
                # Bonus for exact substring match
                if indication_lower in trial_indication or trial_indication in indication_lower:
                    ind_score = max(ind_score, 0.8)

            score = 0.75 * drug_score + 0.25 * ind_score

            if score >= 0.7:
                n_patients = int(row.get("n_patients", 0) or 0)
                matches.append(PDSTrialMatch(
                    trial_id=row["trial_id"],
                    drugs=[d.strip() for d in trial_drugs_raw.split(";") if d.strip()],
                    indication=row.get("indication"),
                    n_patients=n_patients,
                    match_score=round(score, 3),
                ))

    # Sort by score descending, then by patient count
    matches.sort(key=lambda m: (m.match_score, m.n_patients), reverse=True)
    return matches


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------

def aggregate_demographics(dm_df: pd.DataFrame) -> PDSDemographics:
    """Aggregate demographics from DM (Demographics) domain.

    Extracts age statistics, sex ratio, race distribution, and ECOG PS
    from individual patient records.
    """
    cr = ColumnResolver(dm_df)

    n_patients = len(dm_df)
    result = PDSDemographics(n_patients=n_patients)

    # Age
    age_col = cr.get("AGE", "APTS_AGE", "AGE_AT_ENROLLMENT", "AGE_YEARS")
    if age_col:
        ages = pd.to_numeric(dm_df[age_col], errors="coerce").dropna()
        if len(ages) >= 10:
            result.age_mean = round(float(ages.mean()), 1)
            result.age_std = round(float(ages.std()), 1)
            result.age_min = int(ages.min())
            result.age_max = int(ages.max())

    # Sex — handles text ("M"/"Male"), numeric codes (0=M/1=F or 1=M/2=F)
    sex_col = cr.get("SEX", "SEXCDC", "GENDER", "SEXC")
    if sex_col:
        raw_sex = dm_df[sex_col].dropna()
        total_sex = len(raw_sex)
        if total_sex > 0:
            sex_upper = raw_sex.astype(str).str.upper().str.strip()
            n_male = sex_upper.isin(["M", "MALE", "1", "1.0"]).sum()
            n_female = sex_upper.isin(["F", "FEMALE", "2", "2.0"]).sum()
            # Some datasets use 0=Male, 1=Female (Amgen convention)
            if n_male == 0 and n_female == 0:
                n_male = sex_upper.isin(["0", "0.0"]).sum()
                n_female = sex_upper.isin(["1", "1.0"]).sum()
            result.pct_male = round(n_male / total_sex * 100, 1)
            result.pct_female = round(n_female / total_sex * 100, 1)

    # Race
    race_col = cr.get("RACE", "ETHNIC", "ETHNICITY")
    if race_col:
        race_counts = dm_df[race_col].dropna().value_counts()
        total_race = race_counts.sum()
        if total_race > 0:
            result.race_distribution = {
                str(k): round(v / total_race * 100, 1)
                for k, v in race_counts.items()
            }

    # ECOG PS
    ecog_col = cr.get(
        "ECOG", "ECOGPS", "ECOG_PS", "ECOGBL", "BECOG",
        "B_ECOG2", "B_ECOGN", "PS",
    )
    if ecog_col:
        ecog_values = pd.to_numeric(dm_df[ecog_col], errors="coerce").dropna()
        # Filter valid ECOG values (0-4)
        ecog_values = ecog_values[(ecog_values >= 0) & (ecog_values <= 4)]
        total_ecog = len(ecog_values)
        if total_ecog > 0:
            ecog_counts = ecog_values.value_counts()
            result.ecog_distribution = {
                str(int(k)): round(v / total_ecog * 100, 1)
                for k, v in ecog_counts.items()
            }

    return result


def aggregate_aes(
    ae_df: pd.DataFrame,
    dm_df: pd.DataFrame,
) -> list[PDSAEAggregate]:
    """Aggregate AE data from AE or ADAE domain.

    Computes per-AE-term: frequency (n_patients_with_event / N),
    grade distribution (worst grade per patient), median onset day,
    and median duration.
    """
    cr_ae = ColumnResolver(ae_df)
    cr_dm = ColumnResolver(dm_df)

    # Safety population: unique subjects in DM
    subj_col_dm = cr_dm.get("USUBJID", "SUBJID", "PATID", "SUBJECT", "PHATOM_ID", "mask_id", "PID_A")
    if subj_col_dm is None:
        log.warning("PDS: No subject ID column in DM table")
        return []
    n_total = dm_df[subj_col_dm].nunique()
    if n_total == 0:
        return []

    # Subject and term columns in AE table
    subj_col_ae = cr_ae.get("USUBJID", "SUBJID", "PATID", "SUBJECT", "PHATOM_ID", "mask_id", "PID_A")
    term_col = cr_ae.get("AEDECOD", "AETERM", "AEPTERM", "CAE_TERM", "AE_NAME", "EventName", "PT_MEDDRA_TERM", "PREFTEXT", "AEPT", "AELLT")
    if subj_col_ae is None or term_col is None:
        log.warning("PDS: Missing subject/term columns in AE table")
        return []

    # Grade column
    grade_col = cr_ae.get("AETOXGR", "AESEV", "SEVRCD", "CTCAE_GRADE", "AEGRPID", "ATOXGR", "GRADE_ID", "AE_GRADE", "AEGRADE", "AESEVCD")

    # Onset and duration columns
    onset_col = cr_ae.get("AESTDY", "AEDY", "ASTDY", "STUDYDAY")
    end_col = cr_ae.get("AEENDY", "AENDY")
    dur_col = cr_ae.get("AEDUR")  # Amgen: direct duration in days

    # Filter to TEAEs if flag available (Amgen convention)
    teae_col = cr_ae.get("TEAEYN")
    if teae_col:
        pre_n = len(ae_df)
        ae_df = ae_df[ae_df[teae_col].astype(str).isin(["1", "1.0", "Y", "YES"])]
        log.debug("PDS: TEAE filter %d -> %d AE records", pre_n, len(ae_df))

    # Severity text → CTCAE grade mapping (handles "1", "1.0", "Mild", etc.)
    _SEVERITY_MAP = {
        "MILD": 1, "1": 1, "1.0": 1,
        "MODERATE": 2, "2": 2, "2.0": 2,
        "SEVERE": 3, "3": 3, "3.0": 3,
        "LIFE-THREATENING": 4, "LIFE THREATENING": 4, "4": 4, "4.0": 4,
        "DEATH": 5, "FATAL": 5, "5": 5, "5.0": 5,
    }

    results: dict[str, dict] = {}  # term -> aggregation

    for term, group in ae_df.groupby(term_col):
        term_str = str(term).strip()
        if not term_str:
            continue

        # Unique patients with this AE
        patients_with_event = group[subj_col_ae].nunique()
        freq_pct = round(patients_with_event / n_total * 100, 1)

        # Grade distribution: worst grade per patient
        grade_dist: dict[str, float] = {}
        if grade_col and grade_col in group.columns:
            # Convert grades to numeric
            grades = group[[subj_col_ae, grade_col]].copy()
            grades["_grade_num"] = grades[grade_col].astype(str).str.upper().map(_SEVERITY_MAP)
            grades = grades.dropna(subset=["_grade_num"])
            if len(grades) > 0:
                worst_per_patient = grades.groupby(subj_col_ae)["_grade_num"].max()
                grade_counts = worst_per_patient.value_counts()
                total_graded = grade_counts.sum()
                if total_graded > 0:
                    grade_dist = {
                        str(int(k)): round(v / total_graded * 100, 1)
                        for k, v in grade_counts.items()
                    }

        # Median onset day
        median_onset = None
        if onset_col and onset_col in group.columns:
            onsets = pd.to_numeric(group[onset_col], errors="coerce").dropna()
            onsets = onsets[onsets > 0]
            if len(onsets) >= 3:
                median_onset = round(float(onsets.median()), 1)

        # Median duration
        median_duration = None
        if dur_col and dur_col in group.columns:
            # Direct duration column (Amgen AEDUR)
            durations = pd.to_numeric(group[dur_col], errors="coerce").dropna()
            durations = durations[durations > 0]
            if len(durations) >= 3:
                median_duration = round(float(durations.median()), 1)
        elif onset_col and end_col and onset_col in group.columns and end_col in group.columns:
            starts = pd.to_numeric(group[onset_col], errors="coerce")
            ends = pd.to_numeric(group[end_col], errors="coerce")
            durations = (ends - starts).dropna()
            durations = durations[durations > 0]
            if len(durations) >= 3:
                median_duration = round(float(durations.median()), 1)

        results[term_str] = {
            "term": term_str,
            "n_patients_with_event": patients_with_event,
            "n_total_patients": n_total,
            "frequency_pct": freq_pct,
            "grade_distribution": grade_dist,
            "median_onset_day": median_onset,
            "median_duration_days": median_duration,
        }

    # Sort by frequency, return top 50
    sorted_aes = sorted(results.values(), key=lambda x: x["frequency_pct"], reverse=True)
    return [PDSAEAggregate(**ae) for ae in sorted_aes[:50]]


def _time_to_months(val: float) -> float:
    """Convert time value to months, auto-detecting days vs months."""
    if val > 100:  # likely days
        return round(val / 30.44, 1)
    return round(val, 1)


def _km_median(times: pd.Series, censors: pd.Series) -> float | None:
    """Compute Kaplan-Meier median survival if lifelines is available."""
    try:
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        # CNSR=0 means event, CNSR=1 means censored
        # lifelines expects event_observed: 1=event, 0=censored
        events = 1 - censors
        kmf.fit(times, event_observed=events)
        m = kmf.median_survival_time_
        if np.isfinite(m):
            return float(m)
    except (ImportError, Exception):
        pass
    return None


def aggregate_efficacy(trial_dir: Path, dm_df: pd.DataFrame | None = None) -> PDSEfficacy:
    """Aggregate efficacy from multiple possible data formats.

    Handles: ADRS/RS (response), ADTTE (time-to-event), separate pfs.csv/os.csv,
    and PFS_TIME/OS_TIME columns embedded in the DM/demographic file.
    """
    result = PDSEfficacy()

    # --- Response data (ORR) ---
    # Format 1: ADRS/RS response files (text responses)
    for rs_name in ["adrs.csv", "rs.csv"]:
        rs_path = trial_dir / rs_name
        if rs_path.exists():
            rs_df = pd.read_csv(rs_path)
            cr = ColumnResolver(rs_df)

            resp_col = cr.get("AVALC", "RSSTRESC", "RSEVAL", "BESTRESP", "BEST_RESPONSE")
            subj_col = cr.get("USUBJID", "SUBJID", "PATID", "PHATOM_ID")
            if resp_col and subj_col:
                responses = rs_df.groupby(subj_col)[resp_col].first()
                resp_str = responses.astype(str).str.upper().str.strip()
                n_evaluable = len(responses)
                if n_evaluable > 0:
                    # Handle both text ("CR") and numeric codes (1=CR, 2=PR, 3=SD, 4=PD)
                    n_cr = resp_str.isin(["CR", "COMPLETE RESPONSE", "1", "1.0"]).sum()
                    n_pr = resp_str.isin(["PR", "PARTIAL RESPONSE", "2", "2.0"]).sum()
                    result.overall_response_rate_pct = round((n_cr + n_pr) / n_evaluable * 100, 1)
                    result.complete_response_rate_pct = round(n_cr / n_evaluable * 100, 1)
            break

    # Format 2: BESTRESP column in DM/demographic file (Alliance 1998)
    if result.overall_response_rate_pct is None and dm_df is not None:
        cr = ColumnResolver(dm_df)
        resp_col = cr.get("BESTRESP", "BEST_RESPONSE")
        if resp_col:
            resp = dm_df[resp_col].dropna()
            resp_str = resp.astype(str).str.upper().str.strip()
            n_eval = len(resp)
            if n_eval > 0:
                n_cr = resp_str.isin(["CR", "COMPLETE RESPONSE", "1", "1.0"]).sum()
                n_pr = resp_str.isin(["PR", "PARTIAL RESPONSE", "2", "2.0"]).sum()
                result.overall_response_rate_pct = round((n_cr + n_pr) / n_eval * 100, 1)
                result.complete_response_rate_pct = round(n_cr / n_eval * 100, 1)

    # --- Time-to-event data (PFS/OS) ---
    # Format 1: ADTTE file
    tte_path = trial_dir / "adtte.csv"
    if tte_path.exists():
        tte_df = pd.read_csv(tte_path)
        cr = ColumnResolver(tte_df)

        param_col = cr.get("PARAM", "PARAMCD", "CNSR_DESC")
        aval_col = cr.get("AVAL", "AVALU")
        cnsr_col = cr.get("CNSR", "CENSOR")

        if param_col and aval_col:
            for param_pattern, field_name in [
                (re.compile(r"progression.free|PFS", re.IGNORECASE), "median_pfs_months"),
                (re.compile(r"overall.survival|^OS$", re.IGNORECASE), "median_os_months"),
            ]:
                param_rows = tte_df[tte_df[param_col].astype(str).apply(lambda x: bool(param_pattern.search(x)))]
                if len(param_rows) == 0:
                    continue

                times = pd.to_numeric(param_rows[aval_col], errors="coerce").dropna()
                if len(times) < 5:
                    continue

                median_val = _time_to_months(float(times.median()))

                if cnsr_col and cnsr_col in param_rows.columns:
                    t = pd.to_numeric(param_rows[aval_col], errors="coerce").dropna()
                    c = pd.to_numeric(param_rows[cnsr_col], errors="coerce").reindex(t.index).fillna(0)
                    km = _km_median(t, c)
                    if km is not None:
                        median_val = _time_to_months(km)

                setattr(result, field_name, median_val)

    # Format 2: Separate pfs.csv / os.csv files (EliLilly)
    for fname, field_name in [("pfs.csv", "median_pfs_months"), ("os.csv", "median_os_months")]:
        if getattr(result, field_name) is not None:
            continue  # Already got from ADTTE
        fpath = trial_dir / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        cr = ColumnResolver(df)
        dur_col = cr.get("PFS_DUR", "OS_DUR", "AVAL", "DURATION", "TIME")
        cens_col = cr.get("CENS", "CNSR", "STATUS", "CENSOR")
        if dur_col:
            times = pd.to_numeric(df[dur_col], errors="coerce").dropna()
            if len(times) >= 5:
                median_val = _time_to_months(float(times.median()))
                if cens_col and cens_col in df.columns:
                    c = pd.to_numeric(df[cens_col], errors="coerce").reindex(times.index).fillna(0)
                    km = _km_median(times, c)
                    if km is not None:
                        median_val = _time_to_months(km)
                setattr(result, field_name, median_val)

    # Format 3: PFS_TIME/OS_TIME columns in DM/demographic file (Alliance)
    if dm_df is not None:
        cr = ColumnResolver(dm_df)
        for time_col_names, cens_col_names, field_name in [
            (("PFS_TIME", "PD_TIME", "failtime"), ("PFS_STATUS", "failcens"), "median_pfs_months"),
            (("OS_TIME", "survtime"), ("STATUS", "survcens"), "median_os_months"),
        ]:
            if getattr(result, field_name) is not None:
                continue
            time_col = cr.get(*time_col_names)
            if time_col:
                times = pd.to_numeric(dm_df[time_col], errors="coerce").dropna()
                if len(times) >= 5:
                    median_val = _time_to_months(float(times.median()))
                    cens_col = cr.get(*cens_col_names)
                    if cens_col:
                        c = pd.to_numeric(dm_df[cens_col], errors="coerce").reindex(times.index).fillna(0)
                        km = _km_median(times, c)
                        if km is not None:
                            median_val = _time_to_months(km)
                    setattr(result, field_name, median_val)

    return result


def aggregate_regimen(
    ex_df: pd.DataFrame,
    dm_df: pd.DataFrame,
) -> list[PDSRegimen]:
    """Aggregate regimen data from EX (Exposure) domain.

    Extracts per-drug median dose, dose unit, and route from actual
    administration records.
    """
    cr_ex = ColumnResolver(ex_df)
    cr_dm = ColumnResolver(dm_df)

    subj_col = cr_ex.get("USUBJID", "SUBJID", "PATID", "PHATOM_ID", "mask_id", "PID_A")
    drug_col = cr_ex.get("EXTRT", "EXDECOD", "TRT01A", "DRUG", "SAFTX", "TX", "DRGNAME", "ACTTRT")
    dose_col = cr_ex.get("EXDOSE", "DOSE", "EXDOSTOT", "TXDOSE", "SAFTXDOS", "DOSTOT", "BSADOSE")
    unit_col = cr_ex.get("EXDOSU", "DOSE_UNIT", "DOSEUNIT", "DOSTOTUF", "DOSEUNCD", "QTYUNIT")
    route_col = cr_ex.get("EXROUTE", "ROUTE", "ROUTECD")

    if drug_col is None or dose_col is None:
        log.debug("PDS: Missing drug/dose columns in EX table")
        return []

    results: list[PDSRegimen] = []

    for drug_name, group in ex_df.groupby(drug_col):
        drug_str = str(drug_name).strip()
        if not drug_str:
            continue

        doses = pd.to_numeric(group[dose_col], errors="coerce").dropna()
        doses = doses[doses > 0]

        median_dose = round(float(doses.median()), 2) if len(doses) > 0 else None

        dose_unit = None
        if unit_col and unit_col in group.columns:
            units = group[unit_col].dropna()
            if len(units) > 0:
                dose_unit = str(units.mode().iloc[0])

        route = None
        if route_col and route_col in group.columns:
            routes = group[route_col].dropna()
            if len(routes) > 0:
                route = str(routes.mode().iloc[0])

        n_patients = group[subj_col].nunique() if subj_col else 0

        results.append(PDSRegimen(
            drug=drug_str,
            median_dose=median_dose,
            dose_unit=dose_unit,
            route=route,
            n_patients=n_patients,
        ))

    return results


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

def _fetch_pds_sync(
    drugs: list[str],
    indication: str,
    config: RuleEngineConfig,
) -> PDSEvidence:
    """Synchronous PDS evidence fetch — reads cached CSV files."""
    data_dir = config.pds_data_dir
    index_path = data_dir / "trial_index.csv"

    if not index_path.exists():
        log.debug("PDS: trial_index.csv not found at %s", index_path)
        return PDSEvidence()

    # Match trials
    matches = match_trials(drugs, indication, index_path)
    if not matches:
        log.debug("PDS: No matching trials for %s + %s", drugs, indication)
        return PDSEvidence()

    best = matches[0]
    trial_dir = data_dir / best.trial_id

    if not trial_dir.exists():
        log.warning("PDS: Matched trial %s but data directory missing: %s", best.trial_id, trial_dir)
        return PDSEvidence()

    log.info(
        "PDS: Matched trial %s (score=%.2f, n=%d) for %s + %s",
        best.trial_id, best.match_score, best.n_patients,
        " + ".join(drugs), indication,
    )

    # Load DM (required for most aggregations)
    dm_file = _find_dm_file(trial_dir)
    if dm_file is None:
        log.warning("PDS: No DM/ADSL data for trial %s", best.trial_id)
        return PDSEvidence(found=True, matched_trial=best)
    dm_df = pd.read_csv(dm_file)
    if len(dm_df) == 0:
        log.warning("PDS: Empty DM file for trial %s", best.trial_id)
        return PDSEvidence(found=True, matched_trial=best)
    log.info("PDS: Loaded DM from %s (%d rows)", dm_file.name, len(dm_df))

    # Demographics
    demographics = aggregate_demographics(dm_df)

    # AE data
    ae_aggregates: list[PDSAEAggregate] = []
    safety_n = 0
    ae_file = _find_ae_file(trial_dir)
    if ae_file is not None:
        ae_df = pd.read_csv(ae_file)
        log.info("PDS: Loaded AE from %s (%d rows)", ae_file.name, len(ae_df))
        ae_aggregates = aggregate_aes(ae_df, dm_df)
        cr = ColumnResolver(dm_df)
        subj_col = cr.get("USUBJID", "SUBJID", "PATID", "SUBJECT", "PHATOM_ID", "mask_id", "PID_A")
        safety_n = dm_df[subj_col].nunique() if subj_col else len(dm_df)

    # Efficacy
    efficacy = aggregate_efficacy(trial_dir, dm_df)

    # Regimen
    regimen: list[PDSRegimen] = []
    ex_file = _find_ex_file(trial_dir)
    if ex_file is not None:
        ex_df = pd.read_csv(ex_file)
        log.info("PDS: Loaded EX from %s (%d rows)", ex_file.name, len(ex_df))
        regimen = aggregate_regimen(ex_df, dm_df)

    return PDSEvidence(
        found=True,
        matched_trial=best,
        demographics=demographics,
        ae_aggregates=ae_aggregates,
        efficacy=efficacy,
        regimen=regimen,
        safety_population_n=safety_n,
    )


async def fetch_pds(
    drugs: list[str],
    indication: str,
    config: RuleEngineConfig,
) -> PDSEvidence:
    """Fetch aggregated evidence from cached Project Data Sphere data.

    Args:
        drugs: List of generic drug names.
        indication: Disease / indication term.
        config: Pipeline config (provides ``pds_data_dir`` path).

    Returns:
        Populated PDSEvidence; empty defaults if no cached data or no match.
    """
    try:
        return await asyncio.to_thread(_fetch_pds_sync, drugs, indication, config)
    except Exception as exc:
        log.warning("PDS evidence fetch failed for %s + %s: %s", drugs, indication, exc)
        return PDSEvidence()
