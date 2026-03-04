"""Async evidence collector for DailyMed SPL drug labels.

Fetches FDA-approved drug labels from the DailyMed REST API and parses
structured adverse-reaction frequency tables, boxed warnings,
contraindications, dosage, and special-population sections.

The AE frequency data from labels serves as ground truth for incidence
rates — unlike FAERS report counts which lack denominators.
"""

from __future__ import annotations

import asyncio
import logging
import re
from difflib import SequenceMatcher
from xml.etree import ElementTree as ET

import requests

from rule_engine.schema import DailyMedEvidence

log = logging.getLogger(__name__)

DAILYMED_API = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

# SPL section LOINC codes
_SECTION_CODES = {
    "34084-4": "adverse_reactions",
    "34068-7": "dosage",
    "34070-3": "contraindications",
    "34066-1": "boxed_warning",
    "42228-7": "special_populations",   # USE IN SPECIFIC POPULATIONS
    "43684-0": "special_populations",   # USE IN SPECIFIC POPULATIONS (alt)
    "43685-7": "special_populations",   # renal impairment subsection
}

# XML namespace used in SPL documents
_SPL_NS = {"spl": "urn:hl7-org:v3"}


def _text_content(elem: ET.Element) -> str:
    """Extract all text content from an XML element, stripping tags."""
    return " ".join(elem.itertext()).strip()


def _normalize_header(text: str) -> str:
    """Normalize header text: replace non-breaking spaces, collapse whitespace."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def _classify_table(headers: list[str], n_pct_cols: int = 0) -> str:
    """Classify an AE table type from its headers.

    Args:
        headers: All header cell texts (flattened from all rows).
        n_pct_cols: Number of percentage-like columns detected. Tables with
            4+ pct columns (2 per arm) are comparison tables.

    Returns:
        'clinical'  — standard clinical AE table (preferred)
        'lab'       — laboratory abnormalities table
        'comparison'— drug vs placebo comparison table
        'unknown'   — can't determine
    """
    header_text = " ".join(headers)
    if any(kw in header_text for kw in ("laboratory", "lab abnormality", "hematologic", "chemistry")):
        return "lab"
    if any(kw in header_text for kw in ("placebo", "comparator", "control", "alone")):
        return "comparison"
    # Tables with 4+ pct-like columns are comparison tables (drug vs comparator × 2 grade levels)
    if n_pct_cols >= 4:
        return "comparison"
    if any(kw in header_text for kw in ("%", "incidence", "all grades", "grade", "frequency")):
        return "clinical"
    return "unknown"


def _identify_pct_columns(headers: list[str]) -> tuple[int | None, int | None]:
    """Identify which columns hold any-grade and grade 3-4 percentages.

    Priority:
    1. Column with "all grade" in header → any_grade_col
    2. Column with "grade 3" or "grade ≥3" → grade34_col
    3. First column with "%" or "incidence" → any_grade_col (fallback)
    4. Second such column → grade34_col (fallback)

    Returns: (any_grade_col_index, grade34_col_index) — either can be None
    """
    any_grade_col = None
    grade34_col = None
    generic_pct_cols: list[int] = []

    for i, h in enumerate(headers):
        h_lower = h.lower()
        if any(kw in h_lower for kw in ("all grade", "any grade", "all-grade")):
            any_grade_col = i
        elif any(kw in h_lower for kw in ("grade 3", "grade ≥3", "grade ≥ 3", "grade 3-4", "grade 3–4", "severe")):
            grade34_col = i
        elif any(kw in h_lower for kw in ("%", "incidence", "rate", "frequency")):
            generic_pct_cols.append(i)

    # Fallback: use generic pct columns in order
    if any_grade_col is None and generic_pct_cols:
        any_grade_col = generic_pct_cols[0]
    if grade34_col is None and len(generic_pct_cols) >= 2:
        remaining = [c for c in generic_pct_cols if c != any_grade_col]
        if remaining:
            grade34_col = remaining[0]

    return any_grade_col, grade34_col


def _extract_pct(val_str: str) -> float | None:
    """Parse a percentage string into a float, handling ranges and '<' prefixes."""
    val_str = val_str.strip().rstrip("%").strip()
    # Handle ranges like "10-15" → take midpoint
    range_match = re.match(r"(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)", val_str)
    if range_match:
        lo, hi = float(range_match.group(1)), float(range_match.group(2))
        return (lo + hi) / 2.0
    if re.match(r"^\d+\.?\d*$", val_str):
        return float(val_str)
    if val_str.startswith("<"):
        num_match = re.search(r"\d+\.?\d*", val_str)
        return float(num_match.group()) / 2.0 if num_match else None
    return None


def _parse_ae_tables(section_elem: ET.Element) -> list[dict]:
    """Parse <table> elements inside the ADVERSE REACTIONS section.

    Classifies each table (clinical / lab / comparison / unknown) and
    identifies the correct percentage columns (all-grades vs grade 3-4).
    Returns list of dicts with term, incidence_pct, grade34_pct, table_type.
    """
    ae_entries: list[dict] = []

    for table in section_elem.iter("{urn:hl7-org:v3}table"):
        # Gather headers — collect ALL rows for classification, but use
        # the LAST header row for column identification (multi-row headers
        # have drug names in row 1 and grade labels in row 2)
        all_headers: list[str] = []
        last_header_row: list[str] = []
        thead = table.find("{urn:hl7-org:v3}thead")
        if thead is not None:
            for th_row in thead.iter("{urn:hl7-org:v3}tr"):
                row_cells = [_normalize_header(_text_content(th).lower()) for th in th_row]
                all_headers.extend(row_cells)
                if row_cells:
                    last_header_row = row_cells

        if not all_headers:
            # Try first row of tbody as header
            tbody = table.find("{urn:hl7-org:v3}tbody")
            if tbody is None:
                continue
            rows = list(tbody.iter("{urn:hl7-org:v3}tr"))
            if not rows:
                continue
            for td in rows[0]:
                all_headers.append(_normalize_header(_text_content(td).lower()))
            last_header_row = all_headers
            data_rows = rows[1:]
        else:
            tbody = table.find("{urn:hl7-org:v3}tbody")
            if tbody is None:
                continue
            data_rows = list(tbody.iter("{urn:hl7-org:v3}tr"))

        if len(all_headers) < 2:
            continue

        # Determine actual data column count from first non-empty data row
        n_data_cols = 0
        for row in data_rows:
            n_data_cols = len(list(row))
            if n_data_cols >= 2:
                break

        # For column identification: multi-row headers are common in SPL tables.
        # The last header row has per-column labels ("All Grades (%)", "Grades 3-4 (%)").
        # It may have fewer cells than data columns due to rowspan on the term column.
        # Build a padded header list matching data column count.
        col_headers: list[str]
        if last_header_row and n_data_cols > 0:
            offset = n_data_cols - len(last_header_row)
            if 0 <= offset <= 1:
                col_headers = [""] * offset + last_header_row
            elif len(all_headers) == n_data_cols:
                col_headers = all_headers
            else:
                col_headers = all_headers
        else:
            col_headers = all_headers

        # Count pct-like columns for table classification
        n_pct_like = sum(
            1 for h in col_headers
            if any(kw in h for kw in ("all grade", "any grade", "grade 3", "grade 3-4", "%", "incidence"))
        )
        # Use all_headers for classification (captures drug names, "laboratory", etc.)
        table_type = _classify_table(all_headers, n_pct_cols=n_pct_like)

        any_grade_col, grade34_col = _identify_pct_columns(col_headers)

        # Fallback: if no identified columns, use columns 1+ as generic pct
        fallback_pct_cols: list[int] | None = None
        if any_grade_col is None:
            fallback_pct_cols = list(range(1, n_data_cols or len(col_headers)))

        term_col = 0

        for row in data_rows:
            cells = [_text_content(td) for td in row]
            if len(cells) < 2:
                continue

            term = cells[term_col].strip()
            if not term or len(term) > 120:
                continue

            entry: dict = {"term": term, "table_type": table_type}

            # Extract from identified columns
            if any_grade_col is not None and any_grade_col < len(cells):
                pct = _extract_pct(cells[any_grade_col])
                if pct is not None:
                    entry["incidence_pct"] = round(pct, 1)

            if grade34_col is not None and grade34_col < len(cells):
                pct = _extract_pct(cells[grade34_col])
                if pct is not None:
                    entry["grade34_pct"] = round(pct, 1)

            # Fallback: scan columns for first valid pct
            if "incidence_pct" not in entry and fallback_pct_cols:
                for idx in fallback_pct_cols:
                    if idx >= len(cells):
                        continue
                    pct = _extract_pct(cells[idx])
                    if pct is not None:
                        entry["incidence_pct"] = round(pct, 1)
                        break

            if "incidence_pct" in entry:
                ae_entries.append(entry)

    return ae_entries


def _dedup_ae_entries(ae_entries: list[dict]) -> list[dict]:
    """Remove duplicate AE terms, preferring clinical table entries.

    SPL labels for multi-indication drugs (e.g. Pembrolizumab) contain
    separate AE tables per indication plus lab abnormality tables.
    When a term appears in both a clinical and a lab/unknown table,
    the clinical entry is preferred.
    """
    best: dict[str, dict] = {}  # key → best entry
    order: list[str] = []       # insertion order
    _TYPE_PRIORITY = {"clinical": 3, "comparison": 2, "lab": 1, "unknown": 0}

    for entry in ae_entries:
        key = entry.get("term", "").strip().lower()
        if not key:
            continue
        entry_priority = _TYPE_PRIORITY.get(entry.get("table_type", ""), 0)
        existing = best.get(key)
        if existing is None:
            best[key] = entry
            order.append(key)
        else:
            existing_priority = _TYPE_PRIORITY.get(existing.get("table_type", ""), 0)
            if entry_priority > existing_priority:
                best[key] = entry
    return [best[k] for k in order]


_DOSAGE_FORM_WORDS = {
    "injection", "tablet", "capsule", "powder", "solution", "suspension",
    "cream", "ointment", "gel", "patch", "spray", "inhaler", "kit",
    "for", "intravenous", "subcutaneous", "oral", "reconstitution",
    "lyophilized", "concentrate", "infusion",
}

# Biosimilar suffix pattern: 4-letter suffix after hyphen (e.g., -qyyp, -anns, -dkst, -nxki, -oysk)
_BIOSIMILAR_SUFFIX_RE = re.compile(r"-[a-z]{4}$")
# Prefix modifiers used in conjugate/modified drug names (e.g., fam-, ado-)
_PREFIX_MODIFIERS_RE = re.compile(r"^(fam|ado|bv|vc|nax|sac)-", re.IGNORECASE)


def _extract_generic_name(title: str) -> str:
    """Extract and normalize the generic drug name from a DailyMed SPL title.

    Handles format: "BRAND (generic-name) dosage_form [manufacturer]"
    Also handles: "BRAND- generic_name dosage_form"
    """
    title_lower = title.lower().strip()

    # Try extracting from parentheses first — most common DailyMed format
    paren_match = re.search(r"\(([^)]+)\)", title_lower)
    if paren_match:
        drug_part = paren_match.group(1)
    elif "- " in title_lower:
        drug_part = title_lower.split("- ", 1)[1]
    else:
        drug_part = title_lower

    # Remove manufacturer in brackets [...]
    drug_part = re.sub(r"\[.*?\]", "", drug_part).strip()

    # Remove biosimilar suffixes (-qyyp, -anns, etc.)
    words = drug_part.split()
    cleaned = []
    for w in words:
        w = _BIOSIMILAR_SUFFIX_RE.sub("", w)
        w = _PREFIX_MODIFIERS_RE.sub("", w)
        if w and w not in _DOSAGE_FORM_WORDS and w not in {"and", ","}:
            cleaned.append(w.strip(","))

    return " ".join(cleaned).strip()


def _name_similarity(query: str, candidate_title: str) -> float:
    """Score how well a DailyMed SPL title matches the queried drug name.

    DailyMed titles typically: "BRAND (generic_name) dosage_form [manufacturer]"
    Strategy:
      1. Extract and normalize generic name from title
      2. Score: exact → 1.0; prefix with extra words → penalized; fallback SequenceMatcher
    """
    query_lower = query.lower().strip()
    drug_name_clean = _extract_generic_name(candidate_title)

    if not drug_name_clean:
        return 0.0

    # Exact match
    if drug_name_clean == query_lower:
        return 1.0

    # Query is a prefix of the cleaned drug name (e.g., "trastuzumab" vs "trastuzumab deruxtecan")
    if drug_name_clean.startswith(query_lower):
        extra_words = drug_name_clean[len(query_lower):].strip().split()
        # Penalize based on extra words — more extra words = worse match
        penalty = 0.1 * len(extra_words)
        return max(0.5, 0.9 - penalty)

    # Query appears as substring (e.g., combo drugs)
    if query_lower in drug_name_clean:
        return 0.6

    # Fallback: SequenceMatcher
    return SequenceMatcher(None, query_lower, drug_name_clean).ratio() * 0.5


def _fetch_dailymed_sync(drug_name: str, timeout: int) -> DailyMedEvidence:
    """Synchronous DailyMed lookup — runs inside asyncio.to_thread."""
    session = requests.Session()
    session.headers["User-Agent"] = "RuleEngine/1.0"

    # Step 1: resolve SPL setid — fetch multiple candidates and pick best match
    try:
        resp = session.get(
            f"{DAILYMED_API}/spls.json",
            params={"drug_name": drug_name, "page": 1, "pagesize": 5},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.debug("DailyMed SPL search failed for %s: %s", drug_name, exc)
        return DailyMedEvidence()

    spls = data.get("data", [])
    if not spls:
        log.debug("DailyMed: no SPLs found for %s", drug_name)
        return DailyMedEvidence()

    # Score candidates by name similarity and pick the best match
    scored = []
    for candidate in spls:
        title = candidate.get("title", "")
        score = _name_similarity(drug_name, title)
        scored.append((score, candidate))
        log.debug("DailyMed candidate: '%s' → score %.2f", title, score)

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, spl = scored[0]

    if best_score < 0.3:
        log.warning(
            "DailyMed: best match for '%s' has low score %.2f ('%s')",
            drug_name, best_score, spl.get("title", ""),
        )

    set_id = spl.get("setid")
    drug_name_label = spl.get("title", drug_name)

    if not set_id:
        return DailyMedEvidence()

    # Step 2: download full SPL XML
    try:
        xml_resp = session.get(
            f"{DAILYMED_API}/spls/{set_id}.xml",
            timeout=timeout * 2,  # XML can be large
        )
        xml_resp.raise_for_status()
    except Exception as exc:
        log.warning("DailyMed XML fetch failed for %s (setid=%s): %s", drug_name, set_id, exc)
        return DailyMedEvidence(found=True, set_id=set_id, drug_name_label=drug_name_label)

    # Step 3: parse XML sections
    try:
        root = ET.fromstring(xml_resp.content)
    except ET.ParseError as exc:
        log.warning("DailyMed XML parse failed for %s: %s", drug_name, exc)
        return DailyMedEvidence(found=True, set_id=set_id, drug_name_label=drug_name_label)

    adverse_reactions_text = None
    ae_table: list[dict] = []
    boxed_warning = None
    contraindications = None
    dosage_text = None
    special_populations = None

    for section in root.iter("{urn:hl7-org:v3}section"):
        code_elem = section.find("{urn:hl7-org:v3}code")
        if code_elem is None:
            continue
        code = code_elem.get("code", "")

        section_type = _SECTION_CODES.get(code)
        if section_type is None:
            continue

        text = _text_content(section)
        # Truncate very long sections
        if len(text) > 8000:
            text = text[:8000] + "..."

        if section_type == "adverse_reactions":
            adverse_reactions_text = text
            ae_table = _dedup_ae_entries(_parse_ae_tables(section))
        elif section_type == "dosage":
            dosage_text = text
        elif section_type == "contraindications":
            contraindications = text
        elif section_type == "boxed_warning":
            boxed_warning = text
        elif section_type == "special_populations" and special_populations is None:
            special_populations = text

    evidence = DailyMedEvidence(
        found=True,
        set_id=set_id,
        drug_name_label=drug_name_label,
        adverse_reactions_text=adverse_reactions_text,
        ae_table=ae_table,
        boxed_warning=boxed_warning,
        contraindications=contraindications,
        dosage_text=dosage_text,
        special_populations=special_populations,
    )
    log.debug(
        "DailyMed: %s → setid=%s, AE entries=%d, has_warning=%s",
        drug_name, set_id, len(ae_table), boxed_warning is not None,
    )
    return evidence


async def fetch_dailymed(
    drug_name: str,
    timeout: int = 10,
) -> DailyMedEvidence:
    """Fetch DailyMed drug label evidence for *drug_name*.

    Args:
        drug_name: Generic drug name to search for.
        timeout: HTTP request timeout in seconds.

    Returns:
        Populated DailyMedEvidence; empty defaults on any failure.
    """
    try:
        return await asyncio.to_thread(_fetch_dailymed_sync, drug_name, timeout)
    except Exception as exc:
        log.warning("DailyMed request failed for %s: %s", drug_name, exc)
        return DailyMedEvidence()
