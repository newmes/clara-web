"""Generate MedWatch FDA Form 3500A PDF by filling the official FDA template.

Uses the official FDA Form 3500A (10/15) as background and overlays data
using ReportLab, then merges with pypdf.

Public API:
    generate_medwatch_pdf(data, output_path) -> bytes
    generate_medwatch_pdf_from_dict(data, output_path) -> bytes
"""
from __future__ import annotations

import io
import re
import textwrap
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union

from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from .schemas.medwatch import MedWatch3500A

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEMPLATE_PATH = Path(__file__).parent / "templates" / "fda_3500a_template.pdf"
PAGE_W, PAGE_H = LETTER  # 612 x 792

ILD_RED = HexColor("#CC0000")
ILD_BG = HexColor("#FFF0F0")

NS_BLUE = HexColor("#1565C0")
NS_BG = HexColor("#E3F2FD")

PI_ORANGE = HexColor("#E65100")

# Font sizes matching the form's small-print style
FONT_TEXT = 7.5
FONT_SMALL = 6.5
FONT_CHECK = 8


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _parse_date(d: Optional[Union[date, str]]) -> Optional[date]:
    """Parse date from ISO string or date object."""
    if d is None:
        return None
    if isinstance(d, date):
        return d
    try:
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _date_parts(d: Optional[Union[date, str]]) -> tuple[str, str, str]:
    """Return (day, month_abbr, year) strings for FDA date fields."""
    parsed = _parse_date(d)
    if parsed is None:
        return ("", "", "")
    return (
        str(parsed.day).zfill(2),
        parsed.strftime("%b").upper(),
        str(parsed.year),
    )


def _fmt_date(d: Optional[Union[date, str]]) -> str:
    """Format date as DD-MMM-YYYY for free-text fields."""
    parsed = _parse_date(d)
    if parsed is None:
        return ""
    return parsed.strftime("%d-%b-%Y").upper()


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """Strip markdown formatting for plain-text PDF fields."""
    if not text:
        return ""
    text = re.sub(r"^#{1,3}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    return text.strip()


def _parse_drug_info(drug_name: str) -> tuple[str, str]:
    """Parse 'Name (info) (Manufacturer)' into (name+info, manufacturer)."""
    if not drug_name:
        return ("", "")
    # Try to split on last parenthesized group as manufacturer
    parts = re.findall(r"\(([^)]+)\)", drug_name)
    base = re.sub(r"\s*\([^)]*\)\s*$", "", drug_name).strip()
    if len(parts) >= 2:
        manufacturer = parts[-1]
        name_with_info = re.sub(r"\s*\(" + re.escape(manufacturer) + r"\)\s*$", "", drug_name).strip()
        return (name_with_info, manufacturer)
    return (drug_name, "")


def _parse_dose_freq_route(dfr: str) -> tuple[str, str, str]:
    """Parse '5.4 mg/kg, Q3W, Intravenous' into (dose, freq, route)."""
    if not dfr:
        return ("", "", "")
    parts = [p.strip() for p in dfr.split(",")]
    dose = parts[0] if len(parts) > 0 else ""
    freq = parts[1] if len(parts) > 1 else ""
    route = parts[2] if len(parts) > 2 else ""
    return (dose, freq, route)


def _parse_dechallenge_answer(text: str) -> str:
    """Determine 'yes', 'no', or 'na' from dechallenge text."""
    if not text:
        return "na"
    lower = text.lower().strip()
    if lower.startswith("yes"):
        return "yes"
    if lower.startswith("no"):
        return "no"
    if "does not apply" in lower or "n/a" in lower:
        return "na"
    return "na"


def _wrap_text(text: str, width_pts: float, font_size: float) -> list[str]:
    """Wrap text to fit within a given width in points.

    Rough estimation: average char width ~= font_size * 0.5.
    """
    if not text:
        return []
    avg_char_w = font_size * 0.5
    chars_per_line = max(int(width_pts / avg_char_w), 20)
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip():
            lines.extend(textwrap.wrap(paragraph, width=chars_per_line))
        else:
            lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def _draw_text(c: canvas.Canvas, x: float, y: float, text: str,
               font: str = "Helvetica", size: float = FONT_TEXT):
    """Draw a single line of text at (x, y)."""
    if not text:
        return
    c.setFont(font, size)
    c.setFillColor(black)
    c.drawString(x, y, text)


def _draw_text_in_rect(c: canvas.Canvas, rect: list[float], text: str,
                       font: str = "Helvetica", size: float = FONT_TEXT,
                       leading: float = 0):
    """Draw wrapped text within a rectangle [x1, y1, x2, y2]."""
    if not text:
        return
    x1, y1, x2, y2 = rect
    w = x2 - x1
    h = y2 - y1
    if leading == 0:
        leading = size + 1.5

    lines = _wrap_text(text, w - 4, size)
    max_lines = max(int(h / leading), 1)

    c.setFont(font, size)
    c.setFillColor(black)
    y = y2 - size - 1  # Start from top of rect
    for i, line in enumerate(lines[:max_lines]):
        if y < y1:
            break
        c.drawString(x1 + 2, y, line)
        y -= leading


def _draw_check(c: canvas.Canvas, x: float, y: float, size: float = 8):
    """Draw a checkmark at checkbox position."""
    c.setStrokeColor(black)
    c.setLineWidth(1.2)
    # Draw X-style check that fits inside the checkbox
    c.line(x + 1, y + 1, x + size - 1, y + size - 1)
    c.line(x + 1, y + size - 1, x + size - 1, y + 1)


def _draw_ild_banner(c: canvas.Canvas):
    """Draw ILD warning banner at top of page 1."""
    banner_y = PAGE_H - 52
    banner_h = 18
    # Background
    c.setFillColor(ILD_BG)
    c.setStrokeColor(ILD_RED)
    c.setLineWidth(1.5)
    c.rect(30, banner_y, PAGE_W - 60, banner_h, stroke=1, fill=1)
    # Text
    c.setFillColor(ILD_RED)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(
        PAGE_W / 2, banner_y + 5,
        "ILD SIGNAL DETECTED — Interstitial Lung Disease suspected. Immediate clinical review required.",
    )
    # Left red stripe on the page
    c.setStrokeColor(ILD_RED)
    c.setLineWidth(3)
    c.line(12, 20, 12, PAGE_H - 20)


def _draw_non_serious_banner(c: canvas.Canvas):
    """Draw non-serious AE banner at top of page 1 (blue)."""
    banner_y = PAGE_H - 52
    banner_h = 18
    # Background
    c.setFillColor(NS_BG)
    c.setStrokeColor(NS_BLUE)
    c.setLineWidth(1.5)
    c.rect(30, banner_y, PAGE_W - 60, banner_h, stroke=1, fill=1)
    # Text
    c.setFillColor(NS_BLUE)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(
        PAGE_W / 2, banner_y + 5,
        "NON-SERIOUS AE \u2014 Internal tracking only. Not for regulatory submission.",
    )


def _draw_pi_review_marker(c: canvas.Canvas, x: float, y: float):
    """Draw [PI REVIEW] tag in orange at the specified position."""
    c.setFillColor(PI_ORANGE)
    c.setFont("Helvetica-Bold", 7)
    c.drawString(x, y, "[PI REVIEW]")


# ---------------------------------------------------------------------------
# Page overlay builders
# ---------------------------------------------------------------------------

def _build_page1_overlay(data: MedWatch3500A) -> bytes:
    """Build overlay for page 1: Sections A, B (dates/outcomes), C, E."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    a, b, sc, e = data.section_a, data.section_b, data.section_c, data.section_e

    # --- Mfr Report # in top-right header (FDA p.8 requirement) ---
    if data.section_g.report_number:
        _draw_text(c, 440, PAGE_H - 35, data.section_g.report_number,
                   font="Helvetica-Bold", size=7)

    # --- Banners (mutually exclusive: non-serious takes priority) ---
    if data.non_serious_flag:
        _draw_non_serious_banner(c)
    elif data.ild_flag:
        _draw_ild_banner(c)

    # --- Section A: Patient Information ---
    _draw_text(c, 32, 636, a.patient_id, size=8)
    if a.age:
        _draw_text(c, 106, 648, str(a.age))
        _draw_check(c, 132.8, 656)  # age_years checkbox
    # Sex
    sex = (a.sex or "").lower()
    if sex in ("male", "m"):
        _draw_check(c, 222.8, 624)
    elif sex in ("female", "f"):
        _draw_check(c, 222.8, 641)
    # Weight
    if a.weight:
        _draw_text(c, 271, 648, str(a.weight), size=FONT_SMALL)
        _draw_check(c, 273.8, 618)  # kg
    # A5a: Ethnicity (Hispanic / Not Hispanic)
    eth = (a.ethnicity or "").lower()
    if "not hispanic" in eth:
        _draw_check(c, 31.8, 571)
    elif "hispanic" in eth:
        _draw_check(c, 31.8, 584)

    # A5b: Race (separate field from ethnicity per FDA 3500A)
    race = (a.race or "").lower()
    if "asian" in race:
        _draw_check(c, 117.8, 593)
    if "black" in race or "african" in race:
        _draw_check(c, 117.8, 581)
    if "white" in race or "caucasian" in race:
        _draw_check(c, 221.8, 581)
    if "american indian" in race or "alaska" in race:
        _draw_check(c, 117.8, 605)
    if "native hawaiian" in race or "pacific islander" in race:
        _draw_check(c, 117.8, 569)

    # --- Section B: Adverse Event (outcomes & dates) ---
    # B1: Type = Adverse Event
    if b.report_type and "adverse" in b.report_type.lower():
        _draw_check(c, 42.8, 541)
    if "product" in (b.report_type or "").lower():
        _draw_check(c, 146.8, 541)

    # B2: Seriousness outcomes
    if b.seriousness_death:
        _draw_check(c, 32.8, 517)
    if b.seriousness_life_threatening:
        _draw_check(c, 32.8, 505)
    if b.seriousness_hospitalization:
        _draw_check(c, 32.8, 493)
    if b.seriousness_other:
        _draw_check(c, 32.8, 481)
    if b.seriousness_disability:
        _draw_check(c, 173.8, 505)
    if b.seriousness_congenital:
        _draw_check(c, 173.8, 493)

    # Death date
    if b.seriousness_death and b.death_date:
        dd, dm, dy = _date_parts(b.death_date)
        _draw_text(c, 167, 516, dd, size=FONT_SMALL)
        _draw_text(c, 191, 516, dm, size=FONT_SMALL)
        _draw_text(c, 224, 516, dy, size=FONT_SMALL)

    # B3: Event date
    ed, em, ey = _date_parts(b.onset_date)
    _draw_text(c, 47, 444, ed, size=FONT_SMALL)
    _draw_text(c, 71, 444, em, size=FONT_SMALL)
    _draw_text(c, 104, 444, ey, size=FONT_SMALL)

    # B4: Report date
    rd, rm, ry = _date_parts(b.report_date)
    _draw_text(c, 185, 444, rd, size=FONT_SMALL)
    _draw_text(c, 209, 444, rm, size=FONT_SMALL)
    _draw_text(c, 242, 444, ry, size=FONT_SMALL)

    # B5: Narrative (abbreviated — full text goes to page 3)
    narrative_plain = _strip_markdown(b.narrative)
    _draw_text_in_rect(c, [41, 368, 300, 430], narrative_plain,
                       font="Times-Roman", size=6.5, leading=7.5)
    # B6: Lab data (abbreviated)
    _draw_text_in_rect(c, [41, 298, 300, 346], b.lab_data,
                       font="Helvetica", size=6, leading=7)

    # B7: Medical history (abbreviated)
    _draw_text_in_rect(c, [41, 230, 300, 268], b.medical_history,
                       font="Helvetica", size=6, leading=7)

    # --- Section C: Suspect Product ---
    drug_name_info, manufacturer = _parse_drug_info(sc.drug_name)
    _draw_text_in_rect(c, [32, 174, 209, 186], drug_name_info, size=6.5)
    _draw_text_in_rect(c, [32, 152, 209, 164], manufacturer, size=6.5)

    # C1: Lot # (required for biologics per FDA guidance p.16)
    if sc.lot_number:
        _draw_text_in_rect(c, [209, 152, 301, 164], sc.lot_number, size=6.5)

    # Dose, Frequency, Route
    dose, freq, route = _parse_dose_freq_route(sc.dose_frequency_route)
    _draw_text_in_rect(c, [330, 671, 411, 689], dose, size=6.5)
    _draw_text_in_rect(c, [421, 671, 469, 689], freq, size=6.5)
    _draw_text_in_rect(c, [479, 671, 578, 689], route, size=6.5)

    # Therapy dates
    therapy_str = _fmt_date(sc.therapy_start)
    if sc.therapy_end:
        therapy_str += f" to {_fmt_date(sc.therapy_end)}"
    _draw_text_in_rect(c, [327, 613, 475, 625], therapy_str, size=6)

    # Diagnosis
    _draw_text_in_rect(c, [328, 568, 475, 586], sc.indication, size=6.5)

    # Expiration date
    xd, xm, xy = _date_parts(sc.expiry_date)
    _draw_text(c, 328, 470, xd, size=FONT_SMALL)
    _draw_text(c, 352, 470, xm, size=FONT_SMALL)
    _draw_text(c, 385, 470, xy, size=FONT_SMALL)

    # Dechallenge (C5 in 10/15 = "Event Abated?")
    dc_answer = _parse_dechallenge_answer(sc.dechallenge)
    if dc_answer == "yes":
        _draw_check(c, 495.8, 611)
    elif dc_answer == "no":
        _draw_check(c, 523.8, 611)
    else:
        _draw_check(c, 548.8, 611)
    # Rechallenge (C8 = "Event Reappeared?")
    rc_answer = _parse_dechallenge_answer(sc.rechallenge)
    if rc_answer == "yes":
        _draw_check(c, 495.8, 542)
    elif rc_answer == "no":
        _draw_check(c, 523.8, 542)
    else:
        _draw_check(c, 548.8, 542)
    # Concomitant Products
    _draw_text_in_rect(c, [32, 61, 301, 94], sc.concomitant_meds,
                       font="Helvetica", size=6, leading=7)

    # --- Section E: Initial Reporter ---
    # Parse name
    name = e.reporter_name or ""
    name_parts = name.rsplit(" ", 1)
    if len(name_parts) == 2:
        _draw_text(c, 352, 106, name_parts[1], size=FONT_SMALL)  # Last
        _draw_text(c, 496, 106, name_parts[0], size=FONT_SMALL)  # First
    elif name:
        _draw_text(c, 352, 106, name, size=FONT_SMALL)

    # Address parsing (simplified)
    addr = e.reporter_address or ""
    addr_parts = [p.strip() for p in addr.split(",")]
    if len(addr_parts) >= 1:
        _draw_text(c, 345, 93, addr_parts[0], size=FONT_SMALL)
    if len(addr_parts) >= 2:
        _draw_text(c, 330, 80, addr_parts[1], size=FONT_SMALL)  # City
    if len(addr_parts) >= 3:
        # State + ZIP
        state_zip = addr_parts[2].strip()
        sz_parts = state_zip.rsplit(" ", 1)
        if len(sz_parts) == 2:
            _draw_text(c, 514, 80, sz_parts[0], size=FONT_SMALL)
            _draw_text(c, 514, 67, sz_parts[1], size=FONT_SMALL)
        else:
            _draw_text(c, 514, 80, state_zip, size=FONT_SMALL)

    # Phone / Email below address
    if e.reporter_phone:
        _draw_text(c, 345, 58, f"Phone: {e.reporter_phone}", size=FONT_SMALL)
    if e.reporter_email:
        _draw_text(c, 445, 58, f"Email: {e.reporter_email}", size=FONT_SMALL)

    # Health Professional
    qual = (e.reporter_qualification or "").lower()
    if qual and qual != "no":
        _draw_check(c, 320.8, 24.2)  # Yes
    else:
        _draw_check(c, 352.8, 24.2)  # No

    # Reported to FDA
    fda = (e.reported_to_fda or "").lower()
    if fda == "yes":
        _draw_check(c, 501.8, 24.2)
    elif fda == "no":
        _draw_check(c, 530.8, 24.2)
    else:
        _draw_check(c, 555.8, 24.2)  # Unk

    c.save()
    return buf.getvalue()


def _build_page2_overlay(data: MedWatch3500A) -> bytes:
    """Build overlay for page 2: Section G (All Manufacturers)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    g = data.section_g

    # --- Section G: All Manufacturers ---
    # G1: Contact office name
    _draw_text_in_rect(c, [26, 326, 215, 338], g.sponsor_contact, size=6.5)

    # G2: Report Source checkboxes
    source = (g.source or "").lower()
    if "study" in source:
        _draw_check(c, 222, 291.6, size=9)
    elif "foreign" in source:
        _draw_check(c, 222, 303.6, size=9)
    elif "literature" in source:
        _draw_check(c, 222, 279.6, size=9)
    elif "consumer" in source:
        _draw_check(c, 222, 267.6, size=9)
    elif "health" in source:
        _draw_check(c, 222, 255.6, size=9)

    # G3: Date received
    dd, dm, dy = _date_parts(g.awareness_date)
    _draw_text(c, 35, 215, dd, size=FONT_SMALL)
    _draw_text(c, 59, 215, dm, size=FONT_SMALL)
    _draw_text(c, 92, 215, dy, size=FONT_SMALL)

    # G4/G5: IND number
    if g.ind_type and "ind" in g.ind_type.lower():
        _draw_text(c, 171, 202, g.ind_number, size=FONT_SMALL)

    # G6: Type of report
    report_type = (g.initial_followup or "").lower()
    if "initial" in report_type:
        _draw_check(c, 68, 130.9, size=9)  # Initial
    elif "follow" in report_type:
        _draw_check(c, 68, 120, size=9)  # Follow-up
    # Also check 15-day for SAE reports
    _draw_check(c, 28, 118.9, size=9)  # 15-day

    # G7: AE term
    _draw_text_in_rect(c, [138, 72, 297, 103], g.ae_term, size=7)

    # G8: Manufacturer Report Number
    _draw_text_in_rect(c, [28, 84, 132, 103], g.report_number, size=6.5)

    c.save()
    return buf.getvalue()


def _build_page3_overlay(data: MedWatch3500A) -> bytes:
    """Build overlay for page 3: Continuation (full narrative, labs, history)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=LETTER)
    b, sc = data.section_b, data.section_c

    # ILD left stripe on continuation page too
    if data.ild_flag:
        c.setStrokeColor(ILD_RED)
        c.setLineWidth(3)
        c.line(12, 20, 12, PAGE_H - 20)

    # B5: Full narrative
    narrative = _strip_markdown(b.narrative)
    _draw_text_in_rect(c, [36, 474, 589, 680], narrative,
                       font="Times-Roman", size=7.5, leading=9)

    # B6: Relevant tests/lab data
    _draw_text_in_rect(c, [36, 358, 589, 458], b.lab_data,
                       font="Helvetica", size=7, leading=8.5)

    # B7: Other relevant medical history
    _draw_text_in_rect(c, [36, 242, 589, 342], b.medical_history,
                       font="Helvetica", size=7, leading=8.5)

    # Concomitant medications
    _draw_text_in_rect(c, [36, 127, 589, 226], sc.concomitant_meds,
                       font="Helvetica", size=7, leading=8.5)

    # Additional info (ILD details if applicable)
    if data.ild_flag:
        ild_note = (
            "ILD SIGNAL: Sentinel Agent flagged Interstitial Lung Disease. "
            "This case requires immediate PI review and potential SAE expedited reporting."
        )
        _draw_text_in_rect(c, [36, 72, 589, 111], ild_note,
                           font="Helvetica-Bold", size=7, leading=8.5)

    c.save()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def _merge_overlay(template_page, overlay_bytes: bytes):
    """Merge a ReportLab overlay onto a template page."""
    overlay_reader = PdfReader(io.BytesIO(overlay_bytes))
    overlay_page = overlay_reader.pages[0]
    template_page.merge_page(overlay_page)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_medwatch_pdf(
    data: MedWatch3500A,
    output_path: Optional[str] = None,
) -> bytes:
    """Generate a MedWatch FDA 3500A PDF by filling the official template.

    Uses the FDA Form 3500A (10/15) as background and overlays data values
    at the correct field positions using ReportLab.

    Args:
        data: Validated MedWatch3500A Pydantic model.
        output_path: If provided, also writes the PDF to this file path.

    Returns:
        PDF contents as bytes.
    """
    # Read template
    template = PdfReader(str(TEMPLATE_PATH))
    writer = PdfWriter()

    # Build overlays for each page
    overlay1 = _build_page1_overlay(data)
    overlay2 = _build_page2_overlay(data)
    overlay3 = _build_page3_overlay(data)

    # Page 1: Sections A, B, C, E
    page1 = template.pages[0]
    _merge_overlay(page1, overlay1)
    writer.add_page(page1)

    # Page 2: Section G (manufacturer info)
    page2 = template.pages[1]
    _merge_overlay(page2, overlay2)
    writer.add_page(page2)

    # Page 3: Continuation (full text areas)
    page3 = template.pages[2]
    _merge_overlay(page3, overlay3)
    writer.add_page(page3)

    # Write to bytes
    buf = io.BytesIO()
    writer.write(buf)
    pdf_bytes = buf.getvalue()
    buf.close()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes


def generate_medwatch_pdf_from_dict(
    data: dict,
    output_path: Optional[str] = None,
) -> bytes:
    """Generate a MedWatch FDA 3500A PDF from a raw dict.

    Convenience wrapper that validates the dict through MedWatch3500A first.

    Args:
        data: Dict matching the MedWatch3500A schema.
        output_path: If provided, also writes the PDF to this file path.

    Returns:
        PDF contents as bytes.
    """
    model = MedWatch3500A.model_validate(data)
    return generate_medwatch_pdf(model, output_path)
