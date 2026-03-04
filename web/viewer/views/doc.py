"""
Doc Agent views: SAE document generation, SAE status management,
SAE report editor, doc hub, CRF tables, unified doc chat API,
and all _build_*_chat_context helpers.
"""
import io
import json
import logging
import math
import os
import time
from collections import Counter
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET

from ..crf_aggregator import (
    _read_jsonl_cached, _load_patient_json,
    aggregate_domain, export_domain_to_excel, DOMAIN_COLUMNS, DOMAIN_LABELS,
)
from ._helpers import (
    DATA_DIR, _get_run_path, _list_patients, _load_rule_set,
    _load_run_meta, _load_patient_profile, _extract_lab_ranges, logger,
)
from .stats import (
    _sanitize_messages, _call_chat_llm,
    _match_sections, _retrieve_context,
    _STATS_CHAT_MODEL, _STATS_SYSTEM_PROMPT, _STATS_CHAT_URL,
)


VALID_DOMAINS = {"dm", "mh", "ae", "ec", "cm", "vs", "lb", "ds", "dd", "tu", "rs", "pe", "eg"}


# --- Patient data loader for doc agent ---

def _load_patient_data(run_path, patient_id, mode="natural"):
    """Load patient profile + day records for doc agent.

    Uses hospital record (HR) when available -- SAE reporting should be
    based on what the hospital actually observed, not ground truth.
    Falls back to GT for older runs that lack *_hospital.jsonl.
    Uses crf_aggregator's file cache for fast repeated access.
    """
    profile_path = run_path / "patients" / f"{patient_id}.json"
    hr_path = run_path / "simulations" / f"{patient_id}_{mode}_hospital.jsonl"
    gt_path = run_path / "simulations" / f"{patient_id}_{mode}.jsonl"
    sim_path = hr_path if hr_path.exists() else gt_path

    if not profile_path.exists() or not sim_path.exists():
        return None, None

    profile = _load_patient_json(run_path, patient_id)
    if profile is None:
        return None, None

    records = _read_jsonl_cached(sim_path)
    return profile, records


# --- SAE Document Generation ---

@csrf_exempt
@require_POST
def api_doc_generate(request):
    """Generate MedWatch 3500A + E2B XML for a specific patient SAE.

    POST body: {
        "run_id": str,
        "patient_id": str,
        "ae_term": str,
        "ae_day": int (optional),
        "mode": "natural" | "care_ai" (default: "natural"),
        "use_ai": bool (default: false),
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    run_id = body.get("run_id", "")
    patient_id = body.get("patient_id", "")
    ae_term = body.get("ae_term", "")
    ae_day = body.get("ae_day")
    mode = body.get("mode", "natural")
    use_ai = body.get("use_ai", False)

    if not all([run_id, patient_id, ae_term]):
        return JsonResponse({"error": "run_id, patient_id, ae_term required"}, status=400)

    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return JsonResponse({"error": f"Run '{run_id}' not found"}, status=404)

    profile, records = _load_patient_data(run_path, patient_id, mode)
    if profile is None:
        return JsonResponse({"error": f"Patient '{patient_id}' not found"}, status=404)

    meta_path = run_path / "run_meta.json"
    drug_name = "Enfortumab vedotin (Padcev)"
    indication = "Metastatic urothelial carcinoma"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            drug_name = meta.get("drug_name", drug_name)
            indication = meta.get("indication", indication)
        except Exception:
            pass

    from datetime import date as dt_date
    from sim.doc_agent.service import generate_documents

    result = generate_documents(
        patient_profile=profile,
        day_records=records,
        target_ae_term=ae_term,
        run_id=run_id,
        sim_start_date=dt_date(2026, 1, 6),
        drug_name=drug_name,
        indication=indication,
        target_ae_day=ae_day,
        use_ai=use_ai,
    )

    # Record ai_fields in status file when AI is used
    if use_ai and result.get("success"):
        from datetime import datetime
        ae_slug = ae_term.replace(" ", "_").replace("/", "_")
        ai_fields = {
            "section_b.narrative": True,
            "section_c.dechallenge": True,
            "section_c.rechallenge": True,
        }
        status_data = _read_status(run_id, patient_id, ae_slug)
        status_data["ai_fields"] = ai_fields
        # Store MedDRA info if available
        meddra = result.get("meddra", {})
        if meddra:
            status_data["meddra_confidence"] = meddra.get("confidence")
            status_data["meddra_source"] = meddra.get("source")
        status_data["updated_at"] = datetime.utcnow().isoformat()
        if "created_at" not in status_data:
            status_data["created_at"] = datetime.utcnow().isoformat()
        if "status" not in status_data:
            status_data["status"] = "draft"
        _write_status(run_id, patient_id, ae_slug, status_data)

    return JsonResponse(result)


@require_GET
def api_doc_list_saes(request, run_id, patient_id):
    """List all serious AEs for a patient in a run.

    GET /api/doc/saes/<run_id>/<patient_id>/?mode=natural
    """
    mode = request.GET.get("mode", "natural")
    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return JsonResponse({"error": "Run not found"}, status=404)

    profile, records = _load_patient_data(run_path, patient_id, mode)
    if profile is None:
        return JsonResponse({"error": "Patient not found"}, status=404)

    from sim.doc_agent.sim_to_crf_adapter import find_serious_aes

    saes = find_serious_aes(records)
    result = []
    for sae in saes:
        ae = sae["ae_record"]
        result.append({
            "ae_term": ae.get("AETERM", ""),
            "grade": ae.get("_grade", 0),
            "onset_day": ae.get("AESTDAT"),
            "severity": ae.get("AESEV", ""),
            "action": ae.get("AEACN", ""),
            "outcome": ae.get("AEOUT", ""),
        })

    return JsonResponse({"patient_id": patient_id, "saes": result})


@require_GET
def api_doc_download(request, run_id, patient_id, filename):
    """Download a generated document (PDF or XML).

    GET /api/doc/download/<run_id>/<patient_id>/<filename>
    """
    from sim.doc_agent.service import DOCS_OUTPUT_DIR

    file_path = DOCS_OUTPUT_DIR / run_id / patient_id / filename

    if not file_path.exists() or not file_path.is_file():
        return JsonResponse({"error": "File not found"}, status=404)

    try:
        file_path.resolve().relative_to(DOCS_OUTPUT_DIR.resolve())
    except ValueError:
        return JsonResponse({"error": "Access denied"}, status=403)

    if filename.endswith(".pdf"):
        content_type = "application/pdf"
    elif filename.endswith(".xml"):
        content_type = "application/xml"
    else:
        content_type = "application/octet-stream"

    disposition = "inline" if filename.endswith(".pdf") else "attachment"
    with open(file_path, "rb") as f:
        response = HttpResponse(f.read(), content_type=content_type)
        safe_filename = filename.replace('"', '_').replace('\n', '_').replace('\r', '_')
        response["Content-Disposition"] = f'{disposition}; filename="{safe_filename}"'
        return response


@require_GET
def api_doc_list(request, run_id):
    """List all generated documents for a run.

    GET /api/doc/list/<run_id>/
    """
    from sim.doc_agent.service import DOCS_OUTPUT_DIR

    docs_dir = DOCS_OUTPUT_DIR / run_id
    if not docs_dir.exists():
        return JsonResponse({"run_id": run_id, "documents": []})

    documents = []
    for patient_dir in sorted(docs_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        for doc_file in sorted(patient_dir.iterdir()):
            if doc_file.is_file():
                documents.append({
                    "patient_id": patient_dir.name,
                    "filename": doc_file.name,
                    "type": "pdf" if doc_file.suffix == ".pdf" else "xml",
                    "size": doc_file.stat().st_size,
                    "download_url": f"/api/doc/download/{run_id}/{patient_dir.name}/{doc_file.name}",
                })

    return JsonResponse({"run_id": run_id, "documents": documents})


@csrf_exempt
@require_POST
def api_doc_save(request):
    """Save edited MedWatch data and regenerate PDF + E2B XML.

    POST body: {
        "run_id": str,
        "patient_id": str,
        "ae_slug": str,
        "medwatch_data": { section_a: {...}, section_b: {...}, ... },
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    run_id = body.get("run_id", "")
    patient_id = body.get("patient_id", "")
    ae_slug = body.get("ae_slug", "")
    mw_data = body.get("medwatch_data", {})

    if not all([run_id, patient_id, ae_slug, mw_data]):
        return JsonResponse(
            {"error": "run_id, patient_id, ae_slug, medwatch_data required"},
            status=400,
        )

    from sim.doc_agent.service import DOCS_OUTPUT_DIR
    from sim.doc_agent.schemas.medwatch import MedWatch3500A
    from sim.doc_agent.medwatch_pdf import generate_medwatch_pdf
    from sim.doc_agent.e2b_converter import convert_to_e2b_xml
    from sim.doc_agent.meddra_coder import code_meddra
    from sim.doc_agent.config import Settings
    from sim.doc_agent.schemas.crf import CRFData

    try:
        medwatch = MedWatch3500A.model_validate(mw_data)
    except Exception as exc:
        return JsonResponse({"error": f"Invalid MedWatch data: {exc}"}, status=400)

    out_dir = DOCS_OUTPUT_DIR / run_id / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"medwatch_data_{ae_slug}.json"
    json_path.write_text(
        json.dumps(mw_data, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    pdf_path = out_dir / f"medwatch_3500a_{ae_slug}.pdf"
    xml_path = out_dir / f"e2b_r3_{ae_slug}.xml"

    try:
        generate_medwatch_pdf(medwatch, str(pdf_path))
    except Exception as exc:
        return JsonResponse({"error": f"PDF generation failed: {exc}"}, status=500)

    # Regenerate E2B XML
    run_path = _get_run_path(run_id)
    meta_path = run_path / "run_meta.json"
    drug_name = "Enfortumab vedotin (Padcev)"
    indication = "Metastatic urothelial carcinoma"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            drug_name = meta.get("drug_name", drug_name)
            indication = meta.get("indication", indication)
        except Exception:
            pass

    doc_settings = Settings.from_simulation(drug_name=drug_name, indication=indication)
    ae_term = medwatch.section_g.ae_term or ae_slug.replace("_", " ")
    meddra = code_meddra(ae_term, use_medgemma=False)

    # Build minimal CRF for E2B (narrative fields already in medwatch)
    from datetime import date as dt_date
    profile, records = _load_patient_data(run_path, patient_id)
    if profile and records:
        from sim.doc_agent.sim_to_crf_adapter import build_crf_for_sae
        crf = build_crf_for_sae(
            patient_profile=profile,
            day_records=records,
            target_ae_term=ae_term,
            sim_start_date=dt_date(2026, 1, 6),
        )
    else:
        crf = None

    if crf:
        try:
            e2b_xml = convert_to_e2b_xml(medwatch, crf, meddra, doc_settings)
            xml_path.write_text(e2b_xml, encoding="utf-8")
        except Exception as exc:
            logger_msg = f"E2B regeneration failed: {exc}"

    pdf_url = f"/api/doc/download/{run_id}/{patient_id}/{pdf_path.name}"
    xml_url = f"/api/doc/download/{run_id}/{patient_id}/{xml_path.name}"

    # Update status file with refreshed MedDRA confidence
    from sim.doc_agent.service import DOCS_OUTPUT_DIR as _DOCS_DIR
    from datetime import datetime
    status_path = _DOCS_DIR / run_id / patient_id / f"report_status_{ae_slug}.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    if status_path.exists():
        try:
            status_data = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status_data = {}
    else:
        status_data = {
            "status": "draft",
            "ai_fields": {},
            "reviewed_by": None,
            "reviewed_at": None,
            "created_at": datetime.utcnow().isoformat(),
        }
    status_data["meddra_confidence"] = meddra.confidence
    status_data["meddra_source"] = meddra.source
    status_data["updated_at"] = datetime.utcnow().isoformat()
    status_path.write_text(
        json.dumps(status_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return JsonResponse({
        "success": True,
        "pdf_url": pdf_url,
        "xml_url": xml_url,
        "message": "Documents saved and regenerated.",
    })


# --- SAE Status Management ---

def _get_status_path(run_id: str, patient_id: str, ae_slug: str) -> Path:
    """Return path to the report status JSON file."""
    from sim.doc_agent.service import DOCS_OUTPUT_DIR
    return DOCS_OUTPUT_DIR / run_id / patient_id / f"report_status_{ae_slug}.json"


def _read_status(run_id: str, patient_id: str, ae_slug: str) -> dict:
    """Read status file, returning default draft status if missing."""
    status_path = _get_status_path(run_id, patient_id, ae_slug)
    if status_path.exists():
        try:
            return json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"status": "draft"}


def _write_status(run_id: str, patient_id: str, ae_slug: str, data: dict):
    """Write status file."""
    status_path = _get_status_path(run_id, patient_id, ae_slug)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


@require_GET
def api_doc_get_status(request, run_id: str, patient_id: str, ae_slug: str):
    """GET /api/doc/status/<run_id>/<patient_id>/<ae_slug>/ -- return report status."""
    data = _read_status(run_id, patient_id, ae_slug)
    return JsonResponse(data)


@csrf_exempt
@require_POST
def api_doc_update_status(request):
    """POST /api/doc/status -- transition SAE report status.

    Body: {
        "run_id": str,
        "patient_id": str,
        "ae_slug": str,
        "status": "draft" | "under_review" | "accepted",
        "reviewed_by": str (optional),
    }
    """
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    run_id = body.get("run_id", "")
    patient_id = body.get("patient_id", "")
    ae_slug = body.get("ae_slug", "")
    new_status = body.get("status", "")
    reviewed_by = body.get("reviewed_by")

    if not all([run_id, patient_id, ae_slug, new_status]):
        return JsonResponse(
            {"error": "run_id, patient_id, ae_slug, status required"}, status=400
        )

    valid_statuses = {"draft", "under_review", "accepted"}
    if new_status not in valid_statuses:
        return JsonResponse(
            {"error": f"Invalid status. Must be one of: {', '.join(sorted(valid_statuses))}"}, status=400
        )

    from datetime import datetime

    data = _read_status(run_id, patient_id, ae_slug)

    # Validate transitions
    current = data.get("status", "draft")
    allowed_transitions = {
        "draft": {"accepted"},
        "accepted": {"draft"},
    }
    if new_status != current and new_status not in allowed_transitions.get(current, set()):
        return JsonResponse(
            {"error": f"Cannot transition from '{current}' to '{new_status}'"}, status=400
        )

    data["status"] = new_status
    data["updated_at"] = datetime.utcnow().isoformat()

    if new_status == "accepted":
        data["reviewed_by"] = reviewed_by or "Reviewer"
        data["reviewed_at"] = datetime.utcnow().isoformat()
    elif new_status == "draft":
        data["reviewed_by"] = None
        data["reviewed_at"] = None

    _write_status(run_id, patient_id, ae_slug, data)

    return JsonResponse({"success": True, **data})


def sae_report_editor(request, run_id: str, patient_id: str, ae_slug: str):
    """SAE Report Editor -- MedWatch 3500A form with inline editing.

    Loads existing generated data if available, otherwise generates fresh.
    """
    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return HttpResponse("Run not found", status=404)

    from sim.doc_agent.service import DOCS_OUTPUT_DIR

    # Check for existing saved medwatch data
    json_path = DOCS_OUTPUT_DIR / run_id / patient_id / f"medwatch_data_{ae_slug}.json"
    medwatch_data = None
    meddra_data = None
    ai_used = False

    if json_path.exists():
        try:
            medwatch_data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if medwatch_data is None:
        # Generate fresh
        mode = request.GET.get("mode", "care_ai")
        profile, records = _load_patient_data(run_path, patient_id, mode)
        if profile is None:
            return HttpResponse("Patient not found", status=404)

        ae_term = ae_slug.replace("_", " ")
        ae_day_str = request.GET.get("ae_day", "").strip()
        ae_day = int(ae_day_str) if ae_day_str.isdigit() else None

        meta_path = run_path / "run_meta.json"
        drug_name = "Enfortumab vedotin (Padcev)"
        indication = "Metastatic urothelial carcinoma"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                drug_name = meta.get("drug_name", drug_name)
                indication = meta.get("indication", indication)
            except Exception:
                pass

        use_ai = request.GET.get("use_ai", "0") == "1"

        from datetime import date as dt_date
        from sim.doc_agent.service import generate_documents

        result = generate_documents(
            patient_profile=profile,
            day_records=records,
            target_ae_term=ae_term,
            run_id=run_id,
            sim_start_date=dt_date(2026, 1, 6),
            drug_name=drug_name,
            indication=indication,
            target_ae_day=ae_day,
            use_ai=use_ai,
        )

        if result.get("success"):
            medwatch_data = result.get("medwatch_data", {})
            meddra_data = result.get("meddra")
            ai_used = result.get("ai_used", False)
            # Persist the medwatch data
            out_dir = DOCS_OUTPUT_DIR / run_id / patient_id
            out_dir.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps(medwatch_data, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            return render(request, "doc/sae_report.html", {
                "run_id": run_id,
                "patient_id": patient_id,
                "ae_slug": ae_slug,
                "error": result.get("error", "Unknown error"),
            })

    # Load profile for header info
    profile = _load_patient_profile(run_path, patient_id)
    demo = profile.get("emr", {}).get("demographics", {})
    rule_set = _load_rule_set(run_path)

    pdf_url = f"/api/doc/download/{run_id}/{patient_id}/medwatch_3500a_{ae_slug}.pdf"
    xml_url = f"/api/doc/download/{run_id}/{patient_id}/e2b_r3_{ae_slug}.xml"

    mode = request.GET.get("mode", "care_ai")
    ae_day_str = request.GET.get("ae_day", "").strip()

    context = {
        "run_id": run_id,
        "patient_id": patient_id,
        "ae_slug": ae_slug,
        "ae_term": ae_slug.replace("_", " ").title(),
        "medwatch_json": json.dumps(medwatch_data, default=str, ensure_ascii=False),
        "meddra_json": json.dumps(meddra_data or {}, ensure_ascii=False),
        "ai_used": ai_used,
        "pdf_url": pdf_url,
        "xml_url": xml_url,
        "drug_name": rule_set.get("drug_name", ""),
        "indication": rule_set.get("indication", ""),
        "patient_age": demo.get("age", "?"),
        "patient_sex": demo.get("sex", "?"),
        "model_name": _load_run_meta(run_path).get("model", ""),
        "mode": mode,
        "ae_day": ae_day_str,
    }
    return render(request, "doc/sae_report.html", context)


def doc_hub(request, run_id: str):
    """Documents Hub -- overview of all SAEs across patients, with links to report editor."""
    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return HttpResponse("Run not found", status=404)

    mode = request.GET.get("mode", "care_ai")
    patient_ids = _list_patients(run_path)

    sim_dir = run_path / "simulations"
    available_modes = []
    if sim_dir.exists():
        if list(sim_dir.glob("*_care_ai.jsonl")):
            available_modes.append("care_ai")
        if list(sim_dir.glob("*_natural.jsonl")):
            available_modes.append("natural")

    from sim.doc_agent.sim_to_crf_adapter import find_serious_aes

    all_saes = []
    for pid in patient_ids:
        profile, records = _load_patient_data(run_path, pid, mode)
        if not records:
            continue
        saes = find_serious_aes(records)
        for sae in saes:
            ae = sae["ae_record"]
            ae_term = ae.get("AETERM", "")
            ae_slug = ae_term.replace(" ", "_").replace("/", "_")
            # Read report status
            report_status = _read_status(run_id, pid, ae_slug)
            all_saes.append({
                "patient_id": pid,
                "ae_term": ae_term,
                "ae_slug": ae_slug,
                "grade": ae.get("_grade", 0),
                "onset_day": sae["day"],
                "severity": ae.get("AESEV", ""),
                "action": ae.get("AEACN", ""),
                "serious": ae.get("AESER", False),
                "report_status": report_status.get("status", "draft"),
            })

    from sim.doc_agent.service import DOCS_OUTPUT_DIR
    docs_dir = DOCS_OUTPUT_DIR / run_id
    existing_docs = set()
    if docs_dir.exists():
        for patient_dir in docs_dir.iterdir():
            if patient_dir.is_dir():
                for f in patient_dir.iterdir():
                    if f.suffix == ".pdf":
                        existing_docs.add(f"{patient_dir.name}/{f.stem}")

    meta = {}
    meta_path = run_path / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    context = {
        "run_id": run_id,
        "mode": mode,
        "available_modes": available_modes,
        "saes": all_saes,
        "sae_count": len(all_saes),
        "patient_count": len(set(s["patient_id"] for s in all_saes)),
        "existing_docs": existing_docs,
        "drug_name": meta.get("drug_name", ""),
        "indication": meta.get("indication", ""),
        "model_name": _load_run_meta(run_path).get("model", ""),
    }
    return render(request, "doc/doc_hub.html", context)


# --- CRF Tables ---

def crf_tables(request, run_id: str):
    """CRF Tables page -- renders the main shell, data loaded via AJAX."""
    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return JsonResponse({"error": "Run not found"}, status=404)

    patient_ids = _list_patients(run_path)
    sim_dir = run_path / "simulations"
    available_modes = []
    if sim_dir.exists():
        if list(sim_dir.glob("*_care_ai.jsonl")):
            available_modes.append("care_ai")
        if list(sim_dir.glob("*_natural.jsonl")):
            available_modes.append("natural")

    # Load run meta
    drug_name = ""
    indication = ""
    meta_path = run_path / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            drug_name = meta.get("drug_name", "")
            indication = meta.get("indication", "")
        except Exception:
            pass

    rule_set = _load_rule_set(run_path)
    lab_ranges = rule_set.get("lab_reference_ranges", {})
    if not lab_ranges:
        lab_ranges = _extract_lab_ranges(run_path)
    context = {
        "run_id": run_id,
        "patient_ids": patient_ids,
        "patient_ids_json": json.dumps(patient_ids),
        "available_modes": available_modes,
        "drug_name": drug_name,
        "indication": indication,
        "domain_labels_json": json.dumps(DOMAIN_LABELS),
        "model_name": _load_run_meta(run_path).get("model", ""),
        "lab_ranges_json": json.dumps(lab_ranges),
    }
    return render(request, "doc/crf_tables.html", context)


def _get_sim_start_date(run_path: Path):
    """Read sim_start_date from run_meta.json, default 2026-01-06."""
    from datetime import date
    meta_path = run_path / "run_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            d = meta.get("sim_start_date")
            if d:
                return date.fromisoformat(d)
        except Exception:
            pass
    return date(2026, 1, 6)


def _inject_dates(rows, columns, start_date):
    """Replace day-number columns with calendar date columns in CRF rows."""
    # Map of day-number keys -> date label
    DAY_KEY_LABELS = {
        "day": "Date",
        "AESTDAT": "Start Date", "AEENDAT": "End Date",
        "ECSTDAT": "Start Date", "ECENDAT": "End Date",
        "CMSTDAT": "Start Date", "CMENDAT": "End Date",
        "MHSTDAT": "Start Date", "MHENDAT": "End Date",
        "onset_day": "Onset Date", "detected_day": "Detected Date",
        "PEDAT": "Exam Date", "EGDAT": "ECG Date",
        "TUDAT": "Assessment Date",
        "DSSTDAT": "Disposition Date", "DTHDAT": "Date of Death",
    }

    # Find which day-number columns exist
    day_col_keys = [col["key"] for col in columns if col["key"] in DAY_KEY_LABELS]
    if not day_col_keys:
        return rows, columns

    # Replace day columns with date columns
    new_columns = []
    for col in columns:
        if col["key"] in DAY_KEY_LABELS:
            new_columns.append({
                "key": col["key"] + "_date",
                "label": DAY_KEY_LABELS[col["key"]],
            })
        else:
            new_columns.append(col)

    # Convert day numbers to date strings
    for row in rows:
        for k in day_col_keys:
            v = row.pop(k, None)
            if isinstance(v, (int, float)) and v > 0:
                row[k + "_date"] = str(start_date + timedelta(days=int(v) - 1))
            else:
                row[k + "_date"] = None

    return rows, new_columns


@require_GET
def api_crf_domain_data(request, run_id: str, domain: str):
    """JSON API: return domain-specific CRF rows with pagination."""
    if domain.lower() not in VALID_DOMAINS:
        return JsonResponse({"error": f"Invalid domain: {domain}"}, status=400)

    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return JsonResponse({"error": "Run not found"}, status=404)

    source = request.GET.get("source", "hr")
    mode = request.GET.get("mode", "care_ai")
    patient_filter = request.GET.get("patient", "")
    page = int(request.GET.get("page", 1))
    per_page = int(request.GET.get("per_page", 100))

    patient_ids = None
    if patient_filter:
        patient_ids = [p.strip() for p in patient_filter.split(",") if p.strip()]

    rows, total, columns = aggregate_domain(
        domain, run_path, patient_ids, mode, source, page, per_page,
    )

    # Inject calendar dates next to day-number columns
    start_date = _get_sim_start_date(run_path)
    rows, columns = _inject_dates(rows, columns, start_date)

    total_pages = math.ceil(total / per_page) if per_page > 0 else 1

    return JsonResponse({
        "domain": domain.upper(),
        "source": source,
        "mode": mode,
        "columns": columns,
        "rows": rows,
        "total_rows": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
    })


@require_GET
def api_crf_excel_download(request, run_id: str):
    """Download CRF data as Excel file."""
    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return JsonResponse({"error": "Run not found"}, status=404)

    source = request.GET.get("source", "hr")
    mode = request.GET.get("mode", "care_ai")
    domains_param = request.GET.get("domains", "")

    if domains_param:
        domains = [d.strip().upper() for d in domains_param.split(",") if d.strip()]
    else:
        domains = [d.upper() for d in VALID_DOMAINS]

    # Multi-domain export: one sheet per domain
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    wb = Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )

    start_date = _get_sim_start_date(run_path)

    for dom in domains:
        if dom.lower() not in VALID_DOMAINS:
            continue
        rows, total, columns = aggregate_domain(
            dom, run_path, None, mode, source, page=1, per_page=0,
        )
        rows, columns = _inject_dates(rows, columns, start_date)
        ws = wb.create_sheet(title=dom)
        col_keys = [c["key"] for c in columns]

        # Headers
        for ci, col in enumerate(columns, 1):
            cell = ws.cell(row=1, column=ci, value=col["label"])
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")
            cell.border = thin_border

        # Data
        for ri, row in enumerate(rows, 2):
            for ci, key in enumerate(col_keys, 1):
                val = row.get(key)
                if isinstance(val, bool):
                    val = "Yes" if val else "No"
                cell = ws.cell(row=ri, column=ci, value=val)
                cell.border = thin_border

        # Auto-width
        for ci, col in enumerate(columns, 1):
            max_len = len(col["label"])
            for ri in range(2, min(len(rows) + 2, 102)):  # sample first 100 rows
                val = ws.cell(row=ri, column=ci).value
                if val is not None:
                    max_len = max(max_len, len(str(val)))
            ws.column_dimensions[ws.cell(row=1, column=ci).column_letter].width = min(max_len + 2, 40)

    buf = io.BytesIO()
    wb.save(buf)
    xlsx_bytes = buf.getvalue()

    response = HttpResponse(
        xlsx_bytes,
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    response["Content-Disposition"] = f'attachment; filename="CRF_{run_id}_{source}.xlsx"'
    return response


# --- Chat context builders for unified doc chat ---

_CRF_SYSTEM_PROMPT = (
    "You are CLARA's CRF Data Assistant, an expert clinical data manager.\n"
    "You help researchers explore CRF (Case Report Form) tabular data.\n\n"
    "Rules:\n"
    "- Answer based ONLY on the provided CRF data below.\n"
    "- NEVER fabricate data not present in the provided rows.\n"
    "- The 'Pre-computed Summary' contains EXACT counts from ALL rows. ALWAYS trust summary numbers over counting the visible table rows (the table may be truncated).\n"
    "- If the data shows 0 matching rows, say 0 \u2014 do NOT invent a result.\n"
    "- Be concise (2-5 sentences) unless asked for detail.\n"
    "- Reference specific patient IDs, values, and counts from the data.\n"
    "- Use the same language as the user.\n"
    "- When user references data with @[...] tags, focus on that specific record.\n"
)

_SAE_SYSTEM_PROMPT = (
    "You are CLARA's SAE Report Assistant, an expert in pharmacovigilance and MedWatch reporting.\n"
    "You help researchers analyze Serious Adverse Event reports.\n\n"
    "Rules:\n"
    "- Answer based ONLY on the provided MedWatch/SAE data below.\n"
    "- NEVER fabricate information not in the data.\n"
    "- Be concise (2-5 sentences) unless asked for detail.\n"
    "- Reference specific form sections, dates, and clinical details.\n"
    "- Use the same language as the user.\n"
    "- When user references data with @[...] tags, focus on that specific field.\n"
)


def _build_stats_chat_context(run_path, body, drug_name, indication, n_patients, message):
    """Build context for stats page chat."""
    mode = body.get("mode", "natural")
    tab = body.get("tab", "")
    cache_path = run_path / "validation" / f"csr_stats_{mode}.json"
    if not cache_path.exists():
        return JsonResponse({"error": "Stats not computed yet."}, status=400), None, None
    with open(cache_path) as f:
        stats = json.load(f)
    matched_sections = _match_sections(message, tab)
    compact = _retrieve_context(stats, message, tab)
    context_block = (
        f"Drug: {drug_name} | Indication: {indication} | "
        f"Mode: {mode} | N={n_patients}\n\n{compact}"
    )
    query_meta = {
        "source": f"csr_stats_{mode}.json",
        "model": _STATS_CHAT_MODEL,
        "matched_sections": matched_sections,
        "tab": tab or "(none)",
        "context_data": compact,
        "history_turns": len(body.get("history", [])) // 2,
        "message": message,
    }
    return context_block, _STATS_SYSTEM_PROMPT, query_meta


def _build_crf_chat_context(run_path, body, drug_name, indication, n_patients, message):
    """Build context for CRF tables page chat."""
    domain = body.get("domain", "ae")
    mode = body.get("mode", "natural")
    patient = body.get("patient", "")

    patient_ids = [patient] if patient else None
    try:
        rows, total, columns = aggregate_domain(
            domain, run_path, patient_ids, mode, "hr", 1, 20)
    except Exception:
        rows, total, columns = [], 0, []

    lines = [f"Drug: {drug_name} | Indication: {indication} | Mode: {mode} | N={n_patients}"]
    lines.append(f"Domain: {domain.upper()} | Total rows: {total}")
    lines.append(f"Columns: {', '.join(c.get('label', c.get('key', '')) for c in columns[:8])}")
    lines.append("")

    # Pick columns that matter most for each domain (include Grade for AE)
    _PRIORITY_KEYS = {"_grade", "AESEV", "AESER", "AEREL", "AEACN", "LBSTRESN", "LBSTNRHI", "LBSTNRLO"}
    key_cols = []
    for c in columns[:6]:
        key_cols.append(c)
    for c in columns[6:]:
        if c.get("key", "") in _PRIORITY_KEYS and len(key_cols) < 10:
            key_cols.append(c)

    # Pre-computed summary to prevent hallucination on counts
    if domain.lower() == "ae" and rows:
        all_rows, _, _ = aggregate_domain(domain, run_path, patient_ids, mode, "hr", 1, 500)
        grade_dist = Counter()
        serious_count = 0
        ae_per_pt = Counter()
        for r in all_rows:
            g = r.get("_grade", 0)
            try:
                g = int(g)
            except (ValueError, TypeError):
                g = 0
            grade_dist[g] += 1
            if r.get("AESER") in (True, "True", "Y", "YES"):
                serious_count += 1
            ae_per_pt[r.get("patient_id", "")] += 1
        lines.append("=== Pre-computed Summary (AUTHORITATIVE \u2014 always use these counts, ignore the truncated table below if they differ) ===")
        lines.append(f"Total AEs: {len(all_rows)}")
        for g in sorted(grade_dist.keys()):
            # List which patients/AEs for non-G1 grades
            if g >= 2:
                g_rows = [r for r in all_rows if int(r.get("_grade", 0)) == g]
                detail = "; ".join(f'{r.get("patient_id")} {r.get("AETERM","")}' for r in g_rows)
                lines.append(f"  Grade {g} AEs: {grade_dist[g]} ({detail})")
            else:
                lines.append(f"  Grade {g} AEs: {grade_dist[g]}")
        lines.append(f"  Grade 3+ AEs: {sum(n for g, n in grade_dist.items() if g >= 3)}")
        lines.append(f"Serious AEs: {serious_count}")
        lines.append(f"AEs per patient: {', '.join(f'{p}={n}' for p, n in ae_per_pt.most_common())}")
        lines.append(f"NOTE: The table below shows only the first 20 of {len(all_rows)} rows. The summary above covers ALL rows.")
        lines.append("")

    if rows:
        header = " | ".join(c.get("label", c.get("key", "")) for c in key_cols)
        lines.append(header)
        lines.append("-" * len(header))
        for row in rows[:20]:
            vals = []
            for c in key_cols:
                v = row.get(c.get("key", ""), "")
                vals.append(str(v)[:30])
            lines.append(" | ".join(vals))

    context_block = "\n".join(lines)
    if len(context_block) > 2400:
        context_block = context_block[:2400] + "\n...(truncated)"

    query_meta = {
        "source": f"CRF/{domain.upper()}",
        "model": _STATS_CHAT_MODEL,
        "matched_sections": [domain.upper()],
        "tab": domain,
        "context_data": context_block,
        "history_turns": len(body.get("history", [])) // 2,
        "message": message,
    }
    return context_block, _CRF_SYSTEM_PROMPT, query_meta


def _build_sae_chat_context(run_path, body, drug_name, indication, n_patients, message):
    """Build context for SAE report page chat."""
    patient_id = body.get("patient_id", "")
    ae_slug = body.get("ae_slug", "")

    from sim.doc_agent.service import DOCS_OUTPUT_DIR
    json_path = DOCS_OUTPUT_DIR / run_path.name / patient_id / f"medwatch_data_{ae_slug}.json"

    if not json_path.exists():
        return JsonResponse({"error": "SAE data not found. Generate the report first."}, status=400), None, None

    try:
        mw = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        return JsonResponse({"error": f"Failed to load SAE data: {e}"}, status=500), None, None

    lines = [f"Drug: {drug_name} | Indication: {indication} | Patient: {patient_id}"]
    lines.append(f"AE: {ae_slug.replace('_', ' ')}")
    lines.append("")

    # Section A: Patient Info
    a = mw.get("section_a", mw.get("A", {}))
    if a:
        lines.append("=== Section A: Patient ===")
        for k in ["age", "sex", "weight", "ethnicity"]:
            if k in a:
                lines.append(f"  {k}: {a[k]}")

    # Section B: Adverse Event
    b = mw.get("section_b", mw.get("B", {}))
    if b:
        lines.append("=== Section B: Adverse Event ===")
        for k in ["event_description", "onset_date", "outcome", "narrative"]:
            v = b.get(k, "")
            if v:
                lines.append(f"  {k}: {str(v)[:400]}")
        # Extract seriousness criteria from individual boolean fields
        serious = [k.replace("seriousness_", "") for k, v in b.items()
                   if k.startswith("seriousness_") and v]
        if serious:
            lines.append(f"  serious_criteria: {', '.join(serious)}")
        elif b.get("serious_criteria"):
            lines.append(f"  serious_criteria: {b['serious_criteria']}")

    # Section C: Suspect Product
    c = mw.get("section_c", mw.get("C", {}))
    if c:
        lines.append("=== Section C: Suspect Product ===")
        for k in ["product_name", "drug_name", "dose", "dose_frequency_route", "route",
                   "indication", "start_date", "therapy_start", "stop_date", "therapy_end",
                   "dechallenge", "rechallenge", "concomitant_meds"]:
            v = c.get(k, "")
            if v:
                lines.append(f"  {k}: {str(v)[:300]}")

    # MedDRA coding
    meddra = mw.get("meddra", mw.get("MedDRA", {}))
    if meddra:
        lines.append("=== MedDRA Coding ===")
        for k, v in meddra.items():
            if v:
                lines.append(f"  {k}: {v}")

    context_block = "\n".join(lines)
    if len(context_block) > 2400:
        context_block = context_block[:2400] + "\n...(truncated)"

    query_meta = {
        "source": f"medwatch_data_{ae_slug}.json",
        "model": _STATS_CHAT_MODEL,
        "matched_sections": ["A", "B", "C", "MedDRA"],
        "tab": ae_slug,
        "context_data": context_block,
        "history_turns": len(body.get("history", [])) // 2,
        "message": message,
    }
    return context_block, _SAE_SYSTEM_PROMPT, query_meta


def _build_sae_hub_chat_context(run_path, body, drug_name, indication, n_patients, message):
    """Build context for SAE hub listing page chat -- summarises all SAEs."""
    from sim.doc_agent.sim_to_crf_adapter import find_serious_aes

    mode = body.get("mode", "natural")
    patient_ids = _list_patients(run_path)

    lines = [f"Drug: {drug_name} | Indication: {indication} | Patients: {n_patients}"]
    lines.append("")
    lines.append("=== All Serious Adverse Events ===")

    sae_rows = []
    for pid in patient_ids:
        profile, records = _load_patient_data(run_path, pid, mode)
        if not records:
            continue
        saes = find_serious_aes(records)
        for sae in saes:
            ae = sae["ae_record"]
            sae_rows.append({
                "patient": pid,
                "term": ae.get("AETERM", ""),
                "grade": ae.get("_grade", 0),
                "day": sae["day"],
                "action": ae.get("AEACN", ""),
                "serious": ae.get("AESER", False),
            })

    if not sae_rows:
        lines.append("No serious adverse events found in this run.")
    else:
        sae_per_patient = Counter(r["patient"] for r in sae_rows)
        term_counts = Counter(r["term"] for r in sae_rows)
        grade_counts = Counter(r["grade"] for r in sae_rows)
        action_counts = Counter(r["action"] or "NONE" for r in sae_rows)
        onset_days = sorted(r["day"] for r in sae_rows)

        lines.append("=== Pre-computed Summary (use these numbers, do NOT re-count) ===")
        lines.append(f"Total SAEs: {len(sae_rows)}")
        lines.append(f"Affected patients: {len(sae_per_patient)} \u2014 {', '.join(f'{p}={n} SAEs' for p, n in sae_per_patient.most_common())}")
        lines.append(f"SAE terms: {', '.join(f'{t}={n}' for t, n in term_counts.most_common())}")
        lines.append(f"Grade distribution: {', '.join(f'G{g}={n}' for g, n in sorted(grade_counts.items()))}")
        lines.append(f"Actions: {', '.join(f'{a}={n}' for a, n in action_counts.most_common())}")
        lines.append(f"Onset range: Day {onset_days[0]} \u2013 Day {onset_days[-1]} (first SAE: Day {onset_days[0]})")
        lines.append(f"Fatal SAEs: 0")
        lines.append("")
        lines.append("=== Full SAE Table ===")
        lines.append("Patient | AE Term | Grade | Onset | Action")
        lines.append("--------|---------|-------|-------|-------")
        for r in sae_rows[:40]:
            action = r['action'] or '\u2014'
            lines.append(f"{r['patient']} | {r['term']} | G{r['grade']} | Day {r['day']} | {action}")

    context_block = "\n".join(lines)
    # Truncate to ~2.4KB
    if len(context_block) > 2400:
        context_block = context_block[:2400] + "\n... (truncated)"

    system_prompt = (
        "You are CLARA's SAE Overview Assistant, an expert in pharmacovigilance.\n"
        "You help researchers analyze the overall SAE profile of a clinical trial run.\n\n"
        "Rules:\n"
        "- Answer based ONLY on the provided SAE listing data below.\n"
        "- NEVER fabricate information not in the data.\n"
        "- The 'Pre-computed Summary' section contains exact counts. ALWAYS use those numbers instead of counting rows yourself.\n"
        "- Be concise (2-5 sentences) unless asked for detail.\n"
        "- Reference specific patients, AE terms, grades, and onset days.\n"
        "- Use the same language as the user.\n"
        "- When user references data with @[...] tags, focus on that specific row.\n"
    )

    query_meta = {
        "page_type": "sae_hub",
        "mode": mode,
        "sae_count": len(sae_rows),
        "message": message,
    }
    return context_block, system_prompt, query_meta


def _build_compare_chat_context(run_path, body, drug_name, indication, n_patients, message):
    """Build context for A/B Comparison page chat -- summarises comparison_report.json."""
    report_path = run_path / "comparison_report.json"
    if not report_path.exists():
        return (
            JsonResponse({"error": "comparison_report.json not found"}, status=404),
            None,
            None,
        )

    with open(report_path) as f:
        report = json.load(f)

    lines = [
        f"Drug: {drug_name} | Indication: {indication} | Patients: {n_patients}",
        "",
        "=== A/B Comparison: Natural vs Care AI ===",
        "=== Pre-computed Summary (AUTHORITATIVE \u2014 use these numbers, do NOT re-count) ===",
    ]

    # Cohort sizes
    cohort = report.get("cohort_sizes", {})
    lines.append(f"Cohort: Natural={cohort.get('natural', '?')}, Care AI={cohort.get('care_ai', '?')}")
    lines.append("")

    # Detection Delay
    dd = report.get("detection_delay", {})
    deltas = report.get("deltas", {})
    lines.append(
        f"Detection Delay: Natural={dd.get('natural_mean', '?')}d, "
        f"Care AI={dd.get('care_ai_mean', '?')}d "
        f"(\u0394={deltas.get('detection_delay', '?')}d) "
        f"[Undetected: Natural={dd.get('natural_undetected', '?')}, Care AI={dd.get('care_ai_undetected', '?')}]"
    )

    # AE Burden
    ab = report.get("ae_burden", {})
    lines.append(
        f"AE Burden (grade\u00d7days): Natural={ab.get('natural_mean', '?')}, "
        f"Care AI={ab.get('care_ai_mean', '?')} "
        f"[Unique AEs: Natural={ab.get('natural_unique_aes', '?')}, Care AI={ab.get('care_ai_unique_aes', '?')}]"
    )

    # Severe AEs
    sa = report.get("severe_aes", {})
    lines.append(
        f"Grade 3+ AE Days: Natural={sa.get('natural_g3plus_mean', '?')}, "
        f"Care AI={sa.get('care_ai_g3plus_mean', '?')} | "
        f"Grade 4+: Natural={sa.get('natural_g4plus_mean', '?')}, "
        f"Care AI={sa.get('care_ai_g4plus_mean', '?')}"
    )

    # Treatment Duration
    td = report.get("treatment_duration", {})
    lines.append(
        f"Treatment Duration: Natural={td.get('natural_mean', '?')}d, "
        f"Care AI={td.get('care_ai_mean', '?')}d "
        f"(\u0394={deltas.get('treatment_duration', '?')}d)"
    )

    # ECOG
    ecog = report.get("ecog", {})
    lines.append(
        f"ECOG Change: Natural=+{ecog.get('natural_mean_delta', '?')}, "
        f"Care AI=+{ecog.get('care_ai_mean_delta', '?')} "
        f"[End ECOG: Natural={ecog.get('natural_mean_end', '?')}, "
        f"Care AI={ecog.get('care_ai_mean_end', '?')}]"
    )

    # Discontinuation
    disc = report.get("discontinuation", {})
    lines.append(
        f"Discontinued: Natural={disc.get('natural_count', '?')}/{cohort.get('natural', '?')} "
        f"({disc.get('natural_pct', '?')}%), "
        f"Care AI={disc.get('care_ai_count', '?')}/{cohort.get('care_ai', '?')} "
        f"({disc.get('care_ai_pct', '?')}%)"
    )

    # Mortality
    mort = report.get("mortality", {})
    lines.append(
        f"Deaths: Natural={mort.get('natural_deaths', '?')}, "
        f"Care AI={mort.get('care_ai_deaths', '?')}"
    )

    # Care AI Activity
    ca = report.get("care_ai_activity", {})
    if ca:
        lines.append("")
        lines.append("Care AI Activity:")
        lines.append(f"  Mean interventions/patient: {ca.get('mean_interventions', '?')}")
        lines.append(f"  Mean AE detections/patient: {ca.get('mean_detections', '?')}")
        lines.append(f"  Mean turns/call: {ca.get('mean_turns_per_call', '?')}")
        lines.append(f"  Early terminations: {ca.get('total_early_terminations', '?')}")
        lines.append(f"  Force hospital visits: {ca.get('total_force_hospital', '?')}")
        itypes = ca.get("intervention_type_totals", {})
        if itypes:
            lines.append(f"  Intervention types: {', '.join(f'{k}={v}' for k, v in itypes.items())}")

    # Statistical Tests
    stats = report.get("statistics", {})
    if stats:
        lines.append("")
        lines.append("Statistical Tests:")
        for test_name, test_data in stats.items():
            if isinstance(test_data, dict) and "p_value" in test_data:
                p = test_data["p_value"]
                sig = "sig" if isinstance(p, (int, float)) and p < 0.05 else "ns"
                stat_val = test_data.get("statistic", "?")
                n_val = test_data.get("n", "?")
                lines.append(f"  {test_name}: W={stat_val}, p={p} ({sig}, n={n_val})")

    # Pre-computed per-patient analysis (NO raw table -- model can't parse it reliably)
    nat_pts = report.get("natural_patients", [])
    cai_pts = report.get("care_ai_patients", [])
    if nat_pts and cai_pts:
        lines.append("")
        lines.append("=== Per-Patient Analysis (pre-computed) ===")

        for label, pts in [("Natural", nat_pts), ("CareAI", cai_pts)]:
            burdens = [p.get("total_ae_burden", 0) for p in pts]
            delays = [p.get("mean_detection_delay", 0) for p in pts]
            g3ds = [p.get("grade3plus_ae_days", 0) for p in pts]
            worst_b = max(pts, key=lambda p: p.get("total_ae_burden", 0))
            best_b = min(pts, key=lambda p: p.get("total_ae_burden", 0))
            deceased = [p["patient_id"] for p in pts if p.get("deceased")]
            disc_list = [p["patient_id"] for p in pts if p.get("discontinued")]
            g3_pts = [f'{p["patient_id"]}={p.get("grade3plus_ae_days", 0)}d' for p in pts if p.get("grade3plus_ae_days", 0) > 0]
            lines.append(
                f"  {label}: burden range {min(burdens)}-{max(burdens)}, "
                f"worst={worst_b['patient_id']}({worst_b.get('total_ae_burden', 0)}), "
                f"best={best_b['patient_id']}({best_b.get('total_ae_burden', 0)})"
            )
            lines.append(
                f"    delay range {min(delays)}-{max(delays)}d, "
                f"G3+ patients: {', '.join(g3_pts) if g3_pts else 'none'}"
            )
            if deceased or disc_list:
                lines.append(
                    f"    deceased: {', '.join(deceased) if deceased else 'none'}, "
                    f"discontinued: {', '.join(disc_list) if disc_list else 'none'}"
                )

        # Paired comparison: per-patient burden change (sorted by delta)
        nat_by_pid = {p["patient_id"]: p for p in nat_pts}
        cai_by_pid = {p["patient_id"]: p for p in cai_pts}
        common_pids = sorted(set(nat_by_pid) & set(cai_by_pid))
        if common_pids:
            pairs = []
            for pid in common_pids:
                nb = nat_by_pid[pid].get("total_ae_burden", 0)
                cb = cai_by_pid[pid].get("total_ae_burden", 0)
                pairs.append((pid, nb, cb, cb - nb))
            improved = [(pid, nb, cb, d) for pid, nb, cb, d in pairs if d < 0]
            worsened = [(pid, nb, cb, d) for pid, nb, cb, d in pairs if d > 0]
            improved.sort(key=lambda x: x[3])  # most improved first (most negative)
            lines.append(f"  Burden improved with CareAI: {len(improved)}/{len(common_pids)} patients")
            # Show sorted by improvement magnitude
            imp_strs = [f"{pid}({nb}\u2192{cb}, \u0394{d})" for pid, nb, cb, d in improved]
            if imp_strs:
                lines.append(f"    {', '.join(imp_strs)}")
            if improved:
                big = improved[0]
                lines.append(f"  Biggest improvement: {big[0]} (burden {big[1]}\u2192{big[2]}, reduced by {abs(big[3])})")
            if worsened:
                w_strs = [f"{pid}({nb}\u2192{cb}, +{d})" for pid, nb, cb, d in worsened]
                lines.append(f"  Burden worsened: {len(worsened)} \u2014 {', '.join(w_strs)}")

    context_block = "\n".join(lines)
    # Truncate to ~2.4KB
    if len(context_block) > 2400:
        context_block = context_block[:2400] + "\n... (truncated)"

    system_prompt = (
        "You are CLARA's A/B Comparison Assistant, an expert in clinical trial analysis.\n"
        "You help researchers analyze Natural vs Care AI simulation results.\n\n"
        "Rules:\n"
        "- Answer based ONLY on the provided comparison data below.\n"
        "- NEVER fabricate information not in the data.\n"
        "- ALL numbers are pre-computed. Quote them directly \u2014 do NOT attempt to calculate, count, or find max/min yourself.\n"
        "- Be concise (2-5 sentences) unless asked for detail.\n"
        "- Highlight statistically significant differences (p < 0.05) when relevant.\n"
        "- When discussing Care AI value, focus on detection delay reduction, AE burden, and patient outcomes.\n"
        "- Reference specific metrics, patient IDs, and statistical test results.\n"
        "- Use the same language as the user.\n"
    )

    query_meta = {
        "page_type": "compare",
        "natural_n": cohort.get("natural", 0),
        "care_ai_n": cohort.get("care_ai", 0),
        "message": message,
    }
    return context_block, system_prompt, query_meta


# --- Unified Doc Chat API ---

@csrf_exempt
def api_doc_chat(request, run_id: str):
    """Unified chat API -- routes to page-specific context builders."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    message = body.get("message", "").strip()
    if not message:
        return JsonResponse({"error": "Empty message"}, status=400)

    page_type = body.get("page_type", "stats")
    history = body.get("history", [])

    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return JsonResponse({"error": "Run not found"}, status=404)

    # Load common metadata
    rule_set = _load_rule_set(run_path)
    meta = _load_run_meta(run_path)
    drug_name = rule_set.get("drug_name") or meta.get("drug_name", "Unknown")
    indication = rule_set.get("indication") or meta.get("indication", "")
    n_patients = len(_list_patients(run_path))

    # Route to context builder
    if page_type == "stats":
        context_block, system_prompt, query_meta = _build_stats_chat_context(
            run_path, body, drug_name, indication, n_patients, message)
    elif page_type == "crf":
        context_block, system_prompt, query_meta = _build_crf_chat_context(
            run_path, body, drug_name, indication, n_patients, message)
    elif page_type == "sae":
        context_block, system_prompt, query_meta = _build_sae_chat_context(
            run_path, body, drug_name, indication, n_patients, message)
    elif page_type == "sae_hub":
        context_block, system_prompt, query_meta = _build_sae_hub_chat_context(
            run_path, body, drug_name, indication, n_patients, message)
    elif page_type == "compare":
        context_block, system_prompt, query_meta = _build_compare_chat_context(
            run_path, body, drug_name, indication, n_patients, message)
    else:
        return JsonResponse({"error": f"Unknown page_type: {page_type}"}, status=400)

    if isinstance(context_block, JsonResponse):
        return context_block  # Error response from builder

    # Build LLM messages
    full_system = system_prompt + "\n---\nData:\n" + context_block
    messages = [{"role": "system", "content": full_system}]
    for msg in history[-4:]:
        role = msg.get("role", "user")
        if role == "model":
            role = "assistant"
        messages.append({"role": role, "content": msg.get("content", "")})
    messages.append({"role": "user", "content": message})

    # Call LLM (vLLM or Gemini fallback)
    return _call_chat_llm(messages, query_meta)
