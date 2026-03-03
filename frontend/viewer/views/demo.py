"""
Demo page views + AntiHallu API + Demo API (auto-select latest run).
"""
import json
import logging
import os

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET

from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from django.conf import settings

from ._helpers import (
    DATA_DIR, PINNED_RUN_ID, _get_run_path, _list_patients,
    _load_run_meta, logger,
)
from ..crf_aggregator import _read_jsonl_cached, _load_patient_json


# --- Demo Pages ---

def demo_anti_hallucination(request):
    """Anti-Hallucination technology demo page."""
    return render(request, "demo/data_analysis_agent.html")


def demo_patient_init(request):
    """Patient Initialization demo -- single patient generation with avatar."""
    return render(request, "demo/patient_init.html")


def demo_daily_sim(request):
    """Daily Simulation demo -- step-by-step daily simulation with hazard engine."""
    return render(request, "demo/daily_sim.html")


def demo_validate_sim(request):
    """Validate Simulation -- rule-set vs simulation statistical comparison."""
    import json as _json
    run_id = PINNED_RUN_ID
    run_dir = DATA_DIR / "runs" / run_id
    ctx = {"run_id": run_id}
    val_path = run_dir / "validation" / "ruleset_validation_natural_v4.json"
    if val_path.exists():
        ctx["validation_json"] = val_path.read_text(encoding="utf-8")
    rs_path = run_dir / "rule_set.json"
    if rs_path.exists():
        rs = _json.loads(rs_path.read_text(encoding="utf-8"))
        ctx["drug_name"] = rs.get("drug_name", "Unknown")
    return render(request, "demo/validate_sim.html", ctx)


# --- AntiHallu API ---

ANTIHALLU_ASSETS = Path(settings.BASE_DIR) / "static_dirs" / "assets" / "antihallu"
_antihallu_log = logging.getLogger("antihallu")

# FastAPI endpoint for antihallu demo
_AH_FASTAPI_URL = os.environ.get("ANTIHALLU_FASTAPI_URL", "http://clara-antihallu:8000").rstrip("/")


@require_GET
def api_antihallu_examples(request):
    """Return AntiHallu example questions."""
    try:
        data = json.loads((ANTIHALLU_ASSETS / "examples.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return JsonResponse({"error": "examples.json not found"}, status=404)
    return JsonResponse(data)


def _fastapi_antihallu_generate(question: str):
    """Call the AntiHallu FastAPI server (/api/generate). Returns dict or None."""
    if not _AH_FASTAPI_URL:
        return None
    payload = {"question": question}
    body_bytes = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{_AH_FASTAPI_URL}/api/generate",
        data=body_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data
    except (URLError, OSError, json.JSONDecodeError, TimeoutError, KeyError) as exc:
        _antihallu_log.warning("AntiHallu FastAPI call failed (%s): %s", _AH_FASTAPI_URL, exc)
        return None


def _cache_lookup(question: str):
    """Look up a question in the local AntiHallu cache. Returns dict or None."""
    try:
        cache = json.loads(
            (ANTIHALLU_ASSETS / "cache.json").read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        return None
    return cache.get(question.lower())


@csrf_exempt
@require_POST
def api_antihallu_generate(request):
    """Generate AntiHallu comparison: FastAPI live inference with cache fallback."""
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    question = body.get("question", "").strip()
    if not question:
        return JsonResponse({"error": "question is required"}, status=400)

    # 1) Try live inference via FastAPI server
    live_result = _fastapi_antihallu_generate(question)
    if live_result is not None:
        live_result["live"] = True
        return JsonResponse(live_result)

    # 2) Fallback to local cache
    entry = _cache_lookup(question)
    if entry is not None:
        return JsonResponse({
            "question": entry["question"],
            "original": entry["original"],
            "defended": entry["defended"],
            "cached": True,
            "live": False,
        })

    return JsonResponse(
        {"error": "AntiHallu server unavailable and question not in cache"},
        status=503,
    )


# --- Demo API (auto-select latest run) ---

def _get_latest_run_id() -> str | None:
    """Return the run_id of the pinned demo run, falling back to latest."""
    pinned = DATA_DIR / "runs" / PINNED_RUN_ID
    if pinned.is_dir() and (pinned / "simulations").exists():
        return PINNED_RUN_ID
    runs_dir = DATA_DIR / "runs"
    if not runs_dir.exists():
        return None
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if d.is_dir() and (d / "simulations").exists():
            return d.name
    return None


def _load_patient_data(run_path, patient_id, mode="natural"):
    """Load patient profile + day records for demo API (reuse doc module's logic)."""
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


@csrf_exempt
@require_GET
def api_demo_saes(request):
    """Demo SAE list -- auto-selects the latest run, returns all SAEs across
    all patients.

    GET /api/demo/saes/?mode=natural
    """
    mode = request.GET.get("mode", "natural")
    run_id = _get_latest_run_id()
    if not run_id:
        return JsonResponse(
            {"error": "No simulation runs found. Run a simulation first."},
            status=404,
        )

    run_path = _get_run_path(run_id)
    patient_ids = _list_patients(run_path)
    if not patient_ids:
        return JsonResponse(
            {"error": f"No patients found in run '{run_id}'."},
            status=404,
        )

    from src.doc_agent.sim_to_crf_adapter import find_serious_aes

    all_saes = []
    for pid in patient_ids:
        profile, records = _load_patient_data(run_path, pid, mode)
        if not records:
            continue
        try:
            saes = find_serious_aes(records)
        except Exception:
            continue
        for sae in saes:
            ae = sae["ae_record"]
            all_saes.append({
                "patient_id": pid,
                "ae_term": ae.get("AETERM", ""),
                "grade": ae.get("_grade", 0),
                "onset_day": sae.get("day") or ae.get("AESTDAT"),
                "severity": ae.get("AESEV", ""),
                "action": ae.get("AEACN", ""),
                "outcome": ae.get("AEOUT", ""),
                "mode": mode,
            })

    return JsonResponse({
        "run_id": run_id,
        "saes": all_saes,
        "all_patient_ids": patient_ids,
    })


@csrf_exempt
@require_POST
def api_demo_generate(request):
    """Demo report generation -- auto-selects the latest run.

    POST body: {
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

    patient_id = body.get("patient_id", "")
    ae_term = body.get("ae_term", "")
    ae_day = body.get("ae_day")
    mode = body.get("mode", "natural")
    use_ai = body.get("use_ai", False)

    if not all([patient_id, ae_term]):
        return JsonResponse(
            {"error": "patient_id and ae_term are required"},
            status=400,
        )

    run_id = _get_latest_run_id()
    if not run_id:
        return JsonResponse(
            {"error": "No simulation runs found. Run a simulation first."},
            status=404,
        )

    run_path = _get_run_path(run_id)
    profile, records = _load_patient_data(run_path, patient_id, mode)
    if profile is None:
        return JsonResponse(
            {"error": f"Patient '{patient_id}' not found in run '{run_id}'."},
            status=404,
        )

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

    try:
        from datetime import date as dt_date
        from src.doc_agent.service import generate_documents

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
    except Exception as exc:
        logging.exception("Demo generate failed")
        return JsonResponse(
            {"error": f"Document generation failed: {exc}"},
            status=500,
        )

    # Record ai_fields in status file when AI is used
    if use_ai and result.get("success"):
        from datetime import datetime
        from .doc import _read_status, _write_status
        ae_slug = ae_term.replace(" ", "_").replace("/", "_")
        ai_fields = {
            "section_b.narrative": True,
            "section_c.dechallenge": True,
            "section_c.rechallenge": True,
        }
        status_data = _read_status(run_id, patient_id, ae_slug)
        status_data["ai_fields"] = ai_fields
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

    # Include run_id in the response so the caller knows which run was used
    if isinstance(result, dict):
        result["run_id"] = run_id

    return JsonResponse(result)


@csrf_exempt
@require_GET
def api_demo_reports(request):
    """List all generated reports for the latest run.

    GET /api/demo/reports/
    """
    import time

    run_id = _get_latest_run_id()
    if not run_id:
        return JsonResponse(
            {"error": "No simulation runs found. Run a simulation first."},
            status=404,
        )

    from src.doc_agent.service import DOCS_OUTPUT_DIR

    docs_dir = DOCS_OUTPUT_DIR / run_id
    if not docs_dir.exists():
        return JsonResponse({"run_id": run_id, "reports": []})

    reports = []
    for patient_dir in sorted(docs_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient_id = patient_dir.name

        # Group files by ae_slug to pair PDF/XML together
        file_map = {}  # ae_slug -> {pdf_path, xml_path, ...}
        for doc_file in sorted(patient_dir.iterdir()):
            if not doc_file.is_file():
                continue
            fname = doc_file.name
            # Skip status files
            if fname.startswith("report_status_"):
                continue
            # Skip medwatch data JSON files
            if fname.startswith("medwatch_data_"):
                continue

            # Extract ae_slug from filename patterns:
            #   medwatch_3500a_{ae_slug}.pdf
            #   e2b_r3_{ae_slug}.xml
            ae_slug = None
            if fname.startswith("medwatch_3500a_") and fname.endswith(".pdf"):
                ae_slug = fname[len("medwatch_3500a_"):-len(".pdf")]
            elif fname.startswith("e2b_r3_") and fname.endswith(".xml"):
                ae_slug = fname[len("e2b_r3_"):-len(".xml")]

            if ae_slug:
                if ae_slug not in file_map:
                    file_map[ae_slug] = {
                        "patient_id": patient_id,
                        "ae_term": ae_slug.replace("_", " ").title(),
                        "ae_slug": ae_slug,
                    }
                if fname.endswith(".pdf"):
                    file_map[ae_slug]["pdf_path"] = (
                        f"/api/doc/download/{run_id}/{patient_id}/{fname}"
                    )
                    file_map[ae_slug]["created_at"] = (
                        time.strftime(
                            "%Y-%m-%dT%H:%M:%S",
                            time.gmtime(doc_file.stat().st_mtime),
                        )
                    )
                elif fname.endswith(".xml"):
                    file_map[ae_slug]["xml_path"] = (
                        f"/api/doc/download/{run_id}/{patient_id}/{fname}"
                    )

        reports.extend(file_map.values())

    return JsonResponse({"run_id": run_id, "reports": reports})
