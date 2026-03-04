"""
Care Agent Demo views: demo_care_agent, api_care_agent_run,
api_care_agent_patients, api_care_agent_media, api_care_agent_chat.
"""
import json
import os
import time
from pathlib import Path

from django.conf import settings
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET

from ._helpers import logger


def _load_virtual_patients():
    """Load virtual patient config for Care Agent demo."""
    config_path = Path(settings.BASE_DIR).parent / "data" / "multimodal" / "care_agent_patients.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {"patients": [], "drug_name": "Unknown", "indication": ""}


def demo_care_agent(request):
    """Care Agent demo page -- Tab 1: MedGemma Vision, Tab 2: Care Agent Live."""
    config = _load_virtual_patients()
    return render(request, "demo/data_collection_agent.html", {"vpatients": config.get("patients", [])})


@csrf_exempt
@require_POST
def api_care_agent_run(request):
    """Run Care Agent for a virtual patient -- SSE stream via Care AI /v1/consult API."""
    import queue, threading, base64, requests as _requests

    body = json.loads(request.body)
    patient_id = body.get("patient_id", "")
    user_api_key = body.get("api_key", "").strip()

    config = _load_virtual_patients()
    vpt = None
    for p in config.get("patients", []):
        if p["id"] == patient_id:
            vpt = p
            break
    if not vpt:
        return JsonResponse({"error": "Virtual patient not found"}, status=404)

    drug_name = config.get("drug_name", "Unknown")
    indication = config.get("indication", "")
    rep = vpt.get("representative_day", {})
    demographics = vpt.get("profile", {})

    q = queue.Queue()

    CARE_AI_URL = os.environ.get("CARE_AI_API_URL", "http://clara-data-collection-agent:8300")

    def _run():
        try:
            import time as _time
            import logging
            log = logging.getLogger("care_agent")

            face_idx = vpt["face_idx"]
            ae_term = rep.get("ae")
            ae_grade = rep.get("grade", 0)
            image_file = rep.get("image", "")
            audio_label = rep.get("audio_label", "none")
            baseline_image = f"normal_{face_idx}.png"
            patient_text = vpt.get("patient_text", "")
            audio_file = vpt.get("audio_file", f"patient_{face_idx}.wav")

            current_labs = vpt.get("current_labs", {})
            baseline_labs = vpt.get("baseline_labs", {})
            current_vitals = vpt.get("current_vitals", {})
            current_meds = vpt.get("current_medications", [])
            med_history = vpt.get("medical_history", [])
            ecog = vpt.get("ecog", 1)
            mood_data = vpt.get("mood", {})

            # --- Load image + audio as base64 ---
            data_dir = Path(settings.BASE_DIR).parent / "data" / "multimodal"
            img_path = data_dir / "v3_images" / image_file
            audio_path = data_dir / "audio" / "patient1" / audio_file

            image_b64 = None
            if img_path.exists():
                image_b64 = base64.b64encode(img_path.read_bytes()).decode()

            audio_b64 = None
            if audio_path.exists():
                audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()

            # --- SSE: session_start ---
            q.put(json.dumps({"type": "session_start", "patient_id": patient_id,
                              "total_days": 1}))

            # --- SSE: day_start ---
            q.put(json.dumps({
                "type": "day_start",
                "image": image_file,
                "baseline_image": baseline_image,
                "audio_label": audio_label,
                "audio_url": f"/api/care-agent/media/audio/{audio_file}" if audio_b64 else None,
                "has_visual_ae": bool(ae_term),
                "gt_ae": ae_term or "none",
                "gt_grade": ae_grade,
            }))

            # --- SSE: patient_greet (patient text -- before inference starts) ---
            q.put(json.dumps({
                "type": "patient_greet",
                "text": patient_text,
                "audio_url": f"/api/care-agent/media/audio/{audio_file}" if audio_b64 else None,
                "symptoms": [],
                "mood": "neutral",
            }))

            # --- SSE: inference_start (thinking bubble appears) ---
            q.put(json.dumps({"type": "inference_start", "has_audio": bool(audio_b64)}))

            # ===== Step 1: HeAR (cough detection) =====
            audio_assessment = None
            audio_text = "No cough detected"
            hear_result = {}
            if audio_b64:
                try:
                    t0 = _time.time()
                    resp = _requests.post(
                        f"{CARE_AI_URL}/v1/cough",
                        json={"audio_b64": audio_b64},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    cough_data = resp.json()
                    audio_assessment = cough_data.get("audio_assessment")
                    cough_ms = cough_data.get("latency_ms", 0)
                    if audio_assessment:
                        hear_result = {
                            "cough_detected": audio_assessment.get("cough_detected", False),
                            "majority_type": audio_assessment.get("majority_type"),
                            "num_cough_segments": audio_assessment.get("num_cough_segments", 0),
                            "num_energy_segments": audio_assessment.get("num_energy_segments", 0),
                            "duration_sec": audio_assessment.get("duration_sec", 0),
                            "vote_counts": audio_assessment.get("vote_counts", {}),
                            "latency_ms": cough_ms,
                        }
                        if audio_assessment.get("cough_detected"):
                            mtype = audio_assessment.get("majority_type", "dry")
                            audio_text = f"{mtype.capitalize()} cough detected"
                except Exception as exc:
                    log.warning("HeAR /v1/cough error: %s", exc)

                # --- SSE: hear_result ---
                q.put(json.dumps({"type": "hear_result", **hear_result}))
            else:
                q.put(json.dumps({"type": "hear_unavailable"}))

            # ===== Step 2: MedASR (transcription) =====
            medical_transcript = None
            if audio_b64:
                try:
                    t0 = _time.time()
                    resp = _requests.post(
                        f"{CARE_AI_URL}/v1/transcribe",
                        json={"audio_b64": audio_b64},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    asr_data = resp.json()
                    medical_transcript = asr_data.get("medical_transcript")
                    medasr_ms = asr_data.get("latency_ms", 0)
                except Exception as exc:
                    log.warning("MedASR /v1/transcribe error: %s", exc)

                # --- SSE: medasr_result ---
                medasr_info = {}
                if medical_transcript:
                    words = len(medical_transcript.split())
                    medasr_info = {"transcript": medical_transcript, "word_count": words, "latency_ms": medasr_ms}
                q.put(json.dumps({"type": "medasr_result", **medasr_info}))
            else:
                q.put(json.dumps({"type": "medasr_unavailable"}))

            # ===== Step 3: SigLIP (visual classification) =====
            visual_assessment = {}
            siglip_findings = []
            siglip_ms = 0
            raw_pred = {}
            if image_b64:
                try:
                    t0 = _time.time()
                    resp = _requests.post(
                        f"{CARE_AI_URL}/v1/classify",
                        json={"image_b64": image_b64},
                        timeout=60,
                    )
                    resp.raise_for_status()
                    classify_data = resp.json()
                    visual_assessment = classify_data.get("visual_assessment", {})
                    siglip_ms = classify_data.get("latency_ms", 0)
                    va_findings = visual_assessment.get("findings", [])
                    raw_pred = visual_assessment.get("raw_prediction", {})
                    for f in va_findings:
                        siglip_findings.append({
                            "ae_term": f.get("ae_term") or "normal",
                            "grade": f.get("estimated_grade") or 0,
                            "confidence": f.get("confidence", 0),
                            "description": f.get("description", ""),
                        })
                except Exception as exc:
                    log.warning("SigLIP /v1/classify error: %s", exc)

            # --- SSE: siglip_result ---
            q.put(json.dumps({
                "type": "siglip_result",
                "findings": siglip_findings,
                "raw_prediction": raw_pred,
                "general_observations": visual_assessment.get("general_observations", []),
                "latency_ms": siglip_ms,
                "baseline_image": baseline_image,
                "current_image": image_file,
            }))

            # --- Build visual_obs for nurse_context ---
            visual_obs = []
            if siglip_findings:
                for vf in siglip_findings:
                    if vf["ae_term"] != "normal":
                        visual_obs.append(f"{vf['ae_term'].replace('_',' ')} G{vf['grade']}")
            if not visual_obs:
                visual_obs.append("No visual AE changes detected")

            visual_assessment_ctx = {"findings": siglip_findings, "general_observations": visual_obs}

            # --- SSE: nurse_context ---
            q.put(json.dumps({
                "type": "nurse_context",
                "visual_assessment": visual_assessment_ctx,
                "audio_assessment": audio_text,
                "patient_info": demographics,
                "current_labs": current_labs,
                "baseline_labs": baseline_labs,
                "current_vitals": current_vitals,
                "current_medications": current_meds,
                "medical_history": [h.get("condition", "") + (" (" + h.get("medication", "") + ")" if h.get("medication") and h["medication"] != "none" else "") for h in med_history],
                "ecog": ecog,
                "mood": mood_data,
                "medical_transcript": medical_transcript,
            }))

            # ===== Step 4: NurseEngine (MedGemma + TTS) =====
            nurse_structured = {}
            nurse_text = ""
            nurse_audio_b64 = None
            session_id = ""
            consult_elapsed = 0
            try:
                t0 = _time.time()
                nurse_payload = {
                        "patient_text": patient_text,
                        "visual_assessment": visual_assessment,
                        "audio_assessment": audio_assessment,
                        "medical_transcript": medical_transcript,
                        "drug_name": drug_name,
                        "indication": indication,
                        "skip_tts": False,
                    }
                if user_api_key:
                    nurse_payload["api_key"] = user_api_key
                resp = _requests.post(
                    f"{CARE_AI_URL}/v1/nurse",
                    json=nurse_payload,
                    timeout=120,
                )
                if resp.status_code == 403:
                    detail = resp.json().get("detail", "Invalid API key")
                    q.put(json.dumps({"type": "error", "message": detail}))
                    q.put(json.dumps({"type": "finished"}))
                    return
                resp.raise_for_status()
                nurse_data = resp.json()
                session_id = nurse_data.get("session_id", "")
                nurse_text = nurse_data.get("nurse_text", "")
                nurse_structured = nurse_data.get("nurse_structured", {})
                nurse_audio_b64 = nurse_data.get("audio_base64")
                consult_elapsed = (nurse_data.get("latency_ms") or {}).get("total_ms", 0)
            except Exception as exc:
                log.error("Nurse /v1/nurse error: %s", exc)
                q.put(json.dumps({"type": "error", "message": f"Nurse API error: {exc}"}))
                q.put(json.dumps({"type": "finished"}))
                return

            nurse_questions = nurse_structured.get("questions", [])
            nurse_concerns = nurse_structured.get("preliminary_concerns", [])

            # Build nurse audio URL if TTS was returned
            nurse_audio_url = None
            if nurse_audio_b64:
                import base64 as _b64
                nurse_wav_name = f"nurse_{face_idx}.wav"
                nurse_audio_dir = data_dir / "audio" / "nurse_1"
                nurse_audio_dir.mkdir(parents=True, exist_ok=True)
                nurse_wav_path = nurse_audio_dir / nurse_wav_name
                nurse_wav_path.write_bytes(_b64.b64decode(nurse_audio_b64))
                nurse_audio_url = f"/api/care-agent/media/audio/{nurse_wav_name}"

            # --- SSE: nurse_turn ---
            q.put(json.dumps({
                "type": "nurse_turn", "turn": 1,
                "text": nurse_text,
                "questions": nurse_questions,
                "concerns": nurse_concerns,
                "audio_url": nurse_audio_url,
                "approach_style": nurse_structured.get("approach_style", "empathetic"),
                "session_id": session_id,
            }))

            # --- SSE: assessment ---
            detected_aes = []
            for f in siglip_findings:
                if f["ae_term"] != "normal" and f["confidence"] > 0.1:
                    detected_aes.append({
                        "ae_term": f["ae_term"],
                        "grade": f["grade"],
                        "confidence": f["confidence"],
                        "source": "visual",
                    })
            for concern in nurse_concerns:
                concern_lower = concern.lower().replace(" ", "_")
                already = any(a["ae_term"] == concern_lower for a in detected_aes)
                if not already:
                    detected_aes.append({
                        "ae_term": concern_lower,
                        "grade": 0,
                        "confidence": 0,
                        "source": "nurse_concern",
                    })

            concern_level = "low"
            if any(a.get("grade", 0) >= 3 for a in detected_aes):
                concern_level = "high"
            elif any(a.get("grade", 0) >= 2 for a in detected_aes):
                concern_level = "moderate"

            q.put(json.dumps({
                "type": "assessment",
                "detected_aes": detected_aes,
                "concern": concern_level,
                "action": "recommend_early_visit" if concern_level in ("high", "moderate") else "continue_monitoring",
                "latency_ms": consult_elapsed,
            }))

            # --- SSE: day_end ---
            gt_list = [{"ae": ae_term, "grade": ae_grade}] if ae_term else []
            q.put(json.dumps({
                "type": "day_end",
                "gt_aes": gt_list,
                "detected_aes": detected_aes,
            }))

            q.put(json.dumps({"type": "finished"}))
        except Exception as e:
            import traceback
            q.put(json.dumps({"type": "error", "message": str(e),
                              "trace": traceback.format_exc()[-500:]}))
        finally:
            q.put(None)

    def _stream():
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {item}\n\n"

    resp = StreamingHttpResponse(_stream(), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    return resp


@require_GET
def api_care_agent_patients(request):
    """List virtual patients for Care Agent demo."""
    config = _load_virtual_patients()
    patients = []
    for p in config.get("patients", []):
        rep = p.get("representative_day", {})
        patients.append({
            "patient_id": p["id"],
            "age": p.get("profile", {}).get("age"),
            "sex": p.get("profile", {}).get("sex"),
            "race": p.get("profile", {}).get("race"),
            "persona": p.get("persona", ""),
            "ae_type": rep.get("ae", ""),
            "ae_grade": rep.get("grade", 0),
            "rep_day": rep.get("day", 0),
            "baseline_image": f"normal_{p.get('face_idx', 0)}.png",
        })
    return JsonResponse({"patients": patients})


@require_GET
def api_care_agent_media(request, media_type: str, filename: str):
    """Serve multimodal assets (images / audio) for Care Agent demo."""
    from django.http import FileResponse
    base = Path(settings.BASE_DIR).parent / "data" / "multimodal"
    if media_type == "image":
        fpath = base / "v3_images" / filename
        content_type = "image/png"
    elif media_type == "audio":
        # Try patient audio first, then nurse audio, then legacy
        fpath = base / "audio" / "patient1" / filename
        if not fpath.exists():
            fpath = base / "audio" / "nurse_1" / filename
        if not fpath.exists():
            fpath = base / "generated_voices_v4" / filename
        content_type = "audio/wav"
    else:
        return HttpResponse("Invalid media type", status=400)

    if not fpath.exists():
        return HttpResponse("Not found", status=404)

    try:
        fpath.resolve().relative_to(base.resolve())
    except ValueError:
        return HttpResponse("Access denied", status=403)

    return FileResponse(open(fpath, "rb"), content_type=content_type)


@csrf_exempt
@require_POST
def api_care_agent_chat(request):
    """Proxy follow-up chat to Care AI /v1/chat endpoint."""
    import requests as _requests

    body = json.loads(request.body)
    session_id = body.get("session_id", "")
    message = body.get("message", "")
    user_api_key = body.get("api_key", "").strip()

    if not session_id or not message:
        return JsonResponse({"error": "session_id and message are required"}, status=400)

    CARE_AI_URL = os.environ.get("CARE_AI_API_URL", "http://clara-data-collection-agent:8300")

    try:
        chat_payload = {
                "session_id": session_id,
                "message": message,
                "skip_tts": body.get("skip_tts", False),
            }
        if user_api_key:
            chat_payload["api_key"] = user_api_key
        resp = _requests.post(
            f"{CARE_AI_URL}/v1/chat",
            json=chat_payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        # If TTS audio returned, save to file and provide URL
        audio_url = None
        if data.get("audio_base64"):
            import base64
            data_dir = Path(settings.BASE_DIR).parent / "data" / "multimodal" / "audio" / "nurse_1"
            data_dir.mkdir(parents=True, exist_ok=True)
            audio_filename = f"chat_{session_id[:8]}_{int(time.time())}.wav"
            audio_path = data_dir / audio_filename
            audio_path.write_bytes(base64.b64decode(data["audio_base64"]))
            audio_url = f"/api/care-agent/media/audio/{audio_filename}"

        return JsonResponse({
            "nurse_text": data.get("nurse_text", ""),
            "nurse_structured": data.get("nurse_structured", {}),
            "audio_url": audio_url,
            "latency_ms": data.get("latency_ms", {}),
        })
    except _requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        detail = ""
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return JsonResponse({"error": detail}, status=status)
    except Exception as exc:
        return JsonResponse({"error": f"Care AI chat error: {exc}"}, status=502)
