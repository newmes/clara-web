"""
MedGemma views: api_medgemma_analyze, api_medgemma_analyze_base,
api_multimodal_enhance.
"""
import json
import os
import time
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from ._helpers import _get_run_path, logger


_MEDGEMMA_PROMPT = (
    "You are a clinical dermatology expert. You are given two images of the same patient:\n"
    "- Image 1: Baseline photograph (before treatment)\n"
    "- Image 2: Current photograph\n\n"
    "Compare the two images and identify any NEW adverse events (AEs) visible in Image 2 "
    "that were NOT present in Image 1. Use the CTCAE categories and grading criteria below "
    "to classify and grade each finding. If the patient's appearance is unchanged between "
    "the two images, return an empty list.\n\n"
    "## AE Categories & Grading\n\n"
    "### rash_maculopapular\n"
    "- Grade 1: Faint pink macules/papules scattered on cheeks (<10% BSA); mild erythema, no scaling, subtle and localized\n"
    "- Grade 2: Visible red macules/papules spreading across cheeks and forehead (10-30% BSA); moderate erythema with fine scaling at lesion edges\n"
    "- Grade 3: Severe confluent rash covering entire face including cheeks, forehead, chin, and nose (>30% BSA); intense erythema, coarse scaling, and facial edema\n\n"
    "### rash_acneiform\n"
    "- Grade 1: Few small papules on forehead (<10% BSA); non-inflamed or mildly inflamed, skin-colored to pink\n"
    "- Grade 2: Multiple erythematous papules and pustules on cheeks and forehead (10-30% BSA); visible pus-filled lesions, surrounding redness\n"
    "- Grade 3: Dense pustules covering entire face - forehead, cheeks, nose, chin (>30% BSA); confluent inflammation, crusting, signs of secondary infection\n\n"
    "### periorbital_edema\n"
    "- Grade 1: Slight puffiness of upper and lower eyelids, barely noticeable; periorbital skin appears mildly swollen\n"
    "- Grade 2: Obvious bilateral periorbital swelling; puffy, baggy eyelids with visible tissue distension; eyes appear partially narrowed\n"
    "- Grade 3: Severe periorbital edema causing near-closure of eyes; tense, shiny skin around orbital rims, eye-opening significantly impaired\n\n"
    "### sjs_prodrome\n"
    "- Grade 1: Lip redness and dryness; vermilion border appears erythematous, slight chapping without blistering\n"
    "- Grade 2: Lip and oral mucosal blistering; fluid-filled vesicles on lip surface and inner mouth; erosions with crusting at lip margins\n"
    "- Grade 3: Beginning of epidermal detachment on lips and perioral skin; large erosions, bleeding mucosa, severe crusting extending beyond lip borders\n\n"
    "### stomatitis\n"
    "- Grade 1: Mild redness or minor aphthous-like ulcer on lip mucosa; slight discomfort, no visible swelling from outside\n"
    "- Grade 2: Visible cracking and erythema at lip corners with shallow erosions; perioral redness, mild swelling of the lips\n"
    "- Grade 3: Severe lip swelling and deep erosions visible on external lip surface; crusting, bleeding, perioral inflammation\n\n"
    "### pruritus\n"
    "- Grade 1: Mild localized skin excoriation marks on forehead or cheeks; faint scratch marks, minimal erythema\n"
    "- Grade 2: Moderate visible scratch marks and erythema across face; dry, irritated skin with diffuse redness\n"
    "- Grade 3: Severe widespread excoriations with lichenification; intense erythema, bleeding scratch marks, facial edema from chronic scratching\n\n"
    "### alopecia\n"
    "- Grade 1: Mild hair thinning visible at temples and frontal hairline; slightly widened part line, subtle compared to baseline\n"
    "- Grade 2: Obvious diffuse hair thinning with clearly visible scalp through hair; temporal recession, noticeably sparse hair\n\n"
    "## Output Format\n"
    "Return a JSON array of detected AEs. Each element:\n"
    '{"ae_term": "<category>", "grade": <1|2|3>, "confidence": <0.0-1.0>, '
    '"reasoning": "<clinical description of what changed from Image 1 to Image 2>"}\n\n'
    "If no change from baseline is detected, return: []"
)


def _medgemma_infer(baseline_name, current_name, vllm_url, model_id):
    """Shared inference logic for MedGemma vLLM calls."""
    import base64
    import re
    import urllib.request
    import urllib.error

    img_dir = Path(settings.BASE_DIR) / "static_dirs" / "assets" / "medgemma"
    baseline_path = img_dir / baseline_name
    current_path = img_dir / current_name

    if not baseline_path.exists() or not current_path.exists():
        return None, "Image not found"

    b64_baseline = base64.b64encode(baseline_path.read_bytes()).decode("utf-8")
    b64_current = base64.b64encode(current_path.read_bytes()).decode("utf-8")

    payload = json.dumps({
        "model": model_id,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_baseline}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_current}"}},
                {"type": "text", "text": _MEDGEMMA_PROMPT},
            ],
        }],
        "max_completion_tokens": 1024,
        "temperature": 0,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{vllm_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        return None, f"vLLM request failed: {e}"
    latency_ms = (time.time() - t0) * 1000

    raw_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

    detected = []
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            detected = data
    except (json.JSONDecodeError, TypeError):
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    detected = data
            except (json.JSONDecodeError, TypeError):
                pass

    return {
        "detected_aes": detected,
        "raw_output": raw_text,
        "latency_ms": round(latency_ms, 1),
    }, None


@csrf_exempt
@require_POST
def api_medgemma_analyze(request):
    """Run live MedGemma 4B finetuned inference via vLLM."""
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    baseline_name = body.get("baseline")
    current_name = body.get("current")
    if not baseline_name or not current_name:
        return JsonResponse({"error": "baseline and current are required"}, status=400)

    vllm_url = os.environ.get("MEDGEMMA4B_VLLM_BASE_URL", "http://clara-medgemma4b-ft:8000/v1")
    model_id = os.environ.get("MEDGEMMA4B_MODEL_ID", "medgemma-4b-ctcae")

    result, err = _medgemma_infer(baseline_name, current_name, vllm_url, model_id)
    if err:
        status = 404 if "not found" in err else 502
        return JsonResponse({"error": err}, status=status)
    return JsonResponse(result)


@csrf_exempt
@require_POST
def api_medgemma_analyze_base(request):
    """Run live MedGemma 4B base (zero-shot) inference via vLLM."""
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    baseline_name = body.get("baseline")
    current_name = body.get("current")
    if not baseline_name or not current_name:
        return JsonResponse({"error": "baseline and current are required"}, status=400)

    vllm_url = os.environ.get("MEDGEMMA4B_BASE_VLLM_BASE_URL", "http://clara-medgemma4b:8000/v1")
    model_id = os.environ.get("MEDGEMMA4B_BASE_MODEL_ID", "google/medgemma-4b-it")

    result, err = _medgemma_infer(baseline_name, current_name, vllm_url, model_id)
    if err:
        status = 404 if "not found" in err else 502
        return JsonResponse({"error": err}, status=status)
    return JsonResponse(result)


# ─── Multimodal Enhance (face + voice generation) ─────────

_mm_bridges = {}


@csrf_exempt
@require_POST
def api_multimodal_enhance(request):
    """Generate face image + voice audio for a care_record turn on demand.

    POST JSON: {run_id, patient_id, day, turn_index (0-based, patient turns only)}
    Returns:   {face_b64, audio_b64, mm_meta} or {error}
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    run_id = body.get("run_id", "")
    patient_id = body.get("patient_id", "")
    day = int(body.get("day", 0))
    turn_index = int(body.get("turn_index", 0))

    if not run_id or not patient_id or day < 1:
        return JsonResponse({"error": "run_id, patient_id, day required"}, status=400)

    run_path = _get_run_path(run_id)
    if not run_path:
        return JsonResponse({"error": f"Run not found: {run_id}"}, status=404)

    # Load patient profile
    patient_file = run_path / "patients" / f"{patient_id}.json"
    if not patient_file.exists():
        return JsonResponse({"error": f"Patient not found: {patient_id}"}, status=404)

    with open(patient_file) as f:
        patient_json = json.load(f)

    # Load day data from care_ai JSONL
    sim_dir = run_path / "simulations"
    care_file = sim_dir / f"{patient_id}_care_ai.jsonl"
    if not care_file.exists():
        return JsonResponse({"error": "No care_ai data for this patient"}, status=404)

    day_data = None
    with open(care_file) as f:
        for line in f:
            d = json.loads(line)
            if d.get("day") == day:
                day_data = d
                break

    if not day_data:
        return JsonResponse({"error": f"Day {day} not found"}, status=404)

    care_records = day_data.get("care_record", [])
    if not care_records:
        return JsonResponse({"error": "No care_record for this day"}, status=404)

    cr = care_records[0] if isinstance(care_records, list) else care_records
    turns = cr.get("turns", [])

    patient_turns = [t for t in turns if t.get("role") == "patient"]
    if turn_index >= len(patient_turns):
        return JsonResponse({"error": f"turn_index {turn_index} out of range"}, status=400)

    # Extract text from patient turn
    turn = patient_turns[turn_index]
    content = turn.get("content", {})
    text_parts = []
    if isinstance(content, str):
        text_parts.append(content)
    else:
        if g := content.get("greeting"):
            text_parts.append(g)
        if wb := content.get("general_wellbeing"):
            text_parts.append(wb)
        for sym in content.get("reported_symptoms", []):
            if isinstance(sym, dict) and sym.get("verbal_expression"):
                text_parts.append(sym["verbal_expression"])
        for resp in content.get("responses", []):
            if a := resp.get("answer"):
                text_parts.append(a)
    text = " ".join(text_parts) if text_parts else "I'm not feeling great today."

    active_aes = day_data.get("AE", [])
    mood_snapshot = cr.get("mood_snapshot", {})

    # Get or create bridge (cached per patient within run)
    bridge_key = f"{run_id}:{patient_id}"
    try:
        from src.multimodal.game_bridge import MultimodalGameBridge

        if bridge_key not in _mm_bridges:
            _mm_bridges[bridge_key] = MultimodalGameBridge(patient_json, enabled=True)

        bridge = _mm_bridges[bridge_key]
        media = bridge.generate_turn_media(
            text=text,
            active_aes=active_aes,
            day=day,
            mood_snapshot=mood_snapshot,
        )
        return JsonResponse({
            "face_b64": media.get("face_b64"),
            "audio_b64": media.get("audio_b64"),
            "mm_meta": media.get("mm_meta", {}),
            "text": text,
            "day": day,
            "patient_id": patient_id,
        })

    except ImportError as e:
        return JsonResponse({"error": f"Multimodal module not available: {e}"}, status=500)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": f"Generation failed: {e}"}, status=500)
