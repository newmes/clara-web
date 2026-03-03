"""
Rule Set Generation views: api_ruleset_drugs, api_ruleset_compare,
api_ruleset_compare_all, demo_ruleset_generation, api_ruleset_generate,
api_ruleset_generate_status.
"""
import json
import os
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from ._helpers import logger


# ─── Rule Set Generation: GT vs Predicted Comparison ────────
_RULESET_DIR = Path(settings.BASE_DIR).parent / "src" / "ruleset_generation"

_GT_TO_OUTPUT = {
    "1_Darbepoetin_alfa": "darbepoetin_alfa_small_cell_lung_cancer",
    "2_Etoposide_Cisplatin": "etoposide+cisplatin_small_cell_lung_cancer",
    "3_CALGB9732_Paclitaxel_Cisplatin_Etoposide": "paclitaxel+cisplatin+etoposide_small_cell_lung_cancer",
    "4_Carboplatin_Etoposide": "etoposide+carboplatin_small_cell_lung_cancer",
    "6_Paclitaxel_Carboplatin_Bevacizumab": "paclitaxel+carboplatin+bevacizumab_non-small_cell_lung_cancer",
    "7_Paclitaxel_Carboplatin": "paclitaxel+carboplatin_non-small_cell_lung_cancer",
    "8_Gemcitabine_Cisplatin": "gemcitabine+cisplatin_squamous_non-small_cell_lung_cancer",
}


def _load_json_safe(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _normalize_ae(term):
    import re
    t = term.lower().strip().replace(" ", "_").replace("-", "_")
    t = re.sub(r"_s$", "", t)
    _syn = {
        "diarrhoea": "diarrhea", "dyspnoea": "dyspnea", "anaemia": "anemia",
        "haemoglobin_decreased": "lower_hemoglobin", "paraesthesia": "paresthesia",
        "hyponatraemia": "hyponatremia", "oedema": "edema", "leucopenia": "leukopenia",
    }
    return _syn.get(t, t)


def _clamp01(x):
    return max(0.0, min(1.0, x))


def _score_ruleset(out, gt):
    """Score predicted vs GT across 9 dimensions. Returns list of {dim, score, detail}."""
    import re

    def _parse_dose(s):
        m = re.search(r"([\d.]+)", str(s or ""))
        return float(m.group(1)) if m else None

    results = []

    # 1. Doses
    out_sched = out.get("administration_schedule", [])
    gt_sched = gt.get("administration_schedule", [])
    out_d = {e.get("drug_name", "").lower().strip(): _parse_dose(e.get("dose_per_administration")) for e in out_sched}
    gt_d = {e.get("drug_name", "").lower().strip(): _parse_dose(e.get("dose_per_administration")) for e in gt_sched}
    if gt_d:
        sc = []
        for drug, gd in gt_d.items():
            if drug in out_d and out_d[drug] and gd:
                r = min(out_d[drug], gd) / max(out_d[drug], gd)
                sc.append(1.0 if r >= 0.9 else r)
            else:
                sc.append(0.0)
        results.append({"dim": "Doses", "score": sum(sc) / len(sc), "detail": f"out={out_d} gt={gt_d}"})
    else:
        results.append({"dim": "Doses", "score": 0.0, "detail": "no GT doses"})

    # 2. ORR
    out_orr = out.get("efficacy", {}).get("overall_response_rate")
    gt_orr = gt.get("efficacy", {}).get("overall_response_rate")
    if gt_orr and out_orr is not None:
        results.append({"dim": "ORR", "score": _clamp01(1.0 - abs(out_orr - gt_orr) / gt_orr), "detail": f"out={out_orr:.3f} gt={gt_orr:.3f}"})
    else:
        results.append({"dim": "ORR", "score": 0.0, "detail": f"out={out_orr} gt={gt_orr}"})

    # 3. Age Range
    oa = out.get("demographics", {}).get("age", {}).get("params", {})
    ga = gt.get("demographics", {}).get("age", {}).get("params", {})
    om, ox, gm, gx = oa.get("min"), oa.get("max"), ga.get("min"), ga.get("max")
    if gm is not None and gx is not None and gx != gm and om is not None and ox is not None:
        results.append({"dim": "Age Range", "score": _clamp01(1.0 - (abs(om - gm) + abs(ox - gx)) / (gx - gm)), "detail": f"out={om}-{ox} gt={gm}-{gx}"})
    else:
        results.append({"dim": "Age Range", "score": 0.0, "detail": f"out={om}-{ox} gt={gm}-{gx}"})

    # 4. Sex Ratio
    os_sex = out.get("demographics", {}).get("sex", {}).get("options", {})
    gs_sex = gt.get("demographics", {}).get("sex", {}).get("options", {})
    om_v = os_sex.get("Male", os_sex.get("M", 0))
    gm_v = gs_sex.get("Male", gs_sex.get("M", 0))
    if isinstance(om_v, dict):
        om_v = om_v.get("probability", 0)
    if isinstance(gm_v, dict):
        gm_v = gm_v.get("probability", 0)
    results.append({"dim": "Sex Ratio", "score": _clamp01(1.0 - abs(float(om_v) - float(gm_v))), "detail": f"out_M={om_v} gt_M={gm_v}"})

    # 5. ECOG
    oe = out.get("demographics", {}).get("ecog_ps", {}).get("options", {})
    ge = gt.get("demographics", {}).get("ecog_ps", {}).get("options", {})
    all_k = set(list(oe.keys()) + list(ge.keys()))
    td = sum(abs(float(oe.get(k, 0)) - float(ge.get(k, 0))) for k in all_k) if all_k else 0
    results.append({"dim": "ECOG", "score": _clamp01(1.0 - td / 2.0), "detail": f"out={dict(oe)} gt={dict(ge)}"})

    # 6. AE Count
    on = len(out.get("ae_profile", []))
    gn = len(gt.get("ae_profile", []))
    mx = max(on, gn)
    results.append({"dim": "AE Count", "score": _clamp01(1.0 - abs(on - gn) / mx) if mx else 1.0, "detail": f"out={on} gt={gn}"})

    # 7. AE Freq
    oa_map = {_normalize_ae(ae.get("ae_term", "")): ae.get("incidence_all_grade", 0) for ae in out.get("ae_profile", [])}
    ga_map = {_normalize_ae(ae.get("ae_term", "")): ae.get("incidence_all_grade", 0) for ae in gt.get("ae_profile", [])}
    common = set(oa_map) & set(ga_map)
    if common:
        fsc = []
        for t in common:
            m = max(oa_map[t], ga_map[t])
            fsc.append(1.0 if m == 0 else 1.0 - abs(oa_map[t] - ga_map[t]) / m)
        results.append({"dim": "AE Freq", "score": sum(fsc) / len(fsc), "detail": f"{len(common)} common AEs"})
    else:
        results.append({"dim": "AE Freq", "score": 0.0, "detail": "no common AEs"})

    # 8. Top AE overlap
    oae_s = sorted([(k, v) for k, v in oa_map.items()], key=lambda x: -x[1])
    gae_s = sorted([(k, v) for k, v in ga_map.items()], key=lambda x: -x[1])
    ot10 = set(t for t, _ in oae_s[:10])
    gt10 = set(t for t, _ in gae_s[:10])
    overlap = ot10 & gt10
    results.append({"dim": "Top AE", "score": len(overlap) / 10.0, "detail": f"overlap={sorted(overlap)}"})

    # 9. PFS/OS
    oe_eff = out.get("efficacy", {})
    ge_eff = gt.get("efficacy", {})
    surv_sc = []
    for key in ["progression_free_survival_months", "overall_survival_months"]:
        ov = oe_eff.get(key, {}).get("params", {}).get("mean")
        gv = ge_eff.get(key, {}).get("params", {}).get("mean")
        if gv and gv > 0 and ov is not None:
            surv_sc.append(_clamp01(1.0 - abs(ov - gv) / gv))
        elif gv is None and ov is None:
            surv_sc.append(1.0)
        else:
            surv_sc.append(0.0)
    results.append({"dim": "PFS/OS", "score": sum(surv_sc) / len(surv_sc) if surv_sc else 0.0,
                     "detail": f"PFS: out={oe_eff.get('progression_free_survival_months',{}).get('params',{}).get('mean')} gt={ge_eff.get('progression_free_survival_months',{}).get('params',{}).get('mean')}"})

    return results


def api_ruleset_drugs(request):
    """List available drugs with GT + predicted data."""
    gt_dir = _RULESET_DIR / "ground_truth"
    out_dir = _RULESET_DIR / "output"
    drugs = []
    for gt_folder, out_folder in _GT_TO_OUTPUT.items():
        gt_path = gt_dir / gt_folder / "base.json"
        out_path = out_dir / out_folder / "base.json"
        gt_data = _load_json_safe(gt_path)
        out_data = _load_json_safe(out_path)
        if gt_data:
            drugs.append({
                "id": gt_folder,
                "drug_name": gt_data.get("drug_name", gt_folder),
                "indication": gt_data.get("indication", ""),
                "has_gt": True,
                "has_predicted": out_data is not None,
            })
    return JsonResponse({"drugs": drugs})


def api_ruleset_compare(request, drug_id):
    """Compare GT vs Predicted for a specific drug."""
    gt_dir = _RULESET_DIR / "ground_truth"
    out_dir = _RULESET_DIR / "output"
    out_folder = _GT_TO_OUTPUT.get(drug_id)
    if not out_folder:
        return JsonResponse({"error": f"Unknown drug: {drug_id}"}, status=404)

    gt_data = _load_json_safe(gt_dir / drug_id / "base.json")
    out_data = _load_json_safe(out_dir / out_folder / "base.json")

    if not gt_data:
        return JsonResponse({"error": "GT data not found"}, status=404)
    if not out_data:
        return JsonResponse({"error": "Predicted data not found"}, status=404)

    scores = _score_ruleset(out_data, gt_data)
    avg = sum(s["score"] for s in scores) / len(scores) if scores else 0

    gt_aes = sorted(
        [{"term": ae.get("ae_term", ""), "incidence": ae.get("incidence_all_grade", 0)}
         for ae in gt_data.get("ae_profile", [])],
        key=lambda x: -x["incidence"]
    )[:15]
    pred_aes = sorted(
        [{"term": ae.get("ae_term", ""), "incidence": ae.get("incidence_all_grade", 0)}
         for ae in out_data.get("ae_profile", [])],
        key=lambda x: -x["incidence"]
    )[:15]

    def _extract_fields(data):
        demo = data.get("demographics", {})
        age_raw = demo.get("age", {})
        sex_raw = demo.get("sex", {})
        eff = data.get("efficacy", {})
        sched = data.get("administration_schedule", [])
        doses = []
        for s in sched:
            doses.append({
                "drug": s.get("drug_name", ""),
                "dose": s.get("dose_per_administration", ""),
                "route": s.get("route", ""),
                "schedule_days": s.get("schedule_days", []),
            })
        ecog_raw = demo.get("ecog_ps", {})

        age_params = age_raw.get("params", age_raw) if isinstance(age_raw, dict) else {}
        sex_opts = sex_raw.get("options", sex_raw) if isinstance(sex_raw, dict) else {}
        ecog_opts = ecog_raw.get("options", ecog_raw) if isinstance(ecog_raw, dict) else {}

        pct_male = sex_opts.get("pct_male") or sex_opts.get("Male")
        pct_female = sex_opts.get("pct_female") or sex_opts.get("Female")
        if pct_male and pct_male <= 1:
            pct_male = pct_male * 100
        if pct_female and pct_female <= 1:
            pct_female = pct_female * 100

        pfs = eff.get("progression_free_survival") or eff.get("progression_free_survival_months", {})
        os_data = eff.get("overall_survival") or eff.get("overall_survival_months", {})

        return {
            "n_aes": len(data.get("ae_profile", [])),
            "orr": eff.get("overall_response_rate"),
            "cycle_days": data.get("trial_design", {}).get("cycle_length_days"),
            "doses": doses,
            "age_mean": age_params.get("mean"),
            "age_std": age_params.get("std"),
            "age_min": age_params.get("min"),
            "age_max": age_params.get("max"),
            "pct_male": pct_male,
            "pct_female": pct_female,
            "ecog": ecog_opts if isinstance(ecog_opts, dict) else {},
            "pfs_median": pfs.get("median") if isinstance(pfs, dict) else None,
            "os_median": os_data.get("median") if isinstance(os_data, dict) else None,
        }

    gt_fields = _extract_fields(gt_data)
    gt_fields["top_aes"] = gt_aes
    pred_fields = _extract_fields(out_data)
    pred_fields["top_aes"] = pred_aes

    return JsonResponse({
        "drug_id": drug_id,
        "drug_name": gt_data.get("drug_name", drug_id),
        "indication": gt_data.get("indication", ""),
        "scores": scores,
        "average_score": round(avg, 3),
        "gt_summary": gt_fields,
        "pred_summary": pred_fields,
    }, json_dumps_params={"ensure_ascii": False})


def api_ruleset_compare_all(request):
    """Compare all 7 drugs at once — summary table."""
    gt_dir = _RULESET_DIR / "ground_truth"
    out_dir = _RULESET_DIR / "output"
    rows = []
    for gt_folder, out_folder in _GT_TO_OUTPUT.items():
        gt_data = _load_json_safe(gt_dir / gt_folder / "base.json")
        out_data = _load_json_safe(out_dir / out_folder / "base.json")
        if not gt_data or not out_data:
            continue
        scores = _score_ruleset(out_data, gt_data)
        avg = sum(s["score"] for s in scores) / len(scores) if scores else 0
        rows.append({
            "drug_id": gt_folder,
            "drug_name": gt_data.get("drug_name", gt_folder),
            "indication": gt_data.get("indication", ""),
            "scores": {s["dim"]: round(s["score"], 3) for s in scores},
            "average": round(avg, 3),
        })
    overall_avg = sum(r["average"] for r in rows) / len(rows) if rows else 0
    dims = list(rows[0]["scores"].keys()) if rows else []
    return JsonResponse({
        "drugs": rows,
        "dimensions": dims,
        "overall_average": round(overall_avg, 3),
    })


def demo_ruleset_generation(request):
    """Rule Set Generation demo page."""
    return render(request, "demo/ruleset_generation.html")


_ruleset_gen_jobs: dict = {}


@csrf_exempt
@require_POST
def api_ruleset_generate(request):
    """Start rule set generation for a custom drug."""
    try:
        body = json.loads(request.body)
    except Exception:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    drug_name = body.get("drug_name", "").strip()
    indication = body.get("indication", "").strip()
    user_api_key = body.get("api_key", "").strip()
    if not drug_name:
        return JsonResponse({"error": "drug_name is required"}, status=400)

    job_id = f"{drug_name}_{indication}".replace(" ", "_")[:60]

    if job_id in _ruleset_gen_jobs and _ruleset_gen_jobs[job_id].get("status") == "running":
        return JsonResponse({
            "job_id": job_id,
            "status": "running",
            "message": "Generation already in progress",
        })

    _ruleset_gen_jobs[job_id] = {"status": "running", "progress": "Starting..."}

    def _run():
        import sys as _sys
        _sys.path.insert(0, str(Path(settings.BASE_DIR).parent / "src" / "ruleset_generation"))
        _sys.path.insert(0, str(Path(settings.BASE_DIR).parent))

        # 사용자 API 키가 있으면 현재 스레드에만 설정 (다른 요청에 영향 없음)
        _prev_rule_key = os.environ.get("RULE_ENGINE_LLM_API_KEY")
        if user_api_key:
            from src.agents.llm_client import set_api_key
            set_api_key(user_api_key)
            os.environ["RULE_ENGINE_LLM_API_KEY"] = user_api_key
        else:
            env_path = Path(settings.BASE_DIR).parent / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ.setdefault(key.strip(), val.strip())

        try:
            _ruleset_gen_jobs[job_id]["progress"] = "Collecting evidence from 10 databases..."
            import asyncio
            from rule_engine.config import RuleEngineConfig
            from rule_engine.evidence.collector import collect_evidence
            from rule_engine.agent import synthesize_rules

            gkey = os.environ.get("GOOGLE_API_KEY", "")
            if gkey and not os.environ.get("RULE_ENGINE_LLM_API_KEY"):
                os.environ["RULE_ENGINE_LLM_API_KEY"] = gkey

            drugs = [d.strip() for d in drug_name.split("+")]
            config = RuleEngineConfig()

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            _ruleset_gen_jobs[job_id]["progress"] = "Collecting evidence..."
            evidence = loop.run_until_complete(collect_evidence(drugs, indication, config))

            _ruleset_gen_jobs[job_id]["progress"] = "LLM synthesis (multi-stage)..."
            rule_set, agent_log = loop.run_until_complete(
                synthesize_rules(drugs, indication, evidence, config)
            )
            loop.close()

            from rule_engine.converter import convert_ruleset, split_base_overlay
            internal = json.loads(rule_set.model_dump_json())
            converted, schema_type = convert_ruleset(internal)
            base_dict, overlay_dict = split_base_overlay(converted, schema_type)

            _ruleset_gen_jobs[job_id] = {
                "status": "completed",
                "progress": "Done!",
                "result": base_dict,
                "schema_type": schema_type,
            }
        except Exception as e:
            _ruleset_gen_jobs[job_id] = {
                "status": "error",
                "progress": f"Failed: {e}",
                "error": str(e),
            }
        finally:
            # 사용자 키 복원 (다른 요청에 유출 방지)
            if _prev_rule_key is not None:
                os.environ["RULE_ENGINE_LLM_API_KEY"] = _prev_rule_key
            elif "RULE_ENGINE_LLM_API_KEY" in os.environ and user_api_key:
                del os.environ["RULE_ENGINE_LLM_API_KEY"]

    import threading
    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return JsonResponse({"job_id": job_id, "status": "running", "message": "Generation started"})


def api_ruleset_generate_status(request, job_id):
    """Check rule set generation status."""
    job = _ruleset_gen_jobs.get(job_id)
    if not job:
        return JsonResponse({"error": "Job not found"}, status=404)
    resp = {"job_id": job_id, "status": job["status"], "progress": job.get("progress", "")}
    if job["status"] == "completed" and "result" in job:
        resp["result"] = job["result"]
    if job["status"] == "error":
        resp["error"] = job.get("error", "")
    return JsonResponse(resp, json_dumps_params={"ensure_ascii": False})
