"""
A/B Comparison Dashboard views.
"""
import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from ._helpers import (
    _get_run_path, _load_rule_set, _load_run_meta, logger,
)


def compare_dashboard(request, run_id: str):
    """A/B comparison dashboard page."""
    run_path = _get_run_path(run_id)
    report_path = run_path / "comparison_report.json"
    sim_dir = run_path / "simulations"
    _run_model = _load_run_meta(run_path).get("model", "")

    # Basic validation: simulations directory exists
    if not sim_dir.exists():
        return render(request, "compare/compare.html", {
            "run_id": run_id, "drug_name": "Unknown", "indication": "",
            "report_json": json.dumps({"error": "Simulation directory not found"}),
            "error": "\uc2dc\ubbac\ub808\uc774\uc158 \ub514\ub809\ud1a0\ub9ac\uac00 \uc874\uc7ac\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4.",
            "model_name": _run_model,
        })

    # natural / care_ai file existence check
    natural_files = list(sim_dir.glob("*_natural.jsonl"))
    care_ai_files = list(sim_dir.glob("*_care_ai.jsonl"))
    if not natural_files or not care_ai_files:
        rule_set = _load_rule_set(run_path)
        missing = []
        if not natural_files:
            missing.append("Natural")
        if not care_ai_files:
            missing.append("Care AI")
        return render(request, "compare/compare.html", {
            "run_id": run_id,
            "drug_name": rule_set.get("drug_name", "Unknown"),
            "indication": rule_set.get("indication", ""),
            "report_json": json.dumps({"error": f"Missing data: {', '.join(missing)}"}),
            "error": f"A/B \ube44\uad50\uc5d0 \ud544\uc694\ud55c \ub370\uc774\ud130\uac00 \ubd80\uc871\ud569\ub2c8\ub2e4: {', '.join(missing)} \ubaa8\ub4dc \ub370\uc774\ud130 \uc5c6\uc74c. "
                     f"(Natural: {len(natural_files)}\uba85, Care AI: {len(care_ai_files)}\uba85)",
            "model_name": _run_model,
        })

    # comparison_report.json -- regenerate if missing or stale
    try:
        needs_regen = not report_path.exists()
        if not needs_regen:
            sim_files = list(sim_dir.glob("*.jsonl"))
            if sim_files:
                newest_sim = max(f.stat().st_mtime for f in sim_files)
                if report_path.stat().st_mtime < newest_sim:
                    needs_regen = True

        if needs_regen:
            from sim.evaluator import run_evaluation
            run_evaluation(run_path)

        with open(report_path) as f:
            report = json.load(f)
    except Exception as e:
        rule_set = _load_rule_set(run_path)
        return render(request, "compare/compare.html", {
            "run_id": run_id,
            "drug_name": rule_set.get("drug_name", "Unknown"),
            "indication": rule_set.get("indication", ""),
            "report_json": json.dumps({"error": str(e)}),
            "error": f"\ube44\uad50 \ub9ac\ud3ec\ud2b8 \uc0dd\uc131 \uc911 \uc624\ub958: {e}",
            "model_name": _run_model,
        })

    rule_set = _load_rule_set(run_path)

    return render(request, "compare/compare.html", {
        "run_id": run_id,
        "drug_name": rule_set.get("drug_name", "Unknown"),
        "indication": rule_set.get("indication", ""),
        "report_json": json.dumps(report, ensure_ascii=False),
        "model_name": _run_model,
    })


@require_GET
def api_compare_data(request, run_id: str):
    """A/B comparison data JSON API."""
    run_path = _get_run_path(run_id)
    sim_dir = run_path / "simulations"
    report_path = run_path / "comparison_report.json"

    if not sim_dir.exists():
        return JsonResponse({"error": "Simulation directory not found"}, status=404)

    natural_count = len(list(sim_dir.glob("*_natural.jsonl")))
    care_ai_count = len(list(sim_dir.glob("*_care_ai.jsonl")))
    if natural_count == 0 or care_ai_count == 0:
        return JsonResponse({
            "error": "Both natural and care_ai data required for comparison",
            "natural_count": natural_count,
            "care_ai_count": care_ai_count,
        }, status=400)

    try:
        if not report_path.exists():
            from sim.evaluator import run_evaluation
            run_evaluation(run_path)

        with open(report_path) as f:
            report = json.load(f)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse(report, json_dumps_params={"ensure_ascii": False})


@require_GET
def api_compare_regenerate(request, run_id: str):
    """Force-regenerate comparison report."""
    from sim.evaluator import run_evaluation
    run_path = _get_run_path(run_id)
    sim_dir = run_path / "simulations"

    if not sim_dir.exists():
        return JsonResponse({"error": "Simulation directory not found"}, status=404)

    natural_count = len(list(sim_dir.glob("*_natural.jsonl")))
    care_ai_count = len(list(sim_dir.glob("*_care_ai.jsonl")))
    if natural_count == 0 or care_ai_count == 0:
        return JsonResponse({
            "error": "Both natural and care_ai data required",
            "natural_count": natural_count,
            "care_ai_count": care_ai_count,
        }, status=400)

    try:
        report = run_evaluation(run_path)
        return JsonResponse({"status": "regenerated", "cohort_sizes": report["cohort_sizes"]})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
