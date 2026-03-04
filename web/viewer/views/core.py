"""
Core page views: landing, simulation_list.
"""
from django.shortcuts import render

from ._helpers import _get_runs, _get_run_path, PINNED_RUN_ID


def landing(request):
    """Landing page: pure technology showcase + demo map preview."""
    # Fixed to the 100-patient Etoposide + Cisplatin run for a visually rich map
    demo_run_id = PINNED_RUN_ID
    # Verify the run exists, fall back to dynamic selection if not
    run_path = _get_run_path(demo_run_id)
    if not run_path.exists() or not (run_path / "simulations").exists():
        runs = _get_runs()
        demo_run_id = ""
        for r in runs:
            if r.get("status") == "completed" and (r.get("n_patients") or 0) >= 5:
                demo_run_id = r["id"]
                break
        if not demo_run_id and runs:
            demo_run_id = runs[0]["id"]
    context = {"demo_run_id": demo_run_id}
    return render(request, "landing/landing.html", context)


def simulation_list(request):
    """Simulation list page: all runs, stats, and new-sim modal."""
    runs = _get_runs()
    context = {"runs": runs}
    return render(request, "simulation/simulation_list.html", context)
