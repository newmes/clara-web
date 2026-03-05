"""
Live Simulation API views: api_sim_start, api_sim_status, api_sim_list,
api_sim_stop, api_sim_log.
"""
import json
import os
import threading
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST, require_GET

from ._helpers import (
    DATA_DIR, _get_runs, _get_run_path, _list_patients, logger,
)

_live_sims: dict[str, dict] = {}  # run_id -> {thread, status, runner, ...}


@csrf_exempt
@require_POST
def api_sim_start(request):
    """Start a new live simulation run.

    POST body: {
        "drug": "Padcev + Pembrolizumab",
        "indication": "metastatic urothelial carcinoma",
        "patients": 10,
        "days": 126,
        "mode": "both",
        "seed": 42  (optional)
    }
    Returns: {"run_id": "...", "status": "started"}
    """
    try:
        body = json.loads(request.body)
    except Exception:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    drug = body.get("drug", "Padcev + Pembrolizumab")
    indication = body.get("indication", "metastatic urothelial carcinoma")
    n_patients = int(body.get("patients", 10))
    n_days = int(body.get("days", 126))
    mode = body.get("mode", "both")
    seed = body.get("seed")
    rule_set_preset = body.get("rule_set_preset")
    skip_rules = rule_set_preset is not None or body.get("skip_rules", True)
    user_api_key = body.get("api_key", "").strip()

    # Create run directory
    from datetime import datetime as _dt
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    safe_drug = drug.replace(" ", "_").replace("+", "_")[:30]
    run_name = f"{ts}_{safe_drug}_{n_patients}pt_{n_days}d"
    run_dir = DATA_DIR / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    def _run_simulation():
        """Background thread for running the simulation."""
        import sys as _sys
        _sys.path.insert(0, str(Path(settings.BASE_DIR).parent))

        if user_api_key:
            from sim.agents.llm_client import set_api_key
            set_api_key(user_api_key)
        else:
            # Load .env
            env_path = Path(settings.BASE_DIR).parent / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ.setdefault(key.strip(), val.strip())

        from sim.orchestrator import SimulationRunner

        try:
            runner = SimulationRunner(
                drug_name=drug,
                indication=indication,
                data_dir=str(run_dir),
                seed=seed,
            )
            # API 키를 runner에 저장 → 워커 스레드에 전파용
            if user_api_key:
                runner._api_key = user_api_key
            _live_sims[run_name]["runner"] = runner

            # Phase 0: Rules
            if skip_rules:
                base_rule_path = None
                if rule_set_preset:
                    _RULE_SETS_DIR = DATA_DIR / "rule_sets"
                    _preset_map = {
                        "rule_set_calibrated_ev302": DATA_DIR / "rule_set_calibrated_ev302.json",
                        "rule_set_darbepoetin_sclc": DATA_DIR / "rule_set_darbepoetin_sclc.json",
                        "rule_set_ep_sclc": DATA_DIR / "rule_set_ep_sclc.json",
                        "rule_set_default": DATA_DIR / "rule_set.json",
                        "rs_1_Darbepoetin_alfa": _RULE_SETS_DIR / "1_Darbepoetin_alfa.json",
                        "rs_2_Etoposide_Cisplatin": _RULE_SETS_DIR / "2_Etoposide_Cisplatin.json",
                        "rs_3_CALGB9732_Paclitaxel_Cisplatin_Etoposide": _RULE_SETS_DIR / "3_CALGB9732_Paclitaxel_Cisplatin_Etoposide.json",
                        "rs_4_Carboplatin_Etoposide": _RULE_SETS_DIR / "4_Carboplatin_Etoposide.json",
                        "rs_6_Paclitaxel_Carboplatin_Bevacizumab": _RULE_SETS_DIR / "6_Paclitaxel_Carboplatin_Bevacizumab.json",
                        "rs_7_Paclitaxel_Carboplatin": _RULE_SETS_DIR / "7_Paclitaxel_Carboplatin.json",
                        "rs_8_Gemcitabine_Cisplatin": _RULE_SETS_DIR / "8_Gemcitabine_Cisplatin.json",
                    }
                    if rule_set_preset in _preset_map:
                        base_rule_path = _preset_map[rule_set_preset]
                    elif rule_set_preset.startswith("gt_"):
                        gt_folder = rule_set_preset[3:]
                        _RULESET_DIR = Path(settings.BASE_DIR).parent / "sim" / "ruleset_generation"
                        gt_path = _RULESET_DIR / "ground_truth" / gt_folder / "base.json"
                        if gt_path.exists():
                            base_rule_path = gt_path
                if not base_rule_path or not base_rule_path.exists():
                    base_rule_path = DATA_DIR / "rule_set_calibrated_ev302.json"
                    if not base_rule_path.exists():
                        base_rule_path = DATA_DIR / "rule_set.json"
                runner.load_rules(str(base_rule_path))
                import shutil
                shutil.copy2(base_rule_path, run_dir / "rule_set.json")
            else:
                runner.discover_rules()

            runner.write_run_meta(n_patients, n_days, mode, 'generating_patients')

            # Phase 1: Patients
            patients = runner.create_patients_parallel(n_patients, max_workers=10)

            # Phase 2: Daily simulation
            modes = [mode] if mode != "both" else ["natural", "care_ai"]
            for sim_mode in modes:
                runner.write_run_meta(n_patients, n_days, sim_mode, 'running')

                all_results = runner.run_parallel(
                    patients, total_days=n_days, mode=sim_mode, max_workers=10
                )

            # Phase 3: Comparison (if both modes)
            if mode == "both":
                runner.write_run_meta(n_patients, n_days, mode, 'comparing')
                try:
                    from sim.evaluator import run_evaluation
                    run_evaluation(run_dir)
                except Exception as e:
                    print(f"Comparison failed: {e}")

            if runner.is_cancelled:
                runner.write_run_meta(n_patients, n_days, mode, 'cancelled')
                _live_sims[run_name]["status"] = "cancelled"
                runner.log("\u26d4 Simulation cancelled by user")
            else:
                runner.write_run_meta(n_patients, n_days, mode, 'completed')
                _live_sims[run_name]["status"] = "completed"
                runner.log("\U0001f3c1 Simulation completed successfully")

        except Exception as e:
            try:
                runner.write_run_meta(n_patients, n_days, mode, 'failed',
                                     extra={'error': str(e)})
            except Exception:
                pass
            _live_sims[run_name]["status"] = "failed"
            _live_sims[run_name]["error"] = str(e)
            import traceback
            traceback.print_exc()

    # Start background thread
    t = threading.Thread(target=_run_simulation, daemon=True)
    _live_sims[run_name] = {
        "thread": t,
        "status": "starting",
        "run_dir": str(run_dir),
        "config": {
            "drug": drug, "indication": indication,
            "patients": n_patients, "days": n_days,
            "mode": mode, "seed": seed,
        },
    }
    t.start()

    return JsonResponse({
        "run_id": run_name,
        "status": "started",
        "url": f"/trial/{run_name}/",
    })


@require_GET
def api_sim_status(request, run_id: str):
    """Get status of a live simulation run.

    Returns progress, current day per patient, overall status.
    """
    run_path = _get_run_path(run_id)
    meta_path = run_path / "run_meta.json"

    result = {"run_id": run_id, "exists": run_path.exists()}

    # Check in-memory status
    if run_id in _live_sims:
        result["live"] = True
        result["status"] = _live_sims[run_id].get("status", "unknown")
        result["config"] = _live_sims[run_id].get("config", {})

    # Check on-disk metadata
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            result["meta"] = meta
            result["status"] = meta.get("status", result.get("status", "unknown"))
        except Exception:
            pass

    # Count current simulation files
    sim_dir = run_path / "simulations"
    if sim_dir.exists():
        natural_files = list(sim_dir.glob("*_natural.jsonl"))
        care_ai_files = list(sim_dir.glob("*_care_ai.jsonl"))
        result["files"] = {
            "natural": len(natural_files),
            "care_ai": len(care_ai_files),
        }
        # Count latest day from a random file
        if natural_files:
            try:
                with open(natural_files[0]) as f:
                    lines = f.readlines()
                if lines:
                    last = json.loads(lines[-1])
                    result["latest_day"] = last.get("day", 0)
            except Exception:
                pass
    else:
        result["files"] = {"natural": 0, "care_ai": 0}

    # Patient count
    patients_dir = run_path / "patients"
    if patients_dir.exists():
        result["patients_generated"] = len(list(patients_dir.glob("*.json")))
    else:
        result["patients_generated"] = 0

    # If no live status and no meta, it's a completed replay run
    if "status" not in result:
        if run_path.exists() and sim_dir.exists():
            result["status"] = "completed"
        else:
            result["status"] = "not_found"

    # Include log line count for live panel
    if run_id in _live_sims:
        runner = _live_sims[run_id].get("runner")
        if runner:
            result["log_count"] = len(runner._log_lines)

    return JsonResponse(result)


@require_GET
def api_sim_list(request):
    """List all runs with their status (live + completed)."""
    runs = _get_runs()

    result = []
    for run in runs:
        run_id = run["id"]
        run_path = _get_run_path(run_id)
        meta_path = run_path / "run_meta.json"

        entry = {
            "id": run_id,
            "modes": run["modes"],
            "status": "completed",  # default for old runs
        }

        # Check if running
        if run_id in _live_sims:
            entry["status"] = _live_sims[run_id].get("status", "unknown")
            entry["live"] = True

        # Check meta
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                entry["drug_name"] = meta.get("drug_name")
                entry["indication"] = meta.get("indication")
                entry["n_patients"] = meta.get("n_patients")
                entry["total_days"] = meta.get("total_days")
                entry["status"] = meta.get("status", entry["status"])
                entry["started_at"] = meta.get("started_at")
                entry["completed_at"] = meta.get("completed_at")
                if meta.get("status") == "completed" and run_id in _live_sims:
                    del _live_sims[run_id]
                    entry.pop("live", None)
            except Exception:
                pass
        else:
            # Extract info from directory name
            parts = run_id.split("_")
            entry["drug_name"] = run_id

        # Patient count
        patients_dir = run_path / "patients"
        if patients_dir.exists():
            entry["n_patients"] = len(list(patients_dir.glob("*.json")))

        result.append(entry)

    return JsonResponse({"runs": result})


@csrf_exempt
@require_POST
def api_sim_stop(request, run_id: str):
    """Stop a running simulation.

    Signals the runner to cancel, then optionally deletes the run data.
    POST body: {"delete": true/false}
    """
    # Optionally delete run data
    try:
        body = json.loads(request.body) if request.body else {}
    except Exception:
        body = {}

    should_delete = body.get("delete", False)

    if run_id not in _live_sims:
        # Server may have restarted -- _live_sims lost but run dir still exists.
        # Allow delete even if not tracked in memory.
        if should_delete:
            run_path = _get_run_path(run_id)
            if run_path.exists():
                import shutil
                shutil.rmtree(run_path, ignore_errors=True)
            return JsonResponse({"status": "stopped_and_deleted", "run_id": run_id})
        return JsonResponse({"error": "Run not found or not a live simulation"},
                            status=404)

    sim_info = _live_sims[run_id]
    runner = sim_info.get("runner")

    if runner:
        runner.cancel()
        runner.log("\u26d4 Stop requested by user")

    sim_info["status"] = "cancelling"

    if should_delete:
        run_path = _get_run_path(run_id)
        if run_path.exists():
            import shutil
            shutil.rmtree(run_path, ignore_errors=True)
        if run_id in _live_sims:
            del _live_sims[run_id]
        return JsonResponse({"status": "stopped_and_deleted", "run_id": run_id})

    return JsonResponse({"status": "stopping", "run_id": run_id})


@require_GET
def api_sim_log(request, run_id: str):
    """Get real-time log lines for a live simulation.

    Query params:
        since: int -- return lines from this index (default 0)
    Returns: {"lines": [...], "next": int, "status": str}
    """
    since = int(request.GET.get("since", 0))

    result = {"run_id": run_id, "lines": [], "next": since, "status": "unknown"}

    # Try in-memory log first
    if run_id in _live_sims:
        runner = _live_sims[run_id].get("runner")
        result["status"] = _live_sims[run_id].get("status", "unknown")
        if runner:
            lines = runner.get_log(since)
            result["lines"] = lines
            result["next"] = since + len(lines)
            return JsonResponse(result)

    # Fallback: read from disk log
    run_path = _get_run_path(run_id)
    log_path = run_path / "sim_log.txt"
    if log_path.exists():
        try:
            all_lines = log_path.read_text(encoding="utf-8").splitlines()
            result["lines"] = all_lines[since:]
            result["next"] = len(all_lines)
            result["status"] = "completed"
        except Exception:
            pass

    return JsonResponse(result)
