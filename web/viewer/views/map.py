"""
Map views: api_map_meta, api_map_tilemap.
"""
import json
import sys
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from ._helpers import _get_run_path, _list_patients, logger


def _ensure_map_for_run(run_path: Path, n_patients: int) -> tuple[dict, dict]:
    """Generate map for a specific run if needed. Returns (tilemap, meta)."""
    map_dir = run_path / "map"
    meta_path = map_dir / "map_meta.json"
    tilemap_path = map_dir / "tilemap.json"

    if meta_path.exists() and tilemap_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("n_patients", 0) == n_patients:
            with open(tilemap_path) as f:
                tilemap = json.load(f)
            return tilemap, meta

    # Generate map sized for this run's patient count
    tools_dir = Path(settings.BASE_DIR) / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from generate_map import generate_map
    tilemap, meta = generate_map(n_patients)

    map_dir.mkdir(parents=True, exist_ok=True)
    tilemap_path.write_text(json.dumps(tilemap), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return tilemap, meta


@require_GET
def api_map_meta(request, run_id: str):
    """Map metadata (home positions, waypoints) for the tilemap."""
    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return JsonResponse({"error": "not found"}, status=404)

    patient_ids = _list_patients(run_path)
    _, meta = _ensure_map_for_run(run_path, len(patient_ids))
    return JsonResponse(meta)


@require_GET
def api_map_tilemap(request, run_id: str):
    """Serve the Tiled JSON tilemap for a specific run."""
    run_path = _get_run_path(run_id)
    if not run_path.exists():
        return JsonResponse({"error": "not found"}, status=404)

    patient_ids = _list_patients(run_path)
    tilemap, _ = _ensure_map_for_run(run_path, len(patient_ids))
    return JsonResponse(tilemap)
