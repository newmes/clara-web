#!/usr/bin/env bash
# auto_validate_runs.sh — Phase 2 natural 완료를 감지하여 자동 검증 실행
#
# Usage: bash validation/auto_validate_runs.sh [expected_patients=200]

set -uo pipefail
cd "$(dirname "$0")/.."

EXPECTED=${1:-200}
POLL_SEC=60

RUNS=(
  "data/runs/20260220_000413_Padcev___Pembrolizumab_200pt_126d"
  "data/runs/20260220_000418_Darbepoetin_alfa_200pt_126d"
  "data/runs/20260220_000422_Etoposide___Cisplatin_200pt_126d"
)

declare -A DONE

for run in "${RUNS[@]}"; do
  DONE["$run"]=0
done

all_done() {
  for run in "${RUNS[@]}"; do
    [[ "${DONE[$run]}" -eq 0 ]] && return 1
  done
  return 0
}

run_validation() {
  local run_dir="$1"
  local name
  name=$(basename "$run_dir")
  echo ""
  echo "================================================================"
  echo "[$(date '+%H:%M:%S')] VALIDATING: $name"
  echo "================================================================"

  echo "[1/2] extract_sim_stats.py ..."
  python validation/extract_sim_stats.py "$run_dir" --mode natural \
    --output "$run_dir/validation/validation_stats_natural.json" 2>&1 || true

  echo "[2/2] validate_vs_ruleset.py ..."
  python validation/validate_vs_ruleset.py "$run_dir" --mode natural 2>&1 || true

  echo "[$(date '+%H:%M:%S')] Validation complete: $name"
  echo "  Reports: $run_dir/validation/"
  echo ""
}

echo "Auto-validation monitor started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Expected patients per run: $EXPECTED"
echo "Polling interval: ${POLL_SEC}s"
echo "Monitoring ${#RUNS[@]} runs:"
for run in "${RUNS[@]}"; do
  echo "  - $(basename "$run")"
done
echo ""

while ! all_done; do
  for run in "${RUNS[@]}"; do
    [[ "${DONE[$run]}" -eq 1 ]] && continue

    name=$(basename "$run")
    if [[ -d "$run/simulations" ]]; then
      nat_count=$(find "$run/simulations" -name "*_natural.jsonl" 2>/dev/null | wc -l)
    else
      nat_count=0
    fi
    if [[ -d "$run/patients" ]]; then
      patients=$(find "$run/patients" -name "*.json" 2>/dev/null | wc -l)
    else
      patients=0
    fi

    if [[ "$nat_count" -ge "$EXPECTED" ]]; then
      echo "[$(date '+%H:%M:%S')] $name — natural COMPLETE ($nat_count/$EXPECTED)"
      DONE["$run"]=1
      run_validation "$run"
    else
      echo "[$(date '+%H:%M:%S')] $name — patients=$patients, natural=$nat_count/$EXPECTED"
    fi
  done

  all_done && break
  sleep "$POLL_SEC"
done

echo ""
echo "================================================================"
echo "ALL 3 RUNS VALIDATED — $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""
echo "Summary:"
for run in "${RUNS[@]}"; do
  name=$(basename "$run")
  report="$run/validation/ruleset_validation_natural_v4.md"
  if [[ -f "$report" ]]; then
    grade=$(grep -oP 'Overall:.*\*\*(\w+)\*\*' "$report" | head -1 || echo "?")
    echo "  $name: $grade"
    echo "    Report: $report"
  else
    echo "  $name: report not found"
  fi
done
