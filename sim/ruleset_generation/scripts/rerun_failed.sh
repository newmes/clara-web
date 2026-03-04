#!/usr/bin/env bash
# Re-run the 3 drugs that failed before code fixes were applied
set -uo pipefail
VENV_PYTHON="/home/ubuntu/samuel/venv/bin/python"
OUTPUT_DIR="/home/ubuntu/samuel/rule_discovery/rule_sets_improved"
LOG_DIR="/home/ubuntu/samuel/rule_discovery/run_logs"

declare -a DRUGS=(
    "Olaparib|Ovarian cancer"
    "Trastuzumab deruxtecan|HER2-positive breast cancer"
    "Cabozantinib|Renal cell carcinoma"
)

for entry in "${DRUGS[@]}"; do
    IFS='|' read -r DRUG INDICATION <<< "$entry"
    LOG_NAME=$(echo "${DRUG}_${INDICATION}" | tr ' ' '_' | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_+')
    LOG_FILE="$LOG_DIR/${LOG_NAME}_retry.log"
    
    echo "$(date) — Re-running: $DRUG / $INDICATION"
    cd /home/ubuntu/samuel/rule_discovery && \
    "$VENV_PYTHON" -m rule_engine generate-single "$DRUG" "$INDICATION" \
        -o "$OUTPUT_DIR" --multi-stage --verbose \
        > "$LOG_FILE" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "  SUCCESS"
    else
        echo "  FAILED — check $LOG_FILE"
    fi
done
echo "$(date) — Re-run complete"
