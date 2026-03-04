#!/usr/bin/env bash
# run_batch_10.sh — Run 10 new drugs through the multi-stage pipeline sequentially.
#
# Usage:  bash scripts/run_batch_10.sh
# Run in tmux to avoid disconnection. Each drug takes ~18-20 min.
# Total estimated time: ~3 hours.
#
# Logs saved to: run_logs/<drug>_<indication>.log
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/run_logs"
VENV_PYTHON="/home/ubuntu/samuel/venv/bin/python"
OUTPUT_DIR="$PROJECT_ROOT/rule_sets_improved"

mkdir -p "$LOG_DIR"

# 10 new drug-indication pairs
declare -a DRUGS=(
    "Olaparib|Ovarian cancer"
    "Temozolomide|Glioblastoma"
    "Trastuzumab deruxtecan|HER2-positive breast cancer"
    "Sorafenib|Hepatocellular carcinoma"
    "Cabozantinib|Renal cell carcinoma"
    "Gemcitabine+Capecitabine|Pancreatic cancer"
    "Enfortumab vedotin|Urothelial carcinoma"
    "Dabrafenib+Trametinib|Cholangiocarcinoma"
    "Gilteritinib|Acute myeloid leukemia"
    "Erdafitinib|Urothelial carcinoma"
)

TOTAL=${#DRUGS[@]}
PASSED=0
FAILED=0
START_TIME=$(date +%s)

echo "=============================================="
echo "  Batch Run: 10 New Drugs (Multi-Stage)"
echo "  Started: $(date)"
echo "=============================================="
echo ""

for i in "${!DRUGS[@]}"; do
    IFS='|' read -r DRUG INDICATION <<< "${DRUGS[$i]}"
    NUM=$((i + 1))

    # Create a log-friendly filename
    LOG_NAME=$(echo "${DRUG}_${INDICATION}" | tr ' ' '_' | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_+')
    LOG_FILE="$LOG_DIR/${LOG_NAME}.log"

    echo "----------------------------------------------"
    echo "[$NUM/$TOTAL] $DRUG — $INDICATION"
    echo "  Log: $LOG_FILE"
    echo "  Started: $(date)"
    echo "----------------------------------------------"

    DRUG_START=$(date +%s)

    cd "$PROJECT_ROOT" && \
    "$VENV_PYTHON" -m rule_engine generate-single "$DRUG" "$INDICATION" \
        -o "$OUTPUT_DIR" --multi-stage --verbose \
        > "$LOG_FILE" 2>&1

    EXIT_CODE=$?
    DRUG_END=$(date +%s)
    DRUG_ELAPSED=$(( DRUG_END - DRUG_START ))
    DRUG_MIN=$(( DRUG_ELAPSED / 60 ))
    DRUG_SEC=$(( DRUG_ELAPSED % 60 ))

    if [ $EXIT_CODE -eq 0 ]; then
        echo "  Result: SUCCESS (${DRUG_MIN}m ${DRUG_SEC}s)"
        PASSED=$((PASSED + 1))
    else
        echo "  Result: FAILED (exit code $EXIT_CODE, ${DRUG_MIN}m ${DRUG_SEC}s)"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( END_TIME - START_TIME ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SEC=$(( TOTAL_ELAPSED % 60 ))

echo "=============================================="
echo "  Batch Complete: $(date)"
echo "  Total time: ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "  Passed: $PASSED/$TOTAL"
echo "  Failed: $FAILED/$TOTAL"
echo "=============================================="
