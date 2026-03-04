#!/usr/bin/env bash
# setup_evidence_data.sh — Download and process all evidence data files.
#
# Usage:  bash scripts/setup_evidence_data.sh [PYTHON_PATH]
#
# Steps:
#   1. Download PrimeKG kg.csv from Harvard Dataverse
#   2. Run process_primekg.py → base nodes/edges
#   3. Run augment_primekg_exhaustion.py → exhaustion-augmented edges
#   4. Run extract_drugbank_from_primekg.py → DrugBank CSVs
#   5. Report status
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${1:-/home/ubuntu/samuel/venv/bin/python}"

cd "$PROJECT_DIR"

echo "=== Evidence Data Setup ==="
echo "Project: $PROJECT_DIR"
echo "Python:  $PYTHON"
echo ""

# ---------------------------------------------------------------
# Step 1: Download PrimeKG raw data
# ---------------------------------------------------------------
PRIMEKG_RAW="data/primekg_raw/kg.csv"
if [ -f "$PRIMEKG_RAW" ]; then
    echo "[1/4] PrimeKG raw data already exists: $PRIMEKG_RAW"
else
    echo "[1/4] Downloading PrimeKG kg.csv from Harvard Dataverse..."
    mkdir -p data/primekg_raw
    # PrimeKG dataset DOI: 10.7910/DVN/IXA7BM — kg.csv file ID
    curl -L -o "$PRIMEKG_RAW" \
        "https://dataverse.harvard.edu/api/access/datafile/6180620" \
        --progress-bar
    echo "    Downloaded: $(wc -l < "$PRIMEKG_RAW") lines"
fi
echo ""

# ---------------------------------------------------------------
# Step 2: Process into base nodes/edges
# ---------------------------------------------------------------
BASE_NODES="data/primekg/nodes.csv"
BASE_EDGES="data/primekg/edges.csv"
if [ -f "$BASE_NODES" ] && [ -f "$BASE_EDGES" ]; then
    echo "[2/4] Base PrimeKG files already exist:"
    echo "    $BASE_NODES ($(wc -l < "$BASE_NODES") lines)"
    echo "    $BASE_EDGES ($(wc -l < "$BASE_EDGES") lines)"
else
    echo "[2/4] Processing PrimeKG into base nodes/edges..."
    "$PYTHON" scripts/process_primekg.py
fi
echo ""

# ---------------------------------------------------------------
# Step 3: Generate exhaustion-augmented edges
# ---------------------------------------------------------------
AUG_EDGES="data/primekg_augmented/edges_exhaustion_augmented.csv"
if [ -f "$AUG_EDGES" ]; then
    echo "[3/4] Augmented edges already exist: $AUG_EDGES"
else
    echo "[3/4] Generating exhaustion-augmented edges..."
    "$PYTHON" scripts/augment_primekg_exhaustion.py
fi
echo ""

# ---------------------------------------------------------------
# Step 4: Extract DrugBank CSVs from PrimeKG
# ---------------------------------------------------------------
DB_VOCAB="data/drugbank/drugbank_vocabulary.csv"
if [ -f "$DB_VOCAB" ]; then
    echo "[4/4] DrugBank CSVs already exist in data/drugbank/"
else
    echo "[4/4] Extracting DrugBank CSVs from PrimeKG..."
    "$PYTHON" scripts/extract_drugbank_from_primekg.py
fi
echo ""

# ---------------------------------------------------------------
# Status report
# ---------------------------------------------------------------
echo "=== Data File Status ==="
check_file() {
    if [ -f "$1" ]; then
        lines=$(wc -l < "$1")
        printf "  ✓ %-50s (%s lines)\n" "$1" "$lines"
    else
        printf "  ✗ %-50s MISSING\n" "$1"
    fi
}

check_file "data/primekg_raw/kg.csv"
check_file "data/primekg/nodes.csv"
check_file "data/primekg/edges.csv"
check_file "data/primekg_augmented/nodes_exhaustion_augmented.csv"
check_file "data/primekg_augmented/edges_exhaustion_augmented.csv"
check_file "data/drugbank/drugbank_vocabulary.csv"
check_file "data/drugbank/drug_protein.csv"
check_file "data/drugbank/drug_drug.csv"
check_file "data/drugbank/moa.csv"

echo ""
echo "=== Setup Complete ==="
