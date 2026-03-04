#!/usr/bin/env bash
# setup.sh — One-click setup for the Rule Discovery pipeline.
#
# Usage:
#   bash setup.sh           # Full setup: venv + deps + download all data (~7 GB)
#   bash setup.sh --no-data # Quick mode: venv + deps only (data already on disk)
#
# Environment variables:
#   RULE_ENGINE_LLM_API_KEY      Gemini API key (required to run pipeline)
#   RULE_ENGINE_PDS_USERNAME     PDS credentials (optional — enables PDS download)
#   RULE_ENGINE_PDS_PASSWORD     PDS credentials (optional — enables PDS download)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
NO_DATA=false

# Parse arguments
for arg in "$@"; do
    case "$arg" in
        --no-data) NO_DATA=true ;;
        -h|--help)
            echo "Usage: bash setup.sh [--no-data]"
            echo ""
            echo "  --no-data   Skip data downloads (only set up Python env + deps)"
            echo "              Use this when data/ is already on disk."
            echo ""
            echo "Environment variables:"
            echo "  RULE_ENGINE_PDS_USERNAME   PDS credentials (optional, enables PDS trial download)"
            echo "  RULE_ENGINE_PDS_PASSWORD   PDS credentials (optional, enables PDS trial download)"
            exit 0
            ;;
    esac
done

# Detect PDS credentials
PDS_AVAILABLE=false
if [ -n "${RULE_ENGINE_PDS_USERNAME:-}" ] && [ -n "${RULE_ENGINE_PDS_PASSWORD:-}" ]; then
    PDS_AVAILABLE=true
fi

if [ "$NO_DATA" = true ]; then
    echo "============================================"
    echo "  Rule Discovery — Quick Setup (no data)"
    echo "============================================"
    TOTAL_STEPS=2
else
    if [ "$PDS_AVAILABLE" = true ]; then
        TOTAL_STEPS=7
    else
        TOTAL_STEPS=6
    fi
    echo "============================================"
    echo "  Rule Discovery — Full Setup"
    echo "============================================"
fi
echo ""

# ---------------------------------------------------------------
# Step 1: Python virtual environment
# ---------------------------------------------------------------
if [ -d "$VENV_DIR" ] && [ -f "$PYTHON" ]; then
    echo "[1/$TOTAL_STEPS] Virtual environment already exists: $VENV_DIR/"
else
    echo "[1/$TOTAL_STEPS] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "    Created: $VENV_DIR/"
fi
echo ""

# ---------------------------------------------------------------
# Step 2: Install Python dependencies
# ---------------------------------------------------------------
echo "[2/$TOTAL_STEPS] Installing Python dependencies..."
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet -r requirements.txt
if [ "$PDS_AVAILABLE" = true ]; then
    echo "    Installing PDS dependency (swat)..."
    "$PIP" install --quiet swat
fi
echo "    Done."
echo ""

# ---------------------------------------------------------------
# Data setup (skipped with --no-data)
# ---------------------------------------------------------------
if [ "$NO_DATA" = true ]; then
    # Quick status check for --no-data mode
    echo "Checking existing data files..."
    MISSING=0
    for f in \
        "data/primekg/nodes.csv" \
        "data/primekg/edges.csv" \
        "data/drugbank/drugbank_vocabulary.csv" \
        "data/onsides/onsides.db" \
        "data/primekg_augmented/edges_exhaustion_augmented.csv"; do
        if [ -f "$f" ]; then
            printf "  ✓ %s (%s)\n" "$f" "$(du -h "$f" | cut -f1)"
        else
            printf "  ✗ %s  MISSING\n" "$f"
            MISSING=$((MISSING + 1))
        fi
    done
    # Check PDS separately
    if [ -f "data/pds/trial_index.csv" ]; then
        N_TRIALS=$(tail -n +2 data/pds/trial_index.csv | wc -l | tr -d ' ')
        printf "  ✓ data/pds/ (%s trials)\n" "$N_TRIALS"
    else
        printf "  - data/pds/  (not downloaded — optional)\n"
    fi
    echo ""
    if [ "$MISSING" -gt 0 ]; then
        echo "WARNING: $MISSING data file(s) missing. Run 'bash setup.sh' (without --no-data) to download."
        echo ""
    fi
else
    # ---------------------------------------------------------------
    # Step 3: Download & process PrimeKG
    # ---------------------------------------------------------------
    PRIMEKG_RAW="data/primekg_raw/kg.csv"
    PRIMEKG_NODES="data/primekg/nodes.csv"
    PRIMEKG_EDGES="data/primekg/edges.csv"

    if [ -f "$PRIMEKG_NODES" ] && [ -f "$PRIMEKG_EDGES" ]; then
        echo "[3/$TOTAL_STEPS] PrimeKG already processed:"
        echo "    $PRIMEKG_NODES ($(wc -l < "$PRIMEKG_NODES") lines)"
        echo "    $PRIMEKG_EDGES ($(wc -l < "$PRIMEKG_EDGES") lines)"
    else
        if [ -f "$PRIMEKG_RAW" ]; then
            echo "[3/$TOTAL_STEPS] PrimeKG raw data exists, processing..."
        else
            echo "[3/$TOTAL_STEPS] Downloading PrimeKG from Harvard Dataverse (~900 MB)..."
            mkdir -p data/primekg_raw
            curl -L -o "$PRIMEKG_RAW" \
                "https://dataverse.harvard.edu/api/access/datafile/6180620" \
                --progress-bar
            echo "    Downloaded: $(du -h "$PRIMEKG_RAW" | cut -f1)"
        fi
        echo "    Processing into nodes/edges..."
        "$PYTHON" scripts/process_primekg.py
        echo "    Done."
    fi
    echo ""

    # ---------------------------------------------------------------
    # Step 4: Extract DrugBank CSVs + augment KG
    # ---------------------------------------------------------------
    DB_VOCAB="data/drugbank/drugbank_vocabulary.csv"
    AUG_EDGES="data/primekg_augmented/edges_exhaustion_augmented.csv"

    if [ -f "$DB_VOCAB" ]; then
        echo "[4/$TOTAL_STEPS] DrugBank CSVs already exist."
    else
        echo "[4/$TOTAL_STEPS] Extracting DrugBank CSVs from PrimeKG..."
        "$PYTHON" scripts/extract_drugbank_from_primekg.py
    fi

    if [ -f "$AUG_EDGES" ]; then
        echo "      Augmented KG edges already exist."
    else
        echo "      Generating augmented KG edges..."
        "$PYTHON" scripts/augment_primekg_exhaustion.py
    fi
    echo ""

    # ---------------------------------------------------------------
    # Step 5: Download & build OnSIDES database
    # ---------------------------------------------------------------
    ONSIDES_DB="data/onsides/onsides.db"

    if [ -f "$ONSIDES_DB" ]; then
        echo "[5/$TOTAL_STEPS] OnSIDES database already exists: $ONSIDES_DB"
    else
        echo "[5/$TOTAL_STEPS] Setting up OnSIDES (download + build SQLite DB)..."
        echo "      This step downloads ~330 MB and builds a ~2 GB database."
        echo "      It may take 5-10 minutes."
        bash scripts/download_onsides.sh "$PYTHON"
    fi
    echo ""

    # ---------------------------------------------------------------
    # Step 6 (conditional): Download PDS trial data
    # ---------------------------------------------------------------
    if [ "$PDS_AVAILABLE" = true ]; then
        PDS_INDEX="data/pds/trial_index.csv"
        if [ -f "$PDS_INDEX" ]; then
            N_TRIALS=$(tail -n +2 "$PDS_INDEX" | wc -l | tr -d ' ')
            echo "[6/$TOTAL_STEPS] PDS trial data already downloaded ($N_TRIALS trials)"
        else
            echo "[6/$TOTAL_STEPS] Downloading PDS trial data from Project Data Sphere..."
            echo "      Credentials detected (RULE_ENGINE_PDS_USERNAME)."
            echo "      Downloading SCLC trials..."
            "$PYTHON" scripts/pds_download.py || {
                echo "    WARNING: SCLC download failed (check credentials). Continuing..."
            }
            echo "      Downloading NSCLC trials..."
            "$PYTHON" scripts/pds_download_nsclc.py || {
                echo "    WARNING: NSCLC download failed (check credentials). Continuing..."
            }
        fi
        echo ""
    else
        echo "[--] PDS: Skipped (no credentials)."
        echo "    To include PDS patient-level data (improves accuracy for SCLC/NSCLC):"
        echo "    1. Register at https://projectdatasphere.org"
        echo "    2. Re-run with credentials:"
        echo "       RULE_ENGINE_PDS_USERNAME=you@email.com RULE_ENGINE_PDS_PASSWORD=pass bash setup.sh"
        echo ""
    fi

    # ---------------------------------------------------------------
    # Healthcheck
    # ---------------------------------------------------------------
    HEALTH_STEP=$TOTAL_STEPS
    echo "[$HEALTH_STEP/$TOTAL_STEPS] Running healthcheck..."
    if "$PYTHON" -m rule_engine healthcheck 2>/dev/null; then
        echo "    Healthcheck passed."
    else
        echo "    Healthcheck had warnings (this is OK if API key is not set yet)."
    fi
    echo ""
fi

# ---------------------------------------------------------------
# Status report
# ---------------------------------------------------------------
echo "============================================"
echo "  Setup Complete"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Set your Gemini API key:"
echo "     export RULE_ENGINE_LLM_API_KEY=\"your-key\""
echo ""
echo "  2. Generate a rule set:"
echo "     $PYTHON -m rule_engine generate-single \"Etoposide+Cisplatin\" \"Small Cell Lung Cancer\" -o output --multi-stage"
echo ""
echo "  3. Validate output:"
echo "     $PYTHON scripts/validate_target_schema.py output/"
echo "     $PYTHON scripts/analyze_hallucinations.py output/"
