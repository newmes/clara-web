#!/usr/bin/env bash
# run_multi_example.sh — Example multi-indication pipeline run
#
# Generates a unified rule set for a patient treated across 3 lung cancer indications:
#   1. Cisplatin + Etoposide | SCLC
#   2. Paclitaxel + Carboplatin + Bevacizumab | NSCLC
#   3. Gemcitabine + Cisplatin | Squamous NSCLC
#
# Usage:
#   bash scripts/run_multi_example.sh
#
# Prerequisites:
#   - RULE_ENGINE_LLM_API_KEY set (Gemini API key)
#   - Python venv activated or use full path
#   - Data files in place (data/drugbank/, data/primekg/, data/onsides/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/rule_sets_multi"
VENV_PYTHON="${VENV_PYTHON:-/home/ubuntu/samuel/venv/bin/python}"

echo "=== Multi-Indication Pipeline Example ==="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create input JSON
INPUT_FILE=$(mktemp /tmp/multi_input_XXXXXX.json)
cat > "$INPUT_FILE" << 'EOF'
{
  "regimens": [
    {"drugs": ["Cisplatin", "Etoposide"], "indication": "SCLC"},
    {"drugs": ["Paclitaxel", "Carboplatin", "Bevacizumab"], "indication": "NSCLC"},
    {"drugs": ["Gemcitabine", "Cisplatin"], "indication": "Squamous NSCLC"}
  ]
}
EOF

echo "Input file: $INPUT_FILE"
cat "$INPUT_FILE"
echo ""

# Run multi-indication pipeline
cd "$PROJECT_ROOT"
"$VENV_PYTHON" -m rule_engine generate-multi "$INPUT_FILE" \
    -o "$OUTPUT_DIR" \
    --verbose

echo ""
echo "=== Pipeline Complete ==="
echo "Output files:"
ls -la "$OUTPUT_DIR"/multi_* 2>/dev/null || echo "  (no output files found)"

# Run hallucination analysis on output
echo ""
echo "=== Hallucination Analysis ==="
"$VENV_PYTHON" "$PROJECT_ROOT/scripts/analyze_hallucinations.py" "$OUTPUT_DIR/" 2>/dev/null || echo "  (analysis script not available or no files to analyze)"

# Cleanup
rm -f "$INPUT_FILE"
