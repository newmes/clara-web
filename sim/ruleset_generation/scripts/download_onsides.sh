#!/usr/bin/env bash
# download_onsides.sh — Download OnSIDES v3.1.0 release data and build a SQLite
#                       database for fast ingredient→ADE lookups.
#
# Usage:  bash scripts/download_onsides.sh [PYTHON_PATH]
#
# Output: data/onsides/onsides.db  (SQLite database)
#         data/onsides/csv/        (raw CSV files from release)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${1:-/home/ubuntu/samuel/venv/bin/python}"
DATA_DIR="$PROJECT_DIR/data/onsides"
DB_PATH="$DATA_DIR/onsides.db"
ZIP_URL="https://github.com/tatonetti-lab/onsides/releases/download/v3.1.0/onsides-v3.1.0.zip"

cd "$PROJECT_DIR"

echo "=== OnSIDES Data Setup ==="

# ---------------------------------------------------------------
# Step 1: Download release zip
# ---------------------------------------------------------------
ZIP_PATH="$DATA_DIR/onsides-v3.1.0.zip"
if [ -f "$DB_PATH" ]; then
    echo "OnSIDES database already exists: $DB_PATH"
    echo "Delete it to re-download."
    exit 0
fi

mkdir -p "$DATA_DIR"

if [ -f "$ZIP_PATH" ]; then
    echo "[1/3] Release zip already downloaded."
else
    echo "[1/3] Downloading OnSIDES v3.1.0 release..."
    curl -L -o "$ZIP_PATH" "$ZIP_URL" --progress-bar
    echo "    Downloaded: $(du -h "$ZIP_PATH" | cut -f1)"
fi

# ---------------------------------------------------------------
# Step 2: Extract CSV files
# ---------------------------------------------------------------
CSV_DIR="$DATA_DIR/csv"
if [ -d "$CSV_DIR" ] && [ -f "$CSV_DIR/product_adverse_effect.csv" ]; then
    echo "[2/3] CSV files already extracted."
else
    echo "[2/3] Extracting CSV files..."
    unzip -o -d "$DATA_DIR" "$ZIP_PATH" "csv/*"
    echo "    Extracted to $CSV_DIR"
fi

# ---------------------------------------------------------------
# Step 3: Build SQLite database using Python
# ---------------------------------------------------------------
echo "[3/3] Building SQLite database..."

"$PYTHON" - "$DATA_DIR" <<'PYEOF'
import csv
import sqlite3
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
csv_dir = data_dir / "csv"
db_path = data_dir / "onsides.db"

# Remove stale DB
db_path.unlink(missing_ok=True)

conn = sqlite3.connect(str(db_path))
cur = conn.cursor()

# Create tables
cur.executescript("""
CREATE TABLE vocab_rxnorm_ingredient (
    rxnorm_id TEXT NOT NULL PRIMARY KEY,
    rxnorm_name TEXT NOT NULL,
    rxnorm_term_type TEXT NOT NULL
);
CREATE TABLE vocab_meddra_adverse_effect (
    meddra_id INTEGER NOT NULL PRIMARY KEY,
    meddra_name TEXT NOT NULL,
    meddra_term_type TEXT NOT NULL
);
CREATE TABLE vocab_rxnorm_product (
    rxnorm_id TEXT NOT NULL PRIMARY KEY,
    rxnorm_name TEXT NOT NULL,
    rxnorm_term_type TEXT NOT NULL
);
CREATE TABLE product_label (
    label_id INTEGER NOT NULL PRIMARY KEY,
    source TEXT NOT NULL,
    source_product_name TEXT NOT NULL,
    source_product_id TEXT NOT NULL,
    source_label_url TEXT
);
CREATE TABLE product_adverse_effect (
    product_label_id INTEGER,
    effect_id INTEGER NOT NULL PRIMARY KEY,
    label_section TEXT NOT NULL,
    effect_meddra_id INTEGER,
    match_method TEXT NOT NULL,
    pred0 REAL,
    pred1 REAL
);
CREATE TABLE product_to_rxnorm (
    label_id INTEGER NOT NULL,
    rxnorm_product_id TEXT NOT NULL,
    PRIMARY KEY (label_id, rxnorm_product_id)
);
CREATE TABLE vocab_rxnorm_ingredient_to_product (
    product_id TEXT NOT NULL,
    ingredient_id TEXT NOT NULL,
    PRIMARY KEY (product_id, ingredient_id)
);
""")

def load_csv(table_name, filename, n_cols):
    """Load a CSV into a table, handling variable column counts."""
    path = csv_dir / filename
    placeholders = ",".join(["?"] * n_cols)
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            # Pad short rows, truncate long rows
            row = row[:n_cols]
            while len(row) < n_cols:
                row.append(None)
            # Convert empty strings to None for numeric columns
            rows.append(tuple(v if v != "" else None for v in row))
            if len(rows) >= 50000:
                cur.executemany(f"INSERT OR IGNORE INTO {table_name} VALUES ({placeholders})", rows)
                rows = []
    if rows:
        cur.executemany(f"INSERT OR IGNORE INTO {table_name} VALUES ({placeholders})", rows)
    conn.commit()

print("  Loading vocab_rxnorm_ingredient...")
load_csv("vocab_rxnorm_ingredient", "vocab_rxnorm_ingredient.csv", 3)

print("  Loading vocab_meddra_adverse_effect...")
load_csv("vocab_meddra_adverse_effect", "vocab_meddra_adverse_effect.csv", 3)

print("  Loading vocab_rxnorm_product...")
load_csv("vocab_rxnorm_product", "vocab_rxnorm_product.csv", 3)

print("  Loading product_label...")
load_csv("product_label", "product_label.csv", 5)

print("  Loading product_adverse_effect (large — may take a few minutes)...")
load_csv("product_adverse_effect", "product_adverse_effect.csv", 7)

print("  Loading product_to_rxnorm...")
load_csv("product_to_rxnorm", "product_to_rxnorm.csv", 2)

print("  Loading vocab_rxnorm_ingredient_to_product...")
load_csv("vocab_rxnorm_ingredient_to_product", "vocab_rxnorm_ingredient_to_product.csv", 2)

# Create indexes
print("  Creating indexes...")
cur.executescript("""
CREATE INDEX idx_ingredient_name ON vocab_rxnorm_ingredient(rxnorm_name COLLATE NOCASE);
CREATE INDEX idx_ing_to_prod ON vocab_rxnorm_ingredient_to_product(ingredient_id);
CREATE INDEX idx_prod_to_rxnorm ON product_to_rxnorm(rxnorm_product_id);
CREATE INDEX idx_pae_label ON product_adverse_effect(product_label_id);
CREATE INDEX idx_pae_meddra ON product_adverse_effect(effect_meddra_id);
CREATE INDEX idx_pae_section ON product_adverse_effect(label_section);
""")

# Build materialized ingredient-ADE summary table
print("  Building ingredient_ade_summary (this is the slow step)...")
cur.executescript("""
CREATE TABLE ingredient_ade_summary AS
SELECT
    ri.rxnorm_id AS ingredient_rxnorm_id,
    ri.rxnorm_name AS ingredient_name,
    mae.meddra_id AS pt_meddra_id,
    mae.meddra_name AS pt_meddra_term,
    COUNT(DISTINCT pae.product_label_id) AS label_count,
    AVG(pae.pred1) AS mean_pred_score,
    MAX(pae.pred1) AS max_pred_score,
    SUM(CASE WHEN pae.label_section = 'BW' THEN 1 ELSE 0 END) > 0 AS is_boxed_warning
FROM vocab_rxnorm_ingredient ri
JOIN vocab_rxnorm_ingredient_to_product ip ON ri.rxnorm_id = ip.ingredient_id
JOIN product_to_rxnorm ptr ON ip.product_id = ptr.rxnorm_product_id
JOIN product_adverse_effect pae ON ptr.label_id = pae.product_label_id
JOIN vocab_meddra_adverse_effect mae ON pae.effect_meddra_id = mae.meddra_id
WHERE mae.meddra_term_type = 'PT'
GROUP BY ri.rxnorm_id, ri.rxnorm_name, mae.meddra_id, mae.meddra_name;

CREATE INDEX idx_ias_ingredient ON ingredient_ade_summary(ingredient_name COLLATE NOCASE);
""")

row_count = cur.execute("SELECT COUNT(*) FROM ingredient_ade_summary").fetchone()[0]
ing_count = cur.execute("SELECT COUNT(DISTINCT ingredient_name) FROM ingredient_ade_summary").fetchone()[0]

conn.close()
print(f"  Database built: {db_path}")
print(f"  Summary table: {row_count} ingredient-ADE pairs across {ing_count} ingredients")
PYEOF

echo ""
echo "=== OnSIDES Setup Complete ==="
