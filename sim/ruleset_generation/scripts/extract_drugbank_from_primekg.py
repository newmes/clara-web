"""Extract DrugBank-format CSVs from PrimeKG raw data.

PrimeKG's kg.csv contains drug-protein and drug-drug edges originally
sourced from DrugBank. This script extracts them into the CSV format
expected by rule_engine/evidence/local_dbs.py:

  data/drugbank/drugbank_vocabulary.csv  — drug name ↔ ID mapping
  data/drugbank/drug_protein.csv         — drug-protein interactions
  data/drugbank/drug_drug.csv            — drug-drug interactions
  data/drugbank/moa.csv                  — empty template (MoA requires DrugBank XML license)
"""

from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).parent.parent
RAW_KG = PROJECT / "data" / "primekg_raw" / "kg.csv"
OUT_DIR = PROJECT / "data" / "drugbank"


def main():
    if not RAW_KG.exists():
        raise FileNotFoundError(f"PrimeKG raw data not found: {RAW_KG}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {RAW_KG}...")
    df = pd.read_csv(RAW_KG, low_memory=False)
    print(f"Loaded {len(df):,} edges")

    # --- 1. Build drug vocabulary from drug nodes ---
    print("Extracting drug vocabulary...")
    x_drugs = df[df["x_type"] == "drug"][["x_id", "x_name", "x_source"]].rename(
        columns={"x_id": "id", "x_name": "name", "x_source": "source"}
    )
    y_drugs = df[df["y_type"] == "drug"][["y_id", "y_name", "y_source"]].rename(
        columns={"y_id": "id", "y_name": "name", "y_source": "source"}
    )
    drugs = pd.concat([x_drugs, y_drugs]).drop_duplicates(subset=["id"])

    # Create vocabulary CSV matching the format expected by local_dbs.py
    vocab = pd.DataFrame({
        "DrugBank ID": drugs["id"].apply(lambda x: f"DB{int(x):05d}" if str(x).isdigit() else str(x)),
        "Common name": drugs["name"],
        "Synonyms": "",  # PrimeKG doesn't have synonyms
    })
    vocab_path = OUT_DIR / "drugbank_vocabulary.csv"
    vocab.to_csv(vocab_path, index=False)
    print(f"  Saved {len(vocab):,} drugs to {vocab_path}")

    # --- 2. Extract drug-protein edges ---
    print("Extracting drug-protein interactions...")
    dp_mask = (
        ((df["x_type"] == "drug") & (df["y_type"] == "gene/protein"))
        | ((df["x_type"] == "gene/protein") & (df["y_type"] == "drug"))
    )
    dp_edges = df[dp_mask].copy()

    drug_protein_rows = []
    for _, row in dp_edges.iterrows():
        if row["x_type"] == "drug":
            drug_id = str(row["x_id"])
            drug_name = row["x_name"]
            protein_id = str(row["y_id"])
            protein_name = row["y_name"]
        else:
            drug_id = str(row["y_id"])
            drug_name = row["y_name"]
            protein_id = str(row["x_id"])
            protein_name = row["x_name"]

        drugbank_id = f"DB{int(drug_id):05d}" if drug_id.isdigit() else drug_id
        drug_protein_rows.append({
            "DrugBank": drugbank_id,
            "DrugName": drug_name,
            "UniProtID": protein_id,
            "UniProtName": protein_name,
            "NCBIGeneID": "",
            "relation": row.get("display_relation", row.get("relation", "")),
        })

    dp_df = pd.DataFrame(drug_protein_rows).drop_duplicates(subset=["DrugBank", "UniProtID"])
    dp_path = OUT_DIR / "drug_protein.csv"
    dp_df.to_csv(dp_path, index=False)
    print(f"  Saved {len(dp_df):,} drug-protein interactions to {dp_path}")

    # --- 3. Extract drug-drug edges ---
    print("Extracting drug-drug interactions...")
    dd_mask = (df["x_type"] == "drug") & (df["y_type"] == "drug")
    dd_edges = df[dd_mask].copy()

    drug_drug_rows = []
    for _, row in dd_edges.iterrows():
        id1 = str(row["x_id"])
        id2 = str(row["y_id"])
        db1 = f"DB{int(id1):05d}" if id1.isdigit() else id1
        db2 = f"DB{int(id2):05d}" if id2.isdigit() else id2
        drug_drug_rows.append({
            "drug1": db1,
            "drug2": db2,
            "name1": row["x_name"],
            "name2": row["y_name"],
            "relation": row.get("display_relation", row.get("relation", "")),
        })

    dd_df = pd.DataFrame(drug_drug_rows).drop_duplicates(subset=["drug1", "drug2"])
    dd_path = OUT_DIR / "drug_drug.csv"
    dd_df.to_csv(dd_path, index=False)
    print(f"  Saved {len(dd_df):,} drug-drug interactions to {dd_path}")

    # --- 4. Empty MoA template ---
    moa_path = OUT_DIR / "moa.csv"
    pd.DataFrame(columns=["ID", "MOA"]).to_csv(moa_path, index=False)
    print(f"  Created empty MoA template at {moa_path}")

    print("\nDone! DrugBank CSVs extracted from PrimeKG.")


if __name__ == "__main__":
    main()
