"""Augment base PrimeKG edges with T-cell exhaustion marker connections.

Reads the base edges from data/primekg/edges.csv, adds edges connecting
the T_cell_exhaustion node (index 154753) to known exhaustion marker
genes, and writes to data/primekg_augmented/edges_exhaustion_augmented.csv.

Also generates data/primekg_augmented/nodes_exhaustion_augmented.csv if it
doesn't already exist (copies base nodes and appends the T_cell_exhaustion node).
"""

from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).parent.parent
BASE_NODES = PROJECT / "data" / "primekg" / "nodes.csv"
BASE_EDGES = PROJECT / "data" / "primekg" / "edges.csv"
OUT_DIR = PROJECT / "data" / "primekg_augmented"
OUT_NODES = OUT_DIR / "nodes_exhaustion_augmented.csv"
OUT_EDGES = OUT_DIR / "edges_exhaustion_augmented.csv"

T_CELL_EXHAUSTION_INDEX = 154753

# Known T-cell exhaustion marker genes
EXHAUSTION_MARKERS = [
    "PDCD1",   # PD-1
    "HAVCR2",  # TIM-3
    "LAG3",
    "TIGIT",
    "CTLA4",
    "TOX",
    "TOX2",
    "ENTPD1",  # CD39
    "CD244",   # 2B4
    "BTLA",
    "CD160",
    "EOMES",
    "TBX21",   # T-bet
    "PRDM1",   # Blimp-1
    "IRF4",
    "BATF",
    "NR4A1",
    "NR4A2",
    "NR4A3",
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not BASE_NODES.exists() or not BASE_EDGES.exists():
        raise FileNotFoundError(
            f"Base PrimeKG files not found. Run process_primekg.py first.\n"
            f"  Expected: {BASE_NODES}\n  Expected: {BASE_EDGES}"
        )

    # Load base data
    nodes_df = pd.read_csv(BASE_NODES)
    edges_df = pd.read_csv(BASE_EDGES)

    print(f"Base nodes: {len(nodes_df):,}")
    print(f"Base edges: {len(edges_df):,}")

    # Build gene name → node_index lookup
    gene_nodes = nodes_df[nodes_df["node_type"] == "gene/protein"]
    name_to_idx = dict(zip(gene_nodes["node_name"].str.upper(), gene_nodes["node_index"], strict=False))

    # Augment nodes — add T_cell_exhaustion if not present
    if T_CELL_EXHAUSTION_INDEX not in nodes_df["node_index"].values:
        new_node = pd.DataFrame([{
            "node_index": T_CELL_EXHAUSTION_INDEX,
            "node_type": "biological_process",
            "node_name": "T_cell_exhaustion",
            "node_id_str": "GO:T_cell_exhaustion",
        }])
        nodes_df = pd.concat([nodes_df, new_node], ignore_index=True)
        print(f"Added T_cell_exhaustion node at index {T_CELL_EXHAUSTION_INDEX}")

    # Add exhaustion marker edges
    new_edges = []
    next_edge_idx = edges_df["edge_index"].max() + 1

    for marker in EXHAUSTION_MARKERS:
        gene_idx = name_to_idx.get(marker.upper())
        if gene_idx is None:
            print(f"  Warning: gene {marker} not found in nodes, skipping")
            continue

        # Forward edge: gene → exhaustion process
        new_edges.append({
            "edge_index": next_edge_idx,
            "x_index": gene_idx,
            "y_index": T_CELL_EXHAUSTION_INDEX,
            "x_type": "gene/protein",
            "y_type": "biological_process",
            "relation": "associated_with",
        })
        next_edge_idx += 1

        # Reverse edge: exhaustion process → gene
        new_edges.append({
            "edge_index": next_edge_idx,
            "x_index": T_CELL_EXHAUSTION_INDEX,
            "y_index": gene_idx,
            "x_type": "biological_process",
            "y_type": "gene/protein",
            "relation": "rev_associated_with",
        })
        next_edge_idx += 1

    if new_edges:
        new_edges_df = pd.DataFrame(new_edges)
        edges_df = pd.concat([edges_df, new_edges_df], ignore_index=True)
        print(f"Added {len(new_edges)} exhaustion marker edges ({len(new_edges) // 2} genes)")

    # Save
    nodes_df.to_csv(OUT_NODES, index=False)
    edges_df.to_csv(OUT_EDGES, index=False)
    print(f"Saved {OUT_NODES} ({len(nodes_df):,} nodes)")
    print(f"Saved {OUT_EDGES} ({len(edges_df):,} edges)")


if __name__ == "__main__":
    main()
