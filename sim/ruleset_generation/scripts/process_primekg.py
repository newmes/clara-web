from pathlib import Path

import pandas as pd

# Configuration
DATA_DIR = Path("data/primekg_raw")
OUTPUT_DIR = Path("data/primekg")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_primekg():
    csv_path = DATA_DIR / "kg.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    # 1. Read CSV
    print(f"Reading {csv_path}...")
    # PrimeKG header: relation,display_relation,x_index,x_id,x_type,x_name,x_source,y_index,y_id,y_type,y_name,y_source
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df):,} edges")

    # 2. Extract Nodes
    print("Extracting nodes...")
    # We need unique nodes. We can use x_index/y_index if they are unique per entity,
    # but to be safe and consistent with PROTON, let's build from (type, name, id) tuples.

    # Extract source nodes (x)
    x_nodes = df[["x_index", "x_type", "x_name", "x_id"]].rename(
        columns={"x_index": "org_index", "x_type": "node_type", "x_name": "node_name", "x_id": "node_id_str"}
    )

    # Extract target nodes (y)
    y_nodes = df[["y_index", "y_type", "y_name", "y_id"]].rename(
        columns={"y_index": "org_index", "y_type": "node_type", "y_name": "node_name", "y_id": "node_id_str"}
    )

    # Concatenate and unique
    all_nodes = pd.concat([x_nodes, y_nodes]).drop_duplicates(subset=["org_index"]).reset_index(drop=True)

    # We will re-index to ensure 0..N continuous range
    all_nodes["node_index"] = range(len(all_nodes))

    # Ensure node_id_str is string
    all_nodes["node_id_str"] = all_nodes["node_id_str"].astype(str)

    print(f"Found {len(all_nodes):,} unique nodes")

    # Create mapping: org_index -> new_node_index
    id_map = dict(zip(all_nodes["org_index"], all_nodes["node_index"], strict=False))

    # 3. Process Edges
    print("Processing edges...")
    # Map raw indices to new sorted continuous indices
    df["x_index_new"] = df["x_index"].map(id_map)
    df["y_index_new"] = df["y_index"].map(id_map)

    edges_df = pd.DataFrame()
    edges_df["x_index"] = df["x_index_new"]
    edges_df["y_index"] = df["y_index_new"]
    edges_df["x_type"] = df["x_type"]
    edges_df["y_type"] = df["y_type"]
    edges_df["relation"] = df["display_relation"]  # Use display_relation (e.g., 'ppi')
    edges_df["direction"] = "forward"

    # 4. Generate Reverse Edges
    print("Generating reverse edges...")
    rev_edges_df = edges_df.copy()
    rev_edges_df["relation"] = "rev_" + rev_edges_df["relation"]
    # Swap
    rev_edges_df = rev_edges_df.rename(
        columns={"x_index": "y_index", "y_index": "x_index", "x_type": "y_type", "y_type": "x_type"}
    )
    rev_edges_df["direction"] = "reverse"

    # 5. Concatenate
    final_edges_df = pd.concat([edges_df, rev_edges_df], ignore_index=True)
    final_edges_df["edge_index"] = range(len(final_edges_df))

    # 6. Save
    print("Saving PROTON-compatible files...")
    nodes_out = OUTPUT_DIR / "nodes.csv"
    edges_out = OUTPUT_DIR / "edges.csv"

    # Output columns for nodes: node_index, node_type, node_name, node_id_str
    all_nodes[["node_index", "node_type", "node_name", "node_id_str"]].to_csv(nodes_out, index=False)

    # Output columns for edges: edge_index, x_index, y_index, x_type, y_type, relation
    final_edges_df[["edge_index", "x_index", "y_index", "x_type", "y_type", "relation"]].to_csv(edges_out, index=False)

    print(f"✅ Saved nodes to {nodes_out}")
    print(f"✅ Saved edges to {edges_out}")
    print("\nSummary:")
    print(all_nodes["node_type"].value_counts())
    print(f"Total edges (forward + reverse): {len(final_edges_df):,}")


if __name__ == "__main__":
    process_primekg()
