"""
Flow-Similarity Graph Construction for GAT-XIDS (CICIoMT2024 Dataset)
======================================================================
Constructs sparse k-nearest-neighbor (k-NN) flow-similarity graphs from preprocessed
CICIoMT2024 traffic records for Graph Attention Networks.

RESEARCH & METHODOLOGY NOTES:
-----------------------------
1. Conceptual Basis:
   The CICIoMT2024 dataset features represent 45 aggregated traffic statistics (flow durations,
   byte rates, flag counts, packet sizes) rather than explicit topological endpoints (IP/MAC/ports).
   Therefore, the graph topology represents statistical behavioral similarity between network flows,
   not physical hardware-to-hardware network wiring.

2. Supervised Learning Integrity:
   Ground-truth attack labels and file origins are STRICTLY EXCLUDED from graph edge construction.
   Graph connectivity is computed entirely using unsupervised spatial distance in the 45-dimensional
   standardized feature space. Labels are attached as node targets (data.y) for supervised learning.

3. Inductive Evaluation Protocol:
   Training and testing graphs are constructed separately to prevent test-to-training data leakage.
   Nodes in the test graph are connected only among test instances, preserving strict inductive isolation.
"""

import os
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import networkx as nx

from src.data.preprocess import (
    EXPECTED_FEATURES,
    CATEGORY_TO_ID,
    ID_TO_CATEGORY,
    SUBTYPE_TO_ID,
    ID_TO_SUBTYPE
)

logger = logging.getLogger("gat_xids.graph_builder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@dataclass
class GraphBuildConfig:
    """Configuration parameters for flow-similarity graph construction."""
    k_neighbors: int = 5
    metric: str = "euclidean"
    make_undirected: bool = True
    remove_self_loops: bool = True
    target_label: str = "category_id"  # "category_id", "subtype_id", "is_attack"
    random_seed: int = 42


def build_flow_similarity_graph(
    features: np.ndarray,
    labels: Optional[Dict[str, Any]] = None,
    config: Optional[GraphBuildConfig] = None
) -> Data:
    """
    Constructs a PyTorch Geometric Data object by building a k-NN flow-similarity graph.

    Args:
        features: 2D array of shape [N, 45] representing preprocessed numerical traffic features.
        labels: Optional dictionary containing label vectors ('category_id', 'subtype_id', 'is_attack', etc.).
        config: GraphBuildConfig instance specifying k, distance metric, etc.

    Returns:
        torch_geometric.data.Data object with:
            - x: [N, 45] FloatTensor
            - edge_index: [2, E] LongTensor (no self-loops)
            - y: [N] LongTensor (or empty if labels is None)
            - metadata attributes
    """
    if config is None:
        config = GraphBuildConfig()

    if config.k_neighbors <= 0:
        raise ValueError(f"k_neighbors must be a positive integer, got {config.k_neighbors}")

    if not isinstance(features, np.ndarray):
        features = np.asarray(features)

    # Empty graph edge case
    if features.size == 0 or features.shape[0] == 0:
        empty_x = torch.empty((0, len(EXPECTED_FEATURES)), dtype=torch.float32)
        empty_edge_index = torch.empty((2, 0), dtype=torch.long)
        empty_y = torch.empty((0,), dtype=torch.long)
        return Data(x=empty_x, edge_index=empty_edge_index, y=empty_y, num_nodes=0)

    if features.ndim != 2:
        raise ValueError(f"features array must be 2D, got shape {features.shape}")

    num_nodes, num_feats = features.shape
    if num_feats != len(EXPECTED_FEATURES):
        raise ValueError(f"Expected {len(EXPECTED_FEATURES)} features, got {num_feats}")

    # Validate numerical integrity
    if np.isnan(features).any() or np.isinf(features).any():
        raise ValueError("features matrix contains NaN or Inf values. Ensure data is preprocessed.")

    # 1. Edge Case: Single node (isolated node, no self-neighbors)
    if num_nodes == 1:
        x_tensor = torch.tensor(features, dtype=torch.float32)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        y_tensor = (
            torch.tensor([labels[config.target_label][0]], dtype=torch.long)
            if labels and config.target_label in labels
            else torch.empty((1,), dtype=torch.long)
        )
        data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor, num_nodes=1)
        _attach_metadata(data, labels, 1)
        return data

    # 2. Determine effective k
    eff_k = min(config.k_neighbors, num_nodes - 1)
    if eff_k < config.k_neighbors:
        logger.warning(
            f"Graph has {num_nodes} nodes, which is <= k={config.k_neighbors}. "
            f"Clamping effective neighbors to {eff_k}."
        )

    # 3. Fit NearestNeighbors index (avoids N x N distance matrix explosion)
    # Query eff_k + 2 to safely discard self-matches and potential duplicate instances
    k_query = min(eff_k + 2, num_nodes)
    nn = NearestNeighbors(n_neighbors=k_query, metric=config.metric, algorithm="auto")
    nn.fit(features)
    distances, indices = nn.kneighbors(features)

    # 4. Build directed edges excluding self-index
    src_list: List[int] = []
    dst_list: List[int] = []
    dist_list: List[float] = []

    for i in range(num_nodes):
        selected_neighbors = 0
        for rank, neighbor_idx in enumerate(indices[i]):
            if neighbor_idx == i:
                continue  # Exclude self-loop
            src_list.append(i)
            dst_list.append(int(neighbor_idx))
            dist_list.append(float(distances[i][rank]))
            selected_neighbors += 1
            if selected_neighbors >= eff_k:
                break

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 5. Remove self-loops and convert to undirected if configured
    if config.remove_self_loops and edge_index.numel() > 0:
        edge_index, _ = remove_self_loops(edge_index)

    if config.make_undirected and edge_index.numel() > 0:
        edge_index = to_undirected(edge_index)

    # Coalesce to ensure sorted, duplicate-free edge indices in both directed and undirected modes
    if edge_index.numel() > 0:
        edge_index = coalesce(edge_index)

    # 6. Assemble PyTorch Geometric Data object
    x_tensor = torch.tensor(features, dtype=torch.float32)

    if labels is not None and config.target_label in labels:
        y_tensor = torch.tensor(labels[config.target_label], dtype=torch.long)
    else:
        y_tensor = torch.empty((num_nodes,), dtype=torch.long)

    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor, num_nodes=num_nodes)
    _attach_metadata(data, labels, num_nodes)

    return data


def _attach_metadata(data: Data, labels: Optional[Dict[str, Any]], num_nodes: int) -> None:
    """Safely attaches traceability metadata to Data object without polluting node features x."""
    if not labels:
        return
    for key in ("category_id", "subtype_id", "is_attack", "source_row_index"):
        if key in labels:
            setattr(data, key, torch.tensor(labels[key][:num_nodes], dtype=torch.long))

    for key in ("label_raw", "label_subtype", "label_category"):
        if key in labels:
            setattr(data, key, list(labels[key][:num_nodes]))


def sample_stratified_nodes(
    df: pd.DataFrame,
    max_nodes: int,
    stratify_col: str = "label_category",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Performs reproducible stratified sampling across attack/benign categories.
    Ensures minority classes are represented in small graph samples.
    """
    if len(df) <= max_nodes:
        return df.copy()

    classes = df[stratify_col].unique()
    per_class = max(1, max_nodes // len(classes))
    sampled_dfs = []

    np.random.seed(random_seed)
    for cls in classes:
        cls_df = df[df[stratify_col] == cls]
        n_sample = min(len(cls_df), per_class)
        sampled_dfs.append(cls_df.sample(n=n_sample, random_state=random_seed))

    combined = pd.concat(sampled_dfs, ignore_index=True)
    if len(combined) < max_nodes:
        remaining_needed = max_nodes - len(combined)
        remaining_pool = df.drop(combined.index, errors="ignore")
        if len(remaining_pool) > 0:
            extra = remaining_pool.sample(
                n=min(remaining_needed, len(remaining_pool)),
                random_state=random_seed
            )
            combined = pd.concat([combined, extra], ignore_index=True)

    # Sort for deterministic node ordering
    return combined.iloc[:max_nodes].sort_values(by="label_category").reset_index(drop=True)


def load_parquet_and_build_graph(
    parquet_path: str,
    max_nodes: Optional[int] = None,
    stratified: bool = True,
    config: Optional[GraphBuildConfig] = None
) -> Tuple[Data, Dict[str, Any]]:
    """
    Loads preprocessed Parquet records, selects samples, and builds a PyG flow-similarity graph.
    """
    if config is None:
        config = GraphBuildConfig()

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at: {parquet_path}")

    start_time = time.time()
    df = pd.read_parquet(parquet_path)
    load_time = time.time() - start_time

    if max_nodes is not None and max_nodes < len(df):
        if stratified and "label_category" in df.columns:
            df = sample_stratified_nodes(df, max_nodes, stratify_col="label_category", random_seed=config.random_seed)
        else:
            df = df.iloc[:max_nodes].reset_index(drop=True)

    # Extract 45 standardized features
    features = df[EXPECTED_FEATURES].values.astype(np.float32)

    # Extract metadata and labels
    labels = {
        "category_id": df["category_id"].values.astype(np.int64),
        "subtype_id": df["subtype_id"].values.astype(np.int64),
        "is_attack": df["is_attack"].values.astype(np.int64),
        "label_raw": df["label_raw"].tolist(),
        "label_subtype": df["label_subtype"].tolist(),
        "label_category": df["label_category"].tolist(),
        "source_row_index": np.arange(len(df), dtype=np.int64),
    }

    t0 = time.time()
    data = build_flow_similarity_graph(features, labels=labels, config=config)
    build_time = time.time() - t0

    # Build metadata summary
    y_vals = data.y.numpy()
    unique_labels, counts = np.unique(y_vals, return_counts=True)
    label_dist = {ID_TO_CATEGORY.get(int(lbl), str(lbl)): int(c) for lbl, c in zip(unique_labels, counts)}

    metadata = {
        "source_file": parquet_path,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
        "num_features": int(data.x.shape[1]),
        "k_neighbors": config.k_neighbors,
        "metric": config.metric,
        "make_undirected": config.make_undirected,
        "class_distribution": label_dist,
        "density": float(data.edge_index.shape[1] / max(1, data.num_nodes * (data.num_nodes - 1))),
        "build_time_seconds": round(build_time, 4),
        "load_time_seconds": round(load_time, 4),
        "methodology": "k-Nearest-Neighbor Flow Statistical Similarity (Labels Excluded from Topology)",
    }

    return data, metadata


def save_graph_artifacts(
    data: Data,
    metadata: Dict[str, Any],
    output_dir: str,
    prefix: str
) -> Tuple[str, str]:
    """Saves graph Data (.pt) and metadata (.json) under output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    graph_path = os.path.join(output_dir, f"{prefix}_graph.pt")
    meta_path = os.path.join(output_dir, f"{prefix}_metadata.json")

    torch.save(data, graph_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved graph ({data.num_nodes} nodes, {data.edge_index.shape[1]} edges) -> {graph_path}")
    return graph_path, meta_path


def load_graph_artifact(graph_pt_path: str, meta_json_path: Optional[str] = None) -> Tuple[Data, Optional[Dict[str, Any]]]:
    """Loads a previously saved PyG Data object and its metadata."""
    if not os.path.exists(graph_pt_path):
        raise FileNotFoundError(f"Graph file not found: {graph_pt_path}")

    # Register Data class with safe globals if supported in PyTorch
    if hasattr(torch.serialization, "add_safe_globals"):
        try:
            torch.serialization.add_safe_globals([Data])
        except Exception:
            pass

    try:
        data = torch.load(graph_pt_path, weights_only=True)
    except Exception:
        data = torch.load(graph_pt_path, weights_only=False)

    meta = None
    if meta_json_path and os.path.exists(meta_json_path):
        with open(meta_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return data, meta


def visualize_flow_graph(
    data: Data,
    output_image_path: str,
    max_nodes: int = 80,
    random_state: int = 42
) -> str:
    """
    Generates a 2D NetworkX visualization of a small diagnostic subgraph.
    Labels are displayed to evaluate cluster separation in feature space.
    """
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    n_nodes = data.num_nodes
    if n_nodes > max_nodes:
        # Uniformly stride or sample across all categories so the diagnostic plot displays all classes
        rng = np.random.RandomState(random_state)
        # Select balanced sample across available categories
        unique_cats = list(set(data.label_category)) if hasattr(data, "label_category") else []
        if unique_cats:
            sub_indices = []
            per_cat = max(1, max_nodes // len(unique_cats))
            for cat in unique_cats:
                cat_indices = [i for i, c in enumerate(data.label_category) if c == cat]
                sub_indices.extend(rng.choice(cat_indices, size=min(len(cat_indices), per_cat), replace=False))
            sub_indices = np.array(sub_indices[:max_nodes])
        else:
            sub_indices = rng.choice(n_nodes, size=max_nodes, replace=False)
    else:
        sub_indices = np.arange(n_nodes)

    sub_mask = np.isin(data.edge_index[0].numpy(), sub_indices) & np.isin(data.edge_index[1].numpy(), sub_indices)
    sub_edge_index = data.edge_index[:, sub_mask].numpy()

    # Build NetworkX Graph
    G = nx.Graph()
    for idx in sub_indices:
        lbl = data.label_category[idx] if hasattr(data, "label_category") else str(data.y[idx].item())
        G.add_node(int(idx), label=lbl)

    for src, dst in zip(sub_edge_index[0], sub_edge_index[1]):
        if src < dst:
            G.add_edge(int(src), int(dst))

    # Distinct color palette for categories
    category_colors = {
        "Benign": "#2ecc71",          # Green
        "ARP Spoofing": "#f39c12",     # Orange
        "MQTT attacks": "#e74c3c",     # Red
        "Reconnaissance": "#3498db",   # Blue
        "TCP/IP DoS/DDoS": "#9b59b6",  # Purple
    }

    node_colors = []
    for node in G.nodes():
        cat = G.nodes[node].get("label", "Unknown")
        node_colors.append(category_colors.get(cat, "#95a5a6"))

    plt.figure(figsize=(11, 8.5), dpi=150)
    pos = nx.spring_layout(G, seed=random_state, k=0.35)

    nx.draw_networkx_nodes(G, pos, node_size=130, node_color=node_colors, alpha=0.90, edgecolors="white", linewidths=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.28, edge_color="#7f8c8d", width=0.85)

    # Custom Legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=10, label=cat)
        for cat, col in category_colors.items()
        if any(G.nodes[n].get("label") == cat for n in G.nodes())
    ]
    plt.legend(handles=legend_handles, title="Ground-Truth Attack Category", loc="upper right", framealpha=0.95, fontsize=9)

    plt.title(
        f"GAT-XIDS Flow-Similarity Graph ({len(G.nodes)} nodes, {len(G.edges)} edges, k=5)\n"
        "[DIAGNOSTIC VISUALIZATION: Node colors indicate class labels for visual analysis only;\n"
        "edges were constructed purely via k-NN on 45 preprocessed flow features without access to labels or network endpoints]",
        fontsize=9.5,
        pad=14
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_image_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved diagnostic graph visualization -> {output_image_path}")
    return output_image_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="GAT-XIDS Flow-Similarity Graph Construction")
    parser.add_argument("--train-parquet", type=str, default="results/preprocessed/train_sample.parquet")
    parser.add_argument("--test-parquet", type=str, default="results/preprocessed/test_sample.parquet")
    parser.add_argument("--train-nodes", type=int, default=500, help="Nodes to sample for training graph (default: 500)")
    parser.add_argument("--test-nodes", type=int, default=200, help="Nodes to sample for test graph (default: 200)")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors (default: 5)")
    parser.add_argument("--metric", type=str, default="euclidean", help="Distance metric (default: euclidean)")
    parser.add_argument("--output-dir", type=str, default="results/graphs", help="Output directory for graph artifacts")
    parser.add_argument("--visualize", action="store_true", default=True, help="Generate diagnostic plot")

    args = parser.parse_args()
    config = GraphBuildConfig(k_neighbors=args.k, metric=args.metric)

    logger.info(f"Building Training Graph (max {args.train_nodes} nodes, k={args.k})...")
    train_graph, train_meta = load_parquet_and_build_graph(
        args.train_parquet,
        max_nodes=args.train_nodes,
        stratified=True,
        config=config
    )
    save_graph_artifacts(train_graph, train_meta, args.output_dir, prefix="train_sample")

    logger.info(f"Building Test Graph (max {args.test_nodes} nodes, k={args.k})...")
    test_graph, test_meta = load_parquet_and_build_graph(
        args.test_parquet,
        max_nodes=args.test_nodes,
        stratified=True,
        config=config
    )
    save_graph_artifacts(test_graph, test_meta, args.output_dir, prefix="test_sample")

    if args.visualize:
        viz_path = os.path.join(args.output_dir, "flow_similarity_graph_sample.png")
        visualize_flow_graph(train_graph, viz_path, max_nodes=80)

    print("\n" + "="*60)
    print("FLOW-SIMILARITY GRAPH CONSTRUCTION SUMMARY")
    print("="*60)
    print(f"Training Graph: {train_graph.num_nodes} nodes, {train_graph.edge_index.shape[1]} edges, {train_graph.x.shape[1]} features")
    print(f"  -> Class Distribution: {train_meta['class_distribution']}")
    print(f"  -> Build Time: {train_meta['build_time_seconds']}s")
    print(f"Test Graph    : {test_graph.num_nodes} nodes, {test_graph.edge_index.shape[1]} edges, {test_graph.x.shape[1]} features")
    print(f"  -> Class Distribution: {test_meta['class_distribution']}")
    print(f"  -> Build Time: {test_meta['build_time_seconds']}s")
    print(f"Artifacts Dir : {args.output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
