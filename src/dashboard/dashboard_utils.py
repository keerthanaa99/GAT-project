"""
Dashboard Utilities for GAT-XIDS
=================================
Helper routines for loading preprocessing metadata, serialized graph artifacts,
training histories, computing interactive Plotly visualizations, and running
real-time attention explanations.
"""

import os
import json
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import plotly.graph_objects as go
import plotly.express as px

from src.data.preprocess import (
    EXPECTED_FEATURES,
    CATEGORY_TO_ID,
    ID_TO_CATEGORY,
    SUBTYPE_TO_ID,
    ID_TO_SUBTYPE
)
from src.graph.graph_builder import load_graph_artifact
from src.models.gat_model import GATClassifier, GATConfig
from src.explain.attention_explainer import AttentionExplainer
from src.explain.explanation_report import format_explanation_report, DISCLAIMER_TEXT

CATEGORY_COLORS = {
    "Benign": "#10b981",          # Emerald
    "ARP Spoofing": "#f59e0b",     # Amber
    "MQTT attacks": "#ef4444",     # Crimson
    "Reconnaissance": "#3b82f6",   # Sapphire Blue
    "TCP/IP DoS/DDoS": "#8b5cf6",  # Violet
}


def load_preprocessing_metadata(artifact_dir: str = "models/preprocessing") -> Dict[str, Any]:
    """Loads feature schema and label mappings."""
    schema_path = os.path.join(artifact_dir, "schema.json")
    labels_path = os.path.join(artifact_dir, "label_mappings.json")
    config_path = os.path.join(artifact_dir, "config.json")

    meta: Dict[str, Any] = {"num_features": len(EXPECTED_FEATURES), "features": EXPECTED_FEATURES}
    if os.path.exists(schema_path):
        with open(schema_path, "r", encoding="utf-8") as f:
            meta["schema"] = json.load(f)
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            meta["labels"] = json.load(f)
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            meta["config"] = json.load(f)
    return meta


def load_graph_split(
    graphs_dir: str = "results/graphs",
    split: str = "train"
) -> Tuple[Optional[Data], Optional[Dict[str, Any]]]:
    """Loads PyG graph Data and metadata JSON for train or test split."""
    prefix = f"{split}_sample"
    pt_path = os.path.join(graphs_dir, f"{prefix}_graph.pt")
    json_path = os.path.join(graphs_dir, f"{prefix}_metadata.json")

    if not os.path.exists(pt_path):
        return None, None
    return load_graph_artifact(pt_path, json_path)


def load_training_history(history_path: str = "results/training_history.json") -> Optional[Dict[str, Any]]:
    """Loads GAT training history and convergence metrics."""
    if not os.path.exists(history_path):
        return None
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_interactive_graph_figure(
    data: Data,
    max_nodes: int = 100,
    highlight_node: Optional[int] = None,
    neighbor_indices: Optional[List[int]] = None
) -> go.Figure:
    """
    Constructs an interactive 2D Plotly network graph.
    Nodes are colored by ground-truth labels for diagnostic inspection.
    """
    n_nodes = data.num_nodes
    if n_nodes == 0:
        fig = go.Figure()
        fig.update_layout(title="Empty Graph")
        return fig

    # Determine node subset
    if n_nodes > max_nodes:
        # Prioritize highlighted node and its neighbors
        selected = set()
        if highlight_node is not None and highlight_node < n_nodes:
            selected.add(highlight_node)
        if neighbor_indices:
            for nb in neighbor_indices:
                if nb < n_nodes:
                    selected.add(nb)

        # Fill remaining slots evenly across categories
        if hasattr(data, "label_category"):
            cats = list(set(data.label_category))
            rng = np.random.RandomState(42)
            per_cat = max(1, (max_nodes - len(selected)) // len(cats))
            for cat in cats:
                cat_indices = [i for i, c in enumerate(data.label_category) if c == cat and i not in selected]
                if cat_indices:
                    chosen = rng.choice(cat_indices, size=min(len(cat_indices), per_cat), replace=False)
                    selected.update(chosen)

        # Fallback if still under max_nodes
        for i in range(n_nodes):
            if len(selected) >= max_nodes:
                break
            selected.add(i)
        sub_nodes = sorted(list(selected))
    else:
        sub_nodes = list(range(n_nodes))

    sub_set = set(sub_nodes)
    node_to_idx = {orig: i for i, orig in enumerate(sub_nodes)}

    # Filter edges
    edge_index = data.edge_index.numpy()
    src_list, dst_list = [], []
    for s, d in zip(edge_index[0], edge_index[1]):
        if s in sub_set and d in sub_set and s < d:
            src_list.append(s)
            dst_list.append(d)

    # Compute 2D circular / force positions deterministically
    import networkx as nx
    G = nx.Graph()
    for n in sub_nodes:
        G.add_node(n)
    for s, d in zip(src_list, dst_list):
        G.add_edge(s, d)

    pos = nx.spring_layout(G, seed=42, k=0.4)

    # Edge traces
    edge_x, edge_y = [], []
    highlight_edge_x, highlight_edge_y = [], []

    for s, d in zip(src_list, dst_list):
        x0, y0 = pos[s]
        x1, y1 = pos[d]
        if highlight_node is not None and (s == highlight_node or d == highlight_node):
            highlight_edge_x.extend([x0, x1, None])
            highlight_edge_y.extend([y0, y1, None])
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig = go.Figure()

    # Standard edges
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color="rgba(148, 163, 184, 0.4)"),
            hoverinfo="none",
            mode="lines",
            name="Flow Similarity Edges"
        ))

    # Highlighted attention edges
    if highlight_edge_x:
        fig.add_trace(go.Scatter(
            x=highlight_edge_x, y=highlight_edge_y,
            line=dict(width=2.5, color="#f43f5e"),
            hoverinfo="none",
            mode="lines",
            name="Target Attention Edges"
        ))

    # Node traces grouped by category for clear interactive legend
    categories = sorted(list(CATEGORY_COLORS.keys()))
    for cat in categories:
        cat_nodes = [
            n for n in sub_nodes
            if (hasattr(data, "label_category") and data.label_category[n] == cat) or
               (not hasattr(data, "label_category") and ID_TO_CATEGORY.get(int(data.y[n].item())) == cat)
        ]
        if not cat_nodes:
            continue

        nx_coords = [pos[n][0] for n in cat_nodes]
        ny_coords = [pos[n][1] for n in cat_nodes]
        sizes = []
        borders = []
        border_widths = []
        hover_texts = []

        for n in cat_nodes:
            is_target = (highlight_node is not None and n == highlight_node)
            is_neighbor = (neighbor_indices is not None and n in neighbor_indices)
            sub = data.label_subtype[n] if hasattr(data, "label_subtype") else "Unknown"

            if is_target:
                sizes.append(18)
                borders.append("#f43f5e")
                border_widths.append(3)
                role = "⭐ TARGET NODE"
            elif is_neighbor:
                sizes.append(14)
                borders.append("#fbbf24")
                border_widths.append(2)
                role = "🔹 INFLUENTIAL NEIGHBOR"
            else:
                sizes.append(9)
                borders.append("#ffffff")
                border_widths.append(1)
                role = "Traffic Flow Node"

            hover_texts.append(
                f"<b>Node Index:</b> {n} ({role})<br>"
                f"<b>Category:</b> {cat}<br>"
                f"<b>Subtype:</b> {sub}<br>"
                f"<b>Class ID:</b> {int(data.y[n].item())}"
            )

        fig.add_trace(go.Scatter(
            x=nx_coords, y=ny_coords,
            mode="markers",
            hovertext=hover_texts,
            hoverinfo="text",
            marker=dict(
                color=CATEGORY_COLORS.get(cat, "#64748b"),
                size=sizes,
                line=dict(color=borders, width=border_widths)
            ),
            name=cat
        ))

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(15, 23, 42, 0.7)"
        ),
        margin=dict(b=20, l=10, r=10, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(15, 23, 42, 0.95)",
        paper_bgcolor="rgba(15, 23, 42, 0.95)",
        font=dict(color="#e2e8f0", family="Inter, system-ui, sans-serif"),
        height=520
    )
    return fig


def plot_training_history_curves(history_data: Dict[str, Any]) -> go.Figure:
    """Renders interactive multi-metric convergence curves."""
    hist = history_data.get("history", {})
    epochs = list(range(1, len(hist.get("train_loss", [])) + 1))

    fig = go.Figure()
    if "train_loss" in hist:
        fig.add_trace(go.Scatter(x=epochs, y=hist["train_loss"], mode="lines+markers", name="Train Loss", line=dict(color="#f43f5e", width=2)))
    if "val_loss" in hist:
        fig.add_trace(go.Scatter(x=epochs, y=hist["val_loss"], mode="lines+markers", name="Validation Loss", line=dict(color="#fb923c", width=2, dash="dash")))
    if "val_accuracy" in hist:
        fig.add_trace(go.Scatter(x=epochs, y=hist["val_accuracy"], mode="lines+markers", name="Val Accuracy", line=dict(color="#10b981", width=2)))
    if "val_macro_f1" in hist:
        fig.add_trace(go.Scatter(x=epochs, y=hist["val_macro_f1"], mode="lines+markers", name="Val Macro F1", line=dict(color="#38bdf8", width=2.5)))

    best_epoch = history_data.get("best_epoch", None)
    if best_epoch is not None:
        fig.add_vline(x=best_epoch, line_width=1.5, line_dash="dot", line_color="#a855f7", annotation_text=f"Best Epoch ({best_epoch})")

    fig.update_layout(
        title="GAT Convergence Trajectory (Loss & Macro F1)",
        xaxis_title="Training Epoch",
        yaxis_title="Score / Loss",
        margin=dict(b=40, l=40, r=20, t=40),
        plot_bgcolor="rgba(15, 23, 42, 0.95)",
        paper_bgcolor="rgba(15, 23, 42, 0.95)",
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
        height=400
    )
    return fig


def create_attention_radar_figure(top_neighbors: List[Dict[str, Any]]) -> go.Figure:
    """Generates a multi-head attention radar plot for top neighbors."""
    fig = go.Figure()
    head_names = [f"Head {i+1}" for i in range(4)]

    for nb in top_neighbors[:5]:
        heads = nb.get("layer_1_heads", [0.25, 0.25, 0.25, 0.25])
        # Close radar loop
        r_vals = heads + [heads[0]]
        theta_vals = head_names + [head_names[0]]

        cat = nb.get("neighbor_category", "Unknown")
        rank = nb.get("rank", 1)
        node_id = nb.get("neighbor_index", 0)

        fig.add_trace(go.Scatterpolar(
            r=r_vals,
            theta=theta_vals,
            fill='toself',
            name=f"#{rank} Node {node_id} ({cat})",
            line=dict(color=CATEGORY_COLORS.get(cat, "#38bdf8"), width=1.8),
            opacity=0.65
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.0], color="#94a3b8"),
            angularaxis=dict(color="#e2e8f0")
        ),
        showlegend=True,
        title="Multi-Head Attention Distribution (Layer 1, Heads 1-4)",
        plot_bgcolor="rgba(15, 23, 42, 0.95)",
        paper_bgcolor="rgba(15, 23, 42, 0.95)",
        font=dict(color="#e2e8f0"),
        height=380,
        margin=dict(t=50, b=20, l=30, r=30)
    )
    return fig
