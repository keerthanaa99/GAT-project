"""
Attention-Based Explainer for GAT-XIDS
======================================
Extracts and analyzes multi-head attention coefficients from trained Graph Attention
Networks to explain intrusion detection decisions on medical IoT network flows.

SCIENTIFIC METHODOLOGY & LIMITATIONS:
-------------------------------------
1. Mechanism:
   GAT-XIDS computes dynamic self-attention coefficients alpha_ij representing the normalized
   importance of neighboring flow j when aggregating context for classifying target flow i.
   
2. Multi-Head Representation:
   In Layer 1, attention is computed across K independent heads (K=4), capturing distinct
   statistical sub-dimensions of network traffic.
   In Layer 2, attention represents the final aggregation weighting contributing to class logits.

3. Interpretability Disclaimer:
   Attention weights highlight the empirical aggregation behavior of the neural network.
   They identify which neighboring traffic instances were prioritized by the model; they do NOT
   prove physical causality, packet forgery origin, or network hardware routing.
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from torch_geometric.data import Data

from src.models.gat_model import GATClassifier, GATConfig
from src.data.preprocess import EXPECTED_FEATURES, ID_TO_CATEGORY, CATEGORY_TO_ID
from src.explain.explanation_report import (
    format_explanation_report,
    save_explanation_artifacts,
    DISCLAIMER_TEXT
)

logger = logging.getLogger("gat_xids.attention_explainer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class AttentionExplainer:
    """
    Extracts, ranks, and structures attention coefficients from a trained GAT model.
    """
    def __init__(
        self,
        checkpoint_path: str = "models/gat_best_checkpoint.pt",
        device: str = "cpu"
    ):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"GAT model checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading GAT model from checkpoint: {checkpoint_path}")
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.model = GATClassifier.from_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        self.config = self.model.config
        self.id_to_class = ID_TO_CATEGORY

    def explain_node(
        self,
        data: Data,
        node_index: int,
        top_k: int = 5,
        include_feature_contrast: bool = True
    ) -> Dict[str, Any]:
        """
        Generates an attention-based explanation for a specific target node.

        Args:
            data: PyTorch Geometric Data object containing graph and features.
            node_index: Index of the target node to explain [0, num_nodes - 1].
            top_k: Number of top contributing neighbor flows to rank.
            include_feature_contrast: If True, computes numerical feature alignment with neighbors.

        Returns:
            Structured explanation dictionary.
        """
        num_nodes = data.num_nodes
        if node_index < 0 or node_index >= num_nodes:
            raise IndexError(
                f"Invalid node_index {node_index}. Graph contains {num_nodes} nodes (valid range: 0 to {num_nodes - 1})."
            )

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        data = data.to(self.device)

        # 1. Execute deterministic inference with attention retention
        with torch.no_grad():
            inference_results = self.model.predict(
                data.x,
                data.edge_index,
                return_attention_weights=True
            )

        logits = inference_results["logits"][node_index]
        probabilities = inference_results["probabilities"][node_index].cpu().numpy()
        pred_class_id = int(inference_results["predicted_classes"][node_index].item())
        pred_category = self.id_to_class.get(pred_class_id, f"Class_{pred_class_id}")
        confidence = float(probabilities[pred_class_id])

        # Form probability distribution map
        class_probs = {
            self.id_to_class.get(c, f"Class_{c}"): float(prob)
            for c, prob in enumerate(probabilities)
        }

        # 2. Retrieve Ground-Truth (for evaluation and diagnostic audit only)
        true_category = None
        true_class_id = None
        if hasattr(data, "y") and data.y is not None and len(data.y) > node_index:
            true_class_id = int(data.y[node_index].item())
            true_category = self.id_to_class.get(true_class_id, f"Class_{true_class_id}")

        # Metadata records for target node
        target_raw = data.label_raw[node_index] if hasattr(data, "label_raw") else None
        target_sub = data.label_subtype[node_index] if hasattr(data, "label_subtype") else None

        # 3. Extract Layer Attention Tensors
        # attentions is a list of tuples: [(edge_index_l, alpha_l), ...]
        attentions = inference_results["attentions"]
        layer_1_edges, layer_1_alpha = attentions[0]
        layer_2_edges, layer_2_alpha = attentions[1]

        # Edges directed towards target node: dst == node_index (edge_index[1] == node_index)
        l1_mask = (layer_1_edges[1] == node_index)
        l2_mask = (layer_2_edges[1] == node_index)

        l1_sources = layer_1_edges[0][l1_mask].cpu().numpy()
        l1_weights = layer_1_alpha[l1_mask].cpu().numpy()  # [num_incoming, heads=4]

        l2_sources = layer_2_edges[0][l2_mask].cpu().numpy()
        l2_weights = layer_2_alpha[l2_mask].cpu().numpy()  # [num_incoming, 1]

        # 4. Map and Aggregate Neighbor Attention Scores
        # Build dictionary per incoming neighbor
        neighbor_data: Dict[int, Dict[str, Any]] = {}

        # Layer 1 Attention (Multi-Head)
        for src, head_alphas in zip(l1_sources, l1_weights):
            src = int(src)
            mean_alpha = float(np.mean(head_alphas))
            neighbor_data[src] = {
                "neighbor_index": src,
                "is_self": (src == node_index),
                "layer_1_mean_attention": round(mean_alpha, 6),
                "layer_1_head_attentions": [round(float(a), 6) for a in head_alphas],
                "layer_2_attention": 0.0,
            }

        # Layer 2 Attention (Decision Layer)
        for src, dec_alpha in zip(l2_sources, l2_weights):
            src = int(src)
            dec_val = float(dec_alpha[0])
            if src in neighbor_data:
                neighbor_data[src]["layer_2_attention"] = round(dec_val, 6)
            else:
                neighbor_data[src] = {
                    "neighbor_index": src,
                    "is_self": (src == node_index),
                    "layer_1_mean_attention": 0.0,
                    "layer_1_head_attentions": [],
                    "layer_2_attention": round(dec_val, 6),
                }

        # Attach metadata to neighbors
        for src, nbr_dict in neighbor_data.items():
            meta = {}
            if hasattr(data, "label_category"):
                meta["category"] = data.label_category[src]
            if hasattr(data, "label_subtype"):
                meta["subtype"] = data.label_subtype[src]
            if hasattr(data, "label_raw"):
                meta["source_file"] = data.label_raw[src]
            nbr_dict["metadata"] = meta

            # Feature alignment analysis (numerical proximity in feature space)
            if include_feature_contrast and not nbr_dict["is_self"]:
                target_feats = data.x[node_index].cpu().numpy()
                nbr_feats = data.x[src].cpu().numpy()
                abs_diffs = np.abs(target_feats - nbr_feats)
                # Find closest 4 features (smallest distance)
                closest_indices = np.argsort(abs_diffs)[:4]
                nbr_dict["aligned_features"] = [
                    (EXPECTED_FEATURES[idx], round(float(abs_diffs[idx]), 4))
                    for idx in closest_indices
                ]

        # Extract Self-Attention values
        self_entry = neighbor_data.get(node_index, {})
        self_l1_mean = self_entry.get("layer_1_mean_attention", 0.0)
        self_l2 = self_entry.get("layer_2_attention", 0.0)

        # 5. Rank External Contributing Neighbors
        # Filter out self-loop from neighbor ranking to show external influence
        external_neighbors = [
            nbr for src, nbr in neighbor_data.items() if not nbr["is_self"]
        ]
        # Primary rank by Layer 2 decision attention; secondary by Layer 1 mean
        external_neighbors.sort(
            key=lambda n: (n["layer_2_attention"], n["layer_1_mean_attention"]),
            reverse=True
        )

        top_contributing = external_neighbors[:top_k]

        explanation = {
            "target_node": {
                "node_index": node_index,
                "true_class_id": true_class_id,
                "true_category": true_category,
                "source_file": target_raw,
                "subtype": target_sub,
            },
            "prediction": {
                "predicted_class_id": pred_class_id,
                "predicted_category": pred_category,
                "confidence": round(confidence, 6),
                "class_probabilities": class_probs,
            },
            "self_attention": {
                "layer_1_mean_self_attention": round(self_l1_mean, 6),
                "layer_2_self_attention": round(self_l2, 6),
                "layer_1_head_attentions": self_entry.get("layer_1_head_attentions", [])
            },
            "total_incoming_neighbors": len(external_neighbors),
            "top_k_requested": top_k,
            "top_contributing_neighbors": top_contributing,
            "all_neighbors_detail": list(neighbor_data.values()),
            "scientific_disclaimer": DISCLAIMER_TEXT
        }

        return explanation


def main():
    parser = argparse.ArgumentParser(description="GAT-XIDS Inherent Attention Explainer")
    parser.add_argument("--checkpoint", type=str, default="models/gat_best_checkpoint.pt", help="Path to GAT model checkpoint")
    parser.add_argument("--graph", type=str, default="results/graphs/test_sample_graph.pt", help="Path to graph artifact (.pt)")
    parser.add_argument("--metadata", type=str, default="results/graphs/test_sample_metadata.json", help="Path to graph metadata (.json)")
    parser.add_argument("--node-index", type=int, default=0, help="Target node index to explain (default: 0)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top neighbors to rank (default: 5)")
    parser.add_argument("--output-dir", type=str, default="results/explanations", help="Output directory for explanation artifacts")
    parser.add_argument("--print-report", action="store_true", default=True, help="Print formatted ASCII report to terminal")

    args = parser.parse_args()

    if not os.path.exists(args.graph):
        raise FileNotFoundError(f"Graph file not found: {args.graph}")

    logger.info(f"Loading graph from {args.graph}...")
    data = torch.load(args.graph, weights_only=False)

    explainer = AttentionExplainer(checkpoint_path=args.checkpoint)
    explanation = explainer.explain_node(data, node_index=args.node_index, top_k=args.top_k)

    # Save artifacts
    prefix = f"node_{args.node_index}_explanation"
    json_path, txt_path = save_explanation_artifacts(explanation, args.output_dir, prefix=prefix)
    logger.info(f"Saved JSON explanation -> {json_path}")
    logger.info(f"Saved Text explanation -> {txt_path}")

    if args.print_report:
        report_text = format_explanation_report(explanation)
        print("\n" + report_text)


if __name__ == "__main__":
    main()
