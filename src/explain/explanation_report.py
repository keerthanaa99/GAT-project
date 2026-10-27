"""
Explanation Reporting and Serialization for GAT-XIDS
=====================================================
Formats and serializes attention-based graph explanation reports for intrusion detection.

DISCLAIMER & SCIENTIFIC PRINCIPLES:
-----------------------------------
Attention coefficients quantify the proportion of representation weight that a node assigned
to its neighbors during graph message passing. High attention indicates strong statistical reliance
in the model's aggregation mechanism; it does NOT constitute mathematical proof of causal maliciousness
or physical attack propagation.
"""

import os
import json
from typing import Dict, Any, Tuple


DISCLAIMER_TEXT = (
    "SCIENTIFIC & FORENSIC DISCLAIMER:\n"
    "Graph attention coefficients quantify the structural aggregation weights assigned by GAT\n"
    "layers during neighborhood message passing. While high attention indicates strong statistical\n"
    "reliance on a neighbor's representation for classification, it reflects model aggregation\n"
    "behavior rather than empirical causal proof of attack transmission or physical device origin."
)


def format_explanation_report(explanation: Dict[str, Any]) -> str:
    """
    Constructs a structured, human-readable text report of an attention-based explanation.
    """
    target = explanation["target_node"]
    pred_info = explanation["prediction"]
    neighbors = explanation["top_contributing_neighbors"]
    self_att = explanation.get("self_attention", {})
    layers = explanation.get("layer_details", {})

    lines = []
    lines.append("=" * 78)
    lines.append("           GAT-XIDS INHERENT ATTENTION EXPLANATION REPORT")
    lines.append("=" * 78)

    # 1. Target Node Overview
    lines.append("\n[1] TARGET FLOW NODE OVERVIEW")
    lines.append("-" * 78)
    lines.append(f"  Node Index in Graph : {target['node_index']}")
    lines.append(f"  Predicted Category  : {pred_info['predicted_category']} (ID: {pred_info['predicted_class_id']})")
    lines.append(f"  Prediction Confidence: {pred_info['confidence']*100:.2f}%")

    if "true_category" in target and target["true_category"] is not None:
        match_str = "MATCH" if target["true_category"] == pred_info["predicted_category"] else "MISMATCH"
        lines.append(f"  Ground-Truth Label  : {target['true_category']} [{match_str}] (Evaluation Only)")
    else:
        lines.append("  Ground-Truth Label  : Not Provided / Unlabeled Inference")

    if "source_file" in target and target["source_file"]:
        lines.append(f"  Source Capture File : {target['source_file']}")
    if "subtype" in target and target["subtype"]:
        lines.append(f"  Traffic Subtype     : {target['subtype']}")

    # 2. Probability Distribution
    lines.append("\n[2] PREDICTED CLASS PROBABILITY DISTRIBUTION")
    lines.append("-" * 78)
    for cat, prob in pred_info["class_probabilities"].items():
        bar_len = int(prob * 35)
        bar = "#" * bar_len
        lines.append(f"  {cat:<20} : {prob*100:6.2f}% | {bar}")

    # 3. Self-Attention vs Neighbor Aggregation
    lines.append("\n[3] ATTENTION AGGREGATION PROFILE")
    lines.append("-" * 78)
    l1_self = self_att.get("layer_1_mean_self_attention", 0.0)
    l2_self = self_att.get("layer_2_self_attention", 0.0)
    lines.append(f"  Layer 1 Self-Attention (Node's Own State) : {l1_self*100:.2f}%")
    lines.append(f"  Layer 1 External Neighbor Reliance        : {(1.0 - l1_self)*100:.2f}%")
    lines.append(f"  Layer 2 Self-Attention (Decision Layer)   : {l2_self*100:.2f}%")
    lines.append(f"  Layer 2 External Neighbor Reliance        : {(1.0 - l2_self)*100:.2f}%")

    # 4. Top Contributing Neighbors
    lines.append(f"\n[4] TOP-{len(neighbors)} CONTRIBUTING NEIGHBOR FLOWS (Ranked by Decision Layer Attention)")
    lines.append("-" * 78)
    lines.append(f"  {'Rank':<5} {'Neighbor':<10} {'Layer 2 Attn':<14} {'Layer 1 Mean':<14} {'Neighbor Category':<20} {'Subtype'}")
    lines.append(f"  {'-'*4} {'-'*8} {'-'*12} {'-'*12} {'-'*18} {'-'*15}")

    for rank, nbr in enumerate(neighbors, 1):
        idx_str = f"Node {nbr['neighbor_index']}"
        l2_str = f"{nbr['layer_2_attention']*100:.2f}%"
        l1_str = f"{nbr['layer_1_mean_attention']*100:.2f}%"
        cat_str = nbr.get("metadata", {}).get("category", "Unknown")
        sub_str = nbr.get("metadata", {}).get("subtype", "Unknown")
        lines.append(f"  {rank:<5} {idx_str:<10} {l2_str:<14} {l1_str:<14} {cat_str:<20} {sub_str}")

    # 5. Contrastive Feature Alignment (if present)
    if any("aligned_features" in nbr for nbr in neighbors):
        lines.append("\n[5] NEIGHBORHOOD STATISTICAL FEATURE ALIGNMENT (Separate Feature-Level Contrast)")
        lines.append("-" * 78)
        lines.append("  Note: Shows numerical proximity between target node and top neighbor in feature space.")
        top_nbr = neighbors[0]
        if "aligned_features" in top_nbr:
            lines.append(f"  Closest Shared Features with Top Neighbor (Node {top_nbr['neighbor_index']}):")
            for feat, diff in top_nbr["aligned_features"][:4]:
                lines.append(f"    - {feat:<24} : absolute distance = {diff:.4f}")

    # 6. Disclaimer
    lines.append("\n" + "=" * 78)
    lines.append(DISCLAIMER_TEXT)
    lines.append("=" * 78 + "\n")

    return "\n".join(lines)


def save_explanation_artifacts(
    explanation: Dict[str, Any],
    output_dir: str,
    prefix: str
) -> Tuple[str, str]:
    """
    Saves explanation as both structured JSON and formatted text.

    Returns:
        (json_path, txt_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{prefix}.json")
    txt_path = os.path.join(output_dir, f"{prefix}.txt")

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(explanation, f, indent=2)

    # Save formatted text
    report_text = format_explanation_report(explanation)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return json_path, txt_path
