"""
GAT-XIDS: Explainable Graph Attention Network for Medical IoT Intrusion Detection
==================================================================================
Interactive Streamlit Application for Graph Exploration, GAT Classification,
and Attention-Based Explainability.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.data.preprocess import (
    EXPECTED_FEATURES,
    CATEGORY_TO_ID,
    ID_TO_CATEGORY,
    SUBTYPE_TO_ID,
    ID_TO_SUBTYPE
)
from src.dashboard.dashboard_utils import (
    load_preprocessing_metadata,
    load_graph_split,
    load_training_history,
    build_interactive_graph_figure,
    plot_training_history_curves,
    create_attention_radar_figure,
    CATEGORY_COLORS
)
from src.explain.attention_explainer import AttentionExplainer
from src.explain.explanation_report import format_explanation_report, DISCLAIMER_TEXT

# Page configuration
st.set_page_config(
    page_title="GAT-XIDS | Medical IoT Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0%, #020617 100%);
        color: #f8fafc;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 18px;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    
    .disclaimer-banner {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 20px;
        font-size: 0.88rem;
        color: #fde68a;
    }
    
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_explainer():
    """Initializes and caches AttentionExplainer."""
    checkpoint_path = "models/gat_best_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        return AttentionExplainer(checkpoint_path=checkpoint_path)
    return None


@st.cache_data
def get_graph_data(split_name: str):
    """Loads and caches graph artifacts."""
    return load_graph_split(split=split_name)


def main():
    # Header
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; padding-bottom: 12px; border-bottom: 1px solid rgba(148,163,184,0.15); margin-bottom: 18px;">
        <div>
            <h1 style="margin: 0; font-size: 2.1rem; font-weight: 700; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                GAT-XIDS
            </h1>
            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                Explainable Graph Attention Network for Medical IoT Intrusion Detection
            </p>
        </div>
        <div style="text-align: right;">
            <span class="badge" style="background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3);">Pipeline Active</span>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">CICIoMT2024 Benchmark</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Scientific Disclaimer Notice
    st.markdown(f"""
    <div class="disclaimer-banner">
        <strong>METHODOLOGICAL NOTICE:</strong> Graph edges represent <em>statistical behavioral similarity</em> between network flows computed over 45 preprocessed features via unsupervised $k$-NN. They do not represent verified physical device-to-device wiring. Graph attention weights reflect model aggregation reliance and do not constitute physical causal proof.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation
    st.sidebar.image("https://img.icons8.com/isometric/100/graph.png", width=64)
    st.sidebar.title("Navigation")
    view = st.sidebar.radio(
        "Select Module:",
        [
            "🌐 Overview & Schema",
            "🕸️ Flow-Similarity Graph Explorer",
            "📊 GAT Model Performance",
            "🔍 Attention-Based Explainer (XAI)",
            "⚡ Flow Feature Inspector"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Info")
    st.sidebar.markdown("""
    - **Dataset**: CICIoMT2024
    - **Features**: 45 Numerical Statistics
    - **Classes**: 5 Broad Categories
    - **Architecture**: 2-Layer GAT (4 Heads)
    - **Evaluation**: Inductive Isolated Split
    """)

    meta = load_preprocessing_metadata()
    history = load_training_history()
    explainer = get_explainer()

    # VIEW 1: Overview & Schema
    if view == "🌐 Overview & Schema":
        st.subheader("System Overview & Preprocessing Architecture")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 0.85rem;">Input Features</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #38bdf8;">45</div>
                <div style="font-size: 0.78rem; color: #10b981;">Standardized float32</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 0.85rem;">Broad Attack Classes</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #818cf8;">5</div>
                <div style="font-size: 0.78rem; color: #94a3b8;">19 Specific Subtypes</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 0.85rem;">Graph Metric</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #f43f5e;">k-NN</div>
                <div style="font-size: 0.78rem; color: #94a3b8;">k=5 (Euclidean distance)</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 0.85rem;">GAT Checkpoint</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #34d399;">F1: 85.4%</div>
                <div style="font-size: 0.78rem; color: #10b981;">Epoch 38 (Early Stop)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Preprocessed Feature Schema (45 Numerical Attributes)")
        feat_df = pd.DataFrame({
            "Feature Index": range(len(EXPECTED_FEATURES)),
            "Feature Name": EXPECTED_FEATURES,
            "Type": ["float32"] * len(EXPECTED_FEATURES),
            "Description": [
                "Protocol duration, packet sizes, rates, flag numbers, or protocol indicators"
            ] * len(EXPECTED_FEATURES)
        })
        st.dataframe(feat_df, use_container_width=True, height=280)

        st.markdown("### Attack Taxonomy Mapping (19 Granular Subtypes -> 5 Broad Categories)")
        cat_data = []
        for cat, cid in CATEGORY_TO_ID.items():
            subtypes = [s for s, c in meta.get("labels", {}).get("category_mapping", {}).items() if c == cat]
            cat_data.append({
                "Category ID": cid,
                "Broad Category": cat,
                "Mapped Subtypes Count": len(subtypes),
                "Subtype Examples": ", ".join(subtypes[:3]) + ("..." if len(subtypes) > 3 else "")
            })
        st.dataframe(pd.DataFrame(cat_data), use_container_width=True)

    # VIEW 2: Flow-Similarity Graph Explorer
    elif view == "🕸️ Flow-Similarity Graph Explorer":
        st.subheader("Flow-Similarity Graph Topology Explorer")

        col_opt1, col_opt2, col_opt3 = st.columns([2, 2, 2])
        with col_opt1:
            split_choice = st.selectbox("Select Graph Split:", ["train", "test"])
        with col_opt2:
            max_nodes_disp = st.slider("Display Node Limit (for smooth rendering):", 20, 150, 75, 5)
        with col_opt3:
            highlight_target = st.number_input("Target Node to Inspect:", min_value=0, max_value=499, value=0)

        graph_data, graph_meta = get_graph_data(split_choice)

        if graph_data is None:
            st.warning(f"Graph artifact for split '{split_choice}' not found under results/graphs/.")
        else:
            # Summary Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Nodes", f"{graph_data.num_nodes:,}")
            m2.metric("Total Edges", f"{graph_data.edge_index.shape[1]:,}")
            m3.metric("Graph Density", f"{graph_meta.get('density', 0.0):.4f}")
            m4.metric("k-Neighbors", f"k={graph_meta.get('k_neighbors', 5)}")

            # Find target neighbors
            target_neighbors = []
            edge_idx = graph_data.edge_index.numpy()
            for s, d in zip(edge_idx[0], edge_idx[1]):
                if s == highlight_target:
                    target_neighbors.append(int(d))

            fig = build_interactive_graph_figure(
                graph_data,
                max_nodes=max_nodes_disp,
                highlight_node=highlight_target,
                neighbor_indices=target_neighbors
            )
            st.plotly_chart(fig, use_container_width=True)

            # Target Node Details
            if highlight_target < graph_data.num_nodes:
                st.markdown(f"### Node {highlight_target} Topology Details")
                t_cat = graph_data.label_category[highlight_target] if hasattr(graph_data, "label_category") else "Unknown"
                t_sub = graph_data.label_subtype[highlight_target] if hasattr(graph_data, "label_subtype") else "Unknown"

                st.markdown(f"""
                - **Category**: `{t_cat}` | **Subtype**: `{t_sub}`
                - **Degree (Connected Similar Flows)**: `{len(target_neighbors)}` neighbors
                - **Neighbor Indices**: `{target_neighbors}`
                """)

    # VIEW 3: GAT Model Performance
    elif view == "📊 GAT Model Performance":
        st.subheader("GAT Model Convergence & Training Metrics")

        if history is None:
            st.info("No training history JSON found at results/training_history.json.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Best Epoch", history.get("best_epoch", 0))
            c2.metric("Best Val Macro F1", f"{history.get('best_val_macro_f1', 0.0)*100:.2f}%")
            c3.metric("Training Time", f"{history.get('training_time_seconds', 0.0):.2f}s")
            c4.metric("Architecture", f"{history.get('config', {}).get('num_layers', 2)} Layers, {history.get('config', {}).get('heads', 4)} Heads")

            fig = plot_training_history_curves(history)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("View Full Hyperparameter Configuration"):
                st.json(history.get("config", {}))

    # VIEW 4: Attention-Based Explainer (XAI)
    elif view == "🔍 Attention-Based Explainer (XAI)":
        st.subheader("Inherent Attention-Based Explanation (XAI)")

        test_data, _ = get_graph_data("test")
        if test_data is None or explainer is None:
            st.error("Missing test graph artifact or GAT model checkpoint.")
        else:
            col_x1, col_x2 = st.columns([2, 1])
            with col_x1:
                node_to_explain = st.number_input(
                    "Select Target Flow Node to Explain:",
                    min_value=0,
                    max_value=test_data.num_nodes - 1,
                    value=0,
                    step=1
                )
            with col_x2:
                top_k = st.slider("Top Influential Neighbors (Top-K):", 1, 10, 5)

            if st.button("Generate Explanation", type="primary"):
                with st.spinner("Executing GAT forward pass and extracting attention weights..."):
                    exp = explainer.explain_node(test_data, node_index=int(node_to_explain), top_k=int(top_k))

                # Display Results
                pred = exp["prediction"]
                target = exp["target_node"]
                conf_pct = pred["confidence"] * 100

                match_status = "✅ MATCH" if target["true_category"] == pred["predicted_category"] else "⚠️ MISMATCH"

                st.markdown(f"""
                <div class="metric-card" style="margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <div style="font-size: 0.85rem; color: #94a3b8;">Predicted Attack Category</div>
                            <div style="font-size: 1.6rem; font-weight: 700; color: #38bdf8;">
                                {pred['predicted_category']} <span style="font-size: 0.95rem; color: #94a3b8;">(Confidence: {conf_pct:.2f}%)</span>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.85rem; color: #94a3b8;">Ground Truth: {target['true_category']}</div>
                            <div style="font-weight: 600; font-size: 0.95rem;">{match_status}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_radar, col_probs = st.columns([1, 1])
                with col_radar:
                    radar_fig = create_attention_radar_figure(exp["top_contributing_neighbors"])
                    st.plotly_chart(radar_fig, use_container_width=True)

                with col_probs:
                    prob_df = pd.DataFrame([
                        {"Category": cat, "Probability": prob}
                        for cat, prob in pred["class_probabilities"].items()
                    ])
                    prob_fig = px.bar(
                        prob_df, x="Category", y="Probability",
                        color="Category",
                        color_discrete_map=CATEGORY_COLORS,
                        title="Class Posterior Probabilities"
                    )
                    prob_fig.update_layout(
                        plot_bgcolor="rgba(15, 23, 42, 0.95)",
                        paper_bgcolor="rgba(15, 23, 42, 0.95)",
                        font=dict(color="#e2e8f0"),
                        showlegend=False,
                        height=380,
                        margin=dict(t=50, b=20, l=30, r=30)
                    )
                    st.plotly_chart(prob_fig, use_container_width=True)

                st.markdown("### Top Contributing Neighbor Flows")
                nb_rows = []
                for nb in exp["top_contributing_neighbors"]:
                    nb_rows.append({
                        "Rank": nb["rank"],
                        "Neighbor Node ID": nb["neighbor_index"],
                        "Category": nb["neighbor_category"],
                        "Subtype": nb["neighbor_subtype"],
                        "Combined Attention": f"{nb['combined_attention']:.4f}",
                        "Layer 2 Weight": f"{nb['layer_2_attention']:.4f}",
                    })
                st.dataframe(pd.DataFrame(nb_rows), use_container_width=True)

                with st.expander("View Full Text Forensic Explanation Report"):
                    report_txt = format_explanation_report(exp)
                    st.code(report_txt, language="text")

    # VIEW 5: Flow Feature Inspector
    elif view == "⚡ Flow Feature Inspector":
        st.subheader("Flow Feature Inspector & Statistical Comparison")
        st.markdown("Compare feature distributions across preprocessed test instances.")

        test_data, _ = get_graph_data("test")
        if test_data is not None:
            n1 = st.selectbox("Select Flow Node A:", range(test_data.num_nodes), index=0)
            n2 = st.selectbox("Select Flow Node B:", range(test_data.num_nodes), index=min(1, test_data.num_nodes - 1))

            feats_a = test_data.x[n1].numpy()
            feats_b = test_data.x[n2].numpy()

            comp_df = pd.DataFrame({
                "Feature": EXPECTED_FEATURES,
                f"Node {n1} ({test_data.label_category[n1]})": feats_a,
                f"Node {n2} ({test_data.label_category[n2]})": feats_b,
                "Delta |A - B|": np.abs(feats_a - feats_b)
            }).sort_values(by="Delta |A - B|", ascending=False)

            st.dataframe(comp_df, use_container_width=True, height=450)


if __name__ == "__main__":
    main()
