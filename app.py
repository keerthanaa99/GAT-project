"""
GAT-XIDS: Explainable Graph Attention Network for Medical IoT Intrusion Detection
Streamlit viewer: a simple, guided tour of how the detector works.

Pages
  Home            - what the project does, in plain words
  See the graph   - how similar network flows are linked together
  Check a flow    - the model's verdict on one flow and WHY it decided that
  Compare flows   - which traffic numbers differ between two flows
  Model report    - how accurate the model is
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.preprocess import EXPECTED_FEATURES
from src.dashboard.dashboard_utils import (
    load_graph_split,
    load_training_history,
    build_interactive_graph_figure,
    plot_training_history_curves,
    create_attention_radar_figure,
    CATEGORY_COLORS,
)
from src.explain.attention_explainer import AttentionExplainer
from src.explain.explanation_report import format_explanation_report

st.set_page_config(
    page_title="GAT-XIDS | Medical IoT Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Plain-language text
# --------------------------------------------------------------------------
PAGES = ["Home", "See the graph", "Check a flow", "Compare flows", "Model report"]

PAGE_HELP = {
    "Home": "Start here.",
    "See the graph": "How similar flows are connected.",
    "Check a flow": "Verdict and the reason for it.",
    "Compare flows": "What differs between two flows.",
    "Model report": "How accurate the model is.",
}

CATEGORY_INFO = {
    "Benign": ("Normal traffic", "Nothing suspicious. This looks like regular device traffic."),
    "ARP Spoofing": ("Fake identity attack", "A device pretends to be another one on the local network."),
    "MQTT attacks": ("Attack on medical messaging", "Floods or malformed messages sent to the MQTT service that devices use to talk."),
    "Reconnaissance": ("Scouting", "Someone is scanning the network to find devices and weak points."),
    "TCP/IP DoS/DDoS": ("Overload attack", "Traffic floods that try to knock a device or service offline."),
}

THEME_BLUE = "#38bdf8"

st.markdown(
    """
<style>
    .block-container { padding-top: 2rem; max-width: 1150px; }
    .hero h1 { font-size: 2.4rem; margin-bottom: 0.2rem; }
    .hero p  { font-size: 1.1rem; color: #94a3b8; margin-top: 0; }
    .card {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px; padding: 18px 20px; height: 100%;
    }
    .card h4 { margin: 0 0 6px 0; font-size: 1.05rem; }
    .card p  { margin: 0; color: #cbd5e1; font-size: 0.95rem; line-height: 1.5; }
    .step-num {
        display: inline-block; width: 28px; height: 28px; line-height: 28px;
        text-align: center; border-radius: 50%; background: #38bdf8; color: #0f172a;
        font-weight: 700; margin-bottom: 8px;
    }
    .verdict { border-radius: 14px; padding: 20px 24px; margin: 8px 0 16px 0; }
    .verdict .label { font-size: 0.85rem; opacity: 0.85; }
    .verdict .big   { font-size: 2rem; font-weight: 700; line-height: 1.2; }
    .verdict .sub   { font-size: 1rem; opacity: 0.9; margin-top: 4px; }
    .note { font-size: 0.85rem; color: #94a3b8; }
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource
def get_explainer():
    path = "models/gat_best_checkpoint.pt"
    return AttentionExplainer(checkpoint_path=path) if os.path.exists(path) else None


@st.cache_data
def get_graph_data(split_name: str):
    return load_graph_split(split=split_name)


@st.cache_data
def get_history():
    return load_training_history()


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def go_to(page: str):
    st.session_state["page"] = page


def card(title: str, body: str):
    st.markdown(f'<div class="card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def verdict_banner(category: str, confidence: float):
    """Big colour-coded result box: green = normal, red = attack."""
    is_ok = category == "Benign"
    color = "16, 185, 129" if is_ok else "239, 68, 68"
    title, desc = CATEGORY_INFO.get(category, (category, ""))
    icon = "✅" if is_ok else "🚨"
    st.markdown(
        f"""
<div class="verdict" style="background: rgba({color}, 0.14); border: 1px solid rgba({color}, 0.5);">
  <div class="label">{icon} Model verdict</div>
  <div class="big">{category} <span style="font-size:1.1rem;font-weight:500;">&nbsp;{confidence * 100:.0f}% sure</span></div>
  <div class="sub"><b>{title}.</b> {desc}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def style_fig(fig, height=380):
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        height=height,
        margin=dict(t=50, b=20, l=30, r=30),
    )
    return fig


def get_labels(data):
    return list(getattr(data, "label_category", []) or [])


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_home():
    history = get_history()
    f1 = history.get("best_val_macro_f1") if history else None

    st.markdown(
        """
<div class="hero">
  <h1>🛡️ GAT-XIDS</h1>
  <p>Spots cyber-attacks on hospital and medical-device networks, and explains why it flagged them.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<span class="step-num">1</span>', unsafe_allow_html=True)
        card("Take network traffic", "Each row is one <b>flow</b>: a summary of a conversation between devices, described by 45 numbers (rate, size, flags...).")
    with c2:
        st.markdown('<span class="step-num">2</span>', unsafe_allow_html=True)
        card("Link similar flows", "Every flow is connected to its <b>5 most similar flows</b>, forming a graph. Similar behaviour sits close together.")
    with c3:
        st.markdown('<span class="step-num">3</span>', unsafe_allow_html=True)
        card("Decide and explain", "The model labels each flow and shows <b>which neighbouring flows</b> influenced the decision most.")

    st.markdown("### What the answer can be")
    cols = st.columns(5)
    for col, (cat, (title, desc)) in zip(cols, CATEGORY_INFO.items()):
        with col:
            st.markdown(
                f"""<div class="card" style="border-top: 4px solid {CATEGORY_COLORS.get(cat, '#64748b')};">
                <h4>{cat}</h4><p>{title}</p></div>""",
                unsafe_allow_html=True,
            )

    st.markdown("### At a glance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Traffic numbers per flow", "45")
    m2.metric("Possible answers", "5")
    m3.metric("Neighbours per flow", "5")
    m4.metric("Model score (F1)", f"{f1 * 100:.1f}%" if f1 else "n/a",
              help="F1 balances catching attacks with avoiding false alarms. 100% is perfect.")

    st.markdown("### Where to go next")
    b1, b2, b3 = st.columns(3)
    b1.button("See the graph  →", on_click=go_to, args=("See the graph",), use_container_width=True)
    b2.button("Check a flow  →", on_click=go_to, args=("Check a flow",), type="primary", use_container_width=True)
    b3.button("Model report  →", on_click=go_to, args=("Model report",), use_container_width=True)

    st.markdown(
        '<p class="note">Lines in the graph mean <i>"behaves similarly"</i>, not "is physically cabled to". '
        "Attention shows what the model relied on; it is not proof of cause.</p>",
        unsafe_allow_html=True,
    )


def page_graph():
    st.title("See the graph")
    st.write("Each dot is one network flow, coloured by its true type. A line joins flows that behave alike.")

    c1, c2, c3 = st.columns(3)
    with c1:
        split_label = st.radio("Data", ["Test flows", "Training flows"], horizontal=True,
                               help="Test flows were never seen during training.")
    split = "test" if split_label == "Test flows" else "train"
    graph_data, graph_meta = get_graph_data(split)

    if graph_data is None:
        st.warning(f"No saved graph found for '{split}' in results/graphs/.")
        return

    with c2:
        max_nodes = st.slider("How many flows to draw", 20, 150, 75, 5,
                              help="Fewer dots = a cleaner picture.")
    with c3:
        target = st.number_input("Highlight flow number", 0, int(graph_data.num_nodes) - 1, 0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Flows", f"{graph_data.num_nodes:,}")
    m2.metric("Connections", f"{graph_data.edge_index.shape[1]:,}")
    m3.metric("Links per flow (k)", f"{graph_meta.get('k_neighbors', 5)}")

    edge_idx = graph_data.edge_index.numpy()
    neighbors = [int(d) for s, d in zip(edge_idx[0], edge_idx[1]) if s == target]

    fig = build_interactive_graph_figure(
        graph_data, max_nodes=max_nodes, highlight_node=int(target), neighbor_indices=neighbors
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Tip: hover a dot to see details. Zoom with the scroll wheel.")

    labels = get_labels(graph_data)
    if labels:
        cat = labels[int(target)]
        title, _ = CATEGORY_INFO.get(cat, (cat, ""))
        st.markdown(f"**Flow {int(target)}** is **{cat}** ({title}) and is linked to **{len(neighbors)}** similar flows: `{neighbors}`")
        if neighbors:
            same = sum(1 for n in neighbors if labels[n] == cat)
            st.write(f"{same} of its {len(neighbors)} neighbours have the same type. "
                     "This is why looking at neighbours helps the model decide.")

    with st.expander("What does this graph mean?"):
        st.write(
            "Flows are scaled, then each is linked to its 5 nearest flows (Euclidean distance over 45 features). "
            "Lines show statistical similarity, not real device wiring."
        )


def page_check():
    st.title("Check a flow")
    st.write("Pick a flow. The model gives its verdict and shows which neighbouring flows it relied on.")

    data, _ = get_graph_data("test")
    explainer = get_explainer()
    if data is None or explainer is None:
        st.error("The test graph or the model file (models/gat_best_checkpoint.pt) is missing.")
        return

    labels = get_labels(data)
    c1, c2 = st.columns([1, 1])
    with c1:
        options = ["Any"] + list(CATEGORY_INFO.keys())
        wanted = st.selectbox("1. Show me an example of", options,
                              help="Filter by the flow's true type.")
    pool = [i for i, l in enumerate(labels) if wanted == "Any" or l == wanted] or list(range(int(data.num_nodes)))
    with c2:
        pos = st.slider("2. Which example", 1, len(pool), 1, help=f"{len(pool)} flows match.")
    node = pool[pos - 1]

    top_k = st.select_slider("3. How many influential neighbours to show", options=list(range(1, 11)), value=5)

    with st.spinner("Asking the model..."):
        exp = explainer.explain_node(data, node_index=int(node), top_k=int(top_k))

    pred, target = exp["prediction"], exp["target_node"]
    verdict_banner(pred["predicted_category"], pred["confidence"])

    if target["true_category"] == pred["predicted_category"]:
        st.success(f"Correct. The real label of flow {node} is {target['true_category']}.")
    else:
        st.warning(f"Mismatch. The real label of flow {node} is {target['true_category']}.")

    nbrs = exp["top_contributing_neighbors"]
    if nbrs:
        best = nbrs[0]
        st.info(
            f"**Why?** The model looked at flows similar to flow {node}. "
            f"The most influential was flow {best['neighbor_index']} ({best['neighbor_category']}), "
            f"with {best['combined_attention'] * 100:.0f}% of the attention."
        )

    left, right = st.columns(2)
    with left:
        prob_df = pd.DataFrame(
            [{"Category": k, "Probability": v} for k, v in pred["class_probabilities"].items()]
        )
        fig = px.bar(prob_df, x="Probability", y="Category", orientation="h", color="Category",
                     color_discrete_map=CATEGORY_COLORS, title="How sure is the model about each type?")
        fig.update_layout(showlegend=False, xaxis_tickformat=".0%")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        nb_df = pd.DataFrame([{
            "Rank": n["rank"],
            "Flow": n["neighbor_index"],
            "Type": n["neighbor_category"],
            "Attention": n["combined_attention"],
        } for n in nbrs])
        fig2 = px.bar(nb_df, x="Attention", y=nb_df["Flow"].astype(str), orientation="h",
                      color="Type", color_discrete_map=CATEGORY_COLORS,
                      title="Which neighbours mattered most?")
        fig2.update_layout(yaxis_title="Neighbour flow", yaxis=dict(autorange="reversed"),
                           xaxis_tickformat=".0%")
        st.plotly_chart(style_fig(fig2), use_container_width=True)

    with st.expander("Advanced: per-head attention and full report"):
        st.plotly_chart(create_attention_radar_figure(nbrs), use_container_width=True)
        st.dataframe(pd.DataFrame([{
            "Rank": n["rank"], "Flow": n["neighbor_index"], "Type": n["neighbor_category"],
            "Subtype": n["neighbor_subtype"],
            "Combined attention": round(n["combined_attention"], 4),
            "Layer 2 attention": round(n["layer_2_attention"], 4),
        } for n in nbrs]), use_container_width=True, hide_index=True)
        st.code(format_explanation_report(exp), language="text")


def page_compare():
    st.title("Compare two flows")
    st.write("See which traffic numbers differ most between two flows, for example a normal one and an attack.")

    data, _ = get_graph_data("test")
    if data is None:
        st.warning("No test graph found.")
        return

    labels = get_labels(data)
    n = int(data.num_nodes)

    def fmt(i):
        return f"Flow {i} ({labels[i]})" if labels else f"Flow {i}"

    c1, c2 = st.columns(2)
    a = c1.selectbox("Flow A", range(n), index=0, format_func=fmt)
    b = c2.selectbox("Flow B", range(n), index=min(1, n - 1), format_func=fmt)

    fa, fb = data.x[a].numpy(), data.x[b].numpy()
    df = pd.DataFrame({"Feature": EXPECTED_FEATURES, "A": fa, "B": fb, "Difference": np.abs(fa - fb)})
    df = df.sort_values("Difference", ascending=False)

    top = df.head(10).iloc[::-1]
    fig = px.bar(top, x="Difference", y="Feature", orientation="h",
                 title="Top 10 features that differ most")
    fig.update_traces(marker_color=THEME_BLUE)
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)
    st.caption("Values are standardised, so differences are comparable across features.")

    with st.expander("Show all 45 features"):
        st.dataframe(df, use_container_width=True, height=420, hide_index=True)


def page_report():
    st.title("Model report")
    history = get_history()
    if history is None:
        st.info("No training history found at results/training_history.json.")
        return

    cfg = history.get("config", {})
    f1 = history.get("best_val_macro_f1", 0.0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall score (F1)", f"{f1 * 100:.1f}%",
              help="Average F1 across the 5 classes on validation data. Higher is better.")
    c2.metric("Best round", f"Epoch {history.get('best_epoch', 0)}",
              help="Training stopped here because it stopped improving.")
    c3.metric("Training time", f"{history.get('training_time_seconds', 0.0):.0f} s")

    st.write("The curves below show the model getting better as it trained. "
             "Loss should go down; accuracy and F1 should go up.")
    st.plotly_chart(plot_training_history_curves(history), use_container_width=True)

    st.markdown("#### Good to know")
    st.markdown(
        "- The training data had far more attack rows (mostly DoS/DDoS) than normal ones, "
        "so the model can be less reliable on **Benign** and **ARP Spoofing** flows.\n"
        "- The test flows form their own graph, so the model is checked on data it never saw."
    )

    with st.expander("Technical settings"):
        st.json(cfg)


# --------------------------------------------------------------------------
# App shell
# --------------------------------------------------------------------------
def main():
    st.session_state.setdefault("page", "Home")

    with st.sidebar:
        st.markdown("## 🛡️ GAT-XIDS")
        st.caption("Medical IoT intrusion detection")
        st.radio("Go to", PAGES, key="page",
                 captions=[PAGE_HELP[p] for p in PAGES])
        st.divider()
        st.caption("Dataset: CICIoMT2024")

    page = st.session_state["page"]
    {
        "Home": page_home,
        "See the graph": page_graph,
        "Check a flow": page_check,
        "Compare flows": page_compare,
        "Model report": page_report,
    }[page]()


if __name__ == "__main__":
    main()
