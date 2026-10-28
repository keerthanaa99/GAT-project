"""
GAT-XIDS Dashboard
==================
One dashboard for the whole project, built around the security-analyst workflow:

    1. Analyze traffic   - upload a CSV (or use sample data)
    2. Alerts            - triage: what was found, sorted by severity
    3. Investigate       - one flow: verdict, why, what to do
    4. Network map       - the flow-similarity graph the model reasons over
    5. Model report      - how well the model performs
    6. Guide             - inputs, classes, glossary

Run:  streamlit run app.py
"""

import io
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from torch_geometric.data import Data

from src.data.preprocess import EXPECTED_FEATURES, ID_TO_CATEGORY, Preprocessor, clean_chunk
from src.graph.graph_builder import GraphBuildConfig, build_flow_similarity_graph
from src.dashboard.dashboard_utils import (
    CATEGORY_COLORS,
    build_interactive_graph_figure,
    load_graph_split,
    load_training_history,
)
from src.dashboard.alert_utils import HIGH_FROM, REVIEW_BELOW, count_alerts, severity_of
from src.dashboard.synthetic_demo import (
    SYNTHETIC_DEMO_ROWS,
    SYNTHETIC_DEMO_WARNING,
    generate_synthetic_demo_csv,
)
from src.explain.attention_explainer import AttentionExplainer
from src.explain.explanation_report import format_explanation_report

st.set_page_config(
    page_title="GAT-XIDS · Medical IoT Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================================
# Constants and plain-language content
# ==========================================================================
CHECKPOINT = "models/gat_best_checkpoint.pt"
PREPROC_DIR = "models/preprocessing"
MAX_ROWS = 2000
MAX_MB = 25
K = 5
CHART_CFG = {"displayModeBar": False, "displaylogo": False}

CLASSES = ["Benign", "ARP Spoofing", "MQTT attacks", "Reconnaissance", "TCP/IP DoS/DDoS"]

CLASS_INFO = {
    "Benign": {
        "short": "Normal traffic",
        "what": "Nothing suspicious. This looks like regular device traffic.",
        "action": "No action needed.",
    },
    "ARP Spoofing": {
        "short": "Fake identity attack",
        "what": "A device is pretending to be another device (often the gateway) to intercept traffic.",
        "action": "Check the gateway's ARP table, enable Dynamic ARP Inspection on the switch, and isolate the spoofing device.",
    },
    "MQTT attacks": {
        "short": "Attack on device messaging",
        "what": "Floods or malformed messages aimed at the MQTT broker that medical devices use to send readings.",
        "action": "Check the MQTT broker logs, require client authentication, rate-limit or block the offending client.",
    },
    "Reconnaissance": {
        "short": "Network scouting",
        "what": "Someone is scanning the network to discover devices, open ports or vulnerabilities.",
        "action": "Find the scanning source, block it at the firewall, and review which services are exposed.",
    },
    "TCP/IP DoS/DDoS": {
        "short": "Overload attack",
        "what": "A traffic flood trying to slow down or knock a device or service offline.",
        "action": "Rate-limit or block the flooding sources and confirm the targeted device is still reachable.",
    },
}

SEVERITY = {  # label: (icon, colour, meaning)
    "High": ("🔴", "#dc2626", "Attack, model is confident (≥ 80%)."),
    "Medium": ("🟠", "#d97706", "Attack, model is fairly sure (50–80%)."),
    "Review": ("🟡", "#ca8a04", "Model is unsure (< 50%). Inspect the flow manually; this is not a detection of a new attack."),
    "Normal": ("🟢", "#16a34a", "Normal traffic."),
}
SEVERITY_ORDER = {"High": 0, "Medium": 1, "Review": 2, "Normal": 3}

FEATURE_HELP = {
    "Header_Length": "header length",
    "Protocol Type": "protocol type",
    "Duration": "time-to-live / duration",
    "Rate": "packet rate",
    "Srate": "outgoing packet rate",
    "Drate": "incoming packet rate",
    "fin_flag_number": "FIN flags (connection close)",
    "syn_flag_number": "SYN flags (connection start)",
    "rst_flag_number": "RST flags (connection reset)",
    "psh_flag_number": "PSH flags (push data)",
    "ack_flag_number": "ACK flags (acknowledge)",
    "ack_count": "ACK packet count",
    "syn_count": "SYN packet count",
    "fin_count": "FIN packet count",
    "rst_count": "RST packet count",
    "TCP": "TCP share", "UDP": "UDP share", "ARP": "ARP share", "ICMP": "ICMP (ping) share",
    "HTTP": "HTTP share", "HTTPS": "HTTPS share", "DNS": "DNS share",
    "Tot sum": "total bytes", "Min": "smallest packet", "Max": "largest packet",
    "AVG": "average packet size", "Std": "packet-size variation", "Tot size": "packet size",
    "IAT": "time between packets", "Number": "number of packets",
    "Magnitue": "size magnitude", "Radius": "size radius", "Covariance": "size covariance",
    "Variance": "size variance", "Weight": "packet weight",
}

# ==========================================================================
# Styling
# ==========================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.01em; }

.ctx { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;
       background:#fff; border:1px solid #e5e7eb; border-radius:14px; padding:12px 18px; margin-bottom:18px; }
.ctx .t { font-weight:700; font-size:1.05rem; color:#111827; }
.ctx .s { color:#6b7280; font-size:0.9rem; }
.chip { display:inline-block; padding:4px 10px; border-radius:999px; font-size:0.75rem; font-weight:600; }
.chip.blue  { background:#e8f0ff; color:#1d4ed8; border:1px solid #c7d8ff; }
.chip.grey  { background:#f3f4f6; color:#4b5563; border:1px solid #e5e7eb; }

.hero { background:linear-gradient(135deg,#1e40af 0%,#2563eb 55%,#3b82f6 100%); color:#fff;
        border-radius:18px; padding:34px 36px; margin-bottom:22px; }
.hero h1 { color:#fff; margin:0 0 8px 0; font-size:2.2rem; }
.hero p  { color:#dbeafe; margin:0; font-size:1.08rem; max-width:760px; line-height:1.55; }

.step { font-size:0.75rem; font-weight:700; color:#2563eb; letter-spacing:.06em; }
.ct { font-weight:700; font-size:1.02rem; color:#111827; margin:2px 0 4px 0; }
.cx { color:#4b5563; font-size:0.92rem; line-height:1.5; margin:0; }

.banner { border-radius:14px; padding:18px 22px; margin:4px 0 16px 0; }
.banner .lab { font-size:0.78rem; font-weight:700; letter-spacing:.05em; }
.banner .big { font-size:1.7rem; font-weight:700; color:#111827; line-height:1.25; margin-top:2px; }
.banner .sub { color:#374151; font-size:0.97rem; margin-top:6px; line-height:1.5; }

.why { background:#f8fafc; border:1px solid #e5e7eb; border-left:4px solid #2563eb; border-radius:10px;
       padding:14px 16px; color:#1f2937; line-height:1.6; margin-bottom:12px; }
.muted { color:#6b7280; font-size:0.86rem; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:12px 16px; }
div[data-testid="stMetricLabel"] p { color:#6b7280; }
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================================================
# Model + data loading (cached)
# ==========================================================================
@st.cache_resource(show_spinner="Loading the GAT model...")
def load_explainer():
    if not os.path.exists(CHECKPOINT):
        return None
    return AttentionExplainer(checkpoint_path=CHECKPOINT)


@st.cache_resource(show_spinner=False)
def load_scaler():
    return Preprocessor.load(PREPROC_DIR).scaler


@st.cache_data(show_spinner=False)
def load_history():
    return load_training_history()


@st.cache_data(show_spinner=False)
def load_training_distribution():
    """Class counts of the graph the GAT was actually trained on."""
    path = "results/graphs/train_sample_metadata.json"
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("class_distribution", {})


@st.cache_resource(show_spinner="Scoring the model on the test set...")
def test_set_scores():
    """Per-type results on the built-in labelled test graph (never seen in training)."""
    try:
        run, _ = analyze_test_graph()
    except ValueError:
        return None
    t = run["table"]
    rows = []
    for c in CLASSES:
        sub = t[t["True label"] == c]
        pred_c = t[t["Prediction"] == c]
        tp = int((sub["Prediction"] == c).sum())
        rows.append({
            "Type": c,
            "Test flows": len(sub),
            "Caught (recall)": tp / len(sub) if len(sub) else 0.0,
            "Correct when flagged (precision)": tp / len(pred_c) if len(pred_c) else 0.0,
        })
    return {"accuracy": float((t["Prediction"] == t["True label"]).mean()), "n": len(t), "per_class": pd.DataFrame(rows)}


@st.cache_data(show_spinner=False)
def read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# ==========================================================================
# Pipeline: CSV -> clean -> scale -> k-NN graph -> GAT -> results
# ==========================================================================
def run_model(x: np.ndarray, name: str, truth=None) -> dict:
    """Builds the similarity graph, classifies every flow and returns one result bundle."""
    explainer = load_explainer()
    cfg = GraphBuildConfig(k_neighbors=K, make_undirected=True, remove_self_loops=True)
    graph = build_flow_similarity_graph(x, labels=None, config=cfg)

    with torch.no_grad():
        out = explainer.model.predict(graph.x, graph.edge_index)
    probs = out["probabilities"].cpu().numpy()
    pred_ids = out["predicted_classes"].cpu().numpy()

    # A clean graph object for display/explanations: labels = model predictions, no ground truth.
    preds = [ID_TO_CATEGORY[int(i)] for i in pred_ids]
    data = Data(x=graph.x, edge_index=graph.edge_index, num_nodes=int(graph.num_nodes),
                y=torch.as_tensor(pred_ids, dtype=torch.long))
    data.label_category = preds
    data.label_subtype = [CLASS_INFO[c]["short"] for c in preds]

    rows = []
    for i, cat in enumerate(preds):
        conf = float(probs[i, int(pred_ids[i])])
        sev = severity_of(cat, conf)
        row = {
            "Flow": i,
            "Severity": f"{SEVERITY[sev][0]} {sev}",
            "_sev": sev,
            "Prediction": cat,
            "Meaning": CLASS_INFO[cat]["short"],
            "Confidence": round(conf * 100, 1),
        }
        if truth is not None:
            row["True label"] = truth[i]
            row["Correct"] = "✅" if truth[i] == cat else "❌"
        for c in range(probs.shape[1]):
            row[f"p_{ID_TO_CATEGORY[c]}"] = float(probs[i, c])
        rows.append(row)

    table = pd.DataFrame(rows)
    return {
        "name": name,
        "n": len(table),
        "x": x,
        "data": data,
        "table": table,
        "has_truth": truth is not None,
        "edges": int(graph.edge_index.shape[1]),
    }


def analyze_csv(raw: bytes, name: str):
    """Validates an uploaded CSV and runs the model. Returns (result, notes) or raises ValueError."""
    notes = []
    if len(raw) == 0:
        raise ValueError("The file is empty.")
    if len(raw) > MAX_MB * 1024 * 1024:
        raise ValueError(f"The file is {len(raw) / 1e6:.1f} MB. The limit is {MAX_MB} MB.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"This doesn't look like a valid CSV file ({e}).") from e
    if df.empty:
        raise ValueError("The CSV has no rows.")

    df.columns = [str(c).strip() for c in df.columns]
    missing = [f for f in EXPECTED_FEATURES if f not in df.columns]
    if missing:
        err = ValueError(f"{len(missing)} of the 45 required columns are missing.")
        err.missing = missing
        raise err

    if len(df) > MAX_ROWS:
        notes.append(f"The file has {len(df):,} rows. Only the first {MAX_ROWS:,} were analysed.")
        df = df.iloc[:MAX_ROWS].copy()
    if len(df) < K + 1:
        notes.append(f"Only {len(df)} rows. Use at least {K + 1} so each flow has {K} neighbours to compare with.")

    feats = df[EXPECTED_FEATURES].apply(pd.to_numeric, errors="coerce")
    clean, n_inf, n_nan = clean_chunk(feats)
    if n_inf or n_nan:
        notes.append(f"Fixed {n_nan + n_inf} empty or invalid values (set to 0).")
    x = load_scaler().transform(clean.values).astype(np.float32)

    truth = None
    if "label_category" in df.columns:
        vals = df["label_category"].astype(str).tolist()
        if all(v in CLASSES for v in vals):
            truth = vals
            notes.append("This file includes true labels, so the dashboard also shows whether each prediction was correct.")
    return run_model(x, name, truth), notes


def analyze_test_graph():
    data, _ = load_graph_split(split="test")
    if data is None:
        raise ValueError("The built-in test set (results/graphs/test_sample_graph.pt) is missing.")
    truth = list(data.label_category) if hasattr(data, "label_category") else None
    return run_model(data.x.cpu().numpy().astype(np.float32), "Built-in test set (200 labelled flows)", truth), []


# ==========================================================================
# Explanations
# ==========================================================================
def explain(run: dict, node: int, top_k: int = K) -> dict:
    src = run["data"]
    # Separate graph without `y`, so the explainer never mistakes predictions for ground truth.
    g = Data(x=src.x, edge_index=src.edge_index, num_nodes=int(src.num_nodes))
    g.label_category = list(src.label_category)
    exp = load_explainer().explain_node(g, node_index=int(node), top_k=int(top_k))
    nbrs = []
    for rank, n in enumerate(exp["top_contributing_neighbors"], start=1):
        nbrs.append({
            "rank": rank,
            "neighbor_index": int(n["neighbor_index"]),
            "neighbor_category": n.get("metadata", {}).get("category", "Unknown"),
            "decision_attention": float(n.get("layer_2_attention", 0.0)),
            "layer1_attention": float(n.get("layer_1_mean_attention", 0.0)),
            "heads": list(n.get("layer_1_head_attentions", [])),
            "shared_features": n.get("aligned_features", []),
        })
    exp["neighbors"] = nbrs
    exp["self_share"] = float(exp.get("self_attention", {}).get("layer_2_self_attention", 0.0))
    return exp


def unusual_features(x_row: np.ndarray, top: int = 6) -> pd.DataFrame:
    """Values are standardised against training data, so |value| = how unusual vs. typical traffic."""
    df = pd.DataFrame({"Feature": EXPECTED_FEATURES, "z": x_row})
    df["abs"] = df["z"].abs()
    df = df.sort_values("abs", ascending=False).head(top)
    df["Meaning"] = df["Feature"].map(lambda f: (lambda m: m[:1].upper() + m[1:])(FEATURE_HELP.get(f, f)))

    def words(z):
        a = abs(z)
        level = "far" if a >= 3 else "clearly" if a >= 1.5 else "slightly"
        return f"{level} {'higher' if z > 0 else 'lower'} than typical"

    df["Compared with typical traffic"] = df["z"].map(words)
    return df


# ==========================================================================
# UI helpers
# ==========================================================================
def light(fig, height=None):
    grid = "#e5e7eb"
    fig.update_layout(
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(color="#111827", family="Inter, system-ui, sans-serif", size=13),
        legend=dict(bgcolor="rgba(255,255,255,0.9)"),
        margin=dict(t=48, b=24, l=10, r=10),
        title=dict(font=dict(size=15)),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid, linecolor=grid)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid, linecolor=grid)
    return fig


def context_bar(title: str, subtitle: str):
    run = st.session_state.get("run")
    chip = (f'<span class="chip blue">Analysing: {run["name"]} · {run["n"]:,} flows</span>'
            if run else '<span class="chip grey">No traffic loaded yet</span>')
    st.markdown(
        f'<div class="ctx"><div><div class="t">{title}</div><div class="s">{subtitle}</div></div>{chip}</div>',
        unsafe_allow_html=True,
    )


def card(step: str, title: str, text: str):
    with st.container(border=True):
        st.markdown(f'<div class="step">{step}</div><div class="ct">{title}</div><p class="cx">{text}</p>',
                    unsafe_allow_html=True)


def banner(kind: str, label: str, big: str, sub: str):
    styles = {
        "bad": ("#fef2f2", "#fecaca", "#b91c1c"),
        "warn": ("#fffbeb", "#fde68a", "#b45309"),
        "ok": ("#f0fdf4", "#bbf7d0", "#15803d"),
        "info": ("#eff6ff", "#bfdbfe", "#1d4ed8"),
    }
    bg, bd, fg = styles[kind]
    st.markdown(
        f'<div class="banner" style="background:{bg};border:1px solid {bd};">'
        f'<div class="lab" style="color:{fg};">{label}</div><div class="big">{big}</div>'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def require_run():
    run = st.session_state.get("run")
    if run is not None:
        return run
    with st.container(border=True):
        st.markdown("#### No traffic analysed yet")
        st.write("Upload a CSV of network flows, or try the built-in sample data to see how it works.")
        c1, c2 = st.columns(2)
        if c1.button("Upload traffic", type="primary", icon=":material/upload:", use_container_width=True):
            st.switch_page(PAGES["analyze"])
        if c2.button("Use the built-in test set", icon=":material/science:", use_container_width=True):
            load_demo()
    return None


def load_demo():
    with st.spinner("Analysing the built-in test set..."):
        try:
            run, notes = analyze_test_graph()
        except ValueError as e:
            st.error(str(e))
            return
    st.session_state["run"] = run
    st.session_state["notes"] = notes
    st.session_state["focus"] = None
    st.switch_page(PAGES["alerts"])


# ==========================================================================
# Pages
# ==========================================================================
def page_home():
    st.markdown(
        """
<div class="hero">
  <h1>🛡️ GAT-XIDS</h1>
  <p>Finds cyber-attacks in hospital and medical-device (IoMT) network traffic, tells you how serious each one is,
     and explains <b>why</b> the model decided that.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        card("STEP 1", "Load traffic", "Upload a CSV where each row is one network <b>flow</b> described by 45 numbers. "
             "No file? Use the built-in test set.")
    with c2:
        card("STEP 2", "Review alerts", "Every flow gets a label and a severity. Attacks are listed first, "
             "most serious at the top.")
    with c3:
        card("STEP 3", "Investigate", "Open any alert to see the verdict, the reason behind it, "
             "and what to do next.")

    st.write("")
    b1, b2, _ = st.columns([1.2, 1.2, 2])
    if b1.button("Upload traffic", type="primary", icon=":material/upload:", use_container_width=True):
        st.switch_page(PAGES["analyze"])
    if b2.button("Try the test set", icon=":material/science:", use_container_width=True):
        load_demo()

    st.markdown("#### What the model can detect")
    cols = st.columns(5)
    for col, cat in zip(cols, CLASSES):
        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div style="height:4px;border-radius:4px;background:{CATEGORY_COLORS.get(cat, "#64748b")};'
                    f'margin-bottom:10px;"></div><div class="ct">{cat}</div><p class="cx">{CLASS_INFO[cat]["short"]}</p>',
                    unsafe_allow_html=True,
                )

    st.markdown("#### How the model thinks")
    with st.container(border=True):
        st.markdown(
            "- Each flow is linked to its **5 most similar flows**, which turns the traffic into a graph.\n"
            "- A **Graph Attention Network** looks at each flow *and* its neighbours, and learns how much to trust each neighbour.\n"
            "- Those trust weights (**attention**) are shown on the Investigate page as the explanation."
        )
        st.markdown('<p class="muted">Links mean "behaves similarly", not "physically connected". '
                    "Attention shows what the model relied on; it is not proof of cause.</p>", unsafe_allow_html=True)


def page_analyze():
    context_bar("Analyze traffic", "Upload network flows and let the model check them.")

    left, right = st.columns([1.6, 1])
    with left:
        with st.container(border=True):
            st.markdown("#### 1 · Choose your data")
            source = st.radio(
                "Data source",
                [
                    "Upload my CSV",
                    "Sample file (50 flows)",
                    "Built-in test set (200 labelled flows)",
                    "Synthetic demo",
                ],
                label_visibility="collapsed",
            )
            uploaded = None
            if source == "Upload my CSV":
                uploaded = st.file_uploader("Network flow CSV", type=["csv"],
                                            help=f"Up to {MAX_ROWS:,} rows and {MAX_MB} MB.")
            elif source == "Synthetic demo":
                st.info(SYNTHETIC_DEMO_WARNING, icon=":material/info:")

            st.markdown("#### 2 · Run the detector")
            go_btn = st.button("Analyze traffic", type="primary", icon=":material/play_arrow:",
                               use_container_width=True,
                               disabled=(source == "Upload my CSV" and uploaded is None))

    with right:
        with st.container(border=True):
            st.markdown("#### Before you upload")
            st.markdown(
                f"- **CSV file**, one row per network flow\n"
                f"- **All 45 feature columns** with the exact CICIoMT2024 names\n"
                f"- Up to **{MAX_ROWS:,} rows** and **{MAX_MB} MB**\n"
                f"- At least **{K + 1} rows** (each flow needs neighbours)\n"
                f"- Extra columns are fine. A `label_category` column lets the dashboard score accuracy."
            )
            if os.path.exists("sample_input.csv"):
                st.download_button("Download a template CSV", read_file_bytes("sample_input.csv"),
                                   file_name="sample_input.csv", mime="text/csv",
                                   icon=":material/download:", use_container_width=True)
            with st.expander("See all 45 required columns"):
                st.code(", ".join(EXPECTED_FEATURES), language="text")

    if not go_btn:
        return

    with st.spinner("Cleaning data, building the similarity graph and running the GAT model..."):
        try:
            if source == "Upload my CSV":
                run, notes = analyze_csv(uploaded.getvalue(), uploaded.name)
            elif source.startswith("Sample file"):
                run, notes = analyze_csv(read_file_bytes("sample_input.csv"), "sample_input.csv")
            elif source == "Synthetic demo":
                demo_csv = generate_synthetic_demo_csv(load_scaler())
                run, notes = analyze_csv(
                    demo_csv,
                    f"Synthetic demo ({SYNTHETIC_DEMO_ROWS} flows)",
                )
                notes.append(SYNTHETIC_DEMO_WARNING)
            else:
                run, notes = analyze_test_graph()
        except ValueError as e:
            st.error(f"**Couldn't analyse this file.** {e}")
            missing = getattr(e, "missing", None)
            if missing:
                st.write("Missing columns:")
                st.code(", ".join(missing), language="text")
                st.caption("Tip: download the template CSV above and compare the column names.")
            return
        except FileNotFoundError as e:
            st.error(f"A model file is missing: {e}")
            return

    st.session_state["run"] = run
    st.session_state["notes"] = notes
    st.session_state["focus"] = None
    st.switch_page(PAGES["alerts"])


def page_alerts():
    context_bar("Alerts", "Everything the model found, most serious first.")
    run = require_run()
    if run is None:
        return

    for note in st.session_state.get("notes", []):
        if note == SYNTHETIC_DEMO_WARNING:
            st.warning(note, icon=":material/warning:")
        else:
            st.info(note, icon=":material/info:")

    t = run["table"]
    n = run["n"]
    sev_counts = t["_sev"].value_counts()
    n_high, n_med = (int(sev_counts.get(s, 0)) for s in ("High", "Medium"))
    n_attack, n_rev = count_alerts(t)
    n_attention = n_attack + n_rev

    # ---- headline ----
    if n_high:
        banner("bad", "🚨 ATTACKS DETECTED", f"{n_attack} of {n} flows look malicious",
               f"{n_high} high-severity alert{'s' if n_high != 1 else ''} need attention. Start with the top of the list below.")
    elif n_attack or n_rev:
        banner("warn", "⚠️ CHECK THESE", f"{n_attention} of {n} flows need a look",
               "Nothing is high severity, but some flows look suspicious or the model is unsure about them.")
    else:
        banner("ok", "✅ ALL CLEAR", f"All {n} flows look normal", "The model found no attack traffic in this data.")

    m = st.columns(4)
    m[0].metric("Flows analysed", f"{n:,}")
    m[1].metric("🔴 High", n_high)
    m[2].metric("🟠 Medium", n_med)
    m[3].metric("🟡 Needs review", n_rev, help=SEVERITY["Review"][2])

    if run["has_truth"]:
        acc = float((t["Prediction"] == t["True label"]).mean())
        st.caption(f"This data has true labels: the model got **{acc:.0%}** of flows right ({int(round(acc * n))} of {n}).")

    # ---- charts ----
    c1, c2 = st.columns([1.3, 1])
    with c1:
        counts = t["Prediction"].value_counts().reindex(CLASSES, fill_value=0).reset_index()
        counts.columns = ["Type", "Flows"]
        fig = px.bar(counts, x="Flows", y="Type", orientation="h", color="Type", text="Flows",
                     color_discrete_map=CATEGORY_COLORS, title="What kind of traffic is this?")
        fig.update_layout(showlegend=False, yaxis=dict(title="", categoryorder="array", categoryarray=CLASSES[::-1]))
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(light(fig, 300), use_container_width=True, config=CHART_CFG)
    with c2:
        sev = t["_sev"].value_counts().reindex(list(SEVERITY), fill_value=0).reset_index()
        sev.columns = ["Severity", "Flows"]
        fig2 = px.bar(sev, x="Severity", y="Flows", color="Severity", text="Flows",
                      color_discrete_map={k: v[1] for k, v in SEVERITY.items()}, title="How serious?")
        fig2.update_layout(showlegend=False, xaxis_title="")
        fig2.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(light(fig2, 300), use_container_width=True, config=CHART_CFG)

    # ---- alert list ----
    st.markdown("#### Alert list")
    f1, f2, f3 = st.columns([1.4, 1.4, 1])
    flagged = [s for s in SEVERITY if s != "Normal" and s in set(t["_sev"])]
    sev_filter = f1.multiselect("Severity", list(SEVERITY), default=flagged or list(SEVERITY),
                                placeholder="All severities")
    type_filter = f2.multiselect("Type", CLASSES, placeholder="All types")
    min_conf = f3.slider("Min. confidence", 0, 100, 0, 5, format="%d%%")

    view = t.copy()
    if sev_filter:
        view = view[view["_sev"].isin(sev_filter)]
    if type_filter:
        view = view[view["Prediction"].isin(type_filter)]
    view = view[view["Confidence"] >= min_conf]
    view = view.assign(_o=view["_sev"].map(SEVERITY_ORDER)).sort_values(["_o", "Confidence"], ascending=[True, False])

    cols = ["Flow", "Severity", "Prediction", "Meaning", "Confidence"] + (["True label", "Correct"] if run["has_truth"] else [])
    shown = view[cols].reset_index(drop=True)

    if shown.empty:
        st.info("No flows match these filters. Try selecting more severities (including Normal).")
    else:
        st.caption("Click a row, then press **Investigate**.")
        event = st.dataframe(
            shown,
            hide_index=True,
            use_container_width=True,
            height=min(420, 38 + 35 * len(shown)),
            on_select="rerun",
            selection_mode="single-row",
            key="alert_table",
            column_config={
                "Flow": st.column_config.NumberColumn("Flow #", width="small"),
                "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.0f%%"),
            },
        )
        rows = event.selection.rows if event and event.selection else []
        picked = int(shown.iloc[rows[0]]["Flow"]) if rows else int(shown.iloc[0]["Flow"])
        b1, b2, _ = st.columns([1.3, 1.3, 2])
        if b1.button(f"Investigate flow {picked}", type="primary", icon=":material/search:", use_container_width=True):
            st.session_state["focus"] = picked
            st.switch_page(PAGES["investigate"])
        export = t.drop(columns=["_sev"]).rename(columns=lambda c: c.replace("p_", "Prob. ") if c.startswith("p_") else c)
        export["Severity"] = t["_sev"]
        b2.download_button("Download results (CSV)", export.to_csv(index=False).encode("utf-8"),
                           file_name=f"{os.path.splitext(run['name'])[0]}_gat_results.csv".replace(" ", "_"),
                           mime="text/csv", icon=":material/download:", use_container_width=True)


def page_investigate():
    context_bar("Investigate a flow", "The verdict, the reason, and what to do next.")
    run = require_run()
    if run is None:
        return

    t = run["table"].assign(_o=lambda d: d["_sev"].map(SEVERITY_ORDER)).sort_values(["_o", "Confidence"], ascending=[True, False])
    order = t["Flow"].astype(int).tolist()
    focus = st.session_state.get("focus")
    idx = order.index(focus) if focus in order else 0

    def label(i):
        r = run["table"].iloc[i]
        return f"Flow {i} · {SEVERITY[r['_sev']][0]} {r['Prediction']} · {r['Confidence']:.0f}%"

    node = st.selectbox("Flow to investigate (most serious first)", order, index=idx, format_func=label)
    st.session_state["focus"] = node
    row = run["table"].iloc[node]
    cat, conf, sev = row["Prediction"], row["Confidence"] / 100, row["_sev"]

    with st.spinner("Reading the model's attention..."):
        exp = explain(run, node)

    # ---- verdict ----
    kind = {"High": "bad", "Medium": "warn", "Review": "warn", "Normal": "ok"}[sev]
    label_txt = {"High": "🔴 HIGH SEVERITY", "Medium": "🟠 MEDIUM SEVERITY",
                 "Review": "🟡 NEEDS REVIEW", "Normal": "🟢 NORMAL"}[sev]
    banner(kind, label_txt, f"{cat} · {conf:.0%} sure", f"<b>{CLASS_INFO[cat]['short']}.</b> {CLASS_INFO[cat]['what']}")
    if run["has_truth"]:
        truth = row["True label"]
        (st.success if truth == cat else st.warning)(
            f"True label: **{truth}** — the model was {'right' if truth == cat else 'wrong'} on this flow.")

    tab_why, tab_traffic, tab_do, tab_tech = st.tabs(["Why this verdict", "Traffic details", "What to do", "Technical details"])

    # ---- WHY ----
    with tab_why:
        nb = exp["neighbors"]
        if nb:
            same = sum(1 for x in nb if x["neighbor_category"] == cat)
            top = nb[0]
            nb_share = max(0.0, 1.0 - exp["self_share"])
            st.markdown(
                f'<div class="why">The model compared flow <b>{node}</b> with the flows that behave most like it. '
                f'Of the <b>{len(nb)}</b> neighbours it relied on most, <b>{same}</b> were also classified as <b>{cat}</b>.<br>'
                f'It based <b>{exp["self_share"]:.0%}</b> of its decision on this flow\'s own traffic and '
                f'<b>{nb_share:.0%}</b> on its neighbours. The most influential neighbour was '
                f'<b>flow {top["neighbor_index"]}</b> ({top["neighbor_category"]}).</div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2)
            with c1:
                share = pd.DataFrame(
                    [{"Source": f"This flow ({node})", "Share": exp["self_share"], "Type": cat}]
                    + [{"Source": f"Flow {x['neighbor_index']}", "Share": x["decision_attention"], "Type": x["neighbor_category"]} for x in nb]
                )
                fig = px.bar(share, x="Share", y="Source", orientation="h", color="Type",
                             color_discrete_map=CATEGORY_COLORS, title="What the decision was based on")
                fig.update_layout(yaxis=dict(autorange="reversed", title=""), xaxis=dict(tickformat=".0%", title="Share of attention"))
                st.plotly_chart(light(fig, 320), use_container_width=True, config=CHART_CFG)
            with c2:
                probs = pd.DataFrame({"Type": CLASSES, "Probability": [float(row[f"p_{c}"]) for c in CLASSES]})
                fig2 = px.bar(probs, x="Probability", y="Type", orientation="h", color="Type",
                              color_discrete_map=CATEGORY_COLORS, title="How sure is the model about each type?")
                fig2.update_layout(showlegend=False, yaxis=dict(title="", categoryorder="array", categoryarray=CLASSES[::-1]),
                                   xaxis=dict(tickformat=".0%", range=[0, 1]))
                st.plotly_chart(light(fig2, 320), use_container_width=True, config=CHART_CFG)
            if st.button("Show this flow on the network map", icon=":material/hub:"):
                st.switch_page(PAGES["network"])
        else:
            st.info("This flow has no similar flows to compare with (the file may be very small), "
                    "so the verdict is based on its own traffic only.")

    # ---- TRAFFIC ----
    with tab_traffic:
        st.write("The traffic numbers that make this flow stand out most compared with typical training traffic:")
        uf = unusual_features(run["x"][node])
        fig = px.bar(uf.iloc[::-1], x="z", y="Meaning", orientation="h",
                     color=np.where(uf.iloc[::-1]["z"] > 0, "Higher than typical", "Lower than typical"),
                     color_discrete_map={"Higher than typical": "#2563eb", "Lower than typical": "#94a3b8"},
                     title="Most unusual traffic features")
        fig.update_layout(yaxis_title="", xaxis_title="Standard deviations from typical", legend_title="")
        st.plotly_chart(light(fig, 330), use_container_width=True, config=CHART_CFG)
        st.dataframe(uf[["Meaning", "Feature", "Compared with typical traffic"]], hide_index=True, use_container_width=True)

    # ---- ACTION ----
    with tab_do:
        if sev == "Review":
            st.warning("The model is uncertain about this flow's classification. Inspect the traffic and verify the recommended action before responding.")
        with st.container(border=True):
            st.markdown(f"**Suggested next step for {cat}**")
            st.write(CLASS_INFO[cat]["action"])
        st.caption("These are general guidelines. Follow your organisation's incident-response process.")

    # ---- TECH ----
    with tab_tech:
        if exp["neighbors"]:
            tech = pd.DataFrame([{
                "Rank": x["rank"], "Neighbour flow": x["neighbor_index"], "Predicted type": x["neighbor_category"],
                "Layer-2 attention": round(x["decision_attention"], 4),
                "Layer-1 mean attention": round(x["layer1_attention"], 4),
                **{f"Head {h + 1}": round(v, 4) for h, v in enumerate(x["heads"])},
            } for x in exp["neighbors"]])
            st.dataframe(tech, hide_index=True, use_container_width=True)
        with st.expander("Full text report"):
            try:
                st.code(format_explanation_report(exp), language="text")
            except Exception:  # noqa: BLE001 - report format is optional
                st.caption("Text report not available for this flow.")


def page_network():
    context_bar("Network map", "Each dot is a flow, coloured by the model's verdict. Lines join flows that behave alike.")
    run = require_run()
    if run is None:
        return
    data = run["data"]
    n = run["n"]

    c1, c2 = st.columns([1, 1])
    focus = st.session_state.get("focus")
    target = c1.number_input("Highlight flow #", 0, n - 1, int(focus) if focus is not None and focus < n else 0)
    max_nodes = c2.slider("Flows to draw", min(20, n), max(20, min(200, n)), min(80, n), help="Fewer dots give a cleaner picture.") if n > 20 else n

    ei = data.edge_index.cpu().numpy()
    neighbors = sorted({int(d) for s, d in zip(ei[0], ei[1]) if s == target})
    fig = build_interactive_graph_figure(data, max_nodes=int(max_nodes), highlight_node=int(target), neighbor_indices=neighbors)
    fig = light(fig, 560)
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, title="")
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, title="")
    fig.update_xaxes(showline=False)
    fig.update_yaxes(showline=False)
    names = {"Flow Similarity Edges": "Similarity link", "Target Attention Edges": "Links of selected flow"}
    fig.for_each_trace(lambda tr: tr.update(name=names.get(tr.name, tr.name)))
    fig.update_layout(title="", legend=dict(orientation="h", y=-0.05))
    st.plotly_chart(fig, use_container_width=True, config=CHART_CFG)

    row = run["table"].iloc[int(target)]
    with st.container(border=True):
        st.markdown(f"**Flow {int(target)}** · {SEVERITY[row['_sev']][0]} {row['Prediction']} ({row['Confidence']:.0f}%) · "
                    f"linked to **{len(neighbors)}** similar flows: {', '.join(map(str, neighbors)) or 'none'}")
        if st.button("Investigate this flow", icon=":material/search:"):
            st.session_state["focus"] = int(target)
            st.switch_page(PAGES["investigate"])
    st.caption(f"{n:,} flows · {run['edges']:,} links · each flow linked to its {K} nearest flows (Euclidean distance on 45 scaled features). "
               "Links mean statistical similarity, not physical wiring.")


def page_report():
    context_bar("Model report", "How well the model learned, and what it was trained on.")
    h = load_history()
    if not h:
        st.info("No training history found at results/training_history.json.")
        return
    hist = h.get("history", {})
    cfg = h.get("config", {})

    m = st.columns(4)
    m[0].metric("Score (macro F1)", f"{h.get('best_val_macro_f1', 0) * 100:.1f}%",
                help="Average F1 over the 5 classes on validation data. Balances missed attacks and false alarms. 100% is perfect.")
    if hist.get("val_acc"):
        m[1].metric("Validation accuracy", f"{max(hist['val_acc']) * 100:.1f}%")
    m[2].metric("Best epoch", h.get("best_epoch", "-"), help="Training stopped once the score stopped improving.")
    m[3].metric("Architecture", f"{cfg.get('num_layers', 2)}-layer GAT · {cfg.get('heads', 4)} heads")

    c1, c2 = st.columns(2)
    epochs = list(range(1, len(hist.get("train_loss", [])) + 1))
    with c1:
        fig = go.Figure()
        for key, name, color in [("train_loss", "Training", "#2563eb"), ("val_loss", "Validation", "#f59e0b")]:
            if hist.get(key):
                fig.add_trace(go.Scatter(x=epochs, y=hist[key], name=name, mode="lines", line=dict(color=color, width=2.5)))
        fig.update_layout(title="Error (loss) — should go down", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(light(fig, 330), use_container_width=True, config=CHART_CFG)
    with c2:
        fig = go.Figure()
        for key, name, color in [("train_acc", "Training accuracy", "#2563eb"), ("val_acc", "Validation accuracy", "#f59e0b"),
                                 ("val_macro_f1", "Validation F1", "#16a34a")]:
            if hist.get(key):
                fig.add_trace(go.Scatter(x=epochs, y=hist[key], name=name, mode="lines", line=dict(color=color, width=2.5)))
        if h.get("best_epoch"):
            fig.add_vline(x=h["best_epoch"], line_dash="dot", line_color="#6b7280", annotation_text="best")
        fig.update_layout(title="Accuracy and F1 — should go up", xaxis_title="Epoch", yaxis=dict(tickformat=".0%"))
        st.plotly_chart(light(fig, 330), use_container_width=True, config=CHART_CFG)

    scores = test_set_scores()
    if scores:
        st.markdown(f"#### Results on the test set ({scores['n']} flows the model never saw)")
        st.caption(f"Overall accuracy: **{scores['accuracy']:.0%}**. "
                   "Recall = share of that type the model caught. Precision = how often a flag of that type was right.")
        pc = scores["per_class"]
        long = pc.melt(id_vars="Type", value_vars=["Caught (recall)", "Correct when flagged (precision)"],
                       var_name="Measure", value_name="Score")
        fig = px.bar(long, x="Score", y="Type", color="Measure", barmode="group", orientation="h",
                     color_discrete_map={"Caught (recall)": "#2563eb", "Correct when flagged (precision)": "#93c5fd"},
                     title="How well each type is detected")
        fig.update_layout(xaxis=dict(tickformat=".0%", range=[0, 1]), yaxis=dict(title="", categoryorder="array",
                          categoryarray=CLASSES[::-1]), legend_title="")
        st.plotly_chart(light(fig, 340), use_container_width=True, config=CHART_CFG)
        weakest = pc.sort_values("Caught (recall)").iloc[0]

    dist = load_training_distribution()
    with st.container(border=True):
        st.markdown("**Good to know**")
        lines = []
        if dist:
            lines.append(f"- The GAT was trained on a **balanced** graph of {sum(dist.values()):,} flows "
                         f"({', '.join(f'{v} {k}' for k, v in dist.items())}). That is small, so results on very different networks may vary.")
        lines.append("- The scaler was fitted on a larger, **imbalanced** sample (mostly DoS/DDoS), so typical values lean towards attack traffic.")
        if scores:
            lines.append(f"- The hardest type to catch on the test set is **{weakest['Type']}** "
                         f"({weakest['Caught (recall)']:.0%} caught).")
        lines.append(f"- Flows with confidence below {REVIEW_BELOW:.0%} are flagged **Needs review**: a simple way to surface "
                     "traffic that may not match any known attack type.")
        st.markdown("\n".join(lines))
    with st.expander("Technical settings"):
        st.json(cfg)


def page_guide():
    context_bar("Guide", "Everything you need to know to use this dashboard.")
    t1, t2, t3 = st.tabs(["Input file", "Results", "Glossary"])
    with t1:
        st.markdown(
            f"**What to upload:** a CSV where each row is one network flow from the CICIoMT2024 feature format.\n\n"
            f"- All **45 numeric columns** must be present with the exact names (order doesn't matter).\n"
            f"- Extra columns (like a label or file name) are ignored.\n"
            f"- **{K + 1} to {MAX_ROWS:,} rows**, up to **{MAX_MB} MB**. Extra rows are cut off.\n"
            f"- Empty or invalid values are replaced with 0.\n"
            f"- Don't scale the numbers yourself; the dashboard uses the scaler saved from training.\n"
            f"- Optional: a `label_category` column with the true type lets the dashboard show accuracy."
        )
        st.code(", ".join(EXPECTED_FEATURES), language="text")
    with t2:
        st.markdown("**The five possible verdicts**")
        st.dataframe(pd.DataFrame([{"Verdict": c, "Meaning": CLASS_INFO[c]["short"], "Description": CLASS_INFO[c]["what"]}
                                   for c in CLASSES]), hide_index=True, use_container_width=True)
        st.markdown("**Severity levels**")
        st.dataframe(pd.DataFrame([{"Severity": f"{v[0]} {k}", "Meaning": v[2]} for k, v in SEVERITY.items()]),
                     hide_index=True, use_container_width=True)
    with t3:
        st.markdown(
            "- **Flow** — a summary of one conversation between devices (rates, sizes, flags...).\n"
            "- **Graph** — flows are dots; each is linked to its 5 most similar flows.\n"
            "- **GAT (Graph Attention Network)** — a neural network that looks at a flow and its neighbours together.\n"
            "- **Attention** — how much the model relied on each neighbour. Shown as the explanation. Not proof of cause.\n"
            "- **Confidence** — the model's probability for its chosen verdict.\n"
            "- **F1 score** — a single accuracy number that balances missed attacks and false alarms.\n"
            "- **CICIoMT2024** — the public medical-IoT attack dataset the model was trained on."
        )


# ==========================================================================
# Navigation
# ==========================================================================
PAGES = {
    "home": st.Page(page_home, title="Home", icon=":material/home:", default=True),
    "analyze": st.Page(page_analyze, title="Analyze traffic", icon=":material/upload_file:", url_path="analyze"),
    "alerts": st.Page(page_alerts, title="Alerts", icon=":material/notifications:", url_path="alerts"),
    "investigate": st.Page(page_investigate, title="Investigate", icon=":material/search:", url_path="investigate"),
    "network": st.Page(page_network, title="Network map", icon=":material/hub:", url_path="network"),
    "report": st.Page(page_report, title="Model report", icon=":material/monitoring:", url_path="report"),
    "guide": st.Page(page_guide, title="Guide", icon=":material/help:", url_path="guide"),
}

nav = st.navigation({
    "": [PAGES["home"]],
    "Detect": [PAGES["analyze"], PAGES["alerts"], PAGES["investigate"]],
    "Explore": [PAGES["network"]],
    "About": [PAGES["report"], PAGES["guide"]],
})

with st.sidebar:
    if load_explainer() is None:
        st.error("Model file missing: models/gat_best_checkpoint.pt")
    run_ = st.session_state.get("run")
    if run_:
        a = int((run_["table"]["Prediction"] != "Benign").sum())
        st.caption(f"**Loaded:** {run_['name']}")
        st.caption(f"{run_['n']:,} flows · {a} flagged as attacks")
    st.caption("GAT-XIDS · CICIoMT2024")

nav.run()
