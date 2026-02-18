import streamlit as st
import json
import random
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sciml_intervener import SciMLIntervener, TinyMLP

intervener = None

# ---------------------------------------------------
# Page Setup
# ---------------------------------------------------
st.set_page_config(
    page_title="Graph-Based Inspection and Intervention Interface",
    layout="wide"
)

st.title("Graph-Based Inspection and Intervention Interface")

st.markdown("""
Structured inspection of computational graphs with interactive visualization,
element-level inspection, and controlled intervention analysis.
""")

st.markdown("---")

# ---------------------------------------------------
# Professional UI Styling
# ---------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #eef2f7 0%, #f8fafc 100%);
}
.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 600;
}
.section-card {
    background-color: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    margin-bottom: 30px;
}
section[data-testid="stSidebar"] {
    background-color: #1e293b;
}
section[data-testid="stSidebar"] * {
    color: white !important;
}
.stButton>button {
    border-radius: 10px;
    padding: 0.6em 1.2em;
    font-weight: 600;
}
[data-testid="stMetric"] {
    background-color: #ffffff;
    color: #000000 !important;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

[data-testid="stMetric"] label {
    color: #333333 !important;
}

[data-testid="stMetricValue"] {
    color: #000000 !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Upload Section
# ---------------------------------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("1. Upload Model and Graph")

col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    model_file = st.file_uploader("Upload Model File (.pt)", type=["pt"])

with col_upload2:
    graph_file = st.file_uploader("Upload Graph File (.json)", type=["json"])

if model_file:
    st.success(f"Model loaded: {model_file.name}")

if graph_file:
    st.success(f"Graph loaded: {graph_file.name}")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------
# Graph Construction
# ---------------------------------------------------
G = nx.DiGraph()

if graph_file:
    try:
        graph_data = json.load(graph_file)
        raw_nodes = graph_data.get("nodes", [])
        raw_edges = graph_data.get("edges", [])

        for node in raw_nodes:
            if isinstance(node, dict):
                G.add_node(node["name"], type=node.get("type", "state"))
            else:
                G.add_node(node, type="state")

        for edge in raw_edges:
            if len(edge) == 3:
                G.add_edge(edge[0], edge[1], weight=edge[2])
            else:
                G.add_edge(edge[0], edge[1], weight=1.0)

    except Exception:
        st.error("Invalid JSON structure.")
        st.stop()
    # ---------------------------------------------------
    # Backend Intervener Initialization
    # ---------------------------------------------------
    intervener = None

    if model_file and graph_file:
        try:
            # Save uploaded files temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_model:
                tmp_model.write(model_file.getbuffer())
                model_path = tmp_model.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_graph:
                tmp_graph.write(graph_file.getbuffer())
                graph_path_backend = tmp_graph.name


            intervener = SciMLIntervener(model_path, graph_path_backend)

        except Exception as e:
            st.error(f"Backend initialization failed: {e}")


# ---------------------------------------------------
# Sidebar Inspector
# ---------------------------------------------------
st.sidebar.header("Inspector")

if G.number_of_nodes() > 0:
    selected_element = st.sidebar.selectbox(
        "Select node / edge",
        ["None"] + list(G.nodes) + [f"{u} → {v}" for u, v in G.edges]
    )
else:
    selected_element = "None"

# ---------------------------------------------------
# State Initialization
# ---------------------------------------------------
if "baseline_metric" not in st.session_state:
    st.session_state.baseline_metric = None

if "intervened_metric" not in st.session_state:
    st.session_state.intervened_metric = None

if "plot_data" not in st.session_state:
    st.session_state.plot_data = None
# ---------------------------------------------------
# Graph Visualization (STATIC ZOOM)
# ---------------------------------------------------
if G.number_of_nodes() > 0:

    st.subheader("2. Causal Graph")

    net = Network(height="650px", width="100%", directed=True,
                  bgcolor="#ffffff", font_color="#000000")

    net.set_options("""
    {
      "physics": { "enabled": true },
      "interaction": {
        "zoomView": false,
        "dragView": true,
        "dragNodes": true
      }
    }
    """)

    for node, attr in G.nodes(data=True):
        color = {
            "input": "#4f9cf9",
            "state": "#f5c542",
            "output": "#34c38f",
        }.get(attr["type"], "#cbd5e1")

        net.add_node(node, label=node, color=color)

    for u, v, attr in G.edges(data=True):
        net.add_edge(u, v, label=str(attr["weight"]))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name)
        graph_path = tmp.name

    html_content = open(graph_path, "r").read()

    zoom_controls = """
    <div style="position:absolute; top:20px; right:20px; z-index:9999;">
        <button onclick="network.moveTo({scale: network.getScale()*1.2});">+</button>
        <button onclick="network.moveTo({scale: network.getScale()*0.8});">-</button>
        <button onclick="network.fit();">Fit</button>
    </div>
    """

    html_content = html_content.replace("<body>", f"<body>{zoom_controls}")

    st.components.v1.html(html_content, height=700)
    os.unlink(graph_path)

# ---------------------------------------------------
# Intervention Configuration
# ---------------------------------------------------
st.subheader("3. Intervention Configuration")

if G.number_of_nodes() > 0:
    selected_node = st.selectbox("Select Target Node", list(G.nodes))
    intervention_type = st.selectbox(
        "Intervention Type", ["mask", "freeze", "perturb"]
    )
    strength = st.slider("Strength", 0.0, 1.0, 0.5, 0.01)
else:
    selected_node = None

# ---------------------------------------------------
# Node Influence Logic
# ---------------------------------------------------
def compute_node_influence(node):
    outgoing = sum(abs(G[node][nbr]["weight"]) for nbr in G.successors(node))
    incoming = sum(abs(G[pred][node]["weight"]) for pred in G.predecessors(node))
    return outgoing + incoming

# ---------------------------------------------------
# Run Intervention
# ---------------------------------------------------
if st.button("Run Intervention Analysis"):

    if intervener and selected_node:

        plot_data = intervener.get_plot_data(
            node_name=selected_node,
            intervention_type=intervention_type,
            strength=strength,
            test_inputs=torch.load("test_inputs.pt"),
            test_targets=torch.load("test_targets.pt")
        )

        st.session_state.plot_data = plot_data
        st.session_state.baseline_metric = plot_data["baseline_metric"]
        st.session_state.intervened_metric = plot_data["intervened_metric"]

# ---------------------------------------------------
# Results Area
# ---------------------------------------------------
st.markdown("---")
st.subheader("Results Area")

if G.number_of_nodes() > 0:

    if st.session_state.plot_data is not None:

        plot_data = st.session_state.plot_data

        x = plot_data["x_values"]
        y_baseline = plot_data["y_preds_baseline"]
        y_intervened = plot_data["y_preds_intervened"]

        # Sort for clean plotting
        sorted_indices = np.argsort(x)

        x = x[sorted_indices]
        y_baseline = y_baseline[sorted_indices]
        y_intervened = y_intervened[sorted_indices]


        # -------- Plot A: Model Predictions --------
        fig1, ax1 = plt.subplots()

        ax1.plot(x, y_baseline, color="blue", label="Baseline")
        ax1.plot(x, y_intervened, color="red", linestyle="--", label="Intervened")

        ax1.set_title("Model Predictions")
        ax1.set_xlabel("Input")
        ax1.set_ylabel("Output")
        ax1.legend()

        st.pyplot(fig1)

        # -------- Plot B: Loss Landscape --------
        fig2, ax2 = plt.subplots()

        ax2.bar(
            ["Baseline", "Intervened"],
            [plot_data["baseline_metric"], plot_data["intervened_metric"]],
            color=["blue", "red"]
        )

        ax2.set_title("Loss Landscape (MSE)")
        ax2.set_ylabel("MSE Value")

        st.pyplot(fig2)

    else:
        st.info("Run intervention to generate model comparison plots.")

else:
    st.info("Upload a valid graph to enable results visualization.")

# ---------------------------------------------------
# Reset
# ---------------------------------------------------
if st.button("Reset Model"):
    st.session_state.intervened_metric = None
    if "plot_data" in st.session_state:
        del st.session_state.plot_data
# ---------------------------------------------------
# Metrics
# ---------------------------------------------------
st.markdown("---")
st.subheader("4. Model Performance Comparison")

colX, colY = st.columns(2)

with colX:
    st.metric("Baseline Metric", st.session_state.baseline_metric)

with colY:
    if st.session_state.intervened_metric is None:
        st.metric("After Intervention", "—")
    else:
        st.metric("After Intervention", st.session_state.intervened_metric)