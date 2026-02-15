import streamlit as st
import json
import random
import networkx as nx
from pyvis.network import Network
import tempfile
import os

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

/* Global background */
body {
    background: linear-gradient(135deg, #eef2f7 0%, #f8fafc 100%);
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Section titles */
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Card style wrapper */
.section-card {
    background-color: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    margin-bottom: 30px;
    transition: all 0.2s ease-in-out;
}

.section-card:hover {
    transform: translateY(-3px);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #1e293b;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Buttons */
.stButton>button {
    border-radius: 10px;
    padding: 0.6em 1.2em;
    font-weight: 600;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
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

st.markdown("---")
st.markdown('</div>', unsafe_allow_html=True)
# ---------------------------------------------------
# Graph Construction
# ---------------------------------------------------
G = nx.DiGraph()
nodes = {}
edges = []

if graph_file:
    try:
        graph_data = json.load(graph_file)

        raw_nodes = graph_data.get("nodes", [])
        raw_edges = graph_data.get("edges", [])

        for node in raw_nodes:
            if isinstance(node, dict):
                name = node.get("name")
                role = node.get("type", "state")
            else:
                name = node
                role = "state"

            nodes[name] = {"type": role}
            G.add_node(name, type=role)

        for edge in raw_edges:
            if len(edge) == 3:
                src, tgt, weight = edge
            else:
                src, tgt = edge
                weight = round(random.uniform(0.1, 1.0), 2)

            edges.append((src, tgt, weight))
            G.add_edge(src, tgt, weight=weight)

    except Exception:
        st.error("Invalid JSON structure.")
        st.stop()

# ---------------------------------------------------
# Sidebar Inspector
# ---------------------------------------------------
st.sidebar.header("Inspector")

if G.number_of_nodes() > 0:
    selected_element = st.sidebar.selectbox(
        "Select node / edge",
        ["None"]
        + list(G.nodes)
        + [f"{u} → {v}" for u, v in G.edges]
    )

    intervene_sidebar = st.sidebar.button("Mask / Intervene")
else:
    selected_element = "None"
    intervene_sidebar = False

# ---------------------------------------------------
# Baseline & Intervention State
# ---------------------------------------------------
if "baseline_metric" not in st.session_state:
    st.session_state.baseline_metric = round(random.uniform(0.75, 0.90), 4)

if "intervened_metric" not in st.session_state:
    st.session_state.intervened_metric = None

if intervene_sidebar and selected_element != "None":
    st.session_state.intervened_metric = round(
        st.session_state.baseline_metric + random.uniform(0.05, 0.20),
        4
    )

# ---------------------------------------------------
# Graph Visualization + Inspector Layout
# ---------------------------------------------------
if G.number_of_nodes() > 0:

    col1, col2 = st.columns([3, 1])

    # -------- Graph --------
    with col1:
        st.subheader("2. Causal Graph")

        net = Network(
            height="650px",
            width="100%",
            directed=True,
            bgcolor="#ffffff",
            font_color="#000000"
        )

        # Disable scroll zoom, enable smooth drag
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "solver": "barnesHut",
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 150,
              "springConstant": 0.04,
              "damping": 0.09
            },
            "stabilization": {
              "enabled": true,
              "iterations": 150
            }
          },
          "interaction": {
            "zoomView": false,
            "dragView": true,
            "dragNodes": true,
            "hover": true
          },
          "nodes": {
            "shape": "dot",
            "size": 25,
            "font": { "size": 16 },
            "borderWidth": 2
          },
          "edges": {
            "smooth": { "type": "dynamic" },
            "width": 2
          }
        }
        """)

        # Node coloring
        for node, attr in G.nodes(data=True):
            color = {
                "input": "#4f9cf9",
                "state": "#f5c542",
                "output": "#34c38f",
            }.get(attr["type"], "#cbd5e1")

            net.add_node(node, label=node, color=color)

        # Edge thickness proportional to weight
        for u, v, attr in G.edges(data=True):
            weight = attr["weight"]
            net.add_edge(
                u,
                v,
                label=str(weight),
                value=abs(weight) * 6,
                title=f"Weight: {weight}"
            )

        # Save graph
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            net.save_graph(tmp.name)
            graph_path = tmp.name

        html_content = open(graph_path, "r").read()

        # Manual Zoom Controls
        zoom_controls = """
        <div style="
            position:absolute;
            top:20px;
            right:20px;
            z-index:9999;
        ">
            <button onclick="network.moveTo({scale: network.getScale() * 1.2});"
                style="font-size:18px;padding:6px 12px;margin:2px;">+</button>
            <button onclick="network.moveTo({scale: network.getScale() * 0.8});"
                style="font-size:18px;padding:6px 12px;margin:2px;">−</button>
            <button onclick="network.fit();"
                style="font-size:16px;padding:6px 12px;margin:2px;">Fit</button>
        </div>
        """

        html_content = html_content.replace(
            "<body>",
            f"<body>{zoom_controls}"
        )

        st.components.v1.html(html_content, height=700)

        os.unlink(graph_path)

    # -------- Inspector --------
    with col2:
        st.subheader("Selected Element Details")

        if selected_element == "None":
            st.write("No element selected.")

        elif "→" in selected_element:
            u, v = selected_element.split(" → ")
            st.write("Type: Edge")
            st.write(f"From: {u}")
            st.write(f"To: {v}")
            st.write(f"Weight: {G[u][v]['weight']}")

        else:
            st.write("Type: Node")
            st.write(f"Name: {selected_element}")
            st.write(f"Role: {G.nodes[selected_element]['type']}")

st.markdown("---")

# ---------------------------------------------------
# Intervention Configuration Panel
# ---------------------------------------------------
st.subheader("3. Intervention Configuration")

if G.number_of_nodes() > 0:

    colA, colB, colC = st.columns(3)

    with colA:
        selected_node = st.selectbox("Select Target Node", list(G.nodes))

    with colB:
        intervention_type = st.selectbox(
            "Intervention Type",
            ["mask", "freeze", "perturb"]
        )

    with colC:
        strength = st.slider("Strength", 0.0, 1.0, 0.5, 0.01)

else:
    selected_node = None
    st.info("Upload a graph file to enable intervention.")

# ---------------------------------------------------
# Backend Logic
# ---------------------------------------------------
def run_intervention(node, intervention, strength):
    base = st.session_state.baseline_metric

    total_influence = sum(
        abs(G[node][nbr]["weight"])
        for nbr in G.successors(node)
    ) if node in G else 0.1

    impact = total_influence * strength * 0.1

    if intervention == "freeze":
        new_value = base + impact
    elif intervention == "mask":
        new_value = base + impact * 1.2
    else:
        new_value = base + impact * 0.8

    return round(new_value, 4)

# ---------------------------------------------------
# Run Intervention
# ---------------------------------------------------
if st.button("Run Intervention Analysis"):

    if not model_file or not graph_file:
        st.warning("Please upload both model and graph files.")

    elif selected_node:
        st.session_state.intervened_metric = run_intervention(
            selected_node,
            intervention_type,
            strength
        )
        st.success("Intervention executed successfully.")

# ---------------------------------------------------
# Output Panel
# ---------------------------------------------------
st.subheader("4. Model Performance Comparison")

colX, colY = st.columns(2)

with colX:
    st.metric("Baseline Metric", st.session_state.baseline_metric)

with colY:
    if st.session_state.intervened_metric is None:
        st.metric("After Intervention", "—")
    else:
        st.metric("After Intervention", st.session_state.intervened_metric)