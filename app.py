"""
Graph-Based Inspection and Intervention Interface
Main Streamlit application entry point.

Architecture:
  - core/model_loader.py      : Dynamic model loading from any .py file
  - core/dynamic_intervener.py: Physics engine (reads all rules from graph.json)
  - core/llm_engine.py        : LLM-based graph.json auto-generation
  - utils/physics_utils.py    : Node explanation builder
  - components/styles.py      : Centralized CSS
"""

# ── Global Plot Muting ───────────────────────────────────────
# This prevents plt.show() from blocking or warning in a headless environment.
try:
    import matplotlib
    matplotlib.use('Agg') # Force non-interactive backend
    import matplotlib.pyplot as plt
    plt.show = lambda *args, **kwargs: None
except ImportError:
    pass

# ── Standard Library ──────────────────────────────────────────
import os
import tempfile
import json

# ── Third-Party ───────────────────────────────────────────────
import streamlit as st
import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt
from streamlit_agraph import agraph, Node, Edge, Config
import pandas as pd

# ── Local Modules ─────────────────────────────────────────────
from core.model_loader import load_model_from_file, get_model_architecture_summary
from core.dynamic_intervener import DynamicSciMLIntervener, generate_test_data
from core.llm_engine import generate_graph_json
from utils.physics_utils import build_node_explanations
from components.styles import apply_custom_css

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Graph-Based Inspection and Intervention Interface",
    layout="wide"
)
apply_custom_css()

st.title("Graph-Based Inspection and Intervention Interface")
st.markdown("""
Structured inspection of computational graphs with interactive visualization,
element-level inspection, and controlled intervention analysis.
""")
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# GLOBAL GRAPH OBJECT
# ─────────────────────────────────────────────────────────────
G = nx.DiGraph()

# ─────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────
_defaults = {
    "intervener":        None,
    "baseline_metric":   None,
    "intervened_metric": None,
    "plot_data":         None,
    "l2_norm":           None,
    "physics_residual":  None,
    "test_inputs":       None,
    "test_targets":      None,
    "zoom_level":        1.0,
    "clicked_node":      None,
    "last_click_ts":     None,
    "mapping_results":   None,
    "alignment_score":   None,
    "node_explanations": {},
    "graph_data":        None,
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────────────────
# SECTION 0: LLM Graph Generator (Optional)
# ─────────────────────────────────────────────────────────────
with st.expander("⚡ Auto-Generate graph.json from Model using LLM (Optional)"):
    st.markdown("Upload your model `.py` file and provide an API key to generate `graph.json` automatically.")
    col_llm1, col_llm2 = st.columns(2)
    with col_llm1:
        llm_model_py = st.file_uploader("Upload Model `.py` file", type=["py"], key="llm_py")
    with col_llm2:
        api_key = st.text_input("OpenRouter API Key", type="password", key="llm_api_key")

    if st.button("🤖 Generate Graph JSON via LLM") and llm_model_py and api_key:
        with st.spinner("Calling LLM to generate graph.json..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="wb") as tmp_py:
                    tmp_py.write(llm_model_py.getbuffer())
                    tmp_py_path = tmp_py.name

                out_path = tmp_py_path.replace(".py", "_graph.json")
                graph = generate_graph_json(
                    model_py_path=tmp_py_path,
                    api_key=api_key,
                    output_path=out_path,
                    verbose=False
                )
                st.success("Graph JSON generated successfully! Download it below.")
                st.download_button(
                    "📥 Download graph.json",
                    data=json.dumps(graph, indent=2),
                    file_name="graph.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"LLM generation failed: {e}")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Upload Model and Graph
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("1. Upload Model and Graph")

col_upload1, col_upload2, col_upload3 = st.columns(3)
with col_upload1:
    model_py_file  = st.file_uploader("Upload Model Definition `.py`", type=["py"])
with col_upload2:
    model_pt_file  = st.file_uploader("Upload Model Weights `.pt`", type=["pt"])
with col_upload3:
    graph_file     = st.file_uploader("Upload Graph File `.json`", type=["json"])

if model_py_file:
    st.success(f"Model definition: {model_py_file.name}")
if model_pt_file:
    st.success(f"Model weights: {model_pt_file.name}")
if graph_file:
    st.success(f"Graph: {graph_file.name}")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# SECTION 1.5: Physics Ruleset (Optional)
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("1.5 Upload Physics Ruleset (Optional)")
rules_file    = st.file_uploader("Upload Physics Ruleset `.json`", type=["json"])
physics_rules = None
if rules_file:
    try:
        rules_file.seek(0)
        physics_rules = json.load(rules_file)
        st.success("Physics rules loaded successfully")
    except Exception as e:
        st.error(f"Invalid rules file: {e}")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────
graph_data = None
x_label, x_unit, y_label, y_unit = "Input", "", "Output", ""

if graph_file:
    try:
        graph_file.seek(0)
        graph_data = json.load(graph_file)
        st.session_state.graph_data = graph_data

        raw_nodes  = graph_data.get("nodes", [])
        raw_edges  = graph_data.get("edges", [])
        metadata   = graph_data.get("model_metadata", graph_data.get("metadata", {}))
        axis_info  = metadata.get("axis_labels", {})

        x_label    = axis_info.get("x_label", "Input")
        x_unit     = axis_info.get("x_unit", "")
        y_label    = axis_info.get("y_label", "Output")
        y_unit     = axis_info.get("y_unit", "")

        # Build NetworkX graph
        for node in raw_nodes:
            if isinstance(node, dict):
                nid = node.get("id", node.get("name", ""))
                if nid:
                    G.add_node(nid, type=node.get("type", "state"), label=node.get("label", nid))
            else:
                G.add_node(str(node), type="state")

        for edge in raw_edges:
            if isinstance(edge, dict):
                G.add_edge(edge.get("source", ""), edge.get("target", ""),
                           label=edge.get("label", ""))
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                G.add_edge(edge[0], edge[1])

        # Build node explanations
        st.session_state.node_explanations = build_node_explanations(graph_data, physics_rules)

    except Exception as e:
        st.error(f"Invalid graph JSON: {e}")
        st.stop()

# Rebuild explanations if physics rules uploaded after graph
if graph_data and physics_rules:
    st.session_state.node_explanations = build_node_explanations(graph_data, physics_rules)

# ─────────────────────────────────────────────────────────────
# BACKEND INITIALIZATION (Dynamic — No hardcoded model class)
# ─────────────────────────────────────────────────────────────
if model_pt_file and graph_file and st.session_state.intervener is None:
    try:
        # Save weights to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_pt:
            tmp_pt.write(model_pt_file.getbuffer())
            tmp_pt_path = tmp_pt.name

        # Save graph to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp_json:
            json.dump(graph_data, tmp_json)
            tmp_json_path = tmp_json.name

        # Dynamic model loading
        if model_py_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="wb") as tmp_py:
                tmp_py.write(model_py_file.getbuffer())
                tmp_py_path = tmp_py.name
            model = load_model_from_file(py_path=tmp_py_path, weights_path=tmp_pt_path)
        else:
            # Fallback: try loading as full model object
            model = torch.load(tmp_pt_path, map_location="cpu", weights_only=False)

        # Initialize the dynamic intervener
        st.session_state.intervener = DynamicSciMLIntervener(model=model, graph=graph_data)

        # Generate test data from graph.json
        t_inp, t_tgt = generate_test_data(graph_data)
        st.session_state.test_inputs  = t_inp
        st.session_state.test_targets = t_tgt

        st.success("✅ Backend initialized successfully!")

        # Show architecture summary
        if model_py_file:
            summary = get_model_architecture_summary(model)
            st.info(
                f"Model: **{summary['class_name']}** | "
                f"Input dim: **{summary['input_dim']}** | "
                f"Output dim: **{summary['output_dim']}** | "
                f"Params: **{summary['total_params']}**"
            )

    except Exception as e:
        st.error(f"Backend initialization failed: {e}")
        st.stop()

# ─────────────────────────────────────────────────────────────
# SECTION 2: Graph Visualization
# ─────────────────────────────────────────────────────────────
colz1, colz2 = st.columns(2)
with colz1:
    if st.button("➕ Zoom In"):
        st.session_state.zoom_level *= 1.2
with colz2:
    if st.button("➖ Zoom Out"):
        st.session_state.zoom_level /= 1.2

if G.number_of_nodes() > 0:
    st.subheader("2. Causal Graph")

    nodes = []
    for n, attr in G.nodes(data=True):
        ntype  = attr.get("type", "state")
        color  = {"input": "#3b82f6", "parameter": "#8b5cf6",
                  "intermediate": "#f59e0b", "output": "#10b981",
                  "state": "#f59e0b"}.get(ntype, "#3b82f6")
        expl   = st.session_state.node_explanations.get(n, {})
        tooltip = expl.get("description", f"Node: {n}")[:120]
        label  = attr.get("label", n)
        nodes.append(Node(id=n, label=label, size=25, color=color, title=tooltip))

    edges = []
    for u, v, attr in G.edges(data=True):
        edges.append(Edge(source=u, target=v, label=attr.get("label", "")))

    config = Config(
        width=900, height=600,
        directed=True, physics=True, hierarchical=False,
        interaction={"hover": True, "tooltipDelay": 200, "zoomView": False},
        manipulation=False,
    )
    clicked_node = agraph(nodes=nodes, edges=edges, config=config)
    if clicked_node and clicked_node != st.session_state.clicked_node:
        st.session_state.clicked_node = clicked_node
        st.rerun()

    st.markdown("""
    <div style="display:flex;gap:24px;padding:10px 0;font-size:13px;color:#475569;">
        <span>🔵 <b>Input</b></span>
        <span>🟣 <b>Parameter</b></span>
        <span>🟡 <b>Intermediate</b></span>
        <span>🟢 <b>Output</b></span>
        <span style="color:#94a3b8;margin-left:8px;">| 💡 Click node to inspect</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR: Node Inspector
# ─────────────────────────────────────────────────────────────
node_to_show      = st.session_state.clicked_node
NODE_EXPLANATIONS = st.session_state.node_explanations

with st.sidebar:
    st.markdown("## 🔍 Node Inspector")

    if node_to_show is None:
        st.markdown("""
        <div style="padding:24px;border-radius:14px;background:#f8fafc;
            border:2px dashed #cbd5e1;text-align:center;margin-top:16px;">
            <div style="font-size:36px;margin-bottom:12px;">🖱️</div>
            <div style="font-size:14px;font-weight:700;color:#475569;">No node selected</div>
            <div style="font-size:12px;margin-top:8px;color:#94a3b8;line-height:1.5;">
                Click any node in the graph<br>to inspect its physics details
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif node_to_show not in NODE_EXPLANATIONS:
        st.markdown(f"""
        <div style="padding:18px;border-radius:12px;background:#fff7ed;border:1px solid #fed7aa;">
            <div style="font-size:20px;margin-bottom:8px;">⚠️</div>
            <div style="font-weight:700;color:#9a3412;font-size:15px;">Node: {node_to_show}</div>
            <div style="font-size:13px;color:#c2410c;margin-top:6px;">
                No description found. Add metadata to graph.json.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        info = NODE_EXPLANATIONS[node_to_show]
        type_colors = {
            "input":        ("#dbeafe", "#1d4ed8"),
            "parameter":    ("#ede9fe", "#5b21b6"),
            "intermediate": ("#fef3c7", "#92400e"),
            "output":       ("#d1fae5", "#065f46"),
            "state":        ("#fef3c7", "#92400e"),
        }
        bg, fg = type_colors.get(info.get("type", "state"), ("#f1f5f9", "#334155"))

        st.markdown(f"""
        <div class="node-card-outer">
            <span class="node-badge">📐 Physics Node</span>
            <span class="node-type-badge" style="background:{bg};color:{fg};">
                {info.get("type", "state").upper()}
            </span>
        """, unsafe_allow_html=True)

        if info.get("domain"):
            st.markdown(f'<div class="domain-tag">🌐 {info["domain"]}</div>', unsafe_allow_html=True)

        st.markdown(f"### {info['title']}")
        st.markdown('<div class="info-section-title">📋 Description</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="desc-box">{info["description"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-section-title">📘 Formula</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="formula-box">{info["formula"]}</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-section-title">🎯 This Node Computes</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="highlight-term">{info["highlight"]}</div>', unsafe_allow_html=True)

        if info.get("variables"):
            st.markdown('<div class="info-section-title">📊 Variables</div>', unsafe_allow_html=True)
            for var, desc in info["variables"].items():
                st.markdown(f"""
                <div class="var-card">
                    <span class="var-symbol">{var}</span>
                    <span class="var-desc"> → {desc}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✕ Clear Inspector", key="clear_node_btn"):
            st.session_state.clicked_node  = None
            st.session_state.last_click_ts = None
            st.rerun()

# ─────────────────────────────────────────────────────────────
# SECTION 3: Intervention Configuration
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("3. Intervention Configuration")

if G.number_of_nodes() > 0:
    selected_node     = st.selectbox("Select Target Node", list(G.nodes))
    intervention_type = st.selectbox("Intervention Type", ["mask", "perturb"])
    strength          = st.slider("Strength", 0.0, 1.0, 0.5, 0.01)

    # ── Mapping Resolver ─────────────────────────────────────
    # Build a lookup: physical node label/id → neural layer name
    # This is populated after Section 5 mapping is run.
    mapping_results = st.session_state.get("mapping_results") or {}
    mapping_lookup  = {}  # { node_label: layer_name }
    for label, info in mapping_results.items():
        if isinstance(info, dict) and "layer" in info:
            mapping_lookup[label] = info["layer"]

    # Also check by node id from graph_data
    gd = st.session_state.get("graph_data") or {}
    for node in gd.get("nodes", []):
        node_id    = node.get("id", "")
        node_label = node.get("label", node_id)
        if node_label in mapping_lookup:
            mapping_lookup[node_id] = mapping_lookup[node_label]

    # Get the list of all actual neural layers
    intervener = st.session_state.get("intervener")
    all_layers = []
    if intervener:
        all_layers = list(intervener.model.state_dict().keys())

    # Resolve the selected physical node to a layer
    auto_layer = mapping_lookup.get(selected_node, None)

    st.markdown("#### 🗺️ Layer Mapping")
    if auto_layer:
        st.success(f"**Auto-mapped**: `{selected_node}` → `{auto_layer}` *(from Section 5 Mapping)*")
    else:
        st.warning(
            f"⚠️ No mapping found for `{selected_node}`. "
            "Run Section 5 (Auto Mapping) first, or select a layer manually below."
        )

    # Manual override dropdown
    layer_options = ["(use auto-mapping)"] + all_layers if auto_layer else all_layers
    manual_layer  = st.selectbox(
        "Manual Layer Override (optional)",
        options=layer_options,
        index=0,
        help="Overrides the auto-mapping. Useful for direct neural layer experiments."
    )

    # Final resolved layer
    if manual_layer and manual_layer != "(use auto-mapping)":
        resolved_layer = manual_layer
        st.info(f"**Using manual override**: `{resolved_layer}`")
    elif auto_layer:
        resolved_layer = auto_layer
    else:
        resolved_layer = None

else:
    selected_node  = None
    resolved_layer = None

if st.button("▶ Run Intervention Analysis"):
    intervener   = st.session_state.intervener
    test_inputs  = st.session_state.test_inputs
    test_targets = st.session_state.test_targets

    if intervener and selected_node and test_inputs is not None:
        if not resolved_layer:
            st.error(
                f"Cannot run intervention: no layer mapped to `{selected_node}`. "
                "Run Section 5 Mapping first or manually select a layer above."
            )
        else:
            try:
                result = intervener.apply_intervention(
                    layer_name=resolved_layer,
                    intervention_type=intervention_type,
                    strength=strength,
                    input_tensor=test_inputs,
                    target_tensor=test_targets,
                )
                st.session_state.baseline_metric   = result["baseline_mse"]
                st.session_state.intervened_metric = result["intervened_mse"]
                st.success(
                    f"Intervention on **{selected_node}** → `{resolved_layer}` complete | "
                    f"Baseline MSE: {result['baseline_mse']:.4f} → "
                    f"Intervened MSE: {result['intervened_mse']:.4f}"
                )
            except Exception as e:
                st.error(f"Intervention failed: {e}")
    else:
        st.warning("Upload model + graph and ensure backend is initialized.")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Full Physics Analysis
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("4. Full Physics Analysis")

if st.button("🔬 Run Full Analysis (L2 + Residual + Sensitivity)"):
    intervener  = st.session_state.intervener
    test_inputs = st.session_state.test_inputs
    test_targets = st.session_state.test_targets

    if intervener and test_inputs is not None:
        with st.spinner("Running full analysis..."):
            try:
                results = intervener.run_full_analysis(
                    input_tensor=test_inputs,
                    target_tensor=test_targets,
                )
                st.session_state.l2_norm        = results.get("l2_norm")
                st.session_state.physics_residual = results.get("physics_residual_norm")

                # L2 Norm
                if results.get("l2_norm") is not None:
                    l2 = results["l2_norm"]
                    if l2 <= 0.05:
                        st.success(f"✅ L2 Norm: {l2:.4f} (< 5% threshold — PASS)")
                    else:
                        st.error(f"❌ L2 Norm: {l2:.4f} (> 5% threshold — FAIL)")

                # Physics Residual
                res = results.get("physics_residual_norm")
                if res is not None and not np.isnan(res):
                    st.info(f"Physics Residual Norm: {res:.6e}")

                # Sensitivity Table
                sensitivity = results.get("sensitivity", {})
                if sensitivity:
                    st.markdown("#### Sensitivity Consistency")
                    df = pd.DataFrame([
                        {
                            "Parameter":      v["label"],
                            "Value":          v["value"],
                            "Gradient dR/dp": f"{v['gradient']:.6f}",
                            "Computed Sign":  v["computed_sign"],
                            "Expected Sign":  v["expected_sign"],
                            "Consistent":     "✅" if v["consistent"] else "❌",
                            "Trend":          v["trend"],
                        }
                        for v in sensitivity.values()
                    ])
                    st.dataframe(df, use_container_width=True)

            except Exception as e:
                st.error(f"Analysis failed: {e}")
    else:
        st.warning("Upload model + graph first.")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Automated Mapping Layer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("5. Automated Mapping Layer")

if st.button("🗺 Run Node Mapping"):
    intervener  = st.session_state.intervener
    test_inputs = st.session_state.test_inputs

    if intervener and test_inputs is not None:
        with st.spinner("Running systematic perturbation + node mapping..."):
            try:
                result = intervener.generate_mapping_with_confidence(
                    input_tensor=test_inputs
                )
                st.session_state.mapping_results = result["mapping"]
                st.session_state.alignment_score = result["summary"]["alignment_pct"]

            except Exception as e:
                st.error(f"Mapping failed: {e}")
    else:
        st.warning("Upload model + graph first.")

if st.session_state.mapping_results:
    st.markdown("#### Mapping Results")
    df = pd.DataFrame([
        {
            "Physical Node": node,
            "Mapped Layer":  info["layer"],
            "Confidence":    info["confidence"],
            "Score":         round(info.get("score", 0), 6),
            "Note":          info.get("note", ""),
        }
        for node, info in st.session_state.mapping_results.items()
    ])
    st.dataframe(df, use_container_width=True)

if st.session_state.alignment_score is not None:
    score = st.session_state.alignment_score
    if score >= 90:
        st.success(f"Alignment Accuracy: {score:.1f}% ✅ — Highly consistent with physics")
    elif score >= 70:
        st.warning(f"Alignment Accuracy: {score:.1f}% ⚠️ — Partial alignment")
    else:
        st.error(f"Alignment Accuracy: {score:.1f}% ❌ — Does not align with physics")

# ─────────────────────────────────────────────────────────────
# SECTION 6: Model Performance Metrics
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("6. Model Performance Comparison")

colX, colY = st.columns(2)
with colX:
    st.metric(
        label="Baseline MSE",
        value=f"{st.session_state.baseline_metric:.4f}" if st.session_state.baseline_metric is not None else "—"
    )
with colY:
    st.metric(
        label="Intervened MSE",
        value=f"{st.session_state.intervened_metric:.4f}" if st.session_state.intervened_metric is not None else "—"
    )

# ─────────────────────────────────────────────────────────────
# RESET
# ─────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("🔄 Reset Session"):
    for key in _defaults:
        st.session_state[key] = _defaults[key]
    st.rerun()
