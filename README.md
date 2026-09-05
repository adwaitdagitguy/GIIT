# GIIT: Graph-Based Inspection and Intervention Tool for PINNs

GIIT is an interactive tool designed to inspect and evaluate Physics-Informed Neural Networks (PINNs).

Rather than evaluating a network solely by its training loss or residual error, GIIT examines the model's internal representations. It determines whether internal network layers correspond to physical variables (such as mass, damping, spring constant, velocity, and acceleration) or if the model relies on localized numerical shortcuts.

---

## Key Features

1. **Interactive Physics Graph**:
   - Visualizes the governing physics equations as a directional causal graph.
   - Illustrates how physical parameters connect to intermediate derivative states and output trajectories.

2. **Automated Neural-to-Physics Mapping**:
   - Evaluates internal weights and gradients of the PyTorch model.
   - Automatically maps network layers to specific physical variables and parameters using sensitivity and perturbation analysis.

3. **Targeted Interventions**:
   - Applies controlled interventions (noise perturbations or masking) directly to internal layers.
   - Evaluates the effect of internal interventions on physical outputs in real time.

4. **Multi-Seed Stability Analysis**:
   - Performs perturbation sweeps across multiple random seeds to measure the consistency of discovered representations.

5. **In-Distribution vs. Out-of-Distribution (OOD) Transfer Analysis**:
   - Evaluates whether internal physical correspondences transfer across distribution shifts (such as temporal extrapolation) or exhibit representation collapse.

---

## Repository Structure (can be slightly different)

```text
app/
├── README.md                   # Project overview and setup instructions
├── app.py                      # Main Streamlit dashboard application
├── reps.tex                    # Research manuscript (LaTeX)
├── inbuilt_models/             # Pretrained PINN models and default graphs
│   ├── damped_oscillator.pt    # Pretrained damped oscillator PyTorch weights
│   ├── damped_oscillator_graph.json # Physics graph specification
│   ├── physics_ruleset.json    # Executable ruleset for derived physical quantities
│   └── harmonic_oscillator_pinn.py
├── core/
│   ├── dynamic_intervener.py   # Core physics and intervention engine
│   ├── model_loader.py         # Dynamic PyTorch model loader
│   ├── intervention.py         # Base intervention routines
│   └── llm_engine.py           # LLM-assisted graph generator
├── utils/
│   └── physics_utils.py        # Ruleset execution and derived metric utilities
└── components/
    └── styles.py               # Application UI styling
```

---

## Getting Started

### 1. Requirements and Installation

Ensure Python 3.10 or newer is installed. Install the necessary dependencies: [refer req.txt]

```bash
pip install torch numpy matplotlib streamlit streamlit-agraph networkx pandas
```

### 2. Running the Dashboard

Start the Streamlit application with:

```bash
streamlit run app.py
```

Once started, open `http://localhost:8501` in your browser.

---

## Workflow Overview

1. **Load Model and Graph**:
   - Select the built-in Damped Harmonic Oscillator or upload a custom PyTorch model (`.pt` or `.py`) along with its physics graph specification (`graph.json`).
2. **Inspect the Graph**:
   - Review the causal structure showing directional relationships between parameters, derivatives, and outputs.
3. **Run Automated Mapping**:
   - Execute the mapping routine to identify which layers align with specific physical trends and gradients.
   - Review the Alignment Accuracy score to verify physical consistency.
4. **Apply Interventions**:
   - Select mapped layers, adjust intervention parameters, and observe the resulting changes in system dynamics.
5. **Evaluate Seed Stability**:
   - Run multi-seed perturbation sweeps to test the stability and distribution of identified representations.

---

## Reproducing Manuscript Results

To reproduce all quantitative metrics, sensitivity values, and in-distribution versus out-of-distribution transfer analyses reported in the manuscript, execute:

```bash
python burgers_temporal_experiment.py
python dho_temporal_experiment.py
```

---

