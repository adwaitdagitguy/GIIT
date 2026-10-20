# SciML Intervention Workbench (React + FastAPI)

A research workbench for inspecting and probing Physics-Informed Neural Networks (PINNs).

## Level 0 Quickstart: Running Backend & Frontend

### 1. Backend (FastAPI) Setup:
Open a PowerShell/terminal window in your project root:
```bash
# 1. Install missing python-multipart (required for file uploads)
pip install python-multipart

# 2. Run the FastAPI backend server on port 8000
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
You can verify the backend API docs at: `http://127.0.0.1:8000/docs`

---

### 2. Frontend (React + Vite + Tailwind + React Flow) Setup:
Open a second PowerShell/terminal window in the `frontend` folder:
```bash
# 1. Install node dependencies (if not already installed)
npm install

# 2. Start the Vite dev server
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## Features Included:
- **Model & Checkpoint Loader**: Drag & drop `.py` architecture and `.pt` weights.
- **Interactive Causal Physics Graph**: Built with React Flow, color-coded node types (Input, Latent, Parameter, Derivative, Loss Residual, Output).
- **Node Inspector**: Math formulas, ODE relations, qualitative trend expectations.
- **Intervention Studio**: Target layer selector, noise injection / weight masking / gain / clamping, strength slider ($0.0 \to 1.0$), "Run Intervention", and "Sweep Strengths".
- **Real-Time Analytics Dock**:
  - Trajectory comparison plot (Plotly.js) with Baseline vs Perturbed vs Ground Truth.
  - Error & loss metrics panel (MSE, RMSE, L2 norms, Max Divergence, % Degradation).
  - Physics Residual & Autograd Sensitivity analysis table.
  - Automated Physical Concept $\to$ Neural Layer Mapping table.
  - Parameter sweep degradation curves.
  - AI Physics Assistant chat.
- **Interactive Guided Tour**: 5-step onboarding walkthrough.
- **Offline / Mock Mode Fallback**: Instant testing with "Load Damped Oscillator Demo" even when backend is offline.
