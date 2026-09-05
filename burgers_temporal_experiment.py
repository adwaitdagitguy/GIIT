"""
burgers_temporal_experiment.py
==============================

Burgers PINN temporal OOD experiment using the GIIT framework.

Follows the same structure as reproduce_all_paper_experiments.py but for the
1D Viscous Burgers equation, with temporal evaluation across several windows.

Steps:
  1. Load existing burgers_1d.pt, burgers_graph.json (with uux/vuxx), physics_ruleset
    2. Discover mappings on the ID region t in [0, 1]
    3. Rediscover mappings independently on t in [0.5, 0.8], [0.5, 1.5], and [1, 2]
    4. Plot the ID mapping heatmap
  5. Save all consistencies to giit_burgers_results.json

Run from d:/MajorProject/app/:
  python burgers_temporal_experiment.py
"""

import os, sys, json, io
# Force UTF-8 output so Unicode chars in dynamic_intervener.py don't crash on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.dynamic_intervener import DynamicSciMLIntervener

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
BASE       = os.path.join(ROOT, "inbuilt_models")
MODEL_PT   = os.path.join(BASE, "burgers_1d.pt")
GRAPH_JSON = os.path.join(BASE, "burgers_graph.json")
RULESET    = os.path.join(BASE, "burgers_ruleset.json")
OUT_DIR    = "giit_burgers_results"

NU = 0.01 / np.pi

# ── Architecture (must match burgers.py / burgers_1d.pt exactly) ──────────────
class FCN(nn.Module):
    def __init__(self, N_INPUT=2, N_OUTPUT=1, N_HIDDEN=20, N_LAYERS=8):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), activation())
        self.fch = nn.Sequential(*[
            nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), activation())
            for _ in range(N_LAYERS - 1)
        ])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        return self.fce(x)


# ── Numerical reference solution ─────────────────────────────────────────────
def burgers_solution(t_np, x_np, nx=256):
    """Solve Burgers with finite differences and sample the requested points."""
    t_array, x_array = np.broadcast_arrays(
        np.asarray(t_np, dtype=float), np.asarray(x_np, dtype=float)
    )
    t_values = t_array.ravel()
    x_values = x_array.ravel()
    if np.any(t_values < 0):
        raise ValueError("Burgers reference times must be non-negative")

    x_grid = np.linspace(-1.0, 1.0, nx)
    dx = x_grid[1] - x_grid[0]
    interior_grid = x_grid[1:-1]
    initial_state = -np.sin(np.pi * interior_grid)
    requested_times = np.unique(t_values)
    max_time = float(requested_times[-1]) if requested_times.size else 0.0

    def rhs(time, state):
        full_state = np.empty(nx, dtype=float)
        full_state[0] = 0.0
        full_state[-1] = 0.0
        full_state[1:-1] = state
        second_derivative = (
            full_state[2:] - 2.0 * full_state[1:-1] + full_state[:-2]
        ) / dx**2
        flux = 0.5 * full_state**2
        wave_speed = np.maximum(np.abs(full_state[:-1]), np.abs(full_state[1:]))
        numerical_flux = 0.5 * (
            flux[:-1] + flux[1:] - wave_speed * (full_state[1:] - full_state[:-1])
        )
        advection = -(
            numerical_flux[1:] - numerical_flux[:-1]
        ) / dx
        return advection + NU * second_derivative

    if max_time > 0.0:
        solution = solve_ivp(
            rhs,
            (0.0, max_time),
            initial_state,
            t_eval=requested_times[requested_times > 0.0],
            method="BDF",
            rtol=1e-6,
            atol=1e-8,
        )
        if not solution.success:
            raise RuntimeError(f"Burgers reference solver failed: {solution.message}")
        solution_times = np.concatenate(([0.0], solution.t))
        solution_states = np.column_stack((initial_state, solution.y))
    else:
        solution_times = np.array([0.0])
        solution_states = initial_state[:, None]

    result = np.empty(t_values.size, dtype=float)
    for index, (time_value, spatial_value) in enumerate(zip(t_values, x_values)):
        time_index = int(np.searchsorted(solution_times, time_value))
        time_index = min(time_index, len(solution_times) - 1)
        state = solution_states[:, time_index]
        result[index] = np.interp(spatial_value, x_grid[1:-1], state)

    boundary_mask = np.isclose(np.abs(x_values), 1.0)
    result[boundary_mask] = 0.0
    initial_mask = np.isclose(t_values, 0.0)
    result[initial_mask] = -np.sin(np.pi * x_values[initial_mask])
    return result.reshape(t_array.shape)


def make_grid_tensor(t_min, t_max, nt=100, nx=100):
    """Build flat [N, 2] (t, x) tensor + ground truth over the requested window."""
    t_vals = np.linspace(t_min, t_max, nt)
    x_vals = np.linspace(-1.0, 1.0, nx)
    T, X = np.meshgrid(t_vals, x_vals)
    t_flat = T.ravel()
    x_flat = X.ravel()
    inp = torch.tensor(np.column_stack([t_flat, x_flat]), dtype=torch.float32)
    target = burgers_solution(t_flat, x_flat)
    if not np.isfinite(target).all():
        raise ValueError("Burgers reference solution contains NaN or Inf values")
    tgt = torch.tensor(target.reshape(-1, 1), dtype=torch.float32)
    return inp, tgt


def make_mapping_trace(t_min, t_max, nt=100, x_value=-0.5):
    """Build a single fixed-x temporal trace for observable-based mapping."""
    t_values = np.linspace(t_min, t_max, nt)
    inputs = torch.tensor(
        np.column_stack([t_values, np.full_like(t_values, x_value)]),
        dtype=torch.float32,
    )
    return inputs


# ── Heatmap helper ────────────────────────────────────────────────────────────
def plot_heatmap(sweep_data, regime_label, out_path):
    """
    Rows = physical nodes, Columns = neural layers.
    Cell = number of seeds (out of 100) for which that layer was top-mapped.
    """
    p_vars  = sweep_data["physical_variables"]
    layers  = sweep_data["all_neural_components"]
    nodes   = list(p_vars.keys())
    matrix  = np.zeros((len(nodes), len(layers)), dtype=int)

    for i, node in enumerate(nodes):
        dist = p_vars[node].get("layer_distribution", {})
        for j, layer in enumerate(layers):
            matrix[i, j] = dist.get(layer, 0)

    fig, ax = plt.subplots(figsize=(max(14, len(layers) * 0.75), max(4.5, len(nodes) * 0.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, label="# Seeds Mapped (out of 100)")

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels(nodes, fontsize=10)
    ax.set_title(f"GIIT Layer Mapping Heatmap -- {regime_label}", fontweight="bold", fontsize=13)
    ax.set_xlabel("Neural Layer", fontsize=11)
    ax.set_ylabel("Physical Node", fontsize=11)

    for i in range(len(nodes)):
        for j in range(len(layers)):
            v = matrix[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if v > 12 else "black")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    with open(out_path, "wb") as output_file:
        fig.savefig(output_file, format="png", dpi=200)
    plt.close()
    print(f"    Heatmap saved -> {out_path}")


def summarize_sweep(sweep_data):
    """Extract the modal ID mapping while retaining every seed assignment."""
    summary = {}
    for node, data in sweep_data["physical_variables"].items():
        total = sweep_data["summary"]["total_seeds"]
        summary[node] = {
            "id": data.get("id"),
            "top_layer": data.get("top_layer", "None"),
            "top_count": data.get("top_layer_count", 0),
            "total_seeds": total,
            "consistency_%": round(data.get("top_layer_count", 0) / total * 100, 1),
            "mapped_layers": list(data.get("mapped_layers", [])),
            "winner_direction_matches": list(data.get("direction_matches", [])),
            "direction_match_count": sum(
                match is True for match in data.get("direction_matches", [])
            ),
            "direction_match_total": sum(
                match is not None for match in data.get("direction_matches", [])
            ),
            "direction_match_%": (
                round(
                    sum(match is True for match in data.get("direction_matches", []))
                    / sum(match is not None for match in data.get("direction_matches", []))
                    * 100,
                    1,
                )
                if any(match is not None for match in data.get("direction_matches", []))
                else None
            ),
        }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {"experiments": {}}

    # 1. Load
    print("=" * 60)
    print("Loading Burgers PINN and GIIT configuration")
    print("=" * 60)
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    with open(RULESET) as f:
        ruleset = json.load(f)

    model = FCN(2, 1, 20, 8)
    model.load_state_dict(torch.load(MODEL_PT, map_location="cpu"))
    model.eval()

    intervener = DynamicSciMLIntervener(model, graph)
    intervener.set_physics_ruleset(ruleset)

    # ── 2. ID evaluation and mapping discovery: t in [0, 1] ───────────────────
    print("\n" + "=" * 60)
    print("IN-DISTRIBUTION DISCOVERY  (t in [0, 1])")
    print("=" * 60)

    inp_id, tgt_id = make_grid_tensor(0.0, 1.0)
    id_full = intervener.run_full_analysis(inp_id, tgt_id)

    print(f"  Predictive L2 Error   : {id_full['l2_norm'] * 100:.2f}%")
    print(f"  Physics Residual Norm : {id_full['physics_residual_norm']:.4e}")
    for sym, pdata in id_full["sensitivity"].items():
        print(f"  dR/d{sym}: {pdata['gradient']:.6f}  "
              f"(Sign: {pdata['computed_sign']}, Expected: {pdata['expected_sign']}, "
              f"Consistent: {pdata['consistent']})")

    print("\n  Running 100-seed ID mapping sweep on a fixed x trace ...")
    inp_id_mapping = make_mapping_trace(0.0, 1.0)
    sweep_id = intervener.run_physical_seed_sweep(
        inp_id_mapping, start_seed=0, end_seed=99, sigma=0.1,
        sensitivity_method="perturbation"
    )

    print("\n  ID Seed Sweep Summary:")
    id_consistency = summarize_sweep(sweep_id)
    for node, vdata in id_consistency.items():
        cnt = vdata["top_count"]
        total = vdata["total_seeds"]
        pct = vdata["consistency_%"]
        print(f"    {node:35s}: Top={vdata.get('top_layer', 'None'):<35s}  {cnt}/{total}  ({pct:.1f}%)")

    plot_heatmap(sweep_id, "ID  t=[0, 1]", os.path.join(OUT_DIR, "heatmap_id.png"))

    all_results["id"] = {
        "l2_pct":        id_full["l2_norm"] * 100,
        "residual_norm": id_full["physics_residual_norm"],
        "sensitivity": {
            sym: {
                "gradient": pdata["gradient"],
                "computed_sign": pdata["computed_sign"],
                "expected_sign": pdata["expected_sign"],
                "consistent": pdata["consistent"],
            }
            for sym, pdata in id_full["sensitivity"].items()
        },
        "consistency": id_consistency,
    }

    # ── 3. Independent mapping-discovery experiments ──────────────────────────
    experiments = {
        "t_0.5_0.8": (0.5, 0.8),
        "t_0.5_1.5": (0.5, 1.5),
        "t_1_2": (1.0, 2.0),
    }
    for experiment_name, (t_min, t_max) in experiments.items():
        print("\n" + "=" * 60)
        print(f"EVALUATION  (t in [{t_min}, {t_max}])")
        print("=" * 60)
        inputs, targets = make_grid_tensor(t_min, t_max)
        mapping_inputs = make_mapping_trace(t_min, t_max)
        full = intervener.run_full_analysis(inputs, targets)
        sweep = intervener.run_physical_seed_sweep(
            mapping_inputs, start_seed=0, end_seed=99, sigma=0.1,
            sensitivity_method="perturbation"
        )
        consistency = summarize_sweep(sweep)
        plot_heatmap(
            sweep,
            f"t=[{t_min}, {t_max}]",
            os.path.join(OUT_DIR, f"heatmap_{experiment_name}.png"),
        )
        all_results["experiments"][experiment_name] = {
            "t_min": t_min,
            "t_max": t_max,
            "l2_pct": full["l2_norm"] * 100,
            "residual_norm": full["physics_residual_norm"],
            "sensitivity": {
                sym: {
                    "gradient": pdata["gradient"],
                    "computed_sign": pdata["computed_sign"],
                    "expected_sign": pdata["expected_sign"],
                    "consistent": pdata["consistent"],
                }
                for sym, pdata in full["sensitivity"].items()
            },
            "mapping": consistency,
        }
        print(f"  Predictive L2 Error   : {full['l2_norm'] * 100:.2f}%")
        print(f"  Physics Residual Norm : {full['physics_residual_norm']:.4e}")
        print("  Mapping evaluation    : independently rediscovered on this interval")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    out_json = os.path.join(OUT_DIR, "burgers_giit_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  All results saved -> {out_json}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"  {'Metric':<35} {'ID (t=0-1)':>14} {'t=[0.5,0.8]':>14} {'t=[0.5,1.5]':>14} {'t=[1,2]':>14}")
    print(f"  {'-' * 70}")
    print(f"  {'L2 Error (%)':<35} {id_full['l2_norm']*100:>14.2f} "
          f"{all_results['experiments']['t_0.5_0.8']['l2_pct']:>14.2f} "
          f"{all_results['experiments']['t_0.5_1.5']['l2_pct']:>14.2f} "
          f"{all_results['experiments']['t_1_2']['l2_pct']:>14.2f}")
    print(f"  {'Physics Residual Norm':<35} {id_full['physics_residual_norm']:>14.4e} "
          f"{all_results['experiments']['t_0.5_0.8']['residual_norm']:>14.4e} "
          f"{all_results['experiments']['t_0.5_1.5']['residual_norm']:>14.4e} "
          f"{all_results['experiments']['t_1_2']['residual_norm']:>14.4e}")
    print(f"  {'Mapping protocol':<35} {'Discovery':>14} {'Rediscovery':>14} {'Rediscovery':>14} {'Rediscovery':>14}")
    print("=" * 60)
    print(f"\n  Outputs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
