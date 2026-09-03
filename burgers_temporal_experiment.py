"""
burgers_temporal_experiment.py
==============================

Burgers PINN temporal OOD experiment using the GIIT framework.

Follows the same structure as reproduce_all_paper_experiments.py but for the
1D Viscous Burgers equation, with temporal extrapolation across t in [0.5, 1.5].

Steps:
  1. Load existing burgers_1d.pt, burgers_graph.json (with uux/vuxx), physics_ruleset
  2. ID region:  t in [0, 1]     -- run_full_analysis + 20-seed run_physical_seed_sweep
  3. OOD region: t in [0.5, 1.5] -- run_full_analysis + 20-seed run_physical_seed_sweep
  4. Plot heatmaps (physical node x neural layer, coloured by seed count)
  5. Save all consistencies to giit_burgers_results.json

Run from d:/MajorProject/app/:
  python burgers_temporal_experiment.py
"""

import os, sys, json, io
# Force UTF-8 output so Unicode chars in dynamic_intervener.py don't crash on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, r"d:\MajorProject\app")
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.dynamic_intervener import DynamicSciMLIntervener

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = r"d:\MajorProject\app\inbuilt_models"
MODEL_PT   = os.path.join(BASE, "burgers_1d.pt")
GRAPH_JSON = os.path.join(BASE, "burgers_graph.json")
RULESET    = r"d:\Downloads\physics_ruleset.json"
OUT_DIR    = r"d:\MajorProject\app\giit_burgers_results"

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


# ── Analytical ground truth (linearised low-nu approximation) ─────────────────
def burgers_approx(t_np, x_np):
    return -np.sin(np.pi * x_np) * np.exp(-NU * np.pi**2 * t_np)


def make_grid_tensor(t_min, t_max, nt=100, nx=100):
    """Build flat [N, 2] (t, x) tensor + ground truth over the requested window."""
    t_vals = np.linspace(t_min, t_max, nt)
    x_vals = np.linspace(-1.0, 1.0, nx)
    T, X = np.meshgrid(t_vals, x_vals)
    t_flat = T.ravel()
    x_flat = X.ravel()
    inp = torch.tensor(np.column_stack([t_flat, x_flat]), dtype=torch.float32)
    tgt = torch.tensor(burgers_approx(t_flat, x_flat).reshape(-1, 1), dtype=torch.float32)
    return inp, tgt


# ── Heatmap helper ────────────────────────────────────────────────────────────
def plot_heatmap(sweep_data, regime_label, out_path):
    """
    Rows = physical nodes, Columns = neural layers.
    Cell = number of seeds (out of 20) for which that layer was top-mapped.
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
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=20)
    fig.colorbar(im, ax=ax, label="# Seeds Mapped (out of 20)")

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
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"    Heatmap saved -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}

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

    # ── 2. ID: t in [0, 1] ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("IN-DISTRIBUTION  (t in [0, 1])")
    print("=" * 60)

    inp_id, tgt_id = make_grid_tensor(0.0, 1.0)
    id_full = intervener.run_full_analysis(inp_id, tgt_id)

    print(f"  Predictive L2 Error   : {id_full['l2_norm'] * 100:.2f}%")
    print(f"  Physics Residual Norm : {id_full['physics_residual_norm']:.4e}")
    for sym, pdata in id_full["sensitivity"].items():
        print(f"  dR/d{sym}: {pdata['gradient']:.6f}  "
              f"(Sign: {pdata['computed_sign']}, Expected: {pdata['expected_sign']}, "
              f"Consistent: {pdata['consistent']})")

    print("\n  Running 20-seed sweep  (seeds 0-19, sigma=0.1, perturbation) ...")
    sweep_id = intervener.run_physical_seed_sweep(
        inp_id, start_seed=0, end_seed=19, sigma=0.1, sensitivity_method="perturbation"
    )

    print("\n  ID Seed Sweep Summary:")
    id_consistency = {}
    for node, vdata in sweep_id["physical_variables"].items():
        cnt   = vdata.get("top_layer_count", 0)
        total = sweep_id["summary"]["total_seeds"]
        pct   = cnt / total * 100
        print(f"    {node:35s}: Top={vdata.get('top_layer', 'None'):<35s}  {cnt}/{total}  ({pct:.1f}%)")
        id_consistency[node] = {
            "top_layer":     vdata.get("top_layer"),
            "top_count":     cnt,
            "total_seeds":   total,
            "consistency_%": round(pct, 1),
        }

    plot_heatmap(sweep_id, "ID  t=[0, 1]", os.path.join(OUT_DIR, "heatmap_id.png"))

    all_results["id"] = {
        "l2_pct":        id_full["l2_norm"] * 100,
        "residual_norm": id_full["physics_residual_norm"],
        "sensitivity":   {
            sym: {
                "gradient":      pdata["gradient"],
                "computed_sign": pdata["computed_sign"],
                "expected_sign": pdata["expected_sign"],
                "consistent":    pdata["consistent"],
            }
            for sym, pdata in id_full["sensitivity"].items()
        },
        "consistency": id_consistency,
    }

    # ── 3. OOD: t in [0.5, 1.5] ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OUT-OF-DISTRIBUTION  (t in [0.5, 1.5])")
    print("=" * 60)

    inp_ood, tgt_ood = make_grid_tensor(0.5, 1.5)
    ood_full = intervener.run_full_analysis(inp_ood, tgt_ood)

    print(f"  Predictive L2 Error   : {ood_full['l2_norm'] * 100:.2f}%")
    print(f"  Physics Residual Norm : {ood_full['physics_residual_norm']:.4e}")
    for sym, pdata in ood_full["sensitivity"].items():
        print(f"  dR/d{sym}: {pdata['gradient']:.6f}  "
              f"(Sign: {pdata['computed_sign']}, Expected: {pdata['expected_sign']}, "
              f"Consistent: {pdata['consistent']})")

    print("\n  Running 20-seed sweep  (seeds 0-19, sigma=0.1, perturbation) ...")
    sweep_ood = intervener.run_physical_seed_sweep(
        inp_ood, start_seed=0, end_seed=19, sigma=0.1, sensitivity_method="perturbation"
    )

    print("\n  OOD Seed Sweep Summary:")
    ood_consistency = {}
    for node, vdata in sweep_ood["physical_variables"].items():
        cnt   = vdata.get("top_layer_count", 0)
        total = sweep_ood["summary"]["total_seeds"]
        pct   = cnt / total * 100
        print(f"    {node:35s}: Top={vdata.get('top_layer', 'None'):<35s}  {cnt}/{total}  ({pct:.1f}%)")
        ood_consistency[node] = {
            "top_layer":     vdata.get("top_layer"),
            "top_count":     cnt,
            "total_seeds":   total,
            "consistency_%": round(pct, 1),
        }

    plot_heatmap(sweep_ood, "OOD  t=[0.5, 1.5]", os.path.join(OUT_DIR, "heatmap_ood.png"))

    all_results["ood"] = {
        "l2_pct":        ood_full["l2_norm"] * 100,
        "residual_norm": ood_full["physics_residual_norm"],
        "sensitivity":   {
            sym: {
                "gradient":      pdata["gradient"],
                "computed_sign": pdata["computed_sign"],
                "expected_sign": pdata["expected_sign"],
                "consistent":    pdata["consistent"],
            }
            for sym, pdata in ood_full["sensitivity"].items()
        },
        "consistency": ood_consistency,
    }

    # ── 4. Stability comparison ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MAPPING STABILITY: ID vs OOD")
    print("=" * 60)

    all_nodes = sorted(set(id_consistency) | set(ood_consistency))
    stability = {}
    for node in all_nodes:
        id_top  = id_consistency.get(node,  {}).get("top_layer", "None")
        ood_top = ood_consistency.get(node, {}).get("top_layer", "None")
        id_pct  = id_consistency.get(node,  {}).get("consistency_%", 0.0)
        ood_pct = ood_consistency.get(node, {}).get("consistency_%", 0.0)
        stable  = (id_top == ood_top)
        stability[node] = {
            "id_top": id_top, "ood_top": ood_top,
            "id_pct": id_pct, "ood_pct": ood_pct,
            "stable": stable,
        }
        marker = "STABLE" if stable else "CHANGED"
        print(f"  {node:35s}  ID: {id_top:<30s}({id_pct:.1f}%)  "
              f"OOD: {ood_top:<30s}({ood_pct:.1f}%)  [{marker}]")

    n_stable = sum(1 for v in stability.values() if v["stable"])
    print(f"\n  {n_stable}/{len(all_nodes)} nodes maintain a stable mapping across the temporal shift.")
    all_results["stability"] = stability

    # ── 5. Save ───────────────────────────────────────────────────────────────
    out_json = os.path.join(OUT_DIR, "burgers_giit_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  All results saved -> {out_json}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"  {'Metric':<35} {'ID (t=0-1)':>14} {'OOD (t=0.5-1.5)':>18}")
    print(f"  {'-' * 70}")
    print(f"  {'L2 Error (%)':<35} {id_full['l2_norm']*100:>14.2f} {ood_full['l2_norm']*100:>18.2f}")
    print(f"  {'Physics Residual Norm':<35} {id_full['physics_residual_norm']:>14.4e} {ood_full['physics_residual_norm']:>18.4e}")
    print(f"  {'Stable mappings':<35} {'N/A':>14} {n_stable}/{len(all_nodes):>14}")
    print("=" * 60)
    print(f"\n  Outputs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
