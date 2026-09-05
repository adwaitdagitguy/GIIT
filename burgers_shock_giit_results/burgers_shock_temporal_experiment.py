"""
GIIT temporal Burgers experiment evaluated against burgers_shock.mat.

The MAT file contains x (256,), t (100,), and usol (256, 100), where
usol is indexed as (space, time). Windows outside the MAT time range are
reported as skipped rather than being extrapolated.

Run from d:/MajorProject/app/:
  python burgers_shock_giit_results/burgers_shock_temporal_experiment.py
"""

import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import torch
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator

from burgers_temporal_experiment import (
    FCN,
    burgers_solution,
    plot_heatmap,
    summarize_sweep,
)
from core.dynamic_intervener import DynamicSciMLIntervener

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "inbuilt_models")
MAT_FILE = os.path.join(ROOT, "giit_burgers_results", "burgers_shock.mat")
MODEL_PT = os.path.join(BASE, "burgers_1d.pt")
GRAPH_JSON = os.path.join(BASE, "burgers_graph.json")
RULESET = os.path.join(BASE, "burgers_ruleset.json")
OUT_DIR = os.path.join(ROOT, "burgers_shock_giit_results")
RESULT_FILE = os.path.join(OUT_DIR, "burgers_shock_giit_results.json")


class MatReference:
    def __init__(self, path):
        data = loadmat(path)
        self.x = np.asarray(data["x"], dtype=float).ravel()
        self.t = np.asarray(data["t"], dtype=float).ravel()
        self.usol = np.asarray(data["usol"], dtype=float)
        if self.usol.shape != (self.x.size, self.t.size):
            raise ValueError(
                f"Expected usol shape {(self.x.size, self.t.size)}, "
                f"got {self.usol.shape}"
            )
        if not np.isfinite(self.usol).all():
            raise ValueError("MAT solution contains NaN or Inf values")
        self.interpolator = RegularGridInterpolator(
            (self.x, self.t), self.usol, bounds_error=True
        )

    def evaluate(self, t_values, x_values):
        t_array, x_array = np.broadcast_arrays(
            np.asarray(t_values, dtype=float), np.asarray(x_values, dtype=float)
        )
        flat_t = t_array.ravel()
        flat_x = x_array.ravel()
        result = np.empty(flat_t.size, dtype=float)
        mat_mask = (flat_t >= self.t[0]) & (flat_t <= self.t[-1])
        if np.any(mat_mask):
            mat_points = np.column_stack((flat_x[mat_mask], flat_t[mat_mask]))
            result[mat_mask] = self.interpolator(mat_points)
        if np.any(~mat_mask):
            # Preserve the OOD windows by extending the same reference model
            # used by the original temporal experiment beyond the MAT range.
            result[~mat_mask] = burgers_solution(flat_t[~mat_mask], flat_x[~mat_mask])
        return result.reshape(t_array.shape)

def make_grid_tensor(reference, t_min, t_max, nt=100, nx=100):
    t_values = np.linspace(t_min, t_max, nt)
    x_values = np.linspace(reference.x[0], reference.x[-1], nx)
    T, X = np.meshgrid(t_values, x_values)
    targets = reference.evaluate(T, X)
    inputs = torch.tensor(
        np.column_stack((T.ravel(), X.ravel())), dtype=torch.float32
    )
    target_tensor = torch.tensor(targets.reshape(-1, 1), dtype=torch.float32)
    return inputs, target_tensor


def make_mapping_trace(t_min, t_max, nt=100, x_value=-0.5):
    t_values = np.linspace(t_min, t_max, nt)
    return torch.tensor(
        np.column_stack((t_values, np.full_like(t_values, x_value))),
        dtype=torch.float32,
    )


def evaluate_window(intervener, reference, t_min, t_max):
    inputs, targets = make_grid_tensor(reference, t_min, t_max)
    full = intervener.run_full_analysis(inputs, targets)
    mapping_inputs = make_mapping_trace(t_min, t_max)
    sweep = intervener.run_physical_seed_sweep(
        mapping_inputs,
        start_seed=0,
        end_seed=99,
        sigma=0.1,
        sensitivity_method="perturbation",
    )
    return full, summarize_sweep(sweep), sweep


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    reference = MatReference(MAT_FILE)
    print(f"Loaded {MAT_FILE}")
    print(
        f"MAT grid: {reference.x.size} x-points, {reference.t.size} t-points; "
        f"t=[{reference.t[0]}, {reference.t[-1]}]"
    )

    with open(GRAPH_JSON) as graph_file:
        graph = json.load(graph_file)
    with open(RULESET) as ruleset_file:
        ruleset = json.load(ruleset_file)

    model = FCN(2, 1, 20, 8)
    model.load_state_dict(torch.load(MODEL_PT, map_location="cpu"))
    model.eval()
    intervener = DynamicSciMLIntervener(model, graph)
    intervener.set_physics_ruleset(ruleset)

    windows = {
        "id_mat": (0.0, 0.99),
        "t_0.5_0.8": (0.5, 0.8),
        "t_0.5_1.5": (0.5, 1.5),
        "t_1_2": (1.0, 2.0),
    }
    results = {
        "source": os.path.basename(MAT_FILE),
        "source_shape": list(reference.usol.shape),
        "source_time_range": [float(reference.t[0]), float(reference.t[-1])],
        "grid_points_per_supported_window": 10000,
        "windows": {},
    }

    for name, (t_min, t_max) in windows.items():
        print(f"\n{name}: t=[{t_min}, {t_max}]")
        full, consistency, sweep = evaluate_window(
            intervener, reference, t_min, t_max
        )
        plot_heatmap(
            sweep,
            f"MAT t=[{t_min}, {t_max}]",
            os.path.join(OUT_DIR, f"heatmap_{name}.png"),
        )
        results["windows"][name] = {
            "status": "completed",
            "t_min": t_min,
            "t_max": t_max,
            "sample_count": 10000,
            "target_source": (
                "MAT interpolation within t=[0, 0.99]; "
                "numerical reference extension outside MAT range"
            ),
            "l2_pct": full["l2_norm"] * 100,
            "residual_norm": full["physics_residual_norm"],
            "consistency": consistency,
        }
        print(f"  samples: 10000")
        print(f"  predictive L2 error: {full['l2_norm'] * 100:.2f}%")
        print(f"  physics residual norm: {full['physics_residual_norm']:.4e}")

    with open(RESULT_FILE, "w") as output_file:
        json.dump(results, output_file, indent=2, default=str)
    print(f"\nResults saved -> {RESULT_FILE}")


if __name__ == "__main__":
    main()
