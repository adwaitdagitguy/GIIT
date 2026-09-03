"""
Reproducible Transfer & Mechanistic Evaluation Script for:
'A Graph-Based Inspection and Intervention Tool for Validating Mechanistic Learning in PINNs'

Outputs exact quantitative metrics reported in Table 1, Section 5, and Appendix:
- In-Distribution (t in [0, 1]) L2 error, residual norm, and sensitivity signs
- Out-of-Distribution (t in [1, 2]) L2 error, residual norm, and sign inversions
- Seed sweep distribution across perturbation noise draws
"""

import os, sys, json
sys.path.insert(0, r"d:\MajorProject\app")
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import numpy as np
import torch.nn as nn
from core.dynamic_intervener import DynamicSciMLIntervener

class FCN(nn.Module):
    def __init__(self, N_INPUT=1, N_OUTPUT=1, N_HIDDEN=32, N_LAYERS=3):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(nn.Linear(N_INPUT, N_HIDDEN), activation())
        self.fch = nn.Sequential(*[nn.Sequential(nn.Linear(N_HIDDEN, N_HIDDEN), activation()) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        return self.fce(self.fch(self.fcs(x)))

def analytical_oscillator(d=2.0, w0=20.0, t=None):
    """Ground truth damped harmonic oscillator solution."""
    w = np.sqrt(w0**2 - d**2)
    phi = np.arctan(-d/w)
    A = 1 / (2 * np.cos(phi))
    cos = np.cos(phi + w * t)
    exp = np.exp(-d * t)
    return exp * 2 * A * cos

def main():
    # 1. Load Causal Graph, Ruleset, and Trained Model
    with open(r'd:\MajorProject\app\inbuilt_models\damped_oscillator_graph.json') as f:
        graph = json.load(f)
    with open(r'd:\MajorProject\app\inbuilt_models\physics_ruleset.json') as f:
        ruleset = json.load(f)

    model = FCN(1, 1, 32, 3)
    model.load_state_dict(torch.load(r'd:\MajorProject\app\inbuilt_models\damped_oscillator.pt', map_location='cpu'))
    model.eval()

    intervener = DynamicSciMLIntervener(model, graph)
    intervener.set_physics_ruleset(ruleset)

    # 2. In-Distribution Domain (t in [0, 1])
    t_id_np = np.linspace(0, 1, 500)
    y_id_np = analytical_oscillator(2.0, 20.0, t_id_np)
    t_id = torch.tensor(t_id_np, dtype=torch.float32).view(-1, 1)
    y_id = torch.tensor(y_id_np, dtype=torch.float32).view(-1, 1)

    id_res = intervener.run_full_analysis(t_id, y_id)
    print("=" * 60)
    print("IN-DISTRIBUTION RESULTS (t in [0, 1]):")
    print(f"  Predictive L2 Error: {id_res['l2_norm'] * 100:.2f}%")
    print(f"  Physics Residual Norm: {id_res['physics_residual_norm']:.4e}")
    for sym, pdata in id_res['sensitivity'].items():
        print(f"  Sensitivity dR/d{sym}: {pdata['gradient']:.4f} (Sign: {pdata['computed_sign']}, Expected: {pdata['expected_sign']})")

    # 3. Out-of-Distribution Domain (t in [1, 2])
    t_ood_np = np.linspace(1.0, 2.0, 500)
    y_ood_np = analytical_oscillator(2.0, 20.0, t_ood_np)
    t_ood = torch.tensor(t_ood_np, dtype=torch.float32).view(-1, 1)
    y_ood = torch.tensor(y_ood_np, dtype=torch.float32).view(-1, 1)

    ood_res = intervener.run_full_analysis(t_ood, y_ood)
    print("\n" + "=" * 60)
    print("OUT-OF-DISTRIBUTION RESULTS (t in [1, 2]):")
    print(f"  Predictive L2 Error: {ood_res['l2_norm'] * 100:.2f}%")
    print(f"  Physics Residual Norm: {ood_res['physics_residual_norm']:.4e}")
    for sym, pdata in ood_res['sensitivity'].items():
        print(f"  Sensitivity dR/d{sym}: {pdata['gradient']:.4f} (Sign: {pdata['computed_sign']}, Expected: {pdata['expected_sign']})")

    # 4. Multi-Seed Stability Sweep across 21 Seeds
    print("\n" + "=" * 60)
    print("RUNNING 21-SEED SWEEP (seeds 0-20, sigma=0.1)...")
    sweep_id = intervener.run_physical_seed_sweep(t_id, start_seed=0, end_seed=20, sigma=0.1, sensitivity_method='perturbation')
    sweep_ood = intervener.run_physical_seed_sweep(t_ood, start_seed=0, end_seed=20, sigma=0.1, sensitivity_method='perturbation')

    print("\nID Seed Sweep Summary:")
    for k, v in sweep_id['physical_variables'].items():
        cnt = v.get('top_layer_count', 0)
        print(f"  {k:20s}: Top={v.get('top_layer')} ({cnt}/21 = {cnt/21*100:.1f}%)")

    print("\nOOD Seed Sweep Summary:")
    for k, v in sweep_ood['physical_variables'].items():
        cnt = v.get('top_layer_count', 0)
        print(f"  {k:20s}: Top={v.get('top_layer')} ({cnt}/21 = {cnt/21*100:.1f}%)")
    print("=" * 60)

if __name__ == '__main__':
    main()
