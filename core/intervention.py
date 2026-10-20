import torch
import torch.nn as nn
import json
import copy
import os

def _gradients(y, x):
    """Compute gradients of y with respect to x."""
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]

def _load_model(model_path, model_class=None):
    """Safely load a model from a .pt file."""
    device = torch.device('cpu')
    if model_class is not None:
        model = model_class()
        # Ensure correct argument: map_location
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    # Ensure correct argument: map_location
    return torch.load(model_path, map_location=device)

class SciMLIntervener:
    """
    Reads physics JSON, loads the PINN, and provides:
      - Relative L2 Norm                     (MT3 Task 3a)
      - Physics Residual Norm                (MT3 Task 3b)
      - Sensitivity  d(meanR2)/d{m,c,k}     (MT3 Task 2)

    Phase 2 capabilities (Major_tasks_4 + Major_Tasks_5):
      - get_perturbable_tensors()            (MT4 Task 1)
      - run_systematic_perturbation()        (MT4 Task 2)
      - auto_map_nodes()                     (MT5 Task 1)
      - generate_mapping_with_confidence()   (MT5 Task 2)
    """

    # ── MT3 Task 1: Init & graph loading ──────────────────────────────────────
    def __init__(self, model_path: str, graph_path: str, model_class=None):
        print("\n" + "="*60)
        print("SciMLIntervener — INITIALIZATION")
        print("="*60)

        print(f"\n[1] Loading PINN from: {model_path}")
        self.model = _load_model(model_path, model_class)
        self.model.eval()
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"    Parameters   : {num_params}")

        print(f"\n[2] Loading physics graph from: {graph_path}")
        with open(graph_path, "r") as f:
            self.graph_data = json.load(f)
        self._validate_and_print_graph()

        # Backup weights (used by reset_model and systematic perturbation)
        self.original_state = copy.deepcopy(self.model.state_dict())

        # Populated by run_systematic_perturbation / auto_map_nodes
        self._perturbation_results: dict = {}
        self._mapping: dict = {}

        print("\nINITIALIZATION COMPLETE")
        print("="*60)

    def _validate_and_print_graph(self):
        required = {"nodes", "edges"}
        missing  = required - set(self.graph_data.keys())
        if missing:
            raise ValueError(f"Physics graph JSON missing keys: {missing}")

        nodes = self.graph_data["nodes"]
        edges = self.graph_data["edges"]

        print(f"\n    Physics Nodes ({len(nodes)}):")
        for n in nodes:
            if isinstance(n, dict):
                label = n.get("label", n.get("name", n.get("id", "?")))
                symbol = n.get("symbol", "")
            else:
                label = str(n)
                symbol = ""
            print(f"      * {label}  (symbol={symbol})")

        print(f"\n    Causal Edges ({len(edges)}):")
        for e in edges:
            if isinstance(e, dict):
                src = e.get("source", e.get("from", "?"))
                tgt = e.get("target", e.get("to", "?"))
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                src = e[0]
                tgt = e[1]
            else:
                src, tgt = "?", "?"
            print(f"      {src}  ->  {tgt}")

    # ── MT3 Task 3a: Relative L2 Norm ─────────────────────────────────────────
    def compute_l2_norm(
        self,
        t: torch.Tensor,
        x_true: torch.Tensor
    ) -> float:
        """
        Relative L2 Norm = ||x_pred - x_true||_2 / ||x_true||_2
        Success criterion: <= 0.05 (5%)
        """
        with torch.no_grad():
            x_pred = self.model(t)
            l2 = (torch.norm(x_pred - x_true) / torch.norm(x_true)).item()
        return l2

    # ── MT3 Task 3b: Physics Residual Norm ────────────────────────────────────
    def compute_physics_residual_norm(
        self,
        t: torch.Tensor,
        m: float,
        c: float,
        k: float
    ) -> float:
        """
        ODE residual: R = m*x'' + c*x' + k*x  (should be ~0)
        Returns mean(R^2).
        """
        t_rg = t.detach().clone().requires_grad_(True)
        x    = self.model(t_rg)
        x_t  = _gradients(x,   t_rg)
        x_tt = _gradients(x_t, t_rg)

        residual      = m * x_tt + c * x_t + k * x
        residual_norm = torch.mean(residual ** 2).item()
        return residual_norm

    # ── MT3 Task 2: Sensitivity via autograd ──────────────────────────────────
    def compute_sensitivity(
        self,
        t: torch.Tensor,
        m: float,
        c: float,
        k: float
    ) -> dict:
        """
        Compute d(mean R^2)/dp for each physical parameter p in {m, c, k}.
        Also checks computed sign vs. analytical expectation.
        """
        t_rg = t.detach().clone().requires_grad_(True)
        x    = self.model(t_rg)
        x_t  = _gradients(x,   t_rg)
        x_tt = _gradients(x_t, t_rg)

        m_t = torch.tensor(float(m), requires_grad=True)
        c_t = torch.tensor(float(c), requires_grad=True)
        k_t = torch.tensor(float(k), requires_grad=True)

        residual = m_t * x_tt + c_t * x_t + k_t * x
        loss_r   = torch.mean(residual ** 2)

        grads = torch.autograd.grad(
            loss_r, [m_t, c_t, k_t],
            create_graph=False,
            allow_unused=True
        )

        # Analytical expected signs
        expected = {
            "m": {
                "sign"  : "+",
                "trend" : "Increasing mass slows decay; amplitude persists longer (under-damped regime)"
            },
            "c": {
                "sign"  : "-",
                "trend" : "Increasing damping accelerates amplitude decay; system loses energy faster"
            },
            "k": {
                "sign"  : "+",
                "trend" : "Increasing spring constant raises natural frequency; minor effect on amplitude"
            },
        }

        params_info = [
            ("m", "Mass",            m, grads[0]),
            ("c", "Damping",         c, grads[1]),
            ("k", "Spring Constant", k, grads[2]),
        ]

        results = {}
        for sym, label, val, grad_t in params_info:
            g_val         = grad_t.item() if grad_t is not None else 0.0
            computed_sign = "+" if g_val > 1e-8 else ("-" if g_val < -1e-8 else "~0")
            exp           = expected[sym]
            consistent    = (exp["sign"] == "~") or (computed_sign == exp["sign"])

            results[sym] = {
                "label"         : label,
                "symbol"        : sym,
                "value"         : val,
                "gradient"      : g_val,
                "computed_sign" : computed_sign,
                "expected_sign" : exp["sign"],
                "consistent"    : consistent,
                "trend"         : exp["trend"],
            }

        return results

    # ── MT3 Combined report ───────────────────────────────────────────────────
    def run_full_analysis(
        self,
        t      : torch.Tensor,
        x_true : torch.Tensor,
        m      : float,
        c      : float,
        k      : float
    ) -> dict:
        """Single call: L2 norm + physics residual + sensitivity."""
        print("\n" + "="*60)
        print("FULL ANALYSIS REPORT")
        print("="*60)

        print("\n[A] RELATIVE L2 NORM  (prediction accuracy)")
        l2      = self.compute_l2_norm(t, x_true)
        l2_pass = l2 <= 0.05
        status  = "PASS" if l2_pass else "FAIL"
        print(f"    Relative L2 : {l2:.6f}  ({l2*100:.3f}%)  [{status}]  [threshold <= 5%]")

        print("\n[B] PHYSICS RESIDUAL NORM  (ODE: m*x'' + c*x' + k*x = 0)")
        res_norm = self.compute_physics_residual_norm(t, m, c, k)
        print(f"    mean(R^2)   : {res_norm:.6e}")

        print("\n[C] SENSITIVITY  (d[mean R^2]/dp  via autograd)")
        sensitivity = self.compute_sensitivity(t, m, c, k)
        print(f"\n    {'Parameter':<18} {'Val':>5}  {'dR/dp':>13}  {'Sign':>5}  {'Exp':>5}  {'OK?'}")
        print(f"    {'-'*65}")
        for sym, info in sensitivity.items():
            ok = "[OK]" if info["consistent"] else "[!!]"
            print(
                f"    {info['label']:<18} {info['value']:>5.2f}  "
                f"{info['gradient']:>13.6f}  "
                f"{info['computed_sign']:>5}  "
                f"{info['expected_sign']:>5}  {ok}"
            )
            print(f"      -> {info['trend']}")

        print("\n" + "-"*60)
        print("SUMMARY")
        print(f"  L2 Norm (relative)   : {l2*100:.3f}%   {'[PASS <=5%]' if l2_pass else '[FAIL >5%]'}")
        print(f"  Physics Residual Norm: {res_norm:.6e}")
        for sym, info in sensitivity.items():
            ok = "consistent" if info["consistent"] else "INCONSISTENT [!!]"
            print(f"  dR/d{sym}  sign={info['computed_sign']}  expected={info['expected_sign']}  -> {ok}")
        print("="*60)

        return {
            "l2_norm"              : l2,
            "l2_pass"              : l2_pass,
            "physics_residual_norm": res_norm,
            "sensitivity"          : sensitivity
        }

    def reset_model(self):
        """Restore original weights (call between perturbation tests)."""
        self.model.load_state_dict(self.original_state)

    def compute_physics_residual(self, t, x_pred):
        """
        Compute physics residual for projectile motion.
        Note: You can generalize this by passing a domain-specific residual function.
        """
        # For generalization, we assume projectile motion if not specified
        m, g = 1.0, 9.81
        t_rg = t.detach().clone().requires_grad_(True)
        x    = self.model(t_rg)
        
        x_t  = _gradients(x,   t_rg)
        x_tt = _gradients(x_t, t_rg)
        
        # ODE: x'' + g = 0 (simplified)
        residual = x_tt + g
        return torch.mean(residual**2).item()

    def compute_input_gradients(self, t):
        """Compute dOutput/dInput for sensitivity analysis."""
        t_rg = t.detach().clone().requires_grad_(True)
        # We assume input is [t, v0, g, theta] based on PhysicsMLP
        y = self.model(t_rg)
        
        grads = torch.autograd.grad(y.sum(), t_rg)[0]
        mean_grads = grads.mean(dim=0)
        
        return {
            "dOutput_dTime":     mean_grads[0].item() if len(mean_grads) > 0 else 0,
            "dOutput_dVelocity": mean_grads[1].item() if len(mean_grads) > 1 else 0,
            "dOutput_dGravity":  mean_grads[2].item() if len(mean_grads) > 2 else 0,
            "dOutput_dAngle":    mean_grads[3].item() if len(mean_grads) > 3 else 0,
        }

    def get_plot_data(self, node_name, intervention_type, strength, test_inputs, test_targets):
        """
        Runs intervention on a specific node and returns data for plotting.
        """
        # 1. Baseline
        self.model.eval()
        with torch.no_grad():
            y_base = self.model(test_inputs)
            baseline_metric = torch.mean((y_base - test_targets)**2).item()

        # 2. Apply Intervention
        # (This is a simplified version; you can expand with specific layer masking)
        original_state = copy.deepcopy(self.model.state_dict())
        
        # For now, we simulate intervention by perturbing the whole model 
        # or specific layers if the mapping exists
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if intervention_type == "mask":
                    param.mul_(1.0 - strength)
                elif intervention_type == "perturb":
                    param.add_(torch.randn_like(param) * strength)
        
        with torch.no_grad():
            y_pert = self.model(test_inputs)
            intervened_metric = torch.mean((y_pert - test_targets)**2).item()

        # Restore original weights
        self.model.load_state_dict(original_state)

        return {
            "x_values":           test_inputs[:, 0].numpy(), # Assuming time is first column
            "y_preds_baseline":   y_base.squeeze().numpy(),
            "y_preds_intervened": y_pert.squeeze().numpy(),
            "baseline_metric":    baseline_metric,
            "intervened_metric":  intervened_metric
        }


    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2  —  Major_tasks_4.pdf  Backend Tasks
    # ══════════════════════════════════════════════════════════════════════════

    # ── MT4 Task 1: Build the Target Inventory (Introspection) ────────────────
    def get_perturbable_tensors(self) -> list:
        """
        Iterate through the model's state_dict and return a flat list of
        every manipulatable tensor name (all weights and biases).

        Returns:
            List of layer name strings, e.g.
            ["net.0.weight", "net.0.bias", "net.2.weight", ...]
        """
        inventory = list(self.model.state_dict().keys())

        print("\n" + "="*60)
        print("TARGET INVENTORY  (get_perturbable_tensors)")
        print("="*60)
        for i, name in enumerate(inventory):
            shape = self.model.state_dict()[name].shape
            print(f"  [{i:02d}] {name:<35}  shape={tuple(shape)}")
        print(f"\n  Total tensors: {len(inventory)}")
        print("="*60)

        return inventory

    # ── MT4 Task 2: Systematic "Ping" Test ────────────────────────────────────
    def run_systematic_perturbation(
        self,
        t      : torch.Tensor,
        sigma  : float = 0.1,
        verbose: bool  = True
    ) -> dict:
        """
        Iterate through every tensor in the inventory. For each tensor:
          1. Inject standardised Gaussian noise (std=sigma, approx +10%).
          2. Run a forward pass and record the mean absolute change delta_y.
          3. Restore original weights for the next clean test.

        Args:
            t      : input tensor for the forward pass (shape [N, input_dim])
            sigma  : noise std deviation (default 0.1)
            verbose: print per-layer results

        Returns:
            dict { layer_name : {"delta_y": float, "shape": tuple} }
            Sorted by delta_y descending (highest impact first).
        """
        inventory = self.get_perturbable_tensors()

        if verbose:
            print("\n" + "="*60)
            print(f"SYSTEMATIC PERTURBATION  (sigma={sigma})")
            print("="*60)
            print(f"  {'Layer':<35}  {'delta_y (mean |delta_output|)':>30}")
            print(f"  {'-'*67}")

        self.model.eval()
        with torch.no_grad():
            y_base = self.model(t)

        results = {}

        for name in inventory:
            state           = self.model.state_dict()
            original_tensor = state[name].clone()

            # 1. Perturb
            noise       = torch.randn_like(original_tensor) * sigma
            state[name] = original_tensor + noise
            self.model.load_state_dict(state)

            # 2. Forward pass -> record delta_y
            with torch.no_grad():
                y_pert = self.model(t)
            delta_y = (y_pert - y_base).abs().mean().item()

            # 3. Restore
            state[name] = original_tensor
            self.model.load_state_dict(state)

            results[name] = {
                "delta_y": delta_y,
                "shape"  : tuple(original_tensor.shape)
            }

            if verbose:
                print(f"  {name:<35}  delta_y = {delta_y:.6f}")

        # Sort by impact (highest first)
        sorted_results = dict(
            sorted(results.items(), key=lambda x: x[1]["delta_y"], reverse=True)
        )

        if verbose:
            print(f"\n  Top 3 most influential tensors:")
            for i, (name, info) in enumerate(sorted_results.items()):
                if i >= 3:
                    break
                print(f"    {i+1}. {name:<35}  delta_y = {info['delta_y']:.6f}")
            print("="*60)

        self._perturbation_results = sorted_results
        return sorted_results


    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2  —  Major_Tasks_5.PDF  Backend Tasks
    # ══════════════════════════════════════════════════════════════════════════

    # ── MT5 Task 1: Heuristic Matching ────────────────────────────────────────
    def auto_map_nodes(
        self,
        expected_trends : dict,
        t               : torch.Tensor,
        sigma           : float = 0.1,
    ) -> dict:
        """
        Automatically discover which PyTorch layer corresponds to which
        physical causal node by comparing perturbation effects against
        the analytical physics rules.

        Args:
            expected_trends : dict of expected trends, e.g.
                              {"Friction"     : "decreases_amplitude",
                               "Mass"         : "increases_amplitude",
                               "Spring"       : "increases_frequency"}
            t               : input tensor  (shape [N, input_dim])
            sigma           : noise std for perturbation

        Supported trend values (case-insensitive):
            "decreases_amplitude"  — perturbation should shrink output range
            "increases_amplitude"  — perturbation should grow output range
            "increases_frequency"  — perturbation should increase zero crossings
            "decreases_frequency"  — perturbation should decrease zero crossings

        Returns:
            dict { physical_node_name : {"layer"      : str,
                                         "confidence" : str,
                                         "delta_amplitude": float,
                                         "note"       : str} }
        """
        print("\n" + "="*60)
        print("AUTO-MAP NODES  (Heuristic Discovery)")
        print("="*60)

        # Run systematic perturbation first (or reuse cached results)
        if not self._perturbation_results:
            print("  [info] No cached perturbation results; running now...")
            self.run_systematic_perturbation(t, sigma=sigma, verbose=False)

        self.model.eval()
        with torch.no_grad():
            y_base = self.model(t).squeeze()

        def _zero_crossings(signal):
            signs = torch.sign(signal)
            return int(((signs[1:] * signs[:-1]) < 0).sum().item())

        # Compute per-layer effect signatures
        layer_signatures = {}
        for name in self._perturbation_results:
            state           = self.model.state_dict()
            original_tensor = state[name].clone()
            noise           = torch.randn_like(original_tensor) * sigma
            state[name]     = original_tensor + noise
            self.model.load_state_dict(state)

            with torch.no_grad():
                y_pert = self.model(t).squeeze()

            base_amp  = (y_base.max() - y_base.min()).item()
            pert_amp  = (y_pert.max() - y_pert.min()).item()
            delta_amp = pert_amp - base_amp          # + means amplitude grew

            base_zc  = _zero_crossings(y_base)
            pert_zc  = _zero_crossings(y_pert)
            delta_zc = pert_zc - base_zc             # + means more crossings

            layer_signatures[name] = {
                "delta_amplitude": delta_amp,
                "delta_frequency": delta_zc,
            }

            # Restore
            state[name] = original_tensor
            self.model.load_state_dict(state)

        # Map each physical node to the best scoring layer
        mapping    = {}
        used_layers = {}   # layer -> [nodes]

        for node, trend in expected_trends.items():
            trend_lower = trend.lower()
            best_layer  = None
            best_score  = -1e9

            for name, sig in layer_signatures.items():
                if "decreases_amplitude" in trend_lower:
                    score = -sig["delta_amplitude"]
                elif "increases_amplitude" in trend_lower:
                    score =  sig["delta_amplitude"]
                elif "increases_frequency" in trend_lower:
                    score =  sig["delta_frequency"]
                elif "decreases_frequency" in trend_lower:
                    score = -sig["delta_frequency"]
                else:
                    # Fallback: raw sensitivity
                    score = self._perturbation_results[name]["delta_y"]

                if score > best_score:
                    best_score = score
                    best_layer = name

            used_layers.setdefault(best_layer, []).append(node)
            mapping[node] = {
                "layer"          : best_layer,
                "delta_amplitude": layer_signatures[best_layer]["delta_amplitude"],
                "delta_frequency": layer_signatures[best_layer]["delta_frequency"],
                "score"          : best_score,
                "trend"          : trend,
            }

        # Assign confidence (MT5 Task 2 logic)
        layer_use_counts = {layer: len(nodes) for layer, nodes in used_layers.items()}
        for node, info in mapping.items():
            layer = info["layer"]
            if layer_use_counts[layer] > 1:
                info["confidence"] = "Ambiguous"
                info["note"] = (
                    f"Layer '{layer}' matched by multiple nodes "
                    f"{used_layers[layer]} — cannot distinguish uniquely."
                )
            elif abs(info["score"]) < 1e-6:
                info["confidence"] = "Low"
                info["note"] = "Score near zero — layer had negligible matching effect."
            else:
                info["confidence"] = "High"
                info["note"] = (
                    f"Layer '{layer}' uniquely matched trend '{info['trend']}'."
                )

        # Print results table
        print(f"\n  {'Physical Node':<22}  {'Mapped Layer':<32}  {'Confidence':<12}  Note")
        print(f"  {'-'*100}")
        for node, info in mapping.items():
            conf_tag = "[HIGH]" if info["confidence"] == "High" else (
                       "[AMBIG]" if info["confidence"] == "Ambiguous" else "[LOW]")
            print(f"  {node:<22}  {info['layer']:<32}  {conf_tag:<12}  {info['note']}")
        print("="*60)

        self._mapping = mapping
        return mapping

    # ── MT5 Task 2: Generate Automated Mapping with Confidence Check ──────────
    def generate_mapping_with_confidence(
        self,
        expected_trends : dict,
        t               : torch.Tensor,
        sigma           : float = 0.1
    ) -> dict:
        """
        Runs auto_map_nodes and formats the final mapping dict for
        the frontend UI, including the alignment accuracy metric.

        Returns:
            {
              "mapping": {
                  physical_node: {
                      "layer"     : str,
                      "confidence": "High" | "Ambiguous" | "Low",
                      "note"      : str,
                      ...
                  }
              },
              "summary": {
                  "total_nodes"   : int,
                  "high_conf"     : int,
                  "ambiguous"     : int,
                  "low_conf"      : int,
                  "alignment_pct" : float   # % of High confidence mappings
              }
            }
        """
        print("\n" + "="*60)
        print("GENERATING AUTOMATED MAPPING WITH CONFIDENCE CHECK")
        print("="*60)

        mapping = self.auto_map_nodes(expected_trends, t, sigma)

        total     = len(mapping)
        high_conf = sum(1 for v in mapping.values() if v["confidence"] == "High")
        ambiguous = sum(1 for v in mapping.values() if v["confidence"] == "Ambiguous")
        low_conf  = sum(1 for v in mapping.values() if v["confidence"] == "Low")
        align_pct = (high_conf / total * 100) if total > 0 else 0.0

        summary = {
            "total_nodes"   : total,
            "high_conf"     : high_conf,
            "ambiguous"     : ambiguous,
            "low_conf"      : low_conf,
            "alignment_pct" : align_pct,
        }

        status = "[PASS >= 90%]" if align_pct >= 90.0 else "[FAIL < 90%]"
        print(f"\n  ALIGNMENT ACCURACY : {align_pct:.1f}%  {status}")
        print(f"  High Confidence    : {high_conf}/{total}")
        print(f"  Ambiguous          : {ambiguous}/{total}")
        print(f"  Low Confidence     : {low_conf}/{total}")
        print("="*60)

        return {
            "mapping": mapping,
            "summary": summary,
        }
