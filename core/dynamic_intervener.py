"""
dynamic_intervener.py
=====================
A fully dynamic replacement for SciMLIntervener + BurgersSciMLIntervener.

This single class reads a graph.json (produced by llm_graph_generator.py)
and performs ALL intervention analyses generically:
  - L2 Norm (prediction accuracy)
  - Physics Residual Norm (from the formula in graph.json)
  - Sensitivity (d(mean R²)/dp for every parameter in graph.json)
  - Systematic Perturbation (already generic — no change)
  - Auto Node Mapping with Confidence (trends from graph.json)

Zero hardcoding. Works with any PINN / physics-ML model.

Usage:
    from dynamic_intervener import DynamicSciMLIntervener

    intervener = DynamicSciMLIntervener(model, graph_json_dict)
    results = intervener.run_full_analysis(input_tensor, target_tensor)
"""

import copy
import json
import torch
import torch.nn as nn
import numpy as np
import contextlib
import io

@contextlib.contextmanager
def _silence_and_mock_plots():
    """Suppress stdout and mock plt.show() during dynamic code execution."""
    original_show = None
    try:
        import matplotlib.pyplot as plt
        original_show = plt.show
        plt.show = lambda *args, **kwargs: None
    except (ImportError, Exception):
        pass

    with io.StringIO() as dummy_out:
        with contextlib.redirect_stdout(dummy_out):
            try:
                yield
            finally:
                if original_show is not None:
                    import matplotlib.pyplot as plt
                    plt.show = original_show


# ══════════════════════════════════════════════════════════════════════════════
# DERIVATIVE COMPUTATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _compute_derivatives(model, input_tensor, input_spec, deriv_spec):
    """
    Dynamically compute all derivatives specified in the graph.json.

    Args:
        model        : The nn.Module PINN
        input_tensor : Input tensor (shape [N, input_dim]), requires_grad enabled
        input_spec   : graph["input_spec"] — {"names": ["t"], "dim": 1}  or
                        {"names": ["x", "t"], "dim": 2}
        deriv_spec   : graph["residual"]["derivatives"] — dict of derivative definitions

    Returns:
        dict mapping derivative variable names to their computed tensors
    """
    input_names = input_spec["names"]
    input_dim = input_spec["dim"]

    # Ensure input requires grad
    inp = input_tensor.detach().clone().requires_grad_(True)

    # Forward pass — get model output
    output = model(inp)  # shape [N, output_dim] or [N, 1]

    # Build a lookup for input columns
    # e.g., {"t": inp} for 1D  or  {"x": inp[:, 0:1], "t": inp[:, 1:2]} for 2D
    # NOTE: We keep slices of the same tensor so autograd works
    input_cols = {}
    if input_dim == 1:
        input_cols[input_names[0]] = inp
    else:
        for i, name in enumerate(input_names):
            input_cols[name] = inp  # We'll use column indexing during grad

    # Compute each derivative
    computed = {}

    for var_name, dinfo in deriv_spec.items():
        order = dinfo.get("order", 0)
        wrt_list = dinfo.get("wrt", [])
        source_var = dinfo.get("of", None)  # What to differentiate (None = model output)

        if order == 0:
            # This is just the raw model output (or a reference)
            computed[var_name] = output
            continue

        # Determine what we're differentiating
        if source_var and source_var in computed:
            current = computed[source_var]
        else:
            current = output

        # Compute derivatives step by step
        # For PDE: we may need partial derivatives wrt specific input columns
        for deriv_order in range(order):
            wrt_name = wrt_list[0] if len(wrt_list) == 1 else wrt_list[min(deriv_order, len(wrt_list) - 1)]

            grad = torch.autograd.grad(
                current, inp,
                grad_outputs=torch.ones_like(current),
                create_graph=True,
            )[0]

            # Select the column corresponding to the variable we differentiated wrt
            if input_dim > 1:
                col_idx = input_names.index(wrt_name)
                current = grad[:, col_idx:col_idx + 1]
            else:
                current = grad

        computed[var_name] = current

    # Also store the raw input tensor reference (for sensitivity computation)
    computed["_input"] = inp

    return computed


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC RESIDUAL EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_residual(formula: str, variables: dict) -> torch.Tensor:
    """
    Evaluate the residual formula string dynamically.

    Args:
        formula   : e.g., "m * x_tt + c * x_t + k * x"
        variables : dict mapping variable names to torch tensors
                    (includes derivative vars + parameter tensors)

    Returns:
        Residual tensor (should be ~0 if physics is satisfied)
    """
    # Restricted eval — only allow the provided variables and basic math
    safe_globals = {"__builtins__": {}}
    # Add torch math functions that might be needed
    safe_globals["abs"] = torch.abs
    safe_globals["sqrt"] = torch.sqrt
    safe_globals["exp"] = torch.exp
    safe_globals["sin"] = torch.sin
    safe_globals["cos"] = torch.cos
    safe_globals["log"] = torch.log

    with _silence_and_mock_plots():
        return eval(formula, safe_globals, variables)


# ══════════════════════════════════════════════════════════════════════════════
# TEST DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_test_data(graph: dict) -> tuple:
    """
    Generate synthetic test data using the code snippet from graph.json.

    Returns:
        (input_tensor, target_tensor) as torch float32 tensors
    """
    params_dict = {}
    for sym, info in graph.get("parameters", {}).items():
        params_dict[sym] = info.get("default_value", 1.0)

    code_str = graph.get("test_data_generator", "")
    if not code_str:
        # Fallback: generate random data within specified ranges
        print("  [!] No test_data_generator in graph.json — generating random input")
        input_spec = graph.get("input_spec", {"dim": 1, "names": ["t"]})
        dim = input_spec["dim"]
        ranges = input_spec.get("ranges", {})

        n_points = 1000
        cols = []
        for name in input_spec["names"]:
            r = ranges.get(name, [0, 1])
            cols.append(np.linspace(r[0], r[1], n_points))

        if dim == 1:
            inp = np.array(cols[0]).reshape(-1, 1)
        else:
            # Create meshgrid for multi-dim
            grids = np.meshgrid(*cols)
            inp = np.column_stack([g.ravel() for g in grids])

        # No ground truth available
        return (
            torch.tensor(inp, dtype=torch.float32),
            None
        )

    # Execute the test data generator code
    # Handle both escaped newlines and actual newlines
    if "\\n" in code_str and "\n" not in code_str.replace("\\n", ""):
        code_str = code_str.replace("\\n", "\n")

    local_ns = {}
    with _silence_and_mock_plots():
        exec(code_str, {"np": np, "numpy": np, "__builtins__": __builtins__}, local_ns)

    if "generate" not in local_ns:
        raise RuntimeError("test_data_generator code must define a function `generate(params)`")

    with _silence_and_mock_plots():
        inp_np, out_np = local_ns["generate"](params_dict)

    inp_tensor = torch.tensor(inp_np, dtype=torch.float32)
    out_tensor = torch.tensor(out_np, dtype=torch.float32)

    if out_tensor.dim() == 1:
        out_tensor = out_tensor.view(-1, 1)
    if inp_tensor.dim() == 1:
        inp_tensor = inp_tensor.view(-1, 1)

    return inp_tensor, out_tensor


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC INTERVENER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class DynamicSciMLIntervener:
    """
    Fully dynamic SciML intervener.
    Reads ALL physics info from graph.json — no hardcoded formulas.

    Provides:
      - compute_l2_norm()
      - compute_physics_residual_norm()
      - compute_sensitivity()
      - run_full_analysis()
      - get_perturbable_tensors()
      - run_systematic_perturbation()
      - auto_map_nodes()
      - generate_mapping_with_confidence()
    """

    def __init__(self, model: nn.Module, graph: dict):
        """
        Args:
            model : An nn.Module instance (already loaded, in eval mode)
            graph : The graph dict (from graph.json)
        """
        print("\n" + "=" * 60)
        print("DynamicSciMLIntervener — INITIALIZATION")
        print("=" * 60)

        self.model = model
        self.model.eval()
        self.graph = graph

        # Extract key specs from graph
        self.metadata = graph.get("metadata", {})
        self.input_spec = graph.get("input_spec", {"names": ["t"], "dim": 1})
        self.output_spec = graph.get("output_spec", {"names": ["x"], "dim": 1})
        self.parameters = graph.get("parameters", {})
        self.residual_spec = graph.get("residual", {})
        self.expected_trends = graph.get("expected_trends", {})

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n  System       : {self.metadata.get('system', 'Unknown')}")
        print(f"  Equation     : {self.metadata.get('equation', 'Unknown')}")
        print(f"  Model params : {num_params}")
        print(f"  Input spec   : {self.input_spec}")
        print(f"  Output spec  : {self.output_spec}")
        print(f"  Physics params: {list(self.parameters.keys())}")
        print(f"  Residual     : {self.residual_spec.get('formula', 'None')}")

        # Validate graph
        self._validate_graph()

        # Backup weights
        self.original_state = copy.deepcopy(self.model.state_dict())

        # Populated by perturbation / mapping
        self._perturbation_results = {}
        self._mapping = {}

        print("\nINITIALIZATION COMPLETE")
        print("=" * 60)

    def _validate_graph(self):
        """Basic validation of the graph structure."""
        nodes = self.graph.get("nodes", [])
        edges = self.graph.get("edges", [])

        print(f"\n  Physics Nodes ({len(nodes)}):")
        for n in nodes:
            label = n.get("label", n.get("id", "?"))
            symbol = n.get("symbol", "")
            ntype = n.get("type", "?")
            print(f"    [{ntype:11s}]  {symbol:6s}  {label}")

        print(f"\n  Causal Edges ({len(edges)}):")
        for e in edges:
            print(f"    {e['source']:12s} → {e['target']}")

        print(f"\n  Expected Trends ({len(self.expected_trends)}):")
        for label, trend in self.expected_trends.items():
            print(f"    {label:22s} → {trend}")

    # ── L2 Norm ───────────────────────────────────────────────────────────────
    def compute_l2_norm(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
    ) -> float:
        """
        Relative L2 Norm = ||pred - true||_2 / ||true||_2
        Works for any model (generic).
        """
        with torch.no_grad():
            pred = self.model(input_tensor)
            l2 = (torch.norm(pred - target_tensor) / torch.norm(target_tensor)).item()
        return l2

    # ── Physics Residual (DYNAMIC) ────────────────────────────────────────────
    def compute_physics_residual_norm(
        self,
        input_tensor: torch.Tensor,
        param_values: dict = None,
    ) -> float:
        """
        Compute mean(R²) where R is the physics residual from graph.json.

        Args:
            input_tensor : Model input (shape [N, input_dim])
            param_values : Dict of parameter values, e.g. {"m": 1.0, "c": 0.2, "k": 1.0}
                           If None, uses defaults from graph.json.

        Returns:
            float: mean(residual²)
        """
        formula = self.residual_spec.get("formula")
        if not formula:
            print("  [!] No residual formula in graph.json — skipping residual computation")
            return float("nan")

        deriv_spec = self.residual_spec.get("derivatives", {})

        # Get parameter values (use defaults if not provided)
        if param_values is None:
            param_values = {}
        for sym, info in self.parameters.items():
            if sym not in param_values:
                param_values[sym] = info.get("default_value", 1.0)

        # Compute all derivatives
        computed = _compute_derivatives(self.model, input_tensor, self.input_spec, deriv_spec)

        # Build variable dict for formula evaluation
        var_dict = dict(computed)
        # Remove internal keys
        var_dict.pop("_input", None)

        # Add parameter values as torch tensors (plain floats for eval)
        for sym, val in param_values.items():
            var_dict[sym] = float(val)

        # Evaluate residual
        residual = _evaluate_residual(formula, var_dict)
        residual_norm = torch.mean(residual ** 2).item()
        return residual_norm

    # ── Sensitivity (DYNAMIC) ─────────────────────────────────────────────────
    def compute_sensitivity(
        self,
        input_tensor: torch.Tensor,
        param_values: dict = None,
    ) -> dict:
        """
        Compute d(mean R²)/dp for EVERY parameter defined in graph.json.
        Compares computed sign against expected sign from graph.

        Returns:
            dict { symbol : { label, value, gradient, computed_sign,
                               expected_sign, consistent, trend } }
        """
        formula = self.residual_spec.get("formula")
        if not formula:
            print("  [!] No residual formula — skipping sensitivity computation")
            return {}

        deriv_spec = self.residual_spec.get("derivatives", {})

        # Get parameter values
        if param_values is None:
            param_values = {}
        for sym, info in self.parameters.items():
            if sym not in param_values:
                param_values[sym] = info.get("default_value", 1.0)

        # Compute derivatives
        computed = _compute_derivatives(self.model, input_tensor, self.input_spec, deriv_spec)

        # Build variable dict
        var_dict = dict(computed)
        var_dict.pop("_input", None)

        # Create torch tensors WITH requires_grad for each parameter
        param_tensors = {}
        for sym, val in param_values.items():
            pt = torch.tensor(float(val), requires_grad=True)
            param_tensors[sym] = pt
            var_dict[sym] = pt

        # Evaluate residual and compute loss
        residual = _evaluate_residual(formula, var_dict)
        loss_r = torch.mean(residual ** 2)

        # Compute gradients wrt all parameters
        grad_targets = list(param_tensors.values())
        grads = torch.autograd.grad(
            loss_r, grad_targets,
            create_graph=False,
            allow_unused=True,
        )

        # Build results
        results = {}
        for (sym, pt), grad_t in zip(param_tensors.items(), grads):
            info = self.parameters.get(sym, {})

            g_val = grad_t.item() if grad_t is not None else 0.0
            computed_sign = "+" if g_val > 1e-8 else ("-" if g_val < -1e-8 else "~0")
            expected_sign = info.get("expected_sensitivity_sign", "~")
            consistent = (expected_sign == "~") or (computed_sign == expected_sign)

            results[sym] = {
                "label": info.get("label", sym),
                "symbol": sym,
                "value": param_values[sym],
                "gradient": g_val,
                "computed_sign": computed_sign,
                "expected_sign": expected_sign,
                "consistent": consistent,
                "trend": info.get("trend", ""),
            }

        return results

    # ── Combined Analysis Report ──────────────────────────────────────────────
    def run_full_analysis(
        self,
        input_tensor: torch.Tensor,
        target_tensor: torch.Tensor = None,
        param_values: dict = None,
    ) -> dict:
        """
        Single call: L2 norm + physics residual + sensitivity.
        Works for ANY model — reads everything from graph.json.
        """
        eqn = self.metadata.get("equation", "unknown")
        system = self.metadata.get("system", "Unknown System")

        print("\n" + "=" * 60)
        print(f"FULL ANALYSIS REPORT — {system}")
        print("=" * 60)

        # ── A: L2 Norm ───────────────────────────────────────────────────────
        l2 = None
        l2_pass = None
        if target_tensor is not None:
            print("\n[A] RELATIVE L2 NORM  (prediction accuracy)")
            l2 = self.compute_l2_norm(input_tensor, target_tensor)
            l2_pass = l2 <= 0.05
            status = "PASS" if l2_pass else "FAIL"
            print(f"    Relative L2 : {l2:.6f}  ({l2 * 100:.3f}%)  [{status}]  [threshold <= 5%]")
        else:
            print("\n[A] RELATIVE L2 NORM  — skipped (no ground truth provided)")

        # ── B: Physics Residual ──────────────────────────────────────────────
        print(f"\n[B] PHYSICS RESIDUAL NORM  ({eqn})")
        res_norm = self.compute_physics_residual_norm(input_tensor, param_values)
        if not np.isnan(res_norm):
            print(f"    mean(R²)   : {res_norm:.6e}")
        else:
            print(f"    mean(R²)   : N/A (no residual formula)")

        # ── C: Sensitivity ───────────────────────────────────────────────────
        print("\n[C] SENSITIVITY  (d[mean R²]/dp  via autograd)")
        sensitivity = self.compute_sensitivity(input_tensor, param_values)

        if sensitivity:
            print(f"\n    {'Parameter':<22} {'Val':>5}  {'dR/dp':>13}  {'Sign':>5}  {'Exp':>5}  {'OK?'}")
            print(f"    {'-' * 70}")
            for sym, info in sensitivity.items():
                ok = "[OK]" if info["consistent"] else "[!!]"
                print(
                    f"    {info['label']:<22} {info['value']:>5.2f}  "
                    f"{info['gradient']:>13.6f}  "
                    f"{info['computed_sign']:>5}  "
                    f"{info['expected_sign']:>5}  {ok}"
                )
                if info["trend"]:
                    print(f"      -> {info['trend']}")

        # ── Summary ──────────────────────────────────────────────────────────
        print("\n" + "-" * 60)
        print("SUMMARY")
        if l2 is not None:
            print(f"  L2 Norm (relative)    : {l2 * 100:.3f}%   {'[PASS <=5%]' if l2_pass else '[FAIL >5%]'}")
        if not np.isnan(res_norm):
            print(f"  Physics Residual Norm : {res_norm:.6e}")
        for sym, info in sensitivity.items():
            ok = "consistent" if info["consistent"] else "INCONSISTENT [!!]"
            print(f"  dR/d{sym}  sign={info['computed_sign']}  expected={info['expected_sign']}  -> {ok}")
        print("=" * 60)

        return {
            "l2_norm": l2,
            "l2_pass": l2_pass,
            "physics_residual_norm": res_norm,
            "sensitivity": sensitivity,
        }

    def reset_model(self):
        """Restore original weights."""
        self.model.load_state_dict(self.original_state)

    # ══════════════════════════════════════════════════════════════════════════
    # PERTURBATION & MAPPING (carried over from task5final.py — already generic)
    # ══════════════════════════════════════════════════════════════════════════

    def get_perturbable_tensors(self) -> list:
        """List every manipulatable tensor in the model."""
        inventory = list(self.model.state_dict().keys())

        print("\n" + "=" * 60)
        print("TARGET INVENTORY  (get_perturbable_tensors)")
        print("=" * 60)
        for i, name in enumerate(inventory):
            shape = self.model.state_dict()[name].shape
            print(f"  [{i:02d}] {name:<35}  shape={tuple(shape)}")
        print(f"\n  Total tensors: {len(inventory)}")
        print("=" * 60)

        return inventory

    def run_systematic_perturbation(
        self,
        input_tensor: torch.Tensor,
        sigma: float = 0.1,
        verbose: bool = True,
    ) -> dict:
        """
        Systematic perturbation test: inject noise into each tensor,
        measure output change (delta_y). Sorted by impact.
        """
        inventory = self.get_perturbable_tensors()

        if verbose:
            print("\n" + "=" * 60)
            print(f"SYSTEMATIC PERTURBATION  (sigma={sigma})")
            print("=" * 60)
            print(f"  {'Layer':<35}  {'delta_y (mean |delta_output|)':>30}")
            print(f"  {'-' * 67}")

        self.model.eval()
        with torch.no_grad():
            y_base = self.model(input_tensor)

        results = {}

        for name in inventory:
            state = self.model.state_dict()
            original_tensor = state[name].clone()

            # Perturb
            noise = torch.randn_like(original_tensor) * sigma
            state[name] = original_tensor + noise
            self.model.load_state_dict(state)

            # Forward pass
            with torch.no_grad():
                y_pert = self.model(input_tensor)
            delta_y = (y_pert - y_base).abs().mean().item()

            # Restore
            state[name] = original_tensor
            self.model.load_state_dict(state)

            results[name] = {
                "delta_y": delta_y,
                "shape": tuple(original_tensor.shape),
            }

            if verbose:
                print(f"  {name:<35}  delta_y = {delta_y:.6f}")

        # Sort by impact
        sorted_results = dict(
            sorted(results.items(), key=lambda x: x[1]["delta_y"], reverse=True)
        )

        if verbose:
            print(f"\n  Top 3 most influential tensors:")
            for i, (name, info) in enumerate(sorted_results.items()):
                if i >= 3:
                    break
                print(f"    {i + 1}. {name:<35}  delta_y = {info['delta_y']:.6f}")
            print("=" * 60)

        self._perturbation_results = sorted_results
        return sorted_results

    def auto_map_nodes(
        self,
        input_tensor: torch.Tensor,
        sigma: float = 0.1,
        expected_trends: dict = None,
    ) -> dict:
        """
        Auto-discover which PyTorch layer corresponds to which physical
        causal node by comparing perturbation effects against expected trends.

        Uses expected_trends from graph.json if not explicitly provided.
        """
        if expected_trends is None:
            expected_trends = self.expected_trends

        if not expected_trends:
            print("  [!] No expected_trends available — cannot auto-map nodes")
            return {}

        print("\n" + "=" * 60)
        print("AUTO-MAP NODES  (Heuristic Discovery)")
        print("=" * 60)

        # Run perturbation if not cached
        if not self._perturbation_results:
            print("  [info] No cached perturbation results; running now...")
            self.run_systematic_perturbation(input_tensor, sigma=sigma, verbose=False)

        self.model.eval()
        with torch.no_grad():
            y_base = self.model(input_tensor).squeeze()

        def _zero_crossings(signal):
            signs = torch.sign(signal)
            return int(((signs[1:] * signs[:-1]) < 0).sum().item())

        # Compute per-layer effect signatures
        layer_signatures = {}
        for name in self._perturbation_results:
            state = self.model.state_dict()
            original_tensor = state[name].clone()
            noise = torch.randn_like(original_tensor) * sigma
            state[name] = original_tensor + noise
            self.model.load_state_dict(state)

            with torch.no_grad():
                y_pert = self.model(input_tensor).squeeze()

            base_amp = (y_base.max() - y_base.min()).item()
            pert_amp = (y_pert.max() - y_pert.min()).item()
            delta_amp = pert_amp - base_amp

            base_zc = _zero_crossings(y_base)
            pert_zc = _zero_crossings(y_pert)
            delta_zc = pert_zc - base_zc

            layer_signatures[name] = {
                "delta_amplitude": delta_amp,
                "delta_frequency": delta_zc,
            }

            state[name] = original_tensor
            self.model.load_state_dict(state)

        # Map each physical node to best layer
        mapping = {}
        used_layers = {}

        for node, trend in expected_trends.items():
            trend_lower = trend.lower()
            best_layer = None
            best_score = -1e9

            for name, sig in layer_signatures.items():
                if "decreases_amplitude" in trend_lower:
                    score = -sig["delta_amplitude"]
                elif "increases_amplitude" in trend_lower:
                    score = sig["delta_amplitude"]
                elif "increases_frequency" in trend_lower:
                    score = sig["delta_frequency"]
                elif "decreases_frequency" in trend_lower:
                    score = -sig["delta_frequency"]
                else:
                    score = self._perturbation_results[name]["delta_y"]

                if score > best_score:
                    best_score = score
                    best_layer = name

            used_layers.setdefault(best_layer, []).append(node)
            mapping[node] = {
                "layer": best_layer,
                "delta_amplitude": layer_signatures[best_layer]["delta_amplitude"],
                "delta_frequency": layer_signatures[best_layer]["delta_frequency"],
                "score": best_score,
                "trend": trend,
            }

        # Assign confidence
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
                info["note"] = f"Layer '{layer}' uniquely matched trend '{info['trend']}'."

        # Print results
        print(f"\n  {'Physical Node':<22}  {'Mapped Layer':<32}  {'Confidence':<12}  Note")
        print(f"  {'-' * 100}")
        for node, info in mapping.items():
            conf_tag = "[HIGH]" if info["confidence"] == "High" else (
                "[AMBIG]" if info["confidence"] == "Ambiguous" else "[LOW]")
            print(f"  {node:<22}  {info['layer']:<32}  {conf_tag:<12}  {info['note']}")
        print("=" * 60)

        self._mapping = mapping
        return mapping

    def _measure_node_sensitivity(
        self,
        node_id: str,
        input_tensor: torch.Tensor,
    ) -> dict:
        """
        Strategy 2: Sensitivity Mapping (for Physical Nodes like y_x, y_xx).

        Computes the gradient of a physical node's mean value with respect
        to every layer's weights. The layer with the largest gradient
        magnitude is declared the "owner" of that physical node.

        Returns:
            dict { layer_name: gradient_magnitude }
        """
        deriv_spec  = self.residual_spec.get("derivatives", {})
        if not deriv_spec:
            return {}

        inp = input_tensor.detach().clone().requires_grad_(True)
        computed = _compute_derivatives(self.model, inp, self.input_spec, deriv_spec)

        if node_id not in computed:
            return {}

        # Back-prop from the physical node value to the model weights
        target = computed[node_id].mean()
        target.backward()

        # Read gradient magnitudes for every named parameter in the model
        sensitivity = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                sensitivity[name] = param.grad.abs().mean().item()
            else:
                sensitivity[name] = 0.0

        # Zero all grads to keep the model clean
        self.model.zero_grad()
        return sensitivity

    def generate_mapping_with_confidence(
        self,
        input_tensor: torch.Tensor,
        sigma: float = 0.1,
        expected_trends: dict = None,
    ) -> dict:
        """
        Combined mapping using two strategies:

        Strategy 1 — Trend Mapping (for parameters like mu, k):
            Perturbs each layer and observes Amplitude/Frequency change.
            Matches the layer whose effect best matches 'expected_trends'.

        Strategy 2 — Sensitivity Mapping (for physical nodes like y_x, y_xx):
            Computes the gradient of the node value wrt each layer's weights.
            The layer with the largest gradient magnitude "owns" that node.
        """
        print("\n" + "=" * 60)
        print("GENERATING AUTOMATED MAPPING WITH CONFIDENCE")
        print("=" * 60)

        mapping = {}
        all_nodes   = self.graph.get("nodes", [])
        param_ids   = set(self.parameters.keys())

        # ── Strategy 1: Trend Mapping for Parameters ─────────────────────────
        trends = expected_trends or self.expected_trends
        if trends:
            print("\n  [Strategy 1] Trend Mapping for Parameters...")
            trend_mapping = self.auto_map_nodes(input_tensor, sigma, trends)
            for node_label, info in trend_mapping.items():
                mapping[node_label] = {**info, "strategy": "Trend (Behavior)"}
        else:
            print("\n  [Strategy 1] No expected_trends — skipping trend mapping.")

        # ── Strategy 2: Sensitivity Mapping for Physical Nodes ────────────────
        print("\n  [Strategy 2] Sensitivity Mapping for Physical Nodes...")
        already_mapped_labels = set(mapping.keys())

        for node in all_nodes:
            node_id    = node.get("id", "")
            node_label = node.get("label", node_id)
            node_type  = node.get("type", "state")

            # Skip parameters (handled by Strategy 1) and already-mapped nodes
            if node_type == "parameter" or node_id in param_ids:
                continue
            if node_label in already_mapped_labels:
                continue

            sens = self._measure_node_sensitivity(node_id, input_tensor)
            if not sens:
                print(f"    [!] {node_label}: no derivatives found — skipping.")
                continue

            best_layer = max(sens, key=sens.get)
            best_score = sens[best_layer]

            confidence = "High" if best_score > 1e-4 else ("Medium" if best_score > 1e-7 else "Low")
            mapping[node_label] = {
                "layer":      best_layer,
                "score":      best_score,
                "confidence": confidence,
                "note":       f"Layer '{best_layer}' has max gradient magnitude ({best_score:.2e}) for node '{node_id}'.",
                "strategy":   "Sensitivity (Gradient)",
            }
            print(f"    [OK] {node_label:22s} → {best_layer:30s}  [{confidence}]  score={best_score:.2e}")

        # ── Summary ───────────────────────────────────────────────────────────
        total     = len(mapping)
        high_conf = sum(1 for v in mapping.values() if v["confidence"] == "High")
        ambiguous = sum(1 for v in mapping.values() if v["confidence"] == "Ambiguous")
        low_conf  = sum(1 for v in mapping.values() if v["confidence"] in ("Low", "Medium"))
        align_pct = (high_conf / total * 100) if total > 0 else 0.0

        summary = {
            "total_nodes": total,
            "high_conf":   high_conf,
            "ambiguous":   ambiguous,
            "low_conf":    low_conf,
            "alignment_pct": align_pct,
        }

        status = "[PASS >= 90%]" if align_pct >= 90.0 else "[FAIL < 90%]"
        print(f"\n  ALIGNMENT ACCURACY : {align_pct:.1f}%  {status}")
        print(f"  High Confidence    : {high_conf}/{total}")
        print(f"  Ambiguous          : {ambiguous}/{total}")
        print(f"  Low / Medium       : {low_conf}/{total}")
        print("=" * 60)

        return {"mapping": mapping, "summary": summary}


    # ══════════════════════════════════════════════════════════════════════════
    # INTERVENTION ENGINE (from intervention_engine.py — made generic)
    # ══════════════════════════════════════════════════════════════════════════

    def apply_intervention(
        self,
        layer_name: str,
        intervention_type: str = "mask",
        strength: float = 1.0,
        input_tensor: torch.Tensor = None,
        target_tensor: torch.Tensor = None,
    ) -> dict:
        """
        Apply an intervention (mask/perturb) on a specific layer and evaluate.

        Args:
            layer_name        : Name of the tensor to intervene on (from state_dict)
            intervention_type : "mask" (zero out) or "noise" (add Gaussian noise)
            strength          : Intervention strength (0.0 = no change, 1.0 = full)
            input_tensor      : Input for evaluation
            target_tensor     : Ground truth for MSE evaluation

        Returns:
            dict with baseline_mse, intervened_mse, delta, percent_change
        """
        self.reset_model()
        criterion = nn.MSELoss()

        # Baseline
        with torch.no_grad():
            baseline_pred = self.model(input_tensor)
            if target_tensor is not None:
                baseline_mse = criterion(baseline_pred, target_tensor).item()
            else:
                baseline_mse = 0.0

        # Apply intervention
        state = self.model.state_dict()
        if layer_name not in state:
            raise ValueError(f"Layer '{layer_name}' not found. Available: {list(state.keys())}")

        original = state[layer_name].clone()

        if intervention_type == "mask":
            state[layer_name] = original * (1.0 - strength)
        elif intervention_type == "noise":
            noise = torch.randn_like(original) * strength
            state[layer_name] = original + noise
        else:
            raise ValueError(f"Unknown intervention type: {intervention_type}")

        self.model.load_state_dict(state)

        # Evaluate
        with torch.no_grad():
            intervened_pred = self.model(input_tensor)
            if target_tensor is not None:
                intervened_mse = criterion(intervened_pred, target_tensor).item()
            else:
                intervened_mse = (intervened_pred - baseline_pred).abs().mean().item()

        delta = intervened_mse - baseline_mse
        pct_change = (delta / baseline_mse * 100) if baseline_mse > 0 else 0.0

        # Restore
        self.reset_model()

        return {
            "layer": layer_name,
            "intervention_type": intervention_type,
            "strength": strength,
            "baseline_mse": baseline_mse,
            "intervened_mse": intervened_mse,
            "delta": delta,
            "percent_change": pct_change,
        }
