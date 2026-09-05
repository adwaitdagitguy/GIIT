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

# Physics ruleset utilities (compute derived quantities from model output)
try:
    from utils.physics_utils import compute_derived_quantity, get_quantity_for_trend
except ImportError:
    # Fallback: define no-ops so the rest of the module loads cleanly
    def compute_derived_quantity(ruleset, quantity_name, t_array, x_array):
        return None
    def get_quantity_for_trend(ruleset, trend_str):
        return None, None


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

        if dinfo.get("type") == "custom":
            op = dinfo.get("operation", "add")
            vars_to_combine = dinfo.get("vars", [])
            if len(vars_to_combine) > 0 and all(v in computed for v in vars_to_combine):
                result = computed[vars_to_combine[0]]
                for v in vars_to_combine[1:]:
                    if op == "add":
                        result = result + computed[v]
                    elif op == "multiply":
                        result = result * computed[v]
                computed[var_name] = result
            else:
                computed[var_name] = output  # Fallback
            continue

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

    PERTURBATION_SEED = 123

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
        self._perturbation_seed = None
        self._mapping = {}
        # Physics ruleset for derived quantity scoring (set via set_physics_ruleset)
        self.physics_ruleset = None

        print("\nINITIALIZATION COMPLETE")
        print("=" * 60)

    def set_physics_ruleset(self, ruleset: dict):
        """
        Attach a physics ruleset (from LLM suggestion or manual upload) to the
        intervener. Once set, auto_map_nodes() will use the ruleset's compute_fn
        functions to score layers by the magnitude and direction of change in
        derived physical quantities (e.g. frequency, amplitude).

        Args:
            ruleset : Dict parsed from the physics ruleset JSON.
        """
        self.physics_ruleset = ruleset
        quantities = list(ruleset.get("quantities", {}).keys()) if ruleset else []
        print(f"\n  [Ruleset] Physics ruleset attached. Quantities: {quantities}")



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
            if isinstance(e, dict):
                source = e.get("source", "")
                target = e.get("target", "")
            elif isinstance(e, (list, tuple)) and len(e) >= 2:
                source, target = e[0], e[1]
            else:
                continue
            print(f"    {str(source):12s} -> {target}")

        print(f"\n  Expected Trends ({len(self.expected_trends)}):")
        for label, trend in self.expected_trends.items():
            print(f"    {label:22s} -> {trend}")

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
        seed: int = None,
    ) -> dict:
        """
        Systematic perturbation test: inject noise into each tensor,
        measure output change (delta_y). Sorted by impact.
        """
        seed = self.PERTURBATION_SEED if seed is None else int(seed)
        torch.manual_seed(seed)
        self._perturbation_seed = seed
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
        seed: int = None,
        physics_ruleset: dict = None,
    ) -> dict:
        """
        Auto-discover which PyTorch layer corresponds to which physical
        causal node by comparing perturbation effects against expected trends.

        Uses expected_trends from graph.json if not explicitly provided.
        When a physics_ruleset is provided (or attached via set_physics_ruleset),
        derived quantities (frequency, amplitude, …) are computed from the
        model output using the ruleset's compute_fn functions.  Scoring then
        uses both the **magnitude** and the **direction** of change:
            raw_score = direction_sign × |Δquantity|
        where direction_sign = +1 when the change matches the expected trend
        direction, and −1 otherwise.

        Falls back to the existing zero-crossing / signal-range heuristics when
        no ruleset is available or no matching quantity is found.

        Args:
            input_tensor   : Model input tensor (shape [N, input_dim]).
            sigma          : Gaussian noise std for perturbation.
            expected_trends: Override the graph.json expected_trends.
            seed           : Random seed for reproducible perturbation.
            physics_ruleset: Override self.physics_ruleset for this call.
        """
        if expected_trends is None:
            expected_trends = self.expected_trends

        if not expected_trends:
            print("  [!] No expected_trends available — cannot auto-map nodes")
            return {}

        # Resolve ruleset: kwarg > self attribute
        ruleset = physics_ruleset if physics_ruleset is not None else self.physics_ruleset

        seed = self.PERTURBATION_SEED if seed is None else int(seed)
        torch.manual_seed(seed)

        if self._perturbation_seed != seed:
            self._perturbation_results = {}

        print("\n" + "=" * 60)
        print("AUTO-MAP NODES  (Heuristic Discovery)")
        if ruleset:
            qnames = list(ruleset.get("quantities", {}).keys())
            print(f"AUTO-MAP NODES  (Ruleset-Enhanced: {qnames})")
        else:
            print("AUTO-MAP NODES  (Heuristic Discovery)")
        print("=" * 60)

        # Run perturbation if not cached
        if not self._perturbation_results:
            print("  [info] No cached perturbation results; running now...")
            self.run_systematic_perturbation(
                input_tensor, sigma=sigma, verbose=False, seed=seed
            )

        self.model.eval()
        with torch.no_grad():
            y_base = self.model(input_tensor).squeeze()

        # ── Precompute base derived quantities using the ruleset ──────────────
        # Convert input/output to numpy once for the baseline
        t_base_np = input_tensor.detach().cpu().numpy()
        if t_base_np.ndim > 1:
            t_base_np = t_base_np[:, 0]  # use first input column (e.g. time)
        x_base_np = y_base.detach().cpu().numpy().ravel()

        base_derived = {}     # { quantity_name: scalar }
        if ruleset:
            for qname in ruleset.get("quantities", {}):
                val = compute_derived_quantity(ruleset, qname, t_base_np, x_base_np)
                base_derived[qname] = val
                print(f"  [Ruleset] Base {qname} = {val}")

        def _zero_crossings(signal):
            signs = torch.sign(signal)
            return int(((signs[1:] * signs[:-1]) < 0).sum().item())

        # Compute per-layer effect signatures
        # ── Compute per-layer effect signatures ───────────────────────────────
        layer_signatures = {}
        for name in self._perturbation_results:
            state = self.model.state_dict()
            original_tensor = state[name].clone()
            torch.manual_seed(seed)
            # Use strictly positive noise for physical parameter directional perturbation
            noise = torch.abs(torch.randn_like(original_tensor)) * sigma
            state[name] = original_tensor + noise
            self.model.load_state_dict(state)

            with torch.no_grad():
                y_pert = self.model(input_tensor).squeeze()

            # ── Fallback heuristics (always computed) ─────────────────────────
            base_amp = (y_base.max() - y_base.min()).item()
            pert_amp = (y_pert.max() - y_pert.min()).item()
            delta_amp = pert_amp - base_amp

            base_zc = _zero_crossings(y_base)
            pert_zc = _zero_crossings(y_pert)
            delta_zc = pert_zc - base_zc

            sig = {
                "delta_amplitude": delta_amp,
                "delta_frequency": delta_zc,
            }

            # ── Ruleset-derived quantity deltas ───────────────────────────────
            if ruleset:
                x_pert_np = y_pert.detach().cpu().numpy().ravel()
                for qname in ruleset.get("quantities", {}):
                    q_pert = compute_derived_quantity(ruleset, qname, t_base_np, x_pert_np)
                    q_base = base_derived.get(qname)
                    if q_pert is not None and q_base is not None:
                        sig[f"delta_{qname}"] = q_pert - q_base
                    else:
                        sig[f"delta_{qname}"] = None

            layer_signatures[name] = sig

            state[name] = original_tensor
            self.model.load_state_dict(state)

        # Map each physical node to best layer
        # ── Map each physical node to the best layer ──────────────────────────
        mapping = {}
        used_layers = {}

        for node, trend in expected_trends.items():
            trend_lower = trend.lower()
            scored_layers = []

            # Check if the ruleset provides a matching quantity for this trend
            q_name, q_info = get_quantity_for_trend(ruleset, trend) if ruleset else (None, None)
            using_ruleset_for_node = (q_name is not None) and (
                f"delta_{q_name}" in next(iter(layer_signatures.values()), {})
            )

            # Determine expected direction: +1 for "increases_X", -1 for "decreases_X"
            if "increases_" in trend_lower:
                expected_direction = +1
            elif "decreases_" in trend_lower:
                expected_direction = -1
            else:
                expected_direction = 0

            for name, sig in layer_signatures.items():
                if using_ruleset_for_node:
                    # ── Ruleset path: direction × magnitude ─────────────────
                    dq = sig.get(f"delta_{q_name}")
                    if dq is None:
                        raw_score = 0.0
                        change = None
                    else:
                        direction_sign = +1 if (expected_direction * dq > 0) else -1
                        raw_score = direction_sign * abs(dq)
                        change = dq
                elif "decreases_amplitude" in trend_lower:
                    raw_score = -sig["delta_amplitude"]
                    change = sig["delta_amplitude"]
                elif "increases_amplitude" in trend_lower:
                    raw_score = sig["delta_amplitude"]
                    change = sig["delta_amplitude"]
                elif "increases_frequency" in trend_lower:
                    raw_score = sig["delta_frequency"]
                    change = sig["delta_frequency"]
                elif "decreases_frequency" in trend_lower:
                    raw_score = -sig["delta_frequency"]
                    change = sig["delta_frequency"]
                else:
                    raw_score = self._perturbation_results[name]["delta_y"]
                    change = None

                scored_layers.append((name, raw_score, change))

            scale = max((abs(item[1]) for item in scored_layers), default=0.0)
            ranked_layers = sorted(scored_layers, key=lambda item: abs(item[1]), reverse=True)

            scoring_mode = f"Ruleset ({q_name})" if using_ruleset_for_node else "Heuristic"
            print(f"\n  Parameter: {node}  |  Trend: {trend}  |  Scoring: {scoring_mode}")
            print(f"    {'Rank':<6} {'Tensor':<32} {'Raw score':>12} {'Normalized':>12} {'Change':>10} {'Direction':>10}")
            for rank, (name, raw_score, change) in enumerate(ranked_layers, start=1):
                normalized_score = raw_score / scale if scale else 0.0
                if change is None:
                    change_sign = "n/a"
                elif change > 1e-8:
                    change_sign = "+"
                elif change < -1e-8:
                    change_sign = "-"
                else:
                    change_sign = "~0"
                # Direction match for ruleset mode
                if using_ruleset_for_node and change is not None:
                    dir_match = "✓" if (expected_direction * change > 0) else "✗"
                else:
                    dir_match = "n/a"
                print(
                    f"    {rank:<6} {name:<32} {raw_score:>12.6g} "
                    f"{normalized_score:>12.6f} {change_sign:>10} {dir_match:>10}"
                )

            best_layer, raw_score, change = max(scored_layers, key=lambda item: abs(item[1]))
            best_score = abs(raw_score) / scale if scale else 0.0

            # Direction match for the best layer
            if using_ruleset_for_node and change is not None:
                direction_match = bool(expected_direction * change > 0)
                delta_q_best = change
            else:
                direction_match = None
                delta_q_best = None

            used_layers.setdefault(best_layer, []).append(node)
            mapping[node] = {
                "layer":           best_layer,
                "delta_amplitude": layer_signatures[best_layer]["delta_amplitude"],
                "delta_frequency": layer_signatures[best_layer]["delta_frequency"],
                "score":           best_score,
                "raw_score":       raw_score,
                "trend":           trend,
                # Ruleset-specific fields (None when heuristic path is used)
                "quantity_name":   q_name,
                "delta_q":         delta_q_best,
                "direction_match": direction_match,
                "scoring_mode":    scoring_mode,
            }

        # Assign confidence
        # ── Assign confidence ─────────────────────────────────────────────────
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
            # Downgrade confidence if direction didn't match
            if info["direction_match"] is False and info["confidence"] == "High":
                info["confidence"] = "Medium"
                info["note"] += " (Direction mismatch — layer moved quantity in wrong direction.)"

        # Print results
        print(f"\n  {'Physical Node':<22}  {'Mapped Layer':<32}  {'Confidence':<12}  {'Dir Match':<10}  Note")
        print(f"  {'-' * 115}")
        for node, info in mapping.items():
            conf_tag = "[HIGH]" if info["confidence"] == "High" else (
                "[MED]" if info["confidence"] == "Medium" else (
                    "[AMBIG]" if info["confidence"] == "Ambiguous" else "[LOW]"
                )
            )
            dir_tag = "✓" if info["direction_match"] is True else (
                "✗" if info["direction_match"] is False else "n/a"
            )
            print(f"  {node:<22}  {info['layer']:<32}  {conf_tag:<12}  {dir_tag:<10}  {info['note']}")
        print("=" * 60)

        self._mapping = mapping
        return mapping

    def _measure_node_sensitivity(
        self,
        node_id: str,
        input_tensor: torch.Tensor,
    ) -> dict:
        """
        Strategy 2: Sensitivity Mapping via Gradient (for Physical Nodes like y_x, y_xx).

        Computes the gradient of a physical node's mean value with respect
        to every layer's weights. The layer with the largest gradient
        magnitude is declared the "owner" of that physical node.

        Returns:
            dict { layer_name: gradient_magnitude }
        """
        deriv_spec = self.residual_spec.get("derivatives", {})
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

    def _measure_node_perturbation_sensitivity(
        self,
        node_id: str,
        input_tensor: torch.Tensor,
        sigma: float = 0.1,
        seed: int = None,
    ) -> dict:
        """
        Strategy 2: Perturbation Sensitivity (delta u / |u|) for Physical Nodes.

        Measures the relative change in a physical node's value when each neural
        network layer tensor is perturbed by Gaussian noise:
            Sensitivity(W) = mean(|u_pert - u_base|) / (mean(|u_base|) + 1e-8)

        Returns:
            dict { layer_name: relative_perturbation_sensitivity }
        """
        deriv_spec = self.residual_spec.get("derivatives", {})

        # 1. Baseline evaluation of the physical node
        self.model.eval()
        base_inp = input_tensor.detach().clone().requires_grad_(True)
        base_computed = _compute_derivatives(self.model, base_inp, self.input_spec, deriv_spec)

        if node_id not in base_computed:
            return {}

        u_base = base_computed[node_id].detach()
        u_denom = u_base.abs().mean().item() + 1e-8

        # 2. Perturb each layer and measure relative change
        seed = self.PERTURBATION_SEED if seed is None else int(seed)
        inventory = self.get_perturbable_tensors()
        sensitivity = {}

        for name in inventory:
            state = self.model.state_dict()
            orig_t = state[name].clone()

            torch.manual_seed(seed)
            noise = torch.randn_like(orig_t) * sigma
            state[name] = orig_t + noise
            self.model.load_state_dict(state)

            # Recompute physical node with perturbed layer
            pert_inp = input_tensor.detach().clone().requires_grad_(True)
            pert_computed = _compute_derivatives(self.model, pert_inp, self.input_spec, deriv_spec)

            if node_id in pert_computed:
                u_pert = pert_computed[node_id].detach()
                delta_u = (u_pert - u_base).abs().mean().item()
                rel_sens = delta_u / u_denom
            else:
                rel_sens = 0.0

            sensitivity[name] = rel_sens

            # Restore layer
            state[name] = orig_t
            self.model.load_state_dict(state)

        return sensitivity

    def generate_mapping_with_confidence(
        self,
        input_tensor: torch.Tensor,
        sigma: float = 0.1,
        expected_trends: dict = None,
        seed: int = None,
        sensitivity_method: str = "gradient",
    ) -> dict:
        """
        Combined mapping using two strategies:

        Strategy 1 — Trend Mapping (for parameters like mu, k):
            Perturbs each layer and observes Amplitude/Frequency change.
            Matches the layer whose effect best matches 'expected_trends'.

        Strategy 2 — Sensitivity Mapping (for physical nodes like y_x, y_xx):
            - If sensitivity_method == 'gradient': uses gradient magnitude wrt weights
            - If sensitivity_method == 'perturbation': uses relative perturbation sensitivity (delta u / u)
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
            ruleset_status = "with ruleset" if self.physics_ruleset else "heuristic fallback"
            print(f"\n  [Strategy 1] Trend Mapping for Parameters ({ruleset_status})...")
            trend_mapping = self.auto_map_nodes(
                input_tensor=input_tensor,
                sigma=sigma,
                expected_trends=trends,
                seed=seed,
                physics_ruleset=self.physics_ruleset,
            )
            for node_label, info in trend_mapping.items():
                mapping[node_label] = {**info, "strategy": "Trend (Behavior)"}
        else:
            print("\n  [Strategy 1] No expected_trends — skipping trend mapping.")


        # ── Strategy 2: Sensitivity Mapping for Physical Nodes ────────────────
        is_perturb = (sensitivity_method.lower() == "perturbation")
        method_label = "Perturbation (Δu/u)" if is_perturb else "Gradient"
        print(f"\n  [Strategy 2] Sensitivity Mapping for Physical Nodes ({method_label})...")
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

            if is_perturb:
                sens = self._measure_node_perturbation_sensitivity(
                    node_id, input_tensor, sigma=sigma, seed=seed
                )
            else:
                sens = self._measure_node_sensitivity(node_id, input_tensor)

            if not sens:
                print(f"    [!] {node_label}: no derivatives found — skipping.")
                continue

            ranked_layers = sorted(sens.items(), key=lambda item: abs(item[1]), reverse=True)
            best_layer, raw_score = ranked_layers[0]
            scale = max((abs(value) for value in sens.values()), default=0.0)
            best_score = abs(raw_score) / scale if scale else 0.0
            rank = 1

            if is_perturb:
                confidence = "High" if raw_score > 0.05 else ("Medium" if raw_score > 0.01 else "Low")
                note_str = f"Layer '{best_layer}' has max relative perturbation Δu/u ({raw_score:.2e}) for node '{node_id}'."
                strategy_str = "Sensitivity (Perturbation Δu/u)"
            else:
                confidence = "High" if raw_score > 1e-4 else ("Medium" if raw_score > 1e-7 else "Low")
                note_str = f"Layer '{best_layer}' has max gradient magnitude ({raw_score:.2e}) for node '{node_id}'."
                strategy_str = "Sensitivity (Gradient)"

            mapping[node_label] = {
                "layer":      best_layer,
                "rank":       rank,
                "score":      best_score,
                "raw_score":  raw_score,
                "confidence": confidence,
                "note":       note_str,
                "strategy":   strategy_str,
                "ranked_layers": [
                    {"rank": i + 1, "layer": layer_name, "score": abs(score), "raw_score": score}
                    for i, (layer_name, score) in enumerate(ranked_layers)
                ],
            }

            col_hdr = "Δu/u" if is_perturb else "abs(grad)"
            print(f"\n    [OK] {node_label:22s} → {best_layer:30s}  [{confidence}]  rank={rank}  score={best_score:.6f} raw={raw_score:.2e}")
            print(f"      {'Rank':<4} {'Layer':<30} {col_hdr:>12}")
            for rnk, (layer_name, score) in enumerate(ranked_layers, start=1):
                print(f"      {rnk:<4} {layer_name:<30} {abs(score):>12.6e}")

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
            "sensitivity_method": sensitivity_method,
        }

        status = "[PASS >= 90%]" if align_pct >= 90.0 else "[FAIL < 90%]"
        print(f"\n  ALIGNMENT ACCURACY : {align_pct:.1f}%  {status}")
        print(f"  High Confidence    : {high_conf}/{total}")
        print(f"  Ambiguous          : {ambiguous}/{total}")
        print(f"  Low / Medium       : {low_conf}/{total}")
        print("=" * 60)

        return {"mapping": mapping, "summary": summary}

    def run_physical_seed_sweep(
        self,
        input_tensor: torch.Tensor,
        start_seed: int = 0,
        end_seed: int = 20,
        sigma: float = 0.1,
        sensitivity_method: str = "gradient",
        verbose: bool = False,
    ) -> dict:
        """
        Run a seed sweep specifically across physical parameters and causal variables.
        
        For each seed in [start_seed, end_seed], evaluates the mapping confidence score,
        response signature, and physical parameter sensitivity.

        Returns:
            dict containing per-physical-variable metric arrays, statistics, and sweep summary.
        """
        seeds = list(range(int(start_seed), int(end_seed) + 1))
        if not seeds:
            seeds = [int(start_seed)]

        # Collect data per physical variable across seeds
        per_var_data = {}

        # 1. Physical nodes from graph
        all_nodes = self.graph.get("nodes", [])
        for node in all_nodes:
            nid = node.get("id", node.get("name", ""))
            nlabel = node.get("label", nid)
            ntype = node.get("type", "state")
            per_var_data[nlabel] = {
                "id": nid,
                "label": nlabel,
                "type": ntype,
                "description": node.get("description", ""),
                "formula": node.get("formula", ""),
                "scores": [],
                "raw_scores": [],
                "mapped_layers": [],
                "direction_matches": [],
                "metric_name": "Mapping Alignment Score",
            }

        # 2. Physical parameters from graph
        for sym, pinfo in self.parameters.items():
            plabel = pinfo.get("label", sym)
            if plabel not in per_var_data and sym not in per_var_data:
                per_var_data[sym] = {
                    "id": sym,
                    "label": plabel,
                    "type": "parameter",
                    "description": pinfo.get("description", ""),
                    "formula": pinfo.get("formula", ""),
                    "scores": [],
                    "raw_scores": [],
                    "mapped_layers": [],
                    "direction_matches": [],
                    "metric_name": "Sensitivity Gradient dR/dp",
                }

        # Run sweep
        for s in seeds:
            # Capture mapping across seeds
            mapping_res = self.generate_mapping_with_confidence(
                input_tensor=input_tensor,
                sigma=sigma,
                seed=s,
                sensitivity_method=sensitivity_method,
            )
            mapping = mapping_res.get("mapping", {})

            for var_key, vdata in per_var_data.items():
                # Check if var is in mapping
                matched_info = mapping.get(var_key) or mapping.get(vdata.get("id"))
                if matched_info:
                    score = float(matched_info.get("score", 0.0))
                    raw_score = float(matched_info.get("raw_score", 0.0))
                    layer = matched_info.get("layer", "Unknown")
                    vdata["scores"].append(score)
                    vdata["raw_scores"].append(raw_score)
                    vdata["mapped_layers"].append(layer)
                    vdata["direction_matches"].append(
                        matched_info.get("direction_match")
                    )
                else:
                    # If not in mapping, check if it has a perturbation or signature fallback
                    vdata["scores"].append(0.0)
                    vdata["raw_scores"].append(0.0)
                    vdata["mapped_layers"].append("None")
                    vdata["direction_matches"].append(None)

        # Get all neural component names
        all_neural_components = list(self.model.state_dict().keys())

        # Compute summary stats and layer distribution per physical variable
        for var_key, vdata in per_var_data.items():
            arr = np.array(vdata["scores"], dtype=float)
            if len(arr) > 0:
                vdata["mean"] = float(np.mean(arr))
                vdata["std"] = float(np.std(arr))
                vdata["min"] = float(np.min(arr))
                vdata["max"] = float(np.max(arr))
                vdata["median"] = float(np.median(arr))
            else:
                vdata["mean"] = 0.0
                vdata["std"] = 0.0
                vdata["min"] = 0.0
                vdata["max"] = 0.0
                vdata["median"] = 0.0

            # Count mapped layers across seeds for this physical variable
            layer_counts = {comp: 0 for comp in all_neural_components}
            for lyr in vdata["mapped_layers"]:
                if lyr in layer_counts:
                    layer_counts[lyr] += 1
                elif lyr != "None":
                    layer_counts[lyr] = layer_counts.get(lyr, 0) + 1

            vdata["layer_distribution"] = layer_counts
            sorted_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)
            vdata["top_layer"] = sorted_layers[0][0] if sorted_layers else "None"
            vdata["top_layer_count"] = sorted_layers[0][1] if sorted_layers else 0

        return {
            "seeds": seeds,
            "all_neural_components": all_neural_components,
            "physical_variables": per_var_data,
            "summary": {
                "start_seed": int(start_seed),
                "end_seed": int(end_seed),
                "total_seeds": len(seeds),
                "sigma": sigma,
            },
        }

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
        seed: int = None,
    ) -> dict:
        """
        Apply an intervention (mask/perturb) on a specific layer and evaluate.

        Args:
            layer_name        : Name of the tensor to intervene on (from state_dict)
            intervention_type : "mask" (zero out) or "perturb" (add Gaussian noise)
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
        elif intervention_type in ("perturb", "noise"):
            perturbation_seed = self.PERTURBATION_SEED if seed is None else int(seed)
            torch.manual_seed(perturbation_seed)
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
            "baseline_pred": baseline_pred.detach().cpu().numpy(),
            "intervened_pred": intervened_pred.detach().cpu().numpy(),
        }
