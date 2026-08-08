"""
llm_graph_generator.py
======================
Reads a user-provided .py file containing a physics-informed ML model,
sends it to an LLM (via OpenRouter), and produces a structured graph.json
describing the physics dependencies, residual formulas, parameters, etc.

The generated graph.json is consumed by dynamic_intervener.py to perform
all intervention analyses without any hardcoding.

LLM Backend: OpenRouter API (OpenAI-compatible)
Default model: google/gemma-4-31b-it

Usage:
    from llm_graph_generator import generate_graph_json

    graph = generate_graph_json(
        model_py_path="my_pinn.py",
        api_key="sk-or-...",
        output_path="graph.json"
    )
"""

import json
import os
import re
import time
import requests


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-31b-it"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2


# ══════════════════════════════════════════════════════════════════════════════
# JSON SCHEMA VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_TOP_LEVEL_KEYS = {"metadata", "nodes", "edges", "parameters", "residual",
                            "expected_trends", "input_spec", "output_spec"}

REQUIRED_NODE_KEYS = {"id", "label", "symbol", "type"}
VALID_NODE_TYPES = {"parameter", "intermediate", "output", "input"}

REQUIRED_PARAM_KEYS = {"label", "default_value", "expected_sensitivity_sign", "trend"}
VALID_SIGNS = {"+", "-", "~0", "~"}

REQUIRED_RESIDUAL_KEYS = {"formula", "derivatives"}


def validate_graph_json(graph: dict) -> list:
    """
    Validate the generated graph.json against the expected schema.
    Returns a list of error strings. Empty list = valid.
    """
    errors = []

    # Top-level keys
    missing = REQUIRED_TOP_LEVEL_KEYS - set(graph.keys())
    if missing:
        errors.append(f"Missing top-level keys: {missing}")

    # Nodes
    if "nodes" in graph:
        if not isinstance(graph["nodes"], list):
            errors.append("'nodes' must be a list")
        else:
            for i, node in enumerate(graph["nodes"]):
                for key in REQUIRED_NODE_KEYS:
                    if key not in node:
                        errors.append(f"Node [{i}] missing key '{key}'")
                if node.get("type") not in VALID_NODE_TYPES:
                    errors.append(f"Node [{i}] has invalid type: {node.get('type')}")

    # Edges
    if "edges" in graph:
        if not isinstance(graph["edges"], list):
            errors.append("'edges' must be a list")
        else:
            for i, edge in enumerate(graph["edges"]):
                if "source" not in edge or "target" not in edge:
                    errors.append(f"Edge [{i}] missing 'source' or 'target'")

    # Parameters
    if "parameters" in graph:
        if not isinstance(graph["parameters"], dict):
            errors.append("'parameters' must be a dict")
        else:
            for sym, info in graph["parameters"].items():
                for key in REQUIRED_PARAM_KEYS:
                    if key not in info:
                        errors.append(f"Parameter '{sym}' missing key '{key}'")
                sign = info.get("expected_sensitivity_sign", "")
                if sign not in VALID_SIGNS:
                    errors.append(
                        f"Parameter '{sym}' has invalid sign '{sign}'. "
                        f"Valid: {VALID_SIGNS}"
                    )

    # Residual
    if "residual" in graph:
        res = graph["residual"]
        for key in REQUIRED_RESIDUAL_KEYS:
            if key not in res:
                errors.append(f"'residual' missing key '{key}'")
        if "formula" in res and not isinstance(res["formula"], str):
            errors.append("'residual.formula' must be a string")
        if "derivatives" in res and not isinstance(res["derivatives"], dict):
            errors.append("'residual.derivatives' must be a dict")

    # Expected trends
    if "expected_trends" in graph:
        if not isinstance(graph["expected_trends"], dict):
            errors.append("'expected_trends' must be a dict")

    # Input/output specs
    for spec_name in ["input_spec", "output_spec"]:
        if spec_name in graph:
            spec = graph[spec_name]
            if "names" not in spec or "dim" not in spec:
                errors.append(f"'{spec_name}' must have 'names' and 'dim'")

    return errors


# ══════════════════════════════════════════════════════════════════════════════
# FEW-SHOT EXAMPLES (embedded in the prompt)
# ══════════════════════════════════════════════════════════════════════════════

EXAMPLE_DHO_INPUT = '''
class PINN(nn.Module):
    """1D PINN for Damped Harmonic Oscillator: input=t, output=x(t)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, t):
        return self.net(t)
'''

EXAMPLE_DHO_OUTPUT = '''{
  "metadata": {
    "system": "1D Damped Harmonic Oscillator",
    "equation": "m*x'' + c*x' + k*x = 0",
    "description": "A mass-spring-damper system modeled as a second-order ODE. The PINN takes time t as input and predicts displacement x(t).",
    "pinn_input": "t (time)",
    "pinn_output": "x(t) (displacement)"
  },
  "nodes": [
    {"id": "m", "label": "Mass", "symbol": "m", "type": "parameter", "unit": "kg", "description": "Inertial resistance to acceleration"},
    {"id": "c", "label": "Damping Coefficient", "symbol": "c", "type": "parameter", "unit": "N*s/m", "description": "Energy dissipation coefficient"},
    {"id": "k", "label": "Spring Constant", "symbol": "k", "type": "parameter", "unit": "N/m", "description": "Restoring force coefficient"},
    {"id": "x", "label": "Displacement", "symbol": "x", "type": "output", "unit": "m", "description": "System displacement predicted by PINN"},
    {"id": "x_dot", "label": "Velocity", "symbol": "x_t", "type": "intermediate", "unit": "m/s", "description": "First time derivative of displacement"},
    {"id": "x_ddot", "label": "Acceleration", "symbol": "x_tt", "type": "intermediate", "unit": "m/s^2", "description": "Second time derivative of displacement"}
  ],
  "edges": [
    {"source": "m", "target": "x_ddot", "label": "scales inertia"},
    {"source": "c", "target": "x_dot", "label": "damps velocity"},
    {"source": "k", "target": "x", "label": "restoring force"},
    {"source": "x_dot", "target": "x_ddot", "label": "feeds into ODE"},
    {"source": "x", "target": "x_ddot", "label": "feeds into ODE"},
    {"source": "x_ddot", "target": "x", "label": "integration over time"}
  ],
  "input_spec": {
    "names": ["t"],
    "dim": 1,
    "ranges": {"t": [0, 10]}
  },
  "output_spec": {
    "names": ["x"],
    "dim": 1
  },
  "parameters": {
    "m": {
      "label": "Mass",
      "default_value": 1.0,
      "expected_sensitivity_sign": "+",
      "trend": "Increasing mass slows decay; amplitude persists longer (under-damped regime)"
    },
    "c": {
      "label": "Damping Coefficient",
      "default_value": 0.2,
      "expected_sensitivity_sign": "-",
      "trend": "Increasing damping accelerates amplitude decay; system loses energy faster"
    },
    "k": {
      "label": "Spring Constant",
      "default_value": 1.0,
      "expected_sensitivity_sign": "+",
      "trend": "Increasing spring constant raises natural frequency; minor effect on amplitude"
    }
  },
  "residual": {
    "formula": "m * x_tt + c * x_t + k * x",
    "description": "ODE residual: m*x'' + c*x' + k*x should equal 0",
    "derivatives": {
      "x": {"wrt": ["t"], "order": 0},
      "x_t": {"wrt": ["t"], "order": 1},
      "x_tt": {"wrt": ["t"], "order": 2}
    }
  },
  "expected_trends": {
    "Damping Coefficient": "decreases_amplitude",
    "Mass": "increases_amplitude",
    "Spring Constant": "increases_frequency"
  },
  "test_data_generator": "import numpy as np\\ndef generate(params):\\n    m, c, k = params['m'], params['c'], params['k']\\n    t = np.linspace(0, 10, 1000)\\n    wn = np.sqrt(k / m)\\n    zeta = c / (2 * np.sqrt(m * k))\\n    wd = wn * np.sqrt(1 - zeta**2)\\n    x0, v0 = 1.0, 0.0\\n    A, B = x0, (v0 + zeta * wn * x0) / wd\\n    x = np.exp(-zeta * wn * t) * (A * np.cos(wd * t) + B * np.sin(wd * t))\\n    return t.reshape(-1, 1), x.reshape(-1, 1)"
}'''

EXAMPLE_BURGERS_INPUT = '''
class BurgersPINN(nn.Module):
    """
    1D Burgers' equation PINN.
    Input : [x, t]  (spatial coord + time)
    Output: u(x, t) (velocity field)
    Equation: du/dt + u*du/dx = nu * d2u/dx2
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, xt):
        return self.net(xt)
'''

EXAMPLE_BURGERS_OUTPUT = '''{
  "metadata": {
    "system": "1D Viscous Burgers Equation",
    "equation": "du/dt + u * du/dx = nu * d2u/dx2",
    "description": "1D viscous Burgers equation PDE. The PINN takes spatial coordinate x and time t as input, outputs velocity field u(x,t).",
    "pinn_input": "[x, t] (spatial coordinate + time)",
    "pinn_output": "u(x, t) (velocity field)"
  },
  "nodes": [
    {"id": "nu", "label": "Kinematic Viscosity", "symbol": "nu", "type": "parameter", "unit": "m^2/s", "description": "Controls diffusion/smoothing of the velocity field"},
    {"id": "convection", "label": "Convection", "symbol": "u", "type": "parameter", "unit": "", "description": "Nonlinear advection term u*du/dx"},
    {"id": "u_out", "label": "Velocity Field", "symbol": "u", "type": "output", "unit": "m/s", "description": "Velocity field predicted by PINN"},
    {"id": "diss", "label": "Energy Dissipation", "symbol": "nu_d2u", "type": "intermediate", "unit": "", "description": "Viscous dissipation term nu*d2u/dx2"}
  ],
  "edges": [
    {"source": "nu", "target": "diss", "label": "viscous dissipation"},
    {"source": "convection", "target": "u_out", "label": "nonlinear advection"},
    {"source": "diss", "target": "u_out", "label": "smoothing effect"}
  ],
  "input_spec": {
    "names": ["x", "t"],
    "dim": 2,
    "ranges": {"x": [-1, 1], "t": [0, 1]}
  },
  "output_spec": {
    "names": ["u"],
    "dim": 1
  },
  "parameters": {
    "nu": {
      "label": "Kinematic Viscosity",
      "default_value": 0.01,
      "expected_sensitivity_sign": "-",
      "trend": "Increasing viscosity smooths the shock wave; sharp gradients dissipate"
    }
  },
  "residual": {
    "formula": "u_t + u_out * u_x - nu * u_xx",
    "description": "PDE residual: du/dt + u*du/dx - nu*d2u/dx2 should equal 0",
    "derivatives": {
      "u_out": {"wrt": ["x", "t"], "order": 0},
      "u_x": {"wrt": ["x"], "order": 1, "of": "u_out"},
      "u_t": {"wrt": ["t"], "order": 1, "of": "u_out"},
      "u_xx": {"wrt": ["x"], "order": 2, "of": "u_out"}
    }
  },
  "expected_trends": {
    "Kinematic Viscosity": "decreases_amplitude",
    "Convection": "increases_amplitude"
  },
  "test_data_generator": "import numpy as np\\ndef generate(params):\\n    nu = params.get('nu', 0.01)\\n    x_space = np.linspace(-1, 1, 256)\\n    t_time = np.linspace(0, 1, 100)\\n    xv, tv = np.meshgrid(x_space, t_time)\\n    u_true = np.exp(-xv**2) * np.exp(-nu * tv)\\n    xt_flat = np.column_stack([xv.ravel(), tv.ravel()])\\n    u_flat = u_true.ravel()\\n    return xt_flat, u_flat.reshape(-1, 1)"
}'''


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an expert in physics-informed neural networks (PINNs) and scientific machine learning.

Your task: Given a PyTorch model definition (.py file), analyze the physics it encodes and produce a structured JSON describing the physical system, its causal graph, residual formula, parameters, and expected sensitivities.

IMPORTANT RULES:
1. The "residual.formula" must be a valid Python expression using only: variable names from "residual.derivatives" keys, parameter symbols from "parameters" keys, and basic math operators (+, -, *, /, **). Do NOT use any function calls -- only arithmetic.
2. Each derivative in "residual.derivatives" must specify:
   - "wrt": list of input variable names to differentiate with respect to
   - "order": integer derivative order
   - "of": (optional) which output variable this is a derivative of (defaults to model output)
3. For "order": 0, the variable is the raw model output (no differentiation needed).
4. Parameter symbols in "parameters" must match exactly what appears in "residual.formula".
5. The "expected_sensitivity_sign" is the sign of d(mean_residual²)/d(parameter). Use "+" if increasing the parameter increases the residual norm, "-" if it decreases it, "~0" if negligible.
6. "expected_trends" maps the LABEL (not symbol) of each parameter to one of: "decreases_amplitude", "increases_amplitude", "increases_frequency", "decreases_frequency".
7. "test_data_generator" should be a Python code string that defines a function `generate(params)` which takes a dict of parameter default values and returns (input_array, output_array) as numpy arrays. Use the analytical solution if one exists, otherwise generate reasonable synthetic data.
8. Output ONLY valid JSON. No markdown, no code fences, no commentary before or after the JSON.
9. The derivative variable names in "residual.derivatives" keys should use underscores for subscripts. E.g., for du/dx use "u_x", for d2u/dx2 use "u_xx", for dx/dt use "x_t".
10. For multi-input models (e.g., input is [x, t]), derivatives "wrt" should reference the specific input column name.
"""


# ══════════════════════════════════════════════════════════════════════════════
# LLM API CALL
# ══════════════════════════════════════════════════════════════════════════════

def _call_openrouter(
    api_key: str,
    model: str,
    messages: list,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> str:
    """
    Call the OpenRouter API (OpenAI-compatible format).
    Returns the assistant message content as a string.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/sciml-intervener",
        "X-Title": "SciML Dynamic Graph Generator",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API error {response.status_code}: {response.text}"
        )

    data = response.json()

    if "error" in data:
        raise RuntimeError(f"OpenRouter API error: {data['error']}")

    return data["choices"][0]["message"]["content"]


def _extract_json_from_response(text: str) -> dict:
    """
    Parse JSON from the LLM response. Handles cases where the LLM
    wraps the JSON in markdown code fences.
    """
    # Strip markdown code fences if present
    text = text.strip()

    # Try to find JSON within code fences
    fence_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(fence_pattern, text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # Also handle case where response starts with text before JSON
    brace_start = text.find('{')
    if brace_start > 0:
        text = text[brace_start:]

    # Find the matching closing brace
    depth = 0
    end_idx = -1
    for i, ch in enumerate(text):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                end_idx = i
                break

    if end_idx > 0:
        text = text[:end_idx + 1]

    return json.loads(text)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def generate_graph_json(
    model_py_path: str,
    api_key: str,
    output_path: str = None,
    model_name: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> dict:
    """
    Read a user's PINN .py file, send it to the LLM, and produce graph.json.

    Args:
        model_py_path : Path to the .py file containing the PINN model.
        api_key       : OpenRouter API key.
        output_path   : Where to save the generated JSON. If None, auto-names
                        based on the .py filename.
        model_name    : OpenRouter model identifier.
        verbose       : Print progress to stdout.

    Returns:
        The parsed graph dict.
    """
    if output_path is None:
        base = os.path.splitext(os.path.basename(model_py_path))[0]
        output_path = f"{base}_graph.json"

    if verbose:
        print("\n" + "=" * 60)
        print("LLM GRAPH GENERATOR")
        print("=" * 60)
        print(f"  Model file    : {model_py_path}")
        print(f"  LLM model     : {model_name}")
        print(f"  Output        : {output_path}")

    # Read the user's source code
    with open(model_py_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    if verbose:
        lines = source_code.count('\n') + 1
        print(f"  Source lines  : {lines}")
        print()

    # Build the few-shot prompt
    user_prompt = f"""Here are two examples of model code and their corresponding graph JSON:

--- EXAMPLE 1: Damped Harmonic Oscillator ---
MODEL CODE:
{EXAMPLE_DHO_INPUT}

GENERATED JSON:
{EXAMPLE_DHO_OUTPUT}

--- EXAMPLE 2: Burgers' Equation ---
MODEL CODE:
{EXAMPLE_BURGERS_INPUT}

GENERATED JSON:
{EXAMPLE_BURGERS_OUTPUT}

--- YOUR TASK ---
Analyze the following model code and generate the graph JSON in exactly the same format.

MODEL CODE:
```python
{source_code}
```

Generate the complete graph JSON now. Output ONLY the JSON, nothing else."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Retry loop
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        if verbose:
            print(f"  [Attempt {attempt}/{MAX_RETRIES}] Calling LLM...")

        try:
            response_text = _call_openrouter(
                api_key=api_key,
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=4096,
            )

            if verbose:
                print(f"  [OK] Received response ({len(response_text)} chars)")

            # Parse JSON
            graph = _extract_json_from_response(response_text)

            # Validate
            errors = validate_graph_json(graph)
            if errors:
                error_msg = "\n".join(f"    - {e}" for e in errors)
                if verbose:
                    print(f"  [!] Validation errors:\n{error_msg}")

                # If it's the last attempt, try to fix by asking LLM to correct
                if attempt < MAX_RETRIES:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"The JSON you produced has these validation errors:\n{error_msg}\n\n"
                            f"Please fix them and output the corrected JSON. Output ONLY the JSON."
                        )
                    })
                    last_error = f"Validation failed: {error_msg}"
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                else:
                    # Use it anyway on last attempt (partial is better than nothing)
                    print(f"  [!!] Using graph with validation warnings after {MAX_RETRIES} attempts.")
            else:
                if verbose:
                    print(f"  [OK] Validation passed!")

            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(graph, f, indent=2)

            if verbose:
                print(f"\n  [OK] Saved graph to: {output_path}")
                _print_graph_summary(graph)
                print("=" * 60)

            return graph

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            if verbose:
                print(f"  [!] {last_error}")
            if attempt < MAX_RETRIES:
                # Ask LLM to fix its output
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your response was not valid JSON. Error: {e}\n"
                        f"Please output ONLY a valid JSON object with no markdown formatting."
                    )
                })
                time.sleep(RETRY_DELAY_SECONDS)
            continue

        except Exception as e:
            last_error = f"API error: {e}"
            if verbose:
                print(f"  [!] {last_error}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
            continue

    raise RuntimeError(
        f"Failed to generate valid graph.json after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )


def _print_graph_summary(graph: dict):
    """Pretty-print a summary of the generated graph."""
    meta = graph.get("metadata", {})
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    params = graph.get("parameters", {})
    residual = graph.get("residual", {})

    print(f"\n  +---------------------------------------------------+")
    print(f"  |  GENERATED GRAPH SUMMARY                          |")
    print(f"  +---------------------------------------------------+")
    print(f"  System      : {meta.get('system', '?')}")
    print(f"  Equation    : {meta.get('equation', '?')}")
    print(f"  PINN Input  : {meta.get('pinn_input', '?')}")
    print(f"  PINN Output : {meta.get('pinn_output', '?')}")
    print(f"  Nodes       : {len(nodes)}")
    for n in nodes:
        print(f"    [{n.get('type', '?'):11s}] {n.get('symbol', '?'):6s}  {n.get('label', '?')}")
    print(f"  Edges       : {len(edges)}")
    for e in edges:
        print(f"    {e['source']:12s} -> {e['target']:12s}  ({e.get('label', '')})")
    print(f"  Parameters  : {len(params)}")
    for sym, info in params.items():
        print(f"    {sym:6s}  default={info.get('default_value', '?')}  "
              f"sign={info.get('expected_sensitivity_sign', '?')}  "
              f"{info.get('label', '')}")
    print(f"  Residual    : {residual.get('formula', '?')}")
    derivs = residual.get("derivatives", {})
    print(f"  Derivatives : {len(derivs)}")
    for dname, dinfo in derivs.items():
        print(f"    {dname:8s}  wrt={dinfo.get('wrt', '?')}  order={dinfo.get('order', '?')}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate physics graph JSON from a PINN model .py file using LLM"
    )
    parser.add_argument("model_py", help="Path to the .py file containing the PINN model")
    parser.add_argument("--api-key", required=True, help="OpenRouter API key")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path (default: <model>_graph.json)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    generate_graph_json(
        model_py_path=args.model_py,
        api_key=args.api_key,
        output_path=args.output,
        model_name=args.model,
        verbose=not args.quiet,
    )
