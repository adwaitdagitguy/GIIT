"""
model_loader.py
===============
Dynamically loads a PyTorch nn.Module from a user-provided .py file.

KEY FEATURE: AST-based pre-processor strips training loops, plt.show(),
print statements, and all top-level execution code before importing.
Users can upload raw Colab scripts or training notebooks without modification.
"""

import importlib.util
import inspect
import sys
import os
import ast
import time
import types
from pathlib import Path

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# AST PRE-PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def _extract_safe_source(source: str) -> str:
    """
    Parse a Python source file and keep ONLY:
      - import statements
      - top-level function definitions (helpers the class may call)
      - class definitions

    Everything else (training loops, plt.show, torch.save, print, etc.)
    is silently discarded before the code is ever executed.
    """
    tree = ast.parse(source)

    safe_node_types = (
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )

    # Keep only safe top-level nodes
    safe_nodes = [node for node in tree.body if isinstance(node, safe_node_types)]

    # Rebuild a clean module AST
    clean_tree = ast.Module(body=safe_nodes, type_ignores=[])
    ast.fix_missing_locations(clean_tree)

    # Unparse back to source string (Python 3.9+)
    clean_source = ast.unparse(clean_tree)
    return clean_source


def _import_module_from_source(module_name: str, clean_source: str):
    """
    Execute a clean source string as a Python module and return it.
    """
    module = types.ModuleType(module_name)
    module.__file__ = f"<stripped:{module_name}>"
    sys.modules[module_name] = module
    exec(compile(clean_source, module_name, "exec"), module.__dict__)
    return module


def _find_model_classes(module) -> list:
    """
    Scan a module for all classes that are subclasses of nn.Module
    (but NOT nn.Module itself).
    Returns list of (name, class) tuples.
    """
    classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, nn.Module) and obj is not nn.Module:
            classes.append((name, obj))
    return classes


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_model_from_file(
    py_path: str,
    weights_path: str = None,
    class_name: str = None,
) -> nn.Module:
    """
    Load an nn.Module from ANY user-provided .py file.
    Training loops, plots, and top-level execution code are automatically stripped.

    Args:
        py_path      : Path to the .py file containing the model class.
        weights_path : Path to a .pt file with saved weights or full model.
        class_name   : Name of the nn.Module subclass to use (auto-detect if None).

    Returns:
        An nn.Module instance in eval mode on CPU.
    """
    start_total = time.time()
    print("\n" + "=" * 60)
    print("MODEL LOADER")
    print("=" * 60)
    print(f"  Source file  : {py_path}")
    print(f"  Weights file : {weights_path or '(none)'}")

    # ── Step 0: Try loading a full model .pt directly ─────────────────────────
    if weights_path and os.path.isfile(weights_path):
        t0 = time.time()
        try:
            obj = torch.load(weights_path, weights_only=False, map_location="cpu")
            if isinstance(obj, nn.Module):
                obj.eval()
                print(f"  [OK] Full model object loaded in {time.time()-t0:.3f}s")
                print(f"  [DONE] Total: {time.time()-start_total:.3f}s")
                print("=" * 60)
                return obj
        except Exception:
            pass  # Fall through to state_dict path

    # ── Step 1: Read source ───────────────────────────────────────────────────
    with open(py_path, "r", encoding="utf-8") as f:
        raw_source = f.read()
    print(f"  [OK] Source read ({len(raw_source.splitlines())} lines)")

    # ── Step 2: AST strip — remove all top-level execution code ──────────────
    t0 = time.time()
    clean_source = _extract_safe_source(raw_source)
    clean_lines = len(clean_source.splitlines())
    print(f"  [OK] AST stripped to {clean_lines} lines in {time.time()-t0:.4f}s")

    # ── Step 3: Execute clean source ─────────────────────────────────────────
    t0 = time.time()
    module_name = f"_user_model_{Path(py_path).stem}"
    module = _import_module_from_source(module_name, clean_source)
    print(f"  [OK] Module executed in {time.time()-t0:.4f}s")

    # ── Step 4: Find nn.Module class ─────────────────────────────────────────
    candidates = _find_model_classes(module)
    if not candidates:
        raise RuntimeError(
            f"No nn.Module subclass found in {py_path}. "
            f"Make sure your model class inherits from torch.nn.Module."
        )

    if class_name:
        match = [c for c in candidates if c[0] == class_name]
        if not match:
            available = [c[0] for c in candidates]
            raise RuntimeError(f"Class '{class_name}' not found. Available: {available}")
        chosen_name, chosen_class = match[0]
    else:
        chosen_name, chosen_class = candidates[0]
        if len(candidates) > 1:
            names = [c[0] for c in candidates]
            print(f"  [!] Multiple classes found: {names}. Using first: {chosen_name}")

    print(f"  [OK] Found class: {chosen_name}")

    # ── Step 5: Instantiate ───────────────────────────────────────────────────
    t0 = time.time()
    try:
        model = chosen_class()
    except TypeError as e:
        raise RuntimeError(
            f"Cannot instantiate {chosen_name}() with no arguments: {e}. "
            f"Add default values to __init__ — e.g., "
            f"def __init__(self, N_INPUT=1, N_OUTPUT=1, N_HIDDEN=32, N_LAYERS=3)"
        )
    print(f"  [OK] Instantiated in {time.time()-t0:.4f}s")

    # ── Step 6: Load state_dict ───────────────────────────────────────────────
    if weights_path and os.path.isfile(weights_path):
        t0 = time.time()
        state = torch.load(weights_path, weights_only=False, map_location="cpu")
        if isinstance(state, dict):
            model.load_state_dict(state)
        else:
            model.load_state_dict(state.state_dict())
        print(f"  [OK] Weights injected in {time.time()-t0:.3f}s")
    elif weights_path:
        print(f"  [!] Weights file not found: {weights_path}. Using random init.")

    model.eval()
    print(f"  [DONE] Total loading time: {time.time()-start_total:.3f}s")
    print("=" * 60)
    return model


def read_model_source(py_path: str) -> str:
    """Read raw source code of the model file (used by LLM graph generator)."""
    with open(py_path, "r", encoding="utf-8") as f:
        return f.read()


def get_model_architecture_summary(model: nn.Module) -> dict:
    """Return a summary dict of layer shapes, input/output dims, and param count."""
    layers = []
    for name, tensor in model.state_dict().items():
        layers.append({"name": name, "shape": list(tensor.shape), "numel": tensor.numel()})

    weight_layers = [l for l in layers if "weight" in l["name"]]
    input_dim  = weight_layers[0]["shape"][1] if weight_layers else None
    output_dim = weight_layers[-1]["shape"][0] if weight_layers else None

    return {
        "class_name":   model.__class__.__name__,
        "total_params": sum(p.numel() for p in model.parameters()),
        "input_dim":    input_dim,
        "output_dim":   output_dim,
        "layers":       layers,
    }
