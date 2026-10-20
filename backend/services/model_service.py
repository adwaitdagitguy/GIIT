"""
model_service.py
================
Service layer for model upload, loading, and architecture inspection.

Wraps existing functions from:
  - core/model_loader.py : load_model_from_file, get_model_architecture_summary
  - core/dynamic_intervener.py : DynamicSciMLIntervener, generate_test_data
"""

import json
import os
import sys
import tempfile
import torch

from core.model_loader import load_model_from_file, get_model_architecture_summary
from core.dynamic_intervener import DynamicSciMLIntervener, generate_test_data
from backend.session_manager import ModelSession, session_store


def handle_model_upload(
    py_bytes: bytes,
    pt_bytes: bytes,
    graph_bytes: bytes,
    filename_py: str = "model.py",
) -> dict:
    """
    Load an uploaded PINN model + weights + graph.json, instantiate the
    intervener, generate synthetic test data, and store everything in the
    in-memory session store.

    Args:
        py_bytes    : Raw bytes of the Python model definition file.
        pt_bytes    : Raw bytes of the PyTorch .pt weights file.
        graph_bytes : Raw bytes of the graph.json file.
        filename_py : Original filename (used to pick the temp file suffix).

    Returns:
        dict ready for ModelUploadResponse serialization.
    """
    # 1. Parse graph JSON
    try:
        graph_dict = json.loads(graph_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid graph.json: {exc}") from exc

    # 2. Write uploaded files to temp paths for loader ingestion
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f_py:
        f_py.write(py_bytes)
        py_path = f_py.name

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f_pt:
        f_pt.write(pt_bytes)
        pt_path = f_pt.name

    try:
        # 3. Load model using AST sanitizer from core/model_loader.py (auto-places on GPU if available)
        model = load_model_from_file(py_path=py_path, weights_path=pt_path)
        device = next(model.parameters(), torch.empty(0)).device

        # 4. Extract architecture summary
        arch_summary = get_model_architecture_summary(model)

        # 5. Instantiate DynamicSciMLIntervener from core/dynamic_intervener.py
        intervener = DynamicSciMLIntervener(model=model, graph=graph_dict)

        # 6. Generate synthetic test data via code generator in graph.json and move to GPU/device
        test_inputs, test_targets = generate_test_data(graph_dict)
        test_inputs = test_inputs.to(device)
        if test_targets is not None:
            test_targets = test_targets.to(device)

        # 7. Create and store ModelSession
        session_id = session_store.new_id()
        session = ModelSession(
            session_id=session_id,
            model=model,
            model_metadata=graph_dict.get("metadata", {}),
            graph_data=graph_dict,
            intervener=intervener,
            test_inputs=test_inputs,
            test_targets=test_targets,
        )
        session_store.create(session)

    finally:
        # Clean up temporary files from disk
        for p in (py_path, pt_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass

    metadata = graph_dict.get("metadata", {})
    return {
        "session_id": session_id,
        "system": metadata.get("system", "Unknown"),
        "equation": metadata.get("equation", "Unknown"),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "architecture": arch_summary,
        "input_spec": graph_dict.get("input_spec", {}),
        "output_spec": graph_dict.get("output_spec", {}),
        "physics_params": list(graph_dict.get("parameters", {}).keys()),
        "test_input_shape": list(test_inputs.shape),
        "has_ground_truth": test_targets is not None,
    }


def get_architecture_summary(session_id: str) -> dict:
    """Retrieve the architecture summary for an existing session."""
    session = session_store.get(session_id)
    return get_model_architecture_summary(session.model)
