"""
model_schema.py
===============
Pydantic schemas for the model upload and architecture endpoints.
"""

from pydantic import BaseModel
from typing import Optional


class ModelUploadResponse(BaseModel):
    """Returned by POST /api/model/upload on success."""

    session_id: str
    system: str             # metadata.system from graph.json
    equation: str           # metadata.equation from graph.json
    num_parameters: int     # total learnable parameters
    architecture: dict      # output of get_model_architecture_summary()
    input_spec: dict        # graph["input_spec"]
    output_spec: dict       # graph["output_spec"]
    physics_params: list    # list of parameter symbol names
    test_input_shape: list  # e.g. [500, 1]
    has_ground_truth: bool  # whether test_targets is not None


class ArchitectureSummary(BaseModel):
    """Lightweight architecture info, embeddable in other responses."""

    num_layers: int
    layer_details: list     # list of {name, type, shape, params}
    total_params: int
    trainable_params: int


class SessionInfoResponse(BaseModel):
    """Summary returned by GET /api/session/{session_id}."""

    session_id: str
    system: str
    has_ruleset: bool
    has_mapping: bool
    created_at: float
