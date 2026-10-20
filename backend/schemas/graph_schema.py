"""
graph_schema.py
===============
Pydantic schemas for the graph retrieval and custom-quantity endpoints.

The GET /api/graph response serialises the physics causal graph in a
React Flow-compatible node/edge format so the frontend can render it
without NetworkX.
"""

from pydantic import BaseModel
from typing import Optional, Any


class ReactFlowPosition(BaseModel):
    x: float
    y: float


class ReactFlowNode(BaseModel):
    """A single node for rendering in React Flow."""

    id: str
    type: str           # "input" | "output" | "intermediate" | "parameter"
    data: dict          # {"label": str, "symbol": str, "explanation": dict}
    position: ReactFlowPosition
    style: Optional[dict] = None


class ReactFlowEdge(BaseModel):
    """A directed edge for rendering in React Flow."""

    id: str
    source: str
    target: str
    animated: bool = False


class GraphResponse(BaseModel):
    """Returned by GET /api/graph?session_id=..."""

    session_id: str
    nodes: list[ReactFlowNode]
    edges: list[ReactFlowEdge]
    metadata: dict
    node_explanations: dict    # node_id -> explanation dict from build_node_explanations()


class CustomQuantityRequest(BaseModel):
    """Body for POST /api/graph/custom-quantity."""

    session_id: str
    name: str                   # e.g. "Q_custom"
    vars: list[str]             # base quantities to combine, e.g. ["u", "u_x"]
    operation: str = "add"      # "add" | "multiply"


class CustomQuantityResponse(BaseModel):
    """Returned by POST /api/graph/custom-quantity."""

    session_id: str
    name: str
    operation: str
    vars: list[str]
    added_to_graph: bool
