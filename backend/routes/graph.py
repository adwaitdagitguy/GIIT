"""
graph.py
========
FastAPI router for causal graph retrieval and custom quantity operations.
"""

from fastapi import APIRouter, HTTPException, Query
from backend.schemas.graph_schema import (
    GraphResponse,
    CustomQuantityRequest,
    CustomQuantityResponse,
)
from backend.services.graph_service import (
    get_react_flow_graph,
    add_custom_quantity,
)

router = APIRouter(prefix="/api/graph", tags=["Graph"])


@router.get("", response_model=GraphResponse)
async def get_graph(
    session_id: str = Query(..., description="Active session UUID"),
):
    """
    Retrieve the causal graph formatted for React Flow, with deterministic
    positions and node explanations.
    """
    try:
        return get_react_flow_graph(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/custom-quantity", response_model=CustomQuantityResponse)
async def create_custom_quantity(request: CustomQuantityRequest):
    """
    Define a composite physical observable (e.g. u * u_x) and add it to
    the intervener's derivative spec and graph nodes.
    """
    try:
        return add_custom_quantity(
            session_id=request.session_id,
            name=request.name,
            vars_to_combine=request.vars,
            operation=request.operation,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
