"""
model.py
========
FastAPI router for model upload and architecture metadata endpoints.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from backend.schemas.model_schema import ModelUploadResponse, ArchitectureSummary
from backend.services.model_service import handle_model_upload, get_architecture_summary

router = APIRouter(prefix="/api/model", tags=["Model"])


@router.post("/upload", response_model=ModelUploadResponse)
async def upload_model(
    model_py: UploadFile = File(..., description="PyTorch model definition (.py)"),
    model_pt: UploadFile = File(..., description="Trained model weights (.pt)"),
    graph_json: UploadFile = File(..., description="Causal graph specification (.json)"),
):
    """
    Upload a PINN model definition, weights, and causal graph JSON.

    Initializes the AST-sanitized model, generates test data, constructs
    the dynamic intervener, and registers the session.
    """
    try:
        py_bytes = await model_py.read()
        pt_bytes = await model_pt.read()
        graph_bytes = await graph_json.read()

        result = handle_model_upload(
            py_bytes=py_bytes,
            pt_bytes=pt_bytes,
            graph_bytes=graph_bytes,
            filename_py=model_py.filename or "model.py",
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}")


@router.get("/architecture", response_model=ArchitectureSummary)
async def get_model_architecture(
    session_id: str = Query(..., description="Active session UUID"),
):
    """Retrieve layer details and parameter counts for the active session's model."""
    try:
        return get_architecture_summary(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
