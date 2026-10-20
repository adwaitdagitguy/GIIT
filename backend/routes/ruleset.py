"""
ruleset.py
==========
FastAPI router for physics ruleset upload and LLM generation.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from backend.schemas.ruleset_schema import (
    RulesetUploadResponse,
    RulesetGenerateRequest,
    RulesetGenerateResponse,
)
from backend.services.ruleset_service import (
    handle_ruleset_upload,
    handle_ruleset_generate,
)

router = APIRouter(prefix="/api/ruleset", tags=["Ruleset"])


@router.post("/upload", response_model=RulesetUploadResponse)
async def upload_ruleset(
    session_id: str = Query(..., description="Active session UUID"),
    rules_file: UploadFile = File(..., description="Physics ruleset JSON (.json)"),
):
    """Upload and attach a physics ruleset JSON to the session intervener."""
    try:
        ruleset_bytes = await rules_file.read()
        return handle_ruleset_upload(session_id, ruleset_bytes)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/generate", response_model=RulesetGenerateResponse)
async def generate_ruleset(request: RulesetGenerateRequest):
    """
    Call LLM to suggest derived quantity compute functions for the active session's graph.
    """
    try:
        return handle_ruleset_generate(
            session_id=request.session_id,
            api_key=request.api_key,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ruleset generation failed: {exc}")
