"""
main.py
=======
FastAPI application entry point for the SciML Inspection & Intervention backend.

Configures:
  - CORS middleware (allowing local Vite dev server and production frontend origins)
  - Routers for model, graph, ruleset, and intervention endpoints
  - Health check endpoint at GET /
  - Session listing endpoint at GET /api/sessions
"""

import os
import sys

# Ensure repository root is on sys.path regardless of where uvicorn is launched
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch

from backend.routes.model import router as model_router
from backend.routes.graph import router as graph_router
from backend.routes.ruleset import router as ruleset_router
from backend.routes.intervention import router as intervention_router
from backend.session_manager import session_store

app = FastAPI(
    title="SciML Inspection & Intervention API",
    description=(
        "Decoupled REST API backend for inspecting, mapping, and intervening on "
        "Physics-Informed Neural Networks (PINNs) and SciML surrogate models."
    ),
    version="1.0.0",
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
# Allow requests from local frontend development servers (Vite default: 5173/3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "*",  # Permissive for academic local deployment
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount Routers ─────────────────────────────────────────────────────────────
app.include_router(model_router)
app.include_router(graph_router)
app.include_router(ruleset_router)
app.include_router(intervention_router)


# ── System / Health Endpoints ─────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    """Health check and API info, including active GPU/CPU compute device."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    return {
        "status": "online",
        "service": "SciML Inspection & Intervention API",
        "version": "1.0.0",
        "device": "cuda" if cuda_available else "cpu",
        "gpu_name": device_name,
        "docs_url": "/docs",
    }


@app.get("/api/sessions", tags=["Session"])
async def list_sessions():
    """List all active model sessions currently loaded in memory."""
    return {
        "sessions": session_store.list_sessions(),
        "total_active": len(session_store.list_sessions()),
    }


@app.delete("/api/session/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    """Release in-memory session data for a given session ID."""
    session_store.delete(session_id)
    return {"session_id": session_id, "deleted": True}
