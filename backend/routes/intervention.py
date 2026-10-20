"""
intervention.py
===============
FastAPI router for all intervention, sweep, mapping, and physics analysis endpoints:
  POST /api/intervention/run
  POST /api/intervention/sweep
  POST /api/intervention/map
  POST /api/intervention/seed-sweep
  POST /api/intervention/full-analysis
"""

from fastapi import APIRouter, HTTPException
from backend.schemas.intervention_schema import (
    InterventionRunRequest,
    InterventionRunResponse,
    InterventionSweepRequest,
    InterventionSweepResponse,
    MappingRequest,
    MappingResponse,
    SeedSweepRequest,
    SeedSweepResponse,
    FullAnalysisRequest,
    FullAnalysisResponse,
)
from backend.services.intervention_service import (
    run_single_intervention,
    run_intervention_sweep,
    run_node_mapping,
    run_seed_sweep,
    run_full_physics_analysis,
)

router = APIRouter(prefix="/api/intervention", tags=["Intervention"])


@router.post("/run", response_model=InterventionRunResponse)
async def run_intervention(request: InterventionRunRequest):
    """
    Apply a single intervention (mask or perturbation) on a neural layer.

    Returns before/after MSE, relative L2, and the full prediction trajectories
    as flat float arrays ready for Plotly/ECharts.
    """
    try:
        return run_single_intervention(
            session_id=request.session_id,
            layer_name=request.layer_name,
            intervention_type=request.intervention_type,
            strength=request.strength,
            seed=request.seed,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/sweep", response_model=InterventionSweepResponse)
async def run_sweep(request: InterventionSweepRequest):
    """
    Sweep intervention strength monotonically across [0.1 ... 1.0].

    Returns a series of steps with MSE deltas and variance metrics.
    """
    try:
        return run_intervention_sweep(
            session_id=request.session_id,
            layer_name=request.layer_name,
            intervention_type=request.intervention_type,
            strengths=request.strengths,
            seed=request.seed,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/map", response_model=MappingResponse)
async def map_nodes(request: MappingRequest):
    """
    Discover mappings from physical causal nodes to neural network layers
    using trend matching (parameters) and autograd/perturbation sensitivities (states).
    """
    try:
        return run_node_mapping(
            session_id=request.session_id,
            sensitivity_method=request.sensitivity_method,
            seed=request.seed,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/seed-sweep", response_model=SeedSweepResponse)
async def seed_sweep(request: SeedSweepRequest):
    """
    Run perturbation sensitivity sweep across multiple random seeds to evaluate
    mapping stability and layer distribution.
    """
    try:
        return run_seed_sweep(
            session_id=request.session_id,
            start_seed=request.start_seed,
            end_seed=request.end_seed,
            sigma=request.sigma,
            sensitivity_method=request.sensitivity_method,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/full-analysis", response_model=FullAnalysisResponse)
async def full_analysis(request: FullAnalysisRequest):
    """
    Run complete physics analysis: relative L2 norm, mean squared physics
    residual norm, and parameter sensitivity gradients.
    """
    try:
        return run_full_physics_analysis(
            session_id=request.session_id,
            param_values=request.param_values,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
