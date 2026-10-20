"""
intervention_schema.py
======================
Pydantic schemas for all intervention-related endpoints:
  POST /api/intervention/run
  POST /api/intervention/sweep
  POST /api/intervention/map
  POST /api/intervention/seed-sweep
  POST /api/intervention/full-analysis
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


# ── /api/intervention/run ─────────────────────────────────────────────────────

class InterventionRunRequest(BaseModel):
    """Body for POST /api/intervention/run."""

    session_id: str
    layer_name: str                         # state_dict key, e.g. "fch.1.0.weight"
    intervention_type: str = "mask"         # "mask" | "perturb"
    strength: float = Field(0.5, ge=0.0, le=1.0)
    seed: Optional[int] = None


class InterventionRunResponse(BaseModel):
    """Returned by POST /api/intervention/run.

    Mirrors serialize_intervention_result() output so the frontend
    can hand the trajectory arrays directly to Plotly / ECharts.
    """

    session_id: str
    layer: str
    intervention_type: str
    strength: float
    baseline_l2: Optional[float]
    intervened_l2: Optional[float]
    baseline_mse: float
    intervened_mse: float
    mse_delta: float
    mse_percent_change: float
    trajectory_baseline: list[float]
    trajectory_intervened: list[float]
    trajectory_variance: float
    trajectory_mean_deviation: float


# ── /api/intervention/sweep ───────────────────────────────────────────────────

class InterventionSweepRequest(BaseModel):
    """Body for POST /api/intervention/sweep."""

    session_id: str
    layer_name: str
    intervention_type: str = "mask"         # "mask" | "perturb"
    strengths: Optional[list[float]] = None  # default: [0.1 … 1.0]
    seed: Optional[int] = None


class SweepStepResult(BaseModel):
    strength: float
    baseline_l2: Optional[float]
    intervened_l2: Optional[float]
    baseline_mse: float
    intervened_mse: float
    mse_delta: float
    mse_percent_change: float
    trajectory_baseline: list[float]
    trajectory_intervened: list[float]
    trajectory_variance: float
    trajectory_mean_deviation: float


class InterventionSweepResponse(BaseModel):
    """Returned by POST /api/intervention/sweep."""

    session_id: str
    layer: str
    intervention_type: str
    strengths: list[float]
    results: list[SweepStepResult]


# ── /api/intervention/map ─────────────────────────────────────────────────────

class MappingRequest(BaseModel):
    """Body for POST /api/intervention/map."""

    session_id: str
    sensitivity_method: str = "gradient"    # "gradient" | "perturbation"
    seed: Optional[int] = None


class MappingNodeResult(BaseModel):
    layer: str
    rank: Optional[int] = None
    score: float
    raw_score: float
    confidence: str     # "High" | "Medium" | "Ambiguous" | "Low"
    note: str
    strategy: str
    ranked_layers: Optional[list[dict]] = None
    trend: Optional[str] = None
    direction_match: Optional[bool] = None
    quantity: Optional[str] = None
    delta_q: Optional[float] = None
    scoring_mode: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class MappingResponse(BaseModel):
    """Returned by POST /api/intervention/map."""

    session_id: str
    mapping: dict[str, MappingNodeResult]   # physical_node_label -> MappingNodeResult
    summary: dict                           # total_nodes, high_conf, ambiguous, alignment_pct


# ── /api/intervention/seed-sweep ─────────────────────────────────────────────

class SeedSweepRequest(BaseModel):
    """Body for POST /api/intervention/seed-sweep."""

    session_id: str
    start_seed: int = 0
    end_seed: int = 20
    sigma: float = Field(0.1, gt=0.0)
    sensitivity_method: str = "gradient"


class SeedSweepResponse(BaseModel):
    """Returned by POST /api/intervention/seed-sweep.

    physical_variables contains per-variable distribution statistics and
    layer_distribution counts — same data that app.py renders as histograms.
    """

    session_id: str
    seeds: list[int]
    all_neural_components: list[str]
    physical_variables: dict        # var_key -> {mean, std, min, max, median, layer_distribution, ...}
    summary: dict


# ── /api/intervention/full-analysis ──────────────────────────────────────────

class FullAnalysisRequest(BaseModel):
    """Body for POST /api/intervention/full-analysis."""

    session_id: str
    param_values: Optional[dict] = None    # override graph parameter values


class SensitivityEntry(BaseModel):
    label: str
    value: float
    gradient: float
    computed_sign: str
    expected_sign: str
    consistent: bool
    trend: Optional[str]


class FullAnalysisResponse(BaseModel):
    """Returned by POST /api/intervention/full-analysis."""

    session_id: str
    l2_norm: Optional[float]
    l2_pass: Optional[bool]
    physics_residual_norm: float
    sensitivity: dict[str, SensitivityEntry]
