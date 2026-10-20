"""
intervention_service.py
======================
Service layer wrapping all intervention, mapping, and physics analysis methods.

Wraps existing functions from core/dynamic_intervener.py:
  - DynamicSciMLIntervener.apply_intervention
  - DynamicSciMLIntervener.run_monotonic_intervention_sweep
  - DynamicSciMLIntervener.generate_mapping_with_confidence
  - DynamicSciMLIntervener.run_physical_seed_sweep
  - DynamicSciMLIntervener.run_full_analysis
  - serialize_intervention_result
  - serialize_sweep_result

Does NOT re-implement any math, autograd gradients, perturbations, or heuristics.
"""

from typing import Optional
from core.dynamic_intervener import (
    serialize_intervention_result,
    serialize_sweep_result,
)
from backend.session_manager import session_store


def run_single_intervention(
    session_id: str,
    layer_name: str,
    intervention_type: str = "mask",
    strength: float = 0.5,
    seed: Optional[int] = None,
) -> dict:
    """
    Apply a single intervention (mask or perturb) on the specified layer.

    Wraps DynamicSciMLIntervener.apply_intervention() and serializes with
    serialize_intervention_result().

    Args:
        session_id        : UUID of the active ModelSession.
        layer_name        : State dict key to intervene on.
        intervention_type : "mask" or "perturb".
        strength          : Float in [0.0, 1.0].
        seed              : Optional random seed for Gaussian perturbation.

    Returns:
        dict ready for InterventionRunResponse serialization.
    """
    session = session_store.get(session_id)
    intervener = session.intervener

    # Pre-compute baseline L2 if targets available
    baseline_l2 = None
    if session.test_targets is not None:
        try:
            baseline_l2 = intervener.compute_l2_norm(
                session.test_inputs, session.test_targets
            )
        except Exception:
            pass

    # Run intervention using existing engine
    raw_result = intervener.apply_intervention(
        layer_name=layer_name,
        intervention_type=intervention_type,
        strength=strength,
        input_tensor=session.test_inputs,
        target_tensor=session.test_targets,
        seed=seed,
    )

    # Compute intervened L2 if targets available
    intervened_l2 = None
    if session.test_targets is not None:
        try:
            # apply_intervention already computed intervened_pred, compute L2 directly on the target device
            import torch
            tgt_t = session.test_targets
            pred_t = torch.tensor(
                raw_result["intervened_pred"],
                dtype=torch.float32,
                device=tgt_t.device,
            ).view_as(tgt_t)
            intervened_l2 = float(
                (torch.norm(pred_t - tgt_t) / torch.norm(tgt_t)).item()
            )
        except Exception:
            pass

    # Use existing serialization helper
    serialized = serialize_intervention_result(
        result=raw_result,
        l2_baseline=baseline_l2,
        l2_intervened=intervened_l2,
    )
    serialized["session_id"] = session_id
    return serialized


def run_intervention_sweep(
    session_id: str,
    layer_name: str,
    intervention_type: str = "mask",
    strengths: Optional[list[float]] = None,
    seed: Optional[int] = None,
) -> dict:
    """
    Sweep intervention across multiple strengths monotonically.

    Wraps DynamicSciMLIntervener.run_monotonic_intervention_sweep() and
    serializes with serialize_sweep_result().

    Args:
        session_id        : UUID of the active ModelSession.
        layer_name        : State dict key to intervene on.
        intervention_type : "mask" or "perturb".
        strengths         : Optional list of floats in [0.0, 1.0].
        seed              : Optional random seed for perturbations.

    Returns:
        dict ready for InterventionSweepResponse serialization.
    """
    session = session_store.get(session_id)
    intervener = session.intervener

    raw_sweep = intervener.run_monotonic_intervention_sweep(
        layer_name=layer_name,
        intervention_type=intervention_type,
        input_tensor=session.test_inputs,
        target_tensor=session.test_targets,
        strengths=strengths,
        seed=seed,
    )

    serialized = serialize_sweep_result(raw_sweep)
    serialized["session_id"] = session_id
    return serialized


def run_node_mapping(
    session_id: str,
    sensitivity_method: str = "gradient",
    seed: Optional[int] = None,
) -> dict:
    """
    Discover mappings from physical causal nodes to neural layers.

    Wraps DynamicSciMLIntervener.generate_mapping_with_confidence().

    Args:
        session_id         : UUID of the active ModelSession.
        sensitivity_method : "gradient" or "perturbation".
        seed               : Random seed for perturbation mode.

    Returns:
        dict ready for MappingResponse serialization.
    """
    session = session_store.get(session_id)
    intervener = session.intervener

    # Call existing mapping method
    res = intervener.generate_mapping_with_confidence(
        input_tensor=session.test_inputs,
        seed=seed,
        sensitivity_method=sensitivity_method,
    )

    mapping = res.get("mapping", {})
    summary = res.get("summary", {})

    # Cache in session store
    session_store.update(
        session_id,
        mapping_results=mapping,
        alignment_score=summary.get("alignment_pct"),
    )

    return {
        "session_id": session_id,
        "mapping": mapping,
        "summary": summary,
    }


def run_seed_sweep(
    session_id: str,
    start_seed: int = 0,
    end_seed: int = 20,
    sigma: float = 0.1,
    sensitivity_method: str = "gradient",
) -> dict:
    """
    Run physical seed sweep to compute layer mapping distributions.

    Wraps DynamicSciMLIntervener.run_physical_seed_sweep().

    Args:
        session_id         : UUID of the active ModelSession.
        start_seed         : Starting seed index.
        end_seed           : Ending seed index.
        sigma              : Perturbation magnitude.
        sensitivity_method : "gradient" or "perturbation".

    Returns:
        dict ready for SeedSweepResponse serialization.
    """
    session = session_store.get(session_id)
    intervener = session.intervener

    res = intervener.run_physical_seed_sweep(
        input_tensor=session.test_inputs,
        start_seed=start_seed,
        end_seed=end_seed,
        sigma=sigma,
        sensitivity_method=sensitivity_method,
    )

    res["session_id"] = session_id
    return res


def run_full_physics_analysis(
    session_id: str,
    param_values: Optional[dict] = None,
) -> dict:
    """
    Run comprehensive physics analysis: L2 norm + residual norm + parameter sensitivities.

    Wraps DynamicSciMLIntervener.run_full_analysis().

    Args:
        session_id   : UUID of the active ModelSession.
        param_values : Optional override dictionary for physics parameters.

    Returns:
        dict ready for FullAnalysisResponse serialization.
    """
    session = session_store.get(session_id)
    intervener = session.intervener

    res = intervener.run_full_analysis(
        input_tensor=session.test_inputs,
        target_tensor=session.test_targets,
        param_values=param_values,
    )

    return {
        "session_id": session_id,
        "l2_norm": res.get("l2_norm"),
        "l2_pass": res.get("l2_pass"),
        "physics_residual_norm": res.get("physics_residual_norm", 0.0),
        "sensitivity": res.get("sensitivity", {}),
    }
