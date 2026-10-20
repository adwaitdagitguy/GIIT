"""
ruleset_service.py
==================
Service layer for physics ruleset upload, attaching, and LLM suggestion.

Wraps existing functions from:
  - core/llm_engine.py : generate_physics_ruleset, validate_physics_ruleset
  - core/dynamic_intervener.py : DynamicSciMLIntervener.set_physics_ruleset
"""

import json
from core.llm_engine import generate_physics_ruleset, validate_physics_ruleset
from backend.session_manager import session_store


def handle_ruleset_upload(session_id: str, ruleset_bytes: bytes) -> dict:
    """
    Parse uploaded ruleset JSON, attach it to the session intervener, and
    record it in the session store.

    Args:
        session_id    : UUID of the active ModelSession.
        ruleset_bytes : Raw bytes of the ruleset.json file.

    Returns:
        dict ready for RulesetUploadResponse serialization.
    """
    session = session_store.get(session_id)

    try:
        ruleset_dict = json.loads(ruleset_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid ruleset JSON: {exc}") from exc

    # Attach ruleset to intervener
    session.intervener.set_physics_ruleset(ruleset_dict)

    # Persist in session store
    session_store.update(
        session_id,
        physics_ruleset=ruleset_dict,
        physics_ruleset_source="uploaded",
    )

    quantities = list(ruleset_dict.get("quantities", {}).keys())
    return {
        "session_id": session_id,
        "source": "uploaded",
        "quantities": quantities,
        "attached": True,
    }


def handle_ruleset_generate(session_id: str, api_key: str) -> dict:
    """
    Call LLM via OpenRouter to generate derived quantity compute functions
    based on the graph's expected_trends, attach the generated ruleset to
    the intervener, and persist in session store.

    Args:
        session_id : UUID of the active ModelSession.
        api_key    : OpenRouter API key.

    Returns:
        dict ready for RulesetGenerateResponse serialization.
    """
    session = session_store.get(session_id)
    graph = session.graph_data

    # Generate ruleset using existing core/llm_engine.py function
    ruleset_dict = generate_physics_ruleset(graph=graph, api_key=api_key)

    # Attach to intervener
    session.intervener.set_physics_ruleset(ruleset_dict)

    # Persist in session store
    session_store.update(
        session_id,
        physics_ruleset=ruleset_dict,
        physics_ruleset_source="suggested",
    )

    quantities = list(ruleset_dict.get("quantities", {}).keys())
    return {
        "session_id": session_id,
        "source": "suggested",
        "quantities": quantities,
        "ruleset": ruleset_dict,
        "attached": True,
    }
