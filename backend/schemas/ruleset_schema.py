"""
ruleset_schema.py
=================
Pydantic schemas for physics ruleset upload and LLM generation.
"""

from pydantic import BaseModel
from typing import Optional


class RulesetUploadResponse(BaseModel):
    """Returned by POST /api/ruleset/upload."""

    session_id: str
    source: str             # "uploaded"
    quantities: list[str]   # names of quantities in the ruleset
    attached: bool          # True if successfully attached to the session intervener


class RulesetGenerateRequest(BaseModel):
    """Body for POST /api/ruleset/generate."""

    session_id: str
    api_key: str    # OpenRouter API key — never logged or stored


class RulesetGenerateResponse(BaseModel):
    """Returned by POST /api/ruleset/generate."""

    session_id: str
    source: str             # "suggested"
    quantities: list[str]   # names of quantities generated
    ruleset: dict           # the full ruleset dict for download / preview
    attached: bool
