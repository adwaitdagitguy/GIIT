"""
session_manager.py
==================
In-memory session store for the SciML backend.

Each upload of a model + graph creates a ModelSession with a UUID.
The session_id is returned to the caller and used as a query parameter
on every subsequent request.

All data is kept in-process (no Redis, no DB) — sessions are lost on
server restart, which is acceptable for a single-user academic demo.
"""

import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional, Any

import torch
import torch.nn as nn


@dataclass
class ModelSession:
    """All server-side state for one uploaded model + graph pair."""

    session_id: str
    model: nn.Module
    model_metadata: dict
    graph_data: dict
    # Populated lazily after initialization
    intervener: Any  # DynamicSciMLIntervener — typed as Any to avoid import cycle
    test_inputs: torch.Tensor
    test_targets: Optional[torch.Tensor]
    # Optional physics ruleset (uploaded or LLM-generated)
    physics_ruleset: Optional[dict] = None
    physics_ruleset_source: Optional[str] = None  # "uploaded" | "suggested"
    # Cached results from the most recent analysis runs
    mapping_results: Optional[dict] = None
    alignment_score: Optional[float] = None
    created_at: float = field(default_factory=time.time)


class SessionManager:
    """
    Thread-safe dict-backed session store.

    Public API
    ----------
    create(session) -> str
        Persist session and return its ID.
    get(session_id) -> ModelSession
        Retrieve session or raise KeyError.
    update(session_id, **kwargs)
        Patch individual fields on an existing session.
    delete(session_id)
        Remove session from store.
    list_sessions() -> list[dict]
        Summary of all active sessions.
    """

    def __init__(self):
        self._store: dict[str, ModelSession] = {}
        self._lock = threading.Lock()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def create(self, session: ModelSession) -> str:
        """Add a new session to the store and return its ID."""
        with self._lock:
            self._store[session.session_id] = session
        return session.session_id

    def get(self, session_id: str) -> ModelSession:
        """
        Retrieve session by ID.

        Raises
        ------
        KeyError
            If session_id is not found (caller converts to HTTP 404).
        """
        with self._lock:
            if session_id not in self._store:
                raise KeyError(f"Session '{session_id}' not found.")
            return self._store[session_id]

    def update(self, session_id: str, **kwargs):
        """
        Patch one or more fields on an existing session in-place.

        Only fields that are attributes of ModelSession are accepted;
        unknown keys raise AttributeError.
        """
        with self._lock:
            session = self._store[session_id]
            for key, value in kwargs.items():
                if not hasattr(session, key):
                    raise AttributeError(
                        f"ModelSession has no attribute '{key}'. "
                        f"Valid fields: {[f.name for f in session.__dataclass_fields__.values()]}"
                    )
                setattr(session, key, value)

    def delete(self, session_id: str):
        """Remove session. Silent no-op if id does not exist."""
        with self._lock:
            self._store.pop(session_id, None)

    def list_sessions(self) -> list:
        """Return a lightweight summary (no tensors/models) for each session."""
        with self._lock:
            summaries = []
            for sid, sess in self._store.items():
                summaries.append({
                    "session_id": sid,
                    "system": sess.model_metadata.get("system", "Unknown"),
                    "has_ruleset": sess.physics_ruleset is not None,
                    "has_mapping": sess.mapping_results is not None,
                    "created_at": sess.created_at,
                })
            return summaries

    # ── Convenience ───────────────────────────────────────────────────────────

    @staticmethod
    def new_id() -> str:
        """Generate a fresh UUID4 session ID."""
        return str(uuid.uuid4())


# Module-level singleton used by all route handlers.
session_store = SessionManager()
