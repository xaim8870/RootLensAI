"""Process-local runtime state for the RootLensAI API.

The fault controller remains the authority for testbed state.  This module
only tracks presentation/runtime facts that the modular Streamlit application
previously kept in ``st.session_state``.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


INJECT_RCA_LOCK_SECONDS = 30
RESTORE_RCA_LOCK_SECONDS = 60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ApiRuntimeState:
    """Thread-safe state shared by all API requests in one server process."""

    lock: threading.RLock = field(default_factory=threading.RLock)
    operation_lock: threading.Lock = field(default_factory=threading.Lock)
    rca_result: dict[str, Any] | None = None
    rca_stale: bool = False
    rca_unlock_at: float = 0.0
    rca_lock_reason: str = ""
    rca_lock_seconds: int = 0
    last_operation_type: str = ""
    last_operation_duration: float | None = None
    last_operation_finished_at: str | None = None
    last_rca_duration: float | None = None
    last_rag_duration: float | None = None
    rag_analysis: dict[str, Any] | None = None
    rag_source_timestamp: str | None = None

    def invalidate(self, *, seconds: int, reason: str) -> None:
        with self.lock:
            self.rca_result = None
            self.rca_stale = True
            self.rca_unlock_at = time.time() + seconds
            self.rca_lock_reason = reason
            self.rca_lock_seconds = seconds
            self.rag_analysis = None
            self.rag_source_timestamp = None
            self.last_rag_duration = None

    def gate_remaining(self) -> int:
        with self.lock:
            return int(math.ceil(max(0.0, self.rca_unlock_at - time.time())))

    def gate_snapshot(self) -> dict[str, Any]:
        with self.lock:
            remaining = int(math.ceil(max(0.0, self.rca_unlock_at - time.time())))
            return {
                "active": remaining > 0,
                "remaining_seconds": remaining,
                "total_seconds": self.rca_lock_seconds,
                "unlock_at": (
                    datetime.fromtimestamp(self.rca_unlock_at, timezone.utc).isoformat()
                    if self.rca_unlock_at
                    else None
                ),
                "reason": self.rca_lock_reason,
            }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "telemetry_gate": self.gate_snapshot(),
                "rca_stale": self.rca_stale,
                "has_fresh_rca": self.rca_result is not None and not self.rca_stale,
                "last_operation": {
                    "type": self.last_operation_type or None,
                    "duration_seconds": self.last_operation_duration,
                    "finished_at": self.last_operation_finished_at,
                },
                "last_rca_duration_seconds": self.last_rca_duration,
                "last_rag_duration_seconds": self.last_rag_duration,
            }


runtime_state = ApiRuntimeState()

