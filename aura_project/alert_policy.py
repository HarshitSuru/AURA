"""Alert debouncing and cooldown policy for reliable notifications."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict


@dataclass
class AlertState:
    consecutive_hits: int = 0
    last_sent_ts: float = 0.0


class AlertPolicy:
    """Reduce false positives with frame confirmation + per-type cooldown."""

    def __init__(self, min_consecutive: int = 3, cooldown_seconds: float = 12.0) -> None:
        self.min_consecutive = min_consecutive
        self.cooldown_seconds = cooldown_seconds
        self._states: Dict[str, AlertState] = {}

    def should_emit(self, alert_type: str, condition_met: bool) -> bool:
        state = self._states.setdefault(alert_type, AlertState())

        if not condition_met:
            state.consecutive_hits = 0
            return False

        state.consecutive_hits += 1
        if state.consecutive_hits < self.min_consecutive:
            return False

        now = time.time()
        if (now - state.last_sent_ts) < self.cooldown_seconds:
            return False

        state.last_sent_ts = now
        return True
