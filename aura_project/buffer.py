"""Rolling frame buffer for AURA.

Stores frame snapshots for recent history to support Lost & Found search.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List

import numpy as np


@dataclass
class BufferedFrame:
    """One buffered frame with metadata."""

    timestamp: str
    frame_bgr: np.ndarray


class RollingFrameBuffer:
    """Fixed-duration frame buffer backed by deque."""

    def __init__(self, max_minutes: int = 5, fps: int = 5) -> None:
        self.maxlen = max_minutes * 60 * fps
        self._buffer: Deque[BufferedFrame] = deque(maxlen=self.maxlen)

    def add(self, timestamp: str, frame_bgr: np.ndarray) -> None:
        """Push new frame into rolling window."""
        self._buffer.append(BufferedFrame(timestamp=timestamp, frame_bgr=frame_bgr.copy()))

    def items(self) -> List[BufferedFrame]:
        """Get snapshot list of currently buffered frames."""
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)
