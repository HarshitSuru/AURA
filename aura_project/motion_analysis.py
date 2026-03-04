"""Motion analysis / stampede risk detection logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class MotionResult:
    movement_score: float
    is_risk: bool


class MotionAnalyzer:
    """Detect abrupt crowd movement via frame differencing."""

    def __init__(self, movement_threshold: float = 22.0) -> None:
        self.movement_threshold = movement_threshold
        self.previous_gray: Optional[np.ndarray] = None

    def analyze(self, frame_bgr: np.ndarray) -> MotionResult:
        """Compute movement score from previous frame difference."""
        current_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.previous_gray is None:
            self.previous_gray = current_gray
            return MotionResult(movement_score=0.0, is_risk=False)

        diff = cv2.absdiff(current_gray, self.previous_gray)
        score = float(np.mean(diff))
        self.previous_gray = current_gray

        return MotionResult(movement_score=score, is_risk=score > self.movement_threshold)
