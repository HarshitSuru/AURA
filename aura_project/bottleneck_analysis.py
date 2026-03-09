"""Bottleneck risk detection based on crowd concentration across horizontal zones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from aura_project.detection import PersonDetection


@dataclass
class BottleneckResult:
    concentration_score: float
    is_bottleneck: bool
    hot_zone: tuple[int, int]


class BottleneckAnalyzer:
    """Estimate bottleneck risk using occupancy concentration in horizontal bins."""

    def __init__(self, concentration_threshold: float = 0.48, min_people: int = 6, num_bins: int = 6) -> None:
        self.concentration_threshold = concentration_threshold
        self.min_people = min_people
        self.num_bins = num_bins

    def analyze(self, detections: Sequence[PersonDetection], frame_width: int) -> BottleneckResult:
        if not detections or frame_width <= 0:
            return BottleneckResult(0.0, False, (0, 0))

        centers = np.array([(d.bbox[0] + d.bbox[2]) / 2.0 for d in detections], dtype=float)
        bins = np.linspace(0, frame_width, self.num_bins + 1)
        counts, edges = np.histogram(centers, bins=bins)

        max_idx = int(np.argmax(counts))
        max_count = int(counts[max_idx])
        total = len(detections)
        concentration = max_count / total if total else 0.0
        is_bottleneck = total >= self.min_people and concentration >= self.concentration_threshold

        return BottleneckResult(
            concentration_score=concentration,
            is_bottleneck=is_bottleneck,
            hot_zone=(int(edges[max_idx]), int(edges[max_idx + 1])),
        )
