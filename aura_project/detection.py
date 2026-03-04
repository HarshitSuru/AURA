"""Crowd detection and counting using YOLOv8."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class PersonDetection:
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass
class CrowdResult:
    detections: List[PersonDetection]
    people_count: int
    density: float
    is_high_density: bool
    annotated_frame: np.ndarray


class CrowdDetector:
    """Person detector wrapper around YOLOv8."""

    def __init__(self, model_path: str = "yolov8n.pt", density_threshold: float = 0.00008) -> None:
        self.model = YOLO(model_path)
        self.density_threshold = density_threshold

    def detect(self, frame_bgr: np.ndarray) -> CrowdResult:
        """Detect people, compute crowd density, and draw annotations."""
        results = self.model.predict(frame_bgr, verbose=False)
        frame_out = frame_bgr.copy()

        detections: List[PersonDetection] = []
        if results:
            res = results[0]
            boxes = res.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, cls_id in zip(xyxy, conf, cls):
                    if cls_id == 0:  # person class in COCO
                        detections.append(PersonDetection((x1, y1, x2, y2), float(c)))
                        cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame_out,
                            f"person {c:.2f}",
                            (x1, max(10, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )

        h, w = frame_bgr.shape[:2]
        area = h * w
        count = len(detections)
        density = count / area if area else 0.0

        cv2.putText(
            frame_out,
            f"People: {count} | Density: {density:.6f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2,
        )

        return CrowdResult(
            detections=detections,
            people_count=count,
            density=density,
            is_high_density=density > self.density_threshold,
            annotated_frame=frame_out,
        )
