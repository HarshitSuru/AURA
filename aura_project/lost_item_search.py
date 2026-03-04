"""Lost-item visual similarity search with YOLO crops + CLIP embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from ultralytics import YOLO

from aura_project.buffer import RollingFrameBuffer


@dataclass
class SearchMatch:
    timestamp: str
    similarity: float
    annotated_frame: np.ndarray


class LostItemSearcher:
    """Search buffered frames for objects similar to uploaded item image."""

    def __init__(self, yolo_model_path: str = "yolov8n.pt", clip_model_name: str = "ViT-B-32") -> None:
        self.object_detector = YOLO(yolo_model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained="laion2b_s34b_b79k")
        self.clip_model = model.to(self.device).eval()
        self.preprocess = preprocess

    def _encode_image(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.clip_model.encode_image(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.squeeze(0).cpu().numpy()

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def search(self, uploaded_item_bgr: np.ndarray, frame_buffer: RollingFrameBuffer, similarity_threshold: float = 0.28) -> Optional[SearchMatch]:
        """Return the best matching buffered frame if similarity threshold is exceeded."""
        target_embedding = self._encode_image(uploaded_item_bgr)

        best_match: Optional[SearchMatch] = None

        for buffered in frame_buffer.items():
            frame = buffered.frame_bgr
            result = self.object_detector.predict(frame, verbose=False)[0]

            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), conf in zip(boxes, confidences):
                crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                if crop.size == 0:
                    continue

                crop_embedding = self._encode_image(crop)
                sim = self._cosine_similarity(target_embedding, crop_embedding)

                if sim >= similarity_threshold and (best_match is None or sim > best_match.similarity):
                    annotated = frame.copy()
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        annotated,
                        f"Match {sim:.2f}",
                        (x1, max(10, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    best_match = SearchMatch(timestamp=buffered.timestamp, similarity=sim, annotated_frame=annotated)

        return best_match
