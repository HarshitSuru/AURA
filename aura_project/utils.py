"""Utility helpers for AURA.

Contains reusable helpers for image processing, metrics, and data IO.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd


EVENT_COLUMNS = ["timestamp", "event_type", "description", "people_count", "density", "movement_score"]


def ensure_directories() -> None:
    """Ensure required project directories and files exist."""
    Path("aura_project/logs").mkdir(parents=True, exist_ok=True)
    path = Path("aura_project/logs/events.csv")
    if not path.exists():
        pd.DataFrame(columns=EVENT_COLUMNS).to_csv(path, index=False)


def timestamp_now() -> str:
    """Formatted timestamp string for logs and UI."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def resize_frame(frame: np.ndarray, size: Tuple[int, int] = (640, 480)) -> np.ndarray:
    """Resize frame to fixed dashboard input size."""
    return cv2.resize(frame, size)


def frame_area(frame: np.ndarray) -> int:
    """Return area (pixels) of a frame."""
    h, w = frame.shape[:2]
    return h * w


def append_event(event: dict) -> None:
    """Append one event row to events.csv."""
    path = Path("aura_project/logs/events.csv")
    df = pd.DataFrame([event], columns=EVENT_COLUMNS)
    df.to_csv(path, mode="a", index=False, header=not path.exists() or path.stat().st_size == 0)


def load_events() -> pd.DataFrame:
    """Load event log CSV as DataFrame."""
    ensure_directories()
    return pd.read_csv("aura_project/logs/events.csv")


def to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR frame to RGB for Streamlit display."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
