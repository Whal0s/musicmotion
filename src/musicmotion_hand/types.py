from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


Point2 = Tuple[int, int]
Box2 = Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)


@dataclass(frozen=True)
class HandLandmark:
    """A single hand landmark with both normalized and pixel coordinates."""

    idx: int
    x_norm: float
    y_norm: float
    z_norm: float
    x_px: int
    y_px: int


@dataclass(frozen=True)
class HandPosition:
    """Detected hand positions for a single hand."""

    handedness_label: Optional[str]  # "Left" / "Right" (may be None)
    handedness_score: Optional[float]
    landmarks: List[HandLandmark]  # length 21
    bbox_px: Box2
    center_px: Point2
    fingertips_px: Dict[str, Point2]  # thumb/index/middle/ring/pinky


