from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np


def draw_bbox(frame, bbox_px: Tuple[int, int, int, int], color=(0, 255, 0), thickness=2):
    x0, y0, x1, y1 = bbox_px
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)
    return frame


def draw_point(frame, pt: Tuple[int, int], color=(0, 0, 255), radius=5):
    cv2.circle(frame, pt, radius, color, -1)
    return frame


def draw_text(frame, text: str, org: Tuple[int, int], color=(255, 255, 255), scale=0.6, thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return frame


def draw_polyline(frame, points: Iterable[Tuple[int, int]], color=(255, 255, 0), thickness=2, closed=False):
    pts = np.array([(int(x), int(y)) for x, y in points], dtype=np.int32)
    if pts.shape[0] < 2:
        return frame
    cv2.polylines(frame, [pts], closed, color, thickness)
    return frame


