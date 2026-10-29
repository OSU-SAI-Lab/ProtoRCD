"""Small geometry helpers for proposal conversion and evaluation."""

from __future__ import annotations

from typing import Sequence


def xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    """Convert COCO xywh box to xyxy."""
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    """Convert xyxy box to COCO xywh."""
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, x2 - x1, y2 - y1]


def box_area_xyxy(box: Sequence[float]) -> float:
    """Return non-negative xyxy box area."""
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def clip_xyxy(box: Sequence[float], width: int, height: int) -> list[float]:
    """Clip xyxy box to image bounds."""
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    ]

