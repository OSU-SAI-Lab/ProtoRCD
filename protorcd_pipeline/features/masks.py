"""Mask and box helpers for proposal features."""

from __future__ import annotations

from typing import Any

import numpy as np


def xywh_to_xyxy(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def clip_xyxy(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = bbox
    x1_i = max(0, min(int(np.floor(x1)), width - 1))
    y1_i = max(0, min(int(np.floor(y1)), height - 1))
    x2_i = max(0, min(int(np.ceil(x2)), width))
    y2_i = max(0, min(int(np.ceil(y2)), height))
    if x2_i <= x1_i or y2_i <= y1_i:
        return None
    return x1_i, y1_i, x2_i, y2_i


def decode_rle(segmentation: dict[str, Any] | None) -> np.ndarray | None:
    if segmentation is None:
        return None
    from pycocotools import mask as mask_utils

    rle = segmentation
    if isinstance(rle.get("counts"), list):
        h, w = rle["size"]
        rle = mask_utils.frPyObjects(rle, h, w)
    return mask_utils.decode(rle).astype(bool)

