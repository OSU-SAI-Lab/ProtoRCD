"""Crop preparation for crop-level encoders (BioCLIP, DINOv3, debiased DINOv3)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .masks import clip_xyxy, decode_rle, xywh_to_xyxy
from .proposals import Proposal


def proposal_crop(
    image: Image.Image,
    proposal: Proposal,
    region: str,
    mask_background: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image | None:
    image_np = np.asarray(image.convert("RGB")).copy()
    height, width = image_np.shape[:2]
    clipped = clip_xyxy(xywh_to_xyxy(proposal.bbox_xywh), width=width, height=height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    crop = image_np[y1:y2, x1:x2].copy()

    if region == "mask":
        mask = decode_rle(proposal.segmentation)
        if mask is None:
            return None
        crop_mask = mask[y1:y2, x1:x2]
        if crop_mask.shape[:2] != crop.shape[:2] or not bool(crop_mask.any()):
            return None
        bg = np.asarray(mask_background, dtype=np.uint8)
        crop[~crop_mask] = bg

    return Image.fromarray(crop)

