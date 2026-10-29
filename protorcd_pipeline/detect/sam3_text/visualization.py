"""
Visualization utilities for SAM3 detections.
"""
from __future__ import annotations

import os

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image


def draw_detections(
    image_or_path: str | np.ndarray,
    detections: dict,
    output_path: str,
    gt_bboxes: list | None = None,
    max_proposals: int = 100,
    alpha: float = 0.4,
    return_masks: bool = False,
) -> None:
    """
    Draw detections with masks and optionally ground truth boxes on image.

    Args:
        image_or_path: Either path to image file or RGB numpy array
        detections: Detection results dict with 'boxes', 'scores', and optionally 'masks'/'mask_info'
        output_path: Path to save the visualization
        gt_bboxes: Optional list of ground truth boxes in xyxy format (drawn in red)
        max_proposals: Maximum number of proposals to visualize
        alpha: Transparency for mask overlay
        return_masks: Whether masks are expected in the detections
    """
    # Load image
    if isinstance(image_or_path, str):
        img_rgb = np.array(Image.open(image_or_path).convert("RGB"))
    else:
        img_rgb = image_or_path

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # Draw ground truth boxes in red
    if gt_bboxes:
        for box in gt_bboxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    boxes = detections["boxes"]
    scores = detections["scores"]
    labels = detections.get("labels")
    label_names = detections.get("label_names")
    masks = detections.get("masks") if return_masks else None
    mask_info = detections.get("mask_info") if return_masks else None

    # Sort by score and limit to max_proposals
    sorted_indices = np.argsort(scores)[::-1][:max_proposals]

    # Generate consistent colors for masks only
    np.random.seed(42)
    mask_colors = [
        (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
        for _ in range(min(len(boxes), max_proposals))
    ]

    # Pure green for all boxes (BGR: Blue, Green, Red)
    box_color = (0, 255, 0)

    def to_numpy(tensor):
        """Convert tensor to numpy, handling torch tensors."""
        arr = tensor.cpu().numpy() if hasattr(tensor, "cpu") else tensor
        return arr[0] if arr.ndim > 2 else arr

    # Batch resize masks on GPU (3-5x faster than sequential CV2)
    full_masks_cache = {}
    if masks is not None and len(masks) > 0:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mask_list = [to_numpy(masks[idx]) for idx in sorted_indices if idx < len(masks)]
        if mask_list:
            masks_tensor = torch.from_numpy(np.stack(mask_list)).float().to(device)
            if masks_tensor.shape[1:3] != (h, w):
                masks_resized = F.interpolate(
                    masks_tensor.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False
                ).squeeze(1).cpu().numpy()
            else:
                masks_resized = masks_tensor.cpu().numpy()
            full_masks_cache = {sorted_indices[i]: masks_resized[i] for i in range(len(masks_resized))}

    def get_full_mask(idx, h, w):
        """Get full-resolution mask for detection idx."""
        if idx in full_masks_cache:
            return full_masks_cache[idx]
        # SAHI cropped masks with offsets
        if mask_info is not None and idx < len(mask_info):
            info = mask_info[idx]
            mask = to_numpy(info["mask"])
            crop_size = info["crop_size"]
            if mask.shape[0] != crop_size:
                mask = cv2.resize(mask.astype(np.float32), (crop_size, crop_size))
            full_mask = np.zeros((h, w), dtype=np.float32)
            ox, oy = info["offset"]
            dx2, dy2 = min(ox + crop_size, w), min(oy + crop_size, h)
            full_mask[oy:dy2, ox:dx2] = mask[:dy2 - oy, :dx2 - ox]
            return full_mask
        return None

    # Draw each detection
    for i, idx in enumerate(sorted_indices):
        x1, y1, x2, y2 = map(int, boxes[idx])
        score = scores[idx]
        mask_color = mask_colors[i]

        # Draw mask if available (with unique color per detection)
        mask = get_full_mask(idx, h, w)
        if mask is not None:
            overlay = img_bgr.copy()
            overlay[mask > 0.5] = mask_color
            img_bgr = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)

        # Draw thin green bounding box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color=box_color, thickness=3)

        # Draw confidence score with semi-transparent background
        class_name = ""
        if label_names is not None and idx < len(label_names):
            class_name = str(label_names[idx])
        elif labels is not None and idx < len(labels):
            class_name = str(int(labels[idx]))
        label = f"{class_name} {score:.2f}" if class_name else f"{score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )
        label_x1 = x1
        label_y1 = max(y1 - text_h - baseline - 4, 0)
        label_x2 = label_x1 + text_w + 4
        label_y2 = label_y1 + text_h + baseline + 4

        # Draw semi-transparent white background for text
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), (255, 255, 255), -1)
        img_bgr = cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0)

        # Draw text
        cv2.putText(
            img_bgr,
            label,
            (label_x1 + 2, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Save output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img_bgr)
