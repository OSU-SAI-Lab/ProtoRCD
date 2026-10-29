"""
SAHI utilities for SAM3 - slicing and batching functions.
"""
import numpy as np
import torch
from torchvision.ops import nms
from sahi.slicing import slice_image


# Helper to suppress boxes that are nearly fully contained in larger boxes
def suppress_contained_boxes(boxes: torch.Tensor, scores: torch.Tensor,
                             containment_thresh: float = 0.85,
                             area_ratio_thresh: float = 0.6,
                             return_keep: bool = False) -> tuple:
    """
    Suppress boxes that are almost fully contained inside larger boxes.
    Designed to remove partial duplicates while keeping nearby animals.
    Vectorized version for speed.
    """
    if boxes.numel() == 0:
        if return_keep:
            return boxes, scores, torch.empty(0, dtype=torch.bool, device=boxes.device)
        return boxes, scores

    n = len(boxes)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Compute all pairwise intersections (vectorized)
    x1 = torch.max(boxes[:, None, 0], boxes[None, :, 0])  # (n, n)
    y1 = torch.max(boxes[:, None, 1], boxes[None, :, 1])
    x2 = torch.min(boxes[:, None, 2], boxes[None, :, 2])
    y2 = torch.min(boxes[:, None, 3], boxes[None, :, 3])

    inter_areas = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)  # (n, n)

    # Compute containment: inter / smaller_area
    areas_matrix = areas[:, None].expand(n, n)  # (n, n)
    smaller_areas = torch.min(areas_matrix, areas_matrix.t())
    containment = inter_areas / (smaller_areas + 1e-6)

    # Compute area ratios: smaller / larger
    larger_areas = torch.max(areas_matrix, areas_matrix.t())
    area_ratios = smaller_areas / (larger_areas + 1e-6)

    # Find boxes to suppress: high containment + small area ratio
    suppress_mask = (containment > containment_thresh) & (area_ratios < area_ratio_thresh)
    suppress_mask.fill_diagonal_(False)  # Don't suppress self

    # A box is suppressed if it's the smaller one in any pair
    is_smaller = areas[:, None] < areas[None, :]
    should_suppress = suppress_mask & is_smaller
    keep = ~should_suppress.any(dim=1)

    if return_keep:
        return boxes[keep], scores[keep], keep
    return boxes[keep], scores[keep]


def slice_and_batch(
    img_np: np.ndarray,
    slice_h: int = 1008,
    slice_w: int = 1008,
    overlap_h: float = 0.25,
    overlap_w: float = 0.25,
    batch_size: int = 4,
) -> list:
    """
    Slice an image into overlapping crops and group into batches.

    Args:
        img_np: Input image as numpy array (H, W, 3)
        slice_h: Height of each slice
        slice_w: Width of each slice
        overlap_h: Vertical overlap ratio (0-1)
        overlap_w: Horizontal overlap ratio (0-1)
        batch_size: Number of slices per batch

    Returns:
        List of batches, where each batch is a list of slice info dicts
        Each dict has: {"image": np.ndarray, "starting_pixel": [x, y]}
    """
    slices = slice_image(
        image=img_np,
        slice_height=slice_h,
        slice_width=slice_w,
        overlap_height_ratio=overlap_h,
        overlap_width_ratio=overlap_w,
    )
    return [slices[i : i + batch_size] for i in range(0, len(slices), batch_size)]


def merge_detections(
    all_boxes: list,
    all_scores: list,
    all_labels: list = None,
    nms_iou_threshold: float = 0.4,
    containment_thresh: float = 0.85,
    area_ratio_thresh: float = 0.6,
    device: str = "cuda",
    max_box_size: int = None,
) -> tuple:
    """
    Merge detections from all crops and apply global NMS.

    Args:
        all_boxes: List of box tensors from each crop
        all_scores: List of score tensors from each crop
        nms_iou_threshold: IoU threshold for NMS
        containment_thresh: Containment threshold for suppression
        area_ratio_thresh: Area ratio threshold for suppression
        device: Device to use
        max_box_size: Optional maximum box size (applied before containment/area filtering)

    Returns:
        Tuple of (boxes, scores) after NMS
    """
    if not all_boxes:
        empty_boxes = torch.empty(0, 4, device=device)
        empty_scores = torch.empty(0, device=device)
        if all_labels is not None:
            return empty_boxes, empty_scores, torch.empty(0, dtype=torch.long, device=device)
        return empty_boxes, empty_scores

    boxes_cat = torch.cat(all_boxes, dim=0)
    scores_cat = torch.cat(all_scores, dim=0)
    labels_cat = torch.cat(all_labels, dim=0) if all_labels is not None else None

    if boxes_cat.numel() == 0:
        if labels_cat is not None:
            return boxes_cat, scores_cat, labels_cat
        return boxes_cat, scores_cat

    # Apply max_box_size filtering FIRST (before containment/area checks)
    if max_box_size is not None and max_box_size > 0:
        widths = boxes_cat[:, 2] - boxes_cat[:, 0]
        heights = boxes_cat[:, 3] - boxes_cat[:, 1]
        size_mask = (widths <= max_box_size) & (heights <= max_box_size)
        boxes_cat = boxes_cat[size_mask]
        scores_cat = scores_cat[size_mask]
        if labels_cat is not None:
            labels_cat = labels_cat[size_mask]

        if boxes_cat.numel() == 0:
            if labels_cat is not None:
                return boxes_cat, scores_cat, labels_cat
            return boxes_cat, scores_cat

    keep_indices = nms(boxes_cat, scores_cat, nms_iou_threshold)
    boxes_nms = boxes_cat[keep_indices]
    scores_nms = scores_cat[keep_indices]
    labels_nms = labels_cat[keep_indices] if labels_cat is not None else None

    boxes_final, scores_final, keep_mask = suppress_contained_boxes(
        boxes_nms,
        scores_nms,
        containment_thresh=containment_thresh,
        area_ratio_thresh=area_ratio_thresh,
        return_keep=True,
    )
    if labels_nms is not None:
        labels_final = labels_nms[keep_mask]
        return boxes_final, scores_final, labels_final
    return boxes_final, scores_final


def merge_detections_with_masks(
    all_boxes: list,
    all_scores: list,
    all_masks: list,
    all_labels: list = None,
    nms_iou_threshold: float = 0.4,
    containment_thresh: float = 0.85,
    area_ratio_thresh: float = 0.6,
    device: str = "cuda",
    max_box_size: int = None,
) -> tuple:
    """
    Merge detections with masks from all crops and apply global NMS.

    Args:
        all_boxes: List of box tensors from each crop
        all_scores: List of score tensors from each crop
        all_masks: List of mask info dicts with 'mask', 'offset', 'crop_size'
        nms_iou_threshold: IoU threshold for NMS
        containment_thresh: Containment threshold for suppression
        area_ratio_thresh: Area ratio threshold for suppression
        device: Device to use
        max_box_size: Optional maximum box size (applied before containment/area filtering)

    Returns:
        Tuple of (boxes, scores, mask_info_list) after NMS
    """
    if not all_boxes:
        empty_boxes = torch.empty(0, 4, device=device)
        empty_scores = torch.empty(0, device=device)
        if all_labels is not None:
            return empty_boxes, empty_scores, [], torch.empty(0, dtype=torch.long, device=device)
        return empty_boxes, empty_scores, []

    boxes_cat = torch.cat(all_boxes, dim=0)
    scores_cat = torch.cat(all_scores, dim=0)
    labels_cat = torch.cat(all_labels, dim=0) if all_labels is not None else None

    if boxes_cat.numel() == 0:
        if labels_cat is not None:
            return boxes_cat, scores_cat, [], labels_cat
        return boxes_cat, scores_cat, []

    # Flatten mask info list
    masks_flat = []
    for mask_list in all_masks:
        masks_flat.extend(mask_list)

    # Apply max_box_size filtering FIRST (before containment/area checks)
    if max_box_size is not None and max_box_size > 0:
        widths = boxes_cat[:, 2] - boxes_cat[:, 0]
        heights = boxes_cat[:, 3] - boxes_cat[:, 1]
        size_mask = (widths <= max_box_size) & (heights <= max_box_size)
        boxes_cat = boxes_cat[size_mask]
        scores_cat = scores_cat[size_mask]
        if labels_cat is not None:
            labels_cat = labels_cat[size_mask]

        # Filter masks to match
        size_indices = size_mask.nonzero(as_tuple=False).flatten().tolist()
        masks_flat = [masks_flat[i] for i in size_indices]

        if boxes_cat.numel() == 0:
            if labels_cat is not None:
                return boxes_cat, scores_cat, [], labels_cat
            return boxes_cat, scores_cat, []

    keep_indices = nms(boxes_cat, scores_cat, nms_iou_threshold)

    filtered_boxes = boxes_cat[keep_indices]
    filtered_scores = scores_cat[keep_indices]
    filtered_masks = [masks_flat[i] for i in keep_indices.tolist()]
    filtered_labels = labels_cat[keep_indices] if labels_cat is not None else None

    # Apply containment suppression
    boxes_final, scores_final, keep_mask = suppress_contained_boxes(
        filtered_boxes,
        filtered_scores,
        containment_thresh=containment_thresh,
        area_ratio_thresh=area_ratio_thresh,
        return_keep=True,
    )
    final_labels = filtered_labels[keep_mask] if filtered_labels is not None else None

    # Re-map masks to final boxes with exact keep mask from containment stage.
    keep_indices_after_containment = keep_mask.nonzero(as_tuple=False).flatten().tolist()
    final_masks = [filtered_masks[i] for i in keep_indices_after_containment]

    if final_labels is not None:
        return boxes_final, scores_final, final_masks, final_labels
    return boxes_final, scores_final, final_masks


def filter_edge_boxes(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    crop_size: int,
    edge_threshold: float = 0.98,
) -> tuple:
    """
    Filter out boxes that span almost the entire crop (likely false positives).

    Args:
        boxes: Boxes in xyxy format (already in crop pixel coordinates)
        scores: Confidence scores
        crop_size: Size of the crop
        edge_threshold: Filter boxes larger than this fraction of crop

    Returns:
        Tuple of (filtered_boxes, filtered_scores)
    """
    if boxes.numel() == 0:
        return boxes, scores

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    max_size = crop_size * edge_threshold
    keep_mask = (widths < max_size) & (heights < max_size)

    return boxes[keep_mask], scores[keep_mask]
