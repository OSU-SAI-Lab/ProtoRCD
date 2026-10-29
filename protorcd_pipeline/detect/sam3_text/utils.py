"""
Common utilities for SAM3 detection.
"""
from __future__ import annotations

import os
import json
import glob
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml
import numpy as np
from PIL import Image


@dataclass
class Config:
    """Detection configuration loaded from YAML."""

    # SAHI settings
    use_sahi: bool = False
    crop_size: int = 1008
    overlap_h: float = 0.25
    overlap_w: float = 0.25
    batch_size: int = 4

    # Thresholds
    confidence_threshold: float = 0.2
    nms_iou_threshold: float = 0.4
    min_box_size: int | None = None
    max_box_size: int | None = None
    containment_thresh: float = 0.85
    area_ratio_thresh: float = 0.6

    # Text prompt
    text_prompt: Union[str, list[str]] = "visual"

    # Output settings
    return_masks: bool = False
    sahi_skip_intermediate_masks: bool = True

    # Model settings
    compile_model: bool = False

    # Dataset paths
    coco_json: str | None = None
    images_dir: str | None = None
    output_base: str | None = None

    # Visualization
    vis_save_every_n: int = 2
    max_proposals: int = 100

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        config = cls()

        # Project-level defaults from configs/sam3_defaults.yaml.
        defaults = data.get("defaults", {})
        config.use_sahi = defaults.get("use_sahi", config.use_sahi)
        config.confidence_threshold = defaults.get(
            "confidence_threshold", config.confidence_threshold
        )
        config.batch_size = defaults.get("batch_size", config.batch_size)
        config.return_masks = defaults.get("return_masks", config.return_masks)
        config.vis_save_every_n = defaults.get("vis_save_every_n", config.vis_save_every_n)

        # SAHI section
        sahi = data.get("sahi", {})
        config.use_sahi = sahi.get("enabled", config.use_sahi)
        config.crop_size = sahi.get("crop_size", config.crop_size)
        config.overlap_h = sahi.get("overlap_h", config.overlap_h)
        config.overlap_w = sahi.get("overlap_w", config.overlap_w)
        config.batch_size = sahi.get("batch_size", config.batch_size)

        # Thresholds section
        thresholds = data.get("thresholds", {})
        config.confidence_threshold = thresholds.get(
            "confidence", config.confidence_threshold
        )
        config.nms_iou_threshold = thresholds.get("nms_iou", config.nms_iou_threshold)
        config.min_box_size = thresholds.get("min_box_size", config.min_box_size)
        config.max_box_size = thresholds.get("max_box_size", config.max_box_size)
        config.containment_thresh = thresholds.get("containment_thresh", config.containment_thresh)
        config.area_ratio_thresh = thresholds.get("area_ratio_thresh", config.area_ratio_thresh)

        # Text prompt: supports either a single string or a list of strings.
        text_prompt = data.get("text_prompt", config.text_prompt)
        if isinstance(text_prompt, str):
            config.text_prompt = text_prompt
        elif isinstance(text_prompt, list) and all(isinstance(p, str) for p in text_prompt):
            if not text_prompt:
                raise ValueError("text_prompt list cannot be empty")
            config.text_prompt = text_prompt
        else:
            raise TypeError("text_prompt must be a string or a list of strings")

        # Output settings
        config.return_masks = data.get("return_masks", config.return_masks)
        config.sahi_skip_intermediate_masks = data.get("sahi_skip_intermediate_masks", config.sahi_skip_intermediate_masks)

        # Model settings
        model = data.get("model", {})
        config.compile_model = model.get("compile", config.compile_model)

        # Dataset section
        dataset = data.get("dataset", {})
        config.coco_json = dataset.get("coco_json", config.coco_json)
        config.images_dir = dataset.get("images_dir", config.images_dir)
        config.output_base = dataset.get("output_base", config.output_base)

        # Visualization section
        vis = data.get("visualization", {})
        config.vis_save_every_n = vis.get("save_every_n", config.vis_save_every_n)
        config.max_proposals = vis.get("max_proposals", config.max_proposals)

        return config


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path!r}")
    return Config.from_yaml(config_path)


def get_done_ids(ann_dir: str) -> set:
    """
    Get already processed image IDs from annotation directory.

    Args:
        ann_dir: Directory containing detection JSON files

    Returns:
        Set of processed image IDs
    """
    done_ids = set()
    pattern = os.path.join(ann_dir, "**", "*_detections.json")
    skipped_invalid = 0

    for det_path in glob.glob(pattern, recursive=True):
        # Skip empty files
        if os.path.getsize(det_path) == 0:
            skipped_invalid += 1
            continue

        try:
            with open(det_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            skipped_invalid += 1
            continue

        if isinstance(data, dict) and "image_id" in data:
            done_ids.add(data["image_id"])

    if skipped_invalid > 0:
        print(f"Skipped {skipped_invalid} invalid detection JSON file(s) in {ann_dir}")

    return done_ids


def load_coco_annotations(coco_json: str) -> tuple:
    """
    Load COCO annotations and build ground truth lookup.

    Args:
        coco_json: Path to COCO format JSON file

    Returns:
        Tuple of (images, gt_by_image) where:
            - images: List of image info dicts
            - gt_by_image: Dict mapping image_id to list of ground truth boxes in xyxy format
    """
    with open(coco_json, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco.get("annotations", [])

    # Build ground truth lookup (convert from xywh to xyxy)
    gt_by_image = defaultdict(list)
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        gt_by_image[ann["image_id"]].append([x, y, x + w, y + h])

    return images, gt_by_image


def save_detection_json(
    output_path: str,
    image_id: int | str,
    detections: dict,
    include_masks: bool = False,
    file_name: str | None = None,
) -> None:
    """Save detection results in JSON format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    labels = detections.get("labels")
    label_names = detections.get("label_names")

    def encode_mask(mask):
        from pycocotools import mask as mask_util
        rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    def get_label(i: int) -> str:
        if label_names is not None and i < len(label_names):
            return str(label_names[i])
        if labels is not None and i < len(labels):
            return str(int(labels[i]))
        return ""

    raw_anns = [
        {
            "label": get_label(i),
            "score": float(score),
            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            **({"segmentation": encode_mask(detections["masks"][i])}
               if include_masks and "masks" in detections and
                  detections["masks"] is not None and i < len(detections["masks"])
               else {})
        }
        for i, ((x1, y1, x2, y2), score) in enumerate(zip(detections["boxes"], detections["scores"]))
    ]

    output = {"image_id": image_id, "annotations": raw_anns}
    if file_name is not None:
        output["file_name"] = file_name
    if "image_size" in detections:
        image_h, image_w = detections["image_size"]
        output["image_height"] = int(image_h)
        output["image_width"] = int(image_w)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file.

    Args:
        image_path: Path to image file

    Returns:
        RGB numpy array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    return np.array(Image.open(image_path).convert("RGB"))
