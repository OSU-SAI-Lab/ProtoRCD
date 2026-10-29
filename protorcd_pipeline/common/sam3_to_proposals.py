"""Convert raw SAM3 detection JSON files into standardized proposals JSONL."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Iterable

from .geometry import box_area_xyxy, clip_xyxy, xywh_to_xyxy
from .io import read_json, write_jsonl
from .records import ProposalRecord


def iter_sam3_detection_files(raw_dir: str | Path) -> Iterable[Path]:
    pattern = str(Path(raw_dir) / "**" / "*_detections.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        file_path = Path(path)
        if file_path.stat().st_size > 0:
            yield file_path


def detection_file_image_id(path: Path) -> str:
    suffix = "_detections"
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def choose_image_id(raw: dict[str, Any], path: Path, source: str) -> str:
    if source == "filename":
        return detection_file_image_id(path)
    if source == "json":
        return str(raw["image_id"])
    if source != "auto":
        raise ValueError(f"Unsupported image_id_source: {source}")

    raw_image_id = raw.get("image_id")
    if raw_image_id is None or str(raw_image_id) == "0":
        return detection_file_image_id(path)
    return str(raw_image_id)


def mask_area(segmentation: Any) -> int:
    if not segmentation:
        return 0
    if not isinstance(segmentation, dict):
        return 0
    if "size" not in segmentation or "counts" not in segmentation:
        return 0
    try:
        from pycocotools import mask as mask_util

        return int(mask_util.area(segmentation))
    except Exception:
        return 0


def image_size(raw: dict[str, Any], default_width: int = 0, default_height: int = 0) -> tuple[int, int]:
    width = raw.get("image_width", raw.get("width", default_width))
    height = raw.get("image_height", raw.get("height", default_height))
    return int(width or 0), int(height or 0)


def detections_to_proposals(
    raw_json_path: str | Path,
    dataset_name: str,
    prompt_type: str,
    image_id_source: str = "auto",
    default_width: int = 0,
    default_height: int = 0,
) -> list[ProposalRecord]:
    path = Path(raw_json_path)
    raw = read_json(path)
    image_id = choose_image_id(raw, path, image_id_source)
    width, height = image_size(raw, default_width, default_height)

    proposals: list[ProposalRecord] = []
    for idx, ann in enumerate(raw.get("annotations", [])):
        bbox_xyxy = xywh_to_xyxy(ann["bbox"])
        if width > 0 and height > 0:
            bbox_xyxy = clip_xyxy(bbox_xyxy, width=width, height=height)

        segmentation = ann.get("segmentation")
        proposals.append(
            ProposalRecord(
                dataset_name=dataset_name,
                image_id=image_id,
                proposal_id=f"{dataset_name}:{prompt_type}:{image_id}:{idx:06d}",
                prompt_type=prompt_type,
                bbox_xyxy=[round(float(v), 3) for v in bbox_xyxy],
                mask_rle=segmentation if segmentation is not None else None,
                score=float(ann.get("score", 0.0)),
                mask_area=mask_area(segmentation),
                bbox_area=float(box_area_xyxy(bbox_xyxy)),
                image_width=width,
                image_height=height,
            )
        )
    return proposals


def export_proposals_jsonl(
    raw_dir: str | Path,
    output_jsonl: str | Path,
    dataset_name: str,
    prompt_type: str,
    image_id_source: str = "auto",
) -> int:
    count = 0

    def rows() -> Iterable[dict[str, Any]]:
        nonlocal count
        for path in iter_sam3_detection_files(raw_dir):
            for proposal in detections_to_proposals(
                path,
                dataset_name=dataset_name,
                prompt_type=prompt_type,
                image_id_source=image_id_source,
            ):
                count += 1
                yield proposal.to_json()

    write_jsonl(output_jsonl, rows())
    return count

