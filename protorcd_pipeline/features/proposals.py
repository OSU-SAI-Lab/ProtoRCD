"""Load SAM3 proposal detections and COCO image metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ImageInfo:
    image_id: int | str
    file_name: str
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class Proposal:
    image_id: int | str
    file_name: str
    detection_file: str
    proposal_index: int
    bbox_xywh: tuple[float, float, float, float]
    score: float
    segmentation: dict[str, Any] | None = None

    @property
    def proposal_id(self) -> str:
        return f"{self.image_id}:{self.proposal_index}"

    @property
    def has_mask(self) -> bool:
        return self.segmentation is not None


def _id_variants(image_id: int | str) -> list[int | str]:
    vals: list[int | str] = [image_id]
    if isinstance(image_id, str):
        try:
            vals.append(int(image_id))
        except ValueError:
            pass
    else:
        vals.append(str(image_id))
    return vals


def load_coco_images(coco_json: Path) -> dict[int | str, ImageInfo]:
    with coco_json.open("r") as f:
        coco = json.load(f)

    out: dict[int | str, ImageInfo] = {}
    for img in coco.get("images", []):
        image_id = img["id"]
        info = ImageInfo(
            image_id=image_id,
            file_name=str(img["file_name"]),
            width=img.get("width"),
            height=img.get("height"),
        )
        for key in _id_variants(image_id):
            out[key] = info
    return out


def load_coco_images_by_file(coco_json: Path) -> dict[str, ImageInfo]:
    with coco_json.open("r") as f:
        coco = json.load(f)

    out: dict[str, ImageInfo] = {}
    for img in coco.get("images", []):
        info = ImageInfo(
            image_id=img["id"],
            file_name=str(img["file_name"]),
            width=img.get("width"),
            height=img.get("height"),
        )
        out[info.file_name] = info
    return out


def iter_detection_files(detections_dir: Path) -> Iterable[Path]:
    yield from sorted(detections_dir.rglob("*_detections.json"))


def infer_image_file_name(detections_dir: Path, det_path: Path) -> str:
    rel = det_path.relative_to(detections_dir)
    suffix = "_detections"
    if rel.stem.endswith(suffix):
        return str(rel.parent / f"{rel.stem[:-len(suffix)]}.jpg")
    return str(rel.with_suffix(".jpg"))


def load_proposals(
    coco_json: Path,
    detections_dir: Path,
    score_threshold: float,
) -> list[Proposal]:
    images = load_coco_images(coco_json)
    images_by_file = load_coco_images_by_file(coco_json)
    proposals: list[Proposal] = []

    for det_path in iter_detection_files(detections_dir):
        with det_path.open("r") as f:
            det = json.load(f)

        image_id = det.get("image_id")
        inferred_file_name = infer_image_file_name(detections_dir, det_path)
        image_info = images_by_file.get(str(det.get("file_name") or inferred_file_name))
        for key in _id_variants(image_id):
            if image_info is not None:
                break
            image_info = images.get(key)
        if image_info is None and "file_name" not in det:
            continue

        file_name = str(det.get("file_name") or image_info.file_name)
        rel_det = str(det_path.relative_to(detections_dir))

        for idx, ann in enumerate(det.get("annotations", [])):
            score = float(ann.get("score", 1.0))
            if score < score_threshold:
                continue
            bbox = ann.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            x, y, w, h = (float(v) for v in bbox)
            proposals.append(
                Proposal(
                    image_id=image_id,
                    file_name=file_name,
                    detection_file=rel_det,
                    proposal_index=idx,
                    bbox_xywh=(x, y, w, h),
                    score=score,
                    segmentation=ann.get("segmentation"),
                )
            )
    return proposals


def group_by_image(proposals: Iterable[Proposal]) -> dict[str, list[Proposal]]:
    grouped: dict[str, list[Proposal]] = {}
    for proposal in proposals:
        grouped.setdefault(proposal.file_name, []).append(proposal)
    return grouped
