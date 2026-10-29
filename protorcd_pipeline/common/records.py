"""Typed records used by the ProtoRCD data pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ProposalRecord:
    dataset_name: str
    image_id: str
    proposal_id: str
    prompt_type: str
    bbox_xyxy: list[float]
    mask_rle: dict[str, Any] | None
    score: float
    mask_area: int
    bbox_area: float
    image_width: int
    image_height: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManifestRecord:
    dataset_name: str
    image_id: str
    image_path: str
    width: int
    height: int
    has_gt: bool
    num_gt_boxes: int
    gt_labels: str
    split: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

