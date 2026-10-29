"""Proposal pool loading: bridge stage-2 feature stores into stage-3 triage.

A "feature store" is what stage 2 (``protorcd_pipeline.features.extract``) writes:

    <feature_dir>/manifest.json     {"chunk_files": ["chunk_0000.npz", ...], ...}
    <feature_dir>/chunk_*.npz       arrays under key "embeddings"  (rows = proposals)
    <feature_dir>/metadata.jsonl    one JSON object per proposal, in embedding order,
                                    with at least: proposal_id, image_id, file_name,
                                    bbox_xywh.

Triage needs TWO stores per pool -- one per representation:
    background (DINOv3) and foreground (BioCLIP-2) -- extracted from the SAME proposals
    in the SAME order, so row i is the same proposal in both.

``labels_from_coco`` reproduces the research GT-labeling (IoU match to COCO boxes) for
offline mode; live UI mode never needs it.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .acquisition import l2_normalize

__all__ = [
    "ProposalPool",
    "load_feature_store",
    "load_pool",
    "load_pool_multi",
    "labels_from_coco",
    "xywh_to_xyxy",
    "pairwise_iou_one_to_many",
]


# --------------------------------------------------------------------------- #
# Geometry (ported from evaluate_active_triage_features.py)                    #
# --------------------------------------------------------------------------- #
def xywh_to_xyxy(box: Sequence[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in box]
    return np.asarray([x, y, x + max(0.0, w), y + max(0.0, h)], dtype=np.float32)


def pairwise_iou_one_to_many(box: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if gt.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], gt[:, 0])
    y1 = np.maximum(box[1], gt[:, 1])
    x2 = np.minimum(box[2], gt[:, 2])
    y2 = np.minimum(box[3], gt[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.maximum(0.0, gt[:, 2] - gt[:, 0]) * np.maximum(0.0, gt[:, 3] - gt[:, 1])
    denom = np.maximum(area_a + area_b - inter, 1e-12)
    return (inter / denom).astype(np.float32)


# --------------------------------------------------------------------------- #
# Feature-store loading                                                        #
# --------------------------------------------------------------------------- #
def load_feature_store(feature_dir: str | Path) -> Tuple[np.ndarray, List[dict], dict]:
    """Load one feature store -> (L2-normalized embeddings, metadata rows, manifest)."""
    feature_dir = Path(feature_dir)
    manifest_path = feature_dir / "manifest.json"
    metadata_path = feature_dir / "metadata.jsonl"
    if not manifest_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing manifest/metadata in {feature_dir}")
    manifest = json.loads(manifest_path.read_text())
    chunks = []
    for rel in manifest["chunk_files"]:
        arr = np.load(feature_dir / rel)["embeddings"].astype(np.float32)
        chunks.append(arr)
    embeddings = l2_normalize(np.vstack(chunks)) if chunks else np.zeros((0, 0), dtype=np.float32)
    metadata = [json.loads(line) for line in metadata_path.open("r") if line.strip()]
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            f"{feature_dir}: metadata rows {len(metadata)} != embeddings {embeddings.shape[0]}"
        )
    return embeddings, metadata, manifest


def _proposal_id(meta: dict, fallback_index: int):
    for key in ("proposal_id", "id"):
        if key in meta and meta[key] not in (None, ""):
            return meta[key]
    return int(fallback_index)


@dataclass
class ProposalPool:
    """A pool of detector proposals with both representations' embeddings.

    Row i of ``bg_x`` and ``fg_x`` is the same proposal; ``proposal_ids[i]`` is its
    stable id (used by LabelProviders); ``metadata[i]`` its full record.
    ``gt_labels`` is present only in offline mode (else None).
    """

    bg_x: np.ndarray                 # (N, Dbg) DINOv3, L2-normalized
    fg_x: np.ndarray                 # (N, Dfg) BioCLIP-2, L2-normalized
    proposal_ids: List              # (N,) stable ids
    metadata: List[dict]            # (N,) per-proposal records
    gt_labels: Optional[np.ndarray] = None   # (N,) offline only; 0 == background

    def __post_init__(self):
        n = self.bg_x.shape[0]
        if self.fg_x.shape[0] != n:
            raise ValueError(f"bg/fg pool size mismatch: {n} vs {self.fg_x.shape[0]}")
        if len(self.proposal_ids) != n or len(self.metadata) != n:
            raise ValueError("proposal_ids/metadata length must match embedding rows")
        self.id_to_index = {pid: i for i, pid in enumerate(self.proposal_ids)}
        if len(self.id_to_index) != n:
            raise ValueError("proposal_ids are not unique")

    @property
    def size(self) -> int:
        return self.bg_x.shape[0]


def load_pool(
    background_feature_dir: str | Path,
    foreground_feature_dir: str | Path,
    *,
    gt_labels: Optional[np.ndarray] = None,
) -> ProposalPool:
    """Load a two-representation proposal pool from two feature stores.

    The two stores must describe the SAME proposals in the SAME order; this is checked
    by proposal_id (falling back to image_id+bbox) alignment.
    """
    bg_x, bg_meta, _ = load_feature_store(background_feature_dir)
    fg_x, fg_meta, _ = load_feature_store(foreground_feature_dir)
    if len(bg_meta) != len(fg_meta):
        raise ValueError(
            f"background/foreground stores have different proposal counts: "
            f"{len(bg_meta)} vs {len(fg_meta)}"
        )
    ids = [_proposal_id(m, i) for i, m in enumerate(bg_meta)]
    fg_ids = [_proposal_id(m, i) for i, m in enumerate(fg_meta)]
    if ids != fg_ids:
        raise ValueError(
            "background and foreground feature stores are not row-aligned "
            "(proposal_id order differs). Re-extract both from the same proposals."
        )
    return ProposalPool(
        bg_x=bg_x, fg_x=fg_x, proposal_ids=ids, metadata=bg_meta, gt_labels=gt_labels
    )


def load_pool_multi(
    background_feature_dirs: Sequence[str | Path],
    foreground_feature_dirs: Sequence[str | Path],
    *,
    split_tags: Optional[Sequence[str]] = None,
    gt_labels: Optional[np.ndarray] = None,
) -> ProposalPool:
    """Load a pool by CONCATENATING several feature stores, in the given order.

    For datasets published as multiple splits (e.g. LoTE train + test) whose proposals
    are pooled for review. Stores are stacked in list order (row ``i`` of the result is
    the ``i``-th proposal across the concatenation). Because a proposal_id need only be
    unique WITHIN a store, ids are qualified by their split tag (``"<tag>::<id>"``) to
    keep them globally unique -- the row position remains the stable identity used by the
    GT provider. ``split_tags`` defaults to ``s0, s1, ...``.

    ``background_feature_dirs`` and ``foreground_feature_dirs`` must be the same length
    and describe the SAME proposals in the SAME per-store order.
    """
    bg_dirs = list(background_feature_dirs)
    fg_dirs = list(foreground_feature_dirs)
    if len(bg_dirs) != len(fg_dirs):
        raise ValueError("background/foreground store lists must have equal length")
    if not bg_dirs:
        raise ValueError("no feature stores given")
    if split_tags is None:
        split_tags = [f"s{i}" for i in range(len(bg_dirs))]
    elif len(split_tags) != len(bg_dirs):
        raise ValueError("split_tags length must match the number of stores")

    bg_chunks: List[np.ndarray] = []
    fg_chunks: List[np.ndarray] = []
    ids: List = []
    metadata: List[dict] = []
    for tag, bg_dir, fg_dir in zip(split_tags, bg_dirs, fg_dirs):
        bg_x, bg_meta, _ = load_feature_store(bg_dir)
        fg_x, fg_meta, _ = load_feature_store(fg_dir)
        if len(bg_meta) != len(fg_meta):
            raise ValueError(
                f"store '{tag}' bg/fg proposal-count mismatch: {len(bg_meta)} vs {len(fg_meta)}"
            )
        bg_ids = [_proposal_id(m, i) for i, m in enumerate(bg_meta)]
        fg_ids = [_proposal_id(m, i) for i, m in enumerate(fg_meta)]
        if bg_ids != fg_ids:
            raise ValueError(f"store '{tag}' bg/fg feature stores are not row-aligned")
        bg_chunks.append(bg_x)
        fg_chunks.append(fg_x)
        for m, pid in zip(bg_meta, bg_ids):
            row = dict(m)
            row["_review_split"] = tag
            metadata.append(row)
            ids.append(f"{tag}::{pid}")
    return ProposalPool(
        bg_x=np.vstack(bg_chunks),
        fg_x=np.vstack(fg_chunks),
        proposal_ids=ids,
        metadata=metadata,
        gt_labels=gt_labels,
    )


# --------------------------------------------------------------------------- #
# Offline GT labeling (reproducibility mode)                                   #
# --------------------------------------------------------------------------- #
def _load_coco_label_tables(path: Path):
    coco = json.loads(Path(path).read_text())
    names = {0: "background"}
    names.update({int(c["id"]): str(c["name"]) for c in coco.get("categories", [])})
    image_files = {int(im["id"]): str(im["file_name"]) for im in coco.get("images", [])}
    by_image: Dict[int, list] = defaultdict(list)
    by_file: Dict[str, list] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        image_id = int(ann["image_id"])
        item = (xywh_to_xyxy(ann["bbox"]), int(ann["category_id"]))
        by_image[image_id].append(item)
        if image_id in image_files:
            by_file[image_files[image_id]].append(item)
    return by_image, by_file, names


def _label_metadata(
    metadata: Sequence[dict],
    by_image: Dict[int, list],
    by_file: Dict[str, list],
    iou_threshold: float,
) -> np.ndarray:
    """Best-IoU GT class per proposal (0 == background), preferring file-name matches.

    Matches the research ``label_metadata``: try ``by_file`` first (globally unique file
    names), fall back to ``by_image`` only when there is no file match.
    """
    labels = np.zeros((len(metadata),), dtype=np.int64)
    for i, meta in enumerate(metadata):
        gt_items = []
        if meta.get("file_name") is not None:
            gt_items = by_file.get(str(meta["file_name"]), [])
        if not gt_items and meta.get("image_id") is not None:
            gt_items = by_image.get(int(meta["image_id"]), [])
        if not gt_items:
            continue
        gt_boxes = np.stack([it[0] for it in gt_items], axis=0)
        gt_labels = np.asarray([it[1] for it in gt_items], dtype=np.int64)
        ious = pairwise_iou_one_to_many(xywh_to_xyxy(meta["bbox_xywh"]), gt_boxes)
        best = int(np.argmax(ious))
        if float(ious[best]) >= float(iou_threshold):
            labels[i] = int(gt_labels[best])
    return labels


def labels_from_coco(
    metadata: Sequence[dict],
    coco_path: str | Path,
    iou_threshold: float = 0.25,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Assign a GT class (0 == background) to each proposal by best-IoU match.

    Reproduces ``label_metadata`` from the research code. Returns ``(labels, names)``.
    """
    by_image, by_file, names = _load_coco_label_tables(Path(coco_path))
    return _label_metadata(metadata, by_image, by_file, iou_threshold), names


def labels_from_coco_multi(
    metadata: Sequence[dict],
    coco_paths: Sequence[str | Path],
    iou_threshold: float = 0.25,
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Like :func:`labels_from_coco` but merges several COCO files first.

    For multi-split pools (e.g. LoTE train + test) whose proposals are labeled against
    the union of the splits' annotations -- matching the research explicit-all path,
    which extends ``by_image``/``by_file`` across every split before matching.
    """
    merged_image: Dict[int, list] = defaultdict(list)
    merged_file: Dict[str, list] = defaultdict(list)
    names: Dict[int, str] = {0: "background"}
    for path in coco_paths:
        by_image, by_file, split_names = _load_coco_label_tables(Path(path))
        names.update(split_names)
        for key, items in by_image.items():
            merged_image[key].extend(items)
        for key, items in by_file.items():
            merged_file[key].extend(items)
    return _label_metadata(metadata, merged_image, merged_file, iou_threshold), names
