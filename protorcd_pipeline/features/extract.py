"""High-level proposal embedding extraction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from .crops import proposal_crop
from .proposals import Proposal, group_by_image, load_proposals
from .writer import ChunkedEmbeddingWriter


ModelName = Literal["bioclip", "dinov3", "dinov3_debiased"]
RegionName = Literal["box", "mask"]


@dataclass
class ExtractionConfig:
    coco_json: Path
    images_dir: Path
    detections_dir: Path
    output_dir: Path
    model: ModelName
    region: RegionName
    score_threshold: float = 0.4
    chunk_size: int = 50000
    batch_size: int = 16
    device: str = "cuda"
    overwrite: bool = False
    resume: bool = False
    max_images: int | None = None
    normalize: bool = True
    mask_background: tuple[int, int, int] = (0, 0, 0)
    dinov3_model_size: str = "hugeplus"
    dinov3_image_size: int = 1024
    dinov3_repo_or_dir: str = "facebookresearch/dinov3"
    dinov3_weights: str | None = None
    svd_components: int = 500


def resolve_image_path(images_dir: Path, file_name: str) -> Path | None:
    path = images_dir / file_name
    if path.exists():
        return path
    parts = Path(file_name).parts
    if parts and parts[0] == "images":
        stripped = images_dir / Path(*parts[1:])
        if stripped.exists():
            return stripped
    fallback = images_dir / Path(file_name).name
    if fallback.exists():
        return fallback
    return None


def proposal_metadata(proposal: Proposal, model: str, region: str) -> dict:
    return {
        "proposal_id": proposal.proposal_id,
        "image_id": proposal.image_id,
        "file_name": proposal.file_name,
        "detection_file": proposal.detection_file,
        "proposal_index": proposal.proposal_index,
        "bbox_xywh": [float(v) for v in proposal.bbox_xywh],
        "score": float(proposal.score),
        "has_mask": proposal.has_mask,
        "model": model,
        "region": region,
    }


def progress_iter(items: list[tuple[str, list[Proposal]]], description: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return items
    return tqdm(items, desc=description, unit="image", dynamic_ncols=True)


def extract_embeddings(config: ExtractionConfig) -> dict:
    proposals = load_proposals(
        coco_json=config.coco_json,
        detections_dir=config.detections_dir,
        score_threshold=config.score_threshold,
    )
    num_candidate_proposals = len(proposals)
    already_done: set[str] = set()
    metadata_path = config.output_dir / "metadata.jsonl"
    if config.resume and metadata_path.exists():
        with metadata_path.open("r") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid metadata JSON at {metadata_path}:{line_no}") from exc
                proposal_id = row.get("proposal_id")
                if proposal_id is not None:
                    already_done.add(str(proposal_id))
        if already_done:
            proposals = [p for p in proposals if p.proposal_id not in already_done]
    grouped = group_by_image(proposals)
    image_items = list(grouped.items())
    if config.max_images is not None:
        image_items = image_items[: int(config.max_images)]

    writer = ChunkedEmbeddingWriter(
        output_dir=config.output_dir,
        chunk_size=config.chunk_size,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    skipped_missing_image = 0
    skipped_empty_region = 0
    processed_images = 0

    try:
        if config.model == "bioclip":
            from .bioclip import BioCLIPEmbedder

            model = BioCLIPEmbedder(device=config.device, normalize=config.normalize)
            encode_crops = model.encode_crops
        else:
            from .dinov3 import DINOv3PatchEmbedder

            model = DINOv3PatchEmbedder(
                model_size=config.dinov3_model_size,
                image_size=config.dinov3_image_size,
                device=config.device,
                repo_or_dir=config.dinov3_repo_or_dir,
                weights=config.dinov3_weights,
                normalize=config.normalize,
                debiased=config.model == "dinov3_debiased",
                svd_components=config.svd_components,
            )
            encode_crops = model.encode_crops

        description = f"{config.model}/{config.region}"
        for file_name, image_props in progress_iter(image_items, description):
            image_path = resolve_image_path(config.images_dir, file_name)
            if image_path is None:
                skipped_missing_image += 1
                continue
            image = Image.open(image_path).convert("RGB")
            crops = []
            crop_props: list[Proposal] = []
            for proposal in image_props:
                crop = proposal_crop(
                    image,
                    proposal,
                    region=config.region,
                    mask_background=config.mask_background,
                )
                if crop is None:
                    skipped_empty_region += 1
                    continue
                crops.append(crop)
                crop_props.append(proposal)
            if crops:
                embeddings = encode_crops(crops, batch_size=config.batch_size)
                for emb, proposal in zip(embeddings, crop_props):
                    writer.add(emb, proposal_metadata(proposal, config.model, config.region))
            processed_images += 1
    finally:
        writer.close()

    manifest = {
        "config": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(config).items()
        },
        "num_input_proposals": num_candidate_proposals,
        "num_pending_proposals": len(proposals),
        "num_resumed_proposals": len(already_done),
        "num_images_with_proposals": len(grouped),
        "processed_images": processed_images,
        "skipped_missing_image": skipped_missing_image,
        "skipped_empty_region": skipped_empty_region,
    }
    return writer.write_manifest(manifest)


def print_manifest_summary(manifest: dict) -> None:
    print(json.dumps({
        "total_embeddings": manifest.get("total_embeddings"),
        "embedding_dim": manifest.get("embedding_dim"),
        "num_chunks": manifest.get("num_chunks"),
        "processed_images": manifest.get("processed_images"),
        "skipped_missing_image": manifest.get("skipped_missing_image"),
        "skipped_empty_region": manifest.get("skipped_empty_region"),
    }, indent=2))
