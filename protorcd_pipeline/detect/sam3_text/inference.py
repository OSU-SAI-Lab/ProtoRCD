"""Clean local SAM3 text-prompt inference entry point.

This keeps the ProtoRCD project self-contained at the inference-script level while
still allowing the actual `sam3` model package to come from an installed
environment or an explicit source checkout.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm


def add_sam3_source_root(path: str | Path | None) -> None:
    if path is None:
        return
    source_root = Path(path).expanduser().resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


def load_detector_classes(sam3_source_root: str | Path | None):
    add_sam3_source_root(sam3_source_root)
    from .detector import SAM3Detector

    return SAM3Detector


def apply_overrides(config, args: argparse.Namespace) -> None:
    if args.use_sahi:
        config.use_sahi = True
    if args.no_sahi:
        config.use_sahi = False
    if args.confidence_threshold is not None:
        config.confidence_threshold = args.confidence_threshold
    if args.nms_iou_threshold is not None:
        config.nms_iou_threshold = args.nms_iou_threshold
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.crop_size is not None:
        config.crop_size = args.crop_size
    if args.overlap_h is not None:
        config.overlap_h = args.overlap_h
    if args.overlap_w is not None:
        config.overlap_w = args.overlap_w
    if args.min_box_size is not None:
        config.min_box_size = args.min_box_size
    if args.max_box_size is not None:
        config.max_box_size = args.max_box_size
    if args.containment_thresh is not None:
        config.containment_thresh = args.containment_thresh
    if args.area_ratio_thresh is not None:
        config.area_ratio_thresh = args.area_ratio_thresh
    if args.text_prompt is not None:
        config.text_prompt = args.text_prompt[0] if len(args.text_prompt) == 1 else args.text_prompt
    if args.return_masks:
        config.return_masks = True
    if args.no_return_masks:
        config.return_masks = False
    if args.compile_model:
        config.compile_model = True
    if args.no_compile_model:
        config.compile_model = False
    if args.coco_json is not None:
        config.coco_json = str(args.coco_json)
    if args.images_dir is not None:
        config.images_dir = str(args.images_dir)
    if args.output_base is not None:
        config.output_base = str(args.output_base)
    if args.vis_save_every_n is not None:
        config.vis_save_every_n = args.vis_save_every_n
    if args.max_proposals is not None:
        config.max_proposals = args.max_proposals


def run_single_image(detector, config, image_path: str, output_path: str | None, json_output: str | None) -> dict:
    from .utils import load_image, save_detection_json
    from .visualization import draw_detections

    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    print("Running SAM3 text-prompt detection...")
    detections = detector.detect(image)
    print(f"Found {len(detections['boxes'])} detections")

    if len(detections["boxes"]) > 0:
        print(f"Score range: {detections['scores'].min():.3f} - {detections['scores'].max():.3f}")

    if output_path:
        draw_detections(
            image,
            detections,
            output_path,
            max_proposals=config.max_proposals,
            return_masks=config.return_masks,
        )
        print(f"Visualization saved: {output_path}")

    if json_output:
        save_detection_json(
            json_output,
            image_id=Path(image_path).stem,
            detections=detections,
            include_masks=config.return_masks,
        )
        print(f"Detections saved: {json_output}")

    return detections


def run_batch_dataset(detector, config) -> None:
    from .utils import get_done_ids, load_coco_annotations, load_image, save_detection_json
    from .visualization import draw_detections

    if not config.coco_json or not config.images_dir or not config.output_base:
        raise ValueError("--coco-json, --images-dir, and --output-base are required for --batch")

    mode = "sahi" if config.use_sahi else "nosahi"
    vis_dir = Path(config.output_base) / f"sam3_{mode}_images_{config.confidence_threshold}"
    ann_dir = Path(config.output_base) / f"sam3_{mode}_annotations_{config.confidence_threshold}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading annotations from: {config.coco_json}")
    images, gt_by_image = load_coco_annotations(config.coco_json)
    print(f"Total images: {len(images)}")

    done_ids = get_done_ids(str(ann_dir))
    pending_images = [img for img in images if img["id"] not in done_ids]
    print(f"Already processed: {len(done_ids)} images")
    print(f"Remaining: {len(pending_images)} images")
    if not pending_images:
        print("All images already processed.")
        return

    if config.use_sahi:
        print("Processing with SAHI mode")
        for idx, img_info in enumerate(tqdm(pending_images, desc="Processing images")):
            image_path = Path(config.images_dir) / img_info["file_name"]
            if not image_path.exists():
                print(f"Skipping missing: {image_path}")
                continue
            image = load_image(str(image_path))
            detections = detector.detect(image)
            save_outputs_for_image(config, img_info, idx, image, detections, gt_by_image, vis_dir, ann_dir)
    else:
        print(f"Processing with batch mode (batch_size={config.batch_size})")
        for start in tqdm(range(0, len(pending_images), config.batch_size), desc="Processing batches"):
            batch_infos = pending_images[start : start + config.batch_size]
            batch_images = []
            valid_infos = []
            for img_info in batch_infos:
                image_path = Path(config.images_dir) / img_info["file_name"]
                if not image_path.exists():
                    print(f"Skipping missing: {image_path}")
                    continue
                batch_images.append(load_image(str(image_path)))
                valid_infos.append(img_info)
            if not batch_images:
                continue
            batch_detections = detector.detect_batch(batch_images)
            for local_idx, (img_info, image, detections) in enumerate(zip(valid_infos, batch_images, batch_detections)):
                save_outputs_for_image(
                    config,
                    img_info,
                    start + local_idx,
                    image,
                    detections,
                    gt_by_image,
                    vis_dir,
                    ann_dir,
                )

    print(f"Results saved to: {config.output_base}")


def save_outputs_for_image(config, img_info, index, image, detections, gt_by_image, vis_dir: Path, ann_dir: Path) -> None:
    from .utils import save_detection_json
    from .visualization import draw_detections

    image_id = img_info["id"]
    file_name = img_info["file_name"]
    rel_stem = Path(file_name).with_suffix("")
    if rel_stem.is_absolute() or any(part == ".." for part in rel_stem.parts):
        rel_stem = Path(Path(file_name).stem)

    if config.vis_save_every_n > 0 and index % config.vis_save_every_n == 0 and len(detections["boxes"]) > 0:
        vis_path = vis_dir / rel_stem.parent / f"{rel_stem.name}_detections.jpg"
        draw_detections(
            image,
            detections,
            str(vis_path),
            gt_bboxes=gt_by_image.get(image_id, []),
            max_proposals=config.max_proposals,
            return_masks=config.return_masks,
        )

    ann_path = ann_dir / rel_stem.parent / f"{rel_stem.name}_detections.json"
    save_detection_json(
        str(ann_path),
        image_id,
        detections,
        include_masks=config.return_masks,
        file_name=file_name,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local SAM3 text-prompt detection")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image", type=Path, help="Single image path")
    mode.add_argument("--batch", action="store_true", help="Run COCO-format batch inference")

    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config")
    parser.add_argument(
        "--sam3-source-root",
        type=Path,
        default=os.environ.get("PROTORCD_SAM3_SOURCE_ROOT"),
        help="Optional path to a source root containing the `sam3` model package",
    )
    parser.add_argument("--output", type=Path, help="Single-image visualization output")
    parser.add_argument("--json-output", type=Path, help="Single-image detection JSON output")
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save single-image detection JSON next to --output or the input image",
    )

    parser.add_argument("--coco-json", type=Path, help="COCO JSON for batch inference")
    parser.add_argument("--images-dir", type=Path, help="Image root for batch inference")
    parser.add_argument("--output-base", type=Path, help="Output directory for batch inference")

    parser.add_argument(
        "--text-prompt",
        action="append",
        default=None,
        help="Text prompt for SAM3. Use `visual` for broad objectness-style detection. Repeat for multi-prompt detection.",
    )
    parser.add_argument("--use-sahi", action="store_true", help="Enable SAHI slicing")
    parser.add_argument("--no-sahi", action="store_true", help="Disable SAHI slicing")
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--nms-iou-threshold", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--crop-size", type=int)
    parser.add_argument("--overlap-h", type=float)
    parser.add_argument("--overlap-w", type=float)
    parser.add_argument("--min-box-size", type=int)
    parser.add_argument("--max-box-size", type=int)
    parser.add_argument("--containment-thresh", type=float)
    parser.add_argument("--area-ratio-thresh", type=float)
    parser.add_argument("--return-masks", action="store_true")
    parser.add_argument("--no-return-masks", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--no-compile-model", action="store_true")
    parser.add_argument("--vis-save-every-n", type=int, default=None)
    parser.add_argument("--max-proposals", type=int)
    parser.add_argument("--print-config", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.use_sahi and args.no_sahi:
        raise ValueError("Choose only one of --use-sahi or --no-sahi.")
    if args.return_masks and args.no_return_masks:
        raise ValueError("Choose only one of --return-masks or --no-return-masks.")
    if args.compile_model and args.no_compile_model:
        raise ValueError("Choose only one of --compile-model or --no-compile-model.")

    from .utils import Config, load_config

    config = load_config(str(args.config)) if args.config is not None else Config()
    apply_overrides(config, args)
    if config.text_prompt == "visual":
        print("Using text prompt `visual` as the broad objectness-style SAM3 prompt.")

    if args.print_config:
        for key, value in config.__dict__.items():
            print(f"{key}: {value}")
        return 0

    SAM3Detector = load_detector_classes(args.sam3_source_root)
    detector = SAM3Detector(config)
    if args.image is not None:
        json_output = args.json_output
        if json_output is None and args.save_json:
            json_base = args.output if args.output is not None else args.image
            json_stem = json_base.with_suffix("")
            json_output = json_stem.parent / f"{json_stem.name}_detections.json"
        run_single_image(
            detector,
            config,
            str(args.image),
            str(args.output) if args.output else None,
            str(json_output) if json_output else None,
        )
    else:
        run_batch_dataset(detector, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
