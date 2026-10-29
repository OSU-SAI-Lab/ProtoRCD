"""SAM3 detector with text-prompt, batch, and SAHI inference support."""
from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms import v2
from torchvision.ops import nms
from tqdm import tqdm

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.data_misc import FindStage, interpolate
    from sam3.model import box_ops
except ImportError as exc:
    raise ImportError(
        "Could not import the `sam3` model package. Run the local inference "
        "script with --sam3-source-root if the package is not installed in "
        "the active environment."
    ) from exc

from .sahi_utils import (
    filter_edge_boxes,
    merge_detections,
    merge_detections_with_masks,
    slice_and_batch,
)
from .utils import Config


class SAM3Detector:
    """
    SAM3-based object detector supporting SAHI (sliced) and batch (full-image) inference.

    SAHI mode (use_sahi=True): slices large images into overlapping crops, runs
    the model on each crop, then merges proposals with global NMS and containment
    suppression.

    Batch mode (use_sahi=False): runs the model on full images in parallel.

    Both modes share backbone preprocessing, text-embedding caching, confidence
    filtering, box conversion, and the final result-assembly step.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        print(f"Loading SAM3 model (compile={config.compile_model}, return_masks={config.return_masks})...")
        self.model = build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            enable_segmentation=config.return_masks,
            enable_inst_interactivity=False,
            compile=config.compile_model,
        )
        print(f"Segmentation head enabled: {self.model.segmentation_head is not None}")

        self.resolution = 1008
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(self.resolution, self.resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self._text_embeddings = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> dict:
        """Run detection on a single image, dispatching to SAHI or batch mode."""
        if self.config.use_sahi:
            return self._detect_sahi(image)
        return self.detect_batch([image])[0]

    def detect_batch(self, images: list[np.ndarray]) -> list[dict]:
        """Run detection on a list of images (non-SAHI / full-image mode)."""
        if not images:
            return []

        original_sizes = [(img.shape[0], img.shape[1]) for img in images]
        prompt_texts = self._get_text_prompts()

        with torch.inference_mode():
            tensors = [self._preprocess_image(img) for img in images]
            batch_tensor = torch.stack(tensors, dim=0)
            backbone_out = self.model.backbone.forward_image(batch_tensor)
            backbone_out.update(self._get_text_embeddings())
            results = self._decode_backbone_output(backbone_out, original_sizes, prompt_texts)

        return results

    # ------------------------------------------------------------------
    # Shared helpers (used by both modes)
    # ------------------------------------------------------------------

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        tensor = v2.functional.to_image(image).to(self.device)
        return self.transform(tensor)

    def _get_text_prompts(self) -> list[str]:
        if isinstance(self.config.text_prompt, str):
            return [self.config.text_prompt]
        return list(self.config.text_prompt)

    def _get_text_embeddings(self) -> dict:
        if self._text_embeddings is None:
            with torch.inference_mode():
                self._text_embeddings = self.model.backbone.forward_text(
                    self._get_text_prompts(), device=self.device
                )
        return self._text_embeddings

    def _build_find_input(self, batch_size: int, num_text_prompts: int) -> tuple[FindStage, torch.Tensor, torch.Tensor]:
        """Build the SAM3 FindStage query mapping for a batch × prompt grid."""
        img_ids = torch.arange(batch_size, device=self.device, dtype=torch.long).repeat_interleave(num_text_prompts)
        text_ids = torch.arange(num_text_prompts, device=self.device, dtype=torch.long).repeat(batch_size)
        find_input = FindStage(
            img_ids=img_ids,
            text_ids=text_ids,
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        return find_input, img_ids, text_ids

    def _decode_query(
        self,
        outputs: dict,
        query_idx: int,
        return_masks: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Extract and confidence-filter boxes/scores/masks for one query slot."""
        pred_boxes = outputs["pred_boxes"][query_idx]
        scores = outputs["pred_logits"][query_idx].sigmoid().squeeze(-1)
        keep = scores > self.config.confidence_threshold

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(pred_boxes[keep])
        filtered_scores = scores[keep]

        filtered_masks = None
        if return_masks and "pred_masks" in outputs:
            pred_masks = outputs["pred_masks"][query_idx]
            if keep.sum() > 0 and len(pred_masks) > 0:
                filtered_masks = pred_masks[keep]

        return boxes_xyxy, filtered_scores, filtered_masks

    def _apply_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        masks: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        label_names: list[str] | None = None,
    ) -> tuple:
        """Apply NMS and keep all aligned fields in sync."""
        if boxes.numel() == 0:
            return boxes, scores, masks, labels, label_names

        original_count = boxes.shape[0]
        keep = nms(boxes, scores, self.config.nms_iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]

        if masks is not None and hasattr(masks, "__len__") and len(masks) == original_count:
            masks = masks[keep]
        if labels is not None and len(labels) == original_count:
            labels = labels[keep]
        if label_names is not None and len(label_names) == original_count:
            label_names = [label_names[i] for i in keep.cpu().tolist()]

        return boxes, scores, masks, labels, label_names

    def _filter_boxes_by_size(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        masks=None,
        labels: torch.Tensor | None = None,
        apply_max: bool = True,
    ) -> tuple:
        """Filter boxes by min/max pixel size, keeping all aligned fields in sync."""
        if boxes.numel() == 0:
            return boxes, scores, masks, labels

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        keep = torch.ones(boxes.shape[0], dtype=torch.bool, device=boxes.device)

        if self.config.min_box_size:
            keep &= (widths >= self.config.min_box_size) & (heights >= self.config.min_box_size)
        if apply_max and self.config.max_box_size:
            keep &= (widths <= self.config.max_box_size) & (heights <= self.config.max_box_size)

        boxes = boxes[keep]
        scores = scores[keep]

        if masks is not None:
            if isinstance(masks, list):
                keep_idx = keep.nonzero(as_tuple=False).flatten().tolist()
                masks = [masks[i] for i in keep_idx]
            elif len(masks) == keep.shape[0]:
                masks = masks[keep]

        if labels is not None and len(labels) == keep.shape[0]:
            labels = labels[keep]

        return boxes, scores, masks, labels

    def _build_result(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        image_size: tuple[int, int],
        masks=None,
        labels: torch.Tensor | None = None,
        label_names: list[str] | None = None,
    ) -> dict:
        """Assemble a detection result dict from tensors."""
        result = {
            "boxes": boxes.cpu().numpy() if boxes.numel() > 0 else np.empty((0, 4)),
            "scores": scores.cpu().numpy() if scores.numel() > 0 else np.empty(0),
            "image_size": image_size,
        }
        if masks is not None and len(masks) > 0:
            result["masks"] = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
        if labels is not None and len(labels) > 0:
            result["labels"] = labels.cpu().numpy()
            result["label_names"] = label_names
        return result

    # ------------------------------------------------------------------
    # Non-SAHI (batch / full-image) decoder
    # ------------------------------------------------------------------

    def _decode_backbone_output(
        self,
        backbone_out: dict,
        original_sizes: list[tuple[int, int]],
        prompt_texts: list[str],
    ) -> list[dict]:
        """Decode a batched backbone output into per-image detection results."""
        batch_size = len(original_sizes)
        num_text_prompts = len(prompt_texts)
        num_queries = batch_size * num_text_prompts

        find_input, img_ids, text_ids = self._build_find_input(batch_size, num_text_prompts)
        outputs = self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            geometric_prompt=self.model._get_dummy_prompt(num_prompts=num_queries),
            find_target=None,
        )

        per_image_boxes: list[list] = [[] for _ in range(batch_size)]
        per_image_scores: list[list] = [[] for _ in range(batch_size)]
        per_image_masks: list[list] | None = [[] for _ in range(batch_size)] if self.config.return_masks else None
        per_image_labels: list[list] | None = [[] for _ in range(batch_size)] if num_text_prompts > 1 else None
        per_image_label_names: list[list] | None = [[] for _ in range(batch_size)] if num_text_prompts > 1 else None

        for query_idx in range(num_queries):
            image_idx = int(img_ids[query_idx].item())
            text_idx = int(text_ids[query_idx].item())
            original_h, original_w = original_sizes[image_idx]

            boxes_xyxy, filtered_scores, filtered_masks = self._decode_query(
                outputs, query_idx, self.config.return_masks
            )

            if boxes_xyxy.numel() > 0:
                scale = torch.tensor([original_w, original_h, original_w, original_h], device=self.device)
                boxes_xyxy = boxes_xyxy * scale

            boxes_xyxy, filtered_scores, filtered_masks, _ = self._filter_boxes_by_size(
                boxes_xyxy, filtered_scores, filtered_masks
            )
            if boxes_xyxy.numel() == 0:
                continue

            per_image_boxes[image_idx].append(boxes_xyxy)
            per_image_scores[image_idx].append(filtered_scores)

            if num_text_prompts > 1:
                per_image_labels[image_idx].append(
                    torch.full((filtered_scores.shape[0],), text_idx, dtype=torch.long, device=self.device)
                )
                per_image_label_names[image_idx].extend([prompt_texts[text_idx]] * filtered_scores.shape[0])

            if self.config.return_masks and filtered_masks is not None and len(filtered_masks) > 0:
                masks_interp = interpolate(
                    filtered_masks.unsqueeze(1).float(),
                    (original_h, original_w),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid()
                per_image_masks[image_idx].append((masks_interp > 0.5).squeeze(1))

        results = []
        for i, (original_h, original_w) in enumerate(original_sizes):
            boxes = torch.cat(per_image_boxes[i], dim=0) if per_image_boxes[i] else torch.empty((0, 4), device=self.device)
            scores = torch.cat(per_image_scores[i], dim=0) if per_image_scores[i] else torch.empty((0,), device=self.device)
            masks = torch.cat(per_image_masks[i], dim=0) if (per_image_masks and per_image_masks[i]) else None
            labels = torch.cat(per_image_labels[i], dim=0) if (per_image_labels and per_image_labels[i]) else None
            label_names = per_image_label_names[i] if per_image_label_names else None

            boxes, scores, masks, labels, label_names = self._apply_nms(boxes, scores, masks, labels, label_names)
            results.append(self._build_result(boxes, scores, (original_h, original_w), masks, labels, label_names))

        return results

    # ------------------------------------------------------------------
    # SAHI (sliced) decoder
    # ------------------------------------------------------------------

    def _detect_sahi(self, image: np.ndarray) -> dict:
        """Run detection with SAHI: slice → detect each crop → merge."""
        cfg = self.config
        original_h, original_w = image.shape[:2]
        prompt_texts = self._get_text_prompts()
        num_text_prompts = len(prompt_texts)

        # In multi-prompt mode we must collect intermediate masks per crop to
        # track which prompt produced which mask, so skip_intermediate_masks
        # cannot be honoured.
        collect_intermediate_masks = cfg.return_masks and (
            not cfg.sahi_skip_intermediate_masks or num_text_prompts > 1
        )
        if cfg.return_masks and cfg.sahi_skip_intermediate_masks and num_text_prompts > 1:
            print(
                "Multi-prompt SAHI with return_masks=True: "
                "collecting intermediate masks for prompt-aware output."
            )

        with torch.inference_mode():
            text_embeddings = self._get_text_embeddings()
            batches = slice_and_batch(
                image,
                slice_h=cfg.crop_size,
                slice_w=cfg.crop_size,
                overlap_h=cfg.overlap_h,
                overlap_w=cfg.overlap_w,
                batch_size=cfg.batch_size,
            )

            all_boxes: list[torch.Tensor] = []
            all_scores: list[torch.Tensor] = []
            all_masks: list = []
            all_labels: list[torch.Tensor] = []

            for batch in tqdm(batches, desc="Processing crops", leave=False):
                b_boxes, b_scores, b_masks, b_labels = self._process_sahi_batch(
                    batch, text_embeddings, num_text_prompts, collect_intermediate_masks
                )
                all_boxes.extend(b_boxes)
                all_scores.extend(b_scores)
                all_masks.extend(b_masks)
                all_labels.extend(b_labels)

            merge_kwargs = dict(
                nms_iou_threshold=cfg.nms_iou_threshold,
                containment_thresh=cfg.containment_thresh,
                area_ratio_thresh=cfg.area_ratio_thresh,
                device=self.device,
                max_box_size=cfg.max_box_size,
            )
            labels_arg = all_labels if num_text_prompts > 1 else None

            if collect_intermediate_masks and all_masks:
                merged = merge_detections_with_masks(
                    all_boxes, all_scores, all_masks, all_labels=labels_arg, **merge_kwargs
                )
                if num_text_prompts > 1:
                    final_boxes, final_scores, final_masks, final_labels = merged
                else:
                    final_boxes, final_scores, final_masks = merged
                    final_labels = None
            else:
                merged = merge_detections(
                    all_boxes, all_scores, all_labels=labels_arg, **merge_kwargs
                )
                if num_text_prompts > 1:
                    final_boxes, final_scores, final_labels = merged
                else:
                    final_boxes, final_scores = merged
                    final_labels = None
                final_masks = None

            # min_box_size filtering (max_box_size was already applied inside merge_*).
            final_boxes, final_scores, final_masks, final_labels = self._filter_boxes_by_size(
                final_boxes, final_scores, final_masks, final_labels, apply_max=False
            )

            # Two-pass mask generation: faster than collecting per-crop masks.
            if cfg.return_masks and cfg.sahi_skip_intermediate_masks and num_text_prompts == 1 and len(final_boxes) > 0:
                final_masks = self._generate_masks_from_boxes(image, final_boxes).cpu().numpy()

        label_names = [prompt_texts[int(i)] for i in final_labels.cpu().numpy()] if final_labels is not None and len(final_labels) > 0 else None
        return self._build_result(
            final_boxes, final_scores, (original_h, original_w),
            masks=final_masks,
            labels=final_labels if (final_labels is not None and len(final_labels) > 0) else None,
            label_names=label_names,
        )

    def _process_sahi_batch(
        self,
        batch: list[dict],
        text_embeddings: dict,
        num_text_prompts: int,
        collect_masks: bool,
    ) -> tuple[list, list, list, list]:
        """Process one batch of SAHI crops; always returns (boxes, scores, masks, labels)."""
        crop_tensors = []
        crop_offsets = []
        for crop_info in batch:
            crop_tensors.append(self._preprocess_image(crop_info["image"]))
            crop_offsets.append(crop_info["starting_pixel"])

        batch_size = len(crop_tensors)
        num_queries = batch_size * num_text_prompts
        backbone_out = self.model.backbone.forward_image(torch.stack(crop_tensors, dim=0))
        backbone_out.update(text_embeddings)

        find_input, img_ids, text_ids = self._build_find_input(batch_size, num_text_prompts)
        outputs = self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            geometric_prompt=self.model._get_dummy_prompt(num_prompts=num_queries),
            find_target=None,
        )

        all_boxes: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []
        all_masks: list = []
        all_labels: list[torch.Tensor] = []

        for query_idx in range(num_queries):
            crop_idx = int(img_ids[query_idx].item())
            text_idx = int(text_ids[query_idx].item())
            offset_x, offset_y = crop_offsets[crop_idx]

            boxes_xyxy, filtered_scores, filtered_masks = self._decode_query(
                outputs, query_idx, collect_masks
            )

            if boxes_xyxy.numel() == 0:
                continue

            # Scale normalized boxes to model resolution, filter edge artefacts,
            # then project from model resolution into original image coordinates.
            boxes_model = boxes_xyxy * self.resolution
            boxes_model, filtered_scores = filter_edge_boxes(boxes_model, filtered_scores, self.resolution)
            if boxes_model.numel() == 0:
                continue

            scale_factor = self.config.crop_size / self.resolution
            offset = torch.tensor([offset_x, offset_y, offset_x, offset_y], device=boxes_model.device, dtype=boxes_model.dtype)
            boxes_adjusted = boxes_model * scale_factor + offset

            all_boxes.append(boxes_adjusted)
            all_scores.append(filtered_scores)

            if num_text_prompts > 1:
                all_labels.append(
                    torch.full((filtered_scores.shape[0],), text_idx, dtype=torch.long, device=self.device)
                )

            if collect_masks and filtered_masks is not None and len(filtered_masks) > 0:
                all_masks.extend(
                    {"mask": filtered_masks[j], "offset": (offset_x, offset_y), "crop_size": self.config.crop_size}
                    for j in range(len(filtered_masks))
                )

        return all_boxes, all_scores, all_masks, all_labels

    # ------------------------------------------------------------------
    # SAHI two-pass mask generation
    # ------------------------------------------------------------------

    def _generate_masks_from_boxes(self, image: np.ndarray, boxes: torch.Tensor) -> torch.Tensor:
        """Generate full-image masks for already-merged boxes (box-conditioned pass)."""
        if len(boxes) == 0:
            return torch.empty(0, image.shape[0], image.shape[1], device=self.device)

        h, w = image.shape[:2]
        with torch.inference_mode():
            tensor = self._preprocess_image(image)
            backbone_out = self.model.backbone.forward_image(tensor.unsqueeze(0))
            backbone_out.update(self._get_text_embeddings())

            boxes_norm = box_ops.box_xyxy_to_cxcywh(
                boxes / torch.tensor([w, h, w, h], device=boxes.device)
            )
            find_input = FindStage(
                img_ids=torch.tensor([0], device=self.device, dtype=torch.long),
                text_ids=torch.tensor([0], device=self.device, dtype=torch.long),
                input_boxes=boxes_norm.unsqueeze(0),
                input_boxes_mask=torch.ones(1, len(boxes), device=self.device, dtype=torch.bool),
                input_boxes_label=None,
                input_points=None,
                input_points_mask=None,
            )
            outputs = self.model.forward_grounding(
                backbone_out=backbone_out,
                find_input=find_input,
                geometric_prompt=self.model._get_dummy_prompt(),
                find_target=None,
            )

            pred_masks = outputs["pred_masks"][0]
            return interpolate(
                pred_masks.unsqueeze(1),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
