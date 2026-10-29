"""Serve proposal crops from OSC-resident source images.

The UI never accesses the image store or its paths: it asks the backend for a proposal's
crop and gets JPEG bytes. This module resolves a proposal's ``file_name`` against a
per-session image root and cuts its ``bbox_xywh`` with Pillow.

Single- vs multi-root datasets
------------------------------
Most datasets have one image root. Pooled multi-split datasets (e.g. LoTE = train + test)
have per-split image trees, and their proposal ``file_name`` values are relative to each
split's own root with no split prefix -- so a single common ancestor cannot disambiguate
them. For those, ``load_pool_multi`` stamps every proposal's metadata with a
``_review_split`` tag; passing ``image_roots={tag: root}`` here resolves each proposal's
crop against the root for its split. This is general (any tagged multi-root pool), not a
LoTE special case.

Source images remain OSC-resident and their filesystem paths are never exposed to the
client; only the requested (small) cropped JPEG crosses the wire.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, Hashable, Optional, Sequence, Tuple

from PIL import Image

from protorcd_pipeline.triage import ProposalPool

__all__ = ["CropService"]


class CropService:
    """Cut and encode proposal crops on demand from a pool's metadata + image root(s)."""

    def __init__(
        self,
        pool: ProposalPool,
        image_root: str | Path | None = None,
        *,
        image_roots: Optional[Dict[str, str | Path]] = None,
        padding: float = 0.0,
        max_side: Optional[int] = 512,
        jpeg_quality: int = 85,
    ) -> None:
        """``image_root`` is the single/default root; ``image_roots`` maps a proposal's
        ``_review_split`` tag to a root for pooled multi-split datasets (a proposal with a
        tag present in ``image_roots`` uses that root, otherwise ``image_root``). At least
        one of the two must be given.

        ``padding`` expands the bbox by this fraction of its size on each edge (context for
        the reviewer). ``max_side`` downscales the longer crop edge to at most this many
        pixels (0/None disables). Crops are returned as JPEG bytes."""
        if image_root is None and not image_roots:
            raise ValueError("CropService needs image_root or image_roots")
        self.pool = pool
        self.image_root = Path(image_root) if image_root is not None else None
        self.image_roots = {str(k): Path(v) for k, v in (image_roots or {}).items()}
        self.padding = float(padding)
        self.max_side = max_side
        self.jpeg_quality = int(jpeg_quality)

    def _root_for(self, meta: dict) -> Path:
        """Pick the image root for a proposal: its split root if tagged+configured, else
        the default root."""
        tag = meta.get("_review_split")
        if tag is not None and str(tag) in self.image_roots:
            return self.image_roots[str(tag)]
        if self.image_root is not None:
            return self.image_root
        raise FileNotFoundError(
            f"no image root configured for split {tag!r} (have {sorted(self.image_roots)})"
        )

    def _resolve_path(self, file_name: str, meta: dict) -> Path:
        p = Path(file_name)
        return p if p.is_absolute() else self._root_for(meta) / p

    def _box_xyxy(self, bbox_xywh: Sequence[float], w: int, h: int) -> Tuple[int, int, int, int]:
        x, y, bw, bh = (float(v) for v in bbox_xywh)
        if self.padding > 0:
            x -= bw * self.padding
            y -= bh * self.padding
            bw *= 1.0 + 2 * self.padding
            bh *= 1.0 + 2 * self.padding
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(w, int(round(x + bw)))
        y1 = min(h, int(round(y + bh)))
        if x1 <= x0:
            x1 = min(w, x0 + 1)
        if y1 <= y0:
            y1 = min(h, y0 + 1)
        return x0, y0, x1, y1

    def crop_bytes(self, proposal_id: Hashable) -> bytes:
        """Return the JPEG bytes of one proposal's crop."""
        if proposal_id not in self.pool.id_to_index:
            raise KeyError(f"unknown proposal_id: {proposal_id!r}")
        meta = self.pool.metadata[self.pool.id_to_index[proposal_id]]
        file_name = meta.get("file_name")
        bbox = meta.get("bbox_xywh")
        if file_name is None or not bbox:
            raise ValueError(f"proposal {proposal_id!r} lacks file_name/bbox_xywh metadata")
        path = self._resolve_path(str(file_name), meta)
        if not path.exists():
            raise FileNotFoundError(f"source image not found: {path}")

        with Image.open(path) as im:
            im = im.convert("RGB")
            box = self._box_xyxy(bbox, im.width, im.height)
            crop = im.crop(box)
            if self.max_side:
                longest = max(crop.width, crop.height)
                if longest > self.max_side:
                    scale = self.max_side / float(longest)
                    crop = crop.resize(
                        (max(1, int(crop.width * scale)), max(1, int(crop.height * scale))),
                        Image.BILINEAR,
                    )
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=self.jpeg_quality)
            return buf.getvalue()
