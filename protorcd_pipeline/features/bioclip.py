"""BioCLIP proposal embedding extraction."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


class BioCLIPEmbedder:
    model_id = "bioclip"

    def __init__(self, device: str = "cuda", normalize: bool = True) -> None:
        from bioclip.predict import BaseClassifier

        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.normalize = normalize
        self.classifier = BaseClassifier(device=self.device)
        self.classifier.model.eval()

    @torch.no_grad()
    def encode_crops(self, crops: list[Image.Image], batch_size: int) -> np.ndarray:
        if not crops:
            return np.zeros((0, 0), dtype=np.float32)

        chunks = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            feats = self.classifier.create_image_features(batch, normalize=self.normalize)
            chunks.append(feats.detach().cpu())
        return torch.cat(chunks, dim=0).numpy().astype(np.float32)
