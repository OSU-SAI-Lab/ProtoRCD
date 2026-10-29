"""DINOv3 crop-level proposal feature extraction."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .insid3_debias import build_positional_basis, debias_feature_map

_HUB_NAMES = {
    "small": "dinov3_vits16",
    "base": "dinov3_vitb16",
    "large": "dinov3_vitl16",
    "hugeplus": "dinov3_vith16plus",
}

_DEFAULT_WEIGHTS = {
    "hugeplus": (
        "/fs/ess/PAS2699/object_detection_datasets/training_free_paper_expts/"
        "INSID3/pretrain/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
    ),
}


class DINOv3PatchEmbedder:
    def __init__(
        self,
        model_size: str = "hugeplus",
        image_size: int = 1024,
        device: str = "cuda",
        repo_or_dir: str = "facebookresearch/dinov3",
        weights: str | None = None,
        normalize: bool = True,
        debiased: bool = False,
        svd_components: int = 500,
    ) -> None:
        if model_size not in _HUB_NAMES:
            raise ValueError(f"Unsupported DINOv3 model_size: {model_size}")
        self.model_size = model_size
        self.image_size = int(image_size)
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.normalize = bool(normalize)
        self.debiased = bool(debiased)
        self.svd_components = int(svd_components)
        self.weights = weights or _DEFAULT_WEIGHTS.get(model_size)
        self.repo_or_dir = repo_or_dir

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.encoder = self._load_encoder().to(self.device).eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.positional_basis: torch.Tensor | None = None
        if self.debiased:
            self.positional_basis = build_positional_basis(
                self.encoder,
                image_size=self.image_size,
                svd_components=self.svd_components,
                device=self.device,
            )

    @property
    def model_id(self) -> str:
        return "dinov3_debiased" if self.debiased else "dinov3"

    def _load_encoder(self):
        is_local = Path(self.repo_or_dir).exists()
        source = "local" if is_local else "github"
        kwargs = {}
        if self.weights is not None:
            kwargs["weights"] = self.weights
        return torch.hub.load(self.repo_or_dir, _HUB_NAMES[self.model_size], source=source, **kwargs)

    @torch.no_grad()
    def encode_crops(self, crops: list[Image.Image], batch_size: int) -> np.ndarray:
        """Encode proposal crops by resizing each crop and mean-pooling patch features."""
        if not crops:
            return np.zeros((0, 0), dtype=np.float32)

        chunks = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            tensor = torch.stack(
                [self.transform(crop.convert("RGB")) for crop in batch],
                dim=0,
            ).to(self.device)
            fmaps = self._feature_maps_from_tensor(tensor)
            pooled = fmaps.flatten(2).mean(dim=2)
            if self.normalize:
                pooled = F.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.detach().cpu())

        return torch.cat(chunks, dim=0).numpy().astype(np.float32)

    def _feature_maps_from_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        fmaps = self.encoder.get_intermediate_layers(tensor, n=1, reshape=True)[0]
        fmaps = F.normalize(fmaps, p=2, dim=1)
        if self.debiased:
            assert self.positional_basis is not None
            fmaps = debias_feature_map(fmaps, self.positional_basis)
        return fmaps
