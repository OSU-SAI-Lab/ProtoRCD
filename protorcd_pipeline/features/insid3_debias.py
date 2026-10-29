"""INSID3-style positional debiasing for DINOv3 patch features."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def build_positional_basis(
    encoder,
    image_size: int,
    svd_components: int,
    device: str,
) -> torch.Tensor:
    """Estimate the DINO positional subspace using a blank normalized image."""
    from torchvision.transforms.functional import normalize

    blank_img = normalize(
        torch.zeros(1, 3, image_size, image_size),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ).to(device)
    fmaps = encoder.get_intermediate_layers(blank_img, n=1, reshape=True)[0]
    fmaps = F.normalize(fmaps, p=2, dim=1)
    _, channels, _, _ = fmaps.shape
    comps = max(1, min(int(svd_components), channels))

    matrix = fmaps.reshape(channels, -1)
    matrix = matrix - matrix.mean(dim=1, keepdim=True)
    basis, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return basis[:, :comps].contiguous()


def debias_feature_map(fmaps_norm: torch.Tensor, positional_basis: torch.Tensor) -> torch.Tensor:
    """Project normalized patch features away from the positional subspace."""
    batch, channels, height, width = fmaps_norm.shape
    x = fmaps_norm.reshape(batch, channels, height * width)
    basis = positional_basis.to(device=x.device, dtype=x.dtype)
    projection_perp = torch.eye(channels, device=x.device, dtype=x.dtype) - basis @ basis.T
    debiased = torch.matmul(projection_perp.unsqueeze(0), x)
    debiased = debiased.reshape(batch, channels, height, width)
    return F.normalize(debiased, p=2, dim=1)
