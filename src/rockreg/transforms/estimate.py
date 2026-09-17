import torch


def estimate_similarity_batch(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Estimate one similarity transform per point set in a batch.

    Both tensors are shaped (batch, points, 3). Solving in closed form for every
    hypothesis at once is what makes exhaustive triplet matching affordable: the
    scale falls out of the singular values rather than being searched over.
    """
    if source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError(f"Expected source shaped (batch, points, 3), got {tuple(source.shape)}.")
    target = target.expand(source.shape[0], -1, -1)
    point_count = float(source.shape[1])
    source_center = source.mean(dim=1)
    target_center = target.mean(dim=1)
    source_centered = source - source_center[:, None]
    target_centered = target - target_center[:, None]
    covariance = source_centered.transpose(1, 2) @ target_centered / point_count
    u, singular_values, vh = torch.linalg.svd(covariance)
    correction = torch.ones(source.shape[0], 3, device=source.device, dtype=source.dtype)
    correction[:, -1] = torch.where(
        torch.linalg.det(vh.transpose(1, 2) @ u.transpose(1, 2)) < 0.0, -1.0, 1.0
    )
    rotation = vh.transpose(1, 2) @ torch.diag_embed(correction) @ u.transpose(1, 2)
    variance = source_centered.square().sum(dim=(1, 2)).div(point_count).clamp_min(1e-8)
    scale = (singular_values * correction).sum(dim=1) / variance
    linear = scale[:, None, None] * rotation
    translation = target_center - torch.einsum("bij,bj->bi", linear, source_center)
    transforms = torch.eye(4, device=source.device, dtype=source.dtype).repeat(source.shape[0], 1, 1)
    transforms[:, :3, :3] = linear
    transforms[:, :3, 3] = translation
    return transforms
