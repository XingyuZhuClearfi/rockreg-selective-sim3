"""Training-free local refinement for one selected similarity pose."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DenseFieldCandidate:
    transform: torch.Tensor
    score: float
    overlap_fraction: float


@dataclass(frozen=True)
class PartialFieldScore:
    score: torch.Tensor
    valid_fraction: torch.Tensor
    selected_fraction: torch.Tensor
    pointwise_similarity: torch.Tensor


def analytic_intensity_field(
    volume: torch.Tensor,
    radii: tuple[int, ...] = (1, 2, 4),
) -> torch.Tensor:
    """Build the multiscale locally standardized field used for refinement."""
    if volume.ndim != 5 or volume.shape[1] != 1:
        raise ValueError(
            f"Expected volume shaped (B, 1, Z, Y, X), got {tuple(volume.shape)}."
        )
    channels = []
    for radius in radii:
        if radius <= 0:
            raise ValueError("radii must contain positive integers.")
        kernel = 2 * int(radius) + 1
        mean = F.avg_pool3d(volume, kernel, stride=1, padding=int(radius))
        second_moment = F.avg_pool3d(
            volume.square(),
            kernel,
            stride=1,
            padding=int(radius),
        )
        standard_deviation = (
            second_moment - mean.square()
        ).clamp_min(1e-6).sqrt()
        channels.append(
            ((volume - mean) / standard_deviation).clamp(-5.0, 5.0)
        )
    return F.normalize(torch.cat(channels, dim=1), dim=1)


def _sample_trilinear_zyx(
    fixed_field: torch.Tensor,
    coordinates_zyx: torch.Tensor,
) -> torch.Tensor:
    spatial_shape = fixed_field.shape[-3:]
    lower_float = torch.floor(coordinates_zyx)
    lower = lower_float.to(dtype=torch.long)
    fraction = coordinates_zyx - lower_float
    fixed_flat = fixed_field[0].reshape(fixed_field.shape[1], -1)
    sampled = fixed_field.new_zeros(
        (fixed_field.shape[1], *coordinates_zyx.shape[:-1])
    )
    z0, y0, x0 = lower.unbind(dim=-1)
    fz, fy, fx = fraction.unbind(dim=-1)
    for z_offset in (0, 1):
        z_index = (z0 + z_offset).clamp(0, spatial_shape[0] - 1)
        z_weight = fz if z_offset else 1.0 - fz
        for y_offset in (0, 1):
            y_index = (y0 + y_offset).clamp(0, spatial_shape[1] - 1)
            y_weight = fy if y_offset else 1.0 - fy
            for x_offset in (0, 1):
                x_index = (x0 + x_offset).clamp(0, spatial_shape[2] - 1)
                x_weight = fx if x_offset else 1.0 - fx
                flat_index = (
                    z_index * spatial_shape[1] * spatial_shape[2]
                    + y_index * spatial_shape[2]
                    + x_index
                )
                values = fixed_flat[:, flat_index.reshape(-1)].reshape_as(sampled)
                sampled = sampled + values * (
                    z_weight * y_weight * x_weight
                ).unsqueeze(0)
    return sampled.unsqueeze(0)


def sample_fixed_field_at_moving_points(
    fixed_field: torch.Tensor,
    moving_shape: tuple[int, int, int],
    moving_to_fixed: torch.Tensor,
    moving_spacing_um: float,
    fixed_spacing_um: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a fixed field where transformed moving voxel centres land."""
    if fixed_field.ndim != 5 or fixed_field.shape[0] != 1:
        raise ValueError("fixed_field must be shaped (1, C, Z, Y, X).")
    if moving_to_fixed.shape != (4, 4):
        raise ValueError("moving_to_fixed must be shaped (4, 4).")
    device, dtype = fixed_field.device, fixed_field.dtype
    axes = [torch.arange(size, device=device, dtype=dtype) for size in moving_shape]
    moving_zyx = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    moving_um = moving_zyx * float(moving_spacing_um)
    fixed_um = moving_um @ moving_to_fixed[:3, :3].T + moving_to_fixed[:3, 3]
    fixed_zyx = fixed_um / float(fixed_spacing_um)
    fixed_extent = fixed_field.new_tensor(fixed_field.shape[-3:]).sub(1.0).clamp_min(1.0)
    valid = ((fixed_zyx >= 0.0) & (fixed_zyx <= fixed_extent)).all(dim=-1)
    if torch.is_grad_enabled() and moving_to_fixed.requires_grad:
        sampled = _sample_trilinear_zyx(fixed_field, fixed_zyx) * valid[None, None]
    else:
        normalized_zyx = 2.0 * fixed_zyx / fixed_extent - 1.0
        grid_xyz = normalized_zyx[..., [2, 1, 0]][None]
        sampled = F.grid_sample(
            fixed_field,
            grid_xyz,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
    return sampled, valid[None, None]


def _deterministic_avg_pool3d(
    volume: torch.Tensor,
    kernel_size: int,
    padding: int,
) -> torch.Tensor:
    channels = volume.shape[1]
    weight = volume.new_full(
        (channels, 1, kernel_size, kernel_size, kernel_size),
        1.0 / float(kernel_size**3),
    )
    return F.conv3d(
        volume,
        weight,
        stride=1,
        padding=padding,
        groups=channels,
    )


def partial_field_similarity(
    moving_field: torch.Tensor,
    fixed_field: torch.Tensor,
    moving_to_fixed: torch.Tensor,
    moving_spacing_um: float,
    fixed_spacing_um: float,
    *,
    coherence_radius: int = 2,
    minimum_valid_fraction: float = 0.02,
    moving_mask: torch.Tensor | None = None,
    fixed_mask: torch.Tensor | None = None,
) -> PartialFieldScore:
    """Evaluate local-pattern agreement over a candidate's valid overlap."""
    if moving_field.ndim != 5 or moving_field.shape[0] != 1:
        raise ValueError("moving_field must be shaped (1, C, Z, Y, X).")
    if moving_field.shape[1] != fixed_field.shape[1]:
        raise ValueError(
            "moving_field and fixed_field must have the same descriptor dimension."
        )
    if moving_mask is not None and moving_mask.shape != (
        moving_field.shape[0],
        1,
        *moving_field.shape[-3:],
    ):
        raise ValueError("moving_mask must be shaped (1, 1, Z, Y, X).")
    if fixed_mask is not None and fixed_mask.shape != (
        fixed_field.shape[0],
        1,
        *fixed_field.shape[-3:],
    ):
        raise ValueError("fixed_mask must be shaped (1, 1, Z, Y, X).")
    sampling_field = (
        fixed_field
        if fixed_mask is None
        else torch.cat(
            (fixed_field, fixed_mask.to(dtype=fixed_field.dtype)),
            dim=1,
        )
    )
    sampled_fixed, valid = sample_fixed_field_at_moving_points(
        sampling_field,
        tuple(moving_field.shape[-3:]),
        moving_to_fixed.to(
            device=moving_field.device,
            dtype=moving_field.dtype,
        ),
        moving_spacing_um,
        fixed_spacing_um,
    )
    if fixed_mask is not None:
        sampled_fixed, sampled_mask = sampled_fixed[:, :-1], sampled_fixed[:, -1:]
        valid = valid & (sampled_mask >= 1.0 - 1e-6)
    if moving_mask is not None:
        valid = valid & moving_mask.to(device=valid.device, dtype=torch.bool)
    pointwise = (moving_field * sampled_fixed).sum(dim=1, keepdim=True)
    radius = max(int(coherence_radius), 0)
    if radius > 0:
        kernel = 2 * radius + 1
        valid_float = valid.to(dtype=pointwise.dtype)
        if torch.is_grad_enabled() and pointwise.requires_grad:
            local_mass = _deterministic_avg_pool3d(valid_float, kernel, radius)
            coherent = _deterministic_avg_pool3d(
                pointwise * valid_float,
                kernel,
                radius,
            )
        else:
            local_mass = F.avg_pool3d(
                valid_float,
                kernel,
                stride=1,
                padding=radius,
            )
            coherent = F.avg_pool3d(
                pointwise * valid_float,
                kernel,
                stride=1,
                padding=radius,
            )
        coherent = coherent / local_mass.clamp_min(1e-6)
        eligible = valid & (local_mass >= 0.5)
    else:
        coherent = pointwise
        eligible = valid
    values = coherent[eligible]
    denominator = (
        moving_mask.to(device=valid.device, dtype=torch.bool).sum().clamp_min(1)
        if moving_mask is not None
        else pointwise.new_tensor(float(valid.numel()))
    )
    valid_fraction = valid.to(dtype=pointwise.dtype).sum() / denominator
    if values.numel() == 0 or float(valid_fraction) < minimum_valid_fraction:
        score = pointwise.new_tensor(-1.0)
        selected_fraction = pointwise.new_zeros(())
    else:
        score = values.mean()
        selected_fraction = pointwise.new_tensor(values.numel() / int(denominator))
    return PartialFieldScore(
        score=score,
        valid_fraction=valid_fraction,
        selected_fraction=selected_fraction,
        pointwise_similarity=pointwise,
    )


def _rotation_from_vector(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind()
    zero = vector.new_zeros(())
    cross = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    angle = torch.linalg.norm(vector)
    denominator = angle.clamp_min(1e-6)
    small = angle < 1e-4
    sine_factor = torch.where(
        small,
        1.0 - angle.square() / 6.0,
        torch.sin(angle) / denominator,
    )
    cosine_factor = torch.where(
        small,
        0.5 - angle.square() / 24.0,
        (1.0 - torch.cos(angle)) / denominator.square(),
    )
    identity = torch.eye(3, device=vector.device, dtype=vector.dtype)
    return identity + sine_factor * cross + cosine_factor * (cross @ cross)


def gradient_refine_similarity_candidate(
    moving_field: torch.Tensor,
    fixed_field: torch.Tensor,
    spacing_um: float,
    candidate: DenseFieldCandidate,
    *,
    steps: int = 200,
    coherence_radius: int = 2,
    minimum_valid_fraction: float = 0.40,
    coverage_exponent: float = 0.5,
    moving_mask: torch.Tensor | None = None,
    fixed_mask: torch.Tensor | None = None,
) -> DenseFieldCandidate:
    """Refine rotation, translation, and log scale with Adam."""
    base = candidate.transform.detach()
    rotation_vector = torch.nn.Parameter(base.new_zeros(3))
    log_scale = torch.nn.Parameter(base.new_zeros(()))
    translation_voxels = torch.nn.Parameter(base.new_zeros(3))
    optimizer = torch.optim.Adam(
        [
            {"params": [rotation_vector], "lr": 0.01},
            {"params": [translation_voxels], "lr": 0.1},
            {"params": [log_scale], "lr": 0.002},
        ]
    )
    center_um = moving_field.new_tensor(
        [
            (size - 1) * float(spacing_um) / 2.0
            for size in moving_field.shape[-3:]
        ]
    )
    best = candidate
    for _ in range(max(int(steps), 0)):
        rotation = _rotation_from_vector(rotation_vector)
        scale = torch.exp(log_scale)
        delta = torch.eye(4, device=base.device, dtype=base.dtype)
        delta = delta.clone()
        delta[:3, :3] = scale * rotation
        delta[:3, 3] = center_um - delta[:3, :3] @ center_um
        transform = base @ delta
        translation_offset = translation_voxels * float(spacing_um)
        transform = torch.cat(
            [
                torch.cat(
                    [
                        transform[:3, :3],
                        (transform[:3, 3] + translation_offset)[:, None],
                    ],
                    dim=1,
                ),
                transform[3:4],
            ],
            dim=0,
        )
        result = partial_field_similarity(
            moving_field,
            fixed_field,
            transform,
            spacing_um,
            spacing_um,
            coherence_radius=coherence_radius,
            minimum_valid_fraction=minimum_valid_fraction,
            moving_mask=moving_mask,
            fixed_mask=fixed_mask,
        )
        evidence = result.score * result.valid_fraction.pow(coverage_exponent)
        if not evidence.requires_grad:
            break
        optimizer.zero_grad(set_to_none=True)
        (-evidence).backward()
        optimizer.step()
        score = float(evidence.detach())
        if score > best.score:
            best = DenseFieldCandidate(
                transform=transform.detach(),
                score=score,
                overlap_fraction=float(result.valid_fraction.detach()),
            )
    return best
