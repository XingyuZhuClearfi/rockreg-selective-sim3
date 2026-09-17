"""Spatially separated appearance evidence for selective registration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EvidenceScore:
    """Predictive evidence and overlap diagnostics for one pose."""

    value: float | None
    occupied_voxels: int
    overlap_fraction: float


def warp_to_fixed_frame(
    moving_volume: torch.Tensor,
    transform: torch.Tensor,
    fixed_shape: tuple[int, int, int],
    spacing_um: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample a moving volume in the fixed frame and return its valid mask."""
    device = moving_volume.device
    inverse = torch.linalg.inv(transform.double()).float()
    axes = [
        torch.arange(size, device=device, dtype=torch.float32)
        for size in fixed_shape
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1) * spacing_um
    sampled = (grid @ inverse[:3, :3].T + inverse[:3, 3]) / spacing_um
    extent = torch.tensor(
        [size - 1 for size in moving_volume.shape[-3:]],
        device=device,
        dtype=torch.float32,
    )
    valid = ((sampled >= 0) & (sampled <= extent)).all(dim=-1)
    normalized = (2.0 * sampled / extent.clamp_min(1) - 1.0)[..., [2, 1, 0]][None]
    warped = F.grid_sample(
        moving_volume,
        normalized,
        align_corners=True,
        padding_mode="zeros",
    )
    return warped[0, 0], valid


def _gaussian_support_radius(sigma: float) -> int:
    return 0 if sigma <= 0.0 else max(1, int(math.ceil(3.0 * sigma)))


def _gaussian_blur_3d(volume: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return volume
    radius = _gaussian_support_radius(sigma)
    offsets = torch.arange(
        -radius,
        radius + 1,
        device=volume.device,
        dtype=volume.dtype,
    )
    kernel = torch.exp(-0.5 * (offsets / sigma).square())
    kernel = kernel / kernel.sum()
    result = volume[None, None]
    for shape, padding in (
        ((1, 1, -1, 1, 1), (0, 0, 0, 0, radius, radius)),
        ((1, 1, 1, -1, 1), (0, 0, radius, radius, 0, 0)),
        ((1, 1, 1, 1, -1), (radius, radius, 0, 0, 0, 0)),
    ):
        result = F.conv3d(
            F.pad(result, padding, mode="replicate"),
            kernel.view(shape),
        )
    return result[0, 0]


def spatial_three_way_partitions(
    valid: torch.Tensor,
    *,
    guard_band_voxels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Balance three slabs while keeping their feature footprints disjoint."""
    occupied = torch.nonzero(valid)
    if occupied.numel() == 0:
        return None
    spans = occupied.amax(dim=0) - occupied.amin(dim=0)
    longest = int(spans.argmax())
    reduce_axes = tuple(axis for axis in range(3) if axis != longest)
    histogram = valid.sum(dim=reduce_axes).detach().cpu().tolist()
    prefix = [0]
    for count in histogram:
        prefix.append(prefix[-1] + int(count))

    lower = int(occupied[:, longest].min())
    upper = int(occupied[:, longest].max())
    best: tuple[tuple[int, int], int, int] | None = None
    for first_cut in range(lower + guard_band_voxels, upper - guard_band_voxels):
        first_count = prefix[first_cut - guard_band_voxels + 1]
        for second_cut in range(
            first_cut + 2 * guard_band_voxels + 1,
            upper - guard_band_voxels,
        ):
            middle_count = (
                prefix[second_cut - guard_band_voxels + 1]
                - prefix[first_cut + guard_band_voxels + 1]
            )
            final_count = prefix[-1] - prefix[second_cut + guard_band_voxels + 1]
            counts = (first_count, middle_count, final_count)
            objective = (min(counts), sum(counts))
            if best is None or objective > best[0]:
                best = (objective, first_cut, second_cut)
    if best is None:
        return None

    _, first_cut, second_cut = best
    view = [-1 if axis == longest else 1 for axis in range(3)]
    coordinate = torch.arange(valid.shape[longest], device=valid.device).view(view)
    coordinate = coordinate.expand_as(valid)
    first = valid & (coordinate <= first_cut - guard_band_voxels)
    middle = valid & (coordinate > first_cut + guard_band_voxels)
    middle &= coordinate <= second_cut - guard_band_voxels
    final = valid & (coordinate > second_cut + guard_band_voxels)
    return first, middle, final


def _fit_nonnegative_ridge(
    features: torch.Tensor,
    response: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    feature_count = features.shape[1]
    ones = torch.ones(
        (features.shape[0], 1),
        device=features.device,
        dtype=features.dtype,
    )
    best_weights = torch.zeros(
        feature_count + 1,
        device=features.device,
        dtype=features.dtype,
    )
    best_weights[0] = response.mean()
    best_loss = (response - best_weights[0]).square().sum()
    for support_mask in range(1, 1 << feature_count):
        active = [
            index
            for index in range(feature_count)
            if support_mask & (1 << index)
        ]
        design = torch.cat((ones, features[:, active]), dim=1)
        gram = design.T @ design
        regularizer = torch.zeros_like(gram)
        diagonal_scale = gram.diagonal()[1:].mean().clamp_min(1e-30)
        regularizer.diagonal()[1:] = ridge * diagonal_scale
        try:
            active_weights = torch.linalg.solve(
                gram + regularizer,
                design.T @ response,
            )
        except RuntimeError:
            continue
        if not torch.isfinite(active_weights).all() or bool(
            (active_weights[1:] < -1e-10).any()
        ):
            continue
        loss = (response - design @ active_weights).square().sum()
        if loss < best_loss:
            best_loss = loss
            best_weights.zero_()
            best_weights[0] = active_weights[0]
            indices = torch.as_tensor(active, device=features.device) + 1
            best_weights[indices] = active_weights[1:]
    return best_weights


def _constrained_forward_r2(
    feature_volume: torch.Tensor,
    fixed_volume: torch.Tensor,
    fit_mask: torch.Tensor,
    evaluation_mask: torch.Tensor,
    ridge: float,
) -> float:
    design = feature_volume[fit_mask]
    response = fixed_volume[fit_mask].double()
    weights = _fit_nonnegative_ridge(design, response, ridge)
    test_design = feature_volume[evaluation_mask]
    test_response = fixed_volume[evaluation_mask].double()
    prediction = weights[0] + test_design @ weights[1:]
    residual = test_response - prediction
    return float(
        1.0 - residual.var() / test_response.var().clamp_min(1e-12)
    )


def constrained_forward_operator_features(
    warped_moving: torch.Tensor,
    *,
    blur_sigmas: tuple[float, ...] = (0.0, 0.75, 1.5, 2.5),
) -> torch.Tensor:
    """Build the nonnegative multiscale appearance-model channels."""
    if not blur_sigmas or any(sigma < 0.0 for sigma in blur_sigmas):
        raise ValueError("blur_sigmas must contain nonnegative values.")
    return torch.stack(
        [_gaussian_blur_3d(warped_moving, sigma) for sigma in blur_sigmas],
        dim=-1,
    ).double()


def heldout_constrained_forward_operator_three_way_r2(
    warped_moving: torch.Tensor,
    fixed_volume: torch.Tensor,
    valid: torch.Tensor,
    *,
    evaluation_partition: Literal["selection", "acceptance"],
    partition_valid: torch.Tensor | None = None,
    blur_sigmas: tuple[float, ...] = (0.0, 0.75, 1.5, 2.5),
    guard_band_voxels: int = 8,
    minimum_voxels: int = 800,
    minimum_region_voxels: int = 400,
    minimum_overlap_fraction: float = 0.40,
    reference_voxels: int | None = None,
    ridge: float = 1e-4,
) -> float:
    """Fit on V1 and evaluate only selection V2 or reserved verification V3."""
    if evaluation_partition not in ("selection", "acceptance"):
        raise ValueError(
            "evaluation_partition must be 'selection' or 'acceptance'."
        )
    if not blur_sigmas or any(sigma < 0.0 for sigma in blur_sigmas):
        raise ValueError("blur_sigmas must contain nonnegative values.")
    required_guard = max(_gaussian_support_radius(sigma) for sigma in blur_sigmas)
    if guard_band_voxels < required_guard:
        raise ValueError(
            "guard_band_voxels must cover the largest Gaussian support "
            f"({required_guard} voxels for blur_sigmas={blur_sigmas})."
        )
    if partition_valid is not None and partition_valid.shape != valid.shape:
        raise ValueError("partition_valid must have the same shape as valid.")
    occupied_voxels = int(valid.sum())
    if occupied_voxels < minimum_voxels:
        return float("nan")
    if reference_voxels is not None and (
        occupied_voxels < minimum_overlap_fraction * int(reference_voxels)
    ):
        return float("nan")

    partitions = spatial_three_way_partitions(
        valid if partition_valid is None else partition_valid.bool(),
        guard_band_voxels=guard_band_voxels,
    )
    if partitions is None:
        return float("nan")
    fit_mask, selection_mask, acceptance_mask = partitions
    fit_mask &= valid
    evaluation_mask = (
        selection_mask
        if evaluation_partition == "selection"
        else acceptance_mask
    )
    evaluation_mask &= valid
    if (
        int(fit_mask.sum()) < minimum_region_voxels
        or int(evaluation_mask.sum()) < minimum_region_voxels
    ):
        return float("nan")

    feature_volume = constrained_forward_operator_features(
        warped_moving,
        blur_sigmas=blur_sigmas,
    )
    return _constrained_forward_r2(
        feature_volume,
        fixed_volume,
        fit_mask,
        evaluation_mask,
        ridge,
    )


def score_candidate(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    transform: torch.Tensor,
    spacing_um: float,
    reference_voxels: int,
    *,
    evaluation_partition: Literal["selection", "acceptance"] = "selection",
    partition_valid: torch.Tensor | None = None,
    blur_sigmas: tuple[float, ...] = (0.0, 0.75, 1.5, 2.5),
    guard_band_voxels: int = 8,
    minimum_voxels: int = 800,
    minimum_region_voxels: int = 400,
    minimum_overlap_fraction: float = 0.40,
) -> EvidenceScore:
    """Warp one candidate, fit on V1, and score it on V2 or V3."""
    warped, valid = warp_to_fixed_frame(
        moving,
        transform.to(device=moving.device),
        tuple(fixed.shape[-3:]),
        spacing_um,
    )
    occupied_voxels = int(valid.sum().detach().cpu())
    value = heldout_constrained_forward_operator_three_way_r2(
        warped,
        fixed[0, 0],
        valid,
        evaluation_partition=evaluation_partition,
        partition_valid=partition_valid,
        blur_sigmas=blur_sigmas,
        guard_band_voxels=guard_band_voxels,
        minimum_voxels=minimum_voxels,
        minimum_region_voxels=minimum_region_voxels,
        minimum_overlap_fraction=minimum_overlap_fraction,
        reference_voxels=reference_voxels,
    )
    return EvidenceScore(
        value=None if not math.isfinite(value) else value,
        occupied_voxels=occupied_voxels,
        overlap_fraction=min(occupied_voxels / float(reference_voxels), 1.0),
    )
