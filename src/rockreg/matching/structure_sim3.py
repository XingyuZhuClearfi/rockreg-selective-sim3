"""Descriptor-free Sim(3) registration from scale-space blobs.

Both volumes are reduced to blob centres with an associated scale, every triplet is
described by five similarity invariants, and matched triplets are turned into pose
hypotheses in closed form. Nothing here learns anything, nothing compares learned
descriptors, and no stage makes a discrete topological decision -- connected
components, watersheds and persistence pairings all change identity when the
bandwidth changes, so they cannot survive a resolution gap.

Two details are load-bearing and easy to get wrong, both because they silently
degrade instead of failing:

* the neighbour list from a ball query must be SORTED by signature distance before
  it is truncated. ``scipy.spatial.cKDTree.query_ball_point`` returns members in
  unspecified order, so slicing it directly keeps an arbitrary subset. On a coarse
  grid a ball holds a few dozen members and an arbitrary six usually contains the
  true partner; on a fine grid it holds several hundred and the true partner sits
  near the middle, so it is almost never kept. That makes MORE detected structure
  produce WORSE registration, which reads like a limit of the method rather than a
  defect.

* the hypothesis ranking must be continuous. Counting mutual-nearest-neighbour
  inliers inside a hard tolerance yields an integer spanning roughly 0-20, about
  4.3 bits, and it is asked to order upwards of a hundred thousand hypotheses.
  The correct pose reliably attains the MAXIMUM count -- and so do dozens or
  hundreds of others, leaving the ordering among them to the argsort tie-break,
  which is triplet enumeration order and carries no evidence. A narrow Gaussian
  kernel over the same residuals removes the tie.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from rockreg.transforms.estimate import estimate_similarity_batch

__all__ = [
    "StructureSim3Result",
    "detect_scale_space_blobs",
    "triplet_signatures",
    "match_triplet_signatures",
    "soft_inlier_scores",
    "propose_sim3_hypotheses",
]


@dataclass(frozen=True)
class StructureSim3Result:
    """Pose hypotheses ordered by a continuous support score."""

    transforms: torch.Tensor
    support: torch.Tensor
    order: torch.Tensor
    moving_points_um: torch.Tensor
    fixed_points_um: torch.Tensor


def _gaussian_blur(volume: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur on a (1, 1, D, H, W) volume with replicate padding."""
    radius = max(int(np.ceil(3.0 * sigma)), 1)
    offsets = torch.arange(-radius, radius + 1, device=volume.device, dtype=volume.dtype)
    kernel = torch.exp(-offsets.square() / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    for dim in (-3, -2, -1):
        shape = [1, 1, 1, 1, 1]
        shape[dim] = kernel.numel()
        padding = [0] * 6
        padding[(2 - (dim + 3)) * 2] = radius
        padding[(2 - (dim + 3)) * 2 + 1] = radius
        volume = F.conv3d(F.pad(volume, padding, mode="replicate"), kernel.view(shape))
    return volume


def _subvoxel_refine(
    response: torch.Tensor, indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refine integer extrema to sub-voxel positions by a local quadratic fit."""
    extent = torch.tensor(response.shape, device=response.device)
    interior = ((indices > 0) & (indices < (extent - 1))).all(dim=1)
    indices = indices[interior]
    if indices.numel() == 0:
        return indices.float(), interior
    z, y, x = indices[:, 0], indices[:, 1], indices[:, 2]

    def at(dz: int, dy: int, dx: int) -> torch.Tensor:
        return response[z + dz, y + dy, x + dx]

    centre = at(0, 0, 0)
    gradient = torch.stack(
        [
            (at(1, 0, 0) - at(-1, 0, 0)) / 2.0,
            (at(0, 1, 0) - at(0, -1, 0)) / 2.0,
            (at(0, 0, 1) - at(0, 0, -1)) / 2.0,
        ],
        dim=1,
    )
    hzz = at(1, 0, 0) - 2.0 * centre + at(-1, 0, 0)
    hyy = at(0, 1, 0) - 2.0 * centre + at(0, -1, 0)
    hxx = at(0, 0, 1) - 2.0 * centre + at(0, 0, -1)
    hzy = (at(1, 1, 0) - at(1, -1, 0) - at(-1, 1, 0) + at(-1, -1, 0)) / 4.0
    hzx = (at(1, 0, 1) - at(1, 0, -1) - at(-1, 0, 1) + at(-1, 0, -1)) / 4.0
    hyx = (at(0, 1, 1) - at(0, 1, -1) - at(0, -1, 1) + at(0, -1, -1)) / 4.0
    hessian = torch.stack(
        [
            torch.stack([hzz, hzy, hzx], dim=1),
            torch.stack([hzy, hyy, hyx], dim=1),
            torch.stack([hzx, hyx, hxx], dim=1),
        ],
        dim=1,
    )
    hessian = hessian + 1e-6 * torch.eye(3, device=response.device).expand_as(hessian)
    offset = torch.nan_to_num(
        -torch.linalg.solve(hessian, gradient.unsqueeze(-1)).squeeze(-1)
    ).clamp(-0.5, 0.5)
    return indices.float() + offset, interior


def detect_scale_space_blobs(
    volume: torch.Tensor,
    *,
    max_points: int = 64,
    nms_radius_voxels: float = 1.5,
    base_sigma: float = 1.0,
    scale_step: float = 1.35,
    scale_count: int = 7,
    stratify_by_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return blob centres in voxels and the scale each blob was found at.

    Extrema are taken in space AND scale on a scale-normalised difference of
    Gaussians, with both polarities, so neither a global intensity threshold nor a
    choice of foreground is needed -- illumination drift and ring artefacts move
    the intensity level but not the location of an extremum.

    ``scale_count`` sets a hard ceiling on the structure size that can be seen at
    all. Only levels 1..scale_count-3 are usable, because a scale-space extremum
    needs a level above and below it, and a blob is found when its radius is about
    0.62 of the top usable sigma. Measured on isolated Gaussian blobs:

        scale_count=7   top usable sigma 3.32   radius <= 2.0 found, >= 2.5 missed
        scale_count=10  top usable sigma 8.17   radius <= 5.0 found, >= 6.0 missed

    The default suits volumes resampled to the coarser observation's own spacing,
    where rock structure spans a couple of voxels. When the scale ratio between the
    two volumes is unknown the range has to cover BOTH, so a larger scale_count is
    the safe choice -- structures beyond the ceiling are not merely down-weighted,
    they are invisible.

    ``stratify_by_scale`` splits the point budget evenly across levels instead of
    taking a global top-N by response strength. The scale-normalised response grows
    with sigma, so a global ranking drifts towards the coarsest levels and starves
    the fine ones on volumes whose structure is small. It matters because the
    fraction of detected points that turn out to be true correspondences is not
    uniform across levels: on the Berea pairs it rises monotonically with scale
    (10% at sigma 1.35 to 39% at sigma 4.48), while on MRCCM and Green River the
    coarse levels hold almost nothing. Which rule wins therefore depends on the
    volume, and neither is safe to assume.
    """
    if volume.ndim != 5:
        raise ValueError(f"Expected a (1, 1, D, H, W) volume, got {tuple(volume.shape)}.")
    sigmas = [base_sigma * scale_step**index for index in range(scale_count)]
    differences: list[torch.Tensor] = []
    previous = _gaussian_blur(volume, sigmas[0])
    for sigma in sigmas[1:]:
        current = _gaussian_blur(volume, sigma)
        differences.append((previous - current) * sigma**2)
        previous = current

    positions: list[torch.Tensor] = []
    strengths: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    for level in range(1, len(differences) - 1):
        for polarity in (1.0, -1.0):
            response = differences[level] * polarity
            local = F.max_pool3d(response, 3, stride=1, padding=1)
            neighbouring = torch.maximum(
                F.max_pool3d(differences[level - 1] * polarity, 3, stride=1, padding=1),
                F.max_pool3d(differences[level + 1] * polarity, 3, stride=1, padding=1),
            )
            extrema = (response >= local) & (response >= neighbouring) & (response > 0)
            if not extrema.any():
                continue
            refined, interior = _subvoxel_refine(response[0, 0], torch.nonzero(extrema[0, 0]))
            if refined.numel() == 0:
                continue
            positions.append(refined)
            strengths.append(response[0, 0][extrema[0, 0]][interior])
            scales.append(torch.full((refined.shape[0],), sigmas[level], device=volume.device))

    if not positions:
        empty = torch.zeros(0, 3, device=volume.device)
        return empty, empty[:, 0]

    all_positions = torch.cat(positions)
    all_strengths = torch.cat(strengths)
    all_scales = torch.cat(scales)

    def suppress(candidates: torch.Tensor, budget: int, chosen: list[int]) -> None:
        for index in candidates.tolist():
            point = all_positions[index]
            if all(
                float((point - all_positions[other]).norm()) >= nms_radius_voxels
                for other in chosen
            ):
                chosen.append(index)
                if len(chosen) >= budget:
                    return

    kept: list[int] = []
    if stratify_by_scale:
        levels = sorted({float(value) for value in all_scales.tolist()})
        per_level = max(max_points // max(len(levels), 1), 1)
        for level_index, sigma in enumerate(levels, start=1):
            members = torch.nonzero(all_scales == sigma).flatten()
            ordered = members[torch.argsort(all_strengths[members], descending=True)]
            suppress(ordered, min(per_level * level_index, max_points), kept)
            if len(kept) >= max_points:
                break
    if len(kept) < max_points:
        suppress(torch.argsort(all_strengths, descending=True), max_points, kept)
    selection = torch.tensor(kept, device=volume.device)
    return all_positions[selection], all_scales[selection]


def triplet_signatures(
    points_voxels: torch.Tensor,
    blob_sigmas: torch.Tensor,
    *,
    minimum_edge_voxels: float = 5.0,
    degeneracy_margin: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Describe every admissible point triplet by five similarity invariants.

    Edges are sorted ascending and the vertices permuted to follow them, which both
    makes the description canonical and fixes the vertex correspondence, so the six
    permutations of a matched pair never have to be tried. Two edge ratios alone
    give a two-dimensional key, which cannot index tens of thousands of triplets
    without heavy collisions; the three blob-scale ratios are free, since the
    detector already estimated a scale per point, and they take the key to five
    dimensions without touching the combinatorics.

    Near-degenerate triangles are dropped: when two edges are nearly equal the sort
    order is decided by noise, so the vertex correspondence silently permutes and no
    tolerance downstream can recover it.
    """
    count = int(points_voxels.shape[0])
    if count < 3:
        return (np.zeros((0, 3), np.int64), np.zeros((0, 3)), np.zeros((0, 5)))
    triplets = np.array(list(combinations(range(count), 3)), dtype=np.int64)
    points = points_voxels.detach().cpu().numpy()
    sigmas = blob_sigmas.detach().cpu().numpy()
    a, b, c = points[triplets[:, 0]], points[triplets[:, 1]], points[triplets[:, 2]]
    # edge i is the one OPPOSITE vertex i, so sorting edges also orders vertices
    edges = np.stack(
        [
            np.linalg.norm(b - c, axis=1),
            np.linalg.norm(a - c, axis=1),
            np.linalg.norm(a - b, axis=1),
        ],
        axis=1,
    )
    order = np.argsort(edges, axis=1)
    vertices = np.take_along_axis(triplets, order, axis=1)
    sorted_edges = np.take_along_axis(edges, order, axis=1)
    admissible = (
        (sorted_edges[:, 0] > minimum_edge_voxels)
        & (sorted_edges[:, 2] < 0.95 * (sorted_edges[:, 0] + sorted_edges[:, 1]))
        & (sorted_edges[:, 1] - sorted_edges[:, 0] > degeneracy_margin * sorted_edges[:, 2])
        & (sorted_edges[:, 2] - sorted_edges[:, 1] > degeneracy_margin * sorted_edges[:, 2])
    )
    vertices = vertices[admissible]
    sorted_edges = sorted_edges[admissible]
    shortest = sorted_edges[:, 0]
    vertex_sigmas = sigmas[vertices]
    signatures = np.stack(
        [
            np.log(sorted_edges[:, 1] / shortest),
            np.log(sorted_edges[:, 2] / shortest),
            np.log(vertex_sigmas[:, 0] / shortest),
            np.log(vertex_sigmas[:, 1] / shortest),
            np.log(vertex_sigmas[:, 2] / shortest),
        ],
        axis=1,
    )
    return vertices, sorted_edges, signatures


def match_triplet_signatures(
    moving_signatures: np.ndarray,
    fixed_signatures: np.ndarray,
    *,
    edge_tolerance: float = 0.08,
    scale_tolerance: float = 0.15,
    neighbours_per_triplet: int = 6,
) -> np.ndarray:
    """Pair moving and fixed triplets by nearest signatures under a whitened metric.

    Each dimension is divided by the residual it is expected to carry, so the ball
    is queried at unit radius per dimension and every invariant is admitted on the
    same statistical footing.

    The sort before truncation is not an optimisation. A ball can hold hundreds of
    members, ``query_ball_point`` does not order them, and keeping an arbitrary few
    discards the true partner almost every time on a finely sampled volume.
    """
    if moving_signatures.shape[0] == 0 or fixed_signatures.shape[0] == 0:
        return np.zeros((0, 2), np.int64)
    weights = np.array(
        [
            1.0 / edge_tolerance,
            1.0 / edge_tolerance,
            1.0 / scale_tolerance,
            1.0 / scale_tolerance,
            1.0 / scale_tolerance,
        ]
    )
    tree = cKDTree(fixed_signatures * weights)
    queries = moving_signatures * weights
    radius = float(np.sqrt(moving_signatures.shape[1]))
    neighbours = tree.query_ball_point(queries, r=radius)
    pairs: list[tuple[int, int]] = []
    for moving_index, candidates in enumerate(neighbours):
        if not candidates:
            continue
        if len(candidates) > neighbours_per_triplet:
            distances = np.linalg.norm(tree.data[candidates] - queries[moving_index], axis=1)
            keep = np.argsort(distances)[:neighbours_per_triplet]
            candidates = [candidates[position] for position in keep]
        pairs.extend((moving_index, fixed_index) for fixed_index in candidates)
    if not pairs:
        return np.zeros((0, 2), np.int64)
    return np.asarray(pairs, dtype=np.int64)


def soft_inlier_scores(
    transforms: torch.Tensor,
    moving_points_um: torch.Tensor,
    fixed_points_um: torch.Tensor,
    *,
    kernel_sigma_um: float,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Score hypotheses by a narrow Gaussian kernel over mutual-nearest residuals.

    Mutual agreement is required because a one-sided count is trivially inflated by
    a hypothesis that shrinks the moving cloud onto a few fixed points. The kernel
    replaces a hard-tolerance count so that the score is continuous: the correct
    pose otherwise ties with every other hypothesis at the maximum count and its
    rank is decided by enumeration order.

    ``kernel_sigma_um`` should be about one grid spacing. Widening it to the old
    inlier tolerance measurably restores the tie in a different form.
    """
    hypothesis_count = transforms.shape[0]
    moving_count = moving_points_um.shape[0]
    scores = torch.empty(hypothesis_count, device=transforms.device)
    indices = torch.arange(moving_count, device=transforms.device)[None]
    for start in range(0, hypothesis_count, batch_size):
        batch = transforms[start : start + batch_size]
        mapped = (
            moving_points_um[None] @ batch[:, :3, :3].transpose(1, 2) + batch[:, None, :3, 3]
        )
        distances = torch.cdist(mapped, fixed_points_um[None].expand(batch.shape[0], -1, -1))
        nearest_fixed = distances.argmin(dim=-1)
        nearest_moving = distances.argmin(dim=-2)
        mutual = torch.gather(nearest_moving, 1, nearest_fixed) == indices
        residuals = distances.min(dim=-1).values
        kernel = torch.exp(-residuals.square() / (2.0 * kernel_sigma_um**2))
        scores[start : start + batch_size] = (mutual.float() * kernel).sum(dim=-1)
    return scores


def propose_sim3_hypotheses(
    moving_volume: torch.Tensor,
    fixed_volume: torch.Tensor,
    *,
    spacing_um: float,
    max_points: int = 64,
    edge_tolerance: float = 0.08,
    scale_tolerance: float = 0.15,
    neighbours_per_triplet: int = 6,
    minimum_edge_voxels: float = 5.0,
    scale_count: int = 7,
    stratify_by_scale: bool = False,
    kernel_sigma_um: float | None = None,
) -> StructureSim3Result | None:
    """Propose Sim(3) poses from two volumes already resampled to a common grid.

    Returns ``None`` when either volume yields too little structure to form a
    matched triplet, which is a meaningful answer rather than a failure: below a
    handful of repeatable blobs no pose is recoverable, and reporting that is
    better than returning a pose that cannot be checked. If that happens on data
    that plainly has structure, check ``scale_count`` first -- see
    :func:`detect_scale_space_blobs` for the size ceiling it imposes.
    """
    moving_points, moving_sigmas = detect_scale_space_blobs(
        moving_volume,
        max_points=max_points,
        scale_count=scale_count,
        stratify_by_scale=stratify_by_scale,
    )
    fixed_points, fixed_sigmas = detect_scale_space_blobs(
        fixed_volume,
        max_points=max_points,
        scale_count=scale_count,
        stratify_by_scale=stratify_by_scale,
    )
    moving_vertices, _, moving_signatures = triplet_signatures(
        moving_points, moving_sigmas, minimum_edge_voxels=minimum_edge_voxels
    )
    fixed_vertices, _, fixed_signatures = triplet_signatures(
        fixed_points, fixed_sigmas, minimum_edge_voxels=minimum_edge_voxels
    )
    pairs = match_triplet_signatures(
        moving_signatures,
        fixed_signatures,
        edge_tolerance=edge_tolerance,
        scale_tolerance=scale_tolerance,
        neighbours_per_triplet=neighbours_per_triplet,
    )
    if pairs.shape[0] == 0:
        return None

    device = moving_volume.device
    moving_triplets = torch.as_tensor(moving_vertices[pairs[:, 0]], device=device)
    fixed_triplets = torch.as_tensor(fixed_vertices[pairs[:, 1]], device=device)
    moving_points_um = moving_points * spacing_um
    fixed_points_um = fixed_points * spacing_um
    transforms = estimate_similarity_batch(
        moving_points_um[moving_triplets], fixed_points_um[fixed_triplets]
    )
    support = soft_inlier_scores(
        transforms,
        moving_points_um,
        fixed_points_um,
        kernel_sigma_um=spacing_um if kernel_sigma_um is None else kernel_sigma_um,
    )
    order = torch.argsort(support, descending=True)
    return StructureSim3Result(
        transforms=transforms,
        support=support,
        order=order,
        moving_points_um=moving_points_um,
        fixed_points_um=fixed_points_um,
    )
