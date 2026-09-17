"""Branch-aware selective Sim(3) registration pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from rockreg.evidence import (
    EvidenceScore,
    score_candidate,
    spatial_three_way_partitions,
    warp_to_fixed_frame,
)
from rockreg.matching.structure_sim3 import propose_sim3_hypotheses
from rockreg.refinement import (
    DenseFieldCandidate,
    analytic_intensity_field,
    gradient_refine_similarity_candidate,
    sample_fixed_field_at_moving_points,
)


@dataclass(frozen=True)
class RegistrationConfig:
    """Frozen primary settings from the manuscript."""

    max_points: int = 128
    scale_count: int = 7
    candidate_scan_limit: int = 10_000
    max_branches: int = 32
    members_per_branch: int = 4
    branch_rotation_deg: float = 5.0
    branch_center_voxels: float = 2.0
    branch_log_scale: float = 0.05
    blur_sigmas: tuple[float, ...] = (0.0, 0.75, 1.5, 2.5)
    guard_band_voxels: int = 8
    minimum_voxels: int = 800
    minimum_region_voxels: int = 400
    minimum_overlap_fraction: float = 0.40
    minimum_selection_evidence: float = 0.30
    minimum_evidence_margin: float = 0.05
    minimum_acceptance_evidence: float = 0.30
    refinement_radii: tuple[int, ...] = (1, 2, 4)
    refinement_steps: int = 200
    refinement_coherence_radius: int = 2
    refinement_coverage_exponent: float = 0.5

    def __post_init__(self) -> None:
        integer_values = (
            self.max_points,
            self.scale_count,
            self.candidate_scan_limit,
            self.max_branches,
            self.members_per_branch,
            self.minimum_voxels,
            self.minimum_region_voxels,
        )
        if any(value <= 0 for value in integer_values):
            raise ValueError("All count and support settings must be positive.")
        if self.refinement_steps < 0:
            raise ValueError("refinement_steps must be nonnegative.")
        if not 0.0 < self.minimum_overlap_fraction <= 1.0:
            raise ValueError("minimum_overlap_fraction must be in (0, 1].")


@dataclass(frozen=True)
class BranchMember:
    geometry_rank: int
    support: float
    transform: torch.Tensor


@dataclass(frozen=True)
class RegistrationResult:
    """Selective result; transform is present only when the pose is accepted."""

    decision: str
    transform: torch.Tensor | None
    hypothesis_count: int
    branch_count: int
    scored_branch_count: int
    selected_branch: int | None = None
    selection_evidence: float | None = None
    evidence_margin: float | None = None
    post_refinement_selection_evidence: float | None = None
    acceptance_evidence: float | None = None
    refinement_score: float | None = None

    @property
    def accepted(self) -> bool:
        return self.decision == "accept"

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "accepted": self.accepted,
            "transform": (
                self.transform.detach().cpu().tolist()
                if self.transform is not None
                else None
            ),
            "hypothesis_count": self.hypothesis_count,
            "branch_count": self.branch_count,
            "scored_branch_count": self.scored_branch_count,
            "selected_branch": self.selected_branch,
            "selection_evidence": self.selection_evidence,
            "evidence_margin": self.evidence_margin,
            "post_refinement_selection_evidence": (
                self.post_refinement_selection_evidence
            ),
            "acceptance_evidence": self.acceptance_evidence,
            "refinement_score": self.refinement_score,
        }


@dataclass(frozen=True)
class _ScoredBranch:
    index: int
    member: BranchMember
    evidence: EvidenceScore


def transform_scale(transform: torch.Tensor) -> float:
    return float(torch.linalg.norm(transform[:3, :3], dim=0).mean())


def transform_point(point: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    return transform[:3, :3] @ point + transform[:3, 3]


def rotation_distance_deg(first: torch.Tensor, second: torch.Tensor) -> float:
    first_rotation = first[:3, :3] / max(transform_scale(first), 1e-8)
    second_rotation = second[:3, :3] / max(transform_scale(second), 1e-8)
    delta = first_rotation @ second_rotation.T
    cosine = float(((torch.trace(delta) - 1.0) / 2.0).clamp(-1.0, 1.0))
    return math.degrees(math.acos(cosine))


def cluster_pose_branches(
    transforms: torch.Tensor,
    support: torch.Tensor,
    order: torch.Tensor,
    moving_center_um: torch.Tensor,
    *,
    count: int,
    members_per_branch: int,
    scan_limit: int,
    rotation_threshold_deg: float,
    center_threshold_um: float,
    log_scale_threshold: float,
) -> list[list[BranchMember]]:
    """Retain distinct pose branches and several starts within each branch."""
    branches: list[list[BranchMember]] = []
    ranked_indices = order[:scan_limit].detach().cpu().tolist()
    for rank, hypothesis_index in enumerate(ranked_indices, start=1):
        candidate = transforms[hypothesis_index].detach().cpu()
        candidate_center = transform_point(moving_center_um, candidate)
        candidate_scale = transform_scale(candidate)
        matched_branch: list[BranchMember] | None = None
        for branch in branches:
            existing = branch[0].transform
            center_distance = float(
                torch.linalg.norm(
                    candidate_center - transform_point(moving_center_um, existing)
                )
            )
            scale_distance = abs(
                math.log(candidate_scale / max(transform_scale(existing), 1e-8))
            )
            if (
                rotation_distance_deg(candidate, existing) < rotation_threshold_deg
                and center_distance < center_threshold_um
                and scale_distance < log_scale_threshold
            ):
                matched_branch = branch
                break
        member = BranchMember(
            geometry_rank=rank,
            support=float(support[hypothesis_index].detach().cpu()),
            transform=candidate,
        )
        if matched_branch is not None:
            if len(matched_branch) < members_per_branch:
                matched_branch.append(member)
            continue
        if len(branches) < count:
            branches.append([member])
    return branches


def _score(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    transform: torch.Tensor,
    spacing_um: float,
    reference_voxels: int,
    config: RegistrationConfig,
    *,
    evaluation_partition: str = "selection",
    partition_valid: torch.Tensor | None = None,
) -> EvidenceScore:
    return score_candidate(
        moving,
        fixed,
        transform,
        spacing_um,
        reference_voxels,
        evaluation_partition=evaluation_partition,
        partition_valid=partition_valid,
        blur_sigmas=config.blur_sigmas,
        guard_band_voxels=config.guard_band_voxels,
        minimum_voxels=config.minimum_voxels,
        minimum_region_voxels=config.minimum_region_voxels,
        minimum_overlap_fraction=config.minimum_overlap_fraction,
    )


def _refinement_masks(
    moving: torch.Tensor,
    fixed_shape: tuple[int, int, int],
    transform: torch.Tensor,
    spacing_um: float,
    guard_band_voxels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    _, partition_valid = warp_to_fixed_frame(
        moving,
        transform,
        fixed_shape,
        spacing_um,
    )
    partitions = spatial_three_way_partitions(
        partition_valid,
        guard_band_voxels=guard_band_voxels,
    )
    if partitions is None:
        return None
    fit_mask, selection_mask, _ = partitions
    fixed_training_mask = (fit_mask | selection_mask)[None, None]
    sampled_training_mask, moving_valid = sample_fixed_field_at_moving_points(
        fixed_training_mask.to(dtype=moving.dtype),
        tuple(moving.shape[-3:]),
        transform,
        spacing_um,
        spacing_um,
    )
    moving_training_mask = moving_valid & (sampled_training_mask >= 1.0 - 1e-6)
    return partition_valid, moving_training_mask, fixed_training_mask


def register_volumes(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    spacing_um: float,
    config: RegistrationConfig | None = None,
) -> RegistrationResult:
    """Register two normalized volumes already expressed on one physical grid."""
    config = config or RegistrationConfig()
    if moving.ndim != 5 or tuple(moving.shape[:2]) != (1, 1):
        raise ValueError("moving must be shaped (1, 1, Z, Y, X).")
    if fixed.ndim != 5 or tuple(fixed.shape[:2]) != (1, 1):
        raise ValueError("fixed must be shaped (1, 1, Z, Y, X).")
    if moving.device != fixed.device:
        raise ValueError("moving and fixed must be on the same device.")
    if spacing_um <= 0.0:
        raise ValueError("spacing_um must be positive.")

    reference_voxels = min(moving[0, 0].numel(), fixed[0, 0].numel())
    with torch.no_grad():
        proposals = propose_sim3_hypotheses(
            moving,
            fixed,
            spacing_um=spacing_um,
            max_points=config.max_points,
            scale_count=config.scale_count,
        )
    if proposals is None:
        return RegistrationResult(
            decision="abstain_no_hypotheses",
            transform=None,
            hypothesis_count=0,
            branch_count=0,
            scored_branch_count=0,
        )

    hypothesis_count = int(proposals.transforms.shape[0])
    moving_center = torch.tensor(
        [
            (size - 1) * spacing_um / 2.0
            for size in moving.shape[-3:]
        ],
        dtype=torch.float32,
    )
    branches = cluster_pose_branches(
        proposals.transforms,
        proposals.support,
        proposals.order,
        moving_center,
        count=config.max_branches,
        members_per_branch=config.members_per_branch,
        scan_limit=config.candidate_scan_limit,
        rotation_threshold_deg=config.branch_rotation_deg,
        center_threshold_um=config.branch_center_voxels * spacing_um,
        log_scale_threshold=config.branch_log_scale,
    )
    del proposals

    scored_branches: list[_ScoredBranch] = []
    with torch.no_grad():
        for branch_index, members in enumerate(branches, start=1):
            scored_members = [
                (
                    member,
                    _score(
                        moving,
                        fixed,
                        member.transform,
                        spacing_um,
                        reference_voxels,
                        config,
                    ),
                )
                for member in members
            ]
            finite_members = [
                item for item in scored_members if item[1].value is not None
            ]
            if not finite_members:
                continue
            representative, evidence = max(
                finite_members,
                key=lambda item: float(item[1].value),
            )
            scored_branches.append(
                _ScoredBranch(branch_index, representative, evidence)
            )

    scored_branches.sort(
        key=lambda branch: float(branch.evidence.value),
        reverse=True,
    )
    if not scored_branches:
        return RegistrationResult(
            decision="abstain_no_score",
            transform=None,
            hypothesis_count=hypothesis_count,
            branch_count=len(branches),
            scored_branch_count=0,
        )

    selected = scored_branches[0]
    selection_evidence = float(selected.evidence.value)
    evidence_margin = (
        selection_evidence - float(scored_branches[1].evidence.value)
        if len(scored_branches) > 1
        else None
    )
    result_values = {
        "transform": None,
        "hypothesis_count": hypothesis_count,
        "branch_count": len(branches),
        "scored_branch_count": len(scored_branches),
        "selected_branch": selected.index,
        "selection_evidence": selection_evidence,
        "evidence_margin": evidence_margin,
    }
    if selection_evidence < config.minimum_selection_evidence:
        return RegistrationResult(decision="reject_low_evidence", **result_values)
    if (
        evidence_margin is not None
        and evidence_margin < config.minimum_evidence_margin
    ):
        return RegistrationResult(decision="abstain_ambiguous", **result_values)

    initial_transform = selected.member.transform.to(device=moving.device)
    masks = _refinement_masks(
        moving,
        tuple(fixed.shape[-3:]),
        initial_transform,
        spacing_um,
        config.guard_band_voxels,
    )
    if masks is None:
        return RegistrationResult(decision="abstain_no_partition", **result_values)
    partition_valid, moving_training_mask, fixed_training_mask = masks

    moving_field = analytic_intensity_field(moving, config.refinement_radii)
    fixed_field = analytic_intensity_field(fixed, config.refinement_radii)
    refined = gradient_refine_similarity_candidate(
        moving_field,
        fixed_field,
        spacing_um,
        DenseFieldCandidate(
            initial_transform,
            score=float("-inf"),
            overlap_fraction=0.0,
        ),
        steps=config.refinement_steps,
        coherence_radius=config.refinement_coherence_radius,
        minimum_valid_fraction=config.minimum_overlap_fraction,
        coverage_exponent=config.refinement_coverage_exponent,
        moving_mask=moving_training_mask,
        fixed_mask=fixed_training_mask,
    )
    refined_transform = refined.transform.detach()
    with torch.no_grad():
        refined_selection = _score(
            moving,
            fixed,
            refined_transform,
            spacing_um,
            reference_voxels,
            config,
            evaluation_partition="selection",
            partition_valid=partition_valid,
        )
        acceptance = _score(
            moving,
            fixed,
            refined_transform,
            spacing_um,
            reference_voxels,
            config,
            evaluation_partition="acceptance",
            partition_valid=partition_valid,
        )
    final_values = {
        **result_values,
        "post_refinement_selection_evidence": refined_selection.value,
        "acceptance_evidence": acceptance.value,
        "refinement_score": refined.score,
    }
    if acceptance.value is None:
        return RegistrationResult(
            decision="abstain_no_acceptance_score",
            **final_values,
        )
    if acceptance.value < config.minimum_acceptance_evidence:
        return RegistrationResult(
            decision="reject_final_evidence",
            **final_values,
        )
    return RegistrationResult(
        decision="accept",
        transform=refined_transform.detach().cpu(),
        **{key: value for key, value in final_values.items() if key != "transform"},
    )
