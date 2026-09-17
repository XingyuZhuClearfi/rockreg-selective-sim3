"""Command-line interface for registering one moving/fixed volume pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rockreg.io import load_common_volume
from rockreg.pipeline import RegistrationConfig, register_volumes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("moving", type=Path, help="moving (normally HR) 3D volume")
    parser.add_argument("fixed", type=Path, help="fixed (normally LR) 3D volume")
    parser.add_argument("--moving-spacing-um", type=float, required=True)
    parser.add_argument("--fixed-spacing-um", type=float, required=True)
    parser.add_argument(
        "--common-spacing-um",
        type=float,
        default=None,
        help="working-grid spacing; defaults to the fixed spacing",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-points", type=int, default=128)
    parser.add_argument("--scale-count", type=int, default=7)
    parser.add_argument("--candidate-scan-limit", type=int, default=10_000)
    parser.add_argument("--refinement-steps", type=int, default=200)
    parser.add_argument("--output-json", type=Path, default=Path("registration.json"))
    parser.add_argument(
        "--output-transform",
        type=Path,
        default=None,
        help="optional NPY path; written only for an accepted pose",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    common_spacing_um = float(args.common_spacing_um or args.fixed_spacing_um)
    moving = load_common_volume(
        args.moving,
        args.moving_spacing_um,
        common_spacing_um,
        device,
    )
    fixed = load_common_volume(
        args.fixed,
        args.fixed_spacing_um,
        common_spacing_um,
        device,
    )
    config = RegistrationConfig(
        max_points=args.max_points,
        scale_count=args.scale_count,
        candidate_scan_limit=args.candidate_scan_limit,
        refinement_steps=args.refinement_steps,
    )
    result = register_volumes(moving, fixed, common_spacing_um, config)
    payload = {
        "moving": str(args.moving),
        "fixed": str(args.fixed),
        "moving_spacing_um": float(args.moving_spacing_um),
        "fixed_spacing_um": float(args.fixed_spacing_um),
        "common_spacing_um": common_spacing_um,
        "device": str(device),
        "result": result.to_dict(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if result.transform is not None and args.output_transform is not None:
        args.output_transform.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_transform, result.transform.numpy())
    print(json.dumps(payload["result"], indent=2))


if __name__ == "__main__":
    main()
