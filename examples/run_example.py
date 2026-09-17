"""Run the complete registration pipeline on an HR/LR volume pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import tifffile
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rockreg.evidence import warp_to_fixed_frame  # noqa: E402
from rockreg.io import load_common_volume  # noqa: E402
from rockreg.pipeline import register_volumes  # noqa: E402


DEFAULT_HR = PROJECT_ROOT / "examples" / "hr_lr_pair" / "hr_moving.tif"
DEFAULT_LR = PROJECT_ROOT / "examples" / "hr_lr_pair" / "lr_fixed.tif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "hr_path",
        type=Path,
        nargs="?",
        default=DEFAULT_HR,
        help="moving high-resolution 3D image",
    )
    parser.add_argument(
        "lr_path",
        type=Path,
        nargs="?",
        default=DEFAULT_LR,
        help="fixed low-resolution 3D image",
    )
    parser.add_argument("--hr-spacing-um", type=float, default=2.68)
    parser.add_argument("--lr-spacing-um", type=float, default=10.72)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "example_output",
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
    if not args.hr_path.is_file():
        raise FileNotFoundError(f"HR image not found: {args.hr_path}")
    if not args.lr_path.is_file():
        raise FileNotFoundError(f"LR image not found: {args.lr_path}")

    device = resolve_device(args.device)
    common_spacing_um = float(args.lr_spacing_um)
    moving = load_common_volume(
        args.hr_path,
        source_spacing_um=float(args.hr_spacing_um),
        grid_spacing_um=common_spacing_um,
        device=device,
    )
    fixed = load_common_volume(
        args.lr_path,
        source_spacing_um=float(args.lr_spacing_um),
        grid_spacing_um=common_spacing_um,
        device=device,
    )

    started = time.perf_counter()
    result = register_volumes(moving, fixed, spacing_um=common_spacing_um)
    elapsed_seconds = time.perf_counter() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "registration_result.json"
    payload = {
        "input": {
            "hr_path": str(args.hr_path.resolve()),
            "lr_path": str(args.lr_path.resolve()),
            "hr_spacing_um": float(args.hr_spacing_um),
            "lr_spacing_um": float(args.lr_spacing_um),
            "common_spacing_um": common_spacing_um,
        },
        "device": str(device),
        "elapsed_seconds": elapsed_seconds,
        "result": result.to_dict(),
        "output_files": {},
    }

    if result.transform is not None:
        transform_path = args.output_dir / "transform.npy"
        registered_path = args.output_dir / "registered_hr_on_lr_grid.tif"
        np.save(transform_path, result.transform.numpy())
        registered, _ = warp_to_fixed_frame(
            moving,
            result.transform.to(device=device),
            tuple(fixed.shape[-3:]),
            common_spacing_um,
        )
        tifffile.imwrite(
            registered_path,
            registered.detach().cpu().numpy().astype(np.float32),
        )
        payload["output_files"] = {
            "transform": str(transform_path.resolve()),
            "registered_hr_on_lr_grid": str(registered_path.resolve()),
        }

    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
