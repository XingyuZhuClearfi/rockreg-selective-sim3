"""Volume loading and common-grid preprocessing."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Sequence

import numpy as np
from scipy import ndimage
import tifffile
import torch


RAW_NAME_PATTERN = re.compile(
    r"(?P<voxel_size>\d+)micron_(?P<side_length>\d+)cube_"
    r"(?P<bits>\d+)bit_(?P<endian>LE|BE)",
    re.IGNORECASE,
)


def load_tiff_stack(paths: Sequence[str | Path]) -> np.ndarray:
    ordered_paths = sorted(Path(path) for path in paths)
    if not ordered_paths:
        raise ValueError("No TIFF paths were provided.")
    return np.stack([tifffile.imread(path) for path in ordered_paths], axis=0)


def load_raw_volume(path: str | Path) -> np.ndarray:
    volume_path = Path(path)
    matches = list(RAW_NAME_PATTERN.finditer(volume_path.name))
    if not matches:
        raise ValueError(
            "RAW names must contain '<voxel>micron_<side>cube_16bit_LE|BE'."
        )
    metadata = matches[-1]
    side_length = int(metadata.group("side_length"))
    bits = int(metadata.group("bits"))
    if bits != 16:
        raise ValueError(f"Only 16-bit RAW volumes are supported, got {bits}-bit.")
    dtype = np.dtype("<u2" if metadata.group("endian").upper() == "LE" else ">u2")
    data = np.fromfile(volume_path, dtype=dtype)
    expected_voxels = side_length**3
    if data.size != expected_voxels:
        raise ValueError(
            f"RAW size mismatch: expected {expected_voxels} voxels, got {data.size}."
        )
    return data.reshape(side_length, side_length, side_length)


def load_volume(path: str | Path) -> np.ndarray:
    """Load a 3D TIFF, TIFF-stack directory, NPY, NPZ, or named RAW volume."""
    volume_path = Path(path)
    if volume_path.is_dir():
        volume = load_tiff_stack(volume_path.glob("*.tif*"))
    elif volume_path.suffix.lower() == ".raw":
        volume = load_raw_volume(volume_path)
    elif volume_path.suffix.lower() == ".npy":
        volume = np.load(volume_path, allow_pickle=False)
    elif volume_path.suffix.lower() == ".npz":
        with np.load(volume_path, allow_pickle=False) as archive:
            if len(archive.files) != 1:
                raise ValueError("NPZ input must contain exactly one array.")
            volume = archive[archive.files[0]]
    else:
        volume = tifffile.imread(volume_path)
    volume = np.asarray(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}.")
    return volume


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Robustly normalize a volume to [0, 1] using its 1st and 99th percentiles."""
    normalized = volume.astype(np.float32, copy=False)
    lower, upper = np.percentile(normalized, [1.0, 99.0])
    if upper <= lower:
        return np.zeros_like(normalized, dtype=np.float32)
    return np.clip((normalized - lower) / (upper - lower), 0.0, 1.0)


def load_common_volume(
    path: str | Path,
    source_spacing_um: float,
    grid_spacing_um: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Load, band-limit, resample, and normalize one volume on a common grid."""
    if source_spacing_um <= 0.0 or grid_spacing_um <= 0.0:
        raise ValueError("Voxel spacings must be positive.")
    volume = load_volume(path).astype(np.float32, copy=False)
    if grid_spacing_um > source_spacing_um + 1e-6:
        ratio = grid_spacing_um / source_spacing_um
        sigma = 0.5 * math.sqrt(max(ratio**2 - 1.0, 0.0))
        volume = ndimage.gaussian_filter(volume, sigma=sigma, mode="nearest")
    zoom = source_spacing_um / grid_spacing_um
    if abs(zoom - 1.0) > 1e-9:
        volume = ndimage.zoom(
            volume,
            zoom=zoom,
            order=1,
            prefilter=False,
        )
    normalized = normalize_volume(np.ascontiguousarray(volume, dtype=np.float32))
    tensor = torch.from_numpy(np.ascontiguousarray(normalized))[None, None].float()
    return tensor.to(device)
