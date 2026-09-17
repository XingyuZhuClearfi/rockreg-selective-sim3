# Branch-Aware Selective Similarity Registration of Multi-Resolution Rock Tomograms

This repository contains the Python implementation accompanying the
*Computers & Geosciences* manuscript. It registers a moving high-resolution
3D rock tomogram to a fixed low-resolution tomogram using a global similarity
transform (rotation, translation, and isotropic scale). Partial overlap is
supported, and uncertain registrations may be rejected. No training data or
learned weights are required.

## Requirements

- Python 3.10 or newer
- NumPy, SciPy, tifffile, and PyTorch

CUDA is recommended for full-size volumes, but CPU execution is supported.

## Installation

```bash
git clone https://github.com/XingyuZhuClearfi/rockreg-selective-sim3.git
cd rockreg-selective-sim3
python -m pip install -e .
```

## Example

The repository includes one HR/LR 3D image pair. Run it from the repository
root:

```bash
python examples/run_example.py
```

Use `--device cpu` to force CPU execution or `--device cuda` to require CUDA.
Results are written to `example_output/`:

- `registration_result.json`
- `transform.npy` when a pose is accepted
- `registered_hr_on_lr_grid.tif` when a pose is accepted

## Registering another pair

```bash
rockreg-register moving.tif fixed.tif \
  --moving-spacing-um 2.68 \
  --fixed-spacing-um 10.72 \
  --output-json registration.json \
  --output-transform transform.npy
```

The moving image is normally HR and the fixed image is normally LR. Supported
inputs are 3D TIFF, TIFF slice directories, `.npy`, `.npz`, and named 16-bit RAW
files. Voxel spacings are in micrometres per voxel. Run
`rockreg-register --help` for the available options.

## Output

The JSON file reports an `accept`, `reject`, or `abstain` decision and the
estimated transform when accepted. The optional 4 x 4 transform maps moving to
fixed physical coordinates in NumPy axis order `(z, y, x)`:

```text
y_fixed = transform[:3, :3] @ x_moving + transform[:3, 3]
```

## Example data

The example is derived from Alqahtani et al. (2021), *A Multi-Resolution
Complex Carbonates Micro-CT Dataset (MRCCM)*, Digital Rocks Portal:
https://doi.org/10.17612/3T36-Q704. The source data use the ODC Attribution
License 1.0.

## Citation

Please cite the accompanying *Computers & Geosciences* article when using this
code.
