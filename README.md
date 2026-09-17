# Branch-Aware Selective Similarity Registration of Multi-Resolution Rock Tomograms

This folder contains the minimal runnable implementation accompanying the
*Computers & Geosciences* manuscript. It performs training-free, global
similarity registration of two three-dimensional rock tomograms under unknown
rotation, translation, isotropic scale, and partial overlap.

The method implementation and one worked HR/LR example pair are included.
Experimental result collections, full datasets, benchmark manifests, baseline
methods, training code, analysis scripts, figures, and tests are intentionally
excluded.

## Method overview

The implementation follows six stages:

1. Low-pass filter and resample both volumes onto a common physical grid.
2. Detect multiscale bright and dark Difference-of-Gaussian structures.
3. Match relative scale-space triples and solve complete Sim(3) candidates in
   closed form.
4. Group geometrically similar candidates into pose branches and rank one
   representative per branch using spatially held-out image evidence.
5. Refine the selected pose on the first two guarded evidence regions.
6. Accept the refined pose only when its fitted appearance model transfers to a
   reserved third region; otherwise return a rejection or abstention.

No training data or learned weights are required.

## Contents

```text
computers_geosciences_code/
|-- pyproject.toml
|-- README.md
|-- examples/
|   |-- run_example.py                # Complete two-image example
|   `-- hr_lr_pair/
|       |-- hr_moving.tif             # 192 x 192 x 192, 2.68 um/voxel
|       `-- lr_fixed.tif              # 64 x 64 x 64, 10.72 um/voxel
`-- src/rockreg/
    |-- io.py                         # Loading and common-grid preprocessing
    |-- matching/structure_sim3.py   # DoG triples and global Sim(3) proposals
    |-- transforms/estimate.py       # Closed-form similarity estimation
    |-- evidence.py                   # Guarded V1/V2/V3 evidence
    |-- refinement.py                 # Training-free local refinement
    |-- pipeline.py                   # Complete selective registration workflow
    `-- cli.py                        # Command-line entry point
```

## Requirements

- Python 3.10 or newer
- NumPy 1.24 or newer
- SciPy 1.10 or newer
- tifffile 2023.7.10 or newer
- PyTorch 2.1 or newer

A CUDA-capable PyTorch installation is strongly recommended for full-size 3D
volumes, although CPU execution is supported.

## Installation

Create and activate a Python environment, install the appropriate PyTorch build
for the machine, and then install this folder:

```bash
python -m pip install -e .
```

The installation exposes both the `rockreg-register` command and the `rockreg`
Python package.

## Worked HR/LR example

The included example is a real, independently acquired HR/LR pair from the
Middle Eastern Carbonate specimen in the Multi-Resolution Complex Carbonates
Micro-CT Dataset (MRCCM). The moving HR crop has already been materialized with
a 107.34-degree rotation, translation through cropping, and partial overlap with
the fixed LR volume. The realized overlap is approximately 0.71. Registration
does not receive the applied pose or overlap.

From the repository root, run the complete process by supplying the two image
paths:

```bash
python examples/run_example.py \
  examples/hr_lr_pair/hr_moving.tif \
  examples/hr_lr_pair/lr_fixed.tif \
  --device cuda
```

Because these paths and spacings are the defaults, the shorter invocation also
works:

```bash
python examples/run_example.py --device cuda
```

Use `--device cpu` on a machine without CUDA. The full primary settings are used
in both cases. The script creates `example_output/` containing:

- `registration_result.json`: decision, evidence values, and transform;
- `transform.npy`: accepted 4 x 4 moving-to-fixed transform; and
- `registered_hr_on_lr_grid.tif`: transformed, normalized HR image on the
  64 x 64 x 64 LR grid.

A validated run on the included files produced `decision: accept`, 361,372 raw
hypotheses, 32 scored branches, selection evidence 0.9562, a top-two margin of
0.1378, and reserved-region acceptance evidence 0.9747. The recovered transform
was:

```text
[[-0.302233, -0.866517,  0.423097, 614.041870],
 [ 0.818719,  0.003673,  0.592364, 102.018372],
 [-0.509474,  0.519945,  0.700930, 356.858765],
 [ 0.000000,  0.000000,  0.000000,   1.000000]]
```

Against the materialized pose used only for independent evaluation, this result
has a common-grid corner RMSE of 8.81 um (0.82 LR voxels), a rotation error of
0.459 degrees, and a relative scale error of 1.055%. These truth values are not
read by `run_example.py` or the registration package.

Small floating-point differences can occur across PyTorch versions and hardware.
Runtime is hardware dependent; the validated CUDA run took approximately 54 s.

The example volumes are derived from:

> Alqahtani, N., Mostaghimi, P., and Armstrong, R. (2021). A Multi-Resolution
> Complex Carbonates Micro-CT Dataset (MRCCM). Digital Rocks Portal.
> https://doi.org/10.17612/3T36-Q704

The source dataset is distributed under the ODC Attribution License 1.0
(ODC-By-1.0).

## Input data

The command accepts one moving volume and one fixed volume. For the intended
cross-resolution use, the high-resolution image is normally moving and the
low-resolution image is fixed.

Supported inputs are:

- a three-dimensional TIFF file;
- a directory containing a TIFF slice stack;
- a three-dimensional `.npy` file;
- an `.npz` file containing exactly one array; or
- a 16-bit RAW file whose name contains
  `<voxel>micron_<side>cube_16bit_LE` or `BE`.

Both voxel spacings must use micrometres per voxel. The default common-grid
spacing is the fixed-volume spacing.

## Command-line use

```bash
rockreg-register moving.tif fixed.tif \
  --moving-spacing-um 2.68 \
  --fixed-spacing-um 10.72 \
  --output-json registration.json \
  --output-transform transform.npy
```

The equivalent module invocation is:

```bash
python -m rockreg moving.tif fixed.tif \
  --moving-spacing-um 2.68 \
  --fixed-spacing-um 10.72
```

Use `--device cpu` to force CPU execution or `--device cuda` to require CUDA.
Run `rockreg-register --help` for all exposed options.

## Output

The JSON output contains:

- `decision`: `accept`, `reject_*`, or `abstain_*`;
- `transform`: the accepted 4 x 4 moving-to-fixed matrix, otherwise `null`;
- the number of generated hypotheses and retained/scored branches;
- pre-refinement selection evidence and the top-two branch margin;
- post-refinement selection evidence and reserved-region acceptance evidence.

The optional `.npy` transform is written only when `decision` is `accept`.
The matrix acts on physical coordinates in NumPy array-axis order `(z, y, x)`:

```text
y_fixed = transform[:3, :3] @ x_moving + transform[:3, 3]
```

The upper-left block equals `scale * rotation`; translation is in micrometres.

## Python API

```python
from rockreg.io import load_common_volume
from rockreg.pipeline import register_volumes

moving = load_common_volume(
    "moving.tif",
    source_spacing_um=2.68,
    grid_spacing_um=10.72,
    device="cuda",
)
fixed = load_common_volume(
    "fixed.tif",
    source_spacing_um=10.72,
    grid_spacing_um=10.72,
    device="cuda",
)

result = register_volumes(moving, fixed, spacing_um=10.72)
print(result.decision)
print(result.transform)
```

`register_volumes` expects normalized tensors shaped `(1, 1, Z, Y, X)` on the
same device and already expressed on one common grid. `load_common_volume`
performs these preparation steps for file inputs.

## Primary settings

The defaults reproduce the fixed method settings used in the manuscript:

| Setting | Default |
|---|---:|
| DoG levels / structure points | 7 / 128 |
| Candidate scan / branches / members | 10,000 / 32 / 4 |
| Branch rotation / centre / log-scale thresholds | 5 degrees / 2 voxels / 0.05 |
| Forward blur channels | 0, 0.75, 1.5, 2.5 voxels |
| Guard width / minimum region | 8 / 400 voxels |
| Minimum overlap fraction | 0.40 |
| Selection score / branch margin | 0.30 / 0.05 |
| Refinement steps | 200 |
| Final verification score | 0.30 |

These values are experimental operating settings rather than universal physical
constants. Thresholds should be calibrated prospectively before use on new
acquisition conditions.

## Coordinate and spacing notes

When calibrated voxel spacings are available, preprocessing removes the known
sampling ratio. The estimated similarity scale then represents only residual
global physical-size mismatch.

When spacing is unknown, pass `1` for both spacing arguments. In that case the
estimated scale combines the unknown voxel-size ratio with any physical size
difference and must not be interpreted as specimen dilation or shrinkage.

The implementation assumes one global isotropic similarity transform. It does
not model local deformation, anisotropic scale, or spatially varying scanner
response. A returned pose is selective evidence under the configured operating
thresholds, not a formal proof of geometric correctness.

## Citation

Please cite the accompanying *Computers & Geosciences* article when using this
implementation. Final bibliographic details can be added here after publication.
