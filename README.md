# Moby-Dick

Moby-Dick is a code base for deep-learning models of the global ocean circulation. It learns the daily evolution of
the ocean from reanalysis data on unstructured triangular meshes that follow the coastlines. Its first model,
**Leviathan**, is a graph neural network derived from DeepMind's
[GraphCast](https://www.science.org/doi/10.1126/science.adi2336) and adapted from the atmosphere to the ocean: depth
levels instead of pressure levels, land masks, sea ice, and atmospheric and river forcings.

> [!NOTE]
> The project is in an early stage (alpha): models train and validate, but there are no released weights, evaluation
> tools or stable interfaces yet. `TODO.md` lists the open work.

## Data

Training uses ARCO-OCEAN, daily data at 0.25° and up to 50 depth levels that combine the GLORYS12 ocean reanalysis
with ERA5 atmospheric fields and GloFAS river discharge:

| Group | Variables |
|---|---|
| Targets, also inputs | sea surface height `zos`; temperature `thetao`, salinity `so` and currents `uo`, `vo` at depth; sea ice `siconc`, `sithick`, `usi`, `vsi` |
| Static inputs | bathymetry `glorys_deptho`; masks `glorys_mask`, `glofas_mask`; river upstream area `uparea`; `z`, `lsm` |
| Forcings | eight ERA5 atmospheric variables; river discharge `dis24` |

The raw dataset is standardized once into a local store of z-scores (`moby-dick download`): anomalies with respect to
a daily climatology divided by their standard deviation, with explicit conventions for leap days, constant series,
masked points and static fields. Training reads that store with [TensorStore](https://google.github.io/tensorstore/)
and [Grain](https://github.com/google/grain) workers.

## Models

Models live in subpackages of `mobydick` and share data loading, meshes, losses, normalization and the training loop.
A training config selects one with `[model] architecture` (`mobydick.models.MODELS`).

**Leviathan** (`mobydick.leviathan`) is an encoder–processor–decoder graph neural network:

- the **encoder** maps ocean grid points to the mesh vertices within a radius;
- the **processor** passes messages between mesh vertices for `gnn_msg_steps` steps;
- the **decoder** interpolates back to the ocean grid points from the triangles that contain them.

It predicts one day ahead: the residual with respect to the input state, scaled by the standard deviation of the daily
changes (`tendency_std`). The loss has a mask per variable and depth level, weighs levels by depth, and leaves out
land and constant points. `autoregressive.py` and `rollout.py` roll the one-step model out for longer forecasts (no
command for it yet).

**Meshes** are generated from coastline shapefiles with seamsh and gmsh,
from the size fields of a TOML config (uniform, shore proximity, bathymetry). `configs/mesh` has uniform meshes from 25
to 120 km. The best configuration so far is a 50 km mesh with 16 message passing steps
(`configs/training/production_run.toml`). An icosahedral multi-mesh as in GraphCast is available for comparison
(experimental).

**Training** is data parallel over GPUs and nodes (JAX `shard_map`), in bfloat16. It runs on any number of nodes, saves
Orbax checkpoints, validates on held-out years, logs to TensorBoard, and saves and exits cleanly when Slurm preempts the
job. Memory options (`[policy]`) trade compute for GPU memory: rematerialization, host offloading, a scanned processor
and chunked edge updates.

## Installation

The code runs in an Apptainer image built with Spack on the NVIDIA JAX container (`spack.yaml`,
`spack/templates/container/moby-dick.def`):

```bash
spack env create --dir spack-env spack.yaml
spack -e spack-env containerize > moby-dick.def
apptainer build moby-dick.sif moby-dick.def
```

Set `APPTAINERENV_PYTHONPATH` to the checkout's `src` directory to run the sources without rebuilding the image.

Without containers, Python 3.12 and GDAL 3.13.1 are needed; then:

```bash
pip install -e . --group dev
```

The distribution is called `moby-dick` and the import package `mobydick`. It is not on PyPI yet.

## Usage

Every step is a subcommand of `moby-dick`, configured by a TOML file in `configs/`:

| Command | What it does |
|---|---|
| `moby-dick download DATASET CLIMATOLOGY ANOMALY_STD OUTPUT --config configs/download/zarr_v3_lexi.toml` | standardize the raw dataset into a local Zarr store (local or S3 inputs) |
| `moby-dick download-mask DATASET OUTPUT --config configs/download/mask.toml` | write the training mask |
| `moby-dick compute-tendency-std STANDARDIZED OUTPUT --config configs/download/tendency_std.toml` | standard deviation of the daily changes, used to scale the residuals |
| `moby-dick make-ocean-mesh CONFIG OUTPUT.msh` | generate an ocean mesh (`configs/mesh/`) |
| `moby-dick make-icosahedral-mesh CONFIG OUTPUT.msh` | generate the icosahedral multi-mesh (experimental) |
| `moby-dick compute-grid-basis MESH MASK OUTPUT.npz` | linear basis functions of a mesh at the ocean grid points (draft) |
| `moby-dick train CONFIG OUTPUT TRAIN_DIR [--continue]` | train a model (`configs/training/`) and save it to `OUTPUT` |

Every command prints a banner to standard error first; set `MOBYDICK_NO_BANNER=1` to turn it off. The Grain loader
reads its settings from `MOBYDICK_GRAIN_*` environment variables (`mobydick.grain_utils.GrainRuntimeConfig`).

## Running on a cluster

`scripts/train.sh` and `scripts/download.sh` run the commands in the image with the inherited environment, binding the
data under `/data` in the container; they work on any cluster. Cluster-specific settings live under `clusters/<name>/`.
`clusters/jupiter` has the environment, Slurm jobs, a mesh study and JUBE benchmarks for the JUPITER booster (GH200
GPUs, Jülich Supercomputing Centre):

```bash
sbatch --nodes=8 clusters/jupiter/slurm/train.slurm production_run model.npz train_dir
```

## Repository layout

| Path | Contents |
|---|---|
| `src/mobydick/` | shared code: command line, data download and loading, meshes, graph network blocks, losses, normalization, training loop |
| `src/mobydick/leviathan/` | the Leviathan model and its settings |
| `configs/` | download, mesh and training settings |
| `scripts/` | portable launchers for Apptainer |
| `clusters/` | cluster-specific environments, jobs and benchmarks |
| `tests/` | pytest suite (synthetic data, CPU only) |
| `notebooks/` | exploratory notebooks, not maintained with the code |
| `docs/` | logbook, findings on the data, slides of the ESiWACE3 hackathon on JUPITER |

## Development

```bash
apptainer run moby-dick.sif python -m pytest --cov     # tests, coverage of at least 80 %
apptainer exec moby-dick.sif pre-commit run --all-files # ruff, ShellCheck, REUSE, ty
```

The same checks run on GitHub Actions for every pull request. See `CONTRIBUTING.md` for how to contribute.

## License and attribution

Moby-Dick is licensed under the Apache License, Version 2.0 (`LICENSE`); images are licensed under CC BY 4.0. It
started as a fork of [GraphCast](https://github.com/google-deepmind/graphcast) (Copyright 2023 DeepMind Technologies
Limited, Apache-2.0): files with code from GraphCast keep its copyright notice and are listed in `NOTICE`. The license
grants no rights to the names of DeepMind, Google or GraphCast, which are used here only to describe the origin of
the work. Licensing follows the [REUSE specification](https://reuse.software/spec-3.3/).

GraphCast is described in:

> Remi Lam et al., Learning skillful medium-range global weather forecasting, *Science* 382, 1416–1421 (2023),
> [doi:10.1126/science.adi2336](https://www.science.org/doi/10.1126/science.adi2336).
