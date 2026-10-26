# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A fork of DeepMind's GraphCast adapted to **global ocean circulation** (ARCO-OCEAN / GLORYS data, depth levels
instead of pressure levels). The model runs on unstructured triangular ocean meshes (seamsh/gmsh `.msh` files),
not the icosahedral multi-mesh. The main target system is JUPITER (JSC, GH200 GPUs, Slurm partition `booster`).
The top-level README is mostly the upstream GraphCast README and is out of date; `TODO.md` lists open work.

## Environment and commands

Code runs inside an Apptainer image (`graphcast.sif`, built from `spack/templates/container/graphcast.def`:
NVIDIA JAX container `nvcr.io/nvidia/jax:26.09-py3` plus GDAL/PROJ from Spack). `clusters/jupiter/.env` sets
`APPTAINERENV_PYTHONPATH` to `src/`, so source edits take effect without rebuilding the image. Python is 3.12.

- CLI (`src/graphcast/__init__.py`, click): `graphcast train CONFIG.toml OUTPUT TRAIN_CKPT`,
  `graphcast make-ocean-mesh` (mesh from `configs/mesh/*.toml`), `graphcast compute` (basis).
- Training run on JUPITER: `sbatch [--account=… --nodes=N] clusters/jupiter/slurm/train.slurm [--env-file F]
  [--config-dir D] [--tb-dir D] CONFIG_NAME OUTPUT_NAME TRAIN_CKPT_NAME`. It sources `clusters/jupiter/.env`
  (data paths, Grain and zarr settings) and calls `scripts/train.sh`, which binds the dataset, climatology,
  anomaly_std, tendency_std (SquashFS or directories) and `GEOMETRY_DIR` under `/data` in the container.
  Use `--reservation=develbooster` for short tests on up to 4 nodes.
- Standardized anomaly dataset: `scripts/download.sh DATASET CLIMATOLOGY ANOMALY_STD OUTPUT
  [--config configs/download/{zarr_v2,zarr_v3,zarr_v3_lexi}.toml]` runs `graphcast download` (`downloader.py`),
  which applies the preprocessing of `ARCOTimeseriesSource` once. Inputs are local paths or fsspec URLs
  (`s3://ogs-arco-ocean/{datasets,climatology,anomaly_std}/res=0p25`; S3 credentials: `storage_options.profile`,
  e.g. `xarray-s3-writer`, or `anon = true`). On a compute node (no internet: local copies in
  `/e/fscratch/e-dev-2026d09-119/ogs-arco-ocean-50*`): `sbatch clusters/jupiter/slurm/download.slurm` with the same
  arguments. Local stores are read and written with TensorStore (zarr-python caps at ~0.85 GB/s, ~5 cores, whatever
  the threads); TensorStore writes shards in lexicographic order only, so `zarr_v3` (morton) is written by
  zarr-python. Full 10.5 TB (5.1 TB on disk), three booster jobs at once: v3 lexicographic 34 min, v2 and v3 morton
  2 h each; on a login node (20-core, 143 GiB user quota) use 8 Dask and 16 I/O threads: v3 lexicographic ~1 h.
  Outputs (26 Oct 2026): `/e/fscratch/e-dev-2026d09-119/graphcast/data/anomalies/{zarr_v2,zarr_v3,zarr_v3_lexi}`.
- Mesh-size study: `clusters/jupiter/slurm/mesh_study.sh STUDY MESH[:EDGE_CHUNK_SIZE[:HH:MM:SS[:lowmem]]] ...`.
- Benchmarks: JUBE in `clusters/jupiter/benchmarks/jube` (`uv sync`, then
  `uv run jube run graphcast_training.xml --tag <sweep> [modifiers]`; add `dryrun` to check without submitting).
  `run_final.sh` / `run_memfit.sh` submit the standard experiments; `summarize.py RESULTS_DIR` writes the CSVs.
  The README there documents tags and findings; reports live in `results/<date>-<name>/REPORT.md` (gitignored)
  and are copied to `docs/jupiter-hackathon/results/` for the reveal.js presentation (`docs/jupiter-hackathon/index.html`).
- Tests are abseil `absltest` modules next to the code (`src/graphcast/*_test.py`); run one with
  `apptainer exec --nv graphcast.sif python src/graphcast/xarray_jax_test.py` (pytest is not a dependency;
  there is no CI).
- Lint/format: `ruff` with **2-space indentation** (`pyproject.toml`).

## Architecture

**Configuration.** Training is driven by a TOML file (`configs/training/full.toml`; `demo_5000.toml` is a
5000-step demo on 8 nodes) validated by pydantic models in `training_config.py` (`TrainingConfig` with sections
task, model, mesh, mask, dataset, grain_dataset, normalization, policy, optimizer, checkpoints, …). Paths in the
TOML are container paths (`/data/...`). Grain worker/buffer settings come from `GRAPHCAST_GRAIN_*` environment
variables (`grain_utils.GrainRuntimeConfig`), not the TOML; Grain iterator checkpoints depend on the worker count.

**Training loop** (`training.py`, helpers in `training_utils.py`): builds the mesh (`get_mesh` →
`mesh_graph.read_mesh_data`), the predictor, an optax optimizer from named transformations, Orbax checkpoints,
TensorBoard logging, and a data-parallel `fsdp_map` over the device mesh (`sharding.py`). The train step is
lowered and compiled once; its XLA memory analysis (`cli_utils.run_analysis_and_report`, "Memory usage" in the
log) predicts GPU OOM before training starts.

**Predictor stack** (`training_utils.get_predictor`), outermost first:
`Mask` (`mask.py`) → `InputsAndResiduals` (`normalization.py`) → `Bfloat16Cast` (`casting.py`) → `GraphCast`
(`model.py`). Training is one-step; `autoregressive.py` / `rollout.py` handle multi-step.

**GraphCast model** (`model.py`): grid ↔ mesh connectivity comes from `mesh_connectivity.py` (encoder edges by
radius query, decoder edges by containing triangle, both restricted by the ocean mask). With
`model.connected_grid_nodes` only grid points that are encoder senders or decoder receivers become grid nodes;
the decoder writes back only ocean (unmasked) points. Three `DeepTypedGraphNet`s (`deep_typed_graph_net.py`,
built from `typed_graph_net.py` blocks on `typed_graph.TypedGraph`) form encoder (grid2mesh), processor
(mesh_gnn, `gnn_msg_steps` steps) and decoder (mesh2grid). Feature construction is in `model_utils.py`;
geodesic edge features use `gis_utils.py`.

**Memory controls** (`[policy]`):
- Every block is `hk.remat`'d with `save_and_offload_only_these_names`; the names are `embedder`,
  `graph_embedder`, `message_passing`, `graph_output`.
- These names mark block outputs, so offloading them to pinned host memory takes effect only with
  `remat_loss = true`, which checkpoints the whole loss.
- `scan = true` runs the processor with `lax.scan` over parameters stacked from `hk.lift`'d per-step params. It
  shares the parameter tree with the unrolled version (checkpoints are interchangeable), costs a few percent of
  compute and saves a lot of GPU memory.
- Avoid `lax.switch` inside the scan: host offloading there hits an XLA bug ("Expected instruction to have
  shape equal to…").
- `edge_chunk_size > 0` updates large edge sets in remat'd chunks (`typed_graph_net._edge_update`). It saves GPU
  memory but costs compute.

**Data** (`dataloader.py`, `dataloader_utils.py`, `grain_utils.py`, `data_utils.py`, `xarray_jax.py`): zarr
(v2 fastest) or SquashFS sources read with xarray and fed through Grain multiprocessing workers; normalization
statistics come from climatology/anomaly_std/tendency_std datasets. At all 50 depth levels, host memory limits
Grain to about 2 workers.

**Meshes**: `geospatial_mesh.py` (seamsh, coastline shapefiles) generates `.msh` files from `configs/mesh/*.toml`;
generated meshes and geometry live in `data/geometry` (gitignored, bound as `/data/geometry`).

## Repository notes

- Do not commit `graphcast.sif`, `graphcast.def` (generated copy), `spack-env/`, `runs/`, `logs/`, `data/`,
  benchmark `results/` or the local `repro/` reproducers.
- Development branch: `develop@esiwace3_hackathon`; Slurm account for training jobs: `training2647`.
