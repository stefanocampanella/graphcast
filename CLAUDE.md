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
- Standardized anomaly dataset (preprocessing done once): `bash clusters/jupiter/slurm/download.slurm DATASET
  CLIMATOLOGY ANOMALY_STD OUTPUT --config configs/download/{zarr_v2,zarr_v3_lexi,zarr_v3_colexi}.toml` on a login node,
  or the same with `sbatch` on a compute node (no internet: local copies in `/e/fscratch/e-dev-2026d09-119/ogs-arco-ocean-50*`).
  It sources `clusters/jupiter/.env` and runs the portable `scripts/download.sh` → `graphcast download` (`downloader.py`),
  which applies the preprocessing of `ARCOTimeseriesSource`. Inputs: local paths or URLs
  (`s3://ogs-arco-ocean/{datasets,climatology,anomaly_std}/res=0p25`). All arrays are read with xarray-tensorstore
  (`mask_and_scale=False`: masks are int8; xarray still decodes `time` to datetime64, no cftime) and data written with
  TensorStore after `to_zarr(compute=False)` writes metadata and coordinates. Colexicographic order within shards is a
  `transpose` codec (time and depth swapped) before the sharding codec: TensorStore fills shards in lexicographic order
  of the (transposed) chunk grid and still reads single chunks; zarr-python reads whole shards of such arrays (warning
  "Combining a `sharding_indexed` codec disables partial reads"). Morton order (zarr-python only) was dropped: slow.
  S3 credentials come from the environment: metadata via fsspec/s3fs (`AWS_PROFILE=xarray-s3-writer`, with
  `aws_signing_helper` in `~/.local/bin`, or `FSSPEC_S3_ANON=true`), data via TensorStore, which cannot run a
  `credential_process`: with such a profile it logs "Failed to open file ... .aws/credentials" per request and then
  reads anonymously (the bucket is public); for signed reads export the role's keys
  (`eval "$(aws configure export-credentials --profile xarray-s3-writer --format env)"`).
  Full 10.5 TB (5.1 TB on disk), three booster jobs at once: v3 lexicographic 34 min, v2 and v3 morton (zarr-python
  writer) 2 h each; login node (20-core, 143 GiB user quota): 8 Dask and 16 I/O threads, v3 lexicographic ~1 h.
  Outputs: `/e/fscratch/e-dev-2026d09-119/graphcast/data/anomalies/{zarr_v2,zarr_v3,zarr_v3_lexi}` (26 Oct 2026;
  `zarr_v3` is morton, written by the previous version) and `zarr_v3_colexi` (27 Oct 2026).
- Preliminary I/O measurements (27 Oct 2026, login node, 20-core quota, short runs; scripts were throwaway):
  - S3 (`s3://ogs-arco-ocean`, eu-south-1, zarr v2 with one ~4 MB chunk per day and level), 13 GB of `thetao`:
    TensorStore's `s3` kvstore reaches ~4.7 GB/s with `s3_request_concurrency` 256–512 (3.1 GB/s at 128), using
    ~6 cores and ~2× the data in RSS; anonymous and signed reads are equally fast. zarr-python + s3fs stays at
    ~0.29 GB/s whatever the concurrency or Dask chunks.
  - xarray-tensorstore on the anomaly stores, one random sample of 3 days × all time-dependent variables
    (2.78 GB at 50 levels, 0.78 GB at 10): opening takes 1–2 s and 0.15 GB.

    | Store | ts, 4 threads | ts, 16 threads | ts, 64 threads | zarr-python, 8 threads | ts, 16 threads, 10 levels |
    |---|---|---|---|---|---|
    | v2 | 0.73 s | 0.22 s (12.7 GB/s) | 0.20 s | 1.25 s | 0.07 s |
    | v3 morton | 0.70 s | 0.23 s | 0.23 s | 1.74 s | 0.09 s |
    | v3 lexi | 0.79 s | 0.32 s | 0.22 s | 1.55 s | 0.10 s |

    Threads are the `data_copy_concurrency` and `file_io_concurrency` limits. The process RSS after a read is about
    the sample plus 0.4–1.2 GB with 4–16 threads, and plus 1.6–2.8 GB with 64 threads; the peak is at most 0.4 GB
    higher. zarr-python takes about the sample plus 0.2 GB (peak plus 0.4 GB). xarray-tensorstore is 5–8× faster
    than zarr-python, and v2 is slightly faster than v3.
  - Blosc on anomaly chunks (single core): lz4 clevel 1 with byte shuffle compresses 1.89× and decodes at
    1.46 GB/s; bitshuffle compresses less (1.71×) and decodes at 0.74 GB/s, so it is not worth benchmarking further.
    zstd clevel 1 with byte shuffle: 2.01×, decode 1.01 GB/s, encode 0.49 GB/s.
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
