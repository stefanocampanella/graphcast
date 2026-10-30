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
  (data paths, Grain and zarr settings) and calls `scripts/train.sh`, which binds DATASET (the standardized anomaly
  store), MASK, TENDENCY_STD (SquashFS or directories) and `GEOMETRY_DIR` under `/data` in the container.
  Use `--reservation=develbooster` for short tests on up to 4 nodes (its nodes jpbo-101-[01-08] are in maintenance
  until 29 Nov 2026, `maint_row_6.1`: until then use the normal queue). To test the code of a worktree, use an env file
  with `APPTAINERENV_PYTHONPATH` set to the worktree's `src` (the shared `.env` points at the main checkout) and
  submit from the worktree.
- Standardized anomaly stores (preprocessing done once): `bash clusters/jupiter/slurm/download.slurm DATASET
  CLIMATOLOGY ANOMALY_STD OUTPUT --config configs/download/{zarr_v2,zarr_v3_lexi,zarr_v3_colexi}.toml` on a login
  node (S3 or local inputs), or the same with `sbatch` on a booster node (no internet: local inputs only); the mask
  with `bash clusters/jupiter/slurm/download.slurm --mask DATASET OUTPUT --config configs/download/mask.toml`, the
  tendency statistics with `--tendency-std STANDARDIZED OUTPUT --config configs/download/tendency_std.toml`. See
  "Standardized anomaly stores" below.
- Mesh study: `clusters/jupiter/slurm/mesh_study.sh [--account A] STUDY SEEDS MESH:MSG_STEPS[:EDGE_CHUNK_SIZE[:HH:MM:SS]]
  ...` (demo_5000 on 8 nodes per case and seed, validation loss on held-out windows; jobs use the checkout's `src`).
- Benchmarks: JUBE in `clusters/jupiter/benchmarks/jube` (`uv sync`, then
  `uv run jube run graphcast_training.xml --tag <sweep> [modifiers]`; add `dryrun` to check without submitting).
  Everything runs at all 50 levels (`production_run.toml`, coarse mesh × 8 steps unless a tag says otherwise); cases
  are `format:workers:threads:io_threads:nodes:cpus` (baseline `zarr_v2:4:2:16:1:72`), sweeps `loader` (6 layouts × 0x4 to 8x2), `io_threads`, `cpus`, `scaling`, `memfit`; jobs
  use the `src` of the checkout JUBE runs from. `run_final.sh` (loader, io_threads, cpus, scaling, `long`) /
  `run_memfit.sh` submit the standard experiments; `summarize.py RESULTS_DIR` writes the CSVs. Earlier results
  (10 levels, previous dataloader) are not comparable. Results of 29 Oct 2026 (`results/2026-10-29-alllevels`,
  `results/2026-10-29-memfit-alllevels`; REPORT.md and CSVs copied to `docs/jupiter-hackathon/results/
  {benchmark,memfit}-2026-10-29`, shareable report https://claude.ai/artifact/QaPPanmobJWisuDa8MN8ZR):
  - loader (coarse × 8, 1 node): all six layouts and 0x4 to 4x2 run at 0.48–0.55 s per step (compute 0.43–0.49 s, data
    0.013–0.05 s; the old dataloader took 2.4 s at 50 levels with 2x2); SquashFS 3–8 % slower; 8x2 filled the node's
    858 GB and hung at step 4 (12/12 jobs). Node memory peaks: 0 workers 580–630 GB, 1: 605–680, 2: 700–750, 4:
    840–857 GB; a sample is ~5.7 GB, the training processes ~580 GB per node. Threads only (0x4) waits least for data but
    computes 3–5 % slower.
  - io_threads 8/16/32: no difference; 8 CPUs per task instead of 72: 24 % slower; 1/2/4 nodes: 0.515/0.502/0.506 s per
    step (linear scaling); step 0 ends 83–105 s after launch.
  - memfit (50 levels, 9 targets): the defaults fit up to 50 km × 16 (1.29 s per step, 74 GiB) and 40 km × 8 (1.08 s);
    offloading (0 workers) adds 40 km × 14 (1.99 s); edge chunks add 35 km × 12 (2.87 s); 30 km and the fine mesh fit
    with none. Same limits as on 25 Oct. 40,000 steps take 5.6 h (coarse × 8) to 14.3 h (50 km × 16).
  The README there documents tags and findings; reports live in `results/<date>-<name>/REPORT.md` (gitignored)
  and are copied to `docs/jupiter-hackathon/results/` for the reveal.js presentation (`docs/jupiter-hackathon/index.html`).
- Tests: new code is tested with **pytest** in `tests/` (`test_<module>.py`, shared fixtures in `tests/conftest.py`;
  pytest is in the `dev` dependency group, and `[tool.pytest.ini_options]` collects `tests/` and imports `src/`, not
  the package copy installed in the image). The image installs the `dev` group (since 27 Oct 2026; the recipe
  upgrades pip, as the base image's pip 24 cannot install dependency groups): run
  `apptainer exec graphcast.sif python -m pytest` from the repository root (`tests/` takes ~40 s, CPU only, also on
  compute nodes). With an older image, layer pytest on its venv with uv (login nodes only, needs internet):
  `apptainer exec graphcast.sif ~/.local/bin/uv run --no-project --python /opt/software/venv/bin/python --with
  "pytest >= 9" pytest`. Tests use synthetic data (small Zarr stores in `tmp_path`)
  or, if needed, public ARCO-OCEAN data read anonymously: no AWS profile, credentials or network may be assumed.
  The abseil `absltest` modules inherited from GraphCast (`src/graphcast/*_test.py`) are not collected and are
  probably broken; they will be ported to pytest later. There is no CI.
- Lint/format: `ruff` with **2-space indentation** and 120-column lines (`pyproject.toml`); run `ruff format` and
  `ruff check --fix` only on new or refactored modules (the rest is unformatted, to keep diffs reviewable). Type
  checks: `apptainer exec graphcast.sif ~/.local/bin/uvx ty check --python /opt/software/venv --extra-search-path
  src FILES`; new code should pass (except for `pytest`, which is not in the venv). Pre-existing diagnostics in
  `dataloader.py`: unresolved `jax` (installed outside the venv) and two Grain `__getitem__` overrides.
- Image rebuild (needed after dependency changes; `src/` comes from `APPTAINERENV_PYTHONPATH`):
  `spack env create --dir spack-env spack.yaml`, `spack -e spack-env containerize > graphcast.def`, then
  `env APPTAINER_CACHEDIR=$FSCRATCH/.apptainer apptainer build graphcast.sif graphcast.def`. Build to a separate file
  first and ask before replacing the shared `graphcast.sif`: running jobs use it. Run `containerize` from the checkout
  whose `pyproject.toml` and `src/` the image should contain (a worktree works with
  `spack -e /e/project1/e-dev-2026d09-119/graphcast/spack-env`); the build takes ~6 min on a login node. Previous
  images are kept as `graphcast-YYYY-MM-DD.sif`. Past commands are in `~/.bash_eternal_history`.

## Standardized anomaly stores

`graphcast download` (`downloader.py`, settings in `download_config.py`) standardizes the raw dataset once
(climatology alignment, filling, clipping, z-scores, min-max static fields, float32) and writes one local store;
training reads it with `dataloader.StandardizedAnomalySource`, which adds the batch dimension, the `datetime`
coordinate and relative time (the previous `ARCOTimeseriesSource` did all of this per sample from the raw dataset,
climatology and anomaly_std; it was removed on 28 Oct 2026). `graphcast download-mask` writes the training mask: the
mask variable (`name`, default `glorys_mask`) at one depth position (`level`), as a single boolean `(lat, lon)`
DataArray (`configs/download/mask.toml`); training loads it as is (`dataloader_utils.get_grid`: lat, lon, mask).

- `tendency_std` (residual normalization in training) is the zero-mean standard deviation of the one-step differences
  of the standardized anomalies over the whole series, per variable and point (no time dimension). The copy in use,
  `/e/scratch/training2647/graphcast_data/zarr_v2/tendency_std` (also `s3://ogs-arco-ocean/tendency_std/res=0p25`,
  to be deleted there), was computed with arco-make's `diff_std` step, which is expensive: keep it. `graphcast
  compute-tendency-std STANDARDIZED OUTPUT --config configs/download/tendency_std.toml` (`downloader.tendency_std`,
  `TendencyStdConfig`; `download.sh`/`download.slurm --tendency-std`) recomputes it from a standardized store,
  in float64, with Dask chunks of 256 days of one depth level. A std of 0 is written as NaN (`zero_as_missing`);
  training fills NaN with 1 and clips at 1e-16. Full recomputation (28 Oct 2026, booster node, 32 threads): 34.6 min,
  `$FSCRATCH/graphcast/data/anomalies/tendency_std_recomputed`. Comparison with the copy, all 27 variables and levels
  (scripts and CSV in `$PROJECT/claude-work/5ec59923-bcb3-483e-a402-cd793d0e8df5/tendency_report*`); every difference
  has one of four causes:
  1. Feb 29: arco-make converted to a 365-day calendar, dropping it; the store keeps it. Median relative difference
     3–6e-4 everywhere; without Feb 29 the recomputation matches the copy within 1e-7 (float32), including the few
     points of clean series more than 5% apart (an extreme day around Feb 29 carries 11–18% of the sum of squares).
  2. Days with anomaly_std exactly 0 (value = climatology = 0): arco-make's z is 0/0 = NaN there and skipped, the
     store's is 0 and counted, so the new values are smaller (0.65–0.9). Sea-ice variables (`siconc`, `sithick`,
     `usi`, `vsi`: 37–42% of their finite points > 5% apart), the five wave variables (11%: ARCO-OCEAN's `masked_fill`
     wrote 0 where WAVERYS has no waves, under sea ice, inside the static `waverys_mask`), `dis24` (12%), `so` at
     levels 45–47 (130 points), `so_bottom` (31), `lsm` (332).
  3. Constant series with a defined z-score (`z` and `lsm`, ERA5 fields constant in time; 9–10 deep `so` points): the
     climatology and anomaly_std differ from the value by float32 rounding, so z is a constant ratio of rounding
     errors (about ±1); the copy has exactly 0 (590,566 `z`, 143,162 `lsm` points), the recomputation NaN. The
     standardized `z` and `lsm` themselves are such rounding noise (0, ±1, ±1.41), not geopotential or land fraction.
  4. Statistics undefined on all days at points with data on some days: copy NaN, recomputation finite (the store
     filled climatology with 0, anomaly_std with 1, so z is the raw value). Wave variables outside the static
     `waverys_mask` (version 202311 bathymetry; the dynamic fields are 202411): 319 points with data only on
     1993-01-01 (48–80°N; 268 NaN mismatches for `swh`, 191–319 for the other wave variables), plus 82 coastal points
     with data only from 2013-01-01 (50 every day, 32 intermittently; after 2013 the daily NaN count of the wave
     fields varies, before it equals the static land count exactly), which are finite in both but ~0.6 of the copy.
  The four target variables (`thetao`, `so`, `uo`, `vo`), `zos`, ERA5 forcings other than `z`/`lsm`, `tp`,
  `thetao_bottom` only have cause 1 (and the few `so` points of causes 2–3).
- Conventions for the standardized store and tendency_std (decided and implemented on 28 Oct 2026, commit 5b7026c:
  `DownloadConfig.masks`, `static_variables`, `equal_tolerance`, `reference_period`, `TendencyStdConfig.zero_inflated`,
  `dataloader_utils.get_loss_masks`, per-variable `Mask` and losses). The stores, their SquashFS images and
  tendency_std in `$FSCRATCH/graphcast/data/anomalies` were recomputed with them on 28 Oct 2026 (see Outputs); the
  anomaly_std floors take ~2 min to compute:
  1. Feb 29 is kept, standardized with Feb 28's climatology and anomaly_std (they vary smoothly), and its differences
     count in tendency_std (as now; arco-make's copy dropped them).
  2. `z` and `lsm` are static: excluded from climatology/anomaly_std, min-max normalized to 0–1, no tendency_std.
     They should be constant; ARCO-OCEAN has glitches: `z` changes on 2022-10-01 at the two polar rows (lat ±90, all
     1440 longitudes, ~6×), `lsm` differs on 1998-05-04 and 2014-04-05 at 532 points (18–24°S). Take a glitch-free
     day.
  3. Wave variables under sea ice are 0 (already so inside `waverys_mask`, via ARCO-OCEAN's `masked_fill`).
  4. Values equal to the climatology within a relative tolerance, `|x − c| ≤ 4·eps·max(|x|, |c|)` (float32 eps), have
     a z-score of 0 whatever the anomaly_std (constant series such as deep salinity, zero river discharge; without it
     z is a ratio of rounding errors, e.g. ±1).
  5. Anomaly_std floor, per variable (and level): its smallest positive in-sample value inside the mask, so in-sample
     z-scores (and the heavy tails of `dis24`, `so`, `tp`) are unchanged; for data outside 1993–2023, |z| is also
     capped at the variable's in-sample maximum. Smallest positive values: `siconc` 4e-12, `dis24` 1.5e-11, `sithick`
     8e-11, `tp` 1.5e-8, `so_bottom` 2.7e-6, `so` (level 49) 1.4e-5, `swh` 2e-5; ERA5 forcings ≥ 0.04; `uo`/`vo` at
     level 49 are 0 at all 16,664 sea points.
  6. A point whose series is constant (after 4) has tendency_std 0, and is excluded from the loss for that variable
     (no clipping of tendency_std); predictions there keep the input value (also outside the masks, where it is 0).
  7. Sea-ice variables (`siconc`, `sithick`, `usi`, `vsi`): tendency_std is the std of the non-zero differences only
     (their differences are zero-inflated), with no minimum number of them.
  8. Data outside their source's static mask are 0: GLORYS `glorys_mask` (3D for `thetao`, `so`, `uo`, `vo`; level 0
     for `zos`, sea ice, `thetao_bottom`, `so_bottom`), WAVERYS `waverys_mask` (`swh`, `mwp`, `mwd`, `usd`, `vsd`),
     GLOFAS `glofas_mask` (`dis24`); ERA5 has no mask. Wave values that are 0 inside the GLORYS mask (different grids)
     are accepted. GLORYS values outside the 3D mask (6,315 points at level 10, 13,206 at level 40, close to the
     neighbouring sea) are not relevant and become 0. Raw values outside the masks today: sea-ice variables are 0;
     `glorys_deptho` 0 or NaN (and 3 points ≤ 1.3e-7); `waverys_deptho` 0; `uparea` NaN (min-max over the mask maps
     its smallest river point to 0, like land).
  9. Loss masks per target variable and level, replacing the single surface mask: today `Mask` and
     `losses._mean_preserving_batch` broadcast the 2D surface `glorys_mask` over all levels and variables, so land
     below the surface of ocean columns is in the loss (targets 0, tendency_std NaN → 1) and is predicted, and the mean
     counts masked points as zeros. Plan: mask = grid mask ∧ `tendency_std > 0` (zero outside masks and at constant
     points by 6 and 8); per-variable masks in `losses`, averaging over valid points per level before depth weights;
     residuals divided only where tendency_std > 0; `Mask` per variable, keeping the input at excluded points. Targets
     are the ocean variables.
  Normalization statistics over 1993–2023 include the held-out period after `split_date`: accepted for now.
- **Residual connection was disabled from 27 Mar to 28 Oct 2026** (commit 21edee5): `normalization.InputsAndResiduals`
  (since split into `InputNormalization` and `OutputNormalization`) took its residual branch only
  `if self._residual_locations is not None`, always `None`, so training predicted the standardized targets directly and `tendency_std` was loaded but unused; models trained in that period are direct
  predictors. Fixed on 28 Oct 2026: the branch tests `_residual_scales` (residuals when `tendency_std` is given, direct
  prediction otherwise, as the docstring says; `tests/test_normalization.py`). Experiment (28 Oct 2026, `exp_src` in
  claude-work, `GRAPHCAST_EXP_RESIDUALS`): direct vs residual without output normalization vs residual /
  tendency_std, 2000 steps each, with the state error (z-score units) of predictions, persistence and climatology
  logged per step. Results (state error / persistence, steps 1900–2000): direct 0.56, residual unscaled 0.51,
  residual / tendency_std 0.53; all learn. Per variable, direct prediction does not beat persistence for `thetao`
  (1.12) and `so` (1.18), slow variables whose daily changes are near the bfloat16 resolution of the full state;
  residuals do (unscaled 0.88 / 0.96, normalized 0.83 / 0.92). Unscaled residuals do best on `uo`/`vo` (0.43),
  normalized ones on `zos`, `thetao` and `so`: output normalization moves capacity to slow variables. One seed,
  2000 steps, 10 levels.
- Vertical coordinate: z-levels with per-variable, per-level loss masks for now. Sigma coordinates (all levels valid
  sea, depth varying per column) would avoid loss masking and are to be reconsidered later. They would need: a sigma
  level definition (number, stretching, whether the top follows the free surface `zos`); GLORYS level bounds (only
  centres are in the data) for conservative remapping and partial bottom cells; a bottom condition for `uo`/`vo`
  (`thetao_bottom`/`so_bottom` exist, no velocity bottom values); a rule for shallow columns with 1–2 z-levels;
  standardizing before (interpolate z-scores, cheap) or after (recompute climatology/anomaly_std on sigma levels)
  remapping; the inverse remapping for evaluation on z-levels. In the model: the depth of each level becomes a 2D
  (3D with `zos`) feature next to the bathymetry (`glorys_deptho`), and loss level weights come from layer
  thicknesses H·Δσ instead of the 1D `depth` coordinate.
- Training data path: `dataloader_utils.get_dataset` opens the store lazily with xarray-tensorstore and selects
  `task.levels` (`dataloader.select_levels`); `StandardizedAnomalySource` stores only the path, levels, `split_date` and
  `dataset.io_threads` and reopens the store in each process (`__getstate__` drops it), so Grain workers never pickle
  TensorStore objects. A window is read with `xarray_tensorstore.read(...).compute()`. `tendency_std` and the mask
  are small and stay on `xarray.open_dataset`. Grain workers return shared-memory arrays (`xarray_jax.unwrap` to get
  the numpy array; xarray operations reject them), and tests that use Grain must call `flags.FLAGS.mark_as_parsed()`
  (absl), as the CLI does.
- Equivalence (28 Oct 2026): training samples (inputs, targets, forcings after extraction) of the new pipeline on
  `zarr_v2` are bit-identical to those of the old pipeline on the raw local copies (3 samples, 112 arrays); the mask
  equals the old `get_surface_mask` result (693,039 ocean points of 721 × 1440). Samples took 0.08–0.16 s on a login
  node with 16 I/O threads.

- Entry points: `scripts/download.sh` is portable (inherited environment, `APPTAINER_IMAGE` defaulting to
  `PROJECT_DIR/graphcast.sif`); `clusters/jupiter/slurm/download.slurm` sources `clusters/jupiter/.env` (or
  `--env-file`) and runs it. Paths are CLI arguments; the TOML holds settings only (Zarr format, v3 shard sizes,
  order of chunks within shards, Blosc, `num_workers` Dask threads, `io_threads` TensorStore concurrency).
- Inputs: local copies in `/e/fscratch/e-dev-2026d09-119/ogs-arco-ocean-50{,_climatology,_anomaly_std}`, or
  `s3://ogs-arco-ocean/{datasets,climatology,anomaly_std}/res=0p25`.
- Reading: all arrays with xarray-tensorstore (`open_zarr(..., context=, mask_and_scale=False, consolidated=True)`):
  masks stay int8 (cast, not normalized); xarray still decodes `time` to datetime64 (no cftime). Metadata goes through
  xarray/zarr-python/fsspec (s3fs, botocore), data through TensorStore kvstores.
- Writing: `to_zarr(compute=False)` writes metadata and coordinates, then `dask.array.store` into TensorStore arrays
  (`zarr`/`zarr3` drivers, `lock=False`). Dask chunks are shards (8 timesteps with depth, 256 without); chunks are one
  timestep and level.
- Chunk order within v3 shards: colexicographic (by depth, then time) is a `transpose` codec (time and depth swapped)
  before an explicit `ShardingCodec`. TensorStore fills shards in lexicographic order of the transposed grid and still
  reads single chunks, while zarr-python reads whole shards of such arrays ("Combining a `sharding_indexed` codec
  disables partial reads"). Morton order (zarr-python writer only) was dropped: 2 h to write, no faster to read.
- Leap years: `downloader.climatology_index` maps Feb 29 to Feb 28's climatology and later days back by one (the old
  `_handle_leap_years` gave Feb 26 and 27 the same day).
- S3 credentials come from the environment, never from configs (profiles are user-specific): on JUPITER
  `AWS_PROFILE=xarray-s3-writer` (a `credential_process` with `aws_signing_helper` in `~/.local/bin`, hence
  `APPTAINERENV_PREPEND_PATH`), or `FSSPEC_S3_ANON=true`. The default profile is an expired `aws login` session
  (`LoginRefreshRequired`). TensorStore cannot run a `credential_process` (it logs "Failed to open file ...
  .aws/credentials" per request and reads anonymously), so `downloader.get_aws_credentials` resolves the keys with
  botocore, exports them to `os.environ` and sets `aws_credentials: {type: environment}` (or `anonymous`), for S3
  inputs only. Keys are not refreshed (12 h for this role).
- TensorStore auth: `aws_credentials` is a context resource (`ts.Context({"aws_credentials": {"type": ...}})`, passed
  to `xarray_tensorstore.open_zarr(..., context=...)`); types `default` (environment → `~/.aws/credentials` profile →
  web identity → IMDS, then anonymous), `anonymous`, `environment`, `profile`, `imds`, `ecs`; no key fields in specs.
  Concurrency is set by the `data_copy_concurrency`, `file_io_concurrency` and `s3_request_concurrency` resources.
- Dependencies added: `xarray-tensorstore`, `fsspec`, `s3fs >= 2026.9.0` (without the bound, pip falls back to s3fs
  0.4), `botocore[crt]` (for `aws login` sessions).
- Verification: `tests/` compares the downloader and the dataloader with a direct reference computation on synthetic
  data (`tests/conftest.py`); throwaway scripts compared real subsets (20 days around 29 Feb 2020, local and S3 inputs) and windows of the full stores (agreement
  within 1e-6).
- Full 10.5 TB (5.1 TB on disk) on booster nodes: v3 lexicographic 34 min and v2 2 h (three jobs at once), v3
  colexicographic 32 min alone; login node (use 8 Dask and 16 I/O threads): v3 lexicographic ~1 h. Outputs (28 Oct
  2026, with the conventions above; three booster jobs at once: `zarr_v2` 2 h 4 min, `zarr_v3_lexi` 54 min,
  `zarr_v3_colexi` 60 min; identical to an in-memory standardization over windows with Feb 29):
  `/e/fscratch/e-dev-2026d09-119/graphcast/data/anomalies/{zarr_v2,zarr_v3_lexi,zarr_v3_colexi}`, `tendency_std`
  (from `zarr_v2`, 31 min on a booster node; 25 variables, finite, 0 outside the masks) and the mask `glorys_mask`
  (from the local raw copy with `configs/download/mask.toml`, 9 s). The earlier stores (including the morton
  `zarr_v3`) and the arco-make tendency_std copy are incompatible with the conventions and were deleted (the copy
  remains in `/e/scratch/training2647/graphcast_data/zarr_v2/tendency_std` and on S3).
- Training check (28 Oct 2026, job 2568724, 60 steps, 1 node): residuals / tendency_std with per-variable loss masks
  train (loss 0.73 → 0.49, every target variable −30–40%); data wait 0.021 s, compute 0.385 s per step; XLA memory
  49.2 GiB per GPU (47.1 GiB with the single mask: per-variable masks are larger arrays).
- Training check at 50 levels with the `production_run.toml` (then `full.toml`) variables (29 Oct 2026, job 2576513, 60 steps, 1 node, 4 Grain
  workers x 2 threads): loss 0.259 → 0.146; volume variables report ~0.03 (level-weighted), surface ones 1.0–2.1, and
  with the 0.02 weights both groups contribute about equally (0.14 / 0.12). Data 0.041 s, compute 0.397 s per step
  (10 levels: 0.385 s); XLA memory 54.2 GiB per GPU; MaxRSS 18 GB per task.
- Mesh × message passing study (29–30 Oct 2026, `runs/mesh_msg/REPORT.md`, copy in
  `docs/jupiter-hackathon/results/mesh_msg-2026-10-29`):
  - **Setup:** demo_5000 (8 nodes, 5000 steps ≈ 17 epochs), 5 meshes × 8/12/16 steps × 2 seeds, 384 node-h.
  - **Validation loss falls along both axes:** coarse × 8 0.0907, 50 km × 16 0.0818 (−9.8 %, 1.40 s per step),
    40 km × 16 0.0805 (−11.2 %, 3.10 s with edge chunks). Neither axis has saturated.
  - **Noise and overfitting:** seeds differ by under 0.6 %. No overfitting: the minimum is at the last step in all
    runs, and the validation–training gap is 0.010–0.012 in every case.
  - **zos needs reach** (steps × spacing): +9.6 % at 40 km × 8, +0.6 % at 40 km × 16. The other fields gain with
    resolution: sea-ice thickness −28 %, currents −14 % at 40 km × 16.
  - **Value:** at equal cost more steps beat a finer mesh (50 km × 16 beats 40 km × 12).
  - **Next:** 20–24 steps on 50 km and 35 km × 12.
- SquashFS images for the dataloader benchmarks, on the same filesystem as the stores (fair comparison):
  `$FSCRATCH/graphcast/data/anomalies/squashfs/{zarr_v2,zarr_v3_lexi,zarr_v3_colexi}.sqsh`, written on a login node
  by `mksquashfs.sh` there, with the settings of the earlier images in `/e/project1/e-dev-2026d09-119/squashfs` (from
  `~/.bash_eternal_history`): `mksquashfs SRC DST.sqsh -noappend -no-xattrs -noI -noD -noF -b 1M -comp lz4 -progress`
  (uncompressed, as chunks are already Blosc-compressed; 1 MB blocks, the first tries used 4 KB), plus `-mem 32G` on
  login nodes: the default working memory (25% of RAM) equals the 143 GiB per-user limit, and a first run writing
  fscratch → fscratch at ~4 GB/s was killed at 69%. With `-mem 32G` (28 Oct 2026, `jpbl-s02-04`, about 18 GB used, no
  OOM kill): `zarr_v3_lexi` 13 min, `zarr_v3_colexi` 14 min (5.55 TB each, 25,891 entries), `zarr_v2` 159 min (5.52 TB,
  2,525,068 entries: the file count, not the size, sets the time); entry counts equal the sources'. `scripts/train.sh`
  binds `.sqsh` files under `/data` like directories (`image-src=/`); windows read through the bound images are
  identical to those of the directory stores.

### I/O measurements for the dataloader refactoring

Preliminary (27 Oct 2026, login node with a 20-core quota, short runs, throwaway scripts):
- S3 (`s3://ogs-arco-ocean`, eu-south-1, zarr v2 with one ~4 MB chunk per day and level), 13 GB of `thetao`:
  TensorStore's `s3` kvstore reaches ~4.7 GB/s with `s3_request_concurrency` 256–512 (3.1 GB/s at 128), using ~6 cores
  and ~2× the data in RSS; signed and anonymous reads are equally fast. zarr-python + s3fs stays at ~0.29 GB/s whatever
  the concurrency or Dask chunks.
- xarray-tensorstore on the anomaly stores, one random sample of 3 days × all time-dependent variables (2.78 GB at 50
  levels, 0.78 GB at 10); opening takes 1–2 s and 0.15 GB:

  | Store | ts, 4 threads | ts, 16 threads | ts, 64 threads | zarr-python, 8 threads | ts, 16 threads, 10 levels |
  |---|---|---|---|---|---|
  | v2 | 0.73 s | 0.22 s (12.7 GB/s) | 0.20 s | 1.25 s | 0.07 s |
  | v3 morton | 0.70 s | 0.23 s | 0.23 s | 1.74 s | 0.09 s |
  | v3 lexi | 0.79 s | 0.32 s | 0.22 s | 1.55 s | 0.10 s |

  Threads are the `data_copy_concurrency` and `file_io_concurrency` limits. RSS after a read is about the sample plus
  0.4–1.2 GB with 4–16 threads, plus 1.6–2.8 GB with 64 threads (peak at most 0.4 GB higher); zarr-python takes about
  the sample plus 0.2 GB (peak plus 0.4 GB). xarray-tensorstore is 5–8× faster than zarr-python; v2 is slightly faster
  than v3.
- v3 lexicographic vs colexicographic (same sample, 16 samples × 2 runs): 50 levels 0.26 vs 0.19 s (16 threads) and
  0.21 vs 0.16 s (64 threads); 10 levels 0.06–0.07 vs 0.06 s. Colexicographic is not slower (unexpectedly faster at 50
  levels, as a 3-day sample is one contiguous range per shard in lexicographic order; the store was written hours
  later, so the page cache may differ): check again in the dataloader benchmarks.
- Blosc on anomaly chunks (single core): lz4 clevel 1 with byte shuffle compresses 1.89× and decodes at 1.46 GB/s;
  bitshuffle 1.71× and 0.74 GB/s (not worth pursuing); zstd clevel 1 with byte shuffle 2.01×, decode 1.01 GB/s,
  encode 0.49 GB/s.

## Architecture

**Configuration.** Training is driven by a TOML file (`configs/training/production_run.toml`, the best configuration:
50 km mesh × 16 message passing steps, 40,000 steps, validation; `demo_5000.toml` is a
5000-step demo on 8 nodes) validated by pydantic models in `training_config.py` (`TrainingConfig` with sections
task, model, mesh, mask, dataset, grain_dataset, normalization, policy, optimizer, checkpoints, …). Paths in the
TOML are container paths (`/data/...`). Grain worker/buffer settings come from `GRAPHCAST_GRAIN_*` environment
variables (`grain_utils.GrainRuntimeConfig`), not the TOML; Grain iterator checkpoints depend on the worker count.
Variables (29 Oct 2026): targets `zos`, `thetao`, `so`, `uo`, `vo` and sea ice (`siconc`, `sithick`, `usi`, `vsi`),
which are also inputs; static inputs `glorys_deptho`, `glorys_mask`, `glofas_mask`, `uparea`, `z`, `lsm` (no time
dimension, as `land_sea_mask` in GraphCast); forcings the 8 ERA5 variables and `dis24`; year progress in both. Waves
(for a later study), `thetao_bottom` and `so_bottom` are unused. Level weights sum to 1 (upstream GraphCast: mean 1),
so a volume variable's loss is about 1/n_levels of a surface variable's: surface variables have
`per_variable_weights` 0.02 = 1/50 to count as one volume variable each (change them with the number of levels).
`model.level_weights_coord` (default `depth`) and `model.level_weights_decreasing` (default true) set the level
weights: linear in that coordinate, summing to 1.

**Training loop** (`training.py`, helpers in `training_utils.py`): builds the mesh (`get_mesh` →
`mesh_graph.read_mesh_data`), the predictor, an optax optimizer from named transformations, Orbax checkpoints,
TensorBoard logging, and a data-parallel `fsdp_map` over the device mesh (`sharding.py`). The train step is
lowered and compiled once; its XLA memory analysis (`cli_utils.run_analysis_and_report`, "Memory usage" in the
log) predicts GPU OOM before training starts.

**Predictor stack** (`training_utils.get_predictor`), outermost first:
`Mask` (`mask.py`, per-variable loss masks) → `OutputNormalization` (`normalization.py`, residuals / tendency_std) →
[`InputNormalization`, only with input statistics; the standardized store needs none] → `Bfloat16Cast` (`casting.py`)
→ `GraphCast` (`model.py`). Training is one-step; `autoregressive.py` / `rollout.py` handle multi-step.

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

**Data** (`dataloader.py`, `dataloader_utils.py`, `grain_utils.py`, `data_utils.py`, `xarray_jax.py`): windows of
the standardized anomaly store (zarr or SquashFS, read with xarray-tensorstore) fed through Grain multiprocessing
workers; residuals are normalized with `tendency_std`, and the boolean surface mask comes from `graphcast
download-mask`. With the previous dataloader (zarr-python, standardizing per sample), host memory limited Grain to
about 2 workers at all 50 depth levels. With the new one (29 Oct 2026) every setting from 0 x 4 to 4 x 2 keeps the
GPUs busy at 50 levels and 8 x 2 runs out of host memory; the `.env` default is 1 worker x 2 threads.

**Meshes**: `geospatial_mesh.py` (seamsh, coastline shapefiles) generates `.msh` files from `configs/mesh/*.toml`;
generated meshes and geometry live in `data/geometry` (gitignored, bound as `/data/geometry`).
- **Settings:** validated by `mesh_config.MeshGenerationConfig` (pydantic, unknown keys rejected). It holds the gmsh
  options, smoothness, domain shapefile, boundary coarsening, and a list of size fields discriminated by `name`
  (`uniform`, `shore_proximity`, `bathymetry`, `bathymetry_hessian`), each type at most once. The target size is the
  minimum of the fields. Invalid settings and an existing output fail before meshing.
- **Tests:** `tests/test_mesh_config.py` validates the shipped configs (and that uniform ones name their size) and
  meshes a synthetic coastline (a ring at 70°S, 1000 km, 0.5 s).
- **Reproducibility:** regenerating `uniform_120km` with the current code gives identical vertices and triangles.
- **Fixed on 30 Oct 2026:**
  - `write_ocean_mesh` wrote the whole mesh with every view (`PostProcessing.SaveMesh = 1`); now only with the first.
    Files written before then hold the mesh once per view (twice for the uniform meshes): they still read correctly.
  - `read_mesh_data` swapped description and license, so models saved before then carry them swapped.
  - `read_mesh` reused the views of meshes read before (`gmsh.open` keeps them), so reading a second mesh in the same
    process failed with a `KeyError`; it calls `gmsh.clear()` first.
- **Naming:** uniform meshes are named by their target size in km (`uniform_120km` was `uniform_coarse` until 30 Oct
  2026). `data/geometry/uniform_coarse.msh` is a symlink to `uniform_120km.msh` for the main checkout; remove it
  after the PR is merged. Each `.msh` stores the size field it was made with (views `MeshSize` and one per field) and
  the config's description. Measured median edge lengths match the targets: 117.8 km for 120 km, 34.9 for 35, 30.0
  for 30.
- **The 25 km mesh** (`uniform_25km`, formerly `uniform_fine`, 704,078 nodes, 31 Jan 2026):
  - its stored size field is 25 km at every node, and its median edge is 25.0 km;
  - it was made from a config that was never committed: `uniform_fine.toml` asked for 30 km then, and for 35 km later,
    identical to `uniform_35km.toml` (its `domain.sampling` is ignored by the code). It was replaced on 30 Oct 2026 by
    `uniform_25km.toml`; `data/geometry/uniform_fine.msh` is a symlink to `uniform_25km.msh` until the merge;
  - node counts scale as 1/size², so 363,900 × (35/25)² ≈ 713,000;
  - its description (150 km), and that of the 120 km mesh, came from an outdated template; both `.msh` files keep it.
- **Descriptions** of all `configs/mesh/*.toml` match their fields (checked on 30 Oct 2026). The `lambrechts*`
  configs take the smaller of a shore-proximity and a bathymetry field; they said "uniform 300 km" before.
  `domain.sampling` in them is ignored (the shore-proximity field has its own).

## Repository notes

- Do not commit `graphcast.sif`, `graphcast.def` (generated copy), `spack-env/`, `runs/`, `logs/`, `data/`,
  benchmark `results/` or the local `repro/` reproducers.
- Development branch: `develop@esiwace3_hackathon`.
- Slurm accounts (JUPITER booster): `training2647` for small runs (tests, short benchmarks) until it ends on
  31 Dec 2026 or its budget is exhausted; `e-dev-2026d09-119` for large runs (and the downloader), until 28 Feb 2027.
  Both have 1,008,000 core-h on `JUPITER_BOOSTER`; a booster node costs 288 core-h per hour whatever is used (whole
  nodes are charged), i.e. 3500 node-h per project. Check the remaining budget before launching large experiments:
  `jutil project cpuquota -p PROJECT -Z` (`coreh-avail`, `acct-period-end`; updated about daily, so add recent
  `sacct -A PROJECT -X -o JobID,ElapsedRaw,NNodes` usage). On 27 Oct 2026: `training2647` 621k core-h left (~2160
  node-h), `e-dev-2026d09-119` 1.00M (~3490 node-h).
- Budget policy (JSC "Allocation Procedures"): both projects use `acct-mode=fixed`, a one-time quota for the whole
  period with no monthly split, so it can be spent at any time before the period ends; leftovers are lost then. JSC's
  monthly mode (regular projects, not these) spreads the quota evenly over months: available = current month's
  quota + the previous month's leftover; one month may be borrowed from the next (not in the last month) at normal
  priority, beyond that jobs get the lowest priority for the rest of the month. When a quota is exhausted (or
  overdrawn by more than a month), jobs only run when the machine has free capacity, limited to 6 h. The page does
  not say explicitly that the monthly rules are off in fixed mode: that is inferred from `acct-mode=fixed` and from
  `jutil` reporting all usage under the current month.
- Queue: 1-node booster jobs submitted with a 6–12 h limit on 27 Oct 2026 waited ~16 h (reason `Priority`); for
  I/O-only work (e.g. `mksquashfs`), a login node is usually faster to get.
- Temporary files (throwaway scripts, benchmark logs, PR bodies, commit messages): keep them in
  `$PROJECT/claude-work/$CLAUDE_CODE_SESSION_ID/` (`$PROJECT` is `/e/project1/e-dev-2026d09-119`), not in `/tmp` or
  the session scratchpad. `/tmp` is local to each login node, and sessions resumed on another node lose it. The
  session ID stays the same when a session is resumed or compacted, and does not depend on branch or task names; a
  new session gets a new directory, so the logbook records each session's directory. Large data goes to `$FSCRATCH`
  instead; delete a session's directory once its PR is merged.
- Logbook: `docs/logbook.md` records, with date and time (CET), the maintainer's requests, code changes,
  experiments and insights; append entries as work proceeds (CLAUDE.md keeps what stays valid).
- Workflow: one feature branch and PR per task into `develop@esiwace3_hackathon`; the maintainer reviews on GitHub
  or asks for changes directly. Reply to each review comment once addressed. Keep PR descriptions short (purpose,
  usage, design, verification), without detailed benchmark numbers; keep those here. Commits are authored by Claude
  (`git commit --author="Claude <noreply@anthropic.com>"`) with the maintainer as
  `Co-Authored-By: Stefano Campanella <15182642+stefanocampanella@users.noreply.github.com>`.
- Conventions from reviews: no data paths, URLs, hardware or cluster references (JUPITER, ARCO-OCEAN) in code
  comments, docstrings, config comments or `scripts/` (fine as default values and in CLAUDE.md); `scripts/` must be
  portable, cluster-specific settings live under `clusters/<name>/`; user-specific settings (e.g. AWS profiles) come
  from environment variables, not configs; no backward-compatibility code for formats nobody uses; one library per job
  (e.g. TensorStore for all arrays); prefer simple implementations (no resume or blocking machinery unless failures
  show up); avoid `climatology.align_with` (slow).
- Storage quotas (`jutil` also reports compute budgets, see above): `df` shows only the whole filesystem; project quotas (data and inodes, soft and hard limits) are
  shown by `jutil project dataquota -p e-dev-2026d09-119 [-U TB]` (`jutil user dataquota` shows per-user rows without
  limits). Usage is updated a few times a day. Check before writing large data: on 27 Oct 2026 fscratch had 40 TB
  soft / 44 TB hard (8 M inodes), project1 20 / 22 TB (4 M inodes); each standardized store is 5.1 TB.
- Measurements: login nodes are shared (20-core, 143 GiB cgroup quota per user), so run only short, relative
  experiments there and longer ones on booster nodes; peak RSS is `VmHWM` in `/proc/self/status` (`ru_maxrss` was
  inconsistent).
