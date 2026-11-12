# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Moby-Dick** (repository `moby-dick`): deep-learning models of the **global ocean circulation** (ARCO-OCEAN /
GLORYS data, depth levels instead of pressure levels). It started as a fork of DeepMind's GraphCast and was renamed on
8 Nov 2026 (legal reasons, Apache-2.0 §6, and divergence from upstream). Names: distribution and command `moby-dick`,
umbrella import package `mobydick` (shared code), one subpackage per model (first: `mobydick.leviathan`, class
`Leviathan`, Haiku module `leviathan`), environment variables `MOBYDICK_*` for all models (model-specific ones
`MOBYDICK_<MODEL>_*`), image `moby-dick.sif`. GraphCast stays named where the text describes the origin (headers,
NOTICE, comparisons with upstream); data paths and the checkout directory still say `graphcast`.
Leviathan runs on unstructured triangular ocean meshes (seamsh/gmsh `.msh` files), not the icosahedral multi-mesh. The main target system is JUPITER (JSC, GH200 GPUs, Slurm partition `booster`).
The README describes the project (rewritten on 8 Nov 2026); `TODO.md` lists open work.

## Environment and commands

Code runs inside an Apptainer image (`moby-dick.sif`, built from `spack/templates/container/moby-dick.def`:
NVIDIA JAX container `nvcr.io/nvidia/jax:26.09-py3` plus GDAL/PROJ from Spack). `clusters/jupiter/.env` sets
`APPTAINERENV_PYTHONPATH` to `src/`, so source edits take effect without rebuilding the image. Python is 3.12.

- CLI (`src/mobydick/__init__.py`, click; every command first prints the banner `src/mobydick/assets/banner.txt` to
  stderr, on rank 0 only, `MOBYDICK_NO_BANNER=1` to silence it; `python -m mobydick` works too): `moby-dick train CONFIG.toml OUTPUT TRAIN_CKPT`,
  `moby-dick make-ocean-mesh CONFIG.toml OUTPUT.msh` (mesh from `configs/mesh/*.toml`; their paths are container
  paths, as in the training configs: run it in the image with the data mounted, e.g. `apptainer exec --bind
  data/geometry:/data/geometry moby-dick.sif python -m mobydick make-ocean-mesh configs/mesh/uniform_120km.toml
  /data/geometry/uniform_120km.msh`; the `lambrechts*` configs also need the bathymetry at `/data/bathymetry`; the
  coastline coarsening is the minimum of `[[coarsen_boundaries.fields]]`, like the mesh size of `[[fields]]`; there is
  no `--data-path` since 12 Nov 2026), `moby-dick compute-grid-basis MESH MASK OUTPUT.npz`
  (draft: linear basis functions of the mesh at the ocean points of the training grid, barycentric weights in the
  decoder faces, a sparse matrix; `basis_utils.compute_basis` evaluates them at any points, e.g. the vertices of
  another mesh for the planned interpolating decoder and hierarchical meshes; on a flat mesh it equals scipy's
  `LinearNDInterpolator`, on the sphere it differs from interpolating in a projection by O(edge²); coordinates must be
  in metres, as trimesh's closest point uses an absolute tolerance. Cost (7 Nov 2026, login node, one process): 9.1 /
  11.8 / 13.7 / 18.6 s and ~2 GiB peak host memory for 120 / 50 / 35 / 25 km meshes at the 693,039 ocean points, as
  long as building the decoder edges at startup; 50 → 25 km mesh vertices 15.3 s, 1.7 GiB; the matrix is 18.5 MiB.
  So the basis can be computed when the model is built instead of saved),
  `moby-dick make-icosahedral-mesh CONFIG.toml OUTPUT.msh` (experimental: the icosahedral multi-mesh on the WGS84
  ellipsoid, for comparison with the ocean meshes; settings `splits`, `description`, `license` in
  `configs/mesh/icosahedral_*.toml`, `IcosahedralMeshConfig`; `mesh.path` of a training config can point to the
  output; `icosahedral_6.toml`, GraphCast's 6 refinements, gives 40,962 vertices, 327,660 directed edges, finest
  edges 110–132 km, 11.6 MB). Its `.msh`
  holds the finest triangles, the edges of all levels as line elements in the `multi_mesh_edges` physical group (line
  elements outside it are a mesh's boundary), and GraphCast's mesh size (longest finest edge) at every node. Trained
  on 8 Nov 2026 against the ocean meshes (see "Icosahedral multi-mesh vs ocean mesh" below); the file is
  `data/geometry/icosahedral_6.msh`.
- Graph statistics: `moby-dick graph-statistics CONFIG.toml MASK OUTPUT.json MESH...` (package
  `mobydick.graph_statistics`: `config`, `graphs`, `structure`, `geometry`, `quantities`, `report`, `cli`; settings
  `configs/analysis/graph_statistics.toml`): the graphs of a model on the grid of MASK and one mesh or a hierarchy
  (MESH paths finest first), built as the model builds them (a test checks they are equal): encoders grid → mesh 0
  and mesh i−1 → mesh i, processors, decoders mesh i → mesh i−1 and mesh 0 → grid (mesh-to-mesh graphs follow the
  grid's rules until hierarchical models define theirs). Every graph: nodes, edges, geodesic edge lengths; encoders:
  degrees; decoders: the 3-edges-per-receiver check; processors: degrees, density, one-way edges, connected
  components with, per component, the diameter (2-sweep lower bound) and shortest path lengths from `num_sources`
  sampled nodes (0 = exact), and triangle quality (aspect ratio, radius ratio, volume-length, angles, edge ratio);
  coverage of each pair of levels. Prints rich tables to stdout and writes every quantity with a description to the
  JSON; the log has only sizes and timings. networkx runs only the processors' path metrics: 28 s (120 km) to 147 s
  (25 km) per mesh on a login node. networkx's `NETWORKX_BACKEND_PRIORITY=cugraph` would run them on the GPU with
  nx-cugraph (`pip install .[graph-gpu]`, 3.2 GB with its dependencies; not in the image, untested).
  - **Path metrics scale with the mesh** (11 Nov 2026, all uniform meshes, 32 sources): the mean shortest path is
    L = 0.517 · N^0.504 hops (N nodes, within 2.3 % on every mesh), i.e. L × median edge length ≈ 11,400–11,900 km at
    every resolution; diameter ≥ 1.37 · N^0.49 (233 hops at 120 km, 515 at 50 km, 1,034 at 25 km). Average
    betweenness is exactly (L − 1)/(N − 2) and closeness ≈ 1/L (checked with exact networkx values on small
    meshes), so both scale as N^−0.5 and were dropped. The icosahedral M6 multi-mesh: L = 7.8, diameter ≥ 14.
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
  ...` (demo_5000 on 8 nodes per case and seed, validation loss on held-out windows; jobs use the checkout's `src`;
  MESH is a uniform mesh's size, `GEOMETRY_DIR/uniform_MESH.msh`, or another mesh file's name, `GEOMETRY_DIR/MESH.msh`).
- Benchmarks: JUBE in `clusters/jupiter/benchmarks/jube` (`uv sync`, then
  `uv run jube run training.xml --tag <sweep> [modifiers]`; add `dryrun` to check without submitting).
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
  `apptainer run moby-dick.sif python -m pytest` from the repository root (486 tests, ~4 min on a login node when it is not busy, ~3 min on a booster node with `sbatch` (a 30-minute 1-node job runs `apptainer run moby-dick.sif python -m pytest` from the checkout), CPU
  only, also on compute nodes; `run`, like training, goes through NVIDIA's entrypoint, see "CUDA forward
  compatibility on JUPITER"). Add `--cov` for coverage (93 % on 7 Nov 2026, ~4.7 min; `fail_under = 80`). The tests set
  `JAX_PLATFORMS=cpu` and 4 host devices (`tests/conftest.py`). Tests use synthetic data (small Zarr stores in
  `tmp_path`) or, if needed, public ARCO-OCEAN data read anonymously: no AWS profile, credentials or network may be
  assumed. Every test has a docstring saying what it checks and why (the behaviour or the bug it guards against), not
  just restating its name. The `absltest` modules inherited from GraphCast were ported to pytest in Nov 2026.
- Lint, format, types and licensing are pre-commit hooks (`.pre-commit-config.yaml`): `ruff check --fix` and `ruff
  format` (**2-space indentation**, 120-column lines; the whole codebase was formatted in b210470, listed in
  `.git-blame-ignore-revs`: `git config blame.ignoreRevsFile .git-blame-ignore-revs`), ShellCheck, `reuse lint` and
  ty. Run them in the image: `apptainer exec moby-dick.sif pre-commit run --all-files` (the first run downloads the
  hook environments: login nodes only), or `pre-commit install` to run them on each commit. The ty hook uses the
  environment's Python and adds the image's editable jax, jaxlib and flax (`/opt/jax`, `/opt/jaxlibs/jaxlib`,
  `/opt/flax`) as extra search paths; by hand: `apptainer exec moby-dick.sif ty check --python /opt/software/venv
  --extra-search-path /opt/jax --extra-search-path /opt/jaxlibs/jaxlib --extra-search-path /opt/flax src tests`.
  `src` and `tests` pass; where ty or the stubs cannot follow the code, use a targeted `# ty: ignore[rule]` with a
  reason. Notebooks are excluded from ruff.
- CI (`.github/workflows/ci.yml`, GitHub Actions, pushes to `develop`/`main` and pull requests), three independent
  jobs: `lint` (ruff and ShellCheck hooks, `setup-python` and pre-commit only), `reuse` (the REUSE hook) and `tests`:
  Python 3.12 with GDAL 3.13.1 from conda-forge (micromamba; keep in sync with `spack.yaml`), the package and dev group
  from PyPI, the ty hook, and `pytest --cov` with the 80 % gate (run even if ty fails); `coverage.xml` is an artifact and is uploaded to Codecov
  (`codecov/codecov-action`, secret `CODECOV_TOKEN`; upload errors do not fail the run; badge in the README). PROJ is
  not pinned: conda-forge's GDAL 3.13.1 builds need PROJ >= 9.8.1, while the image has GDAL 3.13.1 with PROJ 9.7.0
  (and pyproj 3.8.0 with its own PROJ 9.8.1, from the wheel); the CI log prints the same four versions. Pushing
  workflow files needs a GitHub token with the Workflows permission (fine-grained) or `workflow` scope.
- Image rebuild (needed after dependency changes; `src/` comes from `APPTAINERENV_PYTHONPATH`):
  `spack env create --dir spack-env spack.yaml`, `spack -e spack-env containerize > moby-dick.def`, then
  `env APPTAINER_CACHEDIR=$FSCRATCH/.apptainer apptainer build moby-dick.sif moby-dick.def`. Build to a separate file
  first and ask before replacing the shared `moby-dick.sif`: running jobs use it. Run `containerize` from the checkout
  whose `pyproject.toml` and `src/` the image should contain (a worktree works with
  `spack -e /e/project1/e-dev-2026d09-119/graphcast/spack-env`, as long as that environment was created from the
  current `spack.yaml`: it keeps a copy, and the template is `container/moby-dick.def` since the rename); the build
  takes ~6 min on a login node (40 min when the login node is busy, as on 12 Nov 2026: the current image, with
  seamsh v0.6.0+fork, was built then; the previous one is `moby-dick-2026-11-08.sif`). Previous images are kept as `moby-dick-YYYY-MM-DD.sif` (before the rename
  `graphcast-YYYY-MM-DD.sif`, which install the package as `graphcast`: with sources from `PYTHONPATH` they need
  `python -m mobydick`, and their `graphcast` command runs the old installed code). Past commands are in
  `~/.bash_eternal_history`.

### CUDA forward compatibility on JUPITER

These findings hold for JUPITER's booster nodes and the current image; on other clusters (other kernel drivers, GPUs
or container runtimes) the outcome differs and has to be checked again. The image's CUDA 13.4 user-mode driver
(`libcuda.so.615.71.09`, in `/usr/local/cuda/compat/lib.real`) is newer than the GH200 nodes' kernel driver
(610.57.04, CUDA 13.3), checked on 5 Nov 2026 (job 2698725, claude-work
`875379cd-19a4-4f43-a956-bd19e2b1b6a6/compat`):
- `apptainer run --nv` (training, `scripts/train.sh`) runs NVIDIA's entrypoint. Its bash sources `/etc/shinit_v2`
  (through `BASH_ENV`), which checks the compat library with `cudaCheck` and puts `lib.real` first in
  `LD_LIBRARY_PATH`: JAX loads the 615 library (`cuDriverGetVersion` 13040, "CUDA Forward Compatibility mode
  ENABLED" in the job output).
- `apptainer exec --nv … python` starts no shell: JAX loads the host's `libcuda` from `/.singularity.d/libs`
  (13030, minor version compatibility). Both work. The spack part of the image sets
  `LD_LIBRARY_PATH=/opt/views/view/lib:.:/.singularity.d/libs`, replacing NVIDIA's default (which listed
  `/usr/local/cuda/compat/lib`); the `.` comes from the generated recipe.
- `rm: cannot remove '/usr/local/cuda/compat/lib': Read-only file system` (once or twice per task, from every bash
  that sources `shinit_v2`, also bash children of Python) is harmless: the script removes and recreates the link to
  `lib.real` and touches a per-host marker so that the check runs once, which a read-only image cannot persist, so
  the check (about 1 s) repeats in each container. The link itself was made when the image was built.
- After a kernel driver upgrade to 615 or later, `cudaCheck` decides whether `lib.real` is used, and `exec` uses the
  host driver anyway: no rebuild needed for that.

## Standardized anomaly stores

`moby-dick download` (`downloader.py`, settings in `download_config.py`) standardizes the raw dataset once
(climatology alignment, filling, clipping, z-scores, min-max static fields, float32) and writes one local store;
training reads it with `dataloader.StandardizedAnomalySource`, which adds the batch dimension, the `datetime`
coordinate and relative time (the previous `ARCOTimeseriesSource` did all of this per sample from the raw dataset,
climatology and anomaly_std; it was removed on 28 Oct 2026). `moby-dick download-mask` writes the training mask: the
mask variable (`name`, default `glorys_mask`) at one depth position (`level`), as a single boolean `(lat, lon)`
DataArray (`configs/download/mask.toml`); training loads it as is (`dataloader_utils.get_grid`: lat, lon, mask).

- `tendency_std` (residual normalization in training) is the zero-mean standard deviation of the one-step differences
  of the standardized anomalies over the whole series, per variable and point (no time dimension). The copy in use,
  `/e/scratch/training2647/graphcast_data/zarr_v2/tendency_std` (also `s3://ogs-arco-ocean/tendency_std/res=0p25`,
  to be deleted there), was computed with arco-make's `diff_std` step, which is expensive: keep it. `moby-dick
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
  `PROJECT_DIR/moby-dick.sif`); `clusters/jupiter/slurm/download.slurm` sources `clusters/jupiter/.env` (or
  `--env-file`) and runs it. Paths are CLI arguments; the TOML holds settings only (Zarr format, v3 shard sizes,
  order of chunks within shards, Blosc, `num_workers` Dask threads, `io_threads` TensorStore concurrency).
- **Permanent copies** (3 Nov 2026), in `$PROJECT/data` (`/e/project1/e-dev-2026d09-119/data`; provenance and
  restore commands in its `README.md`):
  - `arco-ocean/{dataset,climatology,anomaly_std}`: raw ARCO-OCEAN as zarr v3, shards of 8 days × 50 levels,
    lexicographic. Converted from the zarr v2 copies on 21 Oct, and verified equal to them array by array on 3 Nov.
  - `anomalies/{zarr_v3_lexi,tendency_std,glorys_mask}`: copied from fscratch and verified (empty `rsync` dry run,
    identical windows).
  - **Working copies** for training and benchmarks stay on fscratch. `$FSCRATCH` purges files not modified or
    accessed for 30 days (and empty directories after 3 days; JUPITER user documentation): restore them from
    `$PROJECT/data` with `rsync` (8 parallel `rsync` on a login node: about 2.6 GiB/s).
  - The other copies in `$PROJECT` were deleted on 3 Nov 2026:
    - `zarr_v2` (raw zarr v2 plus the arco-make tendency_std);
    - `zarr_v3/tendency_std`;
    - `squashfs` (images of the raw data);
    - `graphcast_backup`;
    - `graphcast/data/dataset` (except `glorys12_static_res-0p0625_levels-10`, the bathymetry of the `lambrechts*`
      mesh configs) and `graphcast/data/stats` (2025, 10 levels).
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
- Icosahedral multi-mesh vs ocean mesh (8 Nov 2026, `docs/jupiter-hackathon/results/ico_vs_ocean-2026-11-08`, detailed
  version https://claude.ai/artifact/Bm8DGCiTHcJhL7k9AR5Lwg; jobs 2765280–2765287, 78 node-h):
  - **Setup:** the mesh study's protocol (demo_5000, 2 seeds) on `icosahedral_6` × 16 (GraphCast) and × 8, 50 km × 16
    and 120 km × 8, baselines rerun with the current code.
  - **The ocean mesh wins:** validation loss 120 km × 8 0.0838, icosahedral × 8 0.0837, 50 km × 16 0.0752 (−10.2 %),
    icosahedral × 16 0.0826 (−1.4 %), i.e. 9.8 % worse than 50 km × 16, a gap stable from step 2500.
  - **Only `zos` gains** (−16.7 % against 50 km × 16, from edges up to 7,076 km: the processor's diameter is 14 hops,
    against at least 515 on the 50 km mesh); the other targets are 10–31 % worse (sea-ice thickness +31 %). 23 % of the
    icosahedral nodes are over land; its 31,371 ocean-connected nodes match the 120 km mesh, which it equals on local
    fields. 8 → 16 steps gain only 1.3 % on it.
  - **Cost:** icosahedral × 16 0.78 s per step and 55 GiB, against 1.40 s and 74 GiB.
  - **Next:** long-range edges on the ocean mesh (hierarchy merged into the 50 km processor); icosahedral M7 for a
    resolution-matched comparison.
  - **Rerun on 12 Nov 2026** (current code and meshes: two-way edges, new encoder radius, Black Sea, seamsh
    v0.6.0+fork; study `ico_vs_ocean_v060`, jobs 2864519–2864527, 75.5 node-h): 120 km × 8 0.0833, icosahedral × 8
    0.0838, 50 km × 16 0.0746, icosahedral × 16 0.0826: icosahedral × 16 is 10.7 % worse than 50 km × 16, better only
    for `zos` (−16.0 %); the ocean cases gained 0.6–0.8 % since 8 Nov, the icosahedral ones nothing.
  - **The baselines are ~8 % better than in the mesh study** (120 km × 8 0.0907 → 0.0838, 50 km × 16 0.0818 → 0.0752;
    same config and validation windows), most likely from the edge-direction fix of 6 Nov (`0473751`, not ablated):
    do not compare runs from before 6 Nov with later ones.
- SquashFS images for the dataloader benchmarks, on the same filesystem as the stores (fair comparison):
  `$FSCRATCH/graphcast/data/anomalies/squashfs/{zarr_v2,zarr_v3_lexi,zarr_v3_colexi}.sqsh`, written on a login node
  by `mksquashfs.sh` there, with the settings of the earlier raw-data images (deleted on 3 Nov 2026) (from
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
task, model (the model's own settings, e.g. `leviathan.config.LeviathanConfig`, selected by `architecture`), mesh, mask, dataset, grain_dataset, normalization, policy, optimizer, checkpoints, …). Paths in the
TOML are container paths (`/data/...`). Grain worker/buffer settings come from `MOBYDICK_GRAIN_*` environment
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
- **Step order** (sequential on purpose; overlapping the next batch with the step gained only 2.5 %): batch to the
  GPUs, step, wait, validation if due, preemption check, checkpoint if due, TensorBoard, the step line `Step N
  timings: data D s, compute C s, checkpoint K s, total T s, elapsed E s` (parsed by JUBE). Metrics become host
  values only when saved or logged, and TensorBoard writes on process 0 only.
- **Checkpoints** (`[checkpoints]`, `orbax_utils`): the loop decides the saves, after step n when `(n + 1) %
  save_interval_steps == 0` (a multiple of `validation.interval_steps`, so every regular checkpoint has a
  `valid_loss`), after the last step, and at a preemption. A checkpoint holds the state after its step, so
  `--continue` resumes at latest + 1 (before Nov 2026 it resumed at latest, one step behind). Retention:
  `LatestN(keep_latest)` for resuming (1 is enough: Orbax deletes older checkpoints only after the new one is
  committed, and an interrupted save leaves no committed step), plus `BestN(keep_best)` by validation loss; the
  final model is the best checkpoint, or the latest without validation. Orbax 0.12.4 with `best_fn` kept only the
  `max_to_keep` lowest single-batch losses, so it could delete the latest checkpoint (fixed Nov 2026). Checkpoint
  directories written before then store the metrics as a `(loss, diagnostics)` list and cannot be continued.
- **Preemption:** Slurm sends SIGTERM 120 s before the limit (`train.slurm`, `--signal=TERM@120`; KillWait is 30 s).
  JAX's preemption service catches it, all processes agree on a sync step (`reached_preemption`), the loop saves,
  waits for the commit and exits with status 75 without saving the model: resume with `--continue` (same
  `MOBYDICK_GRAIN_NUM_WORKERS`). Grain workers ignore SIGTERM (`dataloader_utils.ignore_sigterm`), and the loop closes
  the iterator in a `finally` to stop them (Python's exit would send them SIGTERM and wait forever).
- **Orbax:** the per-process mesh workaround for Orbax issue 2545 was removed (fixed in the image's Orbax 0.12.4).
  A save copies one replica of each replicated array to the host (split over the GPUs by replica-parallel saving),
  inside the blocking part (0.4–0.6 s at 120 km × 8); the write runs in the background.
- **Measurements (4 Nov 2026, `training2647`):**
  - **Preemption at 50 km × 16 on 8 nodes** (jobs 2684368/2684369): all 32 processes reached the sync step within the
    second after SIGTERM. The save blocked 0.76 s, 1.3–1.6 s with the commit; the checkpoint is 392 MB (120 km × 8:
    222 MB, 0.42 s, 0.7 s). The continued run resumed at the next step.
  - **Host time per step** (1 node, 120 km × 8, new loop vs old, same node): equal within run-to-run noise; compute
    alternates between 0.40 and 0.42 s per step between runs. TensorBoard costs 7–8 ms per step on process 0: about
    260 µs per `tf.summary.scalar` call (eager TensorFlow dispatch; the writes themselves are queued), not the
    conversions. That is 0.6 % of a 50 km × 16 step, so it was left as is.
  - **Host-to-device copy of a batch:** about 19 ms at 50 levels, partly overlapped with compute already (waiting for
    it raises data time by 19 ms and lowers compute by 12 ms, at 120 km × 8 and 50 km × 16). Moving it into the
    dataloader could save at most 20–30 ms per step: not worth the code (bar for speed-ups: 150 ms per step).
  - **Non-determinism:** two identical runs give different parameters (median ‖a−b‖/‖a‖ per array 2.6e-4 after 60
    steps on 2 nodes). The losses hide it: each GPU's loss is a bfloat16 value (`Bfloat16Cast`), so most steps agree
    bit for bit and the others differ by one bfloat16 unit of one GPU (2^-12 of the 4-GPU mean). The source is XLA's
    non-deterministic GPU operations (atomic scatter-adds and reductions): with `--xla_gpu_deterministic_ops=true`,
    repeated runs and a restored run equal the uninterrupted run bit for bit (parameters and optimizer state), but a
    step takes 8.2 s instead of 0.42 s at 120 km × 8, and 50 km × 16 needs 81.5 GiB (OOM). Use it only to debug or
    to check restores (`APPTAINERENV_XLA_FLAGS`).

**Predictor stack** (`training_utils.get_predictor`), outermost first:
`Mask` (`mask.py`, per-variable loss masks) → `OutputNormalization` (`normalization.py`, residuals / tendency_std) →
[`InputNormalization`, only with input statistics; the standardized store needs none] → `Bfloat16Cast` (`casting.py`)
→ the model of `[model] architecture` (`models.MODELS`; `Leviathan`, `leviathan/model.py`). Training is one-step; `autoregressive.py` / `rollout.py` handle multi-step.

**Leviathan model** (`leviathan/model.py`, settings `leviathan/config.py`): grid ↔ mesh connectivity comes from `mesh_connectivity.py` (encoder edges by
radius query, decoder edges by containing triangle, both restricted by the ocean mask; the k-d tree query uses all
cores since 11 Nov 2026, one thread before, with the same edges). The encoder radius at each mesh vertex is
`radius_query_fraction_edge_length` × the longest triangle edge at the vertex (`mesh_graph.longest_incident_edges`,
since 11 Nov 2026; before, × the mesh size field, i.e. the target size, which left ocean points inside coastal
triangles without encoder edges; GraphCast uses the longest edge of the whole mesh). With
`model.connected_grid_nodes` only grid points that are encoder senders or decoder receivers become grid nodes;
the decoder writes back only ocean (unmasked) points. Three `DeepTypedGraphNet`s (`deep_typed_graph_net.py`,
built from `typed_graph_net.py` blocks on `typed_graph.TypedGraph`) form encoder (grid2mesh), processor
(mesh_gnn, `gnn_msg_steps` steps) and decoder (mesh2grid). Feature construction is in `leviathan/model_utils.py`;
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
workers; residuals are normalized with `tendency_std`, and the boolean surface mask comes from `moby-dick
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
  2026; the `uniform_coarse.msh` and `uniform_fine.msh` symlinks to the new names were removed on 8 Nov 2026). Each
  `.msh` stores the size field it was made with (views `MeshSize` and one per field) and
  the config's description. Measured median edge lengths match the targets: 117.8 km for 120 km, 34.9 for 35, 30.0
  for 30.
- **The 25 km mesh** (`uniform_25km`, formerly `uniform_fine`, 704,078 nodes, 31 Jan 2026):
  - its stored size field is 25 km at every node, and its median edge is 25.0 km;
  - it was made from a config that was never committed: `uniform_fine.toml` asked for 30 km then, and for 35 km later,
    identical to `uniform_35km.toml` (its `domain.sampling` is ignored by the code). It was replaced on 30 Oct 2026 by
    `uniform_25km.toml`;
  - node counts scale as 1/size², so 363,900 × (35/25)² ≈ 713,000;
  - its description (150 km), and that of the 120 km mesh, came from an outdated template; both `.msh` files keep it.
- **Descriptions** of all `configs/mesh/*.toml` match their fields (checked on 30 Oct 2026). The `lambrechts*`
  configs take the smaller of a shore-proximity and a bathymetry field; they said "uniform 300 km" before.
  `domain.sampling` in them is ignored (the shore-proximity field has its own).
- **seamsh limits the domain** (11 Nov 2026, read from the fork's `develop` at `f1ab87d` and measured with the
  standalone `validate_seamsh.py`): `coarsen_boundaries` keeps only the water reachable from the single seed `x0`
  (moby-dick: the stereographic origin, the North Pole) by a flood fill over a Delaunay triangulation of the coastline
  samples that crosses only edges longer than the coarsening size, so basins behind narrower straits are dropped:
  Black Sea and Sea of Azov at 10 km (all configs), plus the Mediterranean and Baltic at 20 km, the Red Sea at 30 km,
  the White Sea at 50 km. `seamsh.gmsh.mesh` also builds one plane surface (largest loop exterior, all others holes),
  so a second basin could not be meshed anyway. Natural Earth's coastline counts ice shelves as ocean. Mesh files
  hold nodes in no triangle: the outlines of islands smaller than the mesh size, which survive the coarsening but are
  covered by triangles (743 of 34,750 nodes at 120 km, 632 islands up to ~100 km across; 655 of 181,980 at 50 km);
  `read_mesh` drops them and logs how many (since 11 Nov 2026). A
  refactoring of seamsh (tests, hooks, fixes, several seeds and one surface per basin) is prepared in a separate PR to
  the fork's `develop`: kit with PR guide, notes (bugs with line numbers), validation script, reference results, data
  and moby-dick's mesh code in `$PROJECT/claude-work/ed336b51-7c1c-4db2-b3ca-b6880ad836fd/seamsh-refactoring-kit{,.tar.gz}` (copy it out before this session's directory is deleted). The script reproduces
  `uniform_120km.msh` exactly (vertices within 1.4e-9 m, same triangles).
- **Black Sea in the configs (11 Nov 2026):** every ocean config refines the coarsening
  (`[[coarsen_boundaries.fields]]`, 250 m, growth 0.02) and the mesh (`[[fields]]`, 3 km, growth 0.3) with the
  `point_refinement` field at the Bosporus, Dardanelles and Kerch Strait.
- **Meshes made with seamsh v0.6.0+fork (12 Nov 2026):** `pyproject.toml` pins the fork's release tag (its
  `doc/fork.rst`: moby-dick's calls run unchanged; `mesh` optimises the triangles by default, so meshes differ; files
  hold no node outside the triangles; `x0` may be several seeds). The ten `data/geometry/uniform_*km.msh` are
  regenerated from the configs with it (copies in `$FSCRATCH/graphcast/data/geometry_seamsh060_2026-11-12`, with their
  graph statistics); the previous generations are in `data/geometry/previous_2026-11-12/` (seamsh `f1ab87d`, with the
  straits) and `data/geometry/previous_2026-10/` (without the Black Sea, mesh stored twice). Runs from before 12 Nov used
  the older meshes. Against seamsh `f1ab87d` with the same configs: one component each, the Black Sea and Sea of Azov
  in; triangles better on the whole (25th percentile of the smallest angle per face +1° at every size, median
  volume-length up), smallest angle better at 85, 70, 50, 30 and 25 km (25 km 5.4° → 9.0°), slightly worse at 100, 40
  and 35 km (100 km 3.7° → 3.4°); 2–3 % fewer encoder edges; ocean points without encoder edges within ±5 % (120 km
  287, 50 km 774, 25 km 2,157; none inside a triangle; 2–9 in the Black Sea box), longest decoder edge 261 km (120 km),
  196 km (50 km); no node dropped when reading. Path scaling over the ten meshes: L = 0.542 · N^0.501 (within 3.1 %),
  L × edge 11,300–12,200 km, diameter ≥ 2.65 · N^0.45. The strait elements carry no grid point but are the only link to
  the Black Sea, Azov and Marmara: without the nodes in neither encoder nor decoder, 4 % of the paths between
  data-carrying nodes break at 120 km (0.3 % at 50 km); the Black Sea is 45 (120 km) and 48 (50 km) hops from the
  Aegean. Mesh-to-mesh encoder 50 → 120 km: 6 finer vertices without edges, all outside the coarser mesh.
- **The Black Sea can be kept with today's seamsh** (11 Nov 2026, `strait_experiment.py` in the kit): the 1:10m
  coastline keeps the straits open (narrowest gaps: Bosporus 1.09 km, Dardanelles 2.36 km, Kerch 2.51 km). A
  coarsening size of 0.25–0.5 km at the three straits growing by at most 0.05 km per km to 10 km (they are 30–60 km
  long: faster growth closes them again), and a mesh size of 1–3 km there growing 0.3 km per km to the base size, give
  one connected ocean with the Black Sea and Sea of Azov: at 120 km 34,449 used nodes with 3 km (34,007 today),
  34,621 with 1 km; at 50 km +849 nodes with 1 km; smallest angles unchanged (7.2° at 120 km, 15.0° at 50 km),
  shortest edges 1.4 km / 0.9 km, meshing time unchanged (now in the configs: see the bullet above). For the processor
  the Black Sea stays a separate basin: 47–61 hops from the Aegean through the strait elements, more than any
  `gnn_msg_steps`; what changes is that its ocean points get mesh vertices. A coarser coastline (30 km) with nine
  straits refined keeps every basin but lowers the smallest angle at 120 km to 2.5°: 10 km stays the better choice.
- **Processor edges go both ways since 11 Nov 2026:** `mesh_graph.faces_to_edges` (from GraphCast) gave each face's
  edges in one direction, which covers both only on a closed mesh. On the ocean meshes the coastline edges belong to
  one face, so they were one-way (3,612 of 194,040 edges at 120 km, 9,357 of 1,061,739 at 50 km): every model trained
  on ocean meshes before then, including the mesh study and the icosahedral comparison, passed messages along the
  coast in one direction only. Their reverses are now appended (`.msh` files unchanged: edges are rebuilt from the
  faces when read). Rerun with both fixes (two-way edges, encoder radius from the longest triangle edge; meshes without the Black Sea; study `edge_fixes`, jobs 2841380–2841383, 46 node-h): validation loss 120 km × 8 0.08382 → 0.08373 (−0.10 %), 50 km × 16 0.07524 → 0.07494 (−0.39 %), within the seed spread (0.04–0.40 %), every variable within ±1.1 %; XLA memory +1.1 GiB (120 km, 54.9 GiB) and +0.8 GiB (50 km, 74.8 GiB); step time +8 % / +3.5 %, within the run-to-run variation. No measurable effect on skill.
- **Coverage of the ocean points** (8 Nov 2026, `moby-dick graph-statistics`): the uniform meshes' domain leaves out the
  Black Sea, so about 900 ocean points of the training mask (40–47° N, 27–42° E) are decoded from the vertices of the
  nearest face in the Aegean, up to 1,232 km away, and are never encoder senders. In all, 2,222 (120 km) and 2,342
  (50 km) ocean points had no encoder edge; the others were coastal points farther from every vertex than the local
  radius (0.6 × target size): 1,129 (120 km), 820 (50 km), 616 (25 km) inside a triangle, the rest outside the mesh.
  The icosahedral mesh covers every point. Since 11 Nov 2026 the radius is 0.6 × the longest triangle edge at each
  vertex, which reaches every point inside a triangle (5–8 % more encoder edges): 1,222 (120 km), 1,725 (50 km) and
  3,099 (25 km) points remain without encoder edges, all outside the mesh domain: the Black Sea (936) and coastal
  channels and ice-shelf fronts (e.g. Tierra del Fuego, the Weddell Sea). A radius of 1.0 × the longest edge leaves
  993 / 1,191 / 1,546 for 2.9 × the encoder edges. Mesh hierarchies: finer vertices without encoder edges to the
  coarser mesh went from 260 to 2 (50 → 120 km) and 802 to 8 (25 → 50 km), the rest outside the coarser mesh.
  To fix the rest: the Black Sea and the coastal gaps in the domain, or out of the mask.
  - Mesh vertices without encoder edges are not lost: on the 25 km mesh 7.8 % of the vertices receive no grid point
    (radius ~15 km against a 0.25° grid, i.e. a mesh finer than the grid), but the processor updates every vertex at
    every step (`_run_mesh_gnn` has no node mask; their encoder state is the embedding of their position) and 99.3 %
    of the vertices send decoder edges. Such a vertex starts without data and holds its neighbours' after one step:
    it costs one step of reach there, not coverage.
  - Vertices in neither the encoder nor the decoder never exchange data with the grid but are full processor nodes:
    they receive and pass on messages at every step. Measured (11 Nov 2026, `relay_nodes.py` in the session
    directory, shortest paths between the other nodes from 32 sources, with and without them): on the icosahedral
    mesh they are the 9,591 land vertices (23 %), and they are needed as relays: without them 20 % of the paths
    between data-carrying nodes get longer (mean × 1.032), across continents. On the ocean meshes they are the 233
    (120 km) and 1,122 (50 km) vertices poleward of 60° S, median 79° S, i.e. under Antarctic ice shelves outside
    the GLORYS mask: dead ends, with no path between data-carrying nodes through them (one 120 km vertex is connected
    to the rest only through them; at 50 km 0.2 % of the paths get longer); they still exchange messages with their
    neighbours.

## Repository notes

<!-- REUSE-IgnoreStart -->
- Licensing follows the [REUSE specification](https://reuse.software/spec-3.3/) (since 7 Nov 2026); check it with
  `uvx reuse lint` (license texts in `LICENSES/`; `LICENSE` stays for GitHub).
  - Code, configs, scripts and docs are Apache-2.0. Images and animations are CC-BY-4.0. The Argonne, GRC and OGS logos
    in `assets/poster.svg` are their owners' trademarks.
  - Every file that takes comments starts (after a shebang) with `SPDX-FileCopyrightText: YEAR Stefano Campanella`
    (year the file was added) and `SPDX-License-Identifier: Apache-2.0`; Python files add the Apache license text.
  - Files without such a header (Markdown, HTML, CSV, notebooks, images, lock files, the container template) are
    covered by `REUSE.toml`: add a pattern there for new kinds of files.
  - Files with code from GraphCast also keep `SPDX-FileCopyrightText: 2023 DeepMind Technologies Limited` and a
    sentence saying they were modified (Apache §4(b)), and are listed in `NOTICE` (§4(d)): update it when code from
    GraphCast is copied into another file. `spack/templates/container/moby-dick.def` is Spack's Singularity template
    (Apache-2.0 OR MIT), modified.
  - The license grants no rights to DeepMind's, Google's or GraphCast's names (§6), hence the rename to Moby-Dick.
  - **Dependencies** (audit of 12 Nov 2026, the image's 224 Python distributions and the Spack libraries; tables in
    `$PROJECT/claude-work/ed336b51-7c1c-4db2-b3ca-b6880ad836fd/licenses/`; not legal advice):
    - All direct dependencies are permissive (Apache-2.0, BSD, MIT, PSF-like; tqdm MPL-2.0 AND MIT, file-level
      copyleft, fine unmodified) **except gmsh** (GPL-2.0-or-later with the Gmsh exception, Ubuntu `python3-gmsh`)
      **and seamsh** (GPL-3.0: README, `COPYING.txt`, classifier; its file headers say LGPL-3.0-or-later, but
      `libseamsh.so` compiles gmsh's `polymesh.c`, GPL-2.0-or-later, so it cannot be LGPL as a whole).
    - Apache-2.0 code may be combined with GPL-3.0 code (one-way compatible; gmsh's "or later" allows GPL-3.0), so
      moby-dick's own files stay Apache-2.0 and the repository (which contains no GPL code) is fine. But every
      `moby-dick` command loads gmsh and seamsh (`mobydick/__init__.py` imports `geospatial_mesh`; training reads
      `.msh` files with gmsh): distributing moby-dick *together with* them (e.g. `moby-dick.sif`, or a bundle) is
      distributing a combined work under GPL-3.0 terms (source of everything, no further restrictions). The same
      process also loads proprietary NVIDIA libraries (CUDA, cuDNN, CUTLASS DSL, through jaxlib), whose terms GPL-3.0
      does not allow in a combined work unless they count as System Libraries: a gray area. Running on JUPITER is not
      distribution; publishing the image would be. Remedy if needed (code change, not done): gmsh and seamsh as an
      optional `meshing` extra, imported lazily, and `.msh` files read without gmsh (e.g. meshio, MIT), so that
      training never loads GPL code.
    - Native libraries: GDAL, PROJ MIT; GEOS LGPL-2.1 (dynamically linked: fine); gcc runtime GPL with the runtime
      exception; Spack's and Ubuntu's readline (GPL-3.0) are not linked by GDAL; OpenCASCADE (with gmsh) LGPL-2.1.
    - The base image `nvcr.io/nvidia/jax` is governed by the NVIDIA Software License Agreement and Product-Specific
      Terms for AI Products (`/workspace/license.txt`): check them before sharing `moby-dick.sif` outside the project.
      The `graph-gpu` extra (nx-cugraph, Apache-2.0) pulls proprietary CUDA libraries.
    - Mesh files are declared CC BY-NC-SA 4.0 in every mesh config (perhaps copied from GraphCast's weights license),
      although their inputs need neither NC nor SA (Natural Earth is public domain; GLORYS bathymetry, used by the
      `lambrechts*` meshes, is under the Copernicus Marine licence, which allows commercial use with attribution, and
      that attribution is missing from their license text). Trained model weights have no declared license, and the
      README does not state the data attributions (Copernicus Marine for GLORYS and WAVERYS, C3S for ERA5, CEMS for
      GloFAS) needed when distributing weights, predictions or derived data. Decisions for the maintainer.
<!-- REUSE-IgnoreEnd -->
- Do not commit `moby-dick.sif`, `moby-dick.def` (generated copy), `spack-env/`, `runs/`, `logs/`, `data/`,
  benchmark `results/` or the local `repro/` reproducers.
- Development branch: `develop` (`develop@esiwace3_hackathon` until Nov 2026).
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
- Workflow: one feature branch and PR per task into `develop`; the maintainer reviews on GitHub
  or asks for changes directly. Reply to each review comment once addressed. Keep PR descriptions short (purpose,
  usage, design, verification), without detailed benchmark numbers; keep those here. Commits are authored by the
  maintainer (the git user, no `--author`), with Claude as co-author through the single `Co-Authored-By: Claude …
  <noreply@anthropic.com>` trailer that Claude Code adds.
- Conventions from reviews: no data paths, dataset URLs (e.g. of the ARCO-OCEAN stores), hardware or cluster
  references (JUPITER, ARCO-OCEAN) in code comments, docstrings, config comments or `scripts/` (fine as default values
  and in CLAUDE.md); links to references (papers, documentation, issues) are welcome in comments; `scripts/` must be
  portable, cluster-specific settings live under `clusters/<name>/`; user-specific settings (e.g. AWS profiles) come
  from environment variables, not configs; no backward-compatibility code for formats nobody uses; one library per job
  (e.g. TensorStore for all arrays); prefer simple implementations (no resume or blocking machinery unless failures
  show up).
- Storage quotas (`jutil` also reports compute budgets, see above): `df` shows only the whole filesystem; project quotas (data and inodes, soft and hard limits) are
  shown by `jutil project dataquota -p e-dev-2026d09-119 [-U TB]` (`jutil user dataquota` shows per-user rows without
  limits). Usage is updated a few times a day. Check before writing large data: on 27 Oct 2026 fscratch had 40 TB
  soft / 44 TB hard (8 M inodes), project1 20 / 22 TB (4 M inodes); each standardized store is 5.1 TB.
- Measurements: login nodes are shared (20-core, 143 GiB cgroup quota per user), so run only short, relative
  experiments there and longer ones on booster nodes; peak RSS is `VmHWM` in `/proc/self/status` (`ru_maxrss` was
  inconsistent).
