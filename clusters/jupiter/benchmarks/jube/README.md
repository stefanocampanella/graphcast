# GraphCast training I/O benchmark (JUBE)

Benchmarks GraphCast training on JUPITER while varying how the data is stored and read:
dataset format (SquashFS images, zarr v2, sharded zarr v3), Grain workers and threads, the
zarr thread pool and request concurrency, CPUs per task and node count. It follows the structure of the
[AI4HPC JUBE benchmark](https://gitlab.jsc.fz-juelich.de/SLFSE/codes/ai4hpc/-/tree/master/Bench):
a self-contained job template filled in by JUBE and submitted with `sbatch`.

| File                        | Purpose                                                                        |
|-----------------------------|--------------------------------------------------------------------------------|
| `graphcast_training.xml`    | JUBE benchmark: cases, job and config substitution, analysis and result tables |
| `train_bench.sbatch.in`     | SLURM job template, one job per workpackage                                    |
| `pyproject.toml`, `uv.lock` | uv environment with JUBE 2.7.1 (installed from GitHub, JUBE is not on PyPI)    |

## Setup

```bash
cd clusters/jupiter/benchmarks/jube
uv sync                       # creates .venv with JUBE
uv run jube --version         # JUBE, version 2.7.1
```

## Running

Every run needs at least one sweep tag:

| Tag       | Cases (format:workers:threads:zarr_pool:zarr_concurrency:nodes:cpus)  | Jobs |
|-----------|-----------------------------------------------------------------------|------|
| `format`  | squashfs, zarr_v2, zarr_v3 at 2x2 and 8x2 Grain workers x threads     | 6    |
| `grain`   | zarr_v3 at 0x4 (threads only), 1x2, 4x2 (baseline), 8x1, 4x4          | 5    |
| `zarr`    | zarr_v3 4x2 with zarr pool/concurrency 4/10, 4/32, 8/10, 32/10, 32/32 | 5    |
| `cpus`    | zarr_v3 4x2 with 1 and 8 CPUs per task                                | 2    |
| `scaling` | zarr_v3 4x2 on 2 and 4 nodes                                          | 2    |

Each case runs twice (`rep=1,2`, different seeds) for 60 training steps. Modifiers:
`quick` (one repetition of 30 steps), `devel` (develbooster reservation), `dryrun` (check the
generated job scripts with `bash -n` instead of submitting them), `all_levels` (train on all 50 depth
levels instead of the 10 of `full.toml`; samples are about 5 times larger). `small_buffers` (one
sample per Grain worker and reader prefetch buffer instead of two) and `no_offload` (no activation
offloading to pinned host memory) reduce host memory; the `lowmem` sweep runs only cases with at most two
Grain workers. Use a separate `--outpath`
for `all_levels` runs, e.g. `bench_run_all_levels`, to keep their results apart.

```bash
# Check what would be submitted, without using any compute time
uv run jube run graphcast_training.xml --tag format grain dryrun --outpath /tmp/jube_dry

# Submit
uv run jube run graphcast_training.xml --tag format          # 12 jobs
uv run jube run graphcast_training.xml --tag format grain zarr cpus scaling   # 40 jobs

uv run jube status bench_run --id last    # wait for jobs
uv run jube continue bench_run --id last  # pick up finished jobs
uv run jube result -a bench_run --id last # tables, also bench_run/<id>/result/result_csv.dat
```

Rough cost: one job is about 5 minutes of startup and compilation plus 60 steps at about 15 s, so about
20 minutes per node. All sweeps are 48 node-jobs, about 16 node-hours (about 5 with `quick`).
The `cpus` case with 1 CPU per task may hit the 30-minute limit. It then reports `status=terminated`
with the steps completed so far.

Each workpackage directory (`bench_run/<id>/<wp>_submit/work`) contains the job script, the
generated `bench.toml`, the SLURM output `job.out`/`job.err` and the per-task logs `task-<rank>.out/err`.
Checkpoints and the saved model are removed at the end of each job.

## What is measured

`graphcast train` logs one line per step:
`Step N timings: data D s, compute C s, checkpoint K s, total T s, elapsed E s`.
The analysis reads rank 0 (`task-0.err`) and reports, over steps 10 and later (after compilation and
prefetch warm-up):

- `data_avg_s`, `data_max_s`: time waiting for the train and test batches, including host-to-device transfer.
  This is the stall the data pipeline causes.
- `compute_avg_s`: the training step itself. It should not depend on the I/O settings and serves as a control.
- `step_avg_s`, `step_std_s`, `samples_per_s` (training samples per second over all GPUs), `data_frac`.
- `first_step_s`: time from start to the end of step 0 (startup, data opening, compilation, first batches);
  `first_data_s`: the data part of step 0.
- `maxrss_gb`: peak memory of a task, from `sacct`, next to `est_mem_gb`, a rough estimate.
  `sacct` can lag behind; a value of 0 means it was not yet available.

## Design notes

These follow from the dataloader investigations (login-node `test_dataloader.py` runs):

- **Reading dominates the pipeline**: in Grain's execution summary the source (zarr reads and
  normalization) accounts for more than 99 % of the wait time. Loader throughput therefore scales with
  the number of samples read concurrently, `num_workers x num_threads`. The zarr thread pool hardly matters
  above about 4 threads per reader, and too small a pool hurts.
- **Demand is low**: each GPU task consumes one train and one test sample per step. At the roughly 15 s per
  step seen in earlier runs, that is about 0.13 samples/s per task, below what even small loader
  configurations deliver on the login node. The benchmark therefore includes small configurations (threads only, 1
  worker) to find where data starts to stall the GPUs. It also reports startup time and
  memory, where the formats and settings can differ even when steady-state steps do not.
- **Train and test streams each start `num_workers` processes**, so a task runs twice as many workers as
  configured, and the memory estimate counts both streams. Booster nodes have about 878 GB RAM (about 210 GB
  per task), shared with the activation offloading to pinned host memory.
- **CPUs per task**: with JUPITER's `CR_CORE` and `task/cgroup` settings, a task gets one core unless
  `--cpus-per-task` is set, and `srun` needs `SRUN_CPUS_PER_TASK` to inherit it. The current
  `clusters/jupiter/slurm/train.slurm` sets neither, so all of a task's loader processes would share one core.
  The jobs here use 72 cores per task; the `cpus` sweep measures 1 and 8.
- **Checkpointing every step** (the default `save_interval_steps = 1`) would dominate step times, so the
  benchmark config saves only step 0. The checkpoint time is still reported separately.
- **Caching**: each workpackage uses a different seed, so that runs do not read samples cached by earlier
  runs. All datasets are on `/e/scratch`, so that only the storage format differs: the SquashFS images in
  `/e/scratch/training2647/ogs_team`, the zarr v2 extraction of these images and the zarr v3 conversion (same chunks,
  lz4 Blosc, sharded 8 time x 50 depth) in `/e/scratch/training2647/graphcast_data`.
- Profiling in the Grain workers is turned off, since its servers would perturb the measurement.

The training configuration is `configs/training/full.toml` with `training_steps`, `seed` and
`checkpoints.save_interval_steps` overridden, so the benchmark follows changes to the model configuration.
Jobs run `scripts/train.sh` with `--config-dir`, `--output-dir`, `--train-dir` and `--tb-dir` in the
workpackage directory. It binds SquashFS images and zarr directories
alike: both launchers accept either source type through `DATASET`, `CLIMATOLOGY`, `ANOMALY_STD`,
and `TENDENCY_STD`.
