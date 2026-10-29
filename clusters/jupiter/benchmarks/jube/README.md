# GraphCast training benchmark (JUBE)

Benchmarks GraphCast training on JUPITER at all 50 depth levels (`configs/training/full.toml`) while varying how the
data is stored and read: layout of the standardized anomaly store (zarr v2, sharded zarr v3, SquashFS images of each),
Grain workers and threads, TensorStore I/O threads, CPUs per task and node count, and the model size that fits in GPU
memory. It follows the structure of the
[AI4HPC JUBE benchmark](https://gitlab.jsc.fz-juelich.de/SLFSE/codes/ai4hpc/-/tree/master/Bench):
a self-contained job template filled in by JUBE and submitted with `sbatch`.

| File                        | Purpose                                                                        |
|-----------------------------|--------------------------------------------------------------------------------|
| `graphcast_training.xml`    | JUBE benchmark: cases, job and config substitution, analysis and result tables |
| `train_bench.sbatch.in`     | SLURM job template, one job per workpackage                                    |
| `usage_sampler.sh`          | Samples node memory and GPU usage every 2 s on each node of a job              |
| `usage_summary.py`          | Summarizes these samples as `BENCH_*` lines in `job.out`                       |
| `run_final.sh`              | Submits the loader, io_threads, cpus and scaling experiments into `results/`   |
| `run_memfit.sh`             | Submits the GPU memory fit experiments into `results/`                         |
| `summarize.py`              | Writes `summary_jobs.csv` and `summary_cases.csv` for a results directory      |
| `pyproject.toml`, `uv.lock` | uv environment with JUBE 2.7.1 (installed from GitHub, JUBE is not on PyPI)    |

## Setup

```bash
cd clusters/jupiter/benchmarks/jube
uv sync                       # creates .venv with JUBE
uv run jube --version         # JUBE, version 2.7.1
```

## Running

Every run needs at least one sweep tag. The baseline is zarr_v2 with 4 Grain workers x 2 threads, 16 TensorStore I/O
threads, 1 node and 72 CPUs per task (the defaults of `.env` and `full.toml`):

| Tag          | Cases (format:workers:threads:io_threads:nodes:cpus)                                     | Jobs |
|--------------|------------------------------------------------------------------------------------------|------|
| `loader`     | zarr_v2, zarr_v3_lexi, zarr_v3_colexi, squashfs_v2, squashfs_v3_lexi, squashfs_v3_colexi at 0x4 (threads only), 1x2, 2x2, 4x2 and 8x2 | 60 |
| `io_threads` | baseline with 8, 16 and 32 I/O threads (`dataset.io_threads`)                            | 6    |
| `cpus`       | baseline with 8 and 72 CPUs per task                                                     | 4    |
| `scaling`    | baseline on 1, 2 and 4 nodes                                                             | 6    |
| `memfit`     | zarr_v2 2x2, 10 uniform meshes (120 km to the fine mesh) x 8 to 16 message passing steps | 100  |

Each case runs twice (`rep=1,2`, different seeds) for 60 training steps. Modifiers: `quick` (one repetition of 30
steps), `long` (1024 steps, 3 hour time limit), `devel` (develbooster reservation), `dryrun` (check the generated job
scripts with `bash -n` instead of submitting them), `small_buffers` (one sample per Grain worker and reader prefetch
buffer instead of two), `offload` (`policy.remat_loss = true`: the named activations go to pinned host memory),
`edge_chunks` (`policy.edge_chunk_size = 262144`), `fine`, `msg12`, `msg16` (model size), and for `memfit`
`one_worker` (1x2) or `no_workers` (0x4), which leave more host memory for offloading.

```bash
# Check what would be submitted, without using any compute time
uv run jube run graphcast_training.xml --tag loader dryrun --outpath /tmp/jube_dry

# Submit
uv run jube run graphcast_training.xml --tag io_threads cpus scaling   # 16 jobs
./run_final.sh            # loader, io_threads, cpus, scaling with long (76 jobs)
./run_memfit.sh           # memfit quick: defaults, offloading with 0 workers, edge chunks (150 jobs)

uv run jube status bench_run --id last    # wait for jobs
uv run jube continue bench_run --id last  # pick up finished jobs
uv run jube result -a bench_run --id last # tables, also bench_run/<id>/result/result_csv.dat
python3 summarize.py results/<date>-<name> # summary_jobs.csv and summary_cases.csv
```

Jobs use the code (`src/`) of the checkout JUBE runs from, so a worktree benchmarks its own branch.

Each workpackage directory (`bench_run/<id>/<wp>_submit/work`) contains the job script, the
generated `bench.toml`, the SLURM output `job.out`/`job.err` and the per-task logs `task-<rank>.out/err`.
Checkpoints and the saved model are removed at the end of each job.

## Earlier results

Before 28 Oct 2026 the benchmark ran at 10 levels by default, with the previous dataloader (zarr-python, standardizing
each sample from the raw dataset, climatology and anomaly_std); those results are in `results/2026-10-2*` and are not
comparable with the current ones. Findings that still guide the sweeps:

- XLA/JAX settings (latency hiding scheduler, PGLE, `JAX_OPTIMIZATION_LEVEL`, `JAX_MEMORY_FITTING_LEVEL`, the
  `cuda_malloc_async` allocator, the compilation cache, `NCCL_IB_DISABLE`) made no measurable difference on the coarse
  mesh (2026-10-22); `JAX_SHARE_BINARY_BETWEEN_HOSTS` hung at the first compilation on 2 nodes. The sweep was dropped.
- GPU memory at all levels (results/2026-10-25-memfit-v3): the processor in a scan (`policy.scan`, default) fits far
  more than unrolled (50 km with 16 message passing steps, 40 km with 8); offloading (`offload`) saves about 10 % of GPU
  memory for 8 to 87 GiB of pinned host memory and fills host memory with 2 Grain workers; edge chunks of 262144 fit 40
  km with 16 steps and 35 km with 12; the fine mesh (704078 nodes) fits with none. `model.connected_grid_nodes` (default)
  saves 4.7 GiB (median); `prevent_cse` did not move the limit. The limit is about 85 GiB per GPU (95 % of a GH200's 96
  GB, `XLA_PYTHON_CLIENT_MEM_FRACTION=.95` in `.env`); training logs the memory XLA plans for a step (`Memory usage` in
  task-0.err), also for cases that do not fit.
- Results measured before 78997d4 (2026-10-22) are not valid: `DeepTypedGraphNet` discarded all message passing steps.

## What is measured

`graphcast train` logs one line per step:
`Step N timings: data D s, compute C s, checkpoint K s, total T s, elapsed E s`.
The analysis reads rank 0 (`task-0.err`) and reports, over steps 10 and later (after compilation and
prefetch warm-up):

- `data_avg_s`, `data_max_s`: time rank 0 waits for its training batch, including host-to-device transfer.
- `compute_avg_s`: the training step of rank 0. Steps synchronize all ranks (gradient all-reduce), so a data stall
  of another rank appears here as compute time. `summarize.py` therefore adds `data_stall_avg` (mean over steps of
  the longest data wait of any rank, from all `task-<rank>.err`) and `compute_median` (robust to such stalls);
  `compute_min_s` is the fastest step. At 10 levels (earlier results), the fastest steps on the coarse mesh computed
  in about 0.41 s (8 message passing steps) to 0.48 s (16); the median was about 0.44 s with 8.
- `step_avg_s`, `step_std_s`, `samples_per_s` (training samples per second over all GPUs), `data_frac`.
- `first_step_s`: time from start to the end of step 0 (startup, data opening, compilation, first batches);
  `first_data_s`: the data part of step 0.
- `maxrss_gb`: peak memory of a task, from `sacct`, next to `est_mem_gb`, a rough estimate.
  `sacct` can lag behind; a value of 0 means it was not yet available.
- Node usage, sampled every 2 s on each node (`node_mem-<host>.csv`, `gpu-<host>.csv` in the workpackage
  directory) and summarized over all nodes: `node_mem_gb` (peak memory used, from `free`, without page cache),
  `node_avail_gb` (lowest available memory), `shm_gb` (peak shared memory, mostly Grain's buffers),
  `gpu_mem_gb` (peak GPU memory of any GPU; JAX preallocates `XLA_PYTHON_CLIENT_MEM_FRACTION` of it, so this
  shows the preallocation rather than use), `gpu_util_pct` and `gpu_w` (mean utilization and power over
  steps 10 and later).
- `status`: the exit status of the training step, or `incomplete` if it did not finish training (e.g. cancelled).
  A job can also finish training and then hang at exit; cancelling its training step (`scancel <job>.0`) lets the
  job script still report. `summarize.py` counts a job as ok if training finished after 1024 steps.

Results measured before 78997d4 (2026-10-22) are not valid: `DeepTypedGraphNet` discarded all message passing
steps, so the model hardly computed (about 25 ms per step regardless of mesh and message passing steps), training
looked loader bound with idle GPUs, and the XLA and model sweeps showed no differences.

## Design notes

These follow from the dataloader investigations (login-node `test_dataloader.py` runs):

- **Reading dominates the pipeline**: in Grain's execution summary the source accounts for more than 99 % of the wait
  time, so loader throughput scales with the number of samples read concurrently, `num_workers x num_threads`, and
  with TensorStore's I/O threads per reader (`dataset.io_threads`).
- **Demand is high**: each GPU task consumes one training sample per step (training computes no test loss). A sample
  holds all 50 levels (about 3.5 GB); `est_mem_gb` is a rough host memory estimate per task to compare with
  `maxrss_gb`. With the previous dataloader, more than 2 Grain workers ran out of host memory at all levels.
- **One stream per task**: a task runs `num_workers` loader processes. Booster nodes have about 857 GB of usable RAM
  (about 210 GB per task), shared with the activation offloading to pinned host memory.
- **CPUs per task**: with JUPITER's `CR_CORE` and `task/cgroup` settings, a task gets one core unless
  `--cpus-per-task` is set, and `srun` needs `SRUN_CPUS_PER_TASK` to inherit it. The jobs here and
  `clusters/jupiter/slurm/train.slurm` use 72 cores per task; the `cpus` sweep compares 8 with 72.
- **Checkpointing every step** (the Orbax default `save_interval_steps = 1`) would dominate step times, so the
  benchmark config saves only step 0 (`full.toml` saves every 500 steps). The checkpoint time is still reported
  separately.
- **Caching**: each workpackage uses a different seed, so that runs do not read samples cached by earlier
  runs. The jobs read the standardized anomaly stores written by `graphcast download`, all in
  `$FSCRATCH/graphcast/data/anomalies` so that only the layout differs: `zarr_v2`, the sharded zarr v3 stores
  (8 time x 50 depth) with the chunks of each shard in lexicographic (`zarr_v3_lexi`) or colexicographic
  (`zarr_v3_colexi`) order, and SquashFS images of each (`squashfs_v2`, `squashfs_v3_lexi`, `squashfs_v3_colexi`). The
  mask (`graphcast download-mask`) and `tendency_std` are small and loaded once, the same for all formats.
- **Concurrent jobs share the file system**: jobs that run while many others read from `/e/scratch` see data
  stalls of seconds on single ranks (in `results/2026-10-22-final-v2`, jobs started after the I/O experiments
  were in full swing took 0.88 instead of 0.47 s per step, with the same median compute). Compare cases that ran
  at the same time, or run sensitive experiments separately. Repetitions with the same seed that start together
  (e.g. the same case in two experiments) read the same samples in lockstep.
- Profiling in the Grain workers is turned off, since its servers would perturb the measurement.

The training configuration is `configs/training/full.toml` with `training_steps`, `seed` and
`checkpoints.save_interval_steps` overridden, so the benchmark follows changes to the model configuration.
Jobs run `scripts/train.sh` with `--config-dir`, `--output-dir`, `--train-dir` and `--tb-dir` in the
workpackage directory. It binds SquashFS images and zarr directories
alike: both launchers accept either source type through `DATASET`, `MASK` and `TENDENCY_STD`.
