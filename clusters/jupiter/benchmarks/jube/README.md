# GraphCast training benchmark (JUBE)

Benchmarks GraphCast training on JUPITER while varying how the data is stored and read:
format of the standardized anomaly store (SquashFS images, zarr v2, sharded zarr v3), Grain workers and threads, the
zarr thread pool and request concurrency, CPUs per task and node count, as well as XLA/JAX settings and the
number of message passing steps. It follows the structure of the
[AI4HPC JUBE benchmark](https://gitlab.jsc.fz-juelich.de/SLFSE/codes/ai4hpc/-/tree/master/Bench):
a self-contained job template filled in by JUBE and submitted with `sbatch`.

| File                        | Purpose                                                                        |
|-----------------------------|--------------------------------------------------------------------------------|
| `graphcast_training.xml`    | JUBE benchmark: cases, job and config substitution, analysis and result tables |
| `train_bench.sbatch.in`     | SLURM job template, one job per workpackage                                    |
| `usage_sampler.sh`          | Samples node memory and GPU usage every 2 s on each node of a job              |
| `usage_summary.py`          | Summarizes these samples as `BENCH_*` lines in `job.out`                       |
| `run_final.sh`              | Submits the final experiments into `results/<date>-<name>/`                    |
| `summarize.py`              | Writes `summary_jobs.csv` and `summary_cases.csv` for a results directory      |
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
| `format`  | squashfs, zarr_v2, zarr_v3, zarr_v3_lexi, zarr_v3_colexi at 2x2 and 8x2; squashfs_v3_lexi, squashfs_v3_colexi at 8x2 | 12   |
| `grain`   | zarr_v3 at 0x4 (threads only), 1x2, 4x2 (baseline), 8x1, 4x4          | 5    |
| `zarr`    | zarr_v3 4x2 with zarr pool/concurrency 4/10, 4/32, 8/10, 32/10, 32/32 | 5    |
| `cpus`    | zarr_v3 4x2 with 1 and 8 CPUs per task                                | 2    |
| `scaling` | zarr_v3 4x2 on 2 and 4 nodes                                          | 2    |
| `opt`     | zarr_v2 8x2 with 9 XLA/JAX setting sets on 1 node and 4 on 2 nodes    | 13   |
| `model`   | zarr_v2 8x2, coarse mesh, 8, 10, 12, 14, 16 message passing steps     | 5    |
| `lowmem`  | squashfs, zarr_v2, zarr_v3, zarr_v3_lexi 2x2; zarr_v3 1x2 and 0x4     | 6    |
| `workers` | zarr_v3, zarr_v3_lexi 2x2, 4x2, 8x2; zarr_v2, squashfs 4x2            | 8    |
| `memfit`  | zarr_v3_lexi 2x2, 8 uniform meshes (120 to 35 km) x 8 to 16 message passing steps | 40   |

`lowmem`, `workers` and `memfit` are meant for `all_levels`. Each case runs twice (`rep=1,2`, different seeds) for 60 training steps. Modifiers:
`quick` (one repetition of 30 steps), `long` (1024 steps, 3 hour time limit), `devel` (develbooster reservation), `dryrun` (check the
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
uv run jube run graphcast_training.xml --tag format          # 24 jobs
uv run jube run graphcast_training.xml --tag format grain zarr cpus scaling   # 52 jobs

uv run jube status bench_run --id last    # wait for jobs
uv run jube continue bench_run --id last  # pick up finished jobs
uv run jube result -a bench_run --id last # tables, also bench_run/<id>/result/result_csv.dat
```

Rough cost: a job spends 1.5 to 3 minutes in startup and compilation. Then a step takes about 0.45 s when
the loader keeps up (compute bound, 8 message passing steps; only zarr_v2 8x2 does at 10 levels) and up to about
6 s when it does not (e.g. zarr_v3 0x4; about 7 s with 1 CPU per task). With `long` (1024 steps) jobs take 10
minutes to 2 hours; all experiments of `run_final.sh` take about 90 node-hours. With all levels, cases with more
than 2 Grain workers run out of host memory within the first 25 steps and are best cancelled at once (a killed
worker otherwise leaves the job waiting until its time limit).

Each workpackage directory (`bench_run/<id>/<wp>_submit/work`) contains the job script, the
generated `bench.toml`, the SLURM output `job.out`/`job.err` and the per-task logs `task-<rank>.out/err`.
Checkpoints and the saved model are removed at the end of each job.

## XLA and JAX settings (`opt`)

The `opt` cases add an eighth field naming a set of settings (`opt_env` in `graphcast_training.xml`), exported
after `clusters/jupiter/.env`: `image` (the container's `XLA_FLAGS`, i.e. the latency hiding scheduler only, as
in all runs before this sweep), `no_lhs`, `env_valid` (the flags of `.env` that this XLA still knows), `pgle`,
`opt_o3`, `fit_o3`, `async_alloc`, `cache`, `all`, and on 2 nodes `no_ib` (`share_bin`, i.e.
`JAX_SHARE_BINARY_BETWEEN_HOSTS`, hung at the first compilation and is no longer swept). `default` uses the flags
of `.env`, which passes them as `APPTAINERENV_XLA_FLAGS` (same as `env_valid`). The container image
sets `XLA_FLAGS` itself, which takes precedence over the host's: XLA flags only reach the container as
`APPTAINERENV_XLA_FLAGS`. Training logs the `XLA_FLAGS` it sees (`Using XLA_FLAGS=...`). The loader is the fastest
one, so that compute is a larger part of a step; `compute_min_s` and `step_min_s` show the fastest step, less
affected by waits for data than the means.

## Model size (`model`, `fine`, `msg12`, `msg16`)

`model` runs the fastest loader with the coarse mesh (`uniform_coarse.msh`) and 8, 10, 12, 14 and 16 message
passing steps (`gnn_msg_steps`; 16 as the original GraphCast), substituted into `bench.toml`. The fine mesh
(`uniform_fine.msh`, tag `fine`) does not fit in GPU memory: XLA asks for single allocations of 279 to 504 GiB
per GPU. 16 message passing steps need `XLA_PYTHON_CLIENT_MEM_FRACTION=.95` (set in `.env`). The modifiers `fine`, `msg12` and `msg16` select one model size for the other sweeps, e.g.
`--tag opt fine msg16`.

`memfit` finds the largest model that fits in GPU memory: uniform meshes with target sizes 120 (coarse), 100, 85,
70, 60, 50, 40 and 35 km (fine), i.e. 34750 to 704078 nodes (`configs/mesh/uniform_*.toml`), each with 8, 10, 12, 14
and 16 message passing steps. Run it as `--tag memfit all_levels quick` (40 jobs of about 5 minutes; GPU OOMs fail
at the first step). At all 50 levels (results/2026-10-24-memfit) the largest cases that fit are the coarse mesh with
16 message passing steps, 100 km with 14 and 85 km with 10; from 70 km (95303 nodes) on not even 8 steps fit. The
limit is about 85 GiB per GPU (95 % of a GH200's 96 GB); `summarize.py` reports the allocation XLA asked for
(`gpu_alloc_GiB`, 85 to 508 GiB for the cases that did not fit).

Those runs offloaded nothing: the policy names mark the outputs of rematerialized blocks, which only an enclosing
checkpoint can offload. With `policy.remat_loss = true` (the default since 2026-10-24; tag `no_remat_loss` turns it
off) 28 instead of 11 cases fit: 100 km and 85 km with 16 message passing steps, 70 km (95303 nodes) with 10. Steps
take about 20 % longer (compute 0.55 to 0.85 s, a step 3.2 to 3.6 s) and node memory peaks at up to 858 GB (2 GB
left): with 2 Grain workers at all levels there is no host memory margin. The fine mesh with 14 and 16 steps ran out
of XLA's pinned host pool (64 GB per GPU, `XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB`; `host_oom` in `summarize.py`).
The tag `prevent_cse` (`policy.prevent_cse = true`) cuts the memory per message passing step from about 25 to about
7 GiB on the fine mesh (174 instead of 236 GiB at 8 steps) but leaves the step-independent part (about 120 GiB on the
fine mesh), so the limit does not move: cases near 85 GiB fit or fail either way. The tag `scan` does not compile
with offloading (XLA INTERNAL error on the host-placed residuals of the scan). **Without offloading
(`--tag memfit all_levels quick scan no_remat_loss`) every mesh up to 40 km (280245 nodes, 8x coarse) fits with 16
message passing steps**, at no cost in speed (compute 0.46 to 1.89 s, steps loader bound at 2.6 to 3.0 s) and with
the same losses as unrolled; only the fine mesh does not fit (115 to 162 GiB, about 5.8 GiB per step).

That scan used `hk.switch` over per-step networks, and XLA cannot compile host offloading inside a switch in a scan
(every JAX image from 26.06 to the 2026-10-24 nightly). `DeepTypedGraphNet` now runs the processor as an
`hk.layer_stack` over stacked per-step parameters (old checkpoints do not load). `run_memfit.sh` submits memfit for
unrolled, unrolled with `prevent_cse` and scan, each with and without offloading (300 jobs; meshes now also 35 and 30
km, 363900 and 491676 nodes). Results in results/2026-10-24-memfit-v2: scan with offloading fits the most (up to
35 km, 10.5x coarse, with 8 steps) but fills host memory with 2 Grain workers (0 GB left; 6 jobs stalled until their
time limit, `timeout` in `summarize.py`); **scan without offloading fits 50 km with 16 steps and 40 km with 8, with at
least 52 GB of host memory left**; unrolled at most 70 km with 10 steps. With less host memory for the loader
(small buffers, 1 or 0 Grain workers), offloading fits 40 km with 14 steps and 35 km with 10, at 3 to 10 s per step.

"At the same speed" held only while the loader limited the steps: the first layer stack (c04cae0) also wrapped the
unrolled networks and the 1-step encoder and decoder and made compute-bound steps about 35 % slower (median compute
0.61 instead of 0.45 s; results/2026-10-24-final). 2189def restored them; the scan itself costs 3 % (8 steps) to 9 %
(16 steps). The final run with 2189def is results/2026-10-24-final-v2 (`run_final.sh`, now with `-offload` variants
and `model-coarse-msg8to16-unrolled`; see its REPORT.md). Since 988d7d0 the processor is a pure step function with
per-step parameters created by `hk.lift`, the same with and without scan (checkpoints load with either setting); speed
and memory fit as with 2189def (results/2026-10-25-lift: scan without offloading fits 35 of 50 memfit cases).
Training logs the memory XLA plans for a step (`Memory usage: {... 'host_temp_size' ..., 'temp_size' ...}` in
task-0.err, e018542): with offloading the GPU needs 1 to 15 GiB less (mostly 5 to 9 GiB, about 10 %) for 8 to 87 GiB of
pinned host memory, with the same losses (results/2026-10-25-offload-check); cases above about 83.5 GiB fail.
Of further options (results/2026-10-25-memopt), `model.connected_grid_nodes` (grid nodes only for grid points that
an edge connects; identical losses) saves 4.7 GiB (median) and is on by default (tag `all_grid_nodes` turns it off);
`policy.edge_chunk_size` (edges in rematerialized chunks; tags `edge_chunks`, `edge_chunks_512k`, `edge_chunks_1m`)
saves 2 to 11 GiB (median; up to 20) for 7 to 35 % more compute; rematerializing each MLP saved nothing.

Results with the current defaults (2026-10-25): results/2026-10-25-final-v3 (`run_final.sh`; zarr_v2 8x2 at 0.48 s
per step, compute bound; the scan costs 2 to 6 %) and results/2026-10-25-memfit-v3 (`run_memfit.sh`, 7 memory
settings; with edge chunks of 262144, 40 km fits 16 message passing steps and 35 km 12 at all levels; the fine mesh
fits with none). The demo training on 9 meshes (`clusters/jupiter/slurm/mesh_study.sh`, runs/mesh_study/REPORT.md)
shows finer meshes lowering the loss of temperature, salinity and currents and raising that of sea surface height.

## What is measured

`graphcast train` logs one line per step:
`Step N timings: data D s, compute C s, checkpoint K s, total T s, elapsed E s`.
The analysis reads rank 0 (`task-0.err`) and reports, over steps 10 and later (after compilation and
prefetch warm-up):

- `data_avg_s`, `data_max_s`: time rank 0 waits for its training batch, including host-to-device transfer.
- `compute_avg_s`: the training step of rank 0. Steps synchronize all ranks (gradient all-reduce), so a data stall
  of another rank appears here as compute time. `summarize.py` therefore adds `data_stall_avg` (mean over steps of
  the longest data wait of any rank, from all `task-<rank>.err`) and `compute_median` (robust to such stalls);
  `compute_min_s` is the fastest step. With message passing working, the fastest steps on the coarse mesh compute
  in about 0.41 s (8 message passing steps) to 0.48 s (16); the median is about 0.44 s with 8.
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

- **Reading dominates the pipeline**: in Grain's execution summary the source (zarr reads and
  normalization) accounts for more than 99 % of the wait time. Loader throughput therefore scales with
  the number of samples read concurrently, `num_workers x num_threads`. The zarr thread pool hardly matters
  above about 4 threads per reader, and too small a pool hurts.
- **Demand is high**: each GPU task consumes one training sample per step (training computes no test loss). At
  about 0.45 s of compute per step, that is about 2.2 samples/s per task. At 10 levels only zarr_v2 with 8 Grain
  workers x 2 threads delivers it (0.47 s per step, 0.02 s waiting for data); zarr_v3 8x2 (0.82 s), SquashFS 8x2
  (1.0 s) and all smaller configurations leave the GPUs waiting for data. The small configurations (threads only, 1
  worker) show how strongly the step time depends on the loader. The benchmark also reports startup time and
  memory.
- **One stream per task**: training computes no test loss, so a task runs `num_workers` loader processes
  (runs before this change had a test stream too, with as many workers again). Booster nodes have about 857 GB
  of usable RAM (about 210 GB per task), shared with the activation offloading to pinned host memory. Most of it
  is used by the training processes themselves (about 480 GB per node without loader workers at 10 levels).
- **CPUs per task**: with JUPITER's `CR_CORE` and `task/cgroup` settings, a task gets one core unless
  `--cpus-per-task` is set, and `srun` needs `SRUN_CPUS_PER_TASK` to inherit it. The jobs here and
  `clusters/jupiter/slurm/train.slurm` use 72 cores per task; the `cpus` sweep measures 1 and 8.
- **Checkpointing every step** (the Orbax default `save_interval_steps = 1`) would dominate step times, so the
  benchmark config saves only step 0 (`full.toml` saves every 500 steps). The checkpoint time is still reported
  separately.
- **Caching**: each workpackage uses a different seed, so that runs do not read samples cached by earlier
  runs. The jobs read the standardized anomaly stores written by `graphcast download`, all in
  `$FSCRATCH/graphcast/data/anomalies` so that only the storage format differs: `zarr_v2`, the sharded zarr v3 stores
  (8 time x 50 depth) with the sub-chunks of each shard in morton (`zarr_v3`), lexicographic (`zarr_v3_lexi`) or
  colexicographic (`zarr_v3_colexi`) order, and SquashFS images of zarr_v2, zarr_v3_lexi and zarr_v3_colexi
  (`squashfs`, `squashfs_v3_lexi`, `squashfs_v3_colexi`). The mask (`graphcast download-mask`) and `tendency_std` are
  small and loaded once, the same for all formats; the zarr pool and concurrency settings apply only to them. Results
  recorded before 28 Oct 2026 are from the previous dataloader, which read the raw dataset, climatology and
  anomaly_std (in `/e/scratch/training2647`) with zarr-python and standardized each sample.
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
alike: both launchers accept either source type through `DATASET`, `CLIMATOLOGY`, `ANOMALY_STD`,
and `TENDENCY_STD`.
