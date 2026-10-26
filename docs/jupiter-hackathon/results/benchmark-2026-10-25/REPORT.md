# GraphCast training benchmark on JUPITER: final results (2026-10-25, final-v3)

Code 451fa9e, the defaults of `full.toml`: processor in a scan over per-step parameters (`policy.scan = true`),
grid nodes only for connected grid points (`model.connected_grid_nodes = true`), nothing offloaded
(`policy.remat_loss = false`), `XLA_PYTHON_CLIENT_MEM_FRACTION=.95`. Coarse mesh, 8 message passing steps, 10 depth
levels and zarr_v2 8x2 unless stated; 1024 steps per job, two repetitions per case, step statistics over steps 10 to
1023. 172 jobs: 148 ok, 20 out of host memory, 4 failed (all 4- and 8-worker cases at all levels, see below).
Per-case numbers: `summary_cases.csv`; per job: `summary_jobs.csv` (both from `summarize.py`).
Previous final runs: results/2026-10-24-final-v2 (code 2189def, all jobs under file system load) and
results/2026-10-22-final-v2 (code 3b06cdc, processor unrolled).

Columns: step = mean step time; stall = mean over steps of the longest data wait of any rank; compute = median
compute time of a step (rank 0); samples/s = training samples per second over all GPUs; node GB = peak node memory
used (of 857 GB); GPU % = mean utilization.

## Summary

- **Training is compute bound with zarr_v2 and 8 Grain workers x 2 threads**: 0.48 s per step, 0.02 s data stall,
  GPUs 94 % busy, 8.4 samples/s per node, as on 22 Oct (0.47 s). All 172 jobs started within 12 minutes, yet the
  file system kept up this time: the slow steps of 24 Oct (0.75 s) came from load beyond our own jobs.
- **The model computes a step in 0.436 s** (median, 8 message passing steps, scan) and 0.537 s with 16 (+23 %);
  unrolled 0.426 and 0.505 s: **the scan costs 2 to 6 %**, and fits meshes up to 50 km at all levels
  (results/2026-10-25-memfit-v3).
- **XLA/JAX settings make no difference** (compute 0.438 to 0.448 s at 8 steps); PGLE and `all` add 35 to 45 s of
  startup; disabling InfiniBand costs 7 to 8 % of compute on 2 nodes.
- **Scaling is nearly linear**: 2.63, 5.20 and 10.25 samples/s on 1, 2 and 4 nodes (1.98x, 3.90x) with zarr_v3 4x2.
- **All levels**: at most 2 Grain workers fit in host memory; zarr_v3_lexi 2x2 is fastest (2.39 to 2.44 s per step).
  Offloading leaves the speed unchanged and takes 15 to 57 GB of host memory; with 4 or 8 workers it runs out of host
  memory as without it.
- **zarr_v3_lexi** is 11 to 13 % slower than zarr_v3 at 10 levels and 15 to 21 % faster at all levels.

## Data loading at 10 levels (`io-10levels`)

| Format, workers x threads | step [s] | stall [s] | compute [s] | samples/s | node GB | GPU % | 24 Oct [s] |
|---|---|---|---|---|---|---|---|
| **zarr_v2 8x2** | **0.48** | **0.02** | 0.444 | **8.41** | 825 | **94** | 0.75 |
| zarr_v3 8x2 | 0.91 | 0.45 | 0.443 | 4.42 | 858 | 80 | 1.28 |
| zarr_v3_lexi 8x2 | 1.01 | 0.55 | 0.451 | 3.96 | 858 | 78 | 1.43 |
| zarr_v3 8x1 | 1.12 | 0.67 | 0.444 | 3.58 | 826 | 74 | 1.81 |
| zarr_v2 2x2 | 1.17 | 0.73 | 0.442 | 3.42 | 593 | 78 | 2.12 |
| squashfs 8x2 | 1.18 | 0.73 | 0.440 | 3.40 | 837 | 76 | 1.96 |
| zarr_v3 4x4 | 1.37 | 0.92 | 0.447 | 2.92 | 801 | 67 | 1.78 |
| zarr_v3 4x2 (baseline) | 1.52 | 1.08 | 0.437 | 2.63 | 710 | 70 | 2.10 |
| squashfs 2x2 | 2.03 | 1.60 | 0.437 | 1.97 | 642 | 74 | 3.18 |
| zarr_v3 2x2 | 2.90 | 2.47 | 0.430 | 1.38 | 655 | 65 | 3.81 |
| zarr_v3_lexi 2x2 | 3.28 | 2.85 | 0.435 | 1.22 | 630 | 52 | 4.36 |
| zarr_v3 0x4 (threads only) | 5.24 | 4.80 | 0.614 | 0.76 | 572 | 19 | 6.22 |
| zarr_v3 1x2 | 5.60 | 5.17 | 1.700 | 0.71 | 584 | 47 | 6.80 |

- **Format**: per worker zarr_v2 is about 3x faster than zarr_v3; zarr_v2 with 2 workers matches SquashFS with 8.
- **Memory**: zarr_v3 8x2 and zarr_v3_lexi 8x2 reach the usable 858 GB again: too close to the limit.
- **zarr thread pool and concurrency** (zarr_v3 4x2): pool 32 gives 1.49 s, pool 8 1.52 to 1.53 s, pool 4 1.66 to
  1.72 s; concurrency makes no difference.
- **CPUs per task** (zarr_v3 4x2): 72 CPUs 1.52 s, 8 CPUs 1.56 s, 1 CPU 6.71 s.
- **Nodes** (zarr_v3 4x2): 2.63, 5.20 and 10.25 samples/s on 1, 2 and 4 nodes (step 1.52 to 1.56 s).

## All 50 levels

| Format, workers x threads | small buffers | small buffers, offload | buffers of 2 (`workers`) | buffers of 2, offload |
|---|---|---|---|---|
| zarr_v3_lexi 2x2 | 2.43 s (109 GB left) | 2.43 s (52 GB) | **2.44 s** (69 GB) | **2.39 s** (47 GB) |
| zarr_v2 2x2 | 2.54 s (132 GB) | 2.46 s (79 GB) | | |
| zarr_v3 2x2 | 3.02 s (87 GB) | 2.94 s (72 GB) | 2.84 s (60 GB) | 3.02 s (17 GB) |
| squashfs 2x2 | 4.37 s (73 GB) | 4.40 s (38 GB) | | |
| zarr_v3 1x2 / 0x4 | 5.95 / 6.38 s | 5.91 / 6.38 s | | |
| 4 or 8 workers (all formats) | | | out of host memory | out of host memory |

GB: lowest available host memory of a node. Training at all levels is loader bound (stall 2.0 of 2.4 s). Every case
with 4 or 8 workers ran out of host memory within the first steps, with and without offloading (20 jobs "oom"; 4
offloading jobs with 8 workers killed or stalled, "failed"). Offloading changes no speed but takes 15 to 57 GB of
host memory.

## zarr_v3_lexi versus zarr_v3

| | zarr_v3 | zarr_v3_lexi | lexi versus v3 |
|---|---|---|---|
| 10 levels, 2x2 | 2.90 s | 3.28 s | 13 % slower |
| 10 levels, 8x2 | 0.91 s | 1.01 s | 11 % slower |
| all levels, 2x2 (4 experiments) | 2.84 to 3.02 s | 2.39 to 2.44 s | 15 to 21 % faster |

The same direction in all three final runs (22 Oct: 11 % faster at all levels; 24 Oct: 21 to 25 %).

## XLA/JAX settings (`xla-coarse-msg8`, `xla-coarse-msg16`)

| Setting | 8 steps, 1 node | 16 steps, 1 node | startup [s] (8 steps) |
|---|---|---|---|
| image, no_lhs, env_valid, opt_o3, fit_o3, async_alloc, cache | 0.438 to 0.448 s | 0.528 to 0.541 s | 72 to 86 |
| pgle | 0.447 s | 0.541 s | 120 |
| all | 0.441 s | 0.539 s | 120 |

On 2 nodes: `image`, `env_valid` and `all` 0.448 to 0.451 s (8 steps) and 0.534 to 0.544 s (16 steps); `no_ib`
(InfiniBand off) 0.483 and 0.584 s, 7 and 8 % slower. Steps 0.47 to 0.51 s (8 steps). None of the settings is worth
changing; the compilation cache gave the shortest startup (72 and 79 s).

## Model size and scan (`model-coarse-msg8to16`, `model-coarse-msg8to16-unrolled`)

| Message passing steps | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|
| scan (default), median compute [s] | 0.436 | 0.470 | 0.488 | 0.502 | 0.537 |
| unrolled, median compute [s] | 0.426 | 0.419 | 0.474 | 0.478 | 0.505 |
| scan, 24 Oct (2189def) [s] | 0.455 | 0.484 | 0.493 | 0.527 | 0.560 |

The scan costs 2 to 6 % (the unrolled 10-step value is low against 8 steps; one case, two repetitions); 16 message
passing steps cost 23 % more than 8 with the scan, 19 % unrolled.

## Caveats

- **Lockstep repetitions**: the same case in two all-levels experiments started together with the same seed and
  read the same samples at the same time; their timings are not independent.
- **Rank 0 attribution**: the JUBE columns read rank 0 only; `data_stall_avg` and `compute_median` use all ranks.
  For the most starved cases (0x4, 1x2) even the median compute includes waits.
- **GPU memory** in the tables is JAX's preallocation (95 %), not use; the memory XLA plans for a step is logged
  ("Memory usage" in task-0.err) and analysed in results/2026-10-25-memfit-v3.
- One small-buffers job with offloading finished its 1024 steps and then hung at exit; the watcher cancelled it
  (counted as ok).

## Recommended settings (the defaults)

zarr_v2 on `/e/scratch/training2647/graphcast_data/zarr_v2`, 8 Grain workers x 2 threads, buffers of 2, 72 CPUs per
task, coarse mesh, 8 message passing steps (16 for the original GraphCast, +23 % compute), processor in a scan
(`false`: 2 to 6 % faster where it fits), grid nodes only for connected grid points, nothing offloaded, checkpoints
every 500 steps, XLA flags as in `.env`, InfiniBand on. For all levels: at most 2 workers, zarr_v3_lexi. For meshes
finer than coarse: results/2026-10-25-memfit-v3.
