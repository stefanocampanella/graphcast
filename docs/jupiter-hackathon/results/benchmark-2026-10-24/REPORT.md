# GraphCast training benchmark on JUPITER: final results (2026-10-24, final-v2)

Code c988ad3: processor in a scan over stacked per-step parameters (`policy.scan = true`, the default of
`full.toml`), nothing offloaded (`policy.remat_loss = false`) unless stated, `XLA_PYTHON_CLIENT_MEM_FRACTION=.95`.
Coarse mesh, 8 message passing steps, 10 depth levels and zarr_v2 8x2 unless stated; 1024 steps per job, two
repetitions per case, step statistics over steps 10 to 1023. 172 jobs: 145 ok, 2 stopped (cancelled at 925 and 941
steps), 10 out of host memory, 14 failed and 1 timeout (all explained below).
Per-case numbers: `summary_cases.csv`; per job: `summary_jobs.csv` (both from `summarize.py`).
Previous final run: results/2026-10-22-final-v2 (code 0bfe880: processor unrolled; offloading configured but, as
found since, never effective). A run with the first stacked processor, which made steps about 35 % slower, is kept
in results/2026-10-24-final (not comparable, see "The slow run").

Columns: step = mean step time; stall = mean over steps of the longest data wait of any rank; compute = median
compute time of a step (rank 0); samples/s = training samples per second over all GPUs; node GB = peak node memory
used (of 857 GB); GPU % = mean utilization.

## Summary

- **All 172 jobs ran at the same time** (started between 22:03 and 22:10): every job ran in the slow regime of the
  previous run's caveat. The compute-bound zarr_v2 8x2 took 0.75 s per step (0.29 s stall) instead of 0.47 s on a
  quieter file system; compute times and rankings within this run are comparable, absolute step times are not. The
  demo training alone on 8 nodes computes a step in 0.474 s (median) with 0.014 s data wait.
- **The model computes a step in 0.455 s** (median, 8 message passing steps, scan) and 0.560 s with 16; unrolled
  0.443 and 0.512 s, as in the previous run (0.438 and 0.503 s). **The scan costs 3 % at 8 steps and 9 % at 16**,
  and fits much larger meshes in GPU memory (results/2026-10-24-memfit-v2: 50 km, 5.2x the coarse nodes, with
  16 steps at all 50 levels; unrolled at most 70 km with 10).
- **Data loading**: the same ranking as before; zarr_v2 8x2 is the only setting close to compute bound.
- **XLA/JAX settings make no difference** (compute 0.447 to 0.468 s at 8 steps); PGLE and `all` add 25 to 35 s of
  startup; disabling InfiniBand costs 6 % (8 steps) to 10 % (16 steps) of compute on 2 nodes.
- **All levels**: at most 2 Grain workers fit in host memory; with offloading (`offload`), 2 workers still fit but
  leave only 1 to 23 GB of host memory (19 to 96 GB without), at the same speed, and 4 or 8 workers stall at the
  second step.
- **zarr_v3_lexi** is 12 to 14 % slower than zarr_v3 at 10 levels and 21 to 25 % faster at all 50 levels.
- **Scaling** (zarr_v3 4x2): 1.90, 3.82 and 7.52 samples/s on 1, 2 and 4 nodes (2.01x, 3.96x).

## Data loading at 10 levels (`io-10levels`)

| Format, workers x threads | step [s] | stall [s] | compute [s] | samples/s | node GB | GPU % | step 22 Oct [s] |
|---|---|---|---|---|---|---|---|
| **zarr_v2 8x2** | **0.75** | **0.29** | 0.459 | **5.31** | 815 | **88** | 0.47 |
| zarr_v3 8x2 | 1.28 | 0.80 | 0.470 | 3.12 | 858 | 79 | 0.83 |
| zarr_v3_lexi 8x2 | 1.43 | 0.95 | 0.467 | 2.81 | 858 | 72 | 0.92 |
| zarr_v3 4x4 | 1.78 | 1.31 | 0.465 | 2.25 | 739 | 73 | 1.49 |
| zarr_v3 8x1 | 1.81 | 1.35 | 0.461 | 2.21 | 792 | 69 | 0.94 |
| squashfs 8x2 | 1.96 | 1.50 | 0.458 | 2.04 | 835 | 77 | 1.03 |
| zarr_v3 4x2 (baseline) | 2.10 | 1.64 | 0.461 | 1.90 | 699 | 66 | 1.78 |
| zarr_v2 2x2 | 2.12 | 1.67 | 0.461 | 1.89 | 606 | 75 | 1.03 |
| squashfs 2x2 | 3.18 | 2.75 | 0.438 | 1.26 | 665 | 75 | 2.81 |
| zarr_v3 2x2 | 3.81 | 3.35 | 0.461 | 1.05 | 625 | 57 | 3.44 |
| zarr_v3_lexi 2x2 | 4.36 | 3.90 | 0.464 | 0.92 | 623 | 51 | 3.91 |
| zarr_v3 0x4 (threads only) | 6.22 | 5.77 | 0.478 | 0.64 | 589 | 20 | 5.91 |
| zarr_v3 1x2 | 6.80 | 6.35 | 2.994 | 0.59 | 598 | 49 | 6.24 |

- **Format and workers**: as on 22 Oct; zarr_v2 per worker about 3x faster than zarr_v3. The fast loaders lost
  most under the concurrent load (zarr_v2 8x2 0.47 to 0.75 s; zarr_v2 2x2 1.03 to 2.12 s); the slowest are within
  10 % of the previous run.
- **Memory**: zarr_v3 8x2 and zarr_v3_lexi 8x2 again reached 858 GB, the usable limit (one of them ran out in the
  slow run).
- **zarr thread pool and concurrency** (zarr_v3 4x2): pool 32 gives 2.03 to 2.04 s, pool 8 2.10 to 2.11 s, pool 4
  2.36 to 2.37 s; concurrency makes no difference.
- **CPUs per task** (zarr_v3 4x2): 72 CPUs 2.10 s, 8 CPUs 2.13 s, 1 CPU 7.21 s.
- **Nodes** (zarr_v3 4x2): 1.90, 3.82 and 7.52 samples/s on 1, 2 and 4 nodes (step 2.09 to 2.13 s).

## All 50 levels

| Format, workers x threads | small buffers | small buffers, offload | buffers of 2 (`workers`) | buffers of 2, offload |
|---|---|---|---|---|
| zarr_v3_lexi 2x2 | 3.64 s (96 GB left) | 3.80 s (15 GB) | **3.49 s** (90 GB) | 3.48 s (4 GB) |
| zarr_v2 2x2 | 3.75 s (75 GB) | 3.66 s (12 GB) | | |
| zarr_v3 2x2 | 4.64 s (66 GB) | 4.55 s (20 GB) | 4.51 s (19 GB) | 4.51 s (1 GB) |
| squashfs 2x2 | 6.14 s (61 GB) | 6.02 s (23 GB) | | |
| zarr_v3 1x2 / 0x4 | 8.87 / 10.44 s | 8.69 / 10.33 s | | |
| 4 or 8 workers (all formats) | | | out of host memory | stalled at step 2 |

GB: lowest available host memory of a node. Training at all levels is loader bound (stall 3.0 of 3.5 s). Without
offloading, every case with 4 or 8 workers ran out of host memory within the first steps (zarr_v3 8x2 and zarr_v3_lexi 8x2:
one repetition each failed with a killed Grain worker, status 134). With offloading they stalled at step 2 and were
cancelled after 20 minutes without progress (12 jobs, "failed" status 137). Offloading changes no speed with 2
workers but takes 18 to 86 GB of host memory: with 2 workers and buffers of 2, 1 to 4 GB remain.

## zarr_v3_lexi versus zarr_v3

| | zarr_v3 | zarr_v3_lexi | lexi versus v3 |
|---|---|---|---|
| 10 levels, 2x2 | 3.81 s | 4.36 s | 14 % slower |
| 10 levels, 8x2 | 1.28 s | 1.43 s | 12 % slower |
| all levels, 2x2 (4 experiments) | 4.51 to 4.64 s | 3.48 to 3.80 s | 21 to 25 % faster |

The same direction as on 22 Oct (11 % at all levels then), with a larger gain at all levels under this load.

## XLA/JAX settings (`xla-coarse-msg8`, `xla-coarse-msg16`)

| Setting | 8 steps, 1 node | 16 steps, 1 node | startup [s] (8 steps) |
|---|---|---|---|
| image, no_lhs, env_valid, opt_o3, fit_o3, async_alloc, cache | 0.452 to 0.465 s | 0.549 to 0.558 s | 87 to 97 |
| pgle | 0.460 s | 0.555 s | 119 |
| all | 0.454 s | 0.555 s | 123 |

On 2 nodes: `image`, `env_valid` and `all` 0.459 to 0.473 s (8 steps) and 0.560 to 0.567 s (16 steps); `no_ib`
(InfiniBand off) 0.488 and 0.622 s, 6 and 10 % slower. None of the settings is worth changing. One repetition of
`image` on 2 nodes failed to start (the coordination service could not bind its port: "Address already in use").

## Model size and scan (`model-coarse-msg8to16`, `model-coarse-msg8to16-unrolled`)

| Message passing steps | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|
| scan (default), median compute [s] | 0.455 | 0.484 | 0.493 | 0.527 | 0.560 |
| unrolled, median compute [s] | 0.443 | 0.467 | 0.485 | 0.486 | 0.512 |
| unrolled on 22 Oct [s] | 0.438 | 0.468 | 0.488 | 0.507 | 0.503 |

The scan costs 3 to 9 %; 16 message passing steps (the original GraphCast) cost 23 % more than 8 with the scan, 16 %
unrolled. In the processor alone (8 steps, coarse size), a scan step takes 87 ms against 62 ms unrolled: XLA
optimizes across unrolled steps but needs 8.5 instead of 3.6 GiB of temporary GPU memory.

## The slow run (results/2026-10-24-final)

The first stacked processor (621aa19) ran every `DeepTypedGraphNet` through a layer stack, including the unrolled
processor and the encoder and decoder with one step each. Median compute rose to 0.61 s at 8 steps (0.74 s
unrolled); the demo training then took 0.629 instead of 0.464 s. c988ad3 restored the unrolled networks (unrolled is
the previous code again) and keeps the layer stack for the scan only. That run's loader, memory and failure results
agree with this one.

## Caveats

- **Concurrent load**: all jobs ran together (see Summary). Compare only within this run.
- **Lockstep repetitions**: the same case in two all-levels experiments started together with the same seed and
  read the same samples at the same time; their timings are not independent.
- **Stopped jobs**: zarr_v3 0x4 at all levels, one repetition each with and without offloading, was cancelled at
  925 and 941 of 1024 steps (statistics over the steps done).
- **Rank 0 attribution**: the JUBE columns read rank 0 only; `data_stall_avg` and `compute_median` use all ranks.
  For the most starved cases (1x2) even the median compute includes waits.
- **GPU memory** in the tables is JAX's preallocation (95 %), not use.
- One log (xla-coarse-msg8, workpackage 18, the startup failure) contained binary bytes JUBE could not read; the
  original is `task-0.err.orig`, `task-0.err` is a copy with them removed.

## Recommended settings (the defaults)

zarr_v2 on `/e/scratch/training2647/graphcast_data/zarr_v2`, 8 Grain workers x 2 threads, buffers of 2, 72 CPUs per
task, coarse mesh, 8 message passing steps (16 for the original GraphCast, +23 % compute), processor in a scan
(`policy.scan = true`; `false` is 3 to 9 % faster where it fits), nothing offloaded, checkpoints every 500 steps, XLA
flags as in `.env`, InfiniBand on. For all levels: at most 2 workers with small buffers, zarr_v3_lexi; for meshes
finer than coarse, see results/2026-10-24-memfit-v2.
