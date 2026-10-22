# GraphCast training benchmark on JUPITER: final results (2026-10-22, final-v2)

Code 0bfe880 (message passing fixed in f43631e, `XLA_PYTHON_CLIENT_MEM_FRACTION=.95`). Coarse mesh, 8 message
passing steps, 10 depth levels and zarr_v2 8x2 unless stated; 1024 steps per job, two repetitions per case, step
statistics over steps 10 to 1023. 162 jobs: 130 ok, 8 stopped (cancelled at 912 to 960 steps, statistics over the
steps done), 24 out of host memory (all expected, see below). 111.6 node-hours.
Per-case numbers: `summary_cases.csv`; per job: `summary_jobs.csv` (both from `summarize.py`).

Columns: step = mean step time; stall = mean over steps of the longest data wait of any rank (the time the GPUs
wait for data); compute = median compute time of a step; samples/s = training samples per second over all GPUs;
node GB = peak node memory used (of 857 GB); GPU % = mean utilization.

## Summary

- **Training is compute bound only with zarr_v2 and 8 Grain workers x 2 threads**: 0.47 s per step, 0.02 s data
  stall, GPUs 94 % busy at 415 W, 8.5 samples/s per node. Every other loader setting makes the GPUs wait for data.
  These settings are now the defaults (`.env`, 1abee0e).
- **The model computes a step in about 0.44 s** (median) with 8 message passing steps; 16 steps cost only about
  15 % more (0.50 s). The fine mesh does not fit in GPU memory.
- **XLA/JAX settings make no difference** to steady-state steps; PGLE adds 30 s of startup; disabling InfiniBand
  costs 5 to 10 % of compute on 2 nodes; `JAX_SHARE_BINARY_BETWEEN_HOSTS` hangs.
- **zarr_v3_lexi** (lexicographic sub-chunk order) is about 11 % slower than zarr_v3 at 10 levels, but about 11 %
  faster at all 50 levels.
- **All levels**: at most 2 Grain workers fit in host memory (every 4- and 8-worker case ran out of memory), so
  training at all levels is loader bound (3.2 s per step at best). Activation offloading makes no difference.
- **Scaling is nearly linear** to 4 nodes (1.98x on 2, 3.85x on 4 nodes).
- **Concurrent jobs slow each other down**: jobs that ran while the I/O experiments loaded `/e/scratch` took 0.88
  instead of 0.47 s per step (see "Caveats").

## Data loading at 10 levels (`io-10levels`)

| Format, workers x threads | step [s] | stall [s] | compute [s] | samples/s | node GB | GPU % |
|---|---|---|---|---|---|---|
| **zarr_v2 8x2** | **0.47** | **0.02** | 0.448 | **8.49** | 804 | **94** |
| zarr_v3 8x2 | 0.83 | 0.36 | 0.457 | 4.81 | 858 | 81 |
| zarr_v3_lexi 8x2 | 0.92 | 0.45 | 0.458 | 4.34 | 858 | 78 |
| zarr_v3 8x1 | 0.94 | 0.48 | 0.453 | 4.27 | 800 | 74 |
| squashfs 8x2 | 1.03 | 0.57 | 0.448 | 3.88 | 840 | 77 |
| zarr_v2 2x2 | 1.03 | 0.59 | 0.438 | 3.88 | 577 | 75 |
| zarr_v3 4x4 | 1.49 | 1.04 | 0.447 | 2.68 | 765 | 68 |
| zarr_v3 4x2 (baseline) | 1.78 | 1.34 | 0.442 | 2.25 | 709 | 66 |
| squashfs 2x2 | 2.81 | 2.37 | 0.442 | 1.42 | 644 | 73 |
| zarr_v3 2x2 | 3.44 | 2.99 | 0.445 | 1.16 | 612 | 63 |
| zarr_v3_lexi 2x2 | 3.91 | 3.47 | 0.442 | 1.02 | 619 | 55 |
| zarr_v3 0x4 (threads only) | 5.91 | 5.48 | 0.519 | 0.68 | 573 | 22 |
| zarr_v3 1x2 | 6.24 | 5.80 | 1.349 | 0.64 | 594 | 53 |

- **Format**: per worker, zarr_v2 is about 3x faster than zarr_v3 and 2.7x faster than SquashFS. zarr_v2 with
  2 workers is as fast as SquashFS with 8.
- **Workers before threads**: step time falls roughly in proportion to the number of workers; doubling threads
  per worker (4x2 to 4x4) gains 16 %, and 8x1 nearly matches 8x2.
- **Memory**: 8 workers per task need 800 to 860 GB per node; zarr_v3 8x2 and zarr_v3_lexi 8x2 reached the usable
  857 GB and are too close to the limit for long training.
- **zarr thread pool and concurrency** (zarr_v3 4x2): pool 32 gives 1.67 to 1.74 s, pool 8 1.76 to 1.78 s, pool 4
  1.91 to 2.00 s; concurrency 10 or 32 makes no consistent difference.
- **CPUs per task** (zarr_v3 4x2): 72 CPUs 1.78 s, 8 CPUs 1.91 s, 1 CPU 7.05 s and 211 s startup (job stopped).
  `train.slurm` now requests 72 CPUs per task.
- **Nodes** (zarr_v3 4x2): 2.25, 4.46 and 8.67 samples/s on 1, 2 and 4 nodes (step 1.78 to 1.84 s).

## All 50 levels

Samples are about 5x larger. With 2 samples buffered per worker, **every case with 4 or 8 workers ran out of host
memory** within 25 steps, with and without activation offloading and for all formats (24 jobs). With one sample
per worker and reader prefetch (`small_buffers`), 2 workers fit (750 to 810 GB per node):

| Format, 2x2 workers x threads | small buffers | small buffers, no offload | buffers of 2 (`workers`) |
|---|---|---|---|
| zarr_v3_lexi | **3.17 s** | **3.17 s** | 3.19 s |
| zarr_v2 | 3.25 s | 3.27 s | out of memory at 4 workers |
| zarr_v3 | 3.56 s | 3.56 s | 3.54 s (3.71 s without offload) |
| squashfs | 4.71 s | 4.71 s | out of memory at 4 workers |
| zarr_v3 1x2 / 0x4 | 6.65 / 7.38 s | 6.65 / 7.39 s | |

Training at all levels is loader bound (stall 2.7 s of 3.2 s). Activation offloading makes no difference to speed
or to whether a case fits. Buffer size makes no difference at 2 workers.

## zarr_v3_lexi versus zarr_v3

Same data, shards (8 time x 50 depth) and chunks; only the order of the sub-chunks in each shard differs.

| | zarr_v3 | zarr_v3_lexi | lexi versus v3 |
|---|---|---|---|
| 10 levels, 2x2 | 3.44 s | 3.91 s | 14 % slower |
| 10 levels, 8x2 | 0.83 s | 0.92 s | 11 % slower |
| all levels, 2x2 (4 experiments) | 3.54 to 3.71 s | 3.17 to 3.19 s | 11 % faster |

With 10 of 50 levels, a sample reads scattered sub-chunks of each shard; with all levels it reads all of them,
which the lexicographic order apparently lays out more favourably. The result is consistent across all four
all-levels experiments; it agrees with the sub-chunk order mattering, not with run-to-run noise.

## XLA/JAX settings (`xla-coarse-msg8`, `xla-coarse-msg16`)

Compared within jobs that ran at the same time (see "Caveats"):

- 8 message passing steps, 1 node, jobs started together: `image`, `no_lhs`, `env_valid`, `opt_o3`, `fit_o3`,
  `async_alloc`, `pgle` all 0.47 to 0.48 s per step, compute 0.448 to 0.454 s.
- 16 message passing steps, 1 node: all nine settings 0.86 to 0.89 s per step, compute 0.500 to 0.526 s (the
  differences are within the spread of repetitions).
- Startup: PGLE (and `all`, which includes it) adds about 30 s (114 versus 84 s); the compilation cache wrote
  entries but the repetitions ran concurrently, so reuse was not measured.
- 2 nodes: disabling InfiniBand (`no_ib`) raises the median compute from 0.45 to 0.475 s (8 steps) and from 0.52
  to 0.57 s (16 steps), i.e. 5 to 10 %. Keep InfiniBand on.

None of the settings is worth changing; the `.env` keeps the three XLA flags this XLA knows, as they cost nothing.

## Model size (`model-coarse-msg8to16`)

| Message passing steps | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|
| median compute [s] | 0.438 | 0.468 | 0.488 | 0.507 | 0.503 |

The processor on the coarse mesh is a small part of a step: 16 steps (the original GraphCast) cost about 15 %
more than 8. All these jobs ran in the slow regime (step 0.88 s, see below), so only the compute times compare.

## Caveats

- **Concurrent load**: 16 jobs that started at 09:51 ran at 0.47 s per step; all 46 xla and model jobs that
  started from 10:13, while about 70 I/O jobs read from `/e/scratch`, took 0.86 to 0.91 s with the same median
  compute. Their slowdown is data stalls of seconds on single ranks, which the other ranks wait for in the gradient
  all-reduce. Compare only cases that ran at the same time; the xla tables above do. Results of single benchmark
  jobs on a quiet file system will be closer to the fast regime.
- **Lockstep repetitions**: the same case in two all-levels experiments started together with the same seed and
  read the same samples at the same time; their timings are not independent.
- **Stopped jobs**: 8 of the slowest jobs were cancelled after 912 to 960 of 1024 steps: both repetitions of
  zarr_v3 4x2 with 1 CPU per task, and at all levels zarr_v3 0x4 (both repetitions, with and without offloading)
  and zarr_v3 1x2 (one repetition each). Their statistics cover the steps done.
- **Rank 0 attribution**: the JUBE columns read rank 0 only; `data_stall_avg` and `compute_median` in the summary
  files use all ranks. For the most starved cases (0x4, 1x2) even the median compute includes waits.
- **GPU memory** in the tables is JAX's preallocation (95 %), not use.

## Recommended settings (applied, 1abee0e)

zarr_v2 on `/e/scratch/training2647/graphcast_data/zarr_v2`, 8 Grain workers x 2 threads, buffers of 2,
72 CPUs per task, coarse mesh, 8 message passing steps (16 for the original GraphCast at +15 % compute),
checkpoints every 500 steps, XLA flags as in `.env`, InfiniBand on. For all levels: at most 2 workers with small
buffers, zarr_v3_lexi.
