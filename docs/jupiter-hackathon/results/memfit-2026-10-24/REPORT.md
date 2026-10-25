# GPU memory fit at all 50 levels with the stacked processor (memfit-v2)

Code 4219cb8: `DeepTypedGraphNet` runs the processor as an `hk.layer_stack` over stacked per-step parameters
(scanned with `scan`, fully unrolled otherwise; no `hk.switch`). 50 cases per experiment: 10 meshes x 8 to 16
message passing steps, all 50 levels, 30 steps, one repetition, zarr_v3_lexi 2x2 workers x threads, 1 node (4 GH200),
`XLA_PYTHON_CLIENT_MEM_FRACTION=.95`. 300 jobs while the whole sweep shared the file system (compute times of rank 0
include waits for other ranks' data and are not comparable to earlier runs; memory is).

Message passing steps that fit, per mesh and experiment:

| Mesh (nodes) | scan-nooffload | scan-offload | unrolled-preventcse-nooffload | unrolled-preventcse-offload | unrolled-offload | unrolled-nooffload |
|---|---|---|---|---|---|---|
| coarse (34750) | 8, 10, 12, 14, 16 | 8, 10, 12, 14, 16 | 8, 10, 12, 14, 16 | 8, 10, 12, 14, 16 | 8, 10, 12, 14, 16 | 8, 10, 12, 14, 16 |
| 100km (48662) | 8, 10, 12, 14, 16 | 8, 10, 12, 14, 16 | 8, 10, 12, 14, 16 | 8, 10, 12, 14, 16 | 8, 10, 12, 14, 16 | 8, 10, 12, 14, 16 |
| 85km (65943) | 8, 10, 12, 14, 16 | 8, 10, 12, 14, 16 | 8, 10, 12, 14, 16 | 8, 10, 12 | 8, 10, 12, 14, 16 | 8, 10, 12 |
| 70km (95303) | 8, 10, 12, 14, 16 | 8, 10, 12, 14, 16 | 8, 10 | 8 | 8, 10 | none |
| 60km (127948) | 8, 10, 12, 14, 16 | 8, 10, 12, 14, 16 | none | none | none | none |
| 50km (181980) | 8, 10, 12, 14, 16 | 10, 12, 14, 16 | none | none | none | none |
| 40km (280245) | 8 | 8, 10 | none | none | none | none |
| 35km (363900) | none | 8 | none | none | none | none |
| 30km (491676) | none | none | none | none | none | none |
| fine (704078) | none | none | none | none | none | none |

| Experiment | ok | GPU OOM | host OOM | timeout | step [s] (mean) | host memory left (min) |
|---|---|---|---|---|---|---|
| scan-nooffload | 31 | 19 | 0 | 0 | 2.52 to 2.88 (2.72) | 51.6 GB |
| scan-offload | 32 | 9 | 3 | 6 | 2.61 to 3.38 (2.97) | 0.0 GB |
| unrolled-preventcse-nooffload | 17 | 33 | 0 | 0 | 2.63 to 2.92 (2.76) | 66.7 GB |
| unrolled-preventcse-offload | 14 | 32 | 3 | 1 | 2.68 to 3.06 (2.91) | 14.5 GB |
| unrolled-offload | 17 | 27 | 3 | 3 | 2.57 to 3.25 (2.78) | 4.9 GB |
| unrolled-nooffload | 13 | 37 | 0 | 0 | 2.58 to 2.82 (2.72) | 46.5 GB |

## Findings

- **scan compiles with offloading now**: the restructure removed the XLA compile error (switch inside scan), and
  scan-offload fits the largest meshes: 50 km with 10 to 16 steps, 40 km with 8 and 10, 35 km (363900 nodes,
  10.5x coarse) with 8.
- **But offloading exhausts host memory**: with 2 Grain workers at all levels, offloaded activations filled the
  node (858 GB peak, 0 GB left); jobs then stalled until the 30 minute limit ("timeout": 50 km with 8 steps, 40 km
  with 12 to 16, 35 km with 10), or XLA's pinned host pool ran out (fine mesh, "host OOM", 64 GB per GPU).
- **scan without offloading is the robust choice**: 60 km (127948 nodes) with 16 steps, 50 km (181980 nodes) with
  16, 40 km with 8; at least 52 GB of host memory left; steps as fast as any (loader bound, 2.72 s).
- **Unrolled is far behind**: at most 70 km with 10 steps (prevent_cse or offloading), 85 km with 12 without either.
- **Near the limit XLA requests about 84 GiB**: e.g. scan-nooffload asked for 82 to 85 GiB at 40 km with 10 to 16
  steps and at 35 km, independent of the steps (XLA's rematerialization fills the memory limit). These cases might
  fit with a larger `XLA_PYTHON_CLIENT_MEM_FRACTION`; not tested.
- **Switch scan vs layer stack**: the switch-based scan (results/2026-10-24-memfit, scan-nooffload) fitted 40 km with
  up to 16 steps; the layer stack fits 8 (asks for about 84 GiB at 10 to 16). The fine mesh needs the same with both
  (114 vs 115 GiB at 8 steps), so the difference is small and decides only cases at the limit.
- 10 jobs (all with offloading) timed out; 22 jobs that hung after an OOM of one rank were cancelled by the monitor.

# Offloading with less host memory for the loader (`scan-offload-smallbuffers`, `-1worker`, `-0workers`)

Code 744a2f2 (same model, new JUBE tags). scan with offloading and small buffers (one sample per Grain worker and in
the reader prefetch), with 2 Grain workers x 2 threads, 1 x 2, or threads only (0 x 4). These ran with only about 6
jobs at a time, so their step times are not comparable to the 300 jobs above.

Message passing steps that fit:

| Mesh (nodes) | scan-nooffload | scan-offload (2x2) | small buffers (2x2) | 1 worker (1x2) | 0 workers (0x4) |
|---|---|---|---|---|---|
| 60 km (127948) and coarser | 8 to 16 | 8 to 16 | 8 to 16 | 8 to 16 | 8 to 16 |
| 50 km (181980) | 8 to 16 | 10 to 16 | 8 to 16 | 8 to 16 | 8 to 16 |
| 40 km (280245) | 8 | 8, 10 | 8 to 12 | 8 to 14 | 8 to 14 |
| 35 km (363900) | none | 8 | 8 | 8 | 8, 10 |
| 30 km and fine | none | none | none | none | none |

| Experiment | ok | GPU OOM | host OOM | timeout | failed | host memory peak | step [s] |
|---|---|---|---|---|---|---|---|
| scan-offload-smallbuffers | 34 | 11 | 3 | 2 | 0 | 798 to 858 GB | 1.66 to 2.33 |
| scan-offload-1worker | 35 | 11 | 3 | 0 | 1 | 694 to 858 GB | 2.95 to 7.84 |
| scan-offload-0workers | 36 | 11 | 3 | 0 | 0 | 651 to 858 GB | 6.39 to 9.72 |

- **Less loader memory lets offloading fit more**: with threads only, 40 km with 14 message passing steps and 35 km
  (363900 nodes, 10.5x coarse) with 10, and no job stalled; with 1 worker 40 km with 14 and 35 km with 8 (35 km with
  10 was killed by the kernel out of host memory, exit 137).
- **The GPU limit stays**: 40 km with 16 steps (82 GiB requested), 35 km with 12 or more and 30 km (80 GiB and up)
  do not fit in GPU memory with any loader; the fine mesh with 12 or more steps runs out of XLA's pinned host pool
  (64 GB per GPU), and of GPU memory with 8 and 10.
- **Host memory is still nearly exhausted** in the largest cases (858 GB peak, 0 to 3 GB left): offloading takes
  whatever the loader frees.
- **The price is the loader**: 1 worker 3 to 8 s per step, threads only 6.4 to 9.7 s, against 2.6 to 2.9 s with
  2 workers (all levels, loader bound). Offloading gains one mesh size step (40 to 35 km) or a few message passing
  steps over scan without offloading, at 2 to 3x the step time.

## Caveat: code of these runs

All memfit-v2 experiments ran before c988ad3, with every `DeepTypedGraphNet` in a layer stack (also the unrolled
processor and the 1-step encoder and decoder). That made steps about 35 % slower (results/2026-10-24-final-v2,
"The slow run"); c988ad3 restored the previous unrolled networks and keeps the layer stack for the scan. GPU memory
of the scan cases changes little, if at all: the processor scan is the same code; the encoder and decoder are
unrolled again.
