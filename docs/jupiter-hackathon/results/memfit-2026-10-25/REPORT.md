# GPU memory fit at all 50 levels, current code (memfit-v3, 2026-10-25)

Code 53ddb9a. memfit: 10 meshes x 8 to 16 message passing steps, all 50 levels, 30 steps, one repetition. Defaults:
processor in a scan, grid nodes only for connected grid points, nothing offloaded, no edge chunks; zarr_v3_lexi 2x2
(the offloading experiments: Grain threads only with small buffers). Every case logs the GPU memory XLA plans for a
step, also when it does not fit.

Planned GPU memory [GiB] with 8 / 12 / 16 message passing steps (\* = trained; h = out of XLA's pinned host pool):

| Mesh (nodes) | unrolled | scan | scan, all grid nodes | chunks 1M | chunks 256k | offload | offload + chunks 256k |
|---|---|---|---|---|---|---|---|
| coarse (34 750) | 59\*/64\*/66\* | 49\*/50\*/47\* | 56\*/54\*/52\* | 46\*/47\*/48\* | 35\*/36\*/37\* | 43\*/44\*/49\* | 33\*/35\*/35\* |
| 100km (48 662) | 65\*/72\*/78\* | 49\*/52\*/49\* | 55\*/57\*/55\* | 46\*/48\*/49\* | 36\*/37\*/39\* | 47\*/45\*/45\* | 35\*/36\*/37\* |
| 85km (65 943) | 72\*/82/87 | 52\*/53\*/52\* | 57\*/55\*/54\* | 48\*/50\*/50\* | 38\*/39\*/41\* | 46\*/46\*/46\* | 36\*/38\*/39\* |
| 70km (95 303) | 81\*/89/91 | 54\*/54\*/54\* | 62\*/65\*/60\* | 50\*/53\*/55\* | 40\*/43\*/45\* | 47\*/47\*/50\* | 39\*/41\*/44\* |
| 60km (127 948) | 88/95/101 | 60\*/58\*/61\* | 63\*/64\*/67\* | 54\*/57\*/60\* | 43\*/47\*/50\* | 50\*/53\*/56\* | 41\*/45\*/48\* |
| 50km (181 980) | 91/100/113 | 62\*/67\*/72\* | 67\*/72\*/77\* | 61\*/66\*/71\* | 49\*/54\*/60\* | 57\*/62\*/66\* | 47\*/52\*/57\* |
| 40km (280 245) | 116/130/160 | 77\*/84/85 | 82\*/83\*/83\* | 70\*/76/83 | 60\*/67\*/75\* | 72\*/72\*/81\* | 58\*/63\*/69\* |
| 35km (363 900) | 126/162/216 | 85/82\*/84 | 86/83\*/90 | 81\*/82\*/83 | 69\*/79\*/83 | 79\*/87/98 | 67\*/74\*/86 |
| 30km (491 676) | 166/223/351 | 84\*/99/113 | 90/103/117 | 84\*/92/105 | 83/85/100 | 86/100/110 | 76\*/92/99 |
| fine (704 078) | 267/381/504 | 112/133/155 | 116/137/159 | 99/119/141 | 93/114/135 | 99/119h/141h | 84/105h/129h |

Experiments: unrolled (`unrolled`), scan (the defaults), scan with all grid nodes (`all_grid_nodes`), edge chunks of
1048576 and 262144 edges (`edge_chunks_1m`, `edge_chunks`), offloading with Grain threads only
(`offload small_buffers no_workers`), and offloading with edge chunks of 262144.

Cases that trained, of 50: unrolled 13/50; scan 34/50; scan-allgrid 36/50; scan-chunks1m 37/50; scan-chunks256k 38/50; scan-offload-0workers 37/50; scan-offload-chunks256k-0workers 39/50.

## Findings

- **The usable GPU memory is about 82 to 85 GiB** and varies between runs: cases planned at 84 GiB trained in one
  experiment and failed in another (30 km with 8 steps: scan 84 GiB trained, chunks of 256k 83 GiB did not).
  Compare planned memory.
- **Scan versus unrolled**: 49 instead of 59 GiB on the coarse mesh with 8 steps, 77 instead of 116 GiB on 40 km;
  unrolled fits 70 km with 8 steps at most, scan 50 km with 16 and 40 km with 8.
- **Connected grid nodes** (the default) save 1 to 8 GiB against all grid nodes (coarse 49 instead of 56 GiB, fine
  112 instead of 116 GiB with 8 steps).
- **Edge chunks** save the most: with chunks of 262144, 35 to 49 GiB on meshes up to 50 km with 8 steps (against 49
  to 62), 40 km with 16 steps fits (75 GiB), and 35 km with 12; they cost 7 to 35 % more compute (memopt).
- **Offloading** saves 6 to 13 GiB more and, with edge chunks, fits the most (39 of 50; 30 km with 8 steps at 76 GiB),
  but needs a slow loader (threads only) to leave host memory for it; the fine mesh with 12 and 16 steps runs out of
  XLA's pinned host pool (64 GB per GPU).
- **The fine mesh** does not fit with any setting: 84 GiB at best (8 steps, offloading and edge chunks), at the limit.
