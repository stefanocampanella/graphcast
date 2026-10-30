# GraphCast ocean training on JUPITER: findings for the current code (29 Oct 2026)

Reference for designing the production training run. Code: branch `feature/anomaly-downloader` (PR #1), commits up to
56e9d78. All measurements use 50 depth levels, the standardized anomaly stores, and the variables of
`configs/training/full.toml`. Results directories (gitignored): `results/2026-10-29-alllevels` (loader, io_threads,
cpus, scaling; 1024 steps, two repetitions per case) and `results/2026-10-29-memfit-alllevels` (GPU memory fit; 30
steps, one repetition). Copies of the summaries: `docs/jupiter-hackathon/results/benchmark-2026-10-29/`.

## Recommended setup

| Setting | Recommendation | Evidence |
|---|---|---|
| Store | `zarr_v3_lexi` (or `zarr_v2`); directories, not SquashFS | Layouts within 5 %; v3 has 25,891 files, v2 2.5 M (8 M inode quota) |
| Grain | 1 worker × 2 threads, buffers 2 (now the `.env` default) | As fast as any setting; ~230 GB node memory left |
| TensorStore I/O threads | 16 (default) | 8, 16, 32 equal |
| CPUs per task | 72 | 8 CPUs: steps 24 % slower |
| Model memory policy | scan, no offloading, no edge chunks (defaults) | Fits up to 50 km × 16 steps at full speed |
| Mesh × message passing | 50 km × 16 (best value) or 40 km × 16 (lowest loss) | Mesh study: −9.8 % / −11.2 % validation loss |
| Nodes | Any: step time is flat from 1 to 4 nodes | Throughput 7.8 → 15.9 → 31.6 samples/s |
| Variables, loss | As in `full.toml`: surface weights 0.02, depth-linear level weights | Surface and volume variables contribute equally |

## Task, normalization and loss

- **Targets:** `zos`, `thetao`, `so`, `uo`, `vo` and sea ice (`siconc`, `sithick`, `usi`, `vsi`). They are also
  inputs.
- **Static inputs:** `glorys_deptho`, `glorys_mask` (one channel per level), `glofas_mask`, `uparea`, `z`, `lsm`.
- **Forcings:** the 8 ERA5 variables and `dis24` at the target time, plus year progress.
- **Unused:** waves (a later study), `thetao_bottom`, `so_bottom`.
- **Predictions:** residuals divided by `tendency_std`. Points with zero `tendency_std` (constant series, outside the
  masks) are excluded from the loss per variable and level, and keep the input value.
- **Level weights:** linear in depth, decreasing, summing to 1 (minimum 0.01). The weights are nearly equal above
  ~500 m.
  - A volume variable's loss is therefore a mean over levels, about 1/50 of a surface variable's.
  - The 0.02 surface weights compensate for this. They must change with the number of levels.
- **Training check** (60 steps, 1 node): the loss went from 0.259 to 0.146.
  - Every target fell: sea ice by 43–75 %, `zos` by 26 %, the volume variables by 33–35 %.
  - Weighted contributions: surface 0.14, volume 0.12.
  - XLA planned 54.2 GiB per GPU.
- **Residuals or direct prediction** (28 Oct, 10 levels, 2000 steps, state error relative to persistence):
  - direct prediction 0.56; residuals unscaled 0.51; residuals / `tendency_std` 0.53;
  - direct prediction does not beat persistence for `thetao` and `so`, while residuals do.

  This was a single seed: worth repeating at 50 levels with the sea-ice targets.

## Data loading (coarse mesh, 8 message passing steps, 1 node)

Step time in seconds, mean of two runs of 1024 steps, after step 10:

| Layout | 0 × 4 | 1 × 2 | 2 × 2 | 4 × 2 | 8 × 2 |
|---|---|---|---|---|---|
| zarr_v2 | 0.489 | 0.497 | 0.502 | 0.486 | host OOM |
| zarr_v3_lexi | 0.504 | 0.494 | 0.491 | 0.484 | host OOM |
| zarr_v3_colexi | 0.513 | 0.485 | 0.498 | 0.516 | host OOM |
| squashfs_v2 | 0.514 | 0.552 | 0.515 | 0.545 | host OOM |
| squashfs_v3_lexi | 0.543 | 0.536 | 0.520 | 0.538 | host OOM |
| squashfs_v3_colexi | 0.531 | 0.537 | 0.515 | 0.550 | host OOM |

- **Compute bound everywhere:** compute takes 0.43–0.49 s and data waits 0.013–0.05 s; GPU utilization is 89–91 %.
  - The old dataloader, which standardized each sample, took 2.4 s per step at 50 levels with 2 × 2. The new store is
    about 5 times faster.
  - Threads only (0 × 4) waits least for data (0.016 s), but its compute median is 3–5 % higher. The loader shares the
    training process.
- **Host memory** per node (858 GB usable):
  - 0 workers: 580–630 GB peak;
  - 1 worker: 605–680 GB;
  - 2 workers: 700–750 GB;
  - 4 workers: 840–857 GB, with as little as 0.25 GB left;
  - 8 workers: memory ran out in all 12 jobs, which then hung at step 4.

  A sample is about 5.7 GB (shared memory grows 45 GB per node per worker with 2 buffers and 4 tasks). The training
  processes alone take about 580 GB.
- **TensorStore I/O threads** (zarr_v2, 4 × 2): 0.497 s with 8 threads, 0.505 s with 16, 0.489 s with 32. No effect.
- **CPUs per task** (zarr_v2, 4 × 2): 72 CPUs gave 0.512 s; 8 CPUs gave 0.635 s, with 0.13 s data waits.
- **Startup:** step 0 ends 83–105 s after launch (compilation ~27 s).

## Scaling (zarr_v2, 4 × 2, coarse × 8)

| Nodes | GPUs | Step | Samples/s |
|---|---|---|---|
| 1 | 4 | 0.515 s | 7.8 |
| 2 | 8 | 0.502 s | 15.9 |
| 4 | 16 | 0.506 s | 31.6 |

Data parallelism scales linearly; 8 nodes (the 5000-step demo of the hackathon) are expected to do the same.

## GPU memory: mesh × message passing steps (50 levels)

Step time in seconds (OOM: does not fit; the limit is about 80–85 GiB planned by XLA, 95 % of 96 GB):

| Mesh (nodes) | Defaults: 8 / 12 / 16 steps | Offloading, 0 workers: 8 / 12 / 16 | Edge chunks 262144: 8 / 12 / 16 |
|---|---|---|---|
| coarse 120 km (34,750) | 0.50 / 0.52 / 0.62 | 0.56 / 0.63 / 0.70 | 0.64 / 0.64 / 0.72 |
| 100 km (48,662) | 0.52 / 0.58 / 0.63 | 0.59 / 0.82 / 0.78 | 1.08 / 1.36 / 1.61 |
| 85 km (65,943) | 0.57 / 0.65 / 0.76 | 1.37 / 0.77 / 0.87 | 0.94 / 1.16 / 1.37 |
| 70 km (95,303) | 0.61 / 0.76 / 0.87 | 0.74 / 0.92 / 1.03 | 1.15 / 1.29 / 1.88 |
| 60 km (127,948) | 0.70 / 0.86 / 1.03 | 0.84 / 1.05 / 1.24 | 0.92 / 1.16 / 1.39 |
| 50 km (181,980) | 0.80 / 1.05 / 1.29 | 1.00 / 1.31 / 1.68 | 1.44 / 1.96 / 2.45 |
| 40 km (280,245) | 1.08 / OOM / OOM | 1.69 / 1.78 / OOM (14: 1.99) | 1.73 / 2.34 / 2.95 |
| 35 km (363,900) | OOM | OOM | 2.09 / 2.87 / OOM |
| 30 km (491,676), fine (704,078) | OOM | OOM (fine 12–16: pinned host OOM) | OOM |

- **Defaults:** the largest models that fit are 50 km with 16 steps (74 GiB) and 40 km with 8 (79 GiB), with no speed
  penalty. 107–198 GB of node memory remain with 2 × 2 workers.
- **Offloading** (`policy.remat_loss = true`) adds 40 km with up to 14 steps.
  - It needs 0 workers and small buffers, and leaves as little as 30 GB of node memory.
  - Compute rises 10–25 %; single runs vary (e.g. 85 km × 8: 1.37 s).
- **Edge chunks** add 35 km with up to 12 steps, at 1.3–3 times the compute.
- **30 km and the fine mesh** fit with none of these settings: the graph would have to be sharded over the GPUs.
- These limits are the same as on 25 Oct with the old dataloader and fewer targets. Sea ice, the static inputs and
  per-variable masks hardly change GPU memory.

## Planning the production run

40,000 steps (as `full.toml`), with step times from the tables. The step time does not depend on the node count
(local batch 1 per GPU).

| Model | Step | Wall time | Node-hours on 1 / 8 nodes |
|---|---|---|---|
| coarse × 8 (`full.toml`) | 0.50 s | 5.6 h | 6 / 44 |
| coarse × 16 | 0.62 s | 6.9 h | 7 / 55 |
| 60 km × 16 | 1.03 s | 11.4 h | 11 / 92 |
| 50 km × 16 | 1.29 s | 14.3 h | 14 / 115 |
| 40 km × 8 | 1.08 s | 12.0 h | 12 / 96 |
| 40 km × 14, offloading | 1.99 s | 22.1 h | 22 / 177 |
| 35 km × 12, edge chunks | 2.87 s | 31.9 h | 32 / 255 |

- **Training data:** 9,677 days up to `split_date` (2019-06-30), i.e. about 9,676 one-day windows.
  - 40,000 steps on 1 node (global batch 4) are about 17 epochs.
  - On 8 nodes (global batch 32) they are about 130 epochs.
  - Choose the node count by global batch and epochs, not by speed. The learning rate schedule (warmup 1000, cosine
    to 40,000) was set for the demo; revisit it with the batch size.
- **Budget:**
  - `training2647` had about 2,150 node-hours on 27 Oct (until 31 Dec 2026).
  - `e-dev-2026d09-119` has about 3,480 (until 28 Feb 2027).
  - Even the largest configurations above cost under 10 % of one project.
- **Long runs:** add `--time` explicitly (the partition default is 1 h) and restart from Orbax checkpoints. Grain
  iterator checkpoints must be restored with the same worker count.
- **Queue:** 1-node jobs waited 4–16 h on 27–29 Oct. `develbooster` is in maintenance until 29 Nov 2026.

## Caveats and open questions

- **No validation loss during training:** training computes no test loss. Evaluate on the held-out period after
  `split_date` separately (rollouts, against persistence and climatology).
- **Normalization leakage:** statistics (climatology, anomaly_std, tendency_std) cover 1993–2023, including the
  held-out period (accepted for now).
- **Mesh choice:** done with residuals and held-out validation (30 Oct 2026, `../mesh_msg-2026-10-29/REPORT.md`).
  - Validation loss falls along both axes; 40 km × 16 is 11.2 % below coarse × 8, and 50 km × 16 is 9.8 % below at
    45 % of the cost.
  - The hackathon's mesh study ran while the residual connection was disabled, and is superseded.
- **Thin statistics:**
  - the residual comparison used one seed, 10 levels, and 2000 steps;
  - memfit used 30 steps and one run;
  - loader timings used the coarse mesh only. On larger meshes the loader matters even less, since the data per
    step is the same.
- **Measurement interference:** jobs share the file system; some timings were taken while up to 40 benchmark jobs
  read the same stores.
- **Deferred:** waves (with `waverys_mask`), `thetao_bottom` and `so_bottom`, sigma coordinates.

## Reproducing

```bash
cd clusters/jupiter/benchmarks/jube && uv sync
./run_final.sh 2026-10-29 alllevels         # loader, io_threads, cpus, scaling (76 jobs, long)
./run_memfit.sh 2026-10-29 memfit-alllevels  # 3 x 50 memfit jobs (quick)
python3 summarize.py results/2026-10-29-alllevels
```
