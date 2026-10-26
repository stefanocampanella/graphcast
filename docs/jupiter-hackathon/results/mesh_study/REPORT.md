# Mesh size and training loss: demo training on every mesh (2026-10-25/26)

The 5000-step demo training (`configs/training/demo_5000.toml`: 500 warm-up steps, cosine decay; 8 nodes, 32 samples
per step, 10 depth levels, zarr_v2 8x2, 8 message passing steps, processor in a scan, connected grid nodes) once per
uniform mesh, code 53ddb9a (`clusters/jupiter/slurm/mesh_study.sh`). Only the mesh changes; 35 km and 30 km (*)
update edges in chunks of 262144 (needed to fit in GPU memory; same results, slower). The parameters do not depend on
the mesh: every run starts from the same weights (seed 42) and reads the same samples in the same order. Training
loss only (no validation), one run per mesh, no tuning: a first clue, not a conclusion.

## Loss, time and memory

| Mesh (nodes) | planned GPU memory | compute per step | 5000 steps | loss 0-99 | loss 4900-4999 | vs coarse |
|---|---|---|---|---|---|---|
| coarse (34,750) | 47.8 GiB | 0.457 s | 43 min | 0.2419 | 0.01185 | +0.0 % |
| 100km (48,662) | 48.8 GiB | 0.459 s | 43 min | 0.2416 | 0.01154 | -2.6 % |
| 85km (65,943) | 50.2 GiB | 0.502 s | 47 min | 0.2415 | 0.01137 | -4.0 % |
| 70km (95,303) | 52.8 GiB | 0.601 s | 55 min | 0.2413 | 0.01129 | -4.7 % |
| 60km (127,948) | 57.4 GiB | 0.659 s | 60 min | 0.2411 | 0.01132 | -4.5 % |
| 50km (181,980) | 65.4 GiB | 0.804 s | 72 min | 0.2409 | 0.01128 | -4.8 % |
| 40km (280,245) | 76.7 GiB | 1.067 s | 94 min | 0.2408 | 0.01128 | -4.8 % |
| 35km* (363,900) | 67.6 GiB | 2.000 s | 171 min | 0.2409 | 0.01168 | -1.5 % |
| 30km* (491,676) | 82.3 GiB | 2.624 s | 224 min | 0.2405 | 0.01169 | -1.4 % |

Loss per variable, mean over steps 4900 to 4999 (the total weights zos by 0.1, the others by 1):

| Mesh | thetao | so | uo | vo | zos |
|---|---|---|---|---|---|
| coarse | 0.00112 | 0.00125 | 0.00369 | 0.00423 | 0.01559 |
| 100km | 0.00110 | 0.00122 | 0.00356 | 0.00409 | 0.01567 |
| 85km | 0.00108 | 0.00119 | 0.00350 | 0.00400 | 0.01594 |
| 70km | 0.00107 | 0.00116 | 0.00344 | 0.00394 | 0.01677 |
| 60km | 0.00106 | 0.00113 | 0.00344 | 0.00394 | 0.01752 |
| 50km | 0.00104 | 0.00109 | 0.00341 | 0.00391 | 0.01824 |
| 40km | 0.00102 | 0.00105 | 0.00339 | 0.00391 | 0.01899 |
| 35km* | 0.00105 | 0.00109 | 0.00350 | 0.00406 | 0.01984 |
| 30km* | 0.00102 | 0.00107 | 0.00349 | 0.00409 | 0.02016 |

The fine mesh (704078 nodes) does not fit: XLA plans 88.6 GiB per GPU with offloading and edge chunks of 262144,
87.2 GiB with chunks of 131072 (usable: about 82 to 85 GiB).

## Findings

- **Temperature, salinity and currents improve with mesh resolution**, steadily down to 40 km: thetao -9 %,
  so -16 %, uo -8 %, vo -8 % from the coarse mesh to 40 km.
- **Sea surface height gets worse**: zos +22 % from coarse to 40 km (+29 % to 30 km), monotonically.
  A possible reason (not tested): with the same 8 message passing steps, a finer mesh sees a smaller area (8 hops of
  120 km on the coarse mesh, of 40 km on the 40 km mesh), while sea surface height is a large-scale field. More
  message passing steps on fine meshes would test this.
- **The total loss is 4 to 5 % lower** from 85 to 40 km than on the coarse mesh, and only 1.5 % lower at 35 and 30 km,
  where the worse sea surface height weighs more.
- **Noise**: the two earlier coarse-mesh demos, which differed only in their initial weights, ended 4 % apart in total
  loss (0.0120 and 0.0115). Differences of a few percent in the total are within that; the monotonic trends of the
  single variables, zos above all, are the clearer signal.
- **Cost**: compute per step grows from 0.46 s (coarse) to 1.07 s (40 km) and 2.0 to 2.6 s with edge chunks (35 and
  30 km); 5000 steps on 8 nodes take 43 min to 3.7 h.

Files: `summary.csv` (all numbers), `scalars/<mesh>.csv` (TensorBoard scalars), `logs/` (job logs), `configs/`.
