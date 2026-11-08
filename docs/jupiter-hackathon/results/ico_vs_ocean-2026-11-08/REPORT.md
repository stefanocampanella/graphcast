# Icosahedral multi-mesh vs ocean mesh (8 Nov 2026)

Detailed version with charts: https://claude.ai/artifact/Bm8DGCiTHcJhL7k9AR5Lwg

## Setup

- **Cases:** GraphCast's icosahedral multi-mesh (`configs/mesh/icosahedral_6.toml`, 6 refinements) with 16 message
  passing steps (GraphCast's setup) and 8; the best ocean setup, uniform 50 km × 16 (`production_run.toml`); the mesh
  study's baseline, uniform 120 km × 8. Two seeds each (42, 43), 8 runs.
- **Protocol:** that of the mesh study (`mesh_msg-2026-10-29`): `configs/training/demo_5000.toml`, 8 nodes, global
  batch 32, 5000 steps (about 17 epochs), validation every 250 steps on the same 128 held-out windows; the metric is the
  training loss (masked, depth- and variable-weighted MSE of residuals / `tendency_std`, one step).
- **Code:** `develop` e7968f8 plus `mesh_study.sh` accepting mesh files other than the uniform meshes (350f636).
  Submitted with `clusters/jupiter/slurm/mesh_study.sh ico_vs_ocean 42,43 icosahedral_6:16 icosahedral_6:8 50km:16
  120km:8`; jobs 2765280–2765287, `training2647`, 78.2 node-h.
- **Baselines rerun:** both are about 8 % better than in the mesh study (see side findings), so its numbers are not
  comparable with these.

## Validation loss at step 5000 (mean of two seeds)

| Case | Valid loss | vs 120 km × 8 | Seed spread | Valid at 2500 | Valid − train | Step (s) | XLA GiB | Node-h (2 runs) |
|---|---|---|---|---|---|---|---|---|
| 120 km × 8 | 0.08382 | 0.0 % | 0.40 % | 0.09210 | 0.0114 | 0.535 | 53.9 | 13.2 |
| icosahedral × 8 | 0.08371 | −0.1 % | 0.05 % | 0.09234 | 0.0116 | 0.585 | 55.3 | 14.2 |
| 50 km × 16 | **0.07524** | **−10.2 %** | 0.29 % | 0.08509 | 0.0109 | 1.395 | 74.0 | 32.3 |
| icosahedral × 16 | 0.08263 | −1.4 % | 0.12 % | 0.09296 | 0.0129 | 0.779 | 55.3 | 18.6 |

- **Icosahedral × 16 is 9.8 % worse than 50 km × 16.** The gap is +1.9 % at step 1000, +9.2 % at step 2500 and +8 to
  +10 % from then on: 5000 steps settle the ranking.
- **At equal resolution the meshes are equal:** icosahedral × 8 and 120 km × 8 differ by 0.1 %. 23 % of the icosahedral
  nodes are over land, which leaves about as many ocean-connected nodes (31,371) as the 120 km mesh (33,781).
- **8 → 16 steps gain little on the multi-mesh** (1.3 %, against 2.5–3.8 % on the ocean meshes in the mesh study): its
  processor's diameter is 14 hops, so 8 steps already reach everywhere.
- Every run has its lowest validation loss at the last step (no overfitting); icosahedral × 16 has the largest
  validation–training gap.

## Per variable (change against 120 km × 8)

| Case | zos | thetao | so | uo | vo | siconc | sithick | usi | vsi |
|---|---|---|---|---|---|---|---|---|---|
| icosahedral × 8 | −9.4 % | +0.6 % | +0.9 % | +0.9 % | +0.9 % | +1.2 % | +3.1 % | +3.8 % | +3.4 % |
| 50 km × 16 | +1.1 % | −9.0 % | −10.2 % | −13.6 % | −14.0 % | −11.6 % | −21.1 % | −7.4 % | −7.0 % |
| icosahedral × 16 | −15.8 % | 0.0 % | +0.8 % | −0.8 % | −0.6 % | +1.9 % | +3.2 % | +3.8 % | +3.8 % |
| **ico × 16 vs 50 km × 16** | **−16.7 %** | +9.9 % | +12.3 % | +14.9 % | +15.5 % | +15.3 % | +30.8 % | +12.1 % | +11.6 % |

120 km × 8 values: zos 0.5003, thetao 0.0156, so 0.0192, uo 0.0127, vo 0.0122, siconc 0.2914, sithick 0.2184,
usi 0.1059, vsi 0.0950.

- **Sea surface height follows reach:** the multi-mesh's edges of up to 7,076 km improve `zos` by 9.4 % (8 steps) and
  15.8 % (16 steps), more than any ocean mesh (best: 120 km × 16, −8.1 % in the mesh study). This confirms the mesh
  study's explanation (reach = steps × spacing).
- **Local fields follow ocean resolution:** temperature, salinity and currents on the icosahedral mesh stay within 1 %
  of the 120 km mesh; sea ice is 3–4 % worse (probably coastline alignment in polar and marginal seas; not checked).
- In the overall loss the `zos` gain is small: 0.0017 (weight 0.02) against 0.0090 lost on the other eight variables.

## Cost

- Icosahedral × 16 takes 0.78 s per step (56 % of 50 km × 16) and 55 GiB, with no edge chunks or offloading;
  40,000 steps on 8 nodes would take 8.7 h (69 node-h) against 15.5 h (124 node-h).
- The grid side sets most of the cost: 0.96–1.09 M encoder and 2.08 M decoder edges in every case; the multi-mesh
  processor has 328 k edges against 1.06 M for the 50 km mesh.
- At similar cost the ocean mesh still looks better: 85 km × 16 ran at 0.76 s and was 5.4 % better than the baseline
  in the mesh study, against 1.4 % for icosahedral × 16 at 0.78 s (two code versions: a hint, not a result).

## Graphs (`moby-dick graph-statistics`, training mask, 32 sampled sources)

| | 120 km | 50 km | icosahedral M6 |
|---|---|---|---|
| processor nodes / directed edges | 34,007 / 194,040 | 181,325 / 1,061,739 | 40,962 / 327,660 |
| edge length, median (min–max) | 118 km (17–173) | 50 km (7–74) | 126 km (110–7,076) |
| diameter / average shortest path (hops) | 233 / 99.4 | 448 / 229.2 | 14 / 7.8 |
| average clustering | 0.417 | 0.395 | 0.415 |
| average betweenness / closeness | 2.8e-3 / 0.010 | 1.3e-3 / 0.0044 | 1.7e-4 / 0.130 |
| mesh nodes connected to ocean points | 99.3 % | 99.4 % | 76.6 % |
| ocean points without encoder edges | 2,222 (0.32 %) | 2,342 (0.34 %) | 0 |
| longest decoder edge | 1,286 km | 1,231 km | 131 km |
| grid points per mesh node, encoder median (max) | 24 (4,041) | 4 (2,104) | 27 (3,753) |
| volume-length, median (min) | 0.994 (0.160) | 0.998 (0.363) | 0.988 (0.973) |
| smallest / largest angle | 7.2° / 163.8° | 14.8° / 143.9° | 53.7° / 72.3° |

The ocean meshes' interiors are slightly more regular than the icosahedron, with slivers along the coastline; triangle
quality does not explain the skill gap. The diameter is networkx's 2-sweep lower bound; path metrics are on the largest
connected component of the undirected graph.

## Side findings

- **The ocean meshes leave out the Black Sea.** About 900 GLORYS ocean points (40–47° N, 27–42° E) are in the mask but
  outside the uniform meshes' domain: the decoder takes their predictions from Aegean vertices up to 1,232 km away, and
  the encoder never reads them (895 of the 120 km mesh's 2,222 points without encoder edges). The other missing points
  are coastal (Chile, Greenland, British Columbia), farther from every vertex than the local encoder radius. It affects
  every model trained on the ocean meshes, but is too small (0.13 % of ocean points) to change this comparison.
- **The baselines improved by about 8 % since the mesh study** (120 km × 8: 0.09068 → 0.08382; 50 km × 16: 0.08181 →
  0.07524), with the same config, protocol and validation windows. The most likely cause is the edge-direction fix of
  6 Nov (`0473751`, azimuth features were meaningless before); not ablated. The ranking is unchanged (50 km × 16 vs
  120 km × 8: −10.2 % now, −9.8 % before).

## Conclusions

- **The 50 km ocean mesh × 16 remains the configuration to use:** 9.8 % better than the icosahedral multi-mesh at equal
  steps, and better on 8 of 9 targets by 10–31 %.
- **The multi-mesh's long edges help `zos`** (−16.7 %), the field the mesh study found limited by reach.
- **Next:** long-range edges on the ocean mesh (hierarchy of coarser ocean meshes merged into the 50 km processor, as
  the multi-mesh does); icosahedral M7 (163,842 nodes, ~55 km, ~125,000 ocean-connected) for a resolution-matched
  comparison; the Black Sea in the mesh domain (or out of the mask); an ablation of the edge-direction fix.
- **Caveats:** 5000 steps and two seeds; one-step loss, not rollout skill (large-scale `zos` errors may weigh more in
  rollouts); no tuning for the multi-mesh (radius 0.6 × mesh size, latent 512).

Files: `summary_cases.csv`, `summary_runs.csv`, `curves.csv` (validation loss every 250 steps per run). Scripts, logs and
graph statistics (JSON) in `$PROJECT/claude-work/ed336b51-7c1c-4db2-b3ca-b6880ad836fd`.
