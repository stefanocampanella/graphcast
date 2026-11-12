# Lambrechts meshes vs uniform meshes (13 Nov 2026)

## Setup

- **Meshes** (seamsh v0.6.0+fork, Natural Earth 1:10m coastline coarsened to 10 km, Black Sea straits refined): the three
  `configs/mesh/lambrechts*.toml`, whose size is the minimum of a field growing with the distance from the coast and
  one growing with the square root of the depth (GLORYS `deptho`, between its 10th and 90th percentiles):
  `lambrechts_mini` (coast 150 → 300 km, depth 150 → 500 km), `lambrechts_coarse` (100 → 200 km, 100 → 500 km),
  `lambrechts` (30 → 200 km, 25 → 500 km, the sizes of Lambrechts et al.); uniform meshes from 200 to 50 km, with new
  configs `uniform_200km` and `uniform_150km` matching the node counts of the two smaller Lambrechts meshes.
- **Protocol:** the mesh study's (`configs/training/demo_5000.toml`, 8 nodes, 5000 steps, validation every 250 steps on
  the same 128 windows), 16 message passing steps, seeds 42 and 43; study `lambrechts_vs_uniform`, jobs
  2868109–2868124, 146.6 node-h on `training2647`; 50 km × 16 from study `ico_vs_ocean_v060` (12 Nov, same code).

## Results (validation loss at step 5000, mean of two seeds)

| Mesh | Nodes | Median edge | Mean path (hops) | Valid loss | Seed spread | vs uniform fit | Step (s) | XLA GiB | Node-h (2 runs) |
|---|---|---|---|---|---|---|---|---|---|
| lambrechts_mini | 11,786 | 177 km | 59 | 0.08836 | 0.03 % | +1.1 % | 0.753 | 52.0 | 17.4 |
| lambrechts_coarse | 23,424 | 125 km | 87 | 0.08515 | 0.07 % | +1.5 % | 0.664 | 53.8 | 15.5 |
| lambrechts | 115,671 | 28 km | 131 | 0.08371 | 0.05 % | +9.7 % | 1.137 | 62.0 | 26.0 |
| 200 km | 13,743 | 192 km | 67 | 0.08664 | 0.02 % | +0.0 % | 0.657 | 52.8 | 15.3 |
| 150 km | 22,960 | 146 km | 83 | 0.08426 | 0.18 % | +0.3 % | 0.612 | 53.1 | 14.3 |
| 120 km | 34,527 | 117 km | 99 | 0.08201 | 0.05 % | +0.0 % | 0.638 | 54.1 | 14.9 |
| 85 km | 65,775 | 84 km | 143 | 0.07847 | 0.36 % | -0.6 % | 0.767 | 54.1 | 17.9 |
| 60 km | 127,768 | 60 km | 195 | 0.07566 | 0.17 % | -0.3 % | 1.104 | 63.8 | 25.3 |
| 50 km | 182,023 | 50 km | 229 | 0.07464 | 0.04 % | +0.5 % | 1.403 | 74.8 | 31.9 |

- **Uniform meshes lie on one power law:** valid loss = 0.153 · N^−0.060 (N mesh nodes), every uniform mesh within
  0.6 % of it.
- **The Lambrechts meshes are worse at equal node count:** +1.1 % (mini) and +1.5 % (coarse) above the uniform fit,
  +9.7 % for the full `lambrechts` mesh, whose loss (0.0837, 116k nodes) is above that of uniform 120 km × 16 (0.0820,
  35k nodes). Against uniform 60 km (128k nodes, same step time, 1.10 s against 1.14 s): +10.6 %.
- **Per variable** (`lambrechts` vs 60 km): `zos` −4.7 %; `thetao` +10.5 %, `so` +12.8 %, `uo` +17.4 %, `vo` +18.3 %,
  `siconc` +6.2 %, `sithick` +14.1 %, `usi` +4.9 %, `vsi` +1.6 %. `lambrechts_coarse` vs 150 km: within ±2.4 % per
  variable; `lambrechts_mini` vs 200 km: +0.1 to +3.5 %.
- **Likely reason (not tested):** the refinement puts nodes on shelves and coasts (median edge 28 km) and leaves the
  open ocean at up to ~200 km, where most ocean points and most of the loss are; the coarse offshore edges also
  shorten the paths (131 hops on average against 195 at 60 km), which fits the gain in `zos`, the variable limited by
  reach in the mesh study.
- Seed spread 0.02–0.36 %: every difference above is larger than the noise.

## Conclusions

- For one-step skill in this loss, uniform resolution beats coast- and depth-driven refinement at equal node count; the
  50 km uniform mesh × 16 remains the configuration to use.
- Refinement in the deep ocean would be the test of the opposite hypothesis (sizes from the variability of the
  targets rather than from the depth).

Files: `cases.csv`, `runs.csv`, `curves.csv`. Scripts and graph statistics in
`$PROJECT/claude-work/ed336b51-7c1c-4db2-b3ca-b6880ad836fd` (`lambrechts_analyze.py`, `lambrechts/`).
