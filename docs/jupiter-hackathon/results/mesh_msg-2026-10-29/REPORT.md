# Mesh × message passing steps: validation loss (29–30 Oct 2026)

## Setup

- **Grid:** 5 uniform meshes × 3 message passing step counts × 2 seeds, i.e. 30 runs of the 5000-step demo
  (`configs/training/demo_5000.toml`).
- **Training:**
  - 8 nodes, global batch 32, 500 warm-up steps, cosine decay to step 5000; about 17 epochs over the 9,676 training
    windows.
  - 50 levels and 9 targets (ocean state and sea ice), `zarr_v3_lexi`, 1 Grain worker.
  - Commit `bbad6cb`, `clusters/jupiter/slurm/mesh_study.sh`.
- **Validation:**
  - every 250 steps, on 128 fixed windows of the held-out period (Jul 2019–Dec 2023);
  - the same windows in every run;
  - the metric is the training loss itself: masked, depth- and variable-weighted MSE of residuals divided by
    `tendency_std`, one step.
- **Seeds:** 42 and 43 change the initial weights and the order of training samples.
- **Edge chunks:** 40 km × 12 and × 16 update edges in chunks of 262144 to fit in GPU memory. Results are the same,
  steps are slower.
- **Cost:** 384 node-hours on `training2647`; all 30 runs completed.

## Validation loss at step 5000 (mean of two seeds; change against coarse × 8)

| Mesh (nodes) | 8 steps | 12 steps | 16 steps |
|---|---|---|---|
| coarse 120 km (34,750) | 0.09068 (0.0 %) | 0.08933 (−1.5 %) | 0.08843 (−2.5 %) |
| 85 km (65,943) | 0.08872 (−2.2 %) | 0.08674 (−4.3 %) | 0.08578 (−5.4 %) |
| 60 km (127,948) | 0.08635 (−4.8 %) | 0.08422 (−7.1 %) | 0.08294 (−8.5 %) |
| 50 km (181,980) | 0.08508 (−6.2 %) | 0.08294 (−8.5 %) | 0.08181 (−9.8 %) |
| 40 km (280,245) | 0.08418 (−7.2 %) | 0.08202 (−9.6 %) | **0.08052 (−11.2 %)** |

- **Seed noise is small:** the two seeds differ by 0.03–0.57 % (median about 0.3 %). Each step along either axis
  improves the loss by 1–3 %, several times that noise.
- **No overfitting yet:** in every run, the lowest validation loss is the last one (step 4999). The gap between
  validation loss and training loss (mean over steps 4900–4999) is 0.010–0.012 in every case. It is slightly smaller
  for finer meshes, so the larger models do not overfit more at 5000 steps.
- **Neither axis has saturated:**
  - from 12 to 16 steps the loss falls 1.0–1.8 % on every mesh;
  - from 50 km to 40 km it falls 1.0–1.6 % at each step count.

## Per variable (validation loss, change against coarse × 8)

| Case | zos | thetao | so | uo | vo | siconc | sithick | usi | vsi |
|---|---|---|---|---|---|---|---|---|---|
| coarse × 16 | −8.1 % | −1.3 % | −0.8 % | −2.5 % | −2.8 % | −1.5 % | −2.0 % | −1.1 % | −0.2 % |
| 85 km × 16 | −6.3 % | −4.2 % | −4.0 % | −6.3 % | −6.8 % | −4.7 % | −9.7 % | −3.2 % | −2.1 % |
| 60 km × 16 | −3.9 % | −7.2 % | −7.8 % | −10.3 % | −10.8 % | −8.0 % | −19.0 % | −5.4 % | −3.8 % |
| 50 km × 8 | +6.7 % | −6.6 % | −8.1 % | −8.2 % | −8.3 % | −4.5 % | −18.8 % | −4.0 % | −3.1 % |
| 50 km × 16 | −1.8 % | −8.8 % | −9.5 % | −12.0 % | −12.4 % | −9.3 % | −23.3 % | −6.2 % | −4.5 % |
| 40 km × 8 | +9.6 % | −7.6 % | −9.8 % | −9.3 % | −9.4 % | −5.9 % | −24.1 % | −5.4 % | −4.2 % |
| 40 km × 16 | +0.6 % | −10.0 % | −11.5 % | −13.6 % | −14.0 % | −11.5 % | −28.3 % | −7.4 % | −6.4 % |

Coarse × 8 values: zos 0.563, thetao 0.0163, so 0.0201, uo 0.0137, vo 0.0133, siconc 0.326, sithick 0.260,
usi 0.115, vsi 0.101. All cases are in `summary_cases.csv`.

- **Resolution:** temperature, salinity, currents and sea ice improve steadily with mesh resolution. Sea-ice thickness
  gains most (−28 %), then currents (−14 %).
- **Sea surface height:**
  - **The finding:** `zos` needs more message passing steps as the mesh gets finer.
    - With 8 steps, `zos` gets worse with resolution: +9.6 % at 40 km.
    - With 16 steps the loss is back to the coarse level (+0.6 % at 40 km). On the coarse mesh, 16 steps improve it
      by 8 %.
  - **The reason:** information travels at most one mesh edge per step, so the reach of the processor is about steps
    × spacing.
    - That is 960 km for coarse × 8, 640 km for 40 km × 16, and 320 km for 40 km × 8.
    - Sea surface height is a large-scale field: it follows that reach, while the local fields follow resolution.
  - **History:** the hackathon's mesh study, with 8 steps only, found the same `zos` trend and suggested this test.

## Cost

| Case | Step (s) | XLA GiB per GPU | Node-hours, 2 × 5000 steps | 40,000 steps on 8 nodes |
|---|---|---|---|---|
| coarse × 8 | 0.52 | 54.0 | 12.6 | 5.8 h, 46 node-h |
| 85 km × 16 | 0.76 | 54.3 | 18.1 | 8.4 h, 68 node-h |
| 60 km × 16 | 1.10 | 63.9 | 25.6 | 12.2 h, 98 node-h |
| 50 km × 16 | 1.40 | 74.1 | 32.2 | 15.6 h, 124 node-h |
| 40 km × 8 | 1.14 | 79.1 | 26.3 | 12.7 h, 101 node-h |
| 40 km × 12 (edge chunks) | 2.43 | 69.8 | 55.2 | 27 h, 216 node-h |
| 40 km × 16 (edge chunks) | 3.10 | 77.8 | 69.9 | 34 h, 276 node-h |

- **Cheapest per step of gain:** at a given cost, more message passing steps on a mid-size mesh beat a finer mesh
  with fewer steps.
  - 50 km × 16 (1.40 s, −9.8 %) beats 40 km × 12 (2.43 s, −9.6 %).
  - 60 km × 12 (0.91 s, −7.1 %) beats 40 km × 8 (1.14 s, −7.2 %).
- **Edge chunks:** 40 km × 12 and × 16 need them, which roughly doubles their step time.

## Conclusions

- **Best:** 40 km × 16, 11.2 % lower validation loss than coarse × 8, at six times the step time.
- **Best value:** 50 km × 16, 1.4 points behind at 45 % of the cost.
  - It needs no edge chunks (74 GiB per GPU).
  - It keeps `zos` at the coarse level (−1.8 %).
  - A 40,000-step run on 8 nodes takes about 16 h.
- **Both axes are still improving:**
  - 20–24 message passing steps on 50 km would need edge chunks (memory);
  - 35 km × 12 also fits with edge chunks.
  
  Both are worth a short follow-up before the production run.
- **Caveats:**
  - 5000 steps is about 17 epochs. Larger models may gain more from the longer production run, and may overfit
    earlier. The validation curves of the production run will show it.
  - The metric is the one-step loss in normalized units, not rollout skill against persistence or climatology.
