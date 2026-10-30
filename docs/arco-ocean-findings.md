# ARCO-OCEAN and its statistics: findings from GraphCast training (Oct 2026)

Findings from preparing ARCO-OCEAN for training GraphCast (PR #1 of this repository, 26–30 Oct 2026). They are meant
as input for future revisions of `arco-make` and of the published dataset.

**Sources examined:**
- the dataset, `s3://ogs-arco-ocean/datasets/res=0p25`;
- its climatology and anomaly_std (`.../climatology/res=0p25`, `.../anomaly_std/res=0p25`);
- the `tendency_std` that arco-make's `diff_std` step computed (`s3://ogs-arco-ocean/tendency_std/res=0p25`, also
  copied to `/e/scratch/training2647/graphcast_data/zarr_v2/tendency_std`);
- local copies of all of them on JUPITER (`/e/fscratch/e-dev-2026d09-119/ogs-arco-ocean-50*`).

**How it was examined:**
- **Reference:** `graphcast download` standardizes the dataset once with explicit conventions, and
  `graphcast compute-tendency-std` recomputes `tendency_std` from the result. The recomputation was compared with
  arco-make's copy over all 27 variables and 50 levels, point by point.
- **Raw data:** they were scanned for values outside the static masks, missing values inside them, and fields that
  should be constant but are not.
- **Records:** throwaway scripts and CSVs are in `$PROJECT/claude-work/5ec59923-bcb3-483e-a402-cd793d0e8df5/`
  (`tendency_report*`, `swh_*`, `floor_and_lsm*`, `outside_masks.py`, `static_outside.py`). The decisions taken in
  this repository are in `CLAUDE.md` ("Standardized anomaly stores"), and the dated record is in `docs/logbook.md`.

## Summary

| # | Finding | Affected | Impact on users | Recommendation |
|---|---|---|---|---|
| 1 | `diff_std` drops Feb 29 (365-day calendar); the dataset keeps it | `tendency_std`, all variables | Median relative difference 3–6e-4; up to 18 % where an extreme day falls near Feb 29 | Keep Feb 29 everywhere; document how the climatology indexes it |
| 2 | z = (x − c) / σ is 0/0 = NaN where σ = 0, and such days are skipped | Sea ice, waves, `dis24`, deep `so`, `lsm` | `tendency_std` 11–54 % too large at 11–42 % of points | Define z = 0 where x equals the climatology; floor σ |
| 3 | Constant fields (`z`, `lsm`) get a climatology and anomaly_std | `z`, `lsm` | Standardized values are float32 rounding noise (0, ±1, ±1.41) | Publish static fields as static (no time), min-max or raw |
| 4 | `z` and `lsm` are not constant | `z` (2022-10-01), `lsm` (1998-05-04, 2014-04-05) | Glitches propagate into statistics | Fix upstream; take a glitch-free day |
| 5 | Wave data outside `waverys_mask` on some days | `swh`, `mwp`, `mwd`, `usd`, `vsd` | Statistics undefined or wrong at 401 points | Use the same bathymetry version for mask and fields; investigate the source |
| 6 | Waves are 0 under sea ice (`masked_fill`) | Wave variables | Zero-inflated series; σ = 0 days | Document it; consider an explicit flag |
| 7 | GLORYS values outside the 3D `glorys_mask` | `thetao`, `so`, `uo`, `vo` | 6,315 points at level 10, 13,206 at level 40 | Make data and mask consistent |
| 8 | Static fields inconsistent outside their masks | `glorys_deptho` (0 or NaN, 3 tiny values), `uparea` (NaN) | Each user must re-mask | Use one fill convention (0 or NaN) per dataset |
| 9 | `uo`/`vo` at depth index 49 are 0 at all sea points | `uo`, `vo` | anomaly_std 0 everywhere at that level | Drop or document the level |
| 10 | `tendency_std` is NaN where a series is constant or missing, never 0 | `tendency_std` | Indistinguishable from "no data"; needs a separate mask | 0 for constant series, NaN only for no data |
| 11 | Sea-ice differences are zero-inflated | `siconc`, `sithick`, `usi`, `vsi` | Plain std is dominated by zeros | Offer the std of the non-zero differences |
| 12 | Statistics cover 1993–2023 | All | Held-out years leak into normalization | Make the reference period configurable |
| 13 | Very small anomaly_std inside the masks | `siconc` 4e-12, `dis24` 1.5e-11, `tp` 1.5e-8 | Huge z-scores | Publish a per-variable floor |
| 14 | Layout: zarr v2, one ~4 MB chunk per day and level | Dataset and statistics | 2.5 M objects for a 5 TB copy; slow with zarr-python | Offer a sharded zarr v3 layout |

## Details

### 1. Feb 29 and the calendar
- **The difference:** arco-make's `compute_climatology` uses a 365-day calendar, with a 29-day Gaussian window. Its
  `diff_std` step also drops Feb 29 from the series whose one-step differences it accumulates. The dataset itself
  keeps Feb 29.
- **Effect:**
  - Across all 27 variables and all levels, the median relative difference between arco-make's `tendency_std` and a
    recomputation that keeps Feb 29 is 3–6e-4.
  - Without Feb 29, the recomputation matches the copy within 1e-7 (float32).
  - At the few clean series more than 5 % apart, an extreme day around Feb 29 carries 11–18 % of the sum of squares.
- **Indexing:** arco-make does not document how a consumer should index the 365-day climatology on Feb 29. We use
  Feb 28's climatology and anomaly_std for Feb 29, and shift later days of a leap year back by one; they vary
  smoothly over a few days.
- **Recommendation:** keep Feb 29 in all derived products. Document the mapping (or publish a 366-day climatology),
  so that standardization and `tendency_std` agree.

### 2. Zero anomaly_std
- **Where it happens:** where the anomaly_std is exactly 0, the value equals the climatology (typically both 0).
  arco-make's z is then 0/0 = NaN, and those days are skipped in `diff_std`.
- **Frequency:** the sea-ice variables (`siconc`, `sithick`, `usi`, `vsi`) and the wave variables under sea ice have
  data every day, but σ = 0 on 40–48 % of days at some points.
- **Effect:** skipping those days leaves out their (zero) differences, so arco-make's `tendency_std` is larger than
  with z = 0 on those days. The recomputation gives 0.65–0.9 of the copy's value.
  - sea ice: 37–42 % of finite points more than 5 % apart;
  - waves: 11 %;
  - `dis24`: 12 %;
  - `so` at levels 45–47: 130 points;
  - `so_bottom`: 31 points;
  - `lsm`: 332 points.
- **Our rule:** z = 0 where |x − c| ≤ 4·eps·max(|x|, |c|) (float32 eps), whatever the anomaly_std. Otherwise
  z = (x − c) / max(σ, floor), with the floor of item 13.
- **Recommendation:** define z on such days (0), and document the rule.

### 3–4. Static fields stored as time series
- **Constant in time:** `z` (geopotential at the surface) and `lsm` (land-sea mask) are ERA5 fields that should not
  change.
- **What arco-make does:** it gives them a climatology and an anomaly_std. Both differ from the value by float32
  rounding, so z is a ratio of rounding errors (0, ±1, ±1.41). The standardized `z` and `lsm` are therefore noise,
  not geopotential or land fraction.
- **Effect on `tendency_std`:** the copy has exactly 0 at 590,566 `z` points and 143,162 `lsm` points. A
  recomputation from the standardized values gives NaN.
- **Glitches:** neither field is exactly constant.
  - `z` changes on 2022-10-01 at the two polar rows (lat ±90, all 1440 longitudes, by about 6×).
  - `lsm` differs from its first day on 1998-05-04 and 2014-04-05, at 532 points (18–24°S).
- **Our workaround:** both are static here. We take the first timestep (glitch-free), min-max normalize it, and give
  no climatology or `tendency_std`.
- **Recommendation:**
  - publish `z` and `lsm` without a time dimension, like the masks and bathymetry, and exclude them from the
    statistics;
  - fix the glitched days upstream, or document them.

### 5–6. Wave variables (WAVERYS)
- **Pipeline:** 3-hourly instantaneous fields are regridded conservatively (`skipna`, `nan_threshold = 1.0`) and
  resampled to daily means (`skipna`). Then `masked_fill` fills NaN with 0, only inside the static `waverys_mask`
  (bathymetry > 0).
- **Under sea ice:** where WAVERYS has no waves (under sea ice), the published value inside the mask is therefore
  0, not NaN. The series are zero-inflated, with σ = 0 days (item 2). We accept this convention, but it is not
  documented in the dataset.
- **Mask version:** `waverys_mask` comes from the WAVERYS bathymetry of version 202311, the dynamic fields from
  202411.
- **Missing values over time:**
  - From 1993-01-02 to 2012-12-31, the daily NaN points of the wave fields are exactly the 393,555 static land
    points.
  - Outside the mask there are data on 1993-01-01 only, at 319 points at 48–80°N (median `swh` 0.09 m).
  - From 2013-01-01 on, there are data at 82 coastal points: 50 every day, 32 intermittently.
  - The same holds for `mwp`, `mwd`, `usd` and `vsd`.
- **Effect on the statistics:**
  - At the 319 points the climatology is NaN on every day. 268 `swh` NaN mismatches (191–319 for the other wave
    variables) come from them.
  - The 82 coastal points are finite in both `tendency_std` versions, but the recomputation is about 0.6 of the copy.
- **Unconfirmed:** the 3-hourly source was not inspected (it needs a CMEMS download). Neither was whether the
  1993-01-01 data come from the regridding of the first timestep.
- **Recommendation:**
  - build `waverys_mask` from the same product version as the fields;
  - check the first day and the 2013 change in the source;
  - document the zero fill under sea ice, or publish a sea-ice flag next to the waves.

### 7–9. Masks and the data they cover
- **Masks per source:** `glorys_mask` (3D: one mask per depth level), `waverys_mask` and `glofas_mask`. ERA5
  variables have no mask.
- **GLORYS values outside the 3D mask:** `thetao`, `so`, `uo` and `vo` have values at 6,315 points at depth index 10
  and 13,206 at index 40, all close to the neighbouring sea. We set them to 0. The surface mask at index 0 has
  693,039 ocean points of 721 × 1440.
- **Raw values outside the masks:**
  - sea-ice variables: 0;
  - `glorys_deptho`: 0 or NaN, plus 3 points ≤ 1.3e-7;
  - `waverys_deptho`: 0;
  - `uparea`: NaN.
- **No missing values inside the masks:** the standardization treats a missing value inside a variable's mask as an
  error, and none occurred in the full dataset.
- **Deepest level:** `uo` and `vo` at depth index 49 are 0 at all 16,664 sea points. Their anomaly_std is therefore 0
  everywhere at that level.
- **Recommendation:**
  - make the data and their masks agree: 0 or NaN outside, the same convention for every variable of a source;
  - document the deepest velocity level, or drop it.

### 10–11. tendency_std conventions
- **NaN conventions in the copy:** it is NaN wherever a variable has no data (land; 66 % of sea points for `siconc`,
  which never see ice; 8 % for `swh`). It is also NaN where the std is 0, and it is never 0.
- **Why that is a problem:** "constant series" and "no data" look the same, so training needs another mask to tell
  them apart. Training on standardized residuals needs exactly this distinction: a constant point is a valid point
  with no change to predict, and it is excluded from the loss.
- **Our convention:** the recomputation, from the standardized store, in float64, with Dask chunks of 256 days of
  one depth level, writes 0 for constant series and inside-mask points. It is finite everywhere: 0 outside the masks.
- **Sea ice:** the one-step differences of the sea-ice variables are zero-inflated (most days have no ice or no
  change). We use the std of the non-zero differences for them, with no minimum count.
- **Cost:** 34.6 min on one JUPITER booster node (32 threads) for all variables and levels, against arco-make's
  expensive `diff_std` run.
- **Recommendation:**
  - 0 for constant series, NaN only where there is never data;
  - an option for the non-zero-difference estimator;
  - Feb 29 kept (item 1).

### 12–13. Reference period and small anomaly_std
- **Leakage:** the climatology, anomaly_std and `tendency_std` cover 1993–2023. Training holds out the years after
  mid-2019, so their statistics leak into the normalization. We accept this for now.
- **Smallest positive anomaly_std inside the masks**, per variable and level:

  | Variable | Smallest positive anomaly_std |
  |---|---|
  | `siconc` | 4e-12 |
  | `dis24` | 1.5e-11 |
  | `sithick` | 8e-11 |
  | `tp` | 1.5e-8 |
  | `so_bottom` | 2.7e-6 |
  | `so` (index 49) | 1.4e-5 |
  | `swh` | 2e-5 |
  | ERA5 forcings | ≥ 0.04 |

  These produce heavy tails in z (`dis24`, `so`, `tp`).
- **Our handling:**
  - the floor is each variable's smallest positive in-sample value, so in-sample z-scores are unchanged;
  - outside 1993–2023, |z| is capped at the variable's in-sample maximum.
- **Recommendation:**
  - compute the statistics over a configurable reference period, e.g. 1993–2018 for training up to 2019;
  - publish the floor (or the smallest positive σ) per variable and level, so that consumers standardize new data
    the same way.

### 14. Storage layout and access
- **Current layout:** the dataset and statistics are zarr v2, with one chunk (~4 MB) per day and depth level. A 5 TB
  local copy has about 2.5 M files.
- **Inode quota:** on JUPITER's fscratch (8 M inodes per project), one such copy takes a third of the quota.
  `mksquashfs` of it took 159 min, against 13–14 min for a sharded copy.
- **Sharded alternative:** zarr v3, shards of 8 days × 50 levels, one chunk per day and level, in lexicographic
  order.
  - It has 25,891 files for the same data, with the same read speed for training at 50 levels: 0.484–0.504 s against
    0.486–0.502 s per training step.
  - zarr-python reads whole shards when a `transpose` codec precedes the sharding codec; TensorStore still reads
    single chunks.
- **Reading from S3** (eu-south-1, 13 GB of `thetao`, a login node with a 20-core quota):
  - TensorStore's `s3` driver reaches about 4.7 GB/s with 256–512 concurrent requests (3.1 GB/s at 128); signed and
    anonymous reads are equally fast.
  - zarr-python with s3fs stays at about 0.29 GB/s whatever the concurrency.
- **Recommendation:**
  - offer (or document how to make) a sharded zarr v3 copy;
  - keep consolidated metadata;
  - recommend TensorStore for bulk reads.

## Conventions adopted in this repository

For reference, the standardized store that training reads (`graphcast download`, `download_config.py`) applies:

1. **Feb 29:** standardized with Feb 28's statistics.
2. **Static fields:** `z` and `lsm` from the first timestep, min-max normalized to 0–1, with no `tendency_std`.
3. **Zero z:** z = 0 within 4 float32 epsilons of the climatology.
4. **Floor:** an anomaly_std floor at the smallest positive in-sample value, and |z| capped outside 1993–2023.
5. **Zeros outside masks:** 0 outside each variable's source mask (GLORYS 3D or level 0, WAVERYS, GLOFAS). A missing
   value inside a mask is an error.
6. **Constant points:** `tendency_std` 0 at constant points, excluded from the loss per variable and level.
7. **Sea ice:** the non-zero-difference estimator for the sea-ice variables.

With these conventions, `thetao`, `so`, `uo`, `vo`, `zos`, the ERA5 forcings other than `z`/`lsm`, `tp` and
`thetao_bottom` differ from arco-make's statistics only by Feb 29 (and a few deep `so` points by items 2 and 3).
