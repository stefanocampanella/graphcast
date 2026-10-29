# Development logbook

Requests from the maintainer, code changes, experiments and insights, in chronological order. Times are CET (UTC+1)
and come from the Claude Code session transcripts, git commits and `sacct`. Each entry is tagged **Request**,
**Change**, **Experiment** or **Insight**. Details that stay valid (commands, layouts, measurements) live in
`CLAUDE.md`; this file records how they came about.

## 2026-10-26: anomaly downloader (PR #1)

- **14:07 Request:** implement `TASK.md`. It asks for a downloader that reads the dataset, climatology and
  anomaly_std (local or S3) and writes one local standardized-anomaly Zarr store (v2 or v3, consolidated, sharded),
  with the preprocessing of `ARCOTimeseriesSource` except the batch dimension. It should have a click CLI, an
  optional pydantic TOML and a bash wrapper. Constraints given on the first plan: no absltest, paths on the CLI and
  settings only in the TOML, simple implementation, no `climatology.align_with`.
- **15:20 Change** `958c0af`: `graphcast download` (`downloader.py`, `download_config.py`), `scripts/download.sh`
  and the configs. It fixes the leap-year mapping (`dataloader.climatology_index`; the old helper gave Feb 26 and
  27 the same climatology day). **15:30** `a59bad8`: pin `s3fs >= 2026.9.0`; without the bound, pip fell back to
  s3fs 0.4 next to `botocore[crt]`.
- **15:33 Experiment:** the output matched `ARCOTimeseriesSource` within 1e-6 (v2, v3, v3 lexicographic) on 20 days
  around 29 Feb 2020, from local copies and from anonymous S3. From local inputs, 30 GB took 38 s, which projects to
  about 3.7 h for the full 10.5 TB; peak memory was 65 GB with 8 threads.
- **15:33 Insight:** the fine-grained `gh` token lacked *Contents: write*, so pushes got a 403 although the
  account has push rights.
- **15:37 Request:** use the `xarray-s3-writer` profile instead of `aws login`; run a small S3 test, longer tests
  locally; estimate the full build time.
- **15:54 Experiment** (login node, local inputs, zarr-python writer): about 0.86 GB/s with 4, 8 or 12 threads
  (about 3.4 h for the full dataset), with peak memory 33, 64 and 87 GB. Only about 5 cores were busy, so the run is
  limited by I/O, not compute. The S3 comparison with the profile passed.
- **15:54 Change** `e61c7bc`: `download.sh` puts `~/.local/bin` on the container `PATH`, because the profile's
  `credential_process` is `aws_signing_helper`. `9172ca2`: throughput in CLAUDE.md.
- **15:58, 16:28 Request:** dry runs of the PR. **15:59:** creating PRs is allowed (a deliberately invalid request
  got 422, not 403), pushing still not. **16:29:** pushing is allowed after the token was fixed.
- **16:34 Request:** find where the time goes; the filesystem had handled more bandwidth before; try
  xarray-tensorstore.
- **Insight:** zarr-python sends all I/O through one event loop. It used about 5 cores, with peaks of 4.5 GB/s for
  reads and 2.5 GB/s for writes, hence 0.85 GB/s overall; the filesystem was not the limit.
- **Experiment:** TensorStore reads the inputs at 11–14 GB/s and writes about 15 GB/s from memory, with identical
  output.
- **16:59 Change** `8db56a5`: read and write with TensorStore; Slurm script for compute nodes. **17:07**
  `1cb1636`: configs for booster nodes. Job 2531614 (16:59–17:04) benchmarked the writer on a booster node.
- **17:07 Insight:** jobs 2531788–90 failed at startup. `/e/scratch/training2647/graphcast_data` belongs to
  another user and is read-only for the group, so the outputs went to `$FSCRATCH/graphcast/data/anomalies`.
- **17:11–19:11 Experiment:** jobs 2531806–08 wrote the full stores, 5.1 TB on disk each: `zarr_v3_lexi` in
  34 min, `zarr_v2` and `zarr_v3` (morton, zarr-python writer) in 1 h 59 min each. All 33 variables matched
  `ARCOTimeseriesSource` on five 3-day windows, including 29 Feb 2000 and 2016. **19:19 Change** `aa83cd3`.
- **19:54 Request:** replace the image, keep the datasets where they are, open the PR. **19:55:** PR #1 opened
  against `develop@esiwace3_hackathon`; the shared `graphcast.sif` now has xarray-tensorstore and s3fs; the previous
  image was kept as `graphcast-2026-10-21.sif`.
- **20:14 Request:** is the session link in the PR safe? **Insight:** the link opens only for its owner or for
  people it is shared with. It does reveal Claude Code use and a session ID, and it is also in the
  `Claude-Session:` trailers of the commits.
- **20:24 Request:** can PR comments be read and acted on? Yes, with `gh` (general, review and inline comments),
  but only when asked or while a session polls; comments from others are reported, not acted on.
- **20:28 Request:** GitHub Actions integrations? `anthropics/claude-code-action` (`@claude` in comments) could
  handle code-only changes, but GitHub runners cannot reach JUPITER data, GPUs, Slurm or the image.

## 2026-10-27: review, I/O measurements, tests, image, quotas

- **10:06 Request** (side quest): the fastest S3 download setup, and the speed and memory use of
  xarray-tensorstore for the later dataloader refactoring; short experiments, findings in CLAUDE.md.
  **10:32:** "Proceed with proposed benchmarks."
- **Experiment** (login node, 20-core quota): on 13 GB of `thetao` from S3, TensorStore's `s3` kvstore reached
  about 4.7 GB/s at `s3_request_concurrency` 256–512 (3.1 GB/s at 128), using about 6 cores and about 2× the data
  in RSS; zarr-python with s3fs stayed at about 0.29 GB/s. A 3-day sample of all variables (2.78 GB) from the
  anomaly stores took 0.2–0.3 s with xarray-tensorstore at 16–64 threads, against 1.25–1.74 s with zarr-python;
  memory was the sample plus 0.4–2.8 GB. Peak RSS was measured with `VmHWM` (`ru_maxrss` was inconsistent).
- **Insight:** the default AWS profile is an expired `aws login` session (`LoginRefreshRequired`). TensorStore
  cannot run a `credential_process`: with `AWS_PROFILE` it reads anonymously and logs credential errors.
- **11:19 Request:** address the 13 review comments:
  - no data paths, URLs, hardware or cluster names in comments, docstrings, config comments or `scripts/`;
  - AWS profiles from the environment;
  - portable `scripts/`;
  - no backward compatibility;
  - drop morton, compare lexicographic and colexicographic;
  - one library for all arrays;
  - bitshuffle benchmarks.

  Also make the PR description shorter, without detailed benchmarks: the preprocessing runs only once.
- **11:30 Change** `b055d14`:
  - TensorStore only: masks are read as int8 with `mask_and_scale=False`, and xarray still decodes `time` (no
    cftime);
  - colexicographic chunk order through a `transpose` codec before the sharding codec;
  - morton and `storage_options` removed;
  - portable `download.sh`, with `clusters/jupiter/slurm/download.slurm` sourcing the cluster env file.
- **Experiment:** in Blosc, bitshuffle compresses less than byte shuffle (1.71× vs 1.89×) and decodes slower (0.74
  vs 1.46 GB/s per core), so it was dropped. Job 2542009 (11:35–12:08) wrote `zarr_v3_colexi` in 32 min.
  TensorStore still reads single chunks of it, while zarr-python reads whole shards. On a 50-level sample,
  colexicographic reads were not slower than lexicographic ones (0.19 vs 0.26 s at 16 threads); to be rechecked in
  the dataloader benchmarks.
- **12:03 Request:** can fsspec plus TensorStore solve the credential helper problem, and at what cost? Drop
  colexicographic order if it reads slower. **Insight:** fsspec reads only the metadata, and the data path stays in
  TensorStore; colexicographic order was kept, since it is not slower.
- **12:05 Request:** explain TensorStore authentication. **Insight:** `aws_credentials` is a context resource
  (`default`, `anonymous`, `environment`, `profile`, `imds`, `ecs`) and specs have no key fields; zarr-python with
  fsspec uses the full botocore chain, including `credential_process` and SSO. **12:12 Change** `a28f22e`.
- **12:16 Request:** handle S3 credentials with botocore. **12:18 Change** `c5a461c`: `get_aws_credentials` resolves
  the keys with botocore, exports them to the environment and selects `environment` (or `anonymous`).
  **Experiment:** signed reads ran at about 4.6 GB/s, as fast as anonymous ones, and bogus keys fail.
- **12:21 Request:** `ty` checks and `ruff format`. **12:22 Change** `cc5379d`: three `ty` fixes (a `Literal` for
  `cname`, `Hashable` encoding keys, `str(name)`). The remaining `dataloader.py` diagnostics predate the PR.
- **13:58 Request:** is the PR updated? Yes; its head matched the branch.
- **14:03 Request:** adopt pytest for all new code, as a dev dependency, in the conventional location, and set the
  ruff line length in `pyproject.toml`. **14:05 Request** (mid-task): commits authored by Claude, with the maintainer
  as co-author. **14:11 Request:** tests may use synthetic or ARCO-OCEAN data, but no AWS profile can be assumed.
- **14:10 Change** `4e4db0f`:
  - pytest in `tests/` (src layout), in a `[dependency-groups] dev` group, with `pythonpath = ["src"]`;
  - 59 tests on synthetic stores: equivalence with `ARCOTimeseriesSource`, read-back with zarr-python and
    TensorStore, layouts, shard order, mocked botocore and the CLI; no network or profile needed;
  - `line-length = 120`.

  **Experiment:** removing the std clip or the transpose codec made 8 tests fail.
- **14:14 Request:** update CLAUDE.md for future tasks. **14:16 Change** `d7241f3`.
- **14:18 Request:** rebuild and replace the image; SquashFS images of `zarr_v2`, `zarr_v3_lexi` and
  `zarr_v3_colexi` for the dataloader benchmarks, on fscratch next to the stores.
- **14:23–14:29 Change:** image rebuilt in 6 min, installing the `dev` group after upgrading pip (the base image's
  pip 24 cannot install dependency groups). The only package changes were pytest and its dependencies plus a
  `pyasn1` patch. All 59 tests pass in the image; it replaced `graphcast.sif`, and the previous one was kept as
  `graphcast-2026-10-26.sif`. **14:31** `732e274`.
- **14:24 Request:** check quotas with `jutil`, not `df`. **Insight:** `jutil project dataquota -p
  e-dev-2026d09-119` gives fscratch 21.1 TB used of 40 TB soft and 44 TB hard (8 M inodes), and project1 14.3 of
  20/22 TB; `jutil user dataquota` shows no limits. **14:26 Request:** keep the obsolete morton `zarr_v3` for now
  and document `jutil`. **Change** `251d19b`.
- **14:31 Request:** check compute budgets with `jutil`; `training2647` is for small runs until December or
  exhaustion, the other project for large runs; look up the JSC allocation policy. **Insight:**
  - **Budgets:** both projects have 1,008,000 core-h on `JUPITER_BOOSTER`, and a booster node costs 288 core-h per
    hour (checked against `sacct`). `training2647` has 621k left until 31 Dec 2026; `e-dev-2026d09-119` has 1.00M
    until 28 Feb 2027.
  - **Policy:** both use `acct-mode=fixed`, one quota for the whole period, with no monthly split. Monthly mode
    allows the previous month's leftover plus one month borrowed at normal priority; once a quota is exhausted, jobs
    run only on free capacity, for at most 6 h.

  **14:33 Change** `96cf710`.
- **14:20–14:37 Experiment:** booster jobs 2544517–19 for the SquashFS images were estimated to start the next day
  at 06:45 (`Priority`), even with the limit lowered to 6 h.
- **14:37 Request:** write the SquashFS images on login nodes, with the settings found in bash history, and discuss
  any different setting first. **Insight:** the history has `mksquashfs SRC DST -noappend -no-xattrs -noI -noD -noF
  -b 1M -comp lz4 -progress` (4 KB blocks in the first tries). The booster jobs were cancelled before starting, and
  the run began at 14:38 on `jpbl-s02-03` at about 4 GB/s.
- **14:56 Experiment:** the run was killed silently at 69% of `zarr_v3_lexi`, leaving a 3.8 TB partial image.

## 2026-10-28: SquashFS images, logbook

- **11:57 Request:** did the previous command finish? **Insight:** no, nothing usable was written; the source
  stores are intact. The scratchpad in `/tmp` is node-local, so files from a session on another login node are gone.
- **12:03 Request:** compare with the earlier, successful runs in bash history. **Insight:**
  - **Options:** identical, apart from `-progress`.
  - **How they ran:** the earlier runs were interactive in `zellij` and wrote fscratch → project1 at about
    0.28 GB/s. This one wrote fscratch → fscratch at about 4 GB/s.
  - **Memory:** mksquashfs's default `-mem` (146497M, 25% of RAM) equals the 143 GiB per-user cgroup limit of
    login nodes, so the kill was most likely out of memory. Not confirmed: the node's cgroup is not visible from
    another login node.
- **12:13 Request:** limit the working memory and run in the background. **12:16 Change:** `mksquashfs.sh` (next to
  the images in `$FSCRATCH/graphcast/data/anomalies/squashfs/`) adds `-mem 32G` and restores `-progress`. It
  relaunched on `jpbl-s02-04` with `nohup setsid`, packing `zarr_v3_lexi`, `zarr_v3_colexi` and `zarr_v2` in turn.
- **12:21 Request:** keep this logbook of requests, changes, experiments and insights, with date and time, as a
  project-wide record to be merged later with earlier sessions.
- **12:27 Request:** keep temporary files in `$PROJECT`, so that work can be resumed on another login node.
  **Change:** CLAUDE.md sets `$PROJECT/claude-work/<branch or task>/` for them; the scripts of this task moved
  there.
- **12:30 Request:** name that directory after the session, which outlives branch and task renames. **Change:**
  `$PROJECT/claude-work/$CLAUDE_CODE_SESSION_ID/`, which stays the same when a session is resumed. This session's
  files are in `/e/project1/e-dev-2026d09-119/claude-work/5ec59923-bcb3-483e-a402-cd793d0e8df5/`.
- **12:29–12:44 Experiment:** `zarr_v3_lexi.sqsh` took 13 min and `zarr_v3_colexi.sqsh` 14 min (5.1 TB each), with
  entry counts matching their sources; with `-mem 32G` memory stayed at about 18 GB. `zarr_v2.sqsh` (2.5 M files) was
  still being packed at 14:51.

## 2026-10-28: dataloader and training on the standardized store (PR #1, widened)

- **14:09 Request:** the PR should now also ship the refactored dataloader, training scripts, related files and
  configs, working with the precomputed store. Update the PR title and description, and run the tests in the image.
  **14:13 Decisions:** read the store with xarray-tensorstore, but keep `xarray.open_dataset` for `tendency_std` and
  the mask (small, loaded once); remove the old on-the-fly path; verify with tests plus a short training run.
- **14:25 Request:** add a downloader command, with its own CLI and config, that writes the mask (e.g.
  `glorys_mask`) at a configurable level to a Zarr store as a single DataArray, loaded as is by training (no level
  selection at training time).
- **Change** (uncommitted until the checks passed):
  - **Mask downloader:** `graphcast download-mask` (`downloader.get_mask`, `MaskDownloadConfig`, `configs/download/mask.toml`;
    `download.sh` and `download.slurm` take `--mask`).
  - **Dataloader:** `dataloader.StandardizedAnomalySource` replaces `ARCOTimeseriesSource`. It reopens the store
    lazily in each process, and `__getstate__` drops it.
  - **Shared helpers:** `get_tensorstore_context` and `select_levels` move to `dataloader.py`, and
    `climatology_index` moves to `downloader.py`.
  - **Data access:** `dataloader_utils.get_mask(MaskConfig)` replaces `get_surface_mask`; `get_climatology` and
    `get_anomaly_std` are removed.
  - **Configs:** `TrainingConfig` has `dataset: StoreConfig(path, io_threads)` and `MaskConfig(path, fill_value)`,
    and no `climatology` or `anomaly_std` sections.
  - **Scripts and benchmarks:** `train.sh` binds DATASET, MASK and TENDENCY_STD. `.env` and JUBE point at the new
    stores; the JUBE format sweep has 12 cases, including colexicographic and SquashFS images.
- **Experiment:** the tests grew to 87, all passing in the image. Two insights from writing them:
  - Grain worker processes need `absl.flags.FLAGS.mark_as_parsed()`, as the CLI does.
  - Batches from workers hold shared-memory arrays, which xarray operations reject (`xarray_jax.unwrap` gets the
    array).
- **14:43 Experiment:** `download-mask` wrote `$FSCRATCH/graphcast/data/anomalies/glorys_mask` from the local raw
  copy in 9 s (7.5 KB). It equals the old `get_surface_mask` result: 693,039 ocean points of 721 × 1440.
- **Experiment:** three training samples (inputs, targets, forcings after extraction) were built with the old
  pipeline, from `git archive HEAD` on the raw local copies, and with the new pipeline on `zarr_v2`. They are
  bit-identical (112 arrays), and the new samples took 0.08–0.16 s each on a login node.
- **14:45–14:49 Experiment:** job 2561874 (`training2647`, develbooster, 1 node, 4 GPUs) ran 60 steps of
  `demo_5000.toml` on the zarr_v2 store with 8 Grain workers. Results:
  - compilation and the memory report (47.1 GiB per GPU) worked, and so did the checkpoints and saving the model;
  - the loss fell from 0.49 (steps 0–4) to 0.14 (steps 55–59);
  - from step 8 on, the data wait was 0.022 s (median) and compute 0.394 s;
  - steps 1–7 waited about 3.1 s each while the Grain workers started, against 6–8 s with the old dataloader
    (`results/2026-10-22-final-v2/io-10levels`, zarr_v2, 8×2).
- **14:52 Change** `96e2fb9`: the above, committed and pushed. PR #1 was renamed "Train on precomputed standardized
  anomalies: downloader, mask, dataloader and training", with a new description.
- **15:04 Request:** add a command, with configs, that recomputes `tendency_std` from the standardized store,
  mimicking arco-make and following the arco-ocean configs. The copy in use (also on S3, to be deleted there) was
  expensive to compute and is kept.
- **Insight:**
  - **The copy:** its attributes describe the standard deviation of the first-order time differences of the z-score
    anomalies, assuming zero mean, over 1993–2023. That is arco-make's `diff_std` processing step (its
    `compute_climatology` computes the climatology and anomaly_std, with a 365-day calendar, 29-day Gaussian
    window). The copy has 27 static variables, float32, lz4 Blosc, one chunk per depth level.
  - **Missing values:** the copy is NaN wherever a variable has no data (land; 66% of sea points for `siconc`, 8%
    for `swh`) and never 0.
- **Change:**
  - **The command:** `graphcast compute-tendency-std STANDARDIZED OUTPUT` (`downloader.tendency_std`,
    `TendencyStdConfig`, `configs/download/tendency_std.toml`; `download.sh`/`download.slurm --tendency-std`).
  - **Computation:** float64 squares and Dask chunks of 256 days of one depth level, since chunks along time alone
    would hold all 50 levels (about 13 GB). A std of 0 is written as NaN (`zero_as_missing`), as in the copy.
  - **Tests:** 97, all passing in the image.
- **15:09 Experiment:** 5 variables recomputed from `zarr_v2` (235 GB in 85 s on a login node) against the copy:
  - **NaN patterns** are equal.
  - **`zos`, `2t` and `so_bottom`:** median relative difference 3–4e-4, with a 99th percentile below 3e-3. The likely
    cause is Feb 29: arco-make dropped it, while the store keeps it.
  - **`siconc` and `swh`:** the median is the same, but the 99th percentile is 0.5–0.7, at points missing only part
    of the time, where the store's 0 filling enters the differences.
- **15:23 Experiment:** the SquashFS run (`-mem 32G`, `jpbl-s02-04`) finished with no OOM kill:

  | Image | Time | Size | Entries (source = image) |
  |---|---|---|---|
  | `zarr_v3_lexi.sqsh` | 13 min | 5.55 TB | 25,891 |
  | `zarr_v3_colexi.sqsh` | 14 min | 5.55 TB | 25,891 |
  | `zarr_v2.sqsh` | 159 min | 5.52 TB | 2,525,068 |

  **Insight:** the file count, not the size, sets the time (v2 took 11× longer). Windows read through each image,
  bound as `scripts/train.sh` binds it, are identical to those of the directory store (3 windows each, 0.1–0.2 s
  on a login node); the `.env` comments list the images as DATASET alternatives.
- **15:31 Request:** explain the NaN differences between the original and the recomputed `tendency_std`.
  **Insight** (correcting 15:09): the large `siconc`/`swh` differences are not points missing part of the time.
  They have data every day, but the anomaly_std is 0 on 40–48% of days, where arco-make's z is 0/0 (skipped) and the
  store's is 0 (counted). The 268 `swh` NaN mismatches have data only on 1993-01-01, while their climatology is NaN
  on every day.
- **15:52 Request:** all variables will be used in training: research the differences across all of them,
  including `thetao`, `so`, `uo` and `vo`, and report in detail. **15:54–16:30 Experiment:** job 2563819
  (`training2647`, develbooster) recomputed `tendency_std` for all 27 variables in 34.6 min, into
  `$FSCRATCH/graphcast/data/anomalies/tendency_std_recomputed`.
- **16:00 Request:** inspect arco-make and the ARCO-OCEAN settings to find why `swh` has data only on some dates.
  **Insight:**
  - **WAVERYS pipeline:** 3-hourly instantaneous fields, regridded conservatively (`skipna`, `nan_threshold = 1.0`)
    and resampled to daily means (`skipna`).
  - **Static mask:** `masked_fill` fills NaN with 0 only inside `waverys_mask` (bathymetry > 0, dataset version
    202311; the dynamic fields are 202411). So the wave variables are 0, not NaN, under sea ice.
  - **NaN pattern over time:** from 1993-01-02 to 2012-12-31 the NaN points are exactly the 393,555 static-land
    points. Outside the mask there is data on 1993-01-01 (319 points, 48–80°N, median 0.09 m) and from 2013-01-01
    (50 points every day, 32 intermittently, coastal). The same holds for `mwp`, `mwd`, `usd` and `vsd`.
  - **Unconfirmed:** the 3-hourly source was not inspected (it needs a CMEMS download).
- **16:30–16:42 Experiment and Insight:** comparison over all variables and levels. Every difference has one of
  four causes (details in CLAUDE.md, "tendency_std"):
  1. **Feb 29:** dropped by arco-make; without it the recomputation matches within 1e-7.
  2. **Zero anomaly_std days:** sea-ice, wave, discharge, some deep `so` points and `lsm`.
  3. **Constant series:** `z` and `lsm` are constant ERA5 fields, so the copy is 0 and the recomputation NaN. Their
     standardized values are float32 rounding noise.
  4. **Statistics undefined at points with data on some days:** the wave points outside the static mask.

  `thetao`, `so`, `uo`, `vo`, `zos` and most forcings have only cause 1.
- **17:46 Request** (decisions on the tendency_std differences):
  - keep Feb 29, standardized with Feb 28's statistics;
  - `z` and `lsm` are static, min-max normalized;
  - waves under sea ice are 0;
  - values equal to the climatology have a z-score of 0;
  - constant series have tendency_std 0;
  - the sea-ice differences are zero-inflated;
  - data outside their source's static mask are 0.

  Is anything still missing? **Answer:** a tolerance for "equal", small anomaly_std, the sea-ice estimator, the
  variable-to-mask mapping, `lsm` not exactly constant, what tendency_std 0 means in training, the loss mask, data
  outside 1993–2023, and normalization over the held-out years.
- **18:11 Request** (decisions): the relative tolerance; heavy tails kept; std of the non-zero differences for the
  sea-ice variables only; the proposed mask mapping; `z` and `lsm` are constant (variations are upstream bugs);
  exclude zero-tendency points from the loss per variable, without clipping; a floor on anomaly_std; leakage
  accepted; recomputing everything is acceptable. Revise the notes and check what is still missing.
- **Experiment and Insight:**
  - **`z`** changes only on 2022-10-01, at the polar rows (lat ±90, all longitudes, about 6×).
  - **`lsm`** differs from its first day on 1998-05-04 and 2014-04-05 only, at 532 points (18–24°S).
  - **`uo`/`vo` at level 49** have an anomaly_std of 0 at all 16,664 sea points.
  - **Smallest positive anomaly_std inside the masks:** `siconc` 4e-12, `dis24` 1.5e-11, `tp` 1.5e-8.
- **18:18 Request:** doesn't the current implementation exclude land below the surface? **Insight:** no. `Mask`
  and the loss broadcast the 2D surface `glorys_mask` over all levels and variables, so land below the surface of
  ocean columns is in the loss and is predicted.
- **18:52 Request** (decisions):
  - floor at the smallest in-sample anomaly_std per variable, and a |z| cap for new data;
  - the targets are the ocean variables;
  - excluded points keep their input values;
  - check that static fields are already 0 outside their masks;
  - no minimum number of non-zero differences for the sea-ice estimator;
  - GLORYS data outside the GLORYS mask are irrelevant;
  - the tolerance multiplier k = 4.

  Questions: what would sigma coordinates need? Run an experiment without output normalization, and what are its
  cons?
- **Insight** (static fields): `waverys_deptho` is 0 outside its mask. `glorys_deptho` is 0 or NaN there (3 points
  ≤ 1.3e-7), and `uparea` is NaN; both become 0 in the store after min-max.
- **Insight:** the residual connection has been disabled since commit 21edee5 (27 Mar 2026).
  `InputsAndResiduals` tests `_residual_locations is not None`, which is always `None`, so the model predicts the
  standardized targets directly and `tendency_std` is unused. Upstream GraphCast always predicts residuals.
- **Experiment** (setup): an experiment copy of `src` in claude-work (`exp_src`). There,
  `GRAPHCAST_EXP_RESIDUALS` selects `direct`, `residual_unit` or `residual_std`, and the state error (z-score
  units) of the predictions, of persistence and of the climatology is logged at every step. The three 2000-step
  runs are jobs 2566428, 2566430 and 2566431 on booster with `training2647`. The develbooster nodes are in the
  `maint_row_6.1` maintenance reservation from 28 Oct 17:00 to 29 Nov, so they run on the regular partition.
- **19:12 Request:** keep the sigma-coordinate notes for later; keep z-levels and mask the loss for now; wait for
  the experiments; fix the residual bug in the PR; recap the requested changes to input and output normalization.
  **Change:** `InputsAndResiduals` branches on `_residual_scales`, so training uses residuals divided by
  `tendency_std` again. `tests/test_normalization.py` fails on the old code. The sigma-coordinate requirements are
  in CLAUDE.md.
- **19:20 Experiment** (jobs 2566428, 2566430, 2566431, 18.6 min each): the state error divided by persistence,
  over steps 1900–2000.
  - **Overall:** direct 0.56, residual unscaled 0.51, residual / tendency_std 0.53. Climatology is 9.5× worse than
    persistence.
  - **Per variable, direct prediction** does not beat persistence for `thetao` (1.12) and `so` (1.18).
  - **Residuals** do: 0.88 / 0.96 unscaled, 0.83 / 0.92 normalized. The unscaled residual is best on `uo`/`vo`
    (0.43), the normalized one on `zos`, `thetao` and `so`.
- **19:31 Request:** implement the requested normalization and loss-masking changes. **Change** `5b7026c`:
  - **Configs:** `DownloadConfig` has `masks`, `static_variables`, `equal_tolerance` and `reference_period`;
    `TendencyStdConfig` has `zero_inflated`, and `zero_as_missing` is gone.
  - **Training:** `get_tendency_std` is strict, and `get_loss_masks` (grid mask and `tendency_std > 0`) is new.
  - **Loss and predictions:** losses use per-variable masked means per level; residual divisors are 1 where
    tendency_std is 0; `Mask` is per variable and keeps the input at excluded points.
  - **Tests:** 121 pass.

  **Experiment:** a real 5-day subset with `zarr_v2.toml` passes all checks: finite values, zero outside the masks,
  static `z` and `lsm`, and floors computed in 136 s.

  **Insight:** `$FSCRATCH` holds 42 TB (`du`) against a 40 TB soft limit (44 TB hard); `jutil` reports 0. The new
  stores need old ones deleted first.
- **20:07 Request:** delete the old stores and images and recompute. **Change:** the old stores, SquashFS images and
  tendency_std were deleted, and the old `zarr_v2` (2.5 M files) was removed in the background. `.env` and JUBE now
  read `tendency_std` from `$FSCRATCH/graphcast/data/anomalies/tendency_std` (`8740ccf`).
- **20:11–22:49 Experiment:** jobs 2567233–2567236 (`e-dev-2026d09-119`) wrote `zarr_v2` in 2 h 4 min,
  `zarr_v3_lexi` in 54 min and `zarr_v3_colexi` in 60 min; then `tendency_std` from `zarr_v2` in 31 min. A
  detached watcher packs the SquashFS images: v3 lexi and colexi in 13 min each (entries match).
- **Verification:**
  - **Stores:** the three are identical to an in-memory standardization of two windows (including Feb 29, 2004),
    with static `z` and `lsm`.
  - **`tendency_std`:** 25 variables, finite, 0 outside the masks; `siconc` is 0 at the 66% of sea points that
    never see ice.
- **Training check:**
  - **First runs:** jobs 2568564 and 2568639 failed before training. The masked mean summed in bfloat16, which
    xarray rejects (`213fa7d`, sums in float32), and then returned float32, while `Bfloat16Cast` expects bfloat16
    (`70e07cd`, returned in the loss's dtype).
  - **Job 2568724** (60 steps): the loss went from 0.73 to 0.49, every target variable by 30–40%. Data wait was
    0.021 s and compute 0.385 s per step; XLA memory was 49.2 GiB per GPU.
- **2026-10-29 00:50 Experiment:** the `zarr_v2` SquashFS image (5.44 TB, 2,502,289 entries, matching the store)
  took 2 h 35 min. All three images of the recomputed stores are in `$FSCRATCH/graphcast/data/anomalies/squashfs`.
  The old `zarr_v2` (`zarr_v2.deleting`) was still being removed.

## 2026-10-29: review of PR #1

- **07:12 Review** (`stefanocampanella`), **08:13 Request** to address it. The review asked to:
  - comment the tests;
  - split input and output normalization into two nested predictors, and revise the log messages;
  - have `get_mask` return the grid arrays, under a new name;
  - rename `mask_da`;
  - check whether the `valid_dates` paths are ever reached;
  - explain the broadcasting in `get_loss_masks`.

  **Change:**
  - **Normalization:** `InputNormalization` and `OutputNormalization` replace `InputsAndResiduals`, and training
    wraps only the latter.
  - **Grid:** `get_grid` returns a `Grid(lat, lon, mask)`; `get_loss_masks` checks the coordinates and comments the
    broadcasting.
  - **Names:** `loss_masks` replaces `mask_da`.
  - **Dead code:** `valid_dates` is removed; only its own test used it.
  - **Tests:** every test has a docstring, and the normalization tests are new (128 pass).

## 2026-10-29: variables and all-levels benchmarks

- **08:20 Request:** review what JUBE benchmarks after the dataloader change and what can be dropped, given that only
  all 50 levels matter. Then: implement the changes, rerun the benchmarks and update CLAUDE.md, and first put all
  variables in the training configs. After revision:
  - waves are left for a later PR;
  - the ocean variables and sea ice are targets;
  - river discharge is a forcing;
  - static fields are inputs;
  - `thetao_bottom` and `so_bottom` are not used;
  - surface loss weights follow option A.
- **Insight (loss weights):** level weights sum to 1 in this fork (upstream: mean 1), so a volume variable's loss is
  about 1/50 of a surface variable's at 50 levels. With option A, surface weights of 0.02 make each surface variable
  count as one volume variable. Option B (weights 1, surface variables dominating) was not chosen.
- **09:00 Change:** in `full.toml` and `demo_5000.toml`:
  - inputs: the 9 targets, the static fields (`glorys_deptho`, `glorys_mask`, `glofas_mask`, `uparea`, `z`, `lsm`)
    and the year progress;
  - targets: `zos`, `thetao`, `so`, `uo`, `vo`, `siconc`, `sithick`, `usi`, `vsi`;
  - forcings: the 8 ERA5 variables, `dis24` and the year progress;
  - `levels = "all"`.

  All ERA5 variables of the store are used. The unused store variables are the waves, `waverys_*` and
  `*_bottom`. Every target has a `tendency_std`, and static inputs carry no time dimension (test updated, 128
  pass).
- **09:10 Change:** JUBE runs everything at all levels.
  - **Case string:** `format:workers:threads:io_threads:nodes:cpus`, with baseline `zarr_v2:4:2:16:1:72`.
  - **Sweeps:**
    - `loader`: 6 layouts × 0x4/1x2/2x2/4x2/8x2;
    - `io_threads`: 8/16/32;
    - `cpus`: 8/72;
    - `scaling`: 1/2/4 nodes;
    - `memfit`: run as scan, scan + offloading with 0 workers, and scan + edge chunks.
  - **Dropped:** the 10-level, zarr pool/concurrency, XLA (`opt`) and `model` sweeps and their tags.
  - **Renamed:** `squashfs` is now `squashfs_v2`.
  - **Fix:** jobs now use the `src` of the checkout JUBE runs from. The shared `.env` pointed them at the main
    checkout.
- **09:15 Experiment:**
  - **Training check:** `develbooster` is in maintenance until 29 Nov 2026 (`maint_row_6.1`), so the 50-level check
    (60 steps, 4 Grain workers) was resubmitted as job 2574217 in the normal queue.
  - **Benchmarks:** `run_final.sh 2026-10-29 alllevels` submitted 76 jobs (`long`, 1024 steps) into
    `results/2026-10-29-alllevels`.
  - **Pending:** memfit is to follow once these finish, so that its reads do not perturb the loader measurements.
- **Request:** level averaging should use `depth`, not the `level` coordinate.

  **Finding:** training already weighted levels by depth: `GraphCast.loss_and_predictions` passed
  `levels_normalization_coord="depth"`, and samples carry `depth` along `level`. A check on levels 0, 10, 20, 30, 40
  and 49 of `zarr_v2` gave depth weights 0.218, 0.218, 0.215, 0.202, 0.137, 0.01, against 0.328 to 0.01 by level
  position. The defaults of `losses.py` were `"level"`, though.

  **Change:**
  - `losses.py`: `depth` is now the default of `normalized_level_weights` and the loss functions.
  - `model.py`: the explicit argument is dropped.
  - Tests: the loss tests use the default; a new test checks that the weights are linear in depth.
- **13:15 Experiment:**
  - **Review check** (job 2573884, 10 levels, 60 steps) passed with the same losses as job 2568724 before the
    refactoring (0.732 → 0.493). Compute was 0.387 s per step (median) and data 0.021 s.
  - **50-level check** (job 2574217) failed at model construction: `GraphCast` refused inputs that are neither
    targets nor forcings ("Input variables should be either predicted or forced.").

  **Change** (`547e098`): the check is removed. Rollouts already treat such inputs as constant and check that they
  have no time dimension (`autoregressive._get_and_validate_constant_inputs`), as upstream GraphCast does.

  **Check:** on a real sample, the static inputs have a batch dimension and no time dimension; `glorys_mask` adds
  one channel per level. At 3 levels this gives 27 input and 11 forcing channels. The check was resubmitted as job
  2576513. The branch is pushed, and the five review comments have replies.
- **13:30 Experiment:** the 50-level check (job 2576513; `full.toml` variables, 60 steps, 1 node, 4 Grain workers x
  2 threads) trains.
  - **Loss:** 0.259 → 0.146 (−44%).
    - Sea ice fell 43–75%, `zos` 26%, and `thetao`, `so`, `uo`, `vo` 33–35%.
    - Volume variables report about 0.03 each (level-weighted mean) and surface variables 1.0–2.1.
    - With option A, surface variables contribute 0.02 × 7.05 = 0.14 and volume variables 0.12: balanced.
  - **Timings:** data 0.041 s, compute 0.397 s per step (median, after step 0). Compute is only 3% above 10 levels:
    the mesh dominates.
  - **Memory:** XLA plans 54.2 GiB per GPU. MaxRSS is 18 GB per task (`sacct`, without Grain's shared memory).
- **13:35 Experiment:** the benchmark jobs started at 13:09, minutes before the static-input fix was in `src/`, so 41
  of them failed (37 `loader`, 4 `scaling`). The 35 that started from 13:13 ran with the fix: 30 completed and 5
  were still running. The 41 failed workpackages were resubmitted from their work directories, with the failed logs
  moved to `attempt1/`.
- **14:40 Experiment:** the loader benchmarks finished (`results/2026-10-29-alllevels`).
  - **8 × 2:** all 12 jobs (6 layouts × 2 repetitions) filled the node's 858 GB and hung at step 4. They were
    cancelled after 1–1.5 h.
  - **All the rest are compute bound** at 0.48–0.55 s per step: 0 × 4 to 4 × 2, I/O threads 8/16/32, and 1/2/4 nodes
    (linear scaling). 8 CPUs per task are 24 % slower.
  - **Insight:** the shared `.env` still had 8 workers, which hangs at 50 levels (now the `full.toml` default).

  **Change:** `.env` now uses 1 worker × 2 threads. Its comment gives the sample size (~5.7 GB) and the measured
  node memory.
- **15:06 Experiment:** memfit at 50 levels (`results/2026-10-29-memfit-alllevels`, 150 jobs: 103 fit, 44 GPU OOM, 3
  pinned-host OOM) shows the same limits as on 25 Oct:
  - defaults up to 50 km × 16 and 40 km × 8;
  - offloading adds 40 km × 14;
  - edge chunks add 35 km × 12.
- **Request:** report the findings for the current code, as a reference for the production training run. Also update
  the slides.

  **Change:**
  - **Report:** `REPORT.md` in the results directory, a copy with the CSVs in
    `docs/jupiter-hackathon/results/{benchmark,memfit}-2026-10-29`, and a shareable page
    (https://claude.ai/artifact/QaPPanmobJWisuDa8MN8ZR).
  - **Slides:** a new stack with three slides (standardized store and variables, loader, GPU memory). The
    recommended configuration is updated, the old 50-level slide is marked as the previous dataloader, and the next
    steps are revised.

## 2026-10-29: validation loss and mesh study design

- **15:15 Request:** make `zarr_v3_lexi` the default dataset, and run a mesh study varying both the mesh and the
  number of message passing steps.

  **Insight:** training logs only the training loss. Ranking models by it favours larger models even when they merely
  fit the training years, so the study needs a held-out loss. The test loss had been removed earlier: its stream had
  its own Grain workers (host memory) and was computed alongside training (compute).
- **Change:**
  - **`.env`:** `DATASET` is now `zarr_v3_lexi`. `zarr_v2` and its SquashFS image are kept.
  - **Config:** an optional `[validation]` section with `interval_steps` (250), `num_samples` (128) and
    `read_threads` (1).
  - **Windows** (`dataloader_utils.get_validation_dataset`): evenly spaced after `split_date`, in order, and the
    same for every seed and mesh. They are read in the training process, with no Grain workers, only during
    validation, with a prefetch buffer of 1.
  - **Evaluation:** `eval_step` is the training loss without gradients. Its results are logged as `valid/*` in
    TensorBoard and as "Step N validation" lines.
  - **Tests:** windows held out, evenly spaced and reproducible; errors (132 tests pass).
- **15:43 Experiment:** check job 2579554 (50 levels, 1 node, 100 steps, `zarr_v3_lexi`, 1 Grain worker,
  validation of 16 windows every 20 steps, i.e. 4 per GPU).
  - **Validation loss:** 0.158 → 0.147.
  - **Validation time:** 1.2–1.3 s, plus 17 s at the first one, mostly compilation.
  - **Training steps unaffected:** data 0.041 s (max 0.064 s, no spike after a validation), compute 0.411 s.
  - **Projected overhead:** about 1% of run time at 250-step intervals (128 windows on 8 nodes are also 4 per GPU).
  - **Memory:** XLA planned 55.2 GiB per GPU for the training step. Host memory was not recorded (`sacct` MaxRSS 0).
