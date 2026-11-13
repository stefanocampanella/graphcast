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
- **16:20 Request:** run the mesh study on `training2647` with two seeds.
  - **Budget:** 619,913 core-hours (2,152 node-hours) per `jutil`, last updated before 27 Oct, minus 45 node-hours
    used since, leaves ~2,107.
  - **Usage share:** 26% of the quota so far (262,081 core-hours), other project members 12.5%.
- **16:25 Experiment:** `mesh_study.sh --account training2647 mesh_msg 42,43 ...` submitted 30 jobs (2580169–2580199)
  at commit `bbad6cb`.
  - **Grid:** meshes coarse, 85 km, 60 km, 50 km and 40 km × 8, 12 and 16 message passing steps × seeds 42 and 43.
  - **Setup:** `demo_5000` on 8 nodes, 50 levels, `zarr_v3_lexi`, 1 Grain worker, validation of 128 windows every
    250 steps.
  - **Fitting 40 km:** with 12 and 16 steps it uses edge chunks of 262144.
  - **Time limits:** 1.3 × the expected time plus 20 min (1:30 to 6:00).
  - **Estimated cost:** ~350 node-hours.
  - **Outputs:** configs in `runs/mesh_msg/configs`, TensorBoard in `$DATA_ROOT/tensorboard/mesh_msg/`.
- **2026-10-30 02:14 Experiment:** all 30 mesh study runs completed (21:43–02:14, 384 node-hours, no errors).
  - **Validation loss at step 5000** (mean of seeds):
    - coarse: 0.0907 / 0.0893 / 0.0884 with 8 / 12 / 16 steps;
    - 85 km: 0.0887 / 0.0867 / 0.0858;
    - 60 km: 0.0864 / 0.0842 / 0.0829;
    - 50 km: 0.0851 / 0.0829 / 0.0818;
    - 40 km: 0.0842 / 0.0820 / 0.0805.
  - **Noise:** the two seeds differ by 0.03–0.57 %.
  - **Overfitting:** none; the minimum is at the last step everywhere, and the validation–training gap is
    0.010–0.012 in every case.
- **Insight:**
  - **Neither axis has saturated:** 12 → 16 steps gains 1.0–1.8 %, and 50 → 40 km gains 1.0–1.6 %.
  - **`zos` follows the processor's reach** (steps × spacing): +9.6 % at 40 km × 8, +0.6 % at 40 km × 16, −8.1 % at
    coarse × 16. This confirms the hypothesis of the hackathon study.
  - **Other fields:** they gain with resolution (40 km × 16: sea-ice thickness −28 %, currents −14 %, salinity
    −12 %, temperature −10 %).
  - **Value:** 50 km × 16 is −9.8 % at 1.40 s per step without edge chunks. 40 km × 16 is −11.2 % at 3.10 s with
    them.
- **Change:**
  - **Reports:** `runs/mesh_msg/REPORT.md` (copied to `docs/jupiter-hackathon/results/mesh_msg-2026-10-29`). The
    benchmark report's mesh caveat is replaced by these results, and the shareable report has a new section.
  - **Slides:** two new slides (the validation grid; sea surface height and reach), and the next steps are updated.
- **Request:** report the findings on ARCO-OCEAN and its statistics in `docs`, for future revisions of arco-make and
  the dataset.

  **Change:** `docs/arco-ocean-findings.md` gives 14 findings, each with the affected variables, the evidence, the
  impact and a recommendation:
  - calendar: Feb 29;
  - zero anomaly_std;
  - static fields stored as time series, and their glitches;
  - wave mask versions and the zero fill under sea ice;
  - masks against data;
  - `tendency_std` NaN conventions and the sea-ice estimator;
  - reference period and small anomaly_std;
  - storage layout and S3 access.

  It also lists the conventions adopted here.
- **Request:** rename `configs/training/full.toml` to `production_run.toml`, set its defaults to the best
  configuration, and update every reference.

  **Change:**
  - **Defaults:** `production_run.toml` uses the 50 km mesh with 16 message passing steps (the best value of the mesh
    study; its comment names 40 km × 16 as the lower-loss alternative). It keeps 40,000 steps and validation.
  - **`demo_5000.toml`:** stays coarse × 8, as the cheap baseline that `mesh_study.sh` varies.
  - **JUBE:** it already substitutes mesh and message passing steps, so it still benchmarks coarse × 8. Only its
    config source and comments changed.
  - **References updated:** the test, CLAUDE.md, the JUBE README and scripts, the slides (recommended configuration
    and material links, with `zarr_v3_lexi`), the 29 Oct report and the shareable page.
  - **Unchanged:** the 25 Oct report and earlier logbook entries keep the old name, as records.
  - **Test fix:** the `[validation]` test now removes the section explicitly, since the shipped configs have one
    (132 tests pass).

## 2026-10-30: second review of PR #1, meshes

- **10:00 Review** (24 threads: 9 on code, 15 on `TODO.md`) and **Request** to address it.

  Also: find out why the fine mesh has more nodes than the 35 km mesh with nearly the same config, and name the
  coarse mesh by its size in km.
- **Change:**
  - **Grain helpers:** `_FlatSource` and `_unflatten_sample` moved to `dataloader_utils`, as private names.
  - **Parameter initialization:** it uses `get_init_sample`, the first training window with a batch of one. The
    iterator round trip `get_first_sample_and_reset` is removed.
  - **Batch transfer:** no wait in `next_batch_on_device`.
  - **Dead code:** `reshard_data` removed.
  - **Logging:** one setup and one `--log-level` option, in `cli_utils`.
  - **Rollout check:** restored as `_check_rollout_inputs`, which allows constant (static) inputs, with
    `tests/test_model.py`.
  - **Level weights:** `model.level_weights_coord` and `model.level_weights_decreasing` (defaults: depth,
    decreasing).
  - **`TODO.md`:** revised as the review asked, with a Done section and reference numbers for GPU memory.
- **Experiment:** check job 2593156 reran job 2576513 (`smoke_50lev`, `zarr_v2`, 4 Grain workers) with these
  changes.
  - **Total loss:** identical at 59 of 60 steps; step 3 differs by one bfloat16 unit, and later steps are identical
    again, so training is unchanged.
  - **Data wait:** 0.019 s per step (0.041 s before).
- **Insight** (meshes):
  - **Origin of the fine mesh:** `uniform_fine.msh` (704,078 nodes, 31 Jan 2026) stores a 25 km size field, and its
    median edge is 25.0 km. It was made from a config that was never committed: `uniform_fine.toml` asked for 30 km
    then and asks for 35 km now, the same as `uniform_35km.toml`. The difference in that file, `domain.sampling`, is
    ignored by the code.
  - **Node count:** 363,900 × (35/25)² ≈ 713,000, close to 704,078.
  - **Descriptions:** the fine and 120 km meshes store a stale description (150 km).
  - **Other meshes:** median edges match their targets (117.8, 34.9 and 30.0 km).
- **Change:**
  - **120 km mesh renamed:** `uniform_coarse` is now `uniform_120km`, both the config (description fixed) and the
    `.msh` file. A symlink `data/geometry/uniform_coarse.msh` is kept for the main checkout until the merge.
  - **Paths updated:** `demo_5000.toml`, JUBE, the `mesh_study.sh` example and two notebooks.
- **12:00 Request:** rename the fine mesh and replace `uniform_fine.toml`. Check that the descriptions of all mesh
  configs match their settings.

  **Change:**
  - **25 km config:** `uniform_25km.toml` (25 km uniform field; the ignored `domain.sampling` dropped) replaces
    `uniform_fine.toml`.
  - **Mesh file:** `uniform_fine.msh` is renamed `uniform_25km.msh`, with a symlink under the old name until the
    merge.
  - **Lambrechts descriptions:** the three `lambrechts*` configs said "a uniform field of 300 km for the interior and
    10 km for coastlines". They now describe their fields: the smaller of a shore-proximity field (30–200, 100–200,
    150–300 km) and a bathymetry field (25–500, 100–500, 150–500 km), with coastlines coarsened to 10 km.
  - **Other configs:** the uniform configs already matched their values.
  - **References:** JUBE (`fine` tag, memfit list), a notebook and CLAUDE.md.
- **Experiment:** `uniform_25km.toml` regenerated with today's code (1 min 43 s on a login node) gives 703,409
  nodes against the January file's 704,078 (−0.1 %), with the same edge lengths (p5/median/p95 20.6/25.0/26.9 km).
  The January file is kept: the experiments used it.
- **Request:** validate the mesh generator's settings with pydantic, as the training script does, with tests and the
  same quality checks.

  **Change:**
  - **Settings:** `mesh_config.py` (`MeshGenerationConfig`).
    - It covers the gmsh options (until now hard-coded behind a FIXME, with the same defaults), smoothness, the
      domain (a `.shp`), boundary coarsening, and the size fields as a union discriminated by `name`.
    - It checks that size_min ≤ size_max, field_min < field_max and q_low < q_high, that each field type appears at
      most once, and that `view_tag_name` differs from the field names.
  - **Command:** `make-ocean-mesh` reads its settings through `cli_utils.read_configs(schema=...)`, so errors become
    CLI errors before meshing. The meshing itself is `write_ocean_mesh`.
  - **Size fields:** `CompositeMeshSizeField` takes the validated configs.
  - **Configs:** the ignored `domain.sampling` was removed from the three `lambrechts*` configs (now rejected).
  - **Bug fix:** `read_mesh_data` swapped the description and license.
  - **Tests:** `tests/test_mesh_config.py` (37: shipped configs, invalid settings, the CLI end to end on a synthetic
    coastline). 171 tests pass; ruff check, ruff format and ty pass on the new modules.
- **Experiment:** `uniform_120km.toml` regenerated with the new code gives identical vertices and triangles.
- **Insight:**
  - **Duplicated mesh:** each field view after the first is appended with the whole mesh again, so `.msh` files hold
    it twice.
  - **Reader:** reading a second mesh in the same process fails (gmsh state).
- **Request:** fix the two mesh I/O bugs, with comments explaining the changes.

  **Experiment** (synthetic coastline, two fields):
  - **The writer:** with `PostProcessing.SaveMesh = 1`, every `gmsh.view.write` also writes the mesh, 3 copies for 3
    views. Writing the first view with the mesh and appending the others without it gives one copy: 99 KB against
    234 KB, with the same data read back. `gmsh.write` with `SaveMesh = 0` writes no mesh at all, and adds a view of
    seamsh's own (`tri`).
  - **The reader:** `gmsh.open` keeps the views of earlier files (6 were left after one read), and `read_mesh` takes
    the first view with the name. `gmsh.clear()` before opening fixes it.

  **Change:**
  - **Writer:** `write_ocean_mesh` sets `SaveMesh` per view (1 for the first, 0 for the appended ones).
  - **Reader:** `read_mesh` calls `gmsh.clear()` before `gmsh.open`.
  - **Tests:** two new tests (one `$Nodes` section with every view readable; three meshes read in sequence). Both
    fail on the previous code. 173 tests pass.
  - **Real data:** the existing two-copy files (120 km, 50 km, 120 km again) read correctly in one process.

## 2026-10-30: blocking operations, Orbax issue 2545

- **Request:** are logger calls blocking in the training loop, and with what effect?

  **Insight** (rank 0 of the 30 mesh study runs):
  - **TensorBoard logging** blocks: `.item()` of each metric, plus `tf.summary` on rank 0. It takes 8 ms per step
    (median, p99 13 ms), 0.3–1.7 % of a step.
  - **The loop is fully serial:** `block_until_ready`, then `push_checkpoint` (3–4 ms, including `.item()` of the
    metrics every step), TensorBoard and the log line. Then the next batch is fetched (37–39 ms).
  - **Host share** of a step: 9.9 % (120 km × 8), 3.7 % (50 km × 16), 1.6 % (40 km × 16).
- **Request:** exploratory analysis of disabling blocking operations.

  **Experiment** (`claude-work/.../exp_loop`, jobs 2603156/2603157, 1 node): four loop variants, each run twice on
  the same node.
  - **Variants:**
    - `baseline`: the current loop;
    - `no_tb`: without TensorBoard;
    - `no_sync`: no wait, and `push_checkpoint` only on save steps;
    - `deferred`: no wait, and the metrics of step n logged after step n+1 is dispatched, so the next batch is
      fetched while a step runs.
  - **Models:** 120 km × 8 (400 steps) and 50 km × 16 (300 steps).
  - **Measure:** throughput from step 50.
- **Request:** take into account that Orbax issue 2545 (saving under a global mesh) is fixed.

  **Insight:** the fix (Orbax commit 5a0bd47) was verified upstream with JAX 0.11.1 and Orbax 0.12.4, the versions in
  the image. Our workaround creates the `CheckpointManager` under a per-process mesh. It is also in
  `check_writable_paths` and `save_model`, and FIXMEs point to it in `push_checkpoint` and `pull_checkpoint`.
- **Experiment:**
  - **Single process:** in the image, the issue's reproduction and a `CheckpointManager` save/restore under a global
    mesh both work.
  - **2 nodes** (8 processes; `claude-work/.../orbax_check`, jobs 2603190 and 2603287): without the workaround, and
    with the batch mesh set globally for the whole run.
    - **Saving:** 60 steps saving every 20, about 0.4 s blocking per save.
    - **Restoring:** the best checkpoint was restored at the end. A second run restored step 40 and continued to 80.
    - **Result:** its losses equal the first run's bit for bit, shifted by one step (19 of 19), so parameters,
      optimizer state and the Grain iterator restore exactly.
- **Insight** (two existing bugs found):
  - **Restart off by one:** `train` resumes at `latest_step`, but the checkpoint of step N is saved after step N, so
    the first resumed step is labelled N while it is really step N+1. Step numbers in logs, TensorBoard and
    checkpoints are one behind after a restart, and the run makes one extra update. The learning-rate schedule
    follows the optimizer's count and is not shifted.
  - **Output path:** `graphcast train` declares `OUTPUT_PATH` as `click.Path(file_okay=False)`, but the model is
    written as a file there. Rerunning into an existing output fails even with `--overwrite`.
- **Insight** (conversions to host values):
  - `push_checkpoint` converts the metrics with `.item()` every step, but Orbax uses them only on save steps.
  - `TensorboardLogger.log` converts them on every process, but only process 0 writes.
  - The other numpy conversions are needed (the Grain worker gather, `save_model`'s format).
- **2026-10-31 00:32 Experiment:** loop variants (jobs 2603156/2603157, 1 node, two runs each on the same node).

  **Throughput** (s per step after warm-up):

  | Variant | 120 km × 8 | vs baseline | 50 km × 16 | vs baseline |
  |---|---|---|---|---|
  | `baseline` | 0.4487 / 0.4489 | — | 1.2680 / 1.2675 | — |
  | `no_tb` | 0.4376 / 0.4527 | −0.8 % | 1.2612 / 1.2720 | −0.1 % |
  | `no_sync` | 0.4430 / 0.4462 | −0.9 % | 1.2616 / 1.2659 | −0.3 % |
  | `deferred` | 0.4374 / 0.4381 | −2.5 % | 1.2349 / 1.2332 | −2.7 % |

  **Losses:** two runs of the same baseline agree exactly on only 63 % / 91 % of steps (relative difference up to
  5e-3: GPU non-determinism of the step). The variants differ from the baseline by no more (at most 2.8e-3), so they
  train equivalently.
- **Insight:**
  - **Logging and syncing alone are within run-to-run noise:** dropping TensorBoard logging, or the waits with
    TensorBoard still logging each step, saves at most about 1 %.
  - **The gain comes from overlap.** Deferring the logging of step n until step n+1 is dispatched lets the next batch
    be fetched and copied to the GPUs while a step runs.
  - **Per step** that saves 11 ms (120 km) and 34 ms (50 km), out of about 49 and 52 ms of host work. Part of it
    stays exposed on the short 120 km step, probably the copy of the 5.7 GB batch competing with compute.
  - **For the production run** (50 km × 16, 40,000 steps, about 15.6 h): about 25 minutes, or 3.4 node-hours on 8
    nodes.
  - **Cost:** the per-part timings (data, compute, checkpoint) can no longer be separated, and JUBE parses them.
- **Request:** is Orbax checkpointing preemptible in the current setup?

  **Experiment** (1 node, 120 km × 8, 100,000 steps, checkpoints only at step 0, 5-minute limit; job 2606545):
  - **The save:** at the time limit, JAX's notifier caught `SIGTERM`, all processes reached a preemption sync point
    at step 381, and Orbax saved it "due to preemption". It blocked 0.56 s, and the commit completed.
  - **After the save:** training continued a few steps, until the Grain workers, also hit by `SIGTERM`, died (exit
    code -15). The job ended as TIMEOUT.
  - **Restore** (job 2608100): `--continue` restored step 381. The 4 steps the preempted run made after its save are
    equal bit for bit to the continued run's, one label lower (the restart off-by-one).
- **Insight:** Orbax detects `SIGTERM` (`jax_enable_preemption_service` is on by default, and `scripts/train.sh`
  `exec`s Apptainer), so the `TODO.md` item is outdated.

  Gaps:
  - the loop does not stop after the preemption save;
  - Grain workers die on `SIGTERM`;
  - the save time at production scale is untested;
  - nothing resumes automatically.
- **Request:** prepare a guide for a new PR covering the bugs, the blocking operations and preemption.

  **Change:** `claude-work/5ec59923-bcb3-483e-a402-cd793d0e8df5/PR_GUIDE_training_loop.md` collects the findings,
  the proposed changes, a validation plan and references.
- **Request:** what is the impact of checkpointing every step, and what is common practice?

  **Insight:**
  - **Cost:** a save blocks for 0.4–0.6 s (222 MB, 51 files at 120 km × 8). Saving every step would make 120 km × 8
    about 2× slower and 50 km × 16 about 35–40 % slower; 500 steps cost under 0.3 %.
  - **Practice:** Young/Daly gives an interval of about √(2·C·MTBF), i.e. 5–13 min for 8-node failures every day to
    every week, so 500 steps (4–11 min) is reasonable. Common practice adds preemption saves (already on) and a clean
    exit.
  - **Bug:** with `best_fn` set, Orbax 0.12.4 keeps only the `max_to_keep` best checkpoints by `metrics[0]`, a single
    batch's training loss (`BestN`; `LatestN` is used only without `best_fn`). The latest checkpoint, including a
    preemption save, can be deleted, `--continue` can resume from an older one, and the final model is chosen by a
    noisy batch loss.

  **Change:** section 5 of the PR guide covers the interval, the retention bug and the fix: `LatestN` plus `BestN` on
  the validation loss.
- **Request:** held-out loss is costly and blocking: how can it score checkpoints?

  **Insight:** validation costs about 1 % today (1.2–1.3 s per 250 steps at 120 km × 8), about 0.7 % estimated at
  50 km × 16. The issues are the blocking, and that a score must exist at save time. Options:
  1. validate at save steps only (no new sync point; about 0.35 % at 500 steps);
  2. prefetch the validation windows in host memory, and dispatch the evaluation without waiting;
  3. spread one window every few steps (cheap, but a mixed-parameter score);
  4. a separate evaluator job on saved checkpoints (no training cost; allows rollout metrics);
  5. cheaper stand-in scores: the averaged training loss, or fewer windows.

  Recommended: 1 + 2 in the next PR, with `LatestN(2)` and `BestN(1)` on the validation loss; 4 later.

  **Change:** section 5.5 of the PR guide.

## 2026-11-03: permanent data copies in $PROJECT

- **Request:** fscratch is purged after a while, so keep one permanent copy in `$PROJECT` of:
  - the raw ARCO-OCEAN dataset, its climatology and anomaly_std;
  - the z-score store, in the fastest training layout (zarr v3 lexicographic).

  Delete the outdated copies in `$PROJECT` and `$PROJECT/graphcast/data`, after checking space with `jutil`.

  **Decisions (asked):** raw data in zarr v3 lexicographic; keep `tendency_std` and `glorys_mask`, but not the
  arco-make tendency_std nor `graphcast_backup`; location `$PROJECT/data`.
- **Insight:**
  - **Quotas** (`jutil`, 09:24): project1 14.3 TB of 20 TB, 2.88 M of 4 M inodes. fscratch reports 0, although its
    files exist.
  - **Purge policy** (JUPITER user documentation): `$FSCRATCH` purges files not modified or accessed for 30 days.
  - **Copies in `$PROJECT`:**
    - `zarr_v2`: raw v2, 4851 GiB, 2.69 M files, with consolidated metadata identical to the fscratch inputs;
    - `zarr_v3`: raw v3 lexicographic, 4841 GiB, 28 k files;
    - `squashfs`: images of the raw data, 4798 GiB;
    - `graphcast_backup`, and the 2025 `graphcast/data/{dataset,stats}`.
- **Experiment** (login node `jpbl-s02-03`): raw zarr v3 compared with raw zarr v2, in full (5 GiB/s, about 70
  min).
  - **Values:** 0 differing blocks out of 38,238 (dataset) and 1,242 each (climatology, anomaly_std).
  - **Structure:** the same, except `_FillValue` (an attribute in v2, array metadata in v3).
- **Change:**
  - **Deleted:**
    - `squashfs`, `graphcast_backup` and `graphcast/data/stats`;
    - `graphcast/data/dataset`, except `glorys12_static_res-0p0625_levels-10`, used by the `lambrechts*` mesh configs.
  - **Moved:** `zarr_v3/{dataset,climatology,anomaly_std}` to `data/arco-ocean/`.
  - **Copied:** `anomalies/{zarr_v3_lexi,tendency_std,glorys_mask}` from fscratch to `data/anomalies/`, with 8
    parallel `rsync` in 57 min (2.6 GiB/s).
    - **Check:** an empty dry run, the same 25,625 entries and 5,442,717,244,660 bytes, and identical windows.
  - **Deleted after that:** `zarr_v2` (14 min) and `zarr_v3/tendency_std`.
  - **End state:** `$PROJECT/data` holds 4840 + 5070 GiB, about 10.9 TB, and about 0.15 M inodes.
  - **Docs:** `$PROJECT/data/README.md` (provenance, checks, restore); CLAUDE.md and a `.env` comment updated.

## 2026-11-04: training loop PR (restart, checkpoints, preemption, determinism)

Session directory: `claude-work/875379cd-19a4-4f43-a956-bd19e2b1b6a6`. All jobs on `training2647`.

- **Request (12:00):** plan the training-loop PR from the guide of 31 Oct (`claude-work/5ec59923-.../PR_GUIDE_training_loop.md`).

  **Decisions (asked):**
  - keep the sequential loop and drop the `deferred` variant (2.5 % gain, harder logging, more code);
  - a speed-up must save at least 150 ms per step (36 ms, 30 min per 50,000 steps, is a bare minimum), and options
    that raise peak GPU memory stay opt-in;
  - investigate copying batches to the GPU inside the dataloader, and the source of non-determinism;
  - no separate evaluator job;
  - Grain workers stop gracefully, SIGTERM comes earlier, test on 8 nodes, no automatic resubmission;
  - branch from and PR into `develop` (not `develop@esiwace3_hackathon`).
- **Change:**
  - **Restart:** `--continue` resumes at latest + 1 (`training.first_step`).
  - **Output path:** `OUTPUT_PATH` is a file argument (`dir_okay=False`), so an existing model can be overwritten.
  - **Checkpoints:** the loop decides the saves, after step n when `(n + 1) % save_interval_steps == 0`, after the
    last step and at a preemption. Validation runs before the save, and its loss is the checkpoint's metric.
  - **Retention:** `[checkpoints]` has `keep_latest` (2) and `keep_best` (1), replacing `max_to_keep`. The
    `save_interval_steps` default is now 500, and it must be a multiple of `validation.interval_steps`. The final
    model is the best checkpoint by validation loss.
  - **Preemption:** the loop waits for the save and exits with status 75. Grain workers ignore SIGTERM, and the loop
    closes the iterator. `train.slurm` has `--signal=TERM@120`.
  - **Host values:** metrics are converted only when saved or logged (one `device_get`), and TensorBoard writes on
    process 0 only.
  - **Orbax:** the per-process meshes (Orbax issue 2545 workaround) are gone.
  - **Tests:** `tests/test_training.py` and a Grain SIGTERM test (180 tests pass).
- **Insight:** the Orbax 0.12.4 source answers three questions.
  - **Ranking:** `BestN` keeps the last n checkpoints of its sort, so losses need `reverse=True`.
  - **Unscored checkpoints:** it keeps checkpoints without metrics unless `keep_checkpoints_without_metrics=False`.
  - **Data movement:** a save copies one replica of each replicated array to the host (replica-parallel unless
    `JAX_PLATFORMS` names the GPU), inside the blocking part.
- **Insight:** in Grain 0.2.18, workers are stopped by `close()` (an event, a join, then SIGKILL). Without `close()`,
  Python's exit sends SIGTERM to daemon processes and joins without a timeout, so workers that ignore SIGTERM need the
  explicit `close()`.
- **Experiment B** (`exp_b`, jobs 2682860/2682861, 1 node, current code). Each variant ran twice: the baseline, a wait
  for the batch's host-to-device copy before the step (B1), and `--xla_gpu_deterministic_ops=true` (B2).
  - **B1:** waiting for the copy raises data time by 19 ms and lowers compute by 12–13 ms (120 km × 8: 0.443 →
    0.448 s, 50 km × 16: 1.277 → 1.283 s per step). The copy is about 19 ms and already partly overlapped, so moving
    it into the dataloader saves at most 20–30 ms. **No prototype** (bar: 150 ms).
  - **B2:** deterministic ops make a step take 8.2 s instead of 0.42 s at 120 km × 8 (68.2 GiB planned). At 50 km ×
    16 they need 81.5 GiB, and the run fails with OOM.
  - **Losses of repeated runs:** they differ at 7–10 of 150–200 steps, each time by exactly 2^-12. Each GPU's loss is
    a bfloat16 value (`Bfloat16Cast`), and one GPU's loss rounds to the neighbouring bfloat16 value.
- **Experiment** (`loop_tests/throughput.sbatch`, job 2683293, 1 node, 120 km × 8): new and old loop, alternating,
  twice each.
  - **Throughput:** the same within noise. The new loop took 0.444 and 0.422 s per step, the old one 0.427 and
    0.444 s; compute alternates between 0.40 and 0.42 s between runs.
  - **Checkpoint call:** 2 ms → 0 on steps that do not save.
  - **TensorBoard:** still 7–8 ms. A CPU microbenchmark gives 263 µs per `tf.summary.scalar` call (eager dispatch;
    the writes are queued), so the conversions were not the cost. Left as is (0.6 % at 50 km × 16).
- **Experiment** (`loop_tests/preempt.sbatch`, jobs 2683295 and 2683296, 1 node, 120 km × 8, 6-minute limit,
  `--signal=TERM@120`):
  - **First job:** SIGTERM arrived 120 s before the limit, and step 250 was saved (0.42 s blocking, 0.7 s with the
    commit). The run exited with status 75, without Grain tracebacks.
  - **Continuation:** it restored step 250, started at step 251, was preempted at step 489 and saved it. The latest
    two checkpoints were kept.
- **Experiment** (`loop_tests/preempt.sbatch`, jobs 2684368 and 2684369, 8 nodes, 50 km × 16, 8-minute limit):
  - **Save:** all 32 processes synced on step 150 within the second after SIGTERM. The save blocked 0.76 s (1.3 s with
    the commit); the checkpoint is 392 MB.
  - **Continuation:** it restored step 150, started at step 151, and saved step 314 at the next preemption (1.6 s).
- **Experiment** (`loop_tests/rc.sbatch`, job 2683621, 2 nodes, 120 km × 8, saves and validation every 20 steps):
  - **Runs:** two fresh 80-step runs, and a 60-step run continued to 80. Restore, `check_writable_paths` and
    `save_model` all ran without the per-process meshes, and the final model came from step 79.
  - **Checkpoints differ** (`compare_ckpt.py`): 220 of 226 parameter arrays differ between the two fresh runs (median
    ‖a−b‖/‖a‖ 2.6e-4), and the continued run differs from them by as much.
  - **Correction of 30 Oct:** bit-identical losses do not show that parameters restore exactly, since the losses are
    bfloat16 values.
- **Experiment** (`loop_tests/rc_det.sbatch`, job 2684366, 1 node, 120 km × 8, deterministic ops): two fresh 20-step
  runs, and a 10-step run continued to 20.
  - **Result:** all checkpoints (steps 9 and 19) are bit-identical, parameters and optimizer state.
  - **Insight:** the non-determinism comes from XLA's non-deterministic GPU operations (atomic scatter-adds and
    reductions). The restore is exact, and the continued run's steps line up with the uninterrupted run's.
- **Change:** CLAUDE.md (training loop, checkpoints, preemption, Orbax, measurements, development branch), the JUBE
  README (saves only after the last step), and `TODO.md` (three items closed). A JUBE `loader` dry run generates
  configs that validate.

## 2026-11-04 18:00: training loop PR, code review fixes

Session directory `$PROJECT/claude-work/875379cd-19a4-4f43-a956-bd19e2b1b6a6`.

- **Request:** the maintainer ran `/code-review` on the PR and asked to fix five of its findings.
- **Change:**
  - The loop checks for preemption after validation, so a SIGTERM during validation is acted on at that step, and a
    save step keeps its validation loss (before, a preemption on a validation step skipped validation).
  - The `try`/`finally` that closes the Grain iterator starts right after the iterator is created, so a failed restore
    does not leave workers that ignore SIGTERM.
  - A stale comment in `check_grain_num_workers`; the SIGTERM test signals only the iterator's own workers.
  - CLAUDE.md: checkpoint directories written before this PR (metrics as a list) cannot be continued.
- **Not changed:** exit status 75 under `srun --kill-on-bad-exit` (the checkpoint is committed on all processes before
  any exits), and the step line's checkpoint time (validation has its own line; the JUBE format stays).
- **Experiment** (`loop_tests/preempt.sbatch`, jobs 2688168 and 2688169, 1 node, 120 km × 8, validation every step,
  6-minute limit): the preemption saved step 85 with its validation loss (0.8 s) and exited with status 75; the
  continuation started at step 86 and saved step 167 at the next preemption, also with its validation loss.

## 2026-11-05 09:00: apptainer run vs exec, CUDA forward compatibility

- **Request:** the maintainer asked why tests ran with `apptainer exec` instead of `run` (which runs NVIDIA's init
  scripts), to repeat them, to explain CUDA forward compatibility in the image, and the `rm: cannot remove
  '/usr/local/cuda/compat/lib'` messages of training jobs.
- **Finding:** CLAUDE.md gave `exec`; `run` differs only by NVIDIA's entrypoint (banners and checks; on a node
  without GPUs it sets `NVIDIA_CPU_ONLY=1`). The tests pass with `run` on a login node and with `run --nv` on a GPU
  node (180 each).
- **Experiment** (job 2698725, `compat/compat.sbatch`, 1 node): kernel driver 610.57.04 (CUDA 13.3). `run --nv` loads
  the image's compat `libcuda.so.615.71.09` (13.4); `exec --nv` loads the host's `libcuda` (13.3); both run JAX. The
  `rm` messages come from `/etc/shinit_v2`, which cannot persist its once-per-host marker in the read-only image;
  harmless. Details in CLAUDE.md ("CUDA forward compatibility").
- **Change:** CLAUDE.md test command uses `run`.

## 2026-11-05 10:30: training loop PR, review comments

- **Request:** address the maintainer's five review comments on PR #2.
- **Change** (350b0c0): `keep_latest` defaults to 1 (Orbax 0.12.4 deletes older checkpoints only after the new one is
  committed, so one suffices to resume); `push_checkpoint`/`pull_checkpoint` renamed `save_checkpoint`/
  `restore_checkpoint`, `force=True` commented (it bypasses Orbax's `should_save`, not strictly needed with the
  default interval of 1); `check_writable_paths` parameters renamed; CLAUDE.md forward-compatibility notes scoped to
  JUPITER.
- **Open:** whether to keep periodic snapshots (`EveryNSteps`, e.g. every 5000 steps) for evaluation beyond the held-out
  loss; the top-k by validation loss would be the last k saves, since the minimum has always been at the last step.

## 2026-11-05 12:30: training loop PR, keep_best in the production run

- **Request:** keep 5 best checkpoints in the production run, to compare per-variable losses (one checkpoint may do
  better on `thetao`, another on `so`); the 5000-step mesh study is too short to say that the best ones will be the
  last saves of a 40,000-step run.
- **Change:** `production_run.toml` sets `keep_best = 5` (the default stays 1). This resolves the open item of the
  previous entry; periodic snapshots are not added.

## 2026-11-05 16:00: tests for the whole codebase, legacy port and report

- **Request:** port the inherited absltest modules to pytest without reading the implementation, find what breaks
  and why, report; then test the whole codebase, with coverage, ruff and ty as pre-commit hooks and a GitHub workflow
  (branch `feature/tests-and-code-quality`, session dir `claude-work/b0a574fd-35c0-4f21-bc1c-92264af27375`).
- **Changes:** commits authored by the maintainer with Claude as co-author (CLAUDE.md); dev group with pytest-cov,
  coverage, pre-commit, ruff 0.17.3 and ty 0.0.97 (pinned), coverage and ty settings in `pyproject.toml`; image
  rebuilt from the branch as `graphcast-new.sif` (not yet replacing `graphcast.sif`); the seven `*_test.py` modules
  ported to `tests/` (67 functions, 85 cases).
- **Findings** (`test_report.md` there): 76 pass, 8 fail, 1 error, none from the port. `merge_meshes` and
  `faces_to_edges` moved to `mesh_graph`; grid coordinates are WGS84 ECEF in metres and `get_mesh_to_grid_edges` takes
  a mask; jax 0.11 dropped `jax_pmap_shmap_merge`, and `xarray_jax.pmap`'s own dimension check is never reached. The
  radius-query smoke test passes with 0 edges (unit-sphere mesh, grid in metres). Latent bugs: `basis.py` calls an
  undefined `cli`; `mesh2grid_edge_normalization_factor` is a float but unpacked as a pair. Coverage 53 %, lowest on
  the model and training path (8–29 %). ty cannot resolve the image's editable jax install.

## 2026-11-05 18:30: plan for fixes and tests, legacy tests updated

- **Decisions** (maintainer, on the report): delete `climatology`; fix `sharding` to fit the current code (drop it if
  that needs large untested changes); keep and test `autoregressive`, `validation_utils`, `graph_pruning`,
  `plot_utils`; make ty see the image's jax through extra search paths in the pre-commit hook, not an image change; fix
  the `basis` draft; coverage gate 80 %; one PR, reviewed after each group of commits.
- **Insights:** `sharding.FullyShardedDataParallel` called the lifted `init` inside `shard_map` (leaking tracers) and
  asked for an rng in `apply`; lifting the shard-mapped `init` instead works (toy predictor on 4 simulated CPU devices:
  same loss, gradients equal). ty with `--extra-search-path` for `/opt/jax`, `/opt/jaxlibs/jaxlib` and `/opt/flax`
  resolves every import (170 → 128 diagnostics). `basis` interpolates a dense identity matrix (2 TB for the 25 km
  mesh). Edge direction features take `sin`/`cos` of azimuths converted to degrees twice (`pyproj.Geod.inv` already
  returns degrees), so they are meaningless for every model trained so far.
- **Changes (group 1):** legacy tests updated to the fork: `merge_meshes`/`faces_to_edges` tests moved to
  `tests/test_mesh_graph.py` with the current semantics; `mesh_connectivity` tests on an Earth-sized mesh against the
  WGS84 formulas, with brute-force checks of the radius query and containing faces, with and without a mask; the
  missing-coordinate test of `data_utils` names the missing coordinate.

## 2026-11-06 10:30: review of group 1, icosahedral multi-mesh on the ellipsoid

- **Review:** project the icosahedral mesh onto the ellipsoid (unit sphere → WGS84 Earth-centred coordinates) instead
  of scaling it to a sphere; fix the spatial reference systems of `mesh_graph.Graph` (the name "cartesian" instead of
  the WKT) and of the icosahedral meshes (labelled WGS84 though on the unit sphere); add a tested function that merges
  the hierarchy, projects it and returns a `MeshGraph`; investigate whether the model then runs on it.
- **Changes:** `icosahedral_mesh.get_multi_mesh_graph(splits, crs=cartesian_crs)`; icosahedral meshes carry the
  unit-sphere WKT; `merge_meshes` keeps (and requires a common) spatial reference system; `Graph` defaults to the WGS84
  WKT; the unused `radius` argument of `get_hierarchy_of_triangular_meshes_for_sphere` (a sphere scaling with a wrong
  WKT) removed; the connectivity tests use the projected multi-mesh.
- **Insights:** the unit sphere → WGS84 transform keeps each vertex's latitude and longitude (geodetic = spherical, no
  datum shift) and puts it on the ellipsoid (6357–6378 km). No other code converts the unit sphere to WGS84
  Earth-centred coordinates: seamsh meshes are written in them directly, `model_utils` uses the unit sphere only for
  node features (from latitude and longitude). Investigation (`claude-work/…/icosahedral_model.py`): a tiny GraphCast
  runs forward on the 642-vertex multi-mesh with a 5° masked grid, with and without scan, given a per-vertex mesh size
  (longest finest-mesh edge at the vertex: the multi-mesh has none). Gaps for experiments: training reads meshes only
  from `.msh` files (`MeshConfig.path`), which store triangles, not the multi-mesh's coarser edges; and `get_transform`
  warns about non-zero elevations of 1e-8–1e-7 m (absolute tolerance 1e-8, below float32 rounding).

## 2026-11-06 11:00: icosahedral multi-mesh file (experimental), GraphCast's mesh size and radius

- **Review:** use GraphCast's mesh size; fix the elevation warning as proposed (group 2); add a CLI command, like
  `make-ocean-mesh`, writing the multi-mesh to a `.msh` file usable in a training config, if gmsh can hold multi-mesh
  edges that are not edges of the finest mesh; is the test's search radius GraphCast's?
- **Changes:** `icosahedral_mesh.get_multi_mesh_data` (mesh size = longest finest edge, 1,054 km at 3 splits, at
  every vertex); `geospatial_mesh_utils.write_multi_mesh` and `graphcast make-icosahedral-mesh SPLITS OUTPUT`;
  `read_mesh` leaves the `multi_mesh_edges` physical group out of the boundary and `read_mesh_data` takes the edges
  from it; `read_mesh` stores the node tags aligned with the vertices (they were the sorted unique tags, equal only
  because seamsh writes sorted tags) and handles meshes without boundary lines (`np.vectorize` on an empty array).
  The radius-query test uses GraphCast's radius (0.6 × mesh size) and a radius per vertex.
- **Insights:** gmsh holds them: discrete entities have no topology, so the directed multi-mesh edges are line
  elements of a discrete curve, kept in order (5,100 at 3 splits), coordinates within 1e-9 m. Once a physical group
  exists gmsh saves only elements in physical groups, so the triangles need one too. seamsh meshes call their boundary
  group `Coastline`. CLI → `.msh` → `read_mesh_data` → tiny GraphCast forward works, with and without scan. The legacy
  test's radius 0.2 (unit sphere, 3 splits) came from the original GraphCast test, about twice GraphCast's 0.6 × longest
  finest edge (0.099); 0.6 exceeds a face's circumradius (L/√3 ≈ 0.577 L), so every grid point is connected.

## 2026-11-06 11:15: make-icosahedral-mesh reads a config

- **Review:** the command takes a configuration file (splits, description, license) committed in `configs/mesh`, as
  the ocean meshes and the other commands do; why the underscore in `license_`?
- **Changes:** `mesh_config.IcosahedralMeshConfig` (unknown keys rejected, `splits` non-negative, also `model_name`
  and `view_tag_name`); `graphcast make-icosahedral-mesh CONFIG OUTPUT [--overwrite]`; `configs/mesh/icosahedral_6.toml`
  (GraphCast's 6 refinements, licensed CC BY-NC-SA 4.0 like the ocean meshes): 40,962 vertices, 327,660 directed
  edges, finest edges 110–132 km, mesh size 132 km, 11.6 MB, reads back without boundary. The `license_` option is
  gone with the config (the underscore avoided shadowing the built-in `license`, which ruff does not check here).

## 2026-11-06 13:30: group 2, fixes

- **Request:** stubs for ty as proposed (group 4), then group 2 (the fixes of the stage 3 plan).
- **Changes** (one commit each, each with a test that failed before):
  - `climatology.py` deleted.
  - `xarray_jax.pmap` checks the leading dimension before `jax.pmap` (the check ran after jax had failed).
  - Edge direction features: `np.deg2rad` of pyproj's degrees (since 15e0389, 6 Feb 2026, `np.rad2deg`: sin/cos of
    meaningless angles on all three edge sets; the forward azimuth feature, off, was not in turns).
  - `edge_normalization` is a length, as `model.mesh2grid_edge_normalization_factor` (was unpacked as a pair).
  - Missing mesh size: `read_mesh` names the missing view (was a KeyError) and returns None without a name;
    `GraphCast` rejects a mesh without mesh size.
  - `get_transform`: elevation tolerance 1e-6 of the ellipsoid's semi-major axis (was 1e-8 absolute, below float32
    rounding; the icosahedral multi-mesh warned at every model construction).
  - `training_utils.fsdp_map` deleted (unused).
  - Durations in days as "1D" in the training configs (pandas deprecates "d"), validated when the config is read.
  - `jax.shard_map` in `training.py`; JAX coordinates assigned as `xarray.Coordinates` without indexes (the rename
    trick warned); DeprecationWarning and FutureWarning from `graphcast` modules are test errors (62 warnings → 1).
  - `sharding.FullyShardedDataParallel` works: shard-mapped pure init/apply, init lifted outside `shard_map`, both
    compiled, a predictor factory (Haiku modules made outside a lifted transform cannot be used in it). Tests on 4
    simulated CPU devices (conftest appends the XLA flag): a tiny GraphCast behind `Mask`, unrolled and scanned, has
    the same parameters, losses, gradients and predictions with and without it. Parameters replicated, not sharded.
  - `graphcast compute MESH MASK OUTPUT`: barycentric weights of each ocean grid point in its decoder face (sparse,
    3 per row); reads the download-mask output like training (`get_grid` takes a path); `python -m graphcast.basis`.
  - `mesh_connectivity` docstrings match the code.
- **Experiment** (B1 check, `claude-work/…/b1_check`, jobs 2718140/2718141, 1 node each, `training2647`):
  `production_run.toml` on 120 km × 8, 500 steps, validation at 250 and 500, code before and after the azimuth fix.
  Validation loss 0.13241 → 0.12827 before, 0.13227 → 0.12815 after (−0.1 %, within seed noise of 0.6 %); compute
  0.44 / 0.46 s per step. The fix does not hurt; no measurable gain at 500 steps.
- **Insight:** jitting the shard-mapped init matters: outside jit, shard_map ran the init of a tiny GraphCast
  operation by operation (46 s instead of 3 s). Coverage 69 % (53 % before group 1).

## 2026-11-06 15:30: review of group 2

- **Review:** rename `graphcast compute`; other configs to update for "1D"; basis at arbitrary points (planned
  interpolating decoder, hierarchical meshes); why `check_vma=True` broke training, and whether it is faster; how to
  pass rng keys to predictors and losses in data-parallel shard_map (a probabilistic variant is planned).
- **Changes:** `graphcast compute-basis`; `basis_utils.compute_basis(mesh, points)` and `compute_grid_basis`;
  `training_utils.data_parallel` (key per device by fold-in, the key an argument of shard_map) used by the train and
  eval steps (keys from the seed and the step, validation by batch), and by `FullyShardedDataParallel`; the scanned
  processor's carry is cast to the axes it varies along, so it works with `check_vma=True`. JUBE and the mesh study
  derive their configs from the shipped ones; 12 generated configs in `runs/` still say "1d" (left as is).
- **Insights** (`claude-work/…/rng_shard_map.py`, `check_vma.py`): closing over a key computed outside shard_map
  reproduces the error of the deleted comment on Explicit axes; passing it as an argument and folding inside works
  (split keys too, after `jax.reshard`). `check_vma=True` failed only with scan: the carry starts with mesh edge
  features equal on every device and returns varying along the batch axis.
- **Experiment** (jobs 2720430/2720432, 1 node, `training2647`, the B1 check settings: 120 km × 8, 500 steps):
  `check_vma` off 0.441 s per step, on 0.516 s (+17 %), XLA memory 54.0 GiB both, validation loss 0.12803 / 0.12805
  (B1 check before the rng change: 0.12815). Kept off: the gradient of the cast to varying is an all-reduce of the mesh
  edge features' cotangents, which the unchecked version does not do. (A first submission failed in 90 s: the copied
  config still said "1d", which the new validation rejected.)

## 2026-11-06 18:30: group 3, tests for the whole codebase

- **Request:** update the old run configs ("1d" → "1D", 12 files in `runs/`, not versioned: done), then group 3.
- **Changes** (tests, 26 modules in `tests/`):
  - **End-to-end training on CPU** (`test_training_run.py`): `graphcast train` of a tiny model on the icosahedral
    multi-mesh and the synthetic store, 4 CPU devices, validation, checkpoints, TensorBoard, model saved; then
    `--continue`. A subprocess, as `jax.distributed` must be initialized first (one-process cluster on localhost);
    coverage measures it (`patch = ["subprocess"]`, pytest-cov 7 no longer does). The synthetic grid is now global
    (-67.5° to 67.5°), as the loss latitude weights require.
  - New test modules: autoregressive, rollout, validation_utils, graph_pruning, plot_utils, casting, cli_utils,
    grain_utils, typed_graph_net, deep_typed_graph_net, mesh size fields; more in model (scan vs unrolled with the same
    parameters, connected grid nodes), mesh_graph, mesh_connectivity, gis_utils, data_utils, checkpoint.
  - `conftest.py`: `JAX_PLATFORMS=cpu` (in a shell of the image on a node without GPUs, NVIDIA's environment made the
    CUDA plugin fail instead of falling back to the CPU, so `apptainer run … pytest` failed a test), `MPLBACKEND=Agg`,
    a shared tiny GraphCast fixture.
- **Bugs found by the tests, fixed** (each its own commit, the test failed before):
  - `GraphCast`: forcings without grid dimensions (only year progress) failed with a KeyError (`lat`).
  - `checkpoint`: empty lists/dicts/tuples did not load ("Missing value"): a model trained on a mesh without
    description could be saved but not loaded.
  - `rollout`: every rollout failed (`xarray.Dataset(dataset)` rejected by current xarray); a concat relied on a
    changing xarray default.
  - `validation_utils`: deprecated `Dataset.drop`.
  - `plot_utils.plot_data`: animation titles assumed nanosecond timedeltas.
  - `data_utils.fix_longitude`, `gis_utils.wrap_longitude` (and so `xarray_to_gdal_raster`): IndexError on data already
    in [-180, 180].
  - `casting.tree_map_cast`: leaves of other dtypes became None.
  - `mesh_graph.latlon_to_nx`: latitude and longitude swapped in the node attributes.
  - `mesh_connectivity.get_masking_indices_fns`: the first unmasked point (index 0) was taken as missing;
    `get_dummy_xscaling_graph`: each mesh's edges built from the other mesh's faces.
- **Result:** 446 tests, 3 min 54 s with coverage through `apptainer run` (login node); coverage 94 % (53 % before
  group 1); every module above 80 % except `predictor_base` (abstract) and the CLI group `__init__`.
- **Candidate for deletion:** `cli_utils.DictParamType` is unused (tested meanwhile); `Configs` is used by notebooks.

## 2026-11-06 20:00: mesh node embedder padding; PGLE

- **Request:** is the zero padding of the mesh node inputs (`model.py`, "to make sure capacity of the embedded is
  identical", from upstream GraphCast) needed? Guess from the code first, then measure; then why the PGLE profile was
  empty, fix it and report gains. Delete `cli_utils.DictParamType` (done).
- **Guess (code only):** the mesh node embedder has its own parameters, so the zero columns add nothing forward and
  get zero gradients; they only make the first layer's initial weights smaller (fan-in C + 4 = 228 instead of 4 at 50
  levels, std 7.5x smaller) and cost a wasted matrix product. Expected: no difference.
- **Experiment** (`claude-work/…/pad_check`, `training2647`): 120 km × 8, 50 levels, 2000 steps, validation every 500,
  seeds 42 and 43, pad (current) vs nopad (positions only); jobs 2722806–2722809 on four nodes. Validation loss at step
  1999: pad 0.11303 / 0.11251, nopad 0.11309 / 0.11274 (+0.13 % on the mean, seeds 0.46 % apart); training losses and
  per-variable validation losses the same within the seed spread. Compute 0.471 / 0.489 vs 0.448 / 0.454 s per step,
  but on four nodes; the same node, alternating (job 2723585, 300 steps each): pad 0.433 / 0.427 s, nopad 0.431 /
  0.415 s (median), within run-to-run noise. XLA memory 54.0 GiB. As guessed: no difference.
- **Change:** the padding is removed (maintainer: compatibility with older checkpoints is not a priority). The mesh
  node embedder's first layer has 4 inputs: checkpoints and models saved before cannot be loaded.
- **PGLE** (`claude-work/…/pgle`):
  - **Cause:** since e018542 (25 Oct 2026) the train step is compiled ahead of time (`lower().compile()`, for the
    memory report) and the compiled executable is called; JAX's automatic PGLE lives in the jit call path
    (`_pjit_call_impl_python`), so it never profiles or recompiles the train step. The "Found 0 instructions from the
    profile" warnings (also on 22 Oct, when the step went through jit) come from tiny one-operation modules (`jit_add`,
    `jit_broadcast_in_dim`, ...), which PGLE profiles too and whose single instruction is not in the trace; XLA falls
    back to its analytical estimator for them. The hypothesis that CUDA forward compatibility breaks CUPTI was wrong
    (probe job 2723388: CUPTI collects kernels with `apptainer run` and `exec`).
  - **Experiment:** a snapshot calling the train step through jit when PGLE is enabled (120 km × 8, 150 steps, same
    nodes): the train step's profile has 1454–1455 instructions (10 missing). Median compute per step: 1 node (job
    2723583) 0.418 s without, 0.429 s with PGLE; 4 nodes (job 2723899) 0.440 / 0.443 s. Startup +~25 s (step 0 +12 s,
    two profiling steps of ~2 s, a 12.5 s recompilation at step 3).
  - **Decision:** no gain, so the training loop is unchanged (it keeps the ahead-of-time compile); `.env` and `TODO.md`
    record why.

## 2026-11-06 18:15: group 4 of the tests and code-quality PR (format, lint, stubs, licenses, ty)

- **Commits:** `ruff format` of the whole codebase (b210470, listed in `.git-blame-ignore-revs`); ruff lint fixes
  (literals, PEP 695 generics, `TypeError` for wrong argument types in `gis_utils.get_transform` and `xarray_jax`);
  type stubs in the dev group (`scipy-stubs ~= 1.18.1`, `pandas-stubs ~= 3.0.5`, `types-networkx`, `types-tqdm`,
  `botocore-stubs`, image `graphcast-g4.sif` built from the branch); ruff skips `notebooks/`.
- **Licenses (maintainer: Apache-2.0 everywhere):** every Python file has the maintainer's copyright, an SPDX line and
  the Apache header; the 28 files derived from GraphCast keep DeepMind's line and say they were modified (§4(b)); the
  four MIT headers became Apache. `NOTICE` names the fork's origin and lists the derived files (§4(d)) and says the
  license grants no rights to DeepMind's, Google's or GraphCast's names (§6): the project will be renamed in a
  follow-up PR (candidates Moby-Dick, Moby-Dix).
- **ty:** 139 diagnostics with the stubs and the image's jax, jaxlib and flax as extra search paths, now 0. Fixes where
  annotations or code were wrong (Optional narrowing, Hashable keys, `get_transform`'s return type, `plot_utils` on
  DataArrays, explicit `TaskConfig` fields, AWS credentials without keys); targeted `# ty: ignore[rule]` with a reason
  where ty or the stubs cannot follow the code (Grain and Predictor overrides, jraph defaults, constrained type
  variables, chex.Array). 437 tests pass.

## 2026-11-07 08:30: REUSE compliance

- **Request:** why the PR dropped `SPDX-FileCopyrightText` (four formerly MIT files) and left configuration files
  without SPDX lines. Answer: the first to give one header form with DeepMind's `Copyright` line (REUSE accepts both),
  the second a scope I set (Python files only) without asking. Maintainer accepted the recommendation: full REUSE.
- **Change** (929797e): `SPDX-FileCopyrightText` in every Python file (DeepMind's and the maintainer's); SPDX lines in
  TOML, YAML, shell, Slurm, .env, .gitignore and JUBE files; `REUSE.toml` for the rest; `LICENSES/` with Apache-2.0 and
  CC-BY-4.0. Images and animations are CC-BY-4.0 (maintainer), except the Argonne, GRC and OGS logos in the poster.
  `spack/templates/container/graphcast.def` turned out to be Spack's `singularity.def` with additions (Apache-2.0 OR
  MIT): Spack's copyright is in `REUSE.toml` and `NOTICE`. `uvx reuse lint`: compliant, 169 files.
- **Next:** a `reuse lint` pre-commit hook in group 5.

## 2026-11-07 08:40: ShellCheck

- **Request:** a linter for the bash scripts. ShellCheck 0.11.0 (`shellcheck-py`, which bundles the binary) on the 14
  shell files found no errors: 33 SC2034 (unused variable) in `.env` and the JUBE job template, false positives since
  both are sourced under `set -a`; 2 SC2126 (`grep -c`) in the JUBE run scripts, false positives as they count over
  several files; notes in `slurm/launch_jupyter.slurm` and `slurm/test_dask_mpi.slurm`, which sourced the removed
  `clusters/leonardo/` (the second also ran a missing script and had `mkdir -p DASK_MPI_LOG_DIR` without `$`).
- **Change** (8b6a78f): both Leonardo scripts deleted (maintainer); the false positives suppressed with reasons.
  ShellCheck is clean. No `shfmt`: the scripts have one consistent style already.
- **Next:** the `shellcheck-py` pre-commit hook in group 5, with a `files` pattern for `*.slurm`, `.env` and
  `*.sbatch.in` (pre-commit detects shell scripts without a `.sh` extension only by the shebang of executables).

## 2026-11-07 16:40: review of PR 3 (compute_basis, masking test, Configs)

- **Requests:** test `compute_basis` against scipy's `LinearNDInterpolator` (the draft interpolated in a stereographic
  projection) and measure its time and memory, to decide whether the basis can be computed on the fly for an
  interpolating decoder; say what bug `test_masking_index_maps` guards against; drop `cli_utils.Configs`.
- **compute_basis** (c4e027a): on a flat triangulation it equals `LinearNDInterpolator` on the same triangles (to
  1e-9); on the sphere it differs from the draft's stereographic interpolation by O(edge²), about 0.9 edge² for edges
  of 3.8–0.47°, far below the interpolation error. A prototype on the unit sphere gave 25 edge² at 0.24° edges:
  trimesh 5.2.0's `closest_point` compares squared distances of candidate faces with `tol.merge` (1e-8, absolute),
  which all squared distances fall below there, and its tie-breaking returned other faces' points (closest-point error
  4.8e-7 against 2e-16 in metres). Meshes are in metres, so training is unaffected; the docstring says so.
- **Cost** (`claude-work/…/basis_bench`, login node jpbl-s01-04, one process each), at the 693,039 ocean points:

  | Mesh | Vertices | Basis | Peak host memory | Decoder edges (built at startup today) |
  |---|---|---|---|---|
  | 120 km | 34,007 | 9.1 s | 1.83 GiB | 9.4 s |
  | 50 km | 181,325 | 11.8 s | 1.92 GiB | 11.8 s |
  | 35 km | 363,338 | 13.7 s | 2.04 GiB | 14.3 s |
  | 25 km | 703,663 | 18.6 s | 2.09 GiB | 19.5 s |

  Mesh to mesh: 50 km at the 25 km vertices 15.3 s, 1.72 GiB; 120 km at the 50 km vertices 3.4 s. The matrix is 18.5
  MiB (3 float32 weights and int32 indices per point). Same query as the decoder edges (closest face with trimesh), so
  an interpolating decoder could build the basis instead of those edges at no extra startup cost; no file needed.
- **Masking test** (bd16994): the docstring names the fix (b8286ea) and that only the unused `unmasked_to_masked` was
  affected.
- **Configs** (4e09373): removed with its test; `spilhaus.ipynb` and `forward.ipynb` use `read_configs` with the
  schemas (not re-run); `forward.ipynb` also uses removed APIs and needs a total rewrite (TODO.md).
- **Benchmark slip:** the first runs imported the image's installed package (no `PYTHONPATH=src`) and did not call
  `gmsh.initialize()`, which `read_mesh`'s callers do; not a bug of the code.

## 2026-11-07 17:30: group 5 of the tests and code-quality PR (pre-commit, CI)

- **Pre-commit** (`.pre-commit-config.yaml`): ruff check --fix and ruff format (v0.17.3), ShellCheck (shellcheck-py
  v0.11.0.1; the pattern also takes `*.slurm`, `*.sbatch.in` and `.env`, checked with a broken probe file), reuse lint
  (v6.3.0), and a local ty hook (the environment's Python; `/opt/jax`, `/opt/jaxlibs/jaxlib`, `/opt/flax` added as
  search paths where they exist). All pass in `graphcast-g4.sif`.
- **CI** (`.github/workflows/tests.yml`): micromamba with Python 3.12, GDAL 3.13.1, PROJ 9.7.0 (conda-forge), the
  package and dev group from PyPI, versions printed, hooks, `pytest --cov` (80 % gate, `coverage.xml` artifact).
  The local dry-run solve with a downloaded micromamba was denied by the auto-mode classifier: the first CI run checks
  it. The push of the workflow was rejected: the fine-grained token of `gh` lacks the Workflows permission
  (maintainer to grant it).
- **Image versions** for comparison with CI: GDAL 3.13.1 / PROJ 9.7.0; pyproj 3.8.0 with its own PROJ 9.8.1 (wheel).
- **Coverage** with the gate: 93.37 %, 441 passed in 4 min 42 s (`graphcast-g4.sif`, login node).
- CLAUDE.md: the absltest, uv-fallback and "format only new modules" notes replaced by the hooks, ty invocation and CI.
- **CI runs** (7 Nov 2026, after the maintainer granted the token the Workflows permission):
  1. 42390949291: the environment did not solve, as conda-forge's GDAL 3.13.1 builds all need PROJ >= 9.8.1: PROJ is
     no longer pinned (253a40b); CI gets GDAL 3.13.1 / PROJ 9.8.1, pyproj 3.8.0 / PROJ 9.8.1.
  2. 42391097167: the newer PyPI releases (pandas 3.1.1, pydantic 2.14.3, jax 0.12.0.1; the image has 3.0.5, 2.13.5,
     0.11.1.dev) gave 16 test failures (`DatetimeIndex.dayofyear` deprecated, turned into errors by the warning
     filter) and 5 ty diagnostics (`default_factory` optional, `tree_map_cast` dtypes): fixed in b0bae10.
  3. 42391894372: green in 5 min 54 s, 441 passed, coverage 93.37 %.

## 2026-11-08 08:35: cleanup after merging PR 3

- **Request:** remove the leftovers of PR 3 and replace the Apptainer image.
- **Commit hashes:** PR 3 was merged by rebase, so its commits have new hashes on `develop`. `.git-blame-ignore-revs`
  listed the format commit by its branch hash (5176406), so `git blame` no longer skipped it; it now lists b210470.
  The hashes cited in CLAUDE.md, `test_masking_index_maps` and this logbook (82 commits, matched by subject) are
  replaced by their copies on `develop`.
- **Image:** rebuilt from `develop` 57da6c0 (`spack -e …/spack-env containerize` from the worktree, ~6 min), with the
  type stubs; same jax 0.11.1.dev20260605, jaxlib, Orbax 0.12.4, Haiku and CUDA compat driver (615.71.09) as the
  27 Oct image, GDAL 3.13.1 / PROJ 9.7.0. In it: 441 tests pass, coverage 93.37 %, all pre-commit hooks pass. No jobs
  were running; the previous image is kept as `graphcast-2026-10-27.sif`. Not checked on a GPU (identical base).
- **Removed:** the `data/geometry/uniform_coarse.msh` and `uniform_fine.msh` symlinks (nothing on `develop` uses the
  old names), the worktree's test images `graphcast-g4.sif` and `graphcast-new.sif` (23 GB), and the session's
  claude-work directory `b0a574fd-35c0-4f21-bc1c-92264af27375` (1.4 GB: scripts and logs of the experiments above).

## 2026-11-08 10:20: rename to Moby-Dick (PR `feature/rename-moby-dick`)

- **Request:** rename the project (legal reasons, Apache-2.0 §6, and divergence from GraphCast); plan first, with the
  points where the rename could break things, a README describing the current project, one banner for all commands
  stored as an asset, and the operations left to the maintainer. Session directory
  `$PROJECT/claude-work/14b6a8df-6a12-47e1-9ce4-a3eab2b94fd5`.
- **Names (decided by the maintainer):** project Moby-Dick, repository `moby-dick`, command group `moby-dick`,
  environment prefix `MOBYDICK_` for all models. Derived: distribution (PyPI) `moby-dick` (free on PyPI on 8 Nov
  2026, as is `mobydick`; `leviathan` is taken), umbrella import package `mobydick`, first model `mobydick.leviathan`
  (class `Leviathan`, Haiku module `leviathan`), image `moby-dick.sif`. The prefix follows the package that reads the
  variables (as `SKLEARN_*` for scikit-learn); a single token avoids the ambiguous `MOBY_DICK_GRAIN_NUM_WORKERS`.
- **Several models:** as in `google-deepmind/weathernext` (the `upstream` remote: one `weathernext` package, shared
  `utils/`, one subpackage per model), shared code stays flat in `mobydick`, the model moves to `mobydick.leviathan`.
  `TaskConfig` moved to `mobydick.task`; `[model]` is `leviathan.config.LeviathanConfig` with `architecture =
  "leviathan"`; `mobydick.models.MODELS` gives the predictor, frozen settings and saved-model type to
  `get_predictor`, `get_model_config` and `save_model`. Grouping the shared modules into subpackages is left for later
  (or never).
- **Commits:** the move (git mv and imports only, in `.git-blame-ignore-revs`); the model hooks; the renames (CLI,
  `MOBYDICK_GRAIN_*`, class and Haiku name, image, spack template `moby-dick.def`, JUBE `training.xml`, configs,
  NOTICE, CI); the banner (`src/mobydick/assets/banner.txt`, package data, stderr, rank 0 only,
  `MOBYDICK_NO_BANNER`); README and CONTRIBUTING (Google's CLA boilerplate) rewritten, README no longer listed as a
  modified copy of GraphCast's.
- **Kept:** GraphCast where it names the origin (DeepMind headers, NOTICE, comparisons with upstream, the paper,
  `notebooks/graphcast_demo.ipynb`), the seamsh fork's `graphcast` branch, data paths and the checkout directory, and
  historical documents (this logbook, the hackathon slides and results, the ARCO-OCEAN findings; only the slides'
  repository links changed).
- **Breaks:** checkpoints and saved models from before (parameters under `graphcast/~/`), `GRAPHCAST_GRAIN_*`
  (silently ignored), images built before (they install `graphcast`; use `python -m mobydick` with them).
- **Verification:**
  - `moby-dick.sif` was built from this branch (~6 min, login node); the shared `graphcast.sif` is untouched. In it:
    452 tests pass with 93.45 % coverage, all pre-commit hooks pass, `pip show moby-dick` gives 0.1, and the banner
    comes from package data.
  - Job 2760080 (`training2647`, 2 nodes, demo_5000 cut to 60 steps, worktree `src`): one banner in the log for
    8 tasks, `num_workers=1` read from `MOBYDICK_GRAIN_*`, validation loss 0.1566 at step 59. The saved model has
    parameters under `leviathan/` and `model_config:architecture`.
  - Job 2760331 (`--continue` to 90 steps): resumed at step 60, validation loss 0.1503 at step 89.
  - The JUBE `training.xml` dry run works.
  - The main checkout's `spack-env/` still holds the old `spack.yaml` (template `container/graphcast.def`): recreate
    it after the merge.
- **Codecov** (maintainer's request, 8 Nov 2026 11:30): CI uploads `coverage.xml` with `codecov/codecov-action@v7`
  (v7.1.1 is the latest release), using the repository secret `CODECOV_TOKEN`; upload errors do not fail the checks.
  The README shows the badge of `stefanocampanella/moby-dick`.
- **CI failures of 8 Nov 2026** (runs 42467982767, 42469164425, 42469390251): `setup-micromamba` got HTTP 504 for
  micromamba from GitHub release assets, before any step of ours ran. From JUPITER, `github.com/` answered 200, but
  repository pages and release assets answered 504 from GitHub's Frankfurt edge (`x-github-edge-region: fra`, GitHub
  request ids, no proxy); `api.github.com` worked, and githubstatus.com reported no incident. The maintainer could
  download the asset, so the failure was partial. Pinning `micromamba-version` did not help (pinned and latest both
  come from GitHub releases) and was reverted.

## 2026-11-08 12:15: CI split into lint / reuse / tests (PR `feature/split-ci`)

- **Request** (agreed before PR #5 merged): one status per check, as a single red job hid that the tests passed.
- **Change:** `tests.yml` became `ci.yml` (workflow `CI`) with three parallel jobs: `lint` (ruff check, ruff format,
  ShellCheck: `setup-python` and pre-commit, `SKIP=reuse,ty`), `reuse` (`pre-commit run reuse`) and `tests`
  (micromamba environment as before, `pre-commit run ty`, then pytest with coverage, artifact and Codecov, also when
  ty fails). Hook versions stay in `.pre-commit-config.yaml`. The split does not help against GitHub release-download
  outages (shellcheck-py and micromamba both download from GitHub releases).
- If branch protection requires the old `checks` status, it has to be replaced with `lint`, `reuse` and `tests`.

## 2026-11-08 14:45: icosahedral multi-mesh vs ocean mesh (branch `experiment/icosahedral-vs-ocean-mesh`)

- **Request:** an experiment comparing performance and skill of the best ocean mesh setup with GraphCast's icosahedral
  multi-mesh, long enough for a trend but shorter than a production run (the mesh study's 5000 steps), in a new branch.
  Session directory `$PROJECT/claude-work/ed336b51-7c1c-4db2-b3ca-b6880ad836fd`.
- **Setup:** `mesh_study.sh` accepts mesh files other than the uniform meshes; `data/geometry/icosahedral_6.msh` made
  with `make-icosahedral-mesh configs/mesh/icosahedral_6.toml`. Cases icosahedral × 16 (GraphCast) and × 8, 50 km × 16
  (`production_run.toml`) and 120 km × 8 (the mesh study's baseline), seeds 42 and 43, demo_5000 on 8 nodes; jobs
  2765280–2765287 (`training2647`, 78.2 node-h, all completed). Report:
  `docs/jupiter-hackathon/results/ico_vs_ocean-2026-11-08/REPORT.md`, detailed version
  https://claude.ai/artifact/Bm8DGCiTHcJhL7k9AR5Lwg.
- **Results** (validation loss at step 5000, mean of seeds): 120 km × 8 0.08382, icosahedral × 8 0.08371 (−0.1 %),
  50 km × 16 0.07524 (−10.2 %), icosahedral × 16 0.08263 (−1.4 %): the icosahedral mesh is 9.8 % worse than 50 km at
  16 steps, a gap stable from step 2500. It wins only `zos` (−16.7 % against 50 km × 16; reach: edges up to 7,076 km,
  processor diameter 14 hops); the other eight targets are 10–31 % worse. 0.78 s per step and 55 GiB, against 1.40 s and
  74 GiB.
- **Graph statistics** (request at 15:30): new module `graph_statistics` and command `moby-dick graph-statistics CONFIG
  MESH MASK OUTPUT.json` (`configs/analysis/graph_statistics.toml`): networkx structure (density, degrees,
  connectivity, diameter, shortest paths, clustering, betweenness, closeness; sampled on large graphs), geodesic edge
  lengths, mesh nodes connected to unmasked grid points and vice versa, triangle quality (aspect ratio, radius ratio,
  volume-length, angles; Shewchuk 2002, https://scicomp.stackexchange.com/a/27095). `[mesh] log_statistics` logs a
  summary at training startup (without networkx). 10–12 min per mesh on a login node, 3–4 GB.
- **Insights:**
  - 23 % of the icosahedral nodes are over land (no grid data); its ocean-connected nodes (31,371) are as many as the
    120 km mesh's, and it equals that mesh on every local field.
  - The ocean meshes leave the Black Sea out of their domain: about 900 ocean points in the mask are decoded from
    Aegean vertices up to 1,232 km away and never encoded. In all, 2,222 (120 km) and 2,342 (50 km) ocean points have
    no encoder edge; the icosahedral mesh covers all.
  - The baselines are about 8 % better than in the mesh study (120 km × 8 0.09068 → 0.08382, 50 km × 16 0.08181 →
    0.07524) with the same config and validation windows; most likely the edge-direction fix of 6 Nov (`0473751`), not
    ablated. Results from before 6 Nov should not be mixed with later ones.

## 2026-11-11 13:00: graph statistics reworked after review (PR #7)

- **Request:** address the review of PR #7 (17 comments on the graph statistics module).
- **Changes:**
  - No statistics in training any more (`[mesh] log_statistics` removed): too slow, `moby-dick graph-statistics` only.
  - Graphs of a mesh hierarchy (`graph-statistics CONFIG MASK OUTPUT.json MESH...`, finest first): encoders grid →
    mesh 0 and mesh i−1 → mesh i, processors, decoders back down; mesh-to-mesh graphs use the grid's rules (new
    `mesh_connectivity.get_points_to_mesh_edges`) until hierarchical models define theirs. Coverage per pair of levels.
  - networkx only for the processors' path metrics, per connected component (no longer only the largest), from
    `single_source_shortest_path_length` (implemented by nx-cugraph): diameter as the largest eccentricity of the
    samples and of the node farthest from them, summary of the shortest path lengths. Dropped clustering, betweenness,
    closeness. Encoders: degrees; decoders: the 3-edges-per-receiver check. Summaries: mean, std, min, quartiles, max;
    angle summary over all angles.
  - Output: rich tables on stdout, a description for every quantity in the JSON, only sizes and timings in the log.
    `--networkx-backend` (or networkx's `NETWORKX_BACKEND_PRIORITY`) selects a backend; nx-cugraph is not in the image
    (`nx-cugraph-cu13` 26.10 installs, 3.2 GB with its dependencies; running it here was not permitted, untested).
  - `gis_utils`: `get_transform` (array in, array out) and `get_components_transform` (one array per coordinate)
    replace `get_transform(..., pack_back=...)`, whose return type depended on the argument and the input.
  - k-d tree queries use all cores by default (`workers=-1` in `mesh_connectivity`, `graph_pruning`, Leviathan): all
    runs so far used 1 thread; the edges do not depend on it (test).
- **Timing:** 21–44 s per mesh on a login node (120 km, icosahedral M6, 50 km), 51 s for 50 km + 120 km as a
  hierarchy; it was 10–12 min with networkx on every graph.
- **Insights:**
  - The processor's boundary edges are one-way (`faces_to_edges` assumes a closed mesh): 3,612 of 194,040 edges (120
    km) and 9,357 of 1,061,739 (50 km) have no reverse; the icosahedral mesh has none. TODO.
  - The 50 km mesh's diameter is at least 515 hops (the networkx 2-sweep of 8 Nov gave 448); the report says "≥ 515".
    The report's claim that 8 steps reach every node of the icosahedral mesh was wrong (median shortest path 8 hops,
    diameter ≥ 14): corrected.

## 2026-11-11 16:30: second review of PR #7 — one-way coastline edges fixed, graph statistics split

- **Request:** fix the one-way coastline edges (a bug), split and comment the graph statistics module, show the latest
  findings in the report artifact; review comments on `get_transform`'s axis order, the backend option, and whether
  betweenness and closeness follow a scaling law.
- **Changes:**
  - `mesh_graph.faces_to_edges` appends the reverse of every boundary edge (closed meshes unchanged); all uniform
    meshes now have 0 one-way edges (120 km: 194,040 → 197,652 edges).
  - `gis_utils.cartesian_to_lat_lon` / `lat_lon_to_cartesian` return and take (lat, lon), replacing the transposes and
    longitude-first unpacking at the call sites; the transforms document their axis order.
  - `mobydick.graph_statistics` is a package (`config`, `graphs`, `structure`, `geometry`, `quantities`, `report`,
    `cli`); `--networkx-backend` dropped for networkx's `NETWORKX_BACKEND_PRIORITY`; extra `graph-gpu` (nx-cugraph).
- **Scaling** (all ten uniform meshes, after the fix, 28–147 s each on a login node): mean shortest path L = 0.517 ·
  N^0.504 hops (within 2.3 %), L × median edge length 11,400–11,900 km at every resolution; diameter ≥ 1.37 · N^0.49.
  Exact networkx values on icosahedral meshes and open patches: average betweenness = (L − 1)/(N − 2) to 6 digits,
  closeness within 3 % of 1/L. Scripts and CSV in the session directory (`scaling.py`, `scaling.csv`,
  `centrality_identity.py`).
- **Open:** the mesh study and the icosahedral comparison trained with one-way coastline edges; the effect of the fix
  is not measured (TODO: rerun 50 km × 16 and 120 km × 8, ~45 node-h).

## 2026-11-11 18:30: encoder radius from the longest triangle edge (PR #7)

- **Request:** check whether the encoder radius at a mesh vertex from its longest edge, instead of the mesh size,
  fixes the grid points without encoder edges (apart from the Black Sea) and the same gaps between meshes of a
  hierarchy; if so, include it.
- **Check** (`radius_rules.py`, `radius_rules.out` in the session directory; points without encoder edges, and
  whether they lie inside a mesh triangle by radial projection):

  | Mesh | 0.6 × mesh size (before) | 0.6 × longest incident edge | 1.0 × longest incident edge |
  |---|---|---|---|
  | 120 km | 2,222 (1,129 inside a triangle) | 1,222 (0 inside; Black Sea 936) | 993 (0 inside), 2.9 × edges |
  | 50 km | 2,342 (820 inside) | 1,725 (0 inside) | 1,191 (0 inside) |
  | 25 km | 3,697 (616 inside) | 3,099 (0 inside) | 1,546 (0 inside) |
  | icosahedral M6 | 0 | 0 (5 % fewer edges) | 0 |
  | 50 → 120 km (finer vertices) | 260 (all inside) | 2 (outside) | 0 |
  | 25 → 50 km | 802 (all inside) | 8 (outside) | 0 |

  Encoder edges grow by 5.5 % (120 km) and 7.6 % (50 km). The remaining points are outside the mesh domain: the
  Black Sea and coastal channels or ice-shelf fronts (Tierra del Fuego, the Weddell Sea).
- **Change:** `mesh_graph.longest_incident_edges` (from the triangles only, not a multi-mesh's coarser edges); the
  model's and graph-statistics' radius is `radius_query_fraction_edge_length` × it. The mesh size field is no longer
  needed by the model. New tests: coverage of every point on an irregular closed mesh, the radius of the model.
- **Not measured:** the effect on skill and on GPU memory (TODO, with the two-way edges rerun).

## 2026-11-11 19:30: mesh vertices without grid points; rerun of the ocean cases with both fixes

- **Side finding corrected:** I had reported that 6–8 % of the 25 km mesh's vertices receive no grid point as a
  problem. They are still updated at every processor step (no node mask in `_run_mesh_gnn`; their encoder state is the
  embedding of their position) and 99.3 % of the 25 km vertices send decoder edges, so their state is used. They
  start without data and hold their neighbours' after one step: one step of reach lost there, not coverage. Only
  vertices in neither the encoder nor the decoder (0.6 % at 25 km; 23 % of the icosahedral mesh, over land) are idle.
- **Rerun** of 50 km × 16 and 120 km × 8 (demo_5000, seeds 42 and 43, the protocol of 8 Nov) with the two-way
  coastline edges and the encoder radius from the longest triangle edge: study `edge_fixes`, jobs 2841380–2841383
  (`training2647`), code of this branch (808fba5). Compare with jobs 2765280–2765287 (validation loss, per variable,
  XLA memory, step time).

## 2026-11-11 20:00: processor nodes outside the encoder and the decoder

- **Question (maintainer):** are the mesh nodes with neither encoder nor decoder edges not needed as intermediates of
  the processor's message passing? I had called them idle relays.
- **Answer:** they are full processor nodes, so yes in general; measured with `relay_nodes.py` (shortest paths between
  the data-carrying nodes from 32 sources, with and without them):
  - icosahedral M6: 9,591 land vertices (23 %); without them the data nodes stay connected (long edges), but 20 % of
    the paths get longer, mean path × 1.032: they relay across continents.
  - 120 km / 50 km: 233 / 1,122 vertices, all poleward of 60° S (median 79° S), under Antarctic ice shelves outside
    the GLORYS mask: dead ends for paths between data nodes (0.0 % / 0.2 % longer; one 120 km vertex reaches the
    rest only through them). They still exchange messages with their neighbours.

## 2026-11-11 21:00: seamsh refactoring kit (side quest of PR #7)

- **Request:** prepare everything known about seamsh for a parallel session that will refactor the fork (tests,
  pre-commit hooks, bug fixes, comments, self-documenting names; PR into its `develop`), with the end goal of meshing
  disconnected domains (basins cut off by unresolved straits); include moby-dick's mesh files, a validation script
  and a PR guide. No moby-dick code changes.
- **Kit:** `$PROJECT/claude-work/ed336b51-7c1c-4db2-b3ca-b6880ad836fd/seamsh-refactoring-kit{,.tar.gz}`: `PR_GUIDE.md`, `NOTES_seamsh.md`, `validation/validate_seamsh.py` (standalone: seamsh, gmsh,
  GDAL, numpy, scipy; `coarsen` and `mesh` subcommands, probe basins, mesh statistics, comparison with a reference
  mesh), `validation/make_references.sh` and `validation/reference/` (current seamsh), Natural Earth coastlines,
  moby-dick's mesh code, configs, tests and CLAUDE.md excerpt.
- **Findings** (fork `develop` `f1ab87d`, read in a clone in the session directory, and measured):
  - The coarsening keeps only the water reachable from one seed (C `color_triangles`: flood fill across Delaunay edges
    longer than the coarsening size). Lost basins with the 1:10m coastline: Black Sea, Sea of Azov at 10 km;
    + Mediterranean, Baltic at 20 km; + Red Sea at 30 km; + White Sea at 50 km. Coarsening takes 3–11 s.
  - Meshing builds one gmsh plane surface: the largest loop is the exterior, all others are holes; disconnected basins
    need one surface per region.
  - The 120 km mesh file holds 49,054 nodes in no triangle (59 %); 0 inverted triangles, one component, smallest
    angle 7.2°. The standalone script reproduces moby-dick's `uniform_120km.msh` (vertices within 1.4e-9 m, same
    triangles), in 14 s.
  - Bugs and risks by reading: dead `get_cavity` with an allocation-size and an out-of-bounds bug, `getid3` calls
    `exit(1)`, a leak in `extract_boundaries`, unchecked allocations, `np.random.seed(0)` on the global state, an
    unbounded sampling loop, the `tri` view left in the model, a license inconsistency (GPLv3 text, LGPL headers);
    details in the notes.

## 2026-11-11 22:00: nodes in no triangle, mesh parameters, keeping the Black Sea

- **Requests:** log the nodes `read_mesh` drops; other findings worth a code change; whether the coastline and mesh
  parameters are reasonable; parameters for one connected ocean including the Black Sea and Sea of Azov.
- **Correction:** the kit's figure of 49,054 nodes in no triangle at 120 km was gmsh's in-memory model (a node on
  every geometry point of the coarsened coastline), not the file. The files hold 743 of 34,750 (120 km) and 655 of
  181,980 (50 km): outlines of islands smaller than the mesh size (632 at 120 km, 1–3 nodes, up to ~100 km across:
  Kurils, Solomons, Canaries, Gotland, Shetland), whose sea is triangulated over them. Kit notes, guide, script (now
  measuring the written file) and references corrected.
- **Change:** `read_mesh` logs "Dropped N of M nodes in no triangle (outlines of islands not meshed)"; test with a
  hand-built mesh holding a small island outline.
- **Keeping the Black Sea** (`strait_experiment.py`, `strait_geometry.py`, `strait_hops.py` in the kit): the 1:10m
  coastline keeps the straits open (Bosporus 1.09 km, Dardanelles 2.36 km, Kerch 2.51 km). Coarsening refined to
  0.1–2 km at the strait centres but growing 0.2 km/km does not connect them (the straits are 30–60 km long); 0.25–0.5
  km growing ≤ 0.05 km/km does. With a mesh size of 1–3 km at the straits (growth 0.3): one component, all basins;
  120 km: 34,449 (3 km) or 34,621 (1 km) used nodes against 34,007; 50 km: 182,174 against 181,325; smallest angles
  unchanged. The Black Sea is then 47–61 hops from the Aegean. A 30 km coastline with nine straits refined keeps every
  basin but gives a 2.5° smallest angle at 120 km.
- **Parameters:** the 10 km coastline coarsening, smoothness 0.3 and uniform fields are reasonable; the only gap is
  the Black Sea, which needs a coarsening size field (configs have one number) and a size field refined around points.

## 2026-11-11 23:00: strait refinement in the mesh configs

- **Change:** `point_refinement` size field (size at points growing with distance), `coarsen_boundaries.fields`
  (coarsening size = minimum of `mesh_size` and the fields; without fields exactly as before), all ocean configs
  refined at the Bosporus, Dardanelles and Kerch Strait; tests (field values, validation, a synthetic inland sea
  behind a 22 km channel kept only with the refinement).
- **Meshes and graph statistics:** see CLAUDE.md (Meshes); comparison in `strait_meshes_comparison.json` in the
  session directory. Not yet done: copy the meshes to `data/geometry`, update the report and the artifact.

## 2026-11-12 09:00: rerun of the ocean cases with the edge fixes

- Rerun with both fixes (two-way edges, encoder radius from the longest triangle edge; meshes without the Black Sea; study `edge_fixes`, jobs 2841380–2841383, 46 node-h): validation loss 120 km × 8 0.08382 → 0.08373 (−0.10 %), 50 km × 16 0.07524 → 0.07494 (−0.39 %), within the seed spread (0.04–0.40 %), every variable within ±1.1 %; XLA memory +1.1 GiB (120 km, 54.9 GiB) and +0.8 GiB (50 km, 74.8 GiB); step time +8 % / +3.5 %, within the run-to-run variation. No measurable effect on skill.
- Per variable (50 km × 16): zos −0.9 %, thetao −0.4 %, so −0.2 %, uo/vo −0.3 %, siconc −0.7 %, sithick −1.1 %,
  usi/vsi +0.4 %. Script `compare_edge_fixes.py`, tables `edge_fixes_runs.csv`, `edge_fixes_curves.csv` in the session
  directory.

## 2026-11-12 10:00: meshes with the Black Sea in data/geometry

- **Request:** replace the meshes in `data/geometry` and delete the duplicate PR replies.
- **Meshes:** the ten `uniform_*km.msh` moved to `data/geometry/previous_2026-10/` and replaced by the meshes
  regenerated with the strait refinement (byte-identical copies of `$FSCRATCH/graphcast/data/geometry_straits_2026-11-11`;
  no job was queued). The new files are a third smaller: the old ones held the mesh twice (writer bug fixed on 30 Oct).
  `lambrechts*.msh` and `icosahedral_6.msh` are unchanged (the `lambrechts` configs have the refinement, their meshes
  were not regenerated).
- **Duplicate replies:** already deleted (no thread of PR #7 has two identical replies).

## 2026-11-12 11:00: license audit of the dependencies

- **Request:** check the license of every dependency and find inconsistencies and licensing problems (e.g. GPL code
  requiring the Apache code to be redistributed as GPL). No code changes.
- **Method:** metadata (License-Expression, License, classifiers) of the 224 Python distributions in the image
  (`licenses/scan_licenses.py`, `licenses.csv`, `direct_deps.txt` in the session directory), Spack libraries and their
  linkage (`ldd`), Ubuntu copyright files, the seamsh sources, the mesh configs and the README.
- **Findings** (details in CLAUDE.md, Repository notes): gmsh (GPL-2.0-or-later) and seamsh (GPL-3.0; LGPL headers
  contradicted by the bundled gmsh code) are the only copyleft direct dependencies and are loaded by every command, so
  a distributed image is a GPL-3.0 combined work, which also loads proprietary NVIDIA libraries (gray area); the
  repository itself is fine under Apache-2.0. Mesh files are CC BY-NC-SA without a reason from their inputs, the
  `lambrechts` meshes lack the Copernicus attribution, and data attributions and a weights license are missing.
  Everything else is permissive or dynamically linked LGPL. Missing metadata: namex and transformer_engine ship
  Apache-2.0 texts; nsys-jax (NVIDIA, base image) has none.

## 2026-11-12 13:00: fixes from the code review of PR #7

- **Request:** fix the findings of `/code-review high 7` except the checkpoint warning (no production run yet).
- **Changes:** graph-statistics' encoder radius described as the model's (longest triangle edge); JSON written before
  the tables, with NaN and infinities as null (strict JSON); tables print "-" for a processor without components and
  for graphs without receivers; volume-length histogram over [-1, 1] (inverted faces counted); `faces_to_edges` keeps
  each directed edge once (a flipped face gave an edge and its reverse twice), with shared helpers
  `directed_edge_codes` / `reverse_is_missing` also used by the statistics, which now count duplicate edges;
  vectorized index mapping in `mesh_connectivity`; `CompositeMeshSizeField` derives from `MinimumField`; a missing test
  docstring; no cluster name in `icosahedral_6.toml`. Edge counts unchanged for the meshes in use (icosahedral M6
  327,660; 120 km 199,720). 484 tests pass.

## 2026-11-12 16:00: third review of PR #7

- **Comments addressed:**
  - No path prefixes: `make-ocean-mesh` lost `--data-path`, the fields and the domain take their paths as they are,
    and the mesh configs use container paths (`/data/geometry/...`; `lambrechts*` bathymetry at `/data/bathymetry/...`).
  - The coastline coarsening takes size fields in full generality: `[[coarsen_boundaries.fields]]` (minimum of the
    fields) replaces `coarsen_boundaries.mesh_size`; every config has a uniform 10 km field there (plus the strait
    refinement). A single uniform field gives exactly the previous coarsening.
  - `CoarsenBoundariesConfig` defined after `FieldConfig` (no forward reference and `model_rebuild`); the check of
    unique field types, now shared, explains why (fields are kept by type name, a repeated type would replace the
    first silently).
  - Edge codes are uint64 (one-to-one up to 2**32 nodes), with an error beyond that or for indices outside
    [0, num_nodes) instead of wrapping around.
  - Chord distance: quantified in `PointRefinementField` (shorter than the geodesic by ~d³/(24 R²): 0.08 % at 500 km,
    beyond which every refinement has reached its base size) and marked "CHORD DISTANCE" in
    `mesh_graph.longest_incident_edges` and `mesh_connectivity.get_points_to_mesh_edges`, which must change together
    when geodesic distances are introduced.
  - The comment on the nodes in no triangle names the seamsh commit (fork `develop` `f1ab87d`, 11 Nov 2026) and says
    why they are dropped and why the log is informational.

## 2026-11-12 22:00: seamsh v0.6.0+fork: pin, image, meshes, reruns

- **Request:** pin seamsh to v0.6.0+fork (merged PR in the fork), rebuild the image, rerun every investigation of the
  report artifact with it; the changes the fork's documentation suggests for moby-dick go to a later PR (TODO).
- **Image:** rebuilt from this branch (Spack environment unchanged; ~40 min on a busy login node), seamsh 0.6.0+fork at
  `3c9304c` with gmsh 4.12.1; installed as `moby-dick.sif` with approval, previous image `moby-dick-2026-11-08.sif`.
  486 tests pass with it (compute node, 3 min).
- **Meshes:** the ten uniform meshes regenerated (job 2864195) and installed in `data/geometry` with approval (previous
  in `data/geometry/previous_2026-11-12/`); details in CLAUDE.md (Meshes). Same structure, better triangles on the
  whole, 2–3 % fewer encoder edges, no node outside the triangles.
- **Checks** (job 2864533): encoder radius (no ocean point inside a triangle without encoder edges), nodes outside the
  encoder and decoder (now needed: the strait elements carry no grid point and are the only link to the Black Sea, Azov
  and Marmara, 4 % of the paths at 120 km break without them), hops through the straits (45 / 48), path scaling
  (L = 0.542 · N^0.501).
- **Training** (study `ico_vs_ocean_v060`, 75.5 node-h): 120 km × 8 0.08333, icosahedral × 8 0.08381, 50 km × 16
  0.07464, icosahedral × 16 0.08262; icosahedral × 16 is 10.7 % worse than 50 km × 16, −16.0 % on `zos`. Ocean cases
  −0.6 / −0.8 % since 8 Nov, icosahedral unchanged. Report section and CSVs added.

## 2026-11-13 08:00: mesh quality in the artifact; Lambrechts vs uniform meshes

- **Requests:** show the change of mesh quality with seamsh v0.6.0+fork in the artifact; test the performance and
  skill of Lambrechts meshes of different sizes against uniform meshes.
- **Quality** (`quality_compare.py`): table and chart of the ten uniform meshes before/after in the artifact; the bulk
  improves at every size (25th percentile of the smallest angle per face +1°, faces with volume-length < 0.8 −3 to
  −20 %), the 6–52 worst faces (< 0.5) do not.
- **Meshes:** `lambrechts_mini`, `lambrechts_coarse`, `lambrechts` regenerated with seamsh v0.6.0+fork (job 2867811,
  11,786 / 23,424 / 115,671 nodes, one component each) and installed in `data/geometry` (previous in
  `previous_2026-11-12/`); new configs `uniform_200km` (13,743 nodes), `uniform_150km` (22,960) to match their node
  counts.
- **Training** (study `lambrechts_vs_uniform`, jobs 2868109–2868124, 146.6 node-h, × 16, two seeds): uniform meshes on
  valid loss = 0.153 · N^−0.060 (within 0.6 %); Lambrechts meshes above it by +1.1 %, +1.5 % and +9.7 % (full mesh:
  0.0837, worse than uniform 120 km × 16 at 0.0820; −4.7 % on `zos`, +10 to +18 % on temperature, salinity and
  currents). Report `docs/jupiter-hackathon/results/lambrechts_vs_uniform-2026-11-13`.

## 2026-11-13 11:30: distances on the meshes; horizon of the squashed grid-to-grid graph

- **Requests:** compare "mean shortest path × median edge length is the typical distance between two ocean points"
  with the mean geodesic distance between ocean points on the mesh; compute the statistics of the squashed grid-to-grid
  graph (encoder, message passing steps on each mesh, mesh-to-mesh encoders and decoders, decoder) and its horizon
  (mean geodesic distance between linked grid points), with sampling and sparse propagation (outline in the comment of
  `graph_pruning.py`; here only the graph's statistics are wanted, no pruning).
- **Code:** `structure.distance_statistics` (geodesic distances and Dijkstra path lengths along the mesh from the
  path metrics' sources, detour), `graph_statistics.squashed` (forward and backward propagation of indicator blocks of
  64 sampled grid nodes through the sparse adjacency matrices, thresholded after each operation; out- and in-degrees,
  horizon weighted by out-degree), `quantities.weighted_summarize`, `[squashed]` settings (8 and 16 steps, 1,024 grid
  nodes, at most 4,096 measured edges per sender), two new tables. Tests check the propagation against a dense matrix
  product for one mesh and a two-mesh hierarchy (forward and backward), the exact statistics with every node sampled,
  and the weighting. 498 tests pass (booster node).
- **Results** (job 2880077, 7 min on one node, all uniform meshes, icosahedral M6, 50 km + 120 km; report
  `docs/jupiter-hackathon/results/graph_horizon-2026-11-13`): hops × median edge (11,290–12,910 km) is the path
  length along the mesh (11,230–12,000 km), 14–22 % longer than the mean geodesic distance between ocean nodes
  (9,880–10,030 km); the artifact's sentence is corrected. Squashed graphs: 50 km × 16 links a grid point to 0.85 % of
  the ocean (mean 495 km, at most 907 km), 120 km × 8 to 1.2 % (623 km), icosahedral M6 × 16 to all of it, 50 km +
  120 km × 16 to 5.9 % (1,604 km).

## 2026-11-13 13:30: graph statistics: settings in the config, booleans, V-cycle with skip connections

- **Requests:** mask and mesh paths in the configuration file (absolute paths, as other commands), with the message
  passing steps of each processor mesh; check that the V-cycle accounts for the U-Net residual (the coarser result
  added to the finer mesh's state of the way down); paths of up to L hops, not exactly L (the OR of the powers in the
  `graph_pruning.py` comment); when to run forward or backward; booleans instead of float32 and thresholding.
- **Answers:** the propagation already ORs every operation into the receiving level's state, so the decoders keep
  the finer mesh's state of the way down (the skip connection) and each step is x ← x OR P x, i.e. (I + P)^L =
  I + P + … + P^L, the nodes within at most L hops (a test on a path graph and one showing that a hierarchy's squashed
  graph contains its finest mesh's). Forward gives domains of influence (out-degree, horizon), backward receptive
  fields (in-degree); pruning needs both, statistics of one end one.
- **Code:** `moby-dick graph-statistics CONFIG OUTPUT.json`; `mask` and `[[meshes]]` (`path`, `message_passing_steps`)
  in the config, absolute paths checked; one squashed graph per run with per-mesh steps; boolean sparse matrices and
  blocks (scipy's boolean product is OR of ANDs, as fast as float32); mean distance per receiver (receptive field);
  the module docstring explains the algebra, the V-cycle and the two directions. The test TOML writer writes inline
  tables. 502 tests pass (booster node).
- **Rerun** (job 2881277, 28 configs, 5.2 min on one node): degrees identical to the float32 run, horizons within
  0.09 % (edge sampling); receptive fields as wide as domains of influence (median per-node mean distance within 1 %).
  Report, CSVs, CLAUDE.md and the artifact updated.

## 2026-11-13 14:30: test of `reach` against a model

- **Request:** a test that builds a GNN with moby-dick's code and checks that perturbing an input grid point changes
  no output outside its domain of influence, and that the input gradients of an output are zero outside its
  receptive field.
- **Test** (`test_reach_matches_the_models_dependencies`, 1 and 2 steps; Leviathan with random weights, 5° grid with a
  continent, icosahedral mesh with 3 refinements; 4 grid points each): perturbed outputs change only inside the
  domain; the forward-mode derivative (JVP) is nonzero exactly on the domain and the gradient (VJP) exactly on the
  receptive field, so `reach` misses no dependency and adds none. A finite perturbation of 1 leaves some points of the
  domain unchanged: their derivative (< 6e-7) is below float32 rounding of outputs of order 1.
- **Mutation check:** with steps x <- P x (walks of exactly L hops) instead of x <- x OR P x the model test still
  passes: on a triangle mesh with edges both ways, P^2 contains I (there and back) and P (around a triangle), so P^L
  already holds every node within L hops for L >= 2, and at L = 1 the encoder reaches neighbouring vertices together.
  The OR is still the right rule in general (one-way graphs, bipartite ones); the path-graph test catches it.

## 2026-11-13 16:00: connectivity of semi-enclosed basins and inner seas

- **Request:** measure the connectivity in the squashed graph between semi-enclosed basins (Mediterranean, Black Sea)
  and inner seas (Gulf of California) and the nearby open ocean, in the report of `graph-statistics`, configurable in
  its TOML.
- **Code:** `[[regions]]` (name, polygon of (longitude, latitude) vertices; module `graph_statistics.regions`,
  matplotlib's point-in-polygon on longitudes brought to [-180, 180)) and `[[basin_connections]]` (basin, ocean) in the
  settings, names checked; `squashed.propagate` (a block whose columns mark sets of nodes: the union of their reaches),
  `steps_to_connect` (bisection over the steps on every mesh, up to `squashed.max_steps_to_connect`, default 128),
  `basin_connectivity` (exact shares of each region's nodes linked to the other, both directions); a report table. The
  committed config pairs 8 basins with their ocean (Mediterranean, Black Sea, Sea of Azov, Red Sea, Persian Gulf,
  Baltic, Gulf of California, Hudson Bay); the polygons were checked on maps of the training mask. Tests: polygons
  (longitude wrap), shares and fewest steps against the dense matrix product (one mesh and a hierarchy), validation of
  the region names. 508 tests pass (booster node).
- **Results** (job 2883258, all meshes × 8 and 16 steps, 5.4 min; report section): the Black Sea needs 28–30 steps to
  reach the Aegean on every uniform mesh (the refined straits), so it is cut off in all models trained so far since
  12 Nov; Hudson Bay and the Baltic are cut off or barely linked at 50 km × 16 (21 and 9 steps needed); the other basins
  connect within 0–6 steps. TODO item added.
