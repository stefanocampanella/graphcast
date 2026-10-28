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
