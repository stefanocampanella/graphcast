#!/usr/bin/env bash
# Submit the GPU memory fit experiments (memfit sweep at all 50 levels, 30 steps, one repetition) into
# results/<date>-<name>/: every remat setting with and without activation offloading (policy.remat_loss).
#   unrolled             processor steps unrolled
#   unrolled-preventcse  unrolled with prevent_cse
#   scan                 processor steps in a scan over stacked parameters
# scan-offload-* repeat scan-offload with less host memory for the loader (small buffers, 1 or 0 Grain workers).
# Each with "-offload" (tag offload: policy around the whole loss) and "-nooffload" (tag no_remat_loss).
# Usage: ./run_memfit.sh [DATE [NAME]]   (run from this directory; DATE defaults to today, NAME to "memfit")
set -euo pipefail
DATE=${1:-$(date +%F)}
ROOT="results/$DATE-${2:-memfit}"
mkdir -p "$ROOT"

EXPERIMENTS=(
    "unrolled-offload                 unrolled offload"
    "unrolled-nooffload               unrolled no_remat_loss"
    "unrolled-preventcse-offload      unrolled prevent_cse offload"
    "unrolled-preventcse-nooffload    unrolled prevent_cse no_remat_loss"
    "scan-offload                     scan offload"
    "scan-nooffload                   scan no_remat_loss"
    # Offloading with less host memory taken by the loader: small buffers, 1 Grain worker, threads only.
    "scan-offload-smallbuffers        scan offload small_buffers"
    "scan-offload-1worker             scan offload small_buffers one_worker"
    "scan-offload-0workers            scan offload small_buffers no_workers"
)

{
    echo "# GPU memory fit at all 50 levels, $DATE"
    echo
    echo "Code: $(git rev-parse --short HEAD) ($(git log -1 --format=%s))"
    echo "Every experiment: tags memfit all_levels quick (30 steps, one repetition) plus those below."
    echo "Results: <experiment>/000000/result/result_csv.dat (all columns) and result.dat (table)."
    echo
    echo "| Experiment | JUBE tags | Jobs |"
    echo "|---|---|---|"
} > "$ROOT/MANIFEST.md"

for entry in "${EXPERIMENTS[@]}"; do
    read -r NAME TAGS <<< "$entry"
    if [[ -d $ROOT/$NAME ]]; then
        echo "skipping $NAME: $ROOT/$NAME exists" >&2
        continue
    fi
    # shellcheck disable=SC2086
    uv run jube run graphcast_training.xml --tag memfit all_levels quick $TAGS --outpath "$ROOT/$NAME" \
        > "$ROOT/$NAME.submit.log" 2>&1
    JOBS=$(grep -h "Submitted batch job" "$ROOT/$NAME"/000000/*_submit/work/stdout 2>/dev/null | wc -l)
    echo "| \`$NAME\` | \`memfit all_levels quick $TAGS\` | $JOBS |" >> "$ROOT/MANIFEST.md"
    echo "$NAME: $JOBS jobs (memfit all_levels quick $TAGS)"
done
