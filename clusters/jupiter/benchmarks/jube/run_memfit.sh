#!/usr/bin/env bash
# Submit the GPU memory fit experiments (memfit sweep at all 50 levels, 30 steps, one repetition) into
# results/<date>-<name>/, for the settings that matter for GPU memory. The defaults of full.toml: processor in a
# scan, grid nodes only for connected grid points, nothing offloaded, no edge chunks.
#   unrolled                          processor steps unrolled
#   scan                              the defaults
#   scan-allgrid                      grid nodes for all grid points (tag all_grid_nodes)
#   scan-chunks256k, scan-chunks1m    edge updates in chunks of 262144 or 1048576 edges
#   scan-offload-0workers             offloading, with Grain threads only (less host memory for the loader)
#   scan-offload-chunks256k-0workers  offloading and edge chunks: the largest models
# Every case logs the memory XLA plans for a step ("Memory usage" in task-0.err), also when it does not fit.
# Usage: ./run_memfit.sh [DATE [NAME]]   (run from this directory; DATE defaults to today, NAME to "memfit")
set -euo pipefail
DATE=${1:-$(date +%F)}
ROOT="results/$DATE-${2:-memfit}"
mkdir -p "$ROOT"

EXPERIMENTS=(
    "unrolled                          unrolled no_remat_loss"
    "scan                              scan no_remat_loss"
    "scan-allgrid                      scan no_remat_loss all_grid_nodes"
    "scan-chunks256k                   scan no_remat_loss edge_chunks"
    "scan-chunks1m                     scan no_remat_loss edge_chunks_1m"
    "scan-offload-0workers             scan offload small_buffers no_workers"
    "scan-offload-chunks256k-0workers  scan offload small_buffers no_workers edge_chunks"
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
