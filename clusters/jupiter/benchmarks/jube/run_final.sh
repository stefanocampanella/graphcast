#!/usr/bin/env bash
# Submit the final benchmark experiments (1024 steps per job, two repetitions) into results/<date>-<name>/.
# Each experiment is one JUBE run directory, all at the 50 depth levels of configs/training/full.toml:
#   loader      store layouts and SquashFS images x Grain workers and threads
#   io_threads  TensorStore I/O threads per reader, on the baseline zarr_v2 4x2
#   cpus        8 and 72 CPUs per task
#   scaling     1, 2 and 4 nodes
# Settings of configs/training/full.toml: processor in a scan, nothing offloaded. GPU memory limits of larger meshes:
# run_memfit.sh.
# Usage: ./run_final.sh [DATE [NAME]]   (run from this directory; DATE defaults to today, NAME to "final")
set -euo pipefail
DATE=${1:-$(date +%F)}
ROOT="results/$DATE-${2:-final}"
mkdir -p "$ROOT"

EXPERIMENTS=(
    "loader      loader"
    "io_threads  io_threads"
    "cpus        cpus"
    "scaling     scaling"
)

{
    echo "# Final GraphCast training benchmark, $DATE"
    echo
    echo "Code: $(git rev-parse --short HEAD) ($(git log -1 --format=%s))"
    echo "Every experiment: tag long (1024 steps, 3 h limit), two repetitions (rep=1,2) per case."
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
    uv run jube run graphcast_training.xml --tag $TAGS long --outpath "$ROOT/$NAME" > "$ROOT/$NAME.submit.log" 2>&1
    JOBS=$(grep -h "Submitted batch job" "$ROOT/$NAME"/000000/*_submit/work/stdout 2>/dev/null | wc -l)
    echo "| \`$NAME\` | \`$TAGS long\` | $JOBS |" >> "$ROOT/MANIFEST.md"
    echo "$NAME: $JOBS jobs ($TAGS long)"
done
