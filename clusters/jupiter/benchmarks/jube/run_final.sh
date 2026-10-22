#!/usr/bin/env bash
# Submit the final benchmark experiments (1024 steps per job, two repetitions) into results/<date>-<name>/.
# Each experiment is one JUBE run directory, named <study>-<data>-<variant>:
#   io-*     data loading: format, Grain and zarr settings, CPUs, nodes (10 levels), memory at all levels
#   xla-*    XLA/JAX optimization settings (opt_env) with the fastest loader, 8 and 16 message passing steps
#   model-*  model size: coarse mesh, 8 to 16 message passing steps (the fine mesh does not fit in GPU memory)
# Usage: ./run_final.sh [DATE [NAME]]   (run from this directory; DATE defaults to today, NAME to "final")
set -euo pipefail
DATE=${1:-$(date +%F)}
ROOT="results/$DATE-${2:-final}"
mkdir -p "$ROOT"

EXPERIMENTS=(
    "io-10levels                     format grain zarr cpus scaling"
    "io-alllevels-smallbuffers       lowmem all_levels small_buffers"
    "io-alllevels-smallbuffers-nooffload lowmem all_levels small_buffers no_offload"
    "io-alllevels-workers            workers all_levels"
    "io-alllevels-workers-nooffload  workers all_levels no_offload"
    "xla-coarse-msg8                 opt"
    "xla-coarse-msg16                opt msg16"
    "model-coarse-msg8to16           model"
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
