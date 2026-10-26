#!/usr/bin/env bash
#SBATCH --job-name=download
#SBATCH --account=e-dev-2026d09-119
#SBATCH --partition=booster
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: sbatch clusters/jupiter/slurm/download.slurm [--env-file FILE] DATASET CLIMATOLOGY ANOMALY_STD OUTPUT
                                                   [--config FILE] [--overwrite] [--log-level LEVEL]

Write the standardized anomaly dataset on a compute node, using all of its CPUs (no GPUs). Compute nodes have no
internet access: DATASET, CLIMATOLOGY and ANOMALY_STD must be local paths (e.g. the copies in
/e/fscratch/e-dev-2026d09-119/ogs-arco-ocean-50*). Arguments are forwarded to scripts/download.sh (submit from the
project directory); settings such as threads are in the --config TOML (configs/download).
For short tests, add --reservation=develbooster before the script path.
EOF
}

if [[ ${1:-} == --help ]]; then
    usage
    exit 0
fi

# sbatch runs a copy of this script: find scripts/download.sh from the submit directory.
PROJECT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
[[ -x $PROJECT_DIR/scripts/download.sh ]] || {
    printf 'Jupiter download.slurm: %s/scripts/download.sh not found; submit from the project directory\n' \
        "$PROJECT_DIR" >&2
    exit 1
}
printf 'Job %s on %s: %s CPUs\n' "${SLURM_JOB_ID:-}" "$(hostname)" "${SLURM_CPUS_PER_TASK:-?}"
exec "$PROJECT_DIR/scripts/download.sh" "$@"
