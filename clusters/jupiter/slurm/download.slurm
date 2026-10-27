#!/usr/bin/env bash
#SBATCH --job-name=download
#SBATCH --account=e-dev-2026d09-119
#SBATCH --partition=booster
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: sbatch clusters/jupiter/slurm/download.slurm [--env-file FILE] DATASET CLIMATOLOGY ANOMALY_STD OUTPUT
                                                   [--config FILE] [--overwrite] [--log-level LEVEL]

Write the standardized anomaly dataset on a compute node, using all of its CPUs (no GPUs). Compute nodes have no
internet access: DATASET, CLIMATOLOGY and ANOMALY_STD must be local paths. On a login node, where S3 URLs work, run
this script with bash instead of sbatch (and fewer threads in the --config TOML).
  --env-file FILE   Bash environment (default: PROJECT_DIR/clusters/jupiter/.env), providing APPTAINER_IMAGE and
                    APPTAINERENV_PYTHONPATH
Other arguments are forwarded to scripts/download.sh (submit from the project directory); see its --help.
For short tests, add --reservation=develbooster before the script path.
EOF
}

fail() {
    printf 'Jupiter download.slurm: %s\n' "$*" >&2
    exit 1
}

# sbatch runs a copy of this script: find the project from the submit directory.
PROJECT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
ENV_FILE=$PROJECT_DIR/clusters/jupiter/.env
case "${1:-}" in
    --help)
        usage
        exit 0
        ;;
    --env-file)
        [[ -n ${2:-} ]] || fail 'missing value for --env-file'
        ENV_FILE=$2
        shift 2
        ;;
esac
[[ -x $PROJECT_DIR/scripts/download.sh ]] || fail "$PROJECT_DIR/scripts/download.sh not found; submit from the project directory"
[[ -f $ENV_FILE && -r $ENV_FILE ]] || fail "environment file is not readable: $ENV_FILE"
set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

printf 'Job %s on %s: %s CPUs\n' "${SLURM_JOB_ID:-}" "$(hostname)" "${SLURM_CPUS_PER_TASK:-?}"
exec "$PROJECT_DIR/scripts/download.sh" "$@"
