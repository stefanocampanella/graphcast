#!/usr/bin/env bash
#SBATCH --job-name=test_training
#SBATCH --account=e-dev-2026d09-119
#SBATCH --partition=booster
#SBATCH --reservation=develbooster
#SBATCH --time=30
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpu-bind=single:1

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: sbatch clusters/jupiter/slurm/train.slurm [options] CONFIG_NAME OUTPUT_NAME TRAIN_CKPT_NAME

Jupiter launcher options:
  --project-dir DIR  Project (default: SLURM_SUBMIT_DIR or current directory)
  --env-file FILE    Bash environment (default: PROJECT_DIR/clusters/jupiter/.env)
  --log-dir DIR     Task logs (default: LOG_DIR or PROJECT_DIR/logs/training)
  --help            Show this help without site setup

Execution options are forwarded to scripts/train.sh; see scripts/train.sh --help.
Relative environment/log paths use PROJECT_DIR. SBATCH directives are literal;
use sbatch options before the script path to override resource allocations.
EOF
}

fail() {
    printf 'Jupiter train.slurm: %s\n' "$*" >&2
    exit 1
}

require_option_value() {
    (($# >= 2)) || fail "missing value for $1"
    [[ $2 != --* ]] || fail "missing value for $1"
    [[ -n $2 ]] || fail "empty value for $1"
}

compactify_logs() {
    local extension combined file
    [[ -n ${SLURM_JOB_ID:-} ]] || return 0
    for extension in out err; do
        combined="$LOG_DIR/${SLURM_JOB_ID}.$extension"
        for file in "$LOG_DIR/${SLURM_JOB_ID}_"*."$extension"; do
            [[ -f $file && ! -L $file ]] || continue
            if {
                printf '===== %s =====\n' "${file##*/}" && cat -- "$file"
            } >> "$combined"; then
                rm -f -- "$file" || printf 'Warning: could not remove task log %s\n' "$file" >&2
            else
                printf 'Warning: could not compact task log %s; original retained\n' "$file" >&2
            fi
        done
    done
}

cleanup() {
    local status=$?
    local index
    trap - EXIT INT TERM
    set +e
    if $STEP_STARTED; then
        compactify_logs
    fi
    for ((index = ${#LOADED_MODULES[@]} - 1; index >= 0; index--)); do
        module unload "${LOADED_MODULES[index]}" || printf 'Warning: module unload failed\n' >&2
    done
    exit "$status"
}

# Extract only launcher settings. Keep execution arguments byte-for-byte rather
# than normalizing them with getopt; the runner owns their parsing/validation.
LAUNCH_PROJECT_DIR=${SLURM_SUBMIT_DIR:-$PWD}
LAUNCH_ENV_FILE=
LAUNCH_LOG_DIR=
FORWARDED_ARGS=()
while (($#)); do
    case "$1" in
        --help)
            usage
            exit 0
            ;;
        --project-dir=*|--env-file=*|--log-dir=*)
            # Normalize launcher --option=value to the two-argument form below.
            set -- "${1%%=*}" "${1#*=}" "${@:2}"
            ;;
        --project-dir)
            require_option_value "$@"
            LAUNCH_PROJECT_DIR=$2
            shift 2
            ;;
        --env-file)
            require_option_value "$@"
            LAUNCH_ENV_FILE=$2
            shift 2
            ;;
        --log-dir)
            require_option_value "$@"
            LAUNCH_LOG_DIR=$2
            shift 2
            ;;
        --image|--bind-data-dir|--config-dir|--output-dir|--train-dir|--tb-dir|--bind|--log-level)
            # A runner value may itself look like a launcher option.
            (($# >= 2)) || fail "missing value for $1"
            FORWARDED_ARGS+=("$1" "$2")
            shift 2
            ;;
        --)
            FORWARDED_ARGS+=("$@")
            break
            ;;
        *)
            FORWARDED_ARGS+=("$1")
            shift
            ;;
    esac
done

# Source the site environment once, exporting its scalar settings to each task.
[[ -d $LAUNCH_PROJECT_DIR ]] || fail "project directory not found: $LAUNCH_PROJECT_DIR"
cd -- "$LAUNCH_PROJECT_DIR" || fail "cannot enter project directory: $LAUNCH_PROJECT_DIR"
LAUNCH_PROJECT_DIR=$PWD
LAUNCH_ENV_FILE=${LAUNCH_ENV_FILE:-$LAUNCH_PROJECT_DIR/clusters/jupiter/.env}
[[ -f $LAUNCH_ENV_FILE && -r $LAUNCH_ENV_FILE ]] || fail "Bash environment file not readable: $LAUNCH_ENV_FILE"
MODULES=()
set -a
# shellcheck source=/dev/null
source "$LAUNCH_ENV_FILE"
set +a

# Explicit launcher paths take precedence over settings from the site file.
PROJECT_DIR=$LAUNCH_PROJECT_DIR
LOG_DIR=${LAUNCH_LOG_DIR:-${LOG_DIR:-$PROJECT_DIR/logs/training}}
[[ $LOG_DIR == /* ]] || LOG_DIR="$PROJECT_DIR/$LOG_DIR"
[[ -x $PROJECT_DIR/scripts/train.sh ]] || fail "runner is not executable: $PROJECT_DIR/scripts/train.sh"
mkdir -p -- "$LOG_DIR" || fail "cannot create log directory: $LOG_DIR"
[[ -w $LOG_DIR ]] || fail "log directory is not writable: $LOG_DIR"

# Remember only successfully loaded modules, and always preserve the step status.
LOADED_MODULES=()
STEP_STARTED=false
trap cleanup EXIT
# Bash defers these traps until the foreground step ends. Cleanup is best effort.
trap 'exit 130' INT
trap 'exit 143' TERM
# Conditional array expansion supports empty arrays under Bash 3's nounset.
for MODULE in ${MODULES[@]+"${MODULES[@]}"}; do
    module load "$MODULE"
    LOADED_MODULES+=("$MODULE")
done

# One shared runner/container per task; only this launcher creates a job step.
STEP_STARTED=true
srun --export=ALL --kill-on-bad-exit=1 \
    --output="$LOG_DIR/%j_%N_%t.out" --error="$LOG_DIR/%j_%N_%t.err" \
    "$PROJECT_DIR/scripts/train.sh" --project-dir "$PROJECT_DIR" \
    ${FORWARDED_ARGS[@]+"${FORWARDED_ARGS[@]}"}