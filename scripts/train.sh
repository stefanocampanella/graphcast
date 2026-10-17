#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/train.sh [options] CONFIG_NAME OUTPUT_NAME TRAIN_CKPT_NAME

Run GraphCast training in Apptainer using the inherited environment.
Options override environment defaults; relative host paths use PROJECT_DIR.
  --project-dir DIR     Project directory (PROJECT_DIR or current directory)
  --image FILE          Image (APPTAINER_IMAGE or PROJECT_DIR/graphcast.sif)
  --bind-data-dir DIR   Container data destination (DATA_DIR or /data)
  --config-dir DIR      Config directory (CONFIG_DIR or PROJECT_DIR/configs/training)
  --output-dir DIR      Model directory (OUTPUT_DIR or PROJECT_DIR/data/models)
  --train-dir DIR       Checkpoints (TRAIN_DIR or DATA_ROOT/training)
  --tb-dir DIR          TensorBoard logs (TB_DIR or DATA_ROOT/tensorboard)
  --bind SRC[:DST[:OPTS]]  Additional bind, repeatable; DST must be absolute
  --continue           Continue training
  --overwrite          Overwrite destination model
  --log-level LEVEL    debug, info, warning, error, critical (default: info)
  --help               Show help without checking dependencies

DATA_ROOT defaults to PROJECT_DIR/data. Required readable SquashFS files:
DATASET_SQSH, CLIMATOLOGY_SQSH, ANOMALY_STD_SQSH, TENDENCY_STD_SQSH.
Extra geometry/normalization inputs depend on the config; see README.md.
Execution requires GNU getopt (util-linux); --help has no dependencies.
Bind paths cannot contain ':' or ',' (Apptainer separators).
Site setup, --env-file and --log-dir belong to the cluster launcher.
The Python CLI still requires automatic JAX distributed initialization.
EOF
}

fail() {
    printf 'train.sh: %s\n' "$*" >&2
    exit 1
}

absolute_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$PROJECT_DIR" "$1" ;;
    esac
}

check_bind_path() {
    [[ $1 != *:* && $1 != *,* ]] || fail "bind path contains an Apptainer separator (: or ,): $1"
}

readable_file() {
    [[ -f $2 && -r $2 ]] || fail "$1 is not a readable file: $2"
}

writable_directory() {
    local directory
    directory=$(absolute_path "$2")
    check_bind_path "$directory"
    mkdir -p -- "$directory" || fail "cannot create $1: $directory"
    [[ -w $directory ]] || fail "$1 is not writable: $directory"
    printf '%s\n' "$directory"
}

add_squashfs_bind() {
    local source
    [[ -n $2 ]] || fail "$1 must name a readable SquashFS file"
    source=$(absolute_path "$2")
    readable_file "$1" "$source"
    check_bind_path "$source"
    APPTAINER_ARGS+=(--bind "$source:$DATA_DIR/$3:image-src=/")
}

# Parse execution options. GNU getopt handles quoting, --option=value and
# options interspersed with positional names, as in the original batch script.
EXTRA_BINDS=()
CONTINUE=false
OVERWRITE=false
LOG_LEVEL=${LOG_LEVEL:-info}

# Help must work even when getopt or Apptainer is not installed.
if [[ $# == 1 && $1 == --help ]]; then
    usage
    exit 0
fi

ARGS=$(getopt --shell bash --options '' \
    --longoptions 'project-dir:,image:,bind-data-dir:,config-dir:,output-dir:,train-dir:,tb-dir:,bind:,continue,overwrite,log-level:,help' \
    --name train.sh -- "$@") || fail 'unknown option or missing value (see --help); GNU getopt is required'
eval "set -- $ARGS"

while true; do
    # All options except these switches require a nonempty value.
    case "$1" in
        --continue|--overwrite|--help|--) ;;
        *)
            [[ -n $2 ]] || fail "empty value for $1"
            [[ $2 != --* ]] || fail "missing value for $1"
            ;;
    esac
    case "$1" in
        --project-dir)
            PROJECT_DIR=$2
            shift 2
            ;;
        --image)
            APPTAINER_IMAGE=$2
            shift 2
            ;;
        --bind-data-dir)
            DATA_DIR=$2
            shift 2
            ;;
        --config-dir)
            CONFIG_DIR=$2
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        --train-dir)
            TRAIN_DIR=$2
            shift 2
            ;;
        --tb-dir)
            TB_DIR=$2
            shift 2
            ;;
        --bind)
            EXTRA_BINDS+=("$2")
            shift 2
            ;;
        --log-level)
            LOG_LEVEL=$2
            shift 2
            ;;
        --continue)
            CONTINUE=true
            shift
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
    esac
done

[[ $# == 3 ]] || fail 'expected exactly CONFIG_NAME OUTPUT_NAME TRAIN_CKPT_NAME (see --help)'
CONFIG_NAME=$1
OUTPUT_NAME=$2
TRAIN_CKPT_NAME=$3
[[ -n $CONFIG_NAME && -n $OUTPUT_NAME && -n $TRAIN_CKPT_NAME ]] || fail 'positional names must not be empty'
case "$LOG_LEVEL" in
    debug|info|warning|error|critical) ;;
    *) fail "invalid log level: $LOG_LEVEL" ;;
esac

# Resolve defaults after parsing so explicit options win over the environment.
PROJECT_DIR=${PROJECT_DIR:-$PWD}
[[ -d $PROJECT_DIR ]] || fail "project directory not found: $PROJECT_DIR"
cd -- "$PROJECT_DIR" || fail "cannot enter project directory: $PROJECT_DIR"
PROJECT_DIR=$PWD
DATA_DIR=${DATA_DIR:-/data}
[[ $DATA_DIR == /* ]] || fail "container data destination must be absolute: $DATA_DIR"
DATA_DIR=${DATA_DIR%/}
DATA_ROOT=${DATA_ROOT:-$PROJECT_DIR/data}
APPTAINER_IMAGE=${APPTAINER_IMAGE:-$PROJECT_DIR/graphcast.sif}
CONFIG_DIR=${CONFIG_DIR:-$PROJECT_DIR/configs/training}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/data/models}
TRAIN_DIR=${TRAIN_DIR:-$DATA_ROOT/training}
TB_DIR=${TB_DIR:-$DATA_ROOT/tensorboard}

APPTAINER_IMAGE=$(absolute_path "$APPTAINER_IMAGE")
readable_file 'Apptainer image' "$APPTAINER_IMAGE"
CONFIG_DIR=$(absolute_path "$CONFIG_DIR")
CONFIG_PATH="$CONFIG_DIR/$CONFIG_NAME.toml"
readable_file 'training config' "$CONFIG_PATH"
check_bind_path "$DATA_DIR"

# Preserve the four existing SquashFS destinations; extra inputs use --bind.
APPTAINER_ARGS=(run --nv --pwd "$PROJECT_DIR")
add_squashfs_bind DATASET_SQSH "${DATASET_SQSH:-}" dataset
add_squashfs_bind CLIMATOLOGY_SQSH "${CLIMATOLOGY_SQSH:-}" climatology
add_squashfs_bind ANOMALY_STD_SQSH "${ANOMALY_STD_SQSH:-}" anomaly_std
add_squashfs_bind TENDENCY_STD_SQSH "${TENDENCY_STD_SQSH:-}" tendency_std

# The conditional array expansion supports empty arrays under Bash 3's nounset.
for BIND in ${EXTRA_BINDS[@]+"${EXTRA_BINDS[@]}"}; do
    [[ $BIND != *,* ]] || fail "use a separate --bind for each source, without commas: $BIND"
    SOURCE=${BIND%%:*}
    [[ -n $SOURCE ]] || fail "empty bind source: $BIND"
    SOURCE=$(absolute_path "$SOURCE")
    [[ -e $SOURCE && -r $SOURCE ]] || fail "bind source does not exist or is unreadable: $SOURCE"
    if [[ $BIND == *:* ]]; then
        DESTINATION=${BIND#*:}
        [[ ${DESTINATION%%:*} == /* ]] || fail "bind destination must be absolute: $BIND"
        APPTAINER_ARGS+=(--bind "$SOURCE:$DESTINATION")
    else
        APPTAINER_ARGS+=(--bind "$SOURCE:$SOURCE")
    fi
done

command -v apptainer >/dev/null 2>&1 || fail 'apptainer not found; install it or load it in the launcher environment'

# Only shared parent directories are created; Python owns checkpoint lifecycle.
OUTPUT_DIR=$(writable_directory OUTPUT_DIR "$OUTPUT_DIR")
TRAIN_DIR=$(writable_directory TRAIN_DIR "$TRAIN_DIR")
TB_DIR=$(writable_directory TB_DIR "$TB_DIR")
OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_NAME"
TRAIN_PATH="$TRAIN_DIR/$TRAIN_CKPT_NAME"
mkdir -p -- "$(dirname -- "$OUTPUT_PATH")" "$(dirname -- "$TRAIN_PATH")"
for DIRECTORY in "$PROJECT_DIR" "$CONFIG_DIR" "$OUTPUT_DIR" "$TRAIN_DIR" "$TB_DIR"; do
    check_bind_path "$DIRECTORY"
    APPTAINER_ARGS+=(--bind "$DIRECTORY:$DIRECTORY")
done

# Replace the shell so the container's exit status reaches the caller unchanged.
TRAIN_ARGS=(
    "--log-level=$LOG_LEVEL"
    "--data-path=${DATA_DIR:-/}"
    "--tensorboard-logdir=$TB_DIR"
)
if $OVERWRITE; then
    TRAIN_ARGS+=(--overwrite)
else
    TRAIN_ARGS+=(--no-overwrite)
fi
if $CONTINUE; then
    TRAIN_ARGS+=(--continue)
fi
exec apptainer "${APPTAINER_ARGS[@]}" "$APPTAINER_IMAGE" graphcast train \
    "${TRAIN_ARGS[@]}" -- "$CONFIG_PATH" "$OUTPUT_PATH" "$TRAIN_PATH"