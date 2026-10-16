#! /usr/bin/env bash
#SBATCH --job-name=test_training
#SBATCH --account=e-dev-2026d09-119
#SBATCH --partition=booster
#SBATCH --reservation=develbooster
#SBATCH --time=30
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --gpu-bind=single:1

usage() {
    more <<EOF
train.slurm
    This script trains the forked GraphCast model on ARCO-OCEAN data on Jupiter.

SYNOPSIS
    usage: $0 --help
    usage: $0 [options] CONFIG_NAME OUTPUT_NAME TRAIN_CKPT_NAME

DESCRIPTION
    Options
        --project-dir                           Project root directory (default to current working directory).
        --env-file FILE                         Environment file to source (default \$PROJECT_ROOT/machines/jupiter/.env).
        --image                                 Apptainer image path (default \$PROJECT_ROOT/graphcast.sif).
        --bind-data-dir                         Where to bind-mount datasets within the container (default to /data).
        --config-dir                            Where to look for configs (default to \$PROJECT_ROOT/configs/training).
        --output-dir                            Where to place the outputs (default to \$PROJECT_ROOT/data/models).
        --train-dir                             Where to place training checkpoints (default to \$DATA_ROOT/training).
        --tb-dir                                Where to place tensorboard logs (default to \$DATA_ROOT/tensorboard).
        --log-dir                               Where to place the logs (default to \$PROJECT_ROOT/logs/training).
        --bind SRC[:DST[:OPTS]]                 Additional path to bind within the Apptainer container (can be repeated).
        --overwrite                             Whether to overwrite destination (default false).
        --continue                              Whether to continue previous training job.
        --log-level LEVEL                       Logging level (debug, info, warning, error, critical). Default: info.
        --help                                  Shows this help.
EOF
}

# Default options
BIND_DATA_DIR="/data"
CONTINUE=false
OVERWRITE=false
LOG_LEVEL="info"
BINDS=()
CLI_OPTIONS=()

# Parse command line options
ARGS=$(getopt --options '' --longoptions "project-dir:,env-file:,image:,bind-data-dir:,config-dir:,output-dir:,train-dir:,tb-dir:,log-dir:,bind:,continue,overwrite,log-level:,help" --name "$0" -- "${@}")
if [[ ${?} -ne 0 ]]; then
    usage
    exit 1
fi

eval "set -- ${ARGS}"
while true; do
    case "$1" in
        (--project-dir)
            PROJECT_ROOT="${2}"
            shift 2
            ;;
        (--env-file)
            ENV_FILE="${2}"
            shift 2
            ;;
        (--image)
            IMAGE_OVERRIDE="${2}"
            shift 2
            ;;
        (--bind-data-dir)
            BIND_DATA_DIR="${2}"
            shift 2
            ;;
        (--config-dir)
            CONFIG_DIR="${2}"
            shift 2
            ;;
        (--output-dir)
            OUTPUT_DIR="${2}"
            shift 2
            ;;
        (--train-dir)
            TRAINING_DIR="${2}"
            shift 2
            ;;
        (--tb-dir)
            TENSORBOARD_LOGDIR="${2}"
            shift 2
            ;;
        (--log-dir)
            LOG_DIR="${2}"
            shift 2
            ;;
        (--bind)
            BINDS+=("${2}")
            shift 2
            ;;
        (--continue)
            CONTINUE=true
            shift
            ;;
        (--overwrite)
            OVERWRITE=true
            shift
            ;;
        (--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        (--help)
            usage
            exit 0
            ;;
        (--)
            shift
            break
            ;;
        (*)
            echo "Error: unrecognized option $1" >&2
            usage
            exit 1
            ;;
    esac
done

# Remaining args: config name, output name, and checkpoint name
if [[ $# -lt 3 ]]; then
    echo "Error: missing required arguments: config name, output name, and training checkpoint name" >&2
    usage
    exit 1
fi

# Change current directory to project root
if [[ -z "${PROJECT_ROOT}" ]]; then
    PROJECT_ROOT="$(pwd)"
fi
cd "${PROJECT_ROOT}" || exit 1

# Source env file first to populate DATA_ROOT and default variables
if [[ -z ${ENV_FILE} ]]; then
    ENV_FILE="${PROJECT_ROOT}/machines/jupiter/.env"
fi
if [[ -f ${ENV_FILE} ]]; then
    set -a
    # shellcheck source=machines/jupiter/.env
    source "${ENV_FILE}"
    set +a
else
    echo "Error: environment file ${ENV_FILE} not found."
    exit 1
fi

if [[ -n "${IMAGE_OVERRIDE}" ]]; then
    APPTAINER_IMAGE="${IMAGE_OVERRIDE}"
elif [[ -z "${APPTAINER_IMAGE}" ]]; then
    APPTAINER_IMAGE="${PROJECT_ROOT}/graphcast.sif"
fi

# Locate config file
if [[ -z ${CONFIG_DIR} ]]; then
    CONFIG_DIR="${PROJECT_ROOT}/configs/training"
fi
CONFIG_PATH="${CONFIG_DIR}/${1}.toml"
shift 1

# Locate output model
if [[ -z ${OUTPUT_DIR} ]]; then
    OUTPUT_DIR="${PROJECT_ROOT}/data/models"
fi
mkdir -p "${OUTPUT_DIR}"
OUTPUT_CKPT_PATH="${OUTPUT_DIR}/${1}"
shift 1

# Locate training checkpoints
if [[ -z ${TRAINING_DIR} ]]; then
    TRAINING_DIR="${DATA_ROOT}/training"
fi
mkdir -p "${TRAINING_DIR}"
TRAIN_CKPT_PATH="${TRAINING_DIR}/${1}"
shift 1

# Locate log directory
if [[ -z ${LOG_DIR} ]]; then
    LOG_DIR="${PROJECT_ROOT}/logs/training"
fi
mkdir -p "${LOG_DIR}"

# Locate TensorBoard directory
if [[ -z ${TENSORBOARD_LOGDIR} ]]; then
    TENSORBOARD_LOGDIR="${DATA_ROOT}/tensorboard"
fi
mkdir -p "${TENSORBOARD_LOGDIR}"

# Load default modules
for MODULE_TO_LOAD in "${MODULES[@]}"; do
    module load "${MODULE_TO_LOAD}"
done

# Check that apptainer is installed and apptainer image exists
apptainer --version || exit 1
if [[ -z ${APPTAINER_IMAGE} ]]; then
    echo "Error: APPTAINER_IMAGE not defined."
    exit 1
fi
if [[ ! -f ${APPTAINER_IMAGE} ]]; then
    echo "Error: APPTAINER_IMAGE file '${APPTAINER_IMAGE}' not found."
    exit 1
fi

# Build apptainer flags for datasets
if [[ -z ${DATASET_SQSH} ]]; then
    echo "Error: DATASET_SQSH image file not defined."
    exit 1
fi
if [[ ! -f ${DATASET_SQSH} ]]; then
    echo "Error: DATASET_SQSH image file '${DATASET_SQSH}' not found."
    exit 1
fi
BINDS+=("${DATASET_SQSH}:${BIND_DATA_DIR}/dataset:image-src=/")

if [[ -z ${CLIMATOLOGY_SQSH} ]]; then
    echo "Error: CLIMATOLOGY_SQSH image file not defined."
    exit 1
fi
if [[ ! -f ${CLIMATOLOGY_SQSH} ]]; then
    echo "Error: CLIMATOLOGY_SQSH image file '${CLIMATOLOGY_SQSH}' not found."
    exit 1
fi
BINDS+=("${CLIMATOLOGY_SQSH}:${BIND_DATA_DIR}/climatology:image-src=/")

if [[ -z ${ANOMALY_STD_SQSH} ]]; then
    echo "Error: ANOMALY_STD_SQSH image file not defined."
    exit 1
fi
if [[ ! -f ${ANOMALY_STD_SQSH} ]]; then
    echo "Error: ANOMALY_STD_SQSH image file '${ANOMALY_STD_SQSH}' not found."
    exit 1
fi
BINDS+=("${ANOMALY_STD_SQSH}:${BIND_DATA_DIR}/anomaly_std:image-src=/")

if [[ -z ${TENDENCY_STD_SQSH} ]]; then
    echo "Error: TENDENCY_STD_SQSH image file not defined."
    exit 1
fi
if [[ ! -f ${TENDENCY_STD_SQSH} ]]; then
    echo "Error: TENDENCY_STD_SQSH image file '${TENDENCY_STD_SQSH}' not found."
    exit 1
fi
BINDS+=("${TENDENCY_STD_SQSH}:${BIND_DATA_DIR}/tendency_std:image-src=/")

APPTAINER_FLAGS=()
for BIND in "${BINDS[@]+"${BINDS[@]}"}"; do
    APPTAINER_FLAGS+=(--bind "${BIND}")
done
RUN_CMD=(apptainer run --nv "${APPTAINER_FLAGS[@]+"${APPTAINER_FLAGS[@]}"}" "${APPTAINER_IMAGE}")

# Build command line flags
CLI_OPTIONS+=("--log-level=${LOG_LEVEL}" "--data-path=${BIND_DATA_DIR}" "--tensorboard-logdir=${TENSORBOARD_LOGDIR}")
if [[ ${OVERWRITE} == true ]]; then
    CLI_OPTIONS+=("--overwrite")
else
    CLI_OPTIONS+=("--no-overwrite")
fi
if [[ ${CONTINUE} == true ]]; then
    CLI_OPTIONS+=("--continue")
fi

# Compactify logs: concatenate all logs for this job and remove the originals
compactify_logs() {
    local ext combined f
    for ext in out err; do
        combined="${LOG_DIR}/${SLURM_JOB_ID}.${ext}"
        while IFS= read -r f; do
            [[ -e "${f}" ]] || continue
            echo "===== ${f##*/} =====" >> "${combined}"
            cat "${f}" >> "${combined}"
            rm -f "${f}"
        done < <(printf '%s\n' "${LOG_DIR}/${SLURM_JOB_ID}"_*."${ext}" | sort -V)
    done
}
trap compactify_logs EXIT

srun --output="${LOG_DIR}"/%j_%N_%t.out --error="${LOG_DIR}"/%j_%N_%t.err \
  "${RUN_CMD[@]}" graphcast train "${CLI_OPTIONS[@]+"${CLI_OPTIONS[@]}"}" -- "${CONFIG_PATH}" "${OUTPUT_CKPT_PATH}" "${TRAIN_CKPT_PATH}"

TRAINING_EXIT_CODE=$?

if [[ ${TRAINING_EXIT_CODE} -eq 0 ]]; then
    echo "----------------------------------------"
    echo "Training completed successfully."
else
    echo "----------------------------------------"
    echo "Training failed with exit code: ${TRAINING_EXIT_CODE}"
    exit ${TRAINING_EXIT_CODE}
fi

# Deactivate all environments
for MODULE_TO_LOAD in "${MODULES[@]}"; do
    module unload "${MODULE_TO_LOAD}"
done