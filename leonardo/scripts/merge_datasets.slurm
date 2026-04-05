#! /usr/bin/env bash
#SBATCH --account=OGS23_PRACE_IT_1
#SBATCH --partition=dcgp_usr_prod
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --time=120
#SBATCH --gres=tmpfs:10G
#SBATCH --requeue

usage() {
    more <<EOF
merge_dataset.slurm
    This script merges multiple downloaded datasets into a single dataset.
    It reads from the data/dataset directory and outputs a merged dataset.

SYNOPSIS
    usage: $0 --help
    usage: $0 [options] config

DESCRIPTION
    Options
        --no-array                              Whether to merge dataset in a single SLURM job (default false).
        --overwrite                             Whether to overwrite destination (default false).
        --start                                 Overwrite config start date.
        --end                                   Overwrite config end date.
        --output-dir                            Where to place the ouputs (default to $LEONARDO_DATA_ROOT/data/dataset).
        --data-dir                              Where to look for data (defaut to $LEONARDO_DATA_ROOT/data/dataset).
        --debug                                 Use synchronous Dask scheduler (default false).
        --progress                              Whether to display a progress bar (default false).
        --help                                  Shows this help.
EOF
}

# Options default values
OVERWRITE=false
PROGRESS=false
DEBUG=false
ARRAY=true

# Parse command line options
ARGS=$(getopt --options '' --longoptions "overwrite,no-array,start:,end:,output-dir:,data-dir:,debug,progress,help" --name "$0" -- "${@}")
if [[ ${?} -ne 0 ]]; then
    usage
    exit 1
fi
eval "set -- ${ARGS}"
while true; do
    case "${1}" in
        (--no-array)
            ARRAY=false
            shift
            ;;
        (--overwrite)
            OVERWRITE=true
            shift
            ;;
        (--start)
            START_DATE="${2}"
            shift 2
            ;;
        (--end)
            END_DATE="${2}"
            shift 2
            ;;
        (--output-dir)
            OUTPUT_DIR="${2}"
            shift 2
            ;;
        (--data-dir)
            DATA_DIR="${2}"
            shift 2
            ;;
        (--debug)
            DEBUG=true
            shift
            ;;
        (--progress)
            PROGRESS=true
            shift
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
            echo "Error: unrecognized option ${1}."
            usage
            exit 1
            ;;
    esac
done

PROJECTROOT=$(git rev-parse --show-toplevel)
cd "${PROJECTROOT}" || exit 1

# Activate environment and set default path envs
source leonardo/environment/activate
source leonardo/.env

# Build command line flags
FLAGS=""

if [[ ${OVERWRITE} == true ]]; then
    FLAGS="${FLAGS} --overwrite"
else
    FLAGS="${FLAGS} --no-overwrite"
fi

if [[ ${PROGRESS} == true ]]; then
    FLAGS="${FLAGS} --progress"
else
    FLAGS="${FLAGS} --no-progress"
fi

if [[ ${DEBUG} == true ]]; then
    FLAGS="${FLAGS} --debug"
else
    FLAGS="${FLAGS} --no-debug"
fi

FLAGS="${FLAGS} --log-level=info"

if [[ ${ARRAY} == true ]]; then
    FLAGS="${FLAGS} --array-id=${SLURM_ARRAY_TASK_ID}"
fi

if [[ -n ${START_DATE} ]]; then
    FLAGS="${FLAGS} --start=${START_DATE}"
fi

if [[ -n ${END_DATE} ]]; then
    FLAGS="${FLAGS} --end=${END_DATE}"
fi

# Determine config file and output paths using the remaining arguments
CONFIG_PATH="${PROJECTROOT}/configs/merge/${1}.toml"

# Set default DATA_DIR
if [[ -z ${DATA_DIR} ]]; then
    DATA_DIR="${LEONARDO_DATA_ROOT}/data/dataset"
fi

# Set OUTPUT_PATH and ensure parent exists
if [[ -z ${OUTPUT_DIR} ]]; then
    OUTPUT_DIR="${LEONARDO_DATA_ROOT}/data/dataset"
fi
if [[ ! -d "${OUTPUT_DIR}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
fi
OUTPUT_PATH="${OUTPUT_DIR}/${1}"

export PYTHONWARNINGS='ignore::RuntimeWarning'

srun python -m graphcast.dataset merge ${FLAGS} -- "${CONFIG_PATH}" "${DATA_DIR}" "${OUTPUT_PATH}"

MERGE_EXIT_CODE=$?

if [[ ${MERGE_EXIT_CODE} -eq 0 ]]; then
    echo "----------------------------------------"
    echo "Dataset merge completed successfully."
else
    echo "----------------------------------------"
    echo "Dataset merge failed with exit code: ${MERGE_EXIT_CODE}"
    exit ${MERGE_EXIT_CODE}
fi

# Deactivate all environments (useless when run in a SLURM job)
source leonardo/environment/deactivate