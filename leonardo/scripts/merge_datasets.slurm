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
        --progress                              Whether to display a progress bar (default false).
        --debug                                 Use synchronous Dask scheduler (default false).
        --help                                  Shows this help.
EOF
}

# Options default values
OVERWRITE=false
PROGRESS=false
DEBUG=false
ARRAY=true

# Parse command line options
ARGS=$(getopt --options '' --longoptions "overwrite,progress,debug,no-array,help" --name "$0" -- "${@}")
if [[ ${?} -ne 0 ]]; then
    usage
    exit 1
fi
eval "set -- ${ARGS}"
while true; do
    case "${1}" in
        (--overwrite)
            OVERWRITE=true
            shift
            ;;
        (--progress)
            PROGRESS=true
            shift
            ;;
        (--debug)
            DEBUG=true
            shift
            ;;
        (--no-array)
            ARRAY=false
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
WORKDIR="${SCRATCH}/xcast"

# Determine config file and output paths using the remaining arguments
CONFIG_PATH="${PROJECTROOT}/configs/merge/${1}.toml"
DATA_DIR="${WORKDIR}/data/dataset"
OUTPUT_PATH="${WORKDIR}/data/dataset/${1}"

source leonardo/environment/activate

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

OUTPUT_DIR=$(dirname "${OUTPUT_PATH}")
if [[ ! -d "${OUTPUT_DIR}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
fi

python -m graphcast.dataset merge ${FLAGS} -- "${CONFIG_PATH}" "${DATA_DIR}" "${OUTPUT_PATH}"

MERGE_EXIT_CODE=$?

if [[ ${MERGE_EXIT_CODE} -eq 0 ]]; then
    echo "----------------------------------------"
    echo "Dataset merge completed successfully."
else
    echo "----------------------------------------"
    echo "Dataset merge failed with exit code: ${MERGE_EXIT_CODE}"
    echo "Job failed. Re-submitting..."
    scontrol requeue $SLURM_JOB_ID
    exit ${MERGE_EXIT_CODE}
fi

# Deactivate all environments (useless when run in a SLURM job)
source leonardo/environment/deactivate