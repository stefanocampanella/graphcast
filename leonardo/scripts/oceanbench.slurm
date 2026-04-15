#! /usr/bin/env bash
#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=240
#SBATCH --mem=30800M
#SBATCH --gres=tmpfs:100G
#SBATCH --requeue

usage() {
    more <<EOF
oceanbench.slurm

SYNOPSIS
    usage: $0 --help
    usage: $0 [options] config

DESCRIPTION
    Options
        --input-dir                             Where look for the challenger dataset (default to $LEONARDO_DATA_ROOT/data/oceanbench).
        --help                                  Shows this help.
EOF
}

# Parse command line options
ARGS=$(getopt --options '' --longoptions "input-dir:,help" --name "$0" -- "${@}")
if [[ ${?} -ne 0 ]]; then
    usage
    exit 1
fi
eval "set -- ${ARGS}"
while true; do
    case "${1}" in
        (--input-dir)
           INPUT_DIR="${2}"
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

# Set CHALLENGER_DATASET_PATH and ensure parent exists
if [[ -z ${INPUT_DIR} ]]; then
    INPUT_DIR="${LEONARDO_DATA_ROOT}/data/oceanbench"
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
    mkdir -p "${INPUT_DIR}"
fi
export CHALLENGER_DATASET_PATH="${INPUT_DIR}/${1}"
export OMP_NUM_THREADS=4

oceanbench evaluate leonardo/scripts/graphcast_challenger.py

OCEANBENCH_EXIT_CODE=$?

if [[ ${OCEANBENCH_EXIT_CODE} -eq 0 ]]; then
    echo "----------------------------------------"
    echo "Download completed successfully."
else
    echo "----------------------------------------"
    echo "Download failed with exit code: ${OCEANBENCH_EXIT_CODE}"
    exit ${OCEANBENCH_EXIT_CODE}
fi

# Deactivate all environments (useless when run in a SLURM job)
source leonardo/environment/deactivate
