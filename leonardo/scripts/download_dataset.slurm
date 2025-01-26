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
download_dataset.slurm
    This script downloads a dataset from Copernicus Marine Service / WeatherBench GCS bucket.
    If the dataset is static (e.g. bathymetry), it will download the entire dataset at once.
    If the dataset is a timeseries, it will download a range of dates computed from SLURM_ARRAY_TASK_ID.

SYNOPSIS
    usage: $0 --help
    usage: $0 [options] config

DESCRIPTION
    Options
        --no-array                              Whether to download dataset in a single SLURM job (default false).
        --overwrite                             Whether to overwrite destination (default false).
        --help                                  Shows this help.
EOF
}

# Options default values
OVERWRITE=false
ARRAY=true
# Parse command line options
ARGS=$(getopt --options '' --longoptions "overwrite,no-array,help" --name "$0" -- "${@}")
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
ARCHIVE="/leonardo_store/DRES_EchoArch/${USER}"
cd "${PROJECTROOT}" || exit 1

# Determine config file and output paths using the remaining argument
CONFIG_PATH="${PROJECTROOT}/configs/download/${1}.toml"
OUTPUT_PATH="${ARCHIVE}/data/dataset/${1}"

source leonardo/environment/activate

if [[ ${OVERWRITE} == true ]]; then
    OVERWRITE_FLAG="--overwrite"
else
    OVERWRITE_FLAG="--no-overwrite"
fi

if [[ ${ARRAY} == true ]]; then
    ARRAY_FLAG="--array-id=${SLURM_ARRAY_TASK_ID}"
else
    ARRAY_FLAG=""
fi

python -m graphcast.dataset download ${OVERWRITE_FLAG} ${ARRAY_FLAG} -- "${CONFIG_PATH}" "${OUTPUT_PATH}"

# Deactivate all environments (useless when run in a SLURM job)
source leonardo/environment/deactivate
