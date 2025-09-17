#! /usr/bin/env bash
#SBATCH --account=OGS_test2528_0
#SBATCH --partition=dcgp_usr_prod
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --time=500
#SBATCH --gres=tmpfs:10G
#SBATCH --requeue

usage() {
    more <<EOF
average_dataset.slurm
    This script computes the time average and daily climatology of a Zarr dataset.

SYNOPSIS
    usage: $0 --help
    usage: $0 [options] dataset_name

DESCRIPTION
    Positional arguments
        dataset_name                            Name of the input dataset

    Options
        --overwrite                             Whether to overwrite destination (default false).
        --progress                              Whether to display a progress bar (default false).
        --skipna                                Skip NaNs when averaging (default false).
        --no-climatology                        Do not compute daily climatology.
        --time-dim DIM                          Name of the time dimension (default: time).
        --climatology-dim DIM                   Name of the climatology time dimension in the outputs (default: dayofyear).
        --workers N                             Max Dask workers to use (default: SLURM_CPUS_PER_TASK or 4).
        --log-level LEVEL                       Logging level (debug, info, warning, error, critical). Default: info.
        --help                                  Shows this help.
EOF
}

# Default options
OVERWRITE=false
PROGRESS=false
SKIPNA=false
CLIMATOLOGY=true
TIME_DIM="time"
CLIMATOLOGY_DIM="dayofyear"
WORKERS=${SLURM_CPUS_PER_TASK:-4}
LOG_LEVEL="info"

# Parse command line options
ARGS=$(getopt --options '' --longoptions "overwrite,progress,skipna,no-climatology,time-dim:,climatology-dim:,workers:,log-level:,help" --name "$0" -- "${@}")
if [[ ${?} -ne 0 ]]; then
    usage
    exit 1
fi

eval "set -- ${ARGS}"
while true; do
    case "$1" in
        (--overwrite)
            OVERWRITE=true; shift ;;
        (--progress)
            PROGRESS=true; shift ;;
        (--skipna)
            SKIPNA=true; shift ;;
        (--no-climatology)
            CLIMATOLOGY=false; shift ;;
        (--time-dim)
            TIME_DIM="$2"; shift 2 ;;
        (--climatology-dim)
            CLIMATOLOGY_DIM="$2"; shift 2 ;;
        (--workers)
            WORKERS="$2"; shift 2 ;;
        (--log-level)
            LOG_LEVEL="$2"; shift 2 ;;
        (--help)
            usage; exit 0 ;;
        (--)
            shift; break ;;
        (*)
            echo "Error: unrecognized option $1" >&2
            usage
            exit 1 ;;
    esac
done

# Remaining args: input_path and output_path
if [[ $# -lt 1 ]]; then
    echo "Error: missing required argument dataset name" >&2
    usage
    exit 1
fi

DATADIR="/leonardo_scratch/large/userexternal/${USER}/xcast/data"
INPUT_PATH="${DATADIR}/dataset/$1"
OUTPUT_PATH="${DATADIR}/stats"
BASENAME_PREFIX="$1"

PROJECTROOT=$(git rev-parse --show-toplevel)
cd "${PROJECTROOT}" || exit 1

# Activate environment
source leonardo/environment/activate

# Cap Dask concurrency to limit memory usage
export DASK_NUM_WORKERS="${WORKERS}"

# Build flags for CLI
FLAGS="--time-dim=${TIME_DIM} --climatology-dim=${CLIMATOLOGY_DIM} --log-level=${LOG_LEVEL}"

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

if [[ ${SKIPNA} == true ]]; then
    FLAGS="${FLAGS} --skipna"
else
    FLAGS="${FLAGS} --no-skipna"
fi

if [[ ${CLIMATOLOGY} == true ]]; then
    FLAGS="${FLAGS} --climatology"
else
    FLAGS="${FLAGS} --no-climatology"
fi

# Ensure output directory exists
OUTPUT_DIR=$(dirname "${OUTPUT_PATH}")
if [[ ! -d "${OUTPUT_DIR}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
fi

python -m graphcast.stats compute ${FLAGS} -- "${INPUT_PATH}" "${OUTPUT_PATH}" "${BASENAME_PREFIX}"

# Deactivate all environments (useless when run in a SLURM job)
source leonardo/environment/deactivate
