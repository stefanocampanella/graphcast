#! /usr/bin/env bash
#SBATCH --account=OGS23_PRACE_IT_1
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=28
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --time=120
#SBATCH --gres=tmpfs:3T
#SBATCH --requeue

usage() {
    more <<EOF
average_dataset.slurm
    This script computes daily stats from a Zarr dataset.

SYNOPSIS
    usage: $0 --help
    usage: $0 [options] dataset_name

DESCRIPTION
    Positional arguments
        dataset_name                            Name of the input dataset

    Options
        --overwrite                             Whether to overwrite destination (default false).
        --skipna                                Skip NaNs when averaging (default false).
        --time-dim DIM                          Name of the time dimension (default: time).
        --log-level LEVEL                       Logging level (debug, info, warning, error, critical). Default: info.
        --help                                  Shows this help.
EOF
}

# Default options
OVERWRITE=false
SKIPNA=false
TIME_DIM="time"
LOG_LEVEL="info"

# Parse command line options
ARGS=$(getopt --options '' --longoptions "overwrite,skipna,time-dim:,log-level:,help" --name "$0" -- "${@}")
if [[ ${?} -ne 0 ]]; then
    usage
    exit 1
fi

eval "set -- ${ARGS}"
while true; do
    case "$1" in
        (--overwrite)
            OVERWRITE=true; shift ;;
        (--skipna)
            SKIPNA=true; shift ;;
        (--time-dim)
            TIME_DIM="$2"; shift 2 ;;
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

PROJECTROOT=$(git rev-parse --show-toplevel)
cd "${PROJECTROOT}" || exit 1

# Activate environment
source leonardo/environment/activate

# Build flags for CLI
FLAGS="--time-dim=${TIME_DIM} --log-level=${LOG_LEVEL} --chunks=time:256,level:1,lat:-1,lon:-1"

if [[ ${OVERWRITE} == true ]]; then
    FLAGS="${FLAGS} --overwrite"
else
    FLAGS="${FLAGS} --no-overwrite"
fi

if [[ ${SKIPNA} == true ]]; then
    FLAGS="${FLAGS} --skipna"
else
    FLAGS="${FLAGS} --no-skipna"
fi

# Ensure output directory exists
OUTPUT_DIR=$(dirname "${OUTPUT_PATH}")
if [[ ! -d "${OUTPUT_DIR}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
fi

# see:
# https://bitbucket.org/mpi4py/mpi4py/issues/80/mpi_thread_multiple-vs-mpi_thread_single
# https://github.com/open-mpi/ompi/issues/6240
# https://github.com/open-mpi/ompi/issues/9763
# https://developer.skao.int/projects/ska-sdp-exec-iotest/en/latest/content/mpi_settings.html#slurm-specific-settings
# https://docs.nvidia.com/networking/display/hpcxv211/hcoll

export OMPI_MCA_btl_tcp_if_exclude=brmgt,enp1s0,lo,virbr0
export OMPI_MCA_coll=^hcoll
export DASK_MPI__INTERFACE=ib0
export DASK_MPI__PROTOCOL="tcp://"
export DASK_MPI__NTHREADS=${SLURM_CPUS_PER_TASK}
export DASK_MPI__LOCAL_DIRECTORY=${TMPDIR}
export DASK_MPI__LOG_DIR="${PROJECTROOT}/logs/dask"

mkdir -p DASK_MPI_LOG_DIR

DATADIR="${SCRATCH}/xcast/data"
ALL_STATS=(mean std diff_std)
for STATS in "${ALL_STATS[@]}"; do
    INPUT_PATH="${DATADIR}/dataset/$1"
    OUTPUT_PATH="${DATADIR}/stats/$1_${STATS}"
    srun -n ${SLURM_NTASKS} python -m graphcast.stats compute ${FLAGS} -- ${STATS} "${INPUT_PATH}" "${OUTPUT_PATH}"
done

# Deactivate all environments (useless when run in a SLURM job)
source leonardo/environment/deactivate
