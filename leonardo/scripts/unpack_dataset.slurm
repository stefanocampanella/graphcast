#! /usr/bin/env bash
#SBATCH --account=OGS23_PRACE_IT_1
#SBATCH --partition=dcgp_usr_prod
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --gres=tmpfs:3T
#SBATCH --requeue

usage() {
    more <<EOF
unpack_dataset.slurm
    This script unpack a dataset stored as a collection of zipped Zarrs.

SYNOPSIS
    usage: $0 --help
    usage: $0 [options] dataset_name

DESCRIPTION
    Positional arguments
        dataset_name                            Name of the input dataset

    Options
        --overwrite                             Whether to overwrite destination (default false).
        --start                                 Overwrite config start date.
        --end                                   Overwrite config end date.
        --output-dir                            Where to place the ouputs (default to $LEONARDO_DATA_ROOT/data/dataset).
        --data-dir                              Where to look for data (defaut to $LEONARDO_DATA_ROOT/data/dataset).
        --debug                                 Use synchronous Dask scheduler (default false).
        --log-level LEVEL                       Logging level (debug, info, warning, error, critical). Default: info.
        --help                                  Shows this help.
EOF
}

# Default options
OVERWRITE=false
TIME_DIM="time"
LOG_LEVEL="info"

# Parse command line options
ARGS=$(getopt --options '' --longoptions "overwrite,start:,end:,output-dir:,data-dir:,time-dim:,log-level:,debug,help" --name "$0" -- "${@}")
if [[ ${?} -ne 0 ]]; then
    usage
    exit 1
fi

eval "set -- ${ARGS}"
while true; do
    case "$1" in
        (--overwrite)
            OVERWRITE=true;
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
        (--time-dim)
            TIME_DIM="$2"; shift 2
            ;;
        (--log-level)
            LOG_LEVEL="$2"; shift 2
            ;;
        (--debug)
            DEBUG=true
            shift
            ;;
        (--help)
            usage;
            exit 0
            ;;
        (--)
            shift;
            break
            ;;
        (*)
            echo "Error: unrecognized option $1" >&2
            usage
            exit 1
            ;;
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

# Activate environment and set default path envs
source leonardo/environment/activate
source leonardo/.env

# Build flags for CLI
FLAGS="--time-dim=${TIME_DIM} --log-level=${LOG_LEVEL} --chunks=time:32,level:-1,lat:-1,lon:-1"

if [[ ${OVERWRITE} == true ]]; then
    FLAGS="${FLAGS} --overwrite"
else
    FLAGS="${FLAGS} --no-overwrite"
fi

if [[ ${DEBUG} == true ]]; then
    FLAGS="${FLAGS} --debug"
else
    FLAGS="${FLAGS} --no-debug"
fi

if [[ -n ${START_DATE} ]]; then
    FLAGS="${FLAGS} --start=${START_DATE}"
fi

if [[ -n ${END_DATE} ]]; then
    FLAGS="${FLAGS} --end=${END_DATE}"
fi

# Set default INPUT_PATH
if [[ -z ${DATA_DIR} ]]; then
    DATA_DIR="${LEONARDO_DATA_ROOT}/data/dataset"
fi
INPUT_PATH="${DATA_DIR}/${1}"

# Set OUTPUT_PATH and ensure parent exists
if [[ -z ${OUTPUT_DIR} ]]; then
    OUTPUT_DIR="${LEONARDO_DATA_ROOT}/data/dataset"
fi
if [[ ! -d "${OUTPUT_DIR}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
fi
OUTPUT_PATH="${OUTPUT_DIR}/${1}_arco"


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
export PYTHONWARNINGS='ignore::RuntimeWarning'


srun --output="${LEONARDO_FAST_ROOT}"/logs/%x-%j_%N_%t.out --error="${LEONARDO_FAST_ROOT}"/logs/%x-%j_%N_%t.err \
  python -m graphcast.dataset unpack ${FLAGS} -- "${INPUT_PATH}" "${OUTPUT_PATH}"

UNPACK_EXIT_CODE=$?

if [[ ${UNPACK_EXIT_CODE} -eq 0 ]]; then
    echo "----------------------------------------"
    echo "Dataset unpacked successfully."
else
    echo "----------------------------------------"
    echo "Dataset unpack failed with exit code: ${UNPACK_EXIT_CODE}"
    exit ${UNPACK_EXIT_CODE}
fi

# Deactivate all environments (useless when run in a SLURM job)
source leonardo/environment/deactivate
