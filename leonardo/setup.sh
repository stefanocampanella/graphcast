#! /usr/bin/env bash

usage() {
    more <<EOF
NAME
    This script prepares the environment and files necessary to run the GraphCast demo on Leonardo.

SYNOPSIS
    usage: $0 --help
    usage: $0 [options]

DESCRIPTION
    Setup options
        --skip-env                              Do not create an environment.
        --skip-data                             Do not download params and datasets.
        --data-dir                              Path where to download data (default 'data/demo').
        --config                                Path to download config file (default 'configs/download/demo.toml').
        --help                                  Shows this help.
EOF
}

# DEFAULTS
ROOT=$(git rev-parse --show-toplevel)
MAKE_ENV=true
DOWNLOAD=true
DATA_DIR="${ROOT}/data/demo"
CONFIG_FILE="${ROOT}/configs/download/demo.toml"

LONGOPTS='help,skip-env,skip-data,data-dir:,config:'
ARGS=$(getopt --options '' --longoptions ${LONGOPTS} -- "${@}")
if [[ $? -ne 0 ]]; then
    usage
    exit 1
fi

eval "set -- ${ARGS}"
while true; do
    case "${1}" in
    (--skip-env)
        MAKE_ENV=false
        shift
        ;;
    (--skip-data)
        DOWNLOAD=false
        shift
        ;;
    (--data-dir)
        DATA_DIR=${2}
        shift
        ;;
    (--config)
        CONFIG_FILE=${2}
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
        exit 1
        ;;
    esac
done

# Change working directory to project root
cd "${ROOT}" || exit

if [[ $MAKE_ENV == true ]]; then
    # Load prerequisite modules, mainly Python and CUDA
    # TODO: it should probably be downgraded to cineca-ai/3.0.1 because of this:
    # https://github.com/google/jax/issues/15384
    # But then one should find a functioning version of jax
    # and jaxlib to be used with graphcast...
    # Seems more reasonable to not have a working jax.profiler.trace
    module load profile/deeplrn cineca-ai/4.1.1

    # Create Python venv
    python -m venv --system-site-packages --upgrade-deps venv || exit

    # Activate Python venv, afterwards download packages (needs internet connection)
    source "${ROOT}/venv/bin/activate"

    JAX_RELEASE_URL=https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    python -m pip download --dest=pkg_cache --find-links=${JAX_RELEASE_URL} "${ROOT}[download,interactive,profile,train]" || exit

    # Install packages on a GPU node
    ACCOUNT=OGS23_PRACE_IT_0
    PARTITION=boost_usr_prod
    TIME=10
    COMMAND="python -m pip install --no-build-isolation --no-index --find-links pkg_cache -e ${ROOT}[download,interactive,profile,train]"
    srun --account ${ACCOUNT} --partition ${PARTITION} --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=${TIME} ${COMMAND} || exit

    deactivate
fi

if [[ $DOWNLOAD == true ]]; then
    mkdir -p "${DATA_DIR}"
    mkdir -p "${ROOT}/logs"

    # Download graphcast demo datasets, weights and stats from GraphCast publicly available bucket on Google Cloud Storage
    sbatch "${ROOT}/scripts/download_demo.slurm"
fi
