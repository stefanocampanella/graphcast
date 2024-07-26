#! /usr/bin/env bash

usage() {
    more <<EOF
NAME
    This script prepare the environment and files to run GraphCast on Leonardo.

SYNOPSIS
    usage: $0 --help
    usage: $0 [options]

DESCRIPTION
    Setup options
        --skip-env                              Do not create an environment.
        --skip-data                             Do not download params and datasets.
        --data-dir                              Path where to download data (default data).
        --config                                Path to configuration file (default leonardo/configs/small.toml).
        --help                                  Shows this help.
EOF
}

# DEFAULTS
ROOT=$(git rev-parse --show-toplevel)
CLEAR=false
MAKE_ENV=true
DOWNLOAD_DATA=true
DATA_DIR=data
CONFIG_FILE="${ROOT}/leonardo/configs/small.toml"

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
        DOWNLOAD_DATA=false
        shift
        ;;
    (--data-dir)
        DATA_DIR=${2}
        shift
        ;;
    (--config)
        CONFIG_FILE=$(realpath "${2}")
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
    python -m pip download --dest=pkg_cache --find-links=${JAX_RELEASE_URL} "${ROOT}[profile,interactive]" || exit

    # Install packages on a GPU node
    ACCOUNT=OGS23_PRACE_IT_0
    PARTITION=boost_usr_prod
    TIME=10
    COMMAND="python -m pip install --no-build-isolation --no-index --find-links pkg_cache \
             -e ${ROOT}[profile,interactive]"
    srun --account ${ACCOUNT} --partition ${PARTITION} --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=${TIME} \
         ${COMMAND} || exit

    deactivate
fi

if [[ $DOWNLOAD_DATA == true ]]; then
    mkdir -p "${DATA_DIR}"
    cd "${DATA_DIR}" || exit

    if [[ -d weatherbench2 ]]; then
        echo "Target directory for weatherbench2 git repo already exists! Skipping."
    else
        module unload cineca-ai profile/deeplrn
        module load python

        WEATHERBENCH2_GIT_URL=https://github.com/google-research/weatherbench2.git
        git clone ${WEATHERBENCH2_GIT_URL}

        python -m venv --system-site-packages --upgrade-deps weatherbench2/venv
        source weatherbench2/venv/bin/activate
        python -m pip install google-cloud-storage gcsfs absl-py "./weatherbench2"
    fi

    # Download graphcast demo datasets, weights and stats from
    # graphcast publicly available bucket on Google Cloud
    if [[ -d dataset && -d params && -d stats ]]; then
        echo "Target directories for GraphCast demo dataset already exists! Skipping."
    else
        sbatch "${ROOT}/leonardo/scripts/download_demo.slurm" "${DATA_DIR}/weatherbench2/venv" "${DATA_DIR}"
    fi

    # Download ERA5 dataset specified in $CONFIG_FILE from
    # GCS (WeatherBench2) and eventually regrid the dataset
    if [[ -d era5 ]]; then
        echo "Target directory for ERA5 dataset already exist! Skipping."
    else
        sbatch "${ROOT}/leonardo/scripts/download_era5.slurm" "${DATA_DIR}/weatherbench2/venv" "${CONFIG_FILE}" "${DATA_DIR}"
    fi
fi
