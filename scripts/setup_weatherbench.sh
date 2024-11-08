#! /usr/bin/env bash

usage() {
    more <<EOF
NAME
    This script creates the environment and downloads the WeatherBench2 dataset.

SYNOPSIS
    usage: $0 --help
    usage: $0 [options]

DESCRIPTION
    Setup options
        --skip-env                              Do not create an environment.
        --env-path                              Environment path (default 'weatherbench_env')
        --skip-data                             Do not download params and datasets.
        --data-dir                              Path where to download data (default 'data/weatherbench').
        --config                                Path to configuration file (default 'configs/download/weatherbench_small.toml').
        --help                                  Shows this help.
EOF
}

# DEFAULTS
ROOT=$(git rev-parse --show-toplevel)
MAKE_ENV=true
DOWNLOAD=true
REGRID=true
CONFIG_FILE="${ROOT}/configs/download/weatherbench_small.toml"
DATA_DIR="${ROOT}/data/weatherbench2"

LONGOPTS='help,skip-env,skip-data,skip-regrid,data-dir:'
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
    (--skip-regrid)
        REGRID=false
        shift
        ;;
    (--data-dir)
        DATA_DIR=${2}
        shift
        ;;
    (--config)
        DATA_DIR=${2}
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

if [[ $MAKE_ENV == true ]]; then

    cd "${ROOT}" || exit

    if [[ -d weatherbench2 ]]; then
        echo "Target directory for weatherbench2 git repo already exists!"
        exit 1
    else
        module load git python

        WEATHERBENCH2_GIT_URL=https://github.com/google-research/weatherbench2.git
        git clone ${WEATHERBENCH2_GIT_URL}

        python -m venv --system-site-packages --upgrade-deps weatherbench2/venv
        source weatherbench2/venv/bin/activate
        python -m pip install google-cloud-storage gcsfs absl-py "./weatherbench2"
    fi
fi

if [[ $DOWNLOAD == true ]]; then
    # Download ERA5 dataset specified in $CONFIG_FILE from GCS (WeatherBench2)
    if [[ -d "${DATA_DIR}/era5" ]]; then
        echo "Target directory for dataset already exist! Skipping."
        exit 1
    else
        sbatch "${ROOT}/scripts/download_weatherbench.slurm" "${ROOT}/weatherbench2/venv" "${CONFIG_FILE}" "${DATA_DIR}/era5"
    fi
fi

if [[ $REGRID == true ]]; then
    if [[ -d "${DATA_DIR}/regridded" ]]; then
        echo "Target directory for regridded dataset already exists! Skipping."
        exit 1
    else
        sbatch "${ROOT}/scripts/regrid_weatherbench.slurm" "${ROOT}/weatherbench2/env" "${ROOT}/weatherbench2" "${CONFIG_FILE}" "${DATA_DIR}/era5" "${DATA_DIR}/regridded"
    fi
fi
