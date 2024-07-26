#! /usr/bin/env bash
#SBATCH --job-name=DOWNLOAD_ERA5_DATA
#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=60
#SBATCH --output=/leonardo_scratch/fast/OGS23_PRACE_IT_0/scampane/graphcast/logs/%x-%j.log

ROOT=$(git rev-parse --show-toplevel)

if [[ -z $1 ]]; then
    VENV="${ROOT}/data/weatherbench2/venv"
else
    VENV=$1
fi

if [[ -z $2 ]]; then
    CONFIG_FILE="${ROOT}/data/weatherbench2/configs/small.toml"
else
    CONFIG_FILE=$2
fi

if [[ -z $3 ]]; then
    DATA_DIR="${ROOT}/data/weatherbench2/data"
else
    DATA_DIR=$3
fi

cd "${ROOT}" || exit

module load python

source "${VENV}/bin/activate"

python leonardo/scripts/download_era5.py "${CONFIG_FILE}" "${DATA_DIR}/era5"
