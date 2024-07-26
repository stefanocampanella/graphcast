#! /usr/bin/env bash
#SBATCH --job-name=REGRID_ERA5
#SBATCH --account=IscrB_TESMHg
#SBATCH --partition=dcgp_usr_prod
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --output=/leonardo_scratch/fast/OGS23_PRACE_IT_0/scampane/graphcast/logs/%x-%j.log

ROOT=$(git rev-parse --show-toplevel)

if [[ -z $1 ]]; then
    VENV="${ROOT}/data/weatherbench2/venv"
else
    VENV=$1
fi

if [[ -z $2 ]]; then
    WEATHERBENCH_DIR="${ROOT}/data/weatherbench2"
else
    WEATHERBENCH_DIR=$2
fi

if [[ -z $3 ]]; then
    SOURCE="${ROOT}/data/weatherbench2/data/era5/training"
else
    SOURCE=$3
fi

if [[ -z $4 ]]; then
    DESTINATION="${ROOT}/data/weatherbench2/data/era5_regridded/training"
else
    DESTINATION=$4
fi


module load python

source "${VENV}/bin/activate"

export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=True

mkdir -p "${DESTINATION}"

python "${WEATHERBENCH_DIR}/scripts/regrid.py" \
    --input_path="${SOURCE}" \
    --output_path="${DESTINATION}" \
    --longitude_nodes=360 --latitude_nodes=181 \
    --num_threads=112 \
    --output_chunks="time=100" \
    --runner=DirectRunner || exit
