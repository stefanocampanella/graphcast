#! /usr/bin/env bash
#SBATCH --job-name=REGRID_WEATHERBENCH2
#SBATCH --account=IscrB_TESMHg
#SBATCH --partition=dcgp_usr_prod
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --output=logs/%x-%j.log

VENV=$1
WEATHERBENCH_DIR=$2
SOURCE=$3
DESTINATION=$4

module load python

source "${VENV}/bin/activate"

export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=True

mkdir -p "${DESTINATION}"

for DIR in ls "${SOURCE}"; do
    python "${WEATHERBENCH_DIR}/scripts/regrid.py" \
        --input_path="${SOURCE}/${DIR}" \
        --output_path="${DESTINATION}/${DIR}" \
        --longitude_nodes=360 --latitude_nodes=181 \
        --num_threads=112 \
        --output_chunks="time=100" \
        --runner=DirectRunner || exit
done
