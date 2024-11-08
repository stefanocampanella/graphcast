#! /usr/bin/env bash
#SBATCH --job-name=DOWNLOAD_ERA5_SURFACE_SMALL
#SBATCH --array=1993-2021
#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=240
#SBATCH --mem=30800M
#SBATCH --gres=tmpfs:100G
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.log


ROOT=$(git rev-parse --show-toplevel)
VENV="${ROOT}/venv"
DATA_DIR="${ROOT}/data"
YEAR=$SLURM_ARRAY_TASK_ID

cd "${ROOT}" || exit

module load profile/deeplrn cineca-ai

source "${VENV}/bin/activate"

python bin/dataset_downloader.py "${ROOT}/configs/download/era5_gcs_small.toml" \
                                 "${DATA_DIR}/ERA5/${YEAR}" \
                                 --year="${YEAR}" \
                                 --provider=gcs \
                                 --no-progress \
                                 --overwrite
