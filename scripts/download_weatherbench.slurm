#! /usr/bin/env bash

#SBATCH --job-name=DOWNLOAD_WEATHERBENCH2
#SBATCH --array=0,1
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

if [[ $SLURM_ARRAY_TASK_ID == 0 ]]; then
    DATASET="training"
    START=2021-12-01
    STOP=2022-02-28
elif [[ $SLURM_ARRAY_TASK_ID == 1 ]]; then
    DATASET="validation"
    START=2023-01-01
    STOP=2023-01-31
fi

cd "${ROOT}" || exit

module load profile/deeplrn cineca-ai

source "${VENV}/bin/activate"

python bin/dataset_downloader.py "${ROOT}/configs/download/weatherbench_small.toml" \
                                 "${DATA_DIR}/WeatherBench2/${DATASET}" \
                                 --start=${START} \
                                 --stop=${STOP} \
                                 --provider=gcs \
                                 --no-progress \
                                 --overwrite