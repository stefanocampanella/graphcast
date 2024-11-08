#! /usr/bin/env bash
#SBATCH --job-name=DOWNLOAD_GRAPHCAST_DEMO
#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20
#SBATCH --gres=tmpfs:10G
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.log

ROOT=$(git rev-parse --show-toplevel)
VENV="${ROOT}/venv"
DATA_DIR="${ROOT}/data"

module load profile/deeplrn cineca-ai/4.1.1

source "${VENV}/bin/activate"

python bin/file_downloader.py gs://dm_graphcast/dataset "${DATA_DIR}/demo/dataset" --overwrite
python bin/file_downloader.py gs://dm_graphcast/params "${DATA_DIR}/demo/params" --overwrite
python bin/file_downloader.py gs://dm_graphcast/stats "${DATA_DIR}/demo/stats" --overwrite
