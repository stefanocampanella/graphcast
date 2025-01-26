#! /usr/bin/env bash
#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --partition=dcgp_usr_prod
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=240
#SBATCH --mem=30800M
#SBATCH --gres=tmpfs:100G
#SBATCH --requeue

# Load prerequisite modules, mainly Python and CUDA
SCRIPT_PATH=$(dirname $0)
source "${SCRIPT_PATH}/modules.sh" || exit

# Change working directory to project root
ROOT=$(git -C "${SCRIPT_PATH}" rev-parse --show-toplevel)
cd "${ROOT}" || exit

# Run Jupyterlab on ephemeral port
source venv/bin/activate
jupyter lab --port=32769
