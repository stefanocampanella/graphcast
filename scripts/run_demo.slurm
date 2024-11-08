#! /usr/bin/env bash
#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --partition=boost_usr_prod
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --output=/leonardo_scratch/fast/OGS23_PRACE_IT_0/scampane/graphcast/logs/%x-%j.log

ROOT=$(git rev-parse --show-toplevel)
cd "${ROOT}" || exit

module load profile/deeplrn cineca-ai
source venv/bin/activate
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true \
                  --xla_gpu_triton_gemm_any=True \
                  --xla_gpu_enable_async_collectives=true \
                  --xla_gpu_enable_latency_hiding_scheduler=true \
                  --xla_gpu_enable_highest_priority_async_stream=true"

python leonardo/scripts/demo_experiment.py $@
