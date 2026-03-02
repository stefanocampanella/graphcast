#! /usr/bin/env bash
#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --requeue

PROJECTROOT=$(git rev-parse --show-toplevel)
cd "${PROJECTROOT}" || exit 1

# Activate environment
source leonardo/environment/activate
# Define code/settings and data paths
source leonardo/.env
# The following might solve memory fragmentation issues
export TF_GPU_ALLOCATOR=cuda_malloc_async
# see: https://docs.jax.dev/en/latest/gpu_performance_tips.html
export XLA_FLAGS='--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true'

CONFIG_PATH="$LEONARDO_CODE_ROOT/configs/training/${1}"
INPUT_CKPT_PATH="$LEONARDO_DATA_ROOT/data/models/${2}"
OUTPUT_CKPT_PATH="$LEONARDO_DATA_ROOT/data/models/${3}"
TRAIN_CKPT_PATH="$LEONARDO_FAST_ROOT/data/training_checkpoints/${4}"

shift 4

srun python -m graphcast.training launch --data-path=$LEONARDO_DATA_ROOT "${@}" -- \
  "${CONFIG_PATH}" "${INPUT_CKPT_PATH}" "${OUTPUT_CKPT_PATH}" "${TRAIN_CKPT_PATH}"

source leonardo/environment/deactivate
