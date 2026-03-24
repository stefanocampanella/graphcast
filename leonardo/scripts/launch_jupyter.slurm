#! /usr/bin/env bash
#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

PROJECTROOT=$(git rev-parse --show-toplevel)
cd "${PROJECTROOT}" || exit 1

# Activate environment
source leonardo/environment/activate
# Define code/settings and data paths
source leonardo/.env

export JAX_PLATFORMS=cuda
export TF_GPU_ALLOCATOR=cuda_malloc_async
export XLA_FLAGS='--xla_gpu_triton_gemm_any=true --xla_gpu_enable_latency_hiding_scheduler=true'
export JAX_TRACEBACK_FILTERING=off

echo "Hostname: $(hostname -s)"
srun jupyter lab --ip="$(hostname -i)" "${@}"

source leonardo/environment/deactivate