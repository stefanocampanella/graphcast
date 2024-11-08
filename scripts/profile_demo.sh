#! /usr/bin/env bash
#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --partition=boost_usr_prod
#SBATCH --time=30
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

ROOT=$(git rev-parse --show-toplevel)
cd "${ROOT}" || exit

module load profile/deeplrn cineca-ai
source venv/bin/activate
export JAX_PLATFORMS=cuda,cpu
export JAX_TRACEBACK_FILTERING=off
export JAX_TRACEBACK_IN_LOCATIONS_LIMIT=-1
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_VMODULE=bfc_allocator=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true \
                  --xla_gpu_triton_gemm_any=True \
                  --xla_gpu_enable_async_collectives=true \
                  --xla_gpu_enable_latency_hiding_scheduler=true \
                  --xla_gpu_enable_highest_priority_async_stream=true"

nsys profile -o "$PWD/$(date -Iminute)" --cuda-graph-trace=node --cuda-memory-usage=true python leonardo/scripts/graphcast_demo.py $@
