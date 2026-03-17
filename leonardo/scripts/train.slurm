#! /usr/bin/env bash
#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue

PROJECTROOT=$(git rev-parse --show-toplevel)
cd "${PROJECTROOT}" || exit 1

# Activate environment
source leonardo/environment/activate
# Define code/settings and data paths
source leonardo/.env

# Try to suppress warnings from TensorFlow/Keras and XLA
export TF_CPP_MIN_LOG_LEVEL=3
# Suppress Python warnings polluting log files (mostly related to protobuf gencode version)
export PYTHONWARNINGS='ignore:::google.protobuf.runtime_version,ignore::FutureWarning'
# Disable cpu backend, might suppress warnings related to TPUs
export JAX_PLATFORMS=cuda
# FIXME: Occasionally, some of the ranks will fail with the following error message:
#  >> INTERNAL: NCCL operation ncclGroupEnd() failed: unhandled system error (run with NCCL_DEBUG=INFO for details).
#  >> Last NCCL warning(error) log entry (may be unrelated) ''
#  Debug messages can be logged with the following, however they appear not informative. Instead, disabling infiniband,
#  as suggested here: https://github.com/NVIDIA/nccl/issues/694, solves the issue. Impact on performance is still to be
#  investigated.
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# The following might solve memory fragmentation issues
export TF_GPU_ALLOCATOR=cuda_malloc_async
# Performance optimization flags, as for: https://docs.jax.dev/en/latest/gpu_performance_tips.html
export XLA_FLAGS='--xla_gpu_triton_gemm_any=true --xla_gpu_enable_latency_hiding_scheduler=true'
#export JAX_ENABLE_PGLE=true
#export JAX_PGLE_PROFILING_RUNS=3
#export JAX_PGLE_AGGREGATION_PERCENTILE=85
# FIXME: Enabling compilation cache appears to cause errors on all ranks but 0, with
#  the following error message:
#  >> jaxlib._jax.XlaRuntimeError: INVALID_ARGUMENT:
#  >> Expected args to execute_sharded_on_local_devices to have 0 shards, got: ...
#  Probably due to an XLA bug, see: https://github.com/openxla/xla/pull/27683.
#  Upstream JAX has a fix for it, check that newer versions solve the issue.
#export JAX_COMPILATION_CACHE_DIR="${LEONARDO_FAST_ROOT}/.jax_cache"

# Flags to consider for execution time optimization
#export JAX_OPTIMIZATION_LEVEL=O3
# Flags to consider for memory optimizations
#export JAX_MEMORY_FITTING_EFFORT=1.0
#export JAX_MEMORY_FITTING_LEVEL=O3
# Could it reduce compilation time on large multi-host setups?
#export JAX_SHARE_BINARY_BETWEEN_HOSTS=True

# When debugging multi-host setups, might be useful
#export JAX_DISTRIBUTED_DEBUG=True
#export JAX_TRANSFER_GUARD_*=log
# Flags to consider for OOM debugging
#export JAX_LOG_CHECKPOINT_RESIDUALS=True
# Other debugging flags
#export JAX_TRACEBACK_FILTERING=off
#export JAX_PPRINT_USE_COLOR=True

CONFIG_PATH="${LEONARDO_CODE_ROOT}/configs/training/${1}"
OUTPUT_CKPT_PATH="${LEONARDO_DATA_ROOT}/data/models/${2}"
TRAIN_CKPT_PATH="${LEONARDO_FAST_ROOT}/data/training_checkpoints/${3}"

shift 3

srun --output="${LEONARDO_FAST_ROOT}"/logs/%x-%j_%N_%t.out --error="${LEONARDO_FAST_ROOT}"/logs/%x-%j_%N_%t.err \
   python -m graphcast.training launch \
   --data-path=$LEONARDO_DATA_ROOT "${@}" -- \
  "${CONFIG_PATH}" "${OUTPUT_CKPT_PATH}" "${TRAIN_CKPT_PATH}"

source leonardo/environment/deactivate
