#! /usr/bin/env bash
#SBATCH --account=OGS_test2528_0
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --gres=tmpfs:10G
#SBATCH --exclusive
#SBATCH --time=3
#SBATCH --requeue

PROJECTROOT=$(git rev-parse --show-toplevel)
cd "${PROJECTROOT}" || exit 1

# Activate environment
source leonardo/environment/activate

# see:
# https://bitbucket.org/mpi4py/mpi4py/issues/80/mpi_thread_multiple-vs-mpi_thread_single
# https://github.com/open-mpi/ompi/issues/6240
# https://github.com/open-mpi/ompi/issues/9763
# https://developer.skao.int/projects/ska-sdp-exec-iotest/en/latest/content/mpi_settings.html#slurm-specific-settings
# https://docs.nvidia.com/networking/display/hpcxv211/hcoll

export OMPI_MCA_btl_tcp_if_exclude=enp1s0,ib0,lo,virbr0
export OMPI_MCA_coll=^hcoll
export DASK_MPI__INTERFACE=brmgt
export DASK_MPI__PROTOCOL=tcp
export DASK_MPI__NTHREADS=1
export DASK_MPI__LOCAL_DIRECTORY=${TMPDIR}
export DASK_MPI__LOG_DIR="${PROJECTROOT}/logs/dask"

mkdir -p DASK_MPI_LOG_DIR
srun -n ${SLURM_NTASKS} python leonardo/bin/test_dask_mpi.py --log-level=debug

# Deactivate all environments (useless when run in a SLURM job)
source leonardo/environment/deactivate
