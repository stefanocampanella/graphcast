#! /usr/bin/env bash

# The following script tries to bootstrap the model using the least possible number of 
# dependencies provided by CINECA. Here's some motivation on why.
# 
# Regarding JAX, on could try with cineca-ai/4.3.0. 
# However, it should probably be downgraded to cineca-ai/3.0.1 because of: 
# 
# https://github.com/google/jax/issues/15384
# 
# But then one should find a functioning version of jax and jaxlib to be used with graphcast...
# In that case, it would seem more reasonable to not have a working jax.profiler.trace.
#
# Also, the provided module for gdal is incompatible with the current version of jaxlib, 
# hence a custom version would have to be built beforehand and loaded here.
#
# A working version of gdal could be built with something similar to the following command:
# $ module load spack/0.21.0-68a
# $ spack install gdal@3.7.3 build_system=cmake build_type=Release generator=ninja %gcc@12.2.0
# $ spack module tcl refresh <new_gdal_hash>
# ... and then loaded with
# $ module load spack/0.21.0-68a
# $ module load /leonardo/pub/userexternal/${USER}/spack-0.21.0-5.2/modules/gdal/3.7.3--gcc--12.2.0
#
# But building jaxlib against cuda/cudnn/nccl provided by CINECA, which are obsolete, is a pain in the neck...
# 
# The approach used here is to rely on spack and pip to build a self-contained reproducible environment. 
# The bare minimum is to just load modules for the compiler and MPI.


usage() {
    more <<EOF
NAME
    This script prepares the environment to run the model on Leonardo.

SYNOPSIS
    usage: $0 --help
    usage: $0 [options]

DESCRIPTION
    Setup options
        --clear                                 Clear all local environment files.
        --skip-spack                            Do not setup spack.
        --skip-venv                             Do not setup Python virtual environment (implies --skip-venv-compile, --skip-venv-download, --skip-venv-install).
        --skip-venv-compile                     Do not compile requirements.txt.
        --skip-venv-download                    Do not download packages from PyPI.
        --skip-venv-install                     Do not install downloaded packages.
        --help                                  Shows this help.
EOF
}

LONGOPTS='help,clear,skip-spack,skip-venv,skip-venv-compile,skip-venv-download,skip-venv-install'
ARGS=$(getopt --options '' --longoptions ${LONGOPTS} -- "${@}")
if [[ ${?} -ne 0 ]]; then
    usage
    exit 1
fi

# OPTIONS DEFAULT VALUES
CLEAR=false
BUILD_SPACK=true
ADD_SPACK_MIRRORS=false
BUILD_VENV=true
COMPILE_VENV=true
DOWNLOAD_VENV=true
INSTALL_VENV=true

eval "set -- ${ARGS}"
while true; do
    case "${1}" in
    (--skip-spack)
        BUILD_SPACK=false
        shift
        ;;
    (--add-spack-mirrors)
        ADD_SPACK_MIRRORS=true
        shift
        ;;
    (--skip-venv)
        BUILD_VENV=false
        shift
        ;;
    (--skip-venv-compile)
        COMPILE_VENV=false
        shift
        ;;
    (--skip-venv-download)
        DOWNLOAD_VENV=false
        shift
        ;;
    (--skip-venv-install)
        INSTALL_VENV=false
        shift
        ;;
    (--clear)
        CLEAR=true
	    shift
	    ;;
    (--help)
        usage
        exit 0
        ;;
    (--)
        shift
        break
        ;;
    (*)
	echo "Error: unrecognized option ${1}."
	    usage
        exit 1
        ;;
    esac
done

# Load provided Leonardo modules.
# Notice: **don't** load other modules beforehand (e.g. cmake), in tests bugged pigz/tar will make the build fail.
module load git/2.45.1 gcc/12.2.0 openmpi/4.1.6--gcc--12.2.0-cuda-12.2

ROOT=$(git rev-parse --show-toplevel)
SPACK_VENV_DIR="${ROOT}/.spack-venv"
SPACK_DIR="${ROOT}/.spack"
PKG_CACHE_DIR="${ROOT}/.pkg_cache"
VENV_DIR="${ROOT}/.venv"

SLURM_ACCOUNT=OGS23_PRACE_IT_0
SLURM_PARTITION=boost_usr_prod
# TODO: Is word splitting a problem here? Nobody really understands bash, everybody writes bash scripts...
SLURM_INSTALL_JAXLIB="pip install --no-index --no-cache-dir --find-links=${PKG_CACHE_DIR} jaxlib"
SLURM_TIME=10


# Change working directory to project root (nonetheless, absolute paths are preferred)
cd "${ROOT}" || exit

if [[ $CLEAR == true ]]; then
	echo "Clearing all local environment files"
	rm -rf "${SPACK_VENV_DIR}" "${SPACK_DIR}" "${VENV_DIR}" "${PKG_CACHE_DIR}" "${ROOT}/graphcast.egg-info"
fi

if [[ $BUILD_SPACK == true ]]; then

    # Create a local environment to bootstrap spack
    python3 -m venv --copies "${SPACK_VENV_DIR}"
    
    # Notice: if Google binary cache for spack is among mirrors, e.g.
    #
    # $ spack mirror add google_binary_cache gs://spack/latest
    # 
    # then one has to install GCS at this point, to allow using it later. 
    # 
    # The following should be sufficient
    # 
    # $ source "${SPACK_VENV_DIR}/bin/activate" # activate .spack-venv
    # $ pip install --upgrade pip
    # $ pip install google-cloud-storage
    # $ deactivate # deactivate .spack-venv, as it will not be needed anymore, only spack will use it via SPACK_PYTHON environment variable
    # 
    # Finally, notice that default user credentials have to be installed on the system beforehand.

    # Create a local installation of spack
    git clone -c feature.manyFiles --depth=2 --branch=releases/v0.23 https://github.com/spack/spack.git .spack

    # Setup spack (now SPACK_PYTHON should point to the .spack-venv python)
    # Probably, defining spack="${SPACK_DIR}/bin/spack" might be enough. 
    # However, the following works for sure.
    export SPACK_PYTHON="${SPACK_VENV_DIR}/bin/python3"
    source "${SPACK_DIR}/share/spack/setup-env.sh"

    # Bootstrap spack
    spack bootstrap root "${SPACK_DIR}/bootstrap"
    spack --insecure bootstrap now
   
    if [[ $ADD_SPACK_MIRRORS == true ]]; then 
        spack mirror add v0.23.1 https://binaries.spack.io/v0.23.1
        # spack mirror add google_binary_cache gs://spack/latest
    fi
    
    spack buildcache keys --install --trust

    # Create and activate spack environment
    spack env create default
    spack env activate default

    # Patch the spack.yaml file to allow multiple versions of the same package (unify: when_possible)
    patch "${SPACK_DIR}/var/spack/environments/default/spack.yaml" "${ROOT}/leonardo/environment/spack.yaml.patch"

    # When using the infiniband interface, dask_mpi fails to launch a cluster due to an error which appears to be solved
    # when using a version of ucx compiled with +thread_multiple.
    # spack add openmpi@4.1: fabrics=ucx schedulers=slurm ^ucx +thread_multiple ^slurm@23.11.10
    # However, it's then very hard to get a functioning version of slurm and openmpi,
    # hence it's better to use the system one and fall back to ethernet interface.

    # Is it useful to add gcc and openmpi externals to the current environment? E.g.
    # spack compiler find
    # spack external find --not-buildable gcc openmpi

    # Add python and gdal to the environment, then install
    spack add python@3.11
    spack add gdal@3.10.0

    spack concretize || exit 1

    spack --insecure install || exit 1
    
    spack env deactivate
fi

if [[ $BUILD_VENV == true ]]; then
    
    # Setup spack (now SPACK_PYTHON should point to the .spack-venv python)
    export SPACK_PYTHON="${SPACK_VENV_DIR}/bin/python3"
    source "${SPACK_DIR}/share/spack/setup-env.sh"
    spack env activate default
    
    # Create a virtualenv with pip, setuptools, and wheel
    python -m venv --clear --copies --upgrade-deps "${VENV_DIR}"

    # Activate python venv
    source "${VENV_DIR}/bin/activate"

    pip install pip-tools

    if [[ $COMPILE_VENV == true ]]; then
        # Freeze environment for later reuse
        pip-compile --extra-index-url=https://download.pytorch.org/whl/cpu --no-strip-extras --all-build-deps --all-extras --output-file="${ROOT}/leonardo/environment/requirements.txt" "${ROOT}/pyproject.toml" || exit 1
    fi
    
    if [[ $DOWNLOAD_VENV == true ]]; then   
        # Download packages on login nodes (needs internet connection)
        pip download --dest="${PKG_CACHE_DIR}" --extra-index-url=https://download.pytorch.org/whl/cpu -r "${ROOT}/leonardo/environment/requirements.txt"
    fi

    if [[ $INSTALL_VENV == true ]]; then
	# Numpy is a build-requisite (together with setuptools and wheel) for gdal, install it beforehand
	pip install --no-index --find-links="${PKG_CACHE_DIR}" numpy || exit 1
	
	# Build gdal (most flags might be unnecessary). Requiring a different version from the one installed via spack will make the script fail (check that leonardo/setup.sh and pyproject.toml are coherent).
    export CFLAGS=$(gdal-config --cflags)
    export CXXFLAGS=$(gdal-config --cflags)
    export LDFLAGS=$(gdal-config --libs)
	pip install --no-cache-dir --no-build-isolation --use-pep517 --no-index --find-links="${PKG_CACHE_DIR}" gdal[numpy]==$(gdal-config --version) || exit 1
	
	# Check that raster (numpy) gdal support is working
	python -c "from osgeo import gdal_array" || exit 1

    # Afterwards, install jaxlib on a GPU node (apparently requires CUDA drivers)
	srun --account=${SLURM_ACCOUNT} --partition=${SLURM_PARTITION} --time=${SLURM_TIME} --ntasks=1 --cpus-per-task=8 --gres=gpu:1 ${SLURM_INSTALL_JAXLIB} || exit 1

	# Finally, install the remaining packages
	pip install --no-index --find-links="${PKG_CACHE_DIR}" -e "${ROOT}[download,interactive,profile,train]" || exit 1

    fi

    # Deactivate venv and spack environments
    deactivate
    spack env deactivate
fi
