# Load provided Leonardo modules.
# Notice: **don't** load other modules beforehand (e.g. certain other versions of cmake), as in tests bugged pigz/tar made the build fail.

export LEONARDO_MODULES_DEFINED=true

function load_modules {
    module load gcc/12.2.0 openmpi/4.1.6--gcc--12.2.0-cuda-12.2 cmake/4.1.2 perl/5.38.0
    export LEONARDO_MODULES_LOADED=true
}

function unload_modules {
    module unload gcc/12.2.0 openmpi/4.1.6--gcc--12.2.0-cuda-12.2 cmake/4.1.2 perl/5.38.0
    export LEONARDO_MODULES_LOADED=false
}
