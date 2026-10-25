#!/usr/bin/env bash
# Train the 5000-step demo (configs/training/demo_5000.toml: 8 nodes, 32 samples per step, 10 levels) once per mesh,
# to compare the loss for different mesh sizes. Only the mesh changes (and, where a mesh does not fit in GPU memory
# otherwise, the edge chunk size, which does not change results); the parameters do not depend on the mesh, so every
# run starts from the same weights (seed 42) and reads the same samples.
# Usage: clusters/jupiter/slurm/mesh_study.sh STUDY_NAME MESH[:EDGE_CHUNK_SIZE[:TIME[:lowmem]]] ...
#   e.g. clusters/jupiter/slurm/mesh_study.sh mesh_study coarse 40km::03:00:00 35km:262144 fine:262144:08:00:00:lowmem
# lowmem: offload activations (policy.remat_loss) and load with 2 Grain workers and small buffers, which leaves host
# memory for the offloaded activations (an environment file next to the configs).
# Configs: runs/STUDY_NAME/configs/demo_5000_MESH.toml; TensorBoard: DATA_ROOT/tensorboard/STUDY_NAME_MESH.
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
STUDY=${1:?study name}
shift
(($# > 0)) || { echo "no meshes given" >&2; exit 1; }
# shellcheck source=/dev/null
source "$PROJECT_DIR/clusters/jupiter/.env"
CONFIG_DIR="$PROJECT_DIR/runs/$STUDY/configs"
mkdir -p "$CONFIG_DIR"
for ENTRY in "$@"; do
    IFS=: read -r MESH CHUNKS HOURS MINUTES SECONDS MODE <<< "$ENTRY"
    CHUNKS=${CHUNKS:-0}
    TIME=03:00:00
    [[ -n ${HOURS:-} ]] && TIME="$HOURS:$MINUTES:$SECONDS"
    ENV_ARGS=()
    [[ -f $GEOMETRY_DIR/uniform_$MESH.msh ]] || { echo "no mesh $GEOMETRY_DIR/uniform_$MESH.msh" >&2; exit 1; }
    NAME="demo_5000_$MESH"
    sed -e "s|^path = \"/data/geometry/[^\"]*\"|path = \"/data/geometry/uniform_$MESH.msh\"|" \
        -e "s|^edge_chunk_size = [0-9]*|edge_chunk_size = $CHUNKS|" \
        "$PROJECT_DIR/configs/training/demo_5000.toml" > "$CONFIG_DIR/$NAME.toml"
    grep -q "uniform_$MESH.msh" "$CONFIG_DIR/$NAME.toml" || { echo "mesh path not set in $NAME.toml" >&2; exit 1; }
    if [[ ${MODE:-} == lowmem ]]; then
        sed -i "s|^remat_loss = false|remat_loss = true|" "$CONFIG_DIR/$NAME.toml"
        sed -e "s|^GRAPHCAST_GRAIN_NUM_WORKERS=.*|GRAPHCAST_GRAIN_NUM_WORKERS=2|" \
            -e "s|^GRAPHCAST_GRAIN_PER_WORKER_BUFFER_SIZE=.*|GRAPHCAST_GRAIN_PER_WORKER_BUFFER_SIZE=1|" \
            -e "s|^GRAPHCAST_GRAIN_PREFETCH_BUFFER_SIZE=.*|GRAPHCAST_GRAIN_PREFETCH_BUFFER_SIZE=1|" \
            "$PROJECT_DIR/clusters/jupiter/.env" > "$CONFIG_DIR/$NAME.env"
        ENV_ARGS=(--env-file "$CONFIG_DIR/$NAME.env")
    fi
    sbatch --account=training2647 --nodes=8 --time="$TIME" --job-name="$STUDY-$MESH" \
        "$PROJECT_DIR/clusters/jupiter/slurm/train.slurm" "${ENV_ARGS[@]}" --config-dir "$CONFIG_DIR" \
        --tb-dir "$DATA_ROOT/tensorboard/${STUDY}_$MESH" "$NAME" "${STUDY}_$MESH" "${STUDY}_$MESH"
done
