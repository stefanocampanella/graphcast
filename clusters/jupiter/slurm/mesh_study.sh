#!/usr/bin/env bash
# Train the 5000-step demo (configs/training/demo_5000.toml: 8 nodes, 32 samples per step, 10 levels) once per mesh,
# to compare the loss for different mesh sizes. Only the mesh changes (and, where a mesh does not fit in GPU memory
# otherwise, the edge chunk size, which does not change results); the parameters do not depend on the mesh, so every
# run starts from the same weights (seed 42) and reads the same samples.
# Usage: clusters/jupiter/slurm/mesh_study.sh STUDY_NAME MESH[:EDGE_CHUNK_SIZE] ...
#   e.g. clusters/jupiter/slurm/mesh_study.sh mesh_study coarse 70km 40km 35km:262144
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
    MESH=${ENTRY%%:*}
    CHUNKS=0
    [[ $ENTRY == *:* ]] && CHUNKS=${ENTRY#*:}
    [[ -f $GEOMETRY_DIR/uniform_$MESH.msh ]] || { echo "no mesh $GEOMETRY_DIR/uniform_$MESH.msh" >&2; exit 1; }
    NAME="demo_5000_$MESH"
    sed -e "s|^path = \"/data/geometry/[^\"]*\"|path = \"/data/geometry/uniform_$MESH.msh\"|" \
        -e "s|^edge_chunk_size = [0-9]*|edge_chunk_size = $CHUNKS|" \
        "$PROJECT_DIR/configs/training/demo_5000.toml" > "$CONFIG_DIR/$NAME.toml"
    grep -q "uniform_$MESH.msh" "$CONFIG_DIR/$NAME.toml" || { echo "mesh path not set in $NAME.toml" >&2; exit 1; }
    sbatch --account=training2647 --nodes=8 --time=03:00:00 --job-name="$STUDY-$MESH" \
        "$PROJECT_DIR/clusters/jupiter/slurm/train.slurm" --config-dir "$CONFIG_DIR" \
        --tb-dir "$DATA_ROOT/tensorboard/${STUDY}_$MESH" "$NAME" "${STUDY}_$MESH" "${STUDY}_$MESH"
done
