#!/usr/bin/env bash
# Train the 5000-step demo (configs/training/demo_5000.toml: 8 nodes, 32 samples per step, validation loss every 250
# steps) once per mesh, number of message passing steps and seed, to compare the validation loss of model sizes. Only
# the mesh, gnn_msg_steps and the seed change (and, where a case does not fit in GPU memory otherwise, the edge chunk
# size, which does not change results). The validation windows are the same in every run; the seed sets the initial
# weights and the order of the training samples.
# Usage: clusters/jupiter/slurm/mesh_study.sh [--account A] STUDY_NAME SEEDS MESH:MSG_STEPS[:EDGE_CHUNK_SIZE[:TIME]] ...
#   e.g. clusters/jupiter/slurm/mesh_study.sh mesh_msg 42,43 coarse:8 50km:16::03:00:00 40km:16:262144:06:00:00
# Configs: runs/STUDY_NAME/configs/demo_5000_MESH_msgN_seedS.toml; TensorBoard: DATA_ROOT/tensorboard/STUDY_NAME/<case>.
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ACCOUNT=training2647
if [[ ${1:-} == --account ]]; then
    ACCOUNT=$2
    shift 2
fi
STUDY=${1:?study name}
SEEDS=${2:?comma-separated seeds}
shift 2
(($# > 0)) || { echo "no cases given" >&2; exit 1; }
# shellcheck source=/dev/null
source "$PROJECT_DIR/clusters/jupiter/.env"
CONFIG_DIR="$PROJECT_DIR/runs/$STUDY/configs"
mkdir -p "$CONFIG_DIR"
# The code of this checkout (e.g. a worktree), not the one of the site defaults.
ENV_FILE="$CONFIG_DIR/study.env"
sed -e "s|^APPTAINERENV_PYTHONPATH=.*|APPTAINERENV_PYTHONPATH=$PROJECT_DIR/src|" "$PROJECT_DIR/clusters/jupiter/.env" \
    > "$ENV_FILE"
grep -qx "APPTAINERENV_PYTHONPATH=$PROJECT_DIR/src" "$ENV_FILE" || { echo "PYTHONPATH not set in $ENV_FILE" >&2; exit 1; }
IFS=, read -r -a SEED_LIST <<< "$SEEDS"
for ENTRY in "$@"; do
    IFS=: read -r MESH MSG CHUNKS HOURS MINUTES SECONDS <<< "$ENTRY"
    [[ $MSG =~ ^[0-9]+$ ]] || { echo "no message passing steps in $ENTRY" >&2; exit 1; }
    CHUNKS=${CHUNKS:-0}
    TIME=03:00:00
    [[ -n ${HOURS:-} ]] && TIME="$HOURS:$MINUTES:$SECONDS"
    [[ -f $GEOMETRY_DIR/uniform_$MESH.msh ]] || { echo "no mesh $GEOMETRY_DIR/uniform_$MESH.msh" >&2; exit 1; }
    for SEED in "${SEED_LIST[@]}"; do
        NAME="demo_5000_${MESH}_msg${MSG}_seed$SEED"
        sed -e "s|^path = \"/data/geometry/[^\"]*\"|path = \"/data/geometry/uniform_$MESH.msh\"|" \
            -e "s|^gnn_msg_steps = [0-9]*|gnn_msg_steps = $MSG|" \
            -e "s|^edge_chunk_size = [0-9]*|edge_chunk_size = $CHUNKS|" \
            -e "s|^seed = [0-9]*|seed = $SEED|" \
            "$PROJECT_DIR/configs/training/demo_5000.toml" > "$CONFIG_DIR/$NAME.toml"
        for EXPECTED in "uniform_$MESH.msh" "gnn_msg_steps = $MSG" "edge_chunk_size = $CHUNKS" "seed = $SEED" "[validation]"; do
            grep -qF "$EXPECTED" "$CONFIG_DIR/$NAME.toml" || { echo "$EXPECTED not in $NAME.toml" >&2; exit 1; }
        done
        sbatch --account="$ACCOUNT" --nodes=8 --time="$TIME" --job-name="$STUDY-$MESH-msg$MSG-s$SEED" \
            "$PROJECT_DIR/clusters/jupiter/slurm/train.slurm" --env-file "$ENV_FILE" --config-dir "$CONFIG_DIR" \
            --tb-dir "$DATA_ROOT/tensorboard/$STUDY/$NAME" "$NAME" "${STUDY}_$NAME" "${STUDY}_$NAME"
    done
done
