#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/download.sh [--env-file FILE] DATASET CLIMATOLOGY ANOMALY_STD OUTPUT [--config FILE] [--overwrite]
                           [--log-level LEVEL]

Write the standardized anomalies of DATASET to the local Zarr store OUTPUT with `graphcast download` in Apptainer.
DATASET, CLIMATOLOGY and ANOMALY_STD are local paths or fsspec URLs (e.g. s3://ogs-arco-ocean/datasets/res=0p25).
Settings (Zarr format, shards, fsspec options, threads) are read from --config, see configs/download.
Runs on login nodes (no GPU); the full ARCO-OCEAN dataset takes a few hours, e.g. run it in screen or tmux.
  --env-file FILE   Environment file (default: clusters/jupiter/.env), providing APPTAINER_IMAGE and
                    APPTAINERENV_PYTHONPATH
  --help            Show help
Local paths must be visible in the container (on JUPITER, /e/... is).
EOF
}

fail() {
    printf 'download.sh: %s\n' "$*" >&2
    exit 1
}

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE=$PROJECT_DIR/clusters/jupiter/.env

case "${1:-}" in
    --help)
        usage
        exit 0
        ;;
    --env-file)
        [[ -n ${2:-} ]] || fail 'missing value for --env-file'
        ENV_FILE=$2
        shift 2
        ;;
esac
[[ $# -ge 4 ]] || { usage >&2; exit 1; }

[[ -f $ENV_FILE && -r $ENV_FILE ]] || fail "environment file is not readable: $ENV_FILE"
set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a
APPTAINER_IMAGE=${APPTAINER_IMAGE:-$PROJECT_DIR/graphcast.sif}
[[ -f $APPTAINER_IMAGE && -r $APPTAINER_IMAGE ]] || fail "image is not readable: $APPTAINER_IMAGE"

exec apptainer exec "$APPTAINER_IMAGE" graphcast download "$@"
