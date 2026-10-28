#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/download.sh DATASET CLIMATOLOGY ANOMALY_STD OUTPUT [--config FILE] [--overwrite] [--log-level LEVEL]
       scripts/download.sh --mask DATASET OUTPUT [--config FILE] [--overwrite] [--log-level LEVEL]

Write the standardized anomalies of DATASET to the local Zarr store OUTPUT with `graphcast download` in Apptainer,
or, with --mask, its mask at one depth level with `graphcast download-mask`, using the inherited environment
(APPTAINER_IMAGE, default PROJECT_DIR/graphcast.sif; APPTAINERENV_*).
DATASET, CLIMATOLOGY and ANOMALY_STD are local paths or URLs (e.g. s3://bucket/path). S3 credentials come from the
environment, e.g. AWS_PROFILE, or FSSPEC_S3_ANON=true for anonymous access.
Settings are read from --config, see configs/download: Zarr format, shards, compression and threads of the
anomalies (zarr_*.toml); mask variable and level (mask.toml).
Local paths must be visible in the container.
EOF
}

fail() {
    printf 'download.sh: %s\n' "$*" >&2
    exit 1
}

if [[ ${1:-} == --help ]]; then
    usage
    exit 0
fi
COMMAND=download
MIN_ARGS=4
if [[ ${1:-} == --mask ]]; then
    COMMAND=download-mask
    MIN_ARGS=2
    shift
fi
[[ $# -ge $MIN_ARGS ]] || { usage >&2; exit 1; }

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
APPTAINER_IMAGE=${APPTAINER_IMAGE:-$PROJECT_DIR/graphcast.sif}
[[ -f $APPTAINER_IMAGE && -r $APPTAINER_IMAGE ]] || fail "image is not readable: $APPTAINER_IMAGE"
# AWS profiles may run a credential_process helper installed in ~/.local/bin.
export APPTAINERENV_PREPEND_PATH=${APPTAINERENV_PREPEND_PATH:-$HOME/.local/bin}

exec apptainer exec "$APPTAINER_IMAGE" graphcast "$COMMAND" "$@"
