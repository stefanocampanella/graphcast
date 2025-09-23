#! /usr/bin/env bash

# This script compares local and remote trees using rsync --dry-run, similar to sync.sh
# Usage:
#   leonardo/bin/diff.sh push [additional rsync args]
#   leonardo/bin/diff.sh pull [additional rsync args]
#
# For each changed path reported by rsync it will:
# - skip directories
# - if it's a file existing only locally: print "ONLY LOCAL: <path>"
# - if it's a file existing only remotely: print "ONLY REMOTE: <path>"
# - if it exists on both sides: print a unified diff

set -euo pipefail

# Default values
LOCAL_ROOT=$(git rev-parse --show-toplevel)
cd "${LOCAL_ROOT}" || exit 1

# Load Leonardo environment
source "leonardo/.env"
REMOTE_HOST="${LEONARDO_USER}@${LEONARDO_DATAMOVER}"
REMOTE_HOST_LOGIN="${LEONARDO_USER}@${LEONARDO_LOGIN}"
REMOTE_ROOT="${LEONARDO_CODE_ROOT}"

# Common SSH options (avoid host key strict checking)
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Determine direction
if [[ ${1:-} = 'pull' ]]; then
  SOURCE="${REMOTE_HOST}:${REMOTE_ROOT}/"
  DESTINATION="./"
  DIRECTION="pull"
elif [[ ${1:-} = 'push' ]]; then
  SOURCE="./"
  DESTINATION="${REMOTE_HOST}:${REMOTE_ROOT}/"
  DIRECTION="push"
else
  echo "First argument must be either 'push' or 'pull'" >&2
  exit 1
fi
shift || true

# Build rsync command for a dry-run with a parsable output
# %i = itemize changes string (first char encodes type: f=file, d=dir, etc.)
# %n = filename (relative path)
# We also honor the same ignore list used by sync.sh
RSYNC_CMD=(rsync -an --exclude-from='leonardo/.ignore' --out-format='%i %n')

# Append any extra args provided by the user (after push/pull)
if [[ $# -gt 0 ]]; then
  RSYNC_CMD+=("$@")
fi

# Capture output
# shellcheck disable=SC2207
mapfile -t CHANGES < <("${RSYNC_CMD[@]}" "${SOURCE}" "${DESTINATION}" | sed '/^sending incremental file list$/d')

# Nothing to do?
if [[ ${#CHANGES[@]} -eq 0 ]]; then
  echo "No differences detected by rsync (direction: ${DIRECTION})."
  exit 0
fi

# Iterate through reported changes
for line in "${CHANGES[@]}"; do
  # Expect format: "<itemize> <path>"
  # Skip empty lines
  [[ -z "${line}" ]] && continue

  itemize=${line%% *}
  relpath=${line#* }

  # Some lines may still be odd; ensure we have both parts
  if [[ -z "${itemize}" || -z "${relpath}" ]]; then
    continue
  fi

  typechar=${itemize:1:1}
  # Skip directories
  if [[ "${typechar}" == "d" ]]; then
    continue
  fi

  # Determine absolute paths on both sides
  local_path="${LOCAL_ROOT}/${relpath}"
  remote_path="${REMOTE_ROOT}/${relpath}"

  # Check existence
  local_exists=false
  remote_exists=false

  if [[ -f "${local_path}" ]]; then
    local_exists=true
  fi

  if ssh ${SSH_OPTS} "${REMOTE_HOST_LOGIN}" bash -lc \"test -f \"${remote_path}\"\"; then
    remote_exists=true
  fi

  if [[ "${local_exists}" == true && "${remote_exists}" == false ]]; then
    echo "ONLY LOCAL: ${relpath}"
    continue
  fi

  if [[ "${local_exists}" == false && "${remote_exists}" == true ]]; then
    echo "ONLY REMOTE: ${relpath}"
    continue
  fi

  if [[ "${local_exists}" == true && "${remote_exists}" == true ]]; then
    echo "DIFF: ${relpath}"
    # Show a unified diff. If files are binary, diff will note it.
    # Indent diff for readability.
    if ! diff -u --label "local:${relpath}" --label "remote:${relpath}" "${local_path}" <(ssh ${SSH_OPTS} "${REMOTE_HOST_LOGIN}" bash -lc \"cat \"${remote_path}\"\") | sed 's/^/    /'; then
      # diff returns non-zero when files differ; that's expected. We already printed the diff.
      true
    fi
    echo
    continue
  fi

  # If neither exists, it's unexpected (e.g., due to filters). Just show the line.
  echo "UNKNOWN (neither side has a regular file): ${relpath} (${itemize})"

done
