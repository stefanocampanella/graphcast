#! /usr/bin/env bash

# Default values
LOCAL_ROOT=$(git rev-parse --show-toplevel)
cd "${LOCAL_ROOT}" || exit

source "leonardo/.env"
REMOTE_ROOT=${LEONARDO_USER}@${LEONARDO_DATAMOVER}:${LEONARDO_CODE_ROOT}

# Common SSH options (avoid host key strict checking)
SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

if [[ ${1} = 'pull' ]]; then
  SOURCE="${REMOTE_ROOT}/"
  DESTINATION="./"
elif [[ ${1} = 'push' ]]; then
  SOURCE="./"
  DESTINATION="${REMOTE_ROOT}/"
else
  echo "First argument must be either 'push' or 'pull'"
  exit 1
fi

rsync -av -e "ssh ${SSH_OPTS}" --exclude-from='leonardo/.ignore' "${@:2}" "${SOURCE}" "${DESTINATION}"