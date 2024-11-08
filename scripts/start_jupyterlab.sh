#! /usr/bin/env bash

# Load prerequisite modules, mainly Python and CUDA
SCRIPT_PATH=$(dirname $0)
source "${SCRIPT_PATH}/modules.sh" || exit

# Change working directory to project root
ROOT=$(git -C "${SCRIPT_PATH}" rev-parse --show-toplevel)
cd "${ROOT}" || exit

# Run Jupyterlab on ephemeral port
source venv/bin/activate
jupyter lab --port=32769
