#! /usr/bin/env bash

PROJECTROOT=$(git rev-parse --show-toplevel)
cd "${PROJECTROOT}" || exit 1

# Activate environment
source leonardo/environment/activate

jupyter lab --ip="$(hostname -i)" "${@}"

source leonardo/environment/deactivate
