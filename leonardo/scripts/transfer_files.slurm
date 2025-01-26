#! /usr/bin/env bash
#SBATCH --account=OGS23_PRACE_IT_0
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=240
#SBATCH --mem=30800M
#SBATCH --gres=tmpfs:100G
#SBATCH --requeue

usage() {
    more <<EOF
transfer_files.slurm
    This script transfers files between SOURCE and DESTINATION directories using rsync.
    It provides options for different transfer modes and can handle large file transfers efficiently.

SYNOPSIS
    usage: $0 --help
    usage: $0 [options] SOURCE DESTINATION

DESCRIPTION
    Arguments
        SOURCE                                  Source directory path
        DESTINATION                             Destination directory path

    Options
        --help                                  Shows this help.
EOF
}

# Parse command line options
ARGS=$(getopt --options '' --longoptions "verbose,help" --name "$0" -- "${@}")
if [[ ${?} -ne 0 ]]; then
    usage
    exit 1
fi
eval "set -- ${ARGS}"
while true; do
    case "${1}" in
        (--help)
            usage
            exit 0
            ;;
        (--)
            shift
            break
            ;;
        (*)
            echo "Error: unrecognized option ${1}."
            usage
            exit 1
            ;;
    esac
done

# Check that we have exactly 2 arguments (SOURCE and DESTINATION)
if [[ $# -ne 2 ]]; then
    echo "Error: exactly 2 arguments required (SOURCE and DESTINATION)."
    usage
    exit 1
fi

SOURCE="${1}"
DESTINATION="${2}"

# Validate that SOURCE exists
if [[ ! -d "${SOURCE}" ]]; then
    echo "Error: SOURCE directory '${SOURCE}' does not exist."
    exit 1
fi

# Create DESTINATION directory if it doesn't exist
DESTINATION_DIR=$(dirname "${DESTINATION}")
if [[ ! -d "${DESTINATION_DIR}" ]]; then
    echo "Creating destination directory: ${DESTINATION_DIR}"
    mkdir -p "${DESTINATION_DIR}"
fi

# Build rsync command with options
RSYNC_CMD="rsync -Pav"

# Add trailing slash to source to copy contents, not the directory itself
if [[ "${SOURCE}" != */ ]]; then
    SOURCE="${SOURCE}/"
fi

echo "Starting file transfer..."
echo "Source: ${SOURCE}"
echo "Destination: ${DESTINATION}"
echo "Command: ${RSYNC_CMD} \"${SOURCE}\" \"${DESTINATION}\""
echo "----------------------------------------"

# Execute rsync command
${RSYNC_CMD} "${SOURCE}" "${DESTINATION}"

RSYNC_EXIT_CODE=$?

if [[ ${RSYNC_EXIT_CODE} -eq 0 ]]; then
    echo "----------------------------------------"
    echo "File transfer completed successfully."
else
    echo "----------------------------------------"
    echo "File transfer failed with exit code: ${RSYNC_EXIT_CODE}"
    echo "Job failed. Re-submitting..."
    scontrol requeue $SLURM_JOB_ID
    exit ${RSYNC_EXIT_CODE}
fi