#!/usr/bin/env bash
set -euo pipefail

# Run LOCALLY to copy output/ from a RunPod pod.
# IP and PORT are in the pod's "Connect" tab under "SSH over exposed TCP".
#
#   bash scripts/download_artifacts.sh <IP> <PORT> [local_dir]
#
# Set SSH_KEY to use a key other than ~/.ssh/id_ed25519:
#
#   SSH_KEY=~/.ssh/id_ed25519_team bash scripts/download_artifacts.sh <IP> <PORT>

IP="${1:?Usage: $0 <IP> <PORT> [local_dir]}"
PORT="${2:?Usage: $0 <IP> <PORT> [local_dir]}"
LOCAL_DIR="${3:-./runpod-artifacts}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519}"

mkdir -p "${LOCAL_DIR}"
scp -r -P "${PORT}" -i "${SSH_KEY}" "root@${IP}:/workspace/eval-awareness/output/." "${LOCAL_DIR}/"
ls -lhR "${LOCAL_DIR}"
echo "After verifying, terminate the pod to stop charges."
