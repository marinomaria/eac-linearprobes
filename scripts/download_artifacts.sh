#!/usr/bin/env bash
set -euo pipefail

# Run LOCALLY to copy output/ from a RunPod pod.
# IP and PORT are in the pod's "Connect" tab under "SSH over exposed TCP".
#
#   bash scripts/download_artifacts.sh <IP> <PORT> [local_dir]

IP="${1:?Usage: $0 <IP> <PORT> [local_dir]}"
PORT="${2:?Usage: $0 <IP> <PORT> [local_dir]}"
LOCAL_DIR="${3:-./runpod-artifacts}"

mkdir -p "${LOCAL_DIR}"
scp -r -P "${PORT}" -i "${HOME}/.ssh/id_ed25519" "root@${IP}:/workspace/eval-awareness/output/." "${LOCAL_DIR}/"
ls -lhR "${LOCAL_DIR}"
echo "After verifying, terminate the pod to stop charges."
