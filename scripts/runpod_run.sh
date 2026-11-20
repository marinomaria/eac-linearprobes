#!/usr/bin/env bash
set -euo pipefail

# Run a script on RunPod and log to output/. Arguments after the script are forwarded to it.
#
#   bash scripts/runpod_run.sh extract.py --model qwen3.8-27b --data cues
#   bash scripts/runpod_run.sh extract.py --model qwen3.8-27b --data data/mix-main.jsonl --batch-size 16

SCRIPT="${1:?Usage: $0 <script.py> [args...]}"
shift

source /root/venvs/eval-awareness/bin/activate
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"

LOG_FILE="output/runpod_${SCRIPT%.py}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p output
python "${SCRIPT}" "$@" 2>&1 | tee "${LOG_FILE}"
echo "Log: ${LOG_FILE}"
