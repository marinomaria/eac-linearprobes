#!/usr/bin/env bash
set -euo pipefail

# Run the pipeline on RunPod and log to a file. Extra args are forwarded to the script
# (SCRIPT, default run.py).
#
#   bash scripts/runpod_run.sh qwen3.8-27b
#   bash scripts/runpod_run.sh qwen3.8-27b --limit 500 --batch-size 16
#   SCRIPT=run_mix.py bash scripts/runpod_run.sh qwen3.8-27b --load-probes output/qwen3.8-27b/probes.pt

MODEL="${1:?Usage: $0 <model> [extra script args...]}"
shift
SCRIPT="${SCRIPT:-run.py}"

source /root/venvs/eval-awareness/bin/activate
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"

LOG_FILE="output/runpod_${SCRIPT%.py}_${MODEL//\//_}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p output
python "${SCRIPT}" --model "${MODEL}" "$@" 2>&1 | tee "${LOG_FILE}"
echo "Log: ${LOG_FILE}"
