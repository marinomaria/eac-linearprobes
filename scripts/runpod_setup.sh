#!/usr/bin/env bash
set -euo pipefail

# Run on the RunPod pod after cloning the repo to /workspace/eval-awareness.
# Assumes: 1x 80GB GPU (H100/A100), PyTorch+CUDA template, volume at /workspace.
#
#   cd /workspace && git clone <repo_url> eval-awareness
#   bash eval-awareness/scripts/runpod_setup.sh

REPO_DIR="/workspace/eval-awareness"
VENV_DIR="/root/venvs/eval-awareness"

python -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --no-cache-dir -r "${REPO_DIR}/requirements.txt"
# Fast kernels for Qwen3.8's linear-attention layers; without them transformers
# falls back to a slow (but correct) PyTorch implementation.
pip install --no-cache-dir flash-linear-attention || echo "flash-linear-attention install failed (continuing, slower)"
pip install --no-cache-dir causal-conv1d --no-build-isolation || echo "causal-conv1d install failed (continuing, slower)"

# HF cache on the volume disk, not the small container disk.
grep -q "HF_HOME=/workspace/hf-cache" ~/.bashrc || echo "export HF_HOME=/workspace/hf-cache" >> ~/.bashrc

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')"
echo
echo "Setup complete. Next: cd ${REPO_DIR} && bash scripts/runpod_run.sh qwen3.8-27b"
