#!/usr/bin/env bash
set -euo pipefail

# Train the probe recipe on everything ("all", the probes that score the mix) and once per held-out source
# (the generalization evidence report.py puts next to the cue AUROCs). No GPU needed.
#
#   bash scripts/train_all.sh output/qwen3.8-27b/acts/train [more activation dirs...]
#
# Each run also gets any extra train_probes.py args from TRAIN_ARGS, e.g. TRAIN_ARGS="--balance label".

TRAIN_ARGS="${TRAIN_ARGS:-}"
run() {
    local name="$1"; shift
    # shellcheck disable=SC2086
    python train_probes.py "${ACTS[@]}" --name "${name}" ${TRAIN_ARGS} "$@"
}

ACTS=("${@:?Usage: $0 <acts dir> [more acts dirs...]}")
run all
run hold-dolly --holdout-family 'sysprompt:*'
run hold-needham --holdout-family 'needham:*'
run hold-awarebench --holdout-family 'awarebench:*'
run hold-chat --holdout-family 'needham:chat:*' 'awarebench:chat:*' 'wildchat:*'
run hold-agent --holdout-family 'needham:agent:*' 'awarebench:agent:*'
if grep -q '"family": "verbalized:' "${ACTS[@]/%//rows.jsonl}"; then
    run hold-verbalized --holdout-family 'verbalized:*'
fi
