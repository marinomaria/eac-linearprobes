#!/usr/bin/env python
"""
Extract last-token activations for labeled (or unlabeled) conversation rows and store them on disk,
for train_probes.py (training) and score.py (scoring with trained probes). Format: see src/rows.py and
src/activations.py.

--data accepts:
  - a rows .jsonl (src/rows.py format; the data mix files already have id/messages/meta)
  - "dolly": the original paired design, every contrastive stem under every system-prompt variant
  - "evalawarebench[:subset]": EvalAwareBench prompts as single user turns (unlabeled)

Layers default to 6 evenly spaced between 30% and 90% of depth; --layers overrides
(indices into hidden_states, i.e. the residual stream entering that block).

Usage:
    python extract.py --model qwen3.5-0.8b --data dolly --limit 40
    python extract.py --model qwen3.8-27b --data data/train_rows.jsonl --batch-size 16
    python extract.py --model qwen3.8-27b --data evalawarebench --out output/qwen3.8-27b/acts/evalawarebench
"""
import argparse
import json
import os
import random

from run import CONTRASTIVE_DATASET, EVALAWAREBENCH_FIELDS, OUTPUT_DIR, SYSTEM_PROMPTS, load_evalawarebench
from src.activations import default_layers, extract
from src.model import load_model
from src.rows import dolly_rows, read_rows


def build_rows(data, limit):
    if data == "dolly":
        with open(CONTRASTIVE_DATASET) as f:
            stems = [row["stem"].strip() for row in json.load(f)]
        if limit is not None:
            stems = random.Random(0).sample(stems, min(limit, len(stems)))
        with open(SYSTEM_PROMPTS) as f:
            return dolly_rows(stems, json.load(f))
    if data.startswith("evalawarebench"):
        subset = data.partition(":")[2] or "prompts"
        rows = [{"id": str(i), "messages": [{"role": "user", "content": r["prompt"]}], "label": None,
                 "group": str(r["task_id"]), "family": "evalawarebench",
                 "meta": {k: r[k] for k in EVALAWAREBENCH_FIELDS}}
                for i, r in enumerate(load_evalawarebench(subset))]
    else:
        rows = read_rows(data)
    if limit is not None:
        rows = random.Random(0).sample(rows, min(limit, len(rows)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Alias from src/model.py:MODELS or HF repo id")
    parser.add_argument("--data", required=True, help='Rows .jsonl, "dolly" or "evalawarebench[:subset]"')
    parser.add_argument("--out", default=None, help="Output directory (default: output/<model>/acts/<data name>)")
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Seeded random sample (stems, for dolly)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=16384)
    args = parser.parse_args()

    rows = build_rows(args.data, args.limit)
    name = os.path.splitext(os.path.basename(args.data))[0].replace(":", "_")
    out = args.out or os.path.join(OUTPUT_DIR, args.model.replace("/", "_"), "acts",
                                   name + ("" if args.limit is None else f"_n{args.limit}"))

    model, tokenizer, n_layers = load_model(args.model)
    layers = args.layers or default_layers(n_layers)
    print(f"{len(rows)} rows, layers {layers} -> {out}")
    extract(model, tokenizer, rows, layers, n_layers, out, args.model, args.batch_size, args.max_batch_tokens)
    print("Done.")


if __name__ == "__main__":
    main()
