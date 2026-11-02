#!/usr/bin/env python
"""
Probe the shared data mix (main and discovery sets) for evaluation awareness.

The mix is built by `evalaware mix` in jlens_evalaware (src/evalaware/mix.py): EvalAwareBench,
bare HarmBench, FORTRESS and MASK prompts, one JSON record per line with `id`, `dataset`,
`messages` and `meta` (`meta["set"]` is main/discovery, `meta["unit"]` the resampling unit).

Steps:
  1. Train one probe per layer per system-prompt variant (or load saved probes), as in run.py
  2. Project every mix prompt onto every variant's and layer's probe. Each prompt is fed as its
     full message list: MASK prompts carry their own system prompt (and doubling_down a scripted
     first exchange), so the probe reads the same conversation every other instrument sees
  3. Save per-prompt projections to output/<model>/projections_<data stem>.jsonl

Usage:
    python run_mix.py --model qwen3.5-0.8b --limit 50
    python run_mix.py --model qwen3.8-27b --batch-size 16 --load-probes output/qwen3.8-27b/probes.pt
    python run_mix.py --model qwen3.8-27b --data data/mix-main.jsonl
"""
import argparse
import json
import os
import random
from pathlib import Path

import torch

from run import OUTPUT_DIR, get_probes
from src.model import load_model
from src.probe import project_conversations

MIX_FILES = ["data/mix-main.jsonl", "data/mix-discovery.jsonl"]


def read_mix(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Alias from src/model.py:MODELS or HF repo id")
    parser.add_argument("--data", nargs="+", default=MIX_FILES, help="Mix .jsonl files written by `evalaware mix`")
    parser.add_argument("--limit", type=int, default=None, help="Randomly sample N prompts per file (seeded)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=16384,
                        help="Cap on a batch's padded tokens; mix prompts reach ~17k tokens")
    parser.add_argument("--load-probes", default=None, help="Path to saved probes.pt (skip training)")
    args = parser.parse_args()

    for path in args.data:
        if not os.path.exists(path):
            parser.error(f"{path} not found: build it with `uv run evalaware mix --out-dir <dir>` in jlens_evalaware")

    out_dir = os.path.join(OUTPUT_DIR, args.model.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    model, tokenizer, n_layers = load_model(args.model)
    probes = get_probes(model, tokenizer, n_layers, out_dir, args.load_probes, args.batch_size)
    names = list(probes)
    stacked = torch.stack([probes[n] for n in names])

    for path in args.data:
        rows = read_mix(path)
        if args.limit is not None:
            rows = random.Random(0).sample(rows, min(args.limit, len(rows)))
        print(f"Projecting {len(rows)} prompts from {path}")
        projections = project_conversations(model, tokenizer, stacked, [r["messages"] for r in rows],
                                            n_layers, args.batch_size, args.max_batch_tokens)

        out_path = os.path.join(out_dir, f"projections_{Path(path).stem}{'' if args.limit is None else f'_n{args.limit}'}.jsonl")
        with open(out_path, "w") as f:
            for row, proj in zip(rows, projections.tolist()):
                f.write(json.dumps({"id": row["id"], "dataset": row["dataset"], "meta": row["meta"],
                                    "projections": dict(zip(names, proj))}) + "\n")
        print(f"Done. {len(rows)} rows x {len(names)} variants x {n_layers} layers saved to {out_path}\n")


if __name__ == "__main__":
    main()
