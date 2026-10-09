#!/usr/bin/env python
"""
Probe EvalAwareBench prompts for evaluation awareness.

Steps:
  1. Train one probe per layer on the contrastive dataset (or load saved probes)
  2. Project every EvalAwareBench prompt onto every layer's probe (forward pass only)
  3. Save per-prompt projections to output/<model>/

Layer selection is deferred to scripts/analyze.py: all layers come from the same
forward pass, so there is no extra cost in keeping them.

Usage:
    python run.py --model qwen3.5-0.8b --limit 200
    python run.py --model qwen3.8-27b
"""
import argparse
import json
import os
import random

import torch
from huggingface_hub import hf_hub_download

from src.model import load_model
from src.probe import project, train_probes

CONTRASTIVE_DATASET = "datasets/contrastive_dataset_dolly.json"
EVALAWAREBENCH = "aisa-group/EvalAwareBench"
OUTPUT_DIR = "output"


def load_evalawarebench(subset):
    path = hf_hub_download(EVALAWAREBENCH, f"data/{subset}.jsonl", repo_type="dataset")
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Alias from src/model.py:MODELS or HF repo id")
    parser.add_argument("--subset", default="prompts", choices=["prompts", "prompts_safety", "prompts_capability"])
    parser.add_argument("--limit", type=int, default=None, help="Randomly sample N prompts (seeded)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--load-probes", default=None, help="Path to saved probes.pt (skip training)")
    args = parser.parse_args()

    out_dir = os.path.join(OUTPUT_DIR, args.model.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    model, tokenizer, n_layers = load_model(args.model)

    if args.load_probes:
        probes = torch.load(args.load_probes)
    else:
        with open(CONTRASTIVE_DATASET) as f:
            stems = [row["stem"].strip() for row in json.load(f)]
        probes = train_probes(model, tokenizer, stems, n_layers, args.batch_size)
        torch.save(probes, os.path.join(out_dir, "probes.pt"))

    rows = load_evalawarebench(args.subset)
    if args.limit is not None:
        rows = random.Random(0).sample(rows, min(args.limit, len(rows)))
    print(f"Projecting {len(rows)} prompts from {EVALAWAREBENCH}/{args.subset}")

    projections = project(model, tokenizer, probes, [r["prompt"] for r in rows], n_layers, args.batch_size)

    out_path = os.path.join(out_dir, f"projections_{args.subset}{'' if args.limit is None else f'_n{args.limit}'}.jsonl")
    with open(out_path, "w") as f:
        for row, proj in zip(rows, projections.tolist()):
            meta = {k: row[k] for k in ("task_id", "task_name", "valence", "factors_varied", "num_factors_varied", "config")}
            f.write(json.dumps({**meta, "projections": proj}) + "\n")

    print(f"\nDone. {len(rows)} rows x {n_layers} layers saved to {out_path}")


if __name__ == "__main__":
    main()
