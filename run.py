#!/usr/bin/env python
"""
Probe EvalAwareBench prompts for evaluation awareness.

Steps:
  1. For each system-prompt variant, train one probe per layer on the contrastive dataset
     (or load saved probes)
  2. Project every EvalAwareBench prompt onto every variant's and layer's probe, in a
     single forward pass per prompt
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
SYSTEM_PROMPTS = "datasets/system_prompts.json"
EVALAWAREBENCH = "aisa-group/EvalAwareBench"
OUTPUT_DIR = "output"


def load_evalawarebench(subset):
    path = hf_hub_download(EVALAWAREBENCH, f"data/{subset}.jsonl", repo_type="dataset")
    with open(path) as f:
        return [json.loads(line) for line in f]


def print_probe_similarity(probes):
    """Cosine similarity between variants' probe directions, summarized over layers.

    Variants with near-identical directions will score the benchmark near-identically.
    """
    names = list(probes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            # Skip layer 0: its probe is all zeros (see train_probes).
            cos = (probes[a][1:] * probes[b][1:]).sum(dim=-1)
            print(f"  cos({a}, {b}) over layers 1+: min={cos.min():.3f} "
                  f"median={cos.median():.3f} max={cos.max():.3f}")


def get_probes(model, tokenizer, n_layers, out_dir, load_path=None, batch_size=8):
    """Load saved probes, or train one probe set per system-prompt variant and save it to out_dir/probes.pt.

    Returns:
        {variant_name: float32 tensor [n_layers, d_model]}
    """
    if load_path:
        probes = torch.load(load_path)
    else:
        with open(CONTRASTIVE_DATASET) as f:
            stems = [row["stem"].strip() for row in json.load(f)]
        with open(SYSTEM_PROMPTS) as f:
            variants = json.load(f)
        probes = {}
        for v in variants:
            print(f"Variant '{v['name']}'")
            probes[v["name"]] = train_probes(model, tokenizer, stems, v["eval"], v["deploy"], n_layers, batch_size)
        torch.save(probes, os.path.join(out_dir, "probes.pt"))
    print_probe_similarity(probes)
    return probes


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

    probes = get_probes(model, tokenizer, n_layers, out_dir, args.load_probes, args.batch_size)
    names = list(probes)

    rows = load_evalawarebench(args.subset)
    if args.limit is not None:
        rows = random.Random(0).sample(rows, min(args.limit, len(rows)))
    print(f"Projecting {len(rows)} prompts from {EVALAWAREBENCH}/{args.subset}")

    stacked = torch.stack([probes[n] for n in names])
    projections = project(model, tokenizer, stacked, [r["prompt"] for r in rows], n_layers, args.batch_size)

    out_path = os.path.join(out_dir, f"projections_{args.subset}{'' if args.limit is None else f'_n{args.limit}'}.jsonl")
    with open(out_path, "w") as f:
        for row, proj in zip(rows, projections.tolist()):
            meta = {k: row[k] for k in ("task_id", "task_name", "valence", "factors_varied", "num_factors_varied", "config")}
            f.write(json.dumps({**meta, "projections": dict(zip(names, proj))}) + "\n")

    print(f"\nDone. {len(rows)} rows x {len(names)} variants x {n_layers} layers saved to {out_path}")


if __name__ == "__main__":
    main()
