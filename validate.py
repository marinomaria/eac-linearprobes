#!/usr/bin/env python
"""
Sanity-check the probes on short hand-written cues (datasets/validation_prompts.json).

Each cue ("evaluation", "chicken", "This is not a test.", ...) is shown to the model in three placements:
  - system: the cue is the system prompt, followed by a contrastive-dataset stem as the user turn
            (the same shape the probes were trained on); averaged over --n-stems stems
  - prefix: the cue is prepended to a stem in a single user turn (the shape of EvalAwareBench cues);
            averaged over --n-stems stems
  - user:   the cue alone is the user turn
The training system prompts of every variant are added as anchors (categories anchor_eval /
anchor_deploy), so scripts/analyze_validation.py can put every cue on a 0 (deploy) .. 1 (eval) scale.

Usage:
    python validate.py --model qwen3.5-0.8b --n-stems 4     # quick local test
    python validate.py --model qwen3.8-27b --batch-size 16
"""
import argparse
import json
import os
import random

import torch

from run import CONTRASTIVE_DATASET, OUTPUT_DIR, SYSTEM_PROMPTS
from src.model import load_model
from src.probe import project_conversations

VALIDATION_PROMPTS = "datasets/validation_prompts.json"


def conversations(text, placement, stems):
    """Message lists for one cue in one placement. An empty cue is the no-cue baseline."""
    if placement == "system":
        system = [{"role": "system", "content": text}] if text else []
        return [system + [{"role": "user", "content": s}] for s in stems]
    if placement == "prefix":
        return [[{"role": "user", "content": f"{text}\n\n{s}" if text else s}] for s in stems]
    return [[{"role": "user", "content": text}]] if text else []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Alias from src/model.py:MODELS or HF repo id")
    parser.add_argument("--probes", default=None, help="Path to probes.pt (default: output/<model>/probes.pt)")
    parser.add_argument("--n-stems", type=int, default=20, help="Stems averaged over in system/prefix placements")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    out_dir = os.path.join(OUTPUT_DIR, args.model.replace("/", "_"))
    probes_path = args.probes or os.path.join(out_dir, "probes.pt")
    if not os.path.exists(probes_path):
        parser.error(f"{probes_path} not found: run run.py first, or pass --probes")
    probes = torch.load(probes_path)  # {variant_name: [n_layers, d_model]}
    names = list(probes)

    with open(VALIDATION_PROMPTS) as f:
        items = json.load(f)
    with open(SYSTEM_PROMPTS) as f:
        for v in json.load(f):
            items.append({"category": "anchor_eval", "expected": "eval", "text": v["eval"]})
            items.append({"category": "anchor_deploy", "expected": "deploy", "text": v["deploy"]})
    with open(CONTRASTIVE_DATASET) as f:
        stems = [row["stem"].strip() for row in json.load(f)]
    stems = random.Random(0).sample(stems, min(args.n_stems, len(stems)))

    # Flatten every (item, placement, stem) into one list so a single batched pass covers all of them.
    convs, groups = [], []  # groups[i] = (item, placement, slice into convs)
    for item in items:
        for placement in ("system", "prefix", "user"):
            c = conversations(item["text"], placement, stems)
            if c:
                groups.append((item, placement, slice(len(convs), len(convs) + len(c))))
                convs += c

    model, tokenizer, n_layers = load_model(args.model)
    print(f"Projecting {len(items)} cues x placements = {len(convs)} conversations")
    stacked = torch.stack([probes[n] for n in names])
    proj = project_conversations(model, tokenizer, stacked, convs, n_layers, args.batch_size)

    out_path = os.path.join(out_dir, "validation.jsonl")
    with open(out_path, "w") as f:
        for item, placement, sl in groups:
            mean = proj[sl].mean(dim=0)  # [n_variants, n_layers]
            f.write(json.dumps({**item, "placement": placement, "n": sl.stop - sl.start,
                                "projections": dict(zip(names, mean.tolist()))}) + "\n")
    print(f"\nDone. {len(groups)} rows saved to {out_path}")


if __name__ == "__main__":
    main()
