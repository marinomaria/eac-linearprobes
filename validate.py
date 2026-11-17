#!/usr/bin/env python
"""
Sanity-check the probes on short hand-written cues: datasets/validation_prompts.json and the zero-tolerance set
datasets/validation_prompts_zt.csv (--prompts; each row records its file as "set").

Each cue ("evaluation", "chicken", "This is not a test.", ...) is shown to the model in three placements:
  - system: the cue is the system prompt, followed by a contrastive-dataset stem as the user turn
            (the same shape the probes were trained on); averaged over --n-stems stems
  - prefix: the cue is prepended to a stem in a single user turn (the shape of EvalAwareBench cues);
            averaged over --n-stems stems
  - user:   the cue alone is the user turn
The training system prompts of every variant are added as anchors (categories anchor_eval /
anchor_deploy), so scripts/analyze_validation.py can put every cue on a 0 (deploy) .. 1 (eval) scale.

Probes:
  --probes   a [n_layers, d_model] direction file (run.py's probes.pt, or train_probes.py's export.pt);
             default output/<model>/probes.pt. Output: output/<model>/validation.jsonl (or --out).
  --trained  one or more train_probes.py directories: every probe in them (linear_L*, mlp_L*,
             linear_ensemble, mlp_ensemble) scores the cues, from one extraction at the union of their layers.
             Output: <trained dir>/validation.jsonl each, with one logit per probe ("projections": {probe: [logit]}).
  Both can be given in one call (one model load); --trained alone skips the direction probes.

Usage:
    python validate.py --model qwen3.5-0.8b --n-stems 4     # quick local test
    python validate.py --model qwen3.8-27b --batch-size 16
    python validate.py --model qwen3.8-27b --probes output/qwen3.8-27b/probes.pt \
        --trained output/qwen3.8-27b/trained/all output/qwen3.8-27b/trained/hold-agent
"""
import argparse
import csv
import json
import os
import random

import torch

from run import CONTRASTIVE_DATASET, OUTPUT_DIR, SYSTEM_PROMPTS
from src.activations import last_token_layers
from src.classifiers import load_trained, score_all
from src.model import format_chat, get_device, load_model
from src.probe import project_conversations

VALIDATION_PROMPTS = ["datasets/validation_prompts.json", "datasets/validation_prompts_zt.csv"]


def read_prompts(path):
    """[{category, expected, text}] from a .json list or a .csv with those columns, tagged with set=<file stem>."""
    with open(path) as f:
        items = list(csv.DictReader(f)) if path.endswith(".csv") else json.load(f)
    name = os.path.splitext(os.path.basename(path))[0]
    return [{"category": i["category"], "expected": i["expected"], "text": i["text"], "set": name} for i in items]


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
    parser.add_argument("--probes", default=None,
                        help="Direction file, run.py's probes.pt or an export.pt (default: output/<model>/probes.pt)")
    parser.add_argument("--trained", nargs="*", default=[], help="train_probes.py output directories")
    parser.add_argument("--prompts", nargs="+", default=VALIDATION_PROMPTS, help="Cue files (.json or .csv)")
    parser.add_argument("--out", default=None, help="Output for --probes (default: output/<model>/validation.jsonl)")
    parser.add_argument("--n-stems", type=int, default=20, help="Stems averaged over in system/prefix placements")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    out_dir = os.path.join(OUTPUT_DIR, args.model.replace("/", "_"))
    probes = None
    if args.probes or not args.trained:
        probes_path = args.probes or os.path.join(out_dir, "probes.pt")
        if not os.path.exists(probes_path):
            parser.error(f"{probes_path} not found: run run.py first, or pass --probes / --trained")
        probes = torch.load(probes_path)  # {variant_name: [n_layers, d_model]}
    trained = {d: load_trained(d) for d in args.trained}
    for d, (saved, _) in trained.items():
        if saved["info"]["model"] != args.model:
            parser.error(f"{d} was trained on {saved['info']['model']} activations, not {args.model}")

    items = [i for path in args.prompts for i in read_prompts(path)]
    with open(SYSTEM_PROMPTS) as f:
        for v in json.load(f):
            items.append({"category": "anchor_eval", "expected": "eval", "text": v["eval"], "set": "anchor"})
            items.append({"category": "anchor_deploy", "expected": "deploy", "text": v["deploy"], "set": "anchor"})
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
    print(f"{len(items)} cues x placements = {len(convs)} conversations")

    def write(path, per_row):
        """per_row(slice) -> {probe: list}, averaged over the stems of one (cue, placement)."""
        with open(path, "w") as f:
            for item, placement, sl in groups:
                f.write(json.dumps({**item, "placement": placement, "n": sl.stop - sl.start,
                                    "projections": per_row(sl)}) + "\n")
        print(f"{len(groups)} rows saved to {path}")

    if probes is not None:
        names = list(probes)
        stacked = torch.stack([probes[n] for n in names])
        proj = project_conversations(model, tokenizer, stacked, convs, n_layers, args.batch_size)
        write(args.out or os.path.join(out_dir, "validation.jsonl"),
              lambda sl: dict(zip(names, proj[sl].mean(dim=0).tolist(), strict=True)))  # [n_variants, n_layers]

    if trained:
        layers = sorted({layer for saved, _ in trained.values() for layer in saved["info"]["layers"]})
        acts = last_token_layers(model, tokenizer, [format_chat(tokenizer, c) for c in convs], layers, n_layers,
                                 args.batch_size)
        for d, (saved, ps) in trained.items():
            scores = score_all(saved, ps, lambda layer: acts[:, layers.index(layer)].float(), get_device())
            write(os.path.join(d, "validation.jsonl"),
                  lambda sl, scores=scores: {k: [float(v[sl].mean())] for k, v in scores.items()})


if __name__ == "__main__":
    main()
