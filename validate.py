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
  --acts     instead of running the model: cue activations stored once with `extract.py --data cues` (same
             cues, placements and stems as here), so any later --trained probes are scored offline in seconds.

Usage:
    python validate.py --model qwen3.5-0.8b --n-stems 4     # quick local test
    python validate.py --model qwen3.8-27b --batch-size 16
    python validate.py --model qwen3.8-27b --probes output/qwen3.8-27b/probes.pt \
        --trained output/qwen3.8-27b/trained/all output/qwen3.8-27b/trained/hold-agent
    python extract.py --model qwen3.8-27b --data cues                  # once, on a GPU
    python validate.py --acts output/qwen3.8-27b/acts/cues --trained output/qwen3.8-27b/trained/all   # anywhere
"""
import argparse
import csv
import json
import os
import random

import torch

from run import CONTRASTIVE_DATASET, OUTPUT_DIR, SYSTEM_PROMPTS
from src.activations import last_token_layers, layer_tensor
from src.activations import load as load_acts
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


def cue_groups(prompts=VALIDATION_PROMPTS, n_stems=20):
    """Every (cue, placement) as a group of conversations (one per stem; one for "user"), flattened so a single
    batched pass covers all of them. Returns (groups, convs), groups[i] = (item, placement, indices into convs)."""
    items = [i for path in prompts for i in read_prompts(path)]
    with open(SYSTEM_PROMPTS) as f:
        for v in json.load(f):
            items.append({"category": "anchor_eval", "expected": "eval", "text": v["eval"], "set": "anchor"})
            items.append({"category": "anchor_deploy", "expected": "deploy", "text": v["deploy"], "set": "anchor"})
    with open(CONTRASTIVE_DATASET) as f:
        stems = [row["stem"].strip() for row in json.load(f)]
    stems = random.Random(0).sample(stems, min(n_stems, len(stems)))
    convs, groups = [], []
    for item in items:
        for placement in ("system", "prefix", "user"):
            c = conversations(item["text"], placement, stems)
            if c:
                groups.append((item, placement, list(range(len(convs), len(convs) + len(c)))))
                convs += c
    return groups, convs


def cue_rows(n_stems=20, prompts=VALIDATION_PROMPTS):
    """The cue conversations as unlabeled rows (src/rows.py), for extract.py --data cues: their activations are
    stored once and any trained probe is scored on them offline (validate.py --acts). meta.g indexes the
    (cue, placement) group whose stems are averaged."""
    groups, convs = cue_groups(prompts, n_stems)
    rows = []
    for g, (item, placement, sl) in enumerate(groups):
        for k, i in enumerate(sl):
            rows.append({"id": f"cue:{g}:{k}", "messages": convs[i], "label": None, "group": f"cue:{g}",
                         "family": f"cues:{item['set']}", "source": "cues",
                         "meta": {**item, "placement": placement, "g": g}})
    return rows


def write_validation(path, groups, per_group):
    """per_group(indices) -> {probe: list}, averaged over the stems of one (cue, placement); one line each."""
    with open(path, "w") as f:
        for item, placement, idx in groups:
            f.write(json.dumps({**item, "placement": placement, "n": len(idx), "projections": per_group(idx)}) + "\n")
    print(f"{len(groups)} rows saved to {path}")


def from_acts(parser, args):
    """--acts: score --trained probes on stored cue activations, without loading the model."""
    acts, rows, info = load_acts(args.acts)
    if not rows or rows[0]["source"] != "cues":
        parser.error(f"{args.acts} does not hold cue activations (extract.py --data cues)")
    if args.probes:
        parser.error("--acts scores --trained probes only (direction probes need every layer)")
    by_g = {}
    for i, r in enumerate(rows):
        by_g.setdefault(r["meta"]["g"], []).append(i)
    groups = []
    for g in sorted(by_g):
        meta = dict(rows[by_g[g][0]]["meta"])
        placement = meta.pop("placement")
        meta.pop("g")
        groups.append((meta, placement, by_g[g]))
    for d in args.trained:
        saved, ps = load_trained(d)
        layers = saved["info"]["layers"]
        if saved["info"]["model"] != info["model"] or not set(layers) <= set(info["layers"]):
            parser.error(f"{d} ({saved['info']['model']}, layers {layers}) does not match the activations "
                         f"({info['model']}, layers {info['layers']})")
        scores = score_all(saved, ps, lambda layer: layer_tensor(acts, info["layers"].index(layer)), get_device())
        write_validation(os.path.join(d, "validation.jsonl"), groups,
                         lambda idx, scores=scores: {k: [float(v[idx].mean())] for k, v in scores.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="Alias from src/model.py:MODELS or HF repo id")
    parser.add_argument("--acts", default=None,
                        help="Cue activations from extract.py --data cues: score --trained offline, no model")
    parser.add_argument("--probes", default=None,
                        help="Direction file, run.py's probes.pt or an export.pt (default: output/<model>/probes.pt)")
    parser.add_argument("--trained", nargs="*", default=[], help="train_probes.py output directories")
    parser.add_argument("--prompts", nargs="+", default=VALIDATION_PROMPTS, help="Cue files (.json or .csv)")
    parser.add_argument("--out", default=None, help="Output for --probes (default: output/<model>/validation.jsonl)")
    parser.add_argument("--n-stems", type=int, default=20, help="Stems averaged over in system/prefix placements")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.acts:
        if not args.trained:
            parser.error("--acts needs --trained")
        return from_acts(parser, args)
    if not args.model:
        parser.error("--model is required (unless --acts)")

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

    groups, convs = cue_groups(args.prompts, args.n_stems)

    model, tokenizer, n_layers = load_model(args.model)
    print(f"{len(groups)} cues x placements = {len(convs)} conversations")

    if probes is not None:
        names = list(probes)
        stacked = torch.stack([probes[n] for n in names])
        proj = project_conversations(model, tokenizer, stacked, convs, n_layers, args.batch_size)
        write_validation(args.out or os.path.join(out_dir, "validation.jsonl"), groups,  # [n_variants, n_layers]
                         lambda idx: dict(zip(names, proj[idx].mean(dim=0).tolist(), strict=True)))

    if trained:
        layers = sorted({layer for saved, _ in trained.values() for layer in saved["info"]["layers"]})
        acts = last_token_layers(model, tokenizer, [format_chat(tokenizer, c) for c in convs], layers, n_layers,
                                 args.batch_size)
        for d, (saved, ps) in trained.items():
            scores = score_all(saved, ps, lambda layer: acts[:, layers.index(layer)].float(), get_device())
            write_validation(os.path.join(d, "validation.jsonl"), groups,
                             lambda idx, scores=scores: {k: [float(v[idx].mean())] for k, v in scores.items()})


if __name__ == "__main__":
    main()
