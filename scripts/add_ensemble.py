#!/usr/bin/env python
"""
Add linear-ensemble scores to a projections file made with a train_probes.py export.pt
(run.py / run_mix.py with --load-probes, or validate.py with --probes).

The ensemble logit is mean over kept layers l of (projection_l + offset_l), from ensemble.json. It is added as
a new variant "<variant>_ensemble" whose per-layer list repeats the ensemble score at every layer, so the
existing analysis scripts work on it unchanged (whatever layer they pick, they read the ensemble).

Usage:
    python scripts/add_ensemble.py output/qwen3.8-27b/validation.jsonl output/qwen3.8-27b/trained/v1/ensemble.json
    # -> output/qwen3.8-27b/validation_ens.jsonl
"""
import argparse
import json
import os

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("projections")
    parser.add_argument("ensemble", help="ensemble.json written by train_probes.py")
    parser.add_argument("--out", default=None, help="Default: <projections stem>_ens.jsonl")
    args = parser.parse_args()

    with open(args.ensemble) as f:
        ens = json.load(f)
    layers = ens["layers"]
    offsets = np.array([ens["offsets"][str(layer)] for layer in layers])
    out = args.out or os.path.splitext(args.projections)[0] + "_ens.jsonl"
    with open(args.projections) as f, open(out, "w") as g:
        for line in f:
            row = json.loads(line)
            p = np.asarray(row["projections"][ens["variant"]])
            score = float((p[layers] + offsets).mean())
            row["projections"][f"{ens['variant']}_ensemble"] = [score] * len(p)
            g.write(json.dumps(row) + "\n")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
