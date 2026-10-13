#!/usr/bin/env python
"""
Summarize validation.jsonl produced by validate.py.

Scores are normalized per placement and layer against the training system prompts (anchors):
0 = mean of anchor_deploy, 1 = mean of anchor_eval. A probe that tracks evaluation awareness should put
eval cues near 1, deploy cues near 0 and neutral cues ("chicken") near the no-cue baseline ("none"),
and should follow negations rather than keywords.

Layer (default): per variant, the layer where the anchors are furthest apart in the system placement,
relative to the spread of neutral words. Anchors are training prompts, so this does not peek at the test cues.

Per variant: AUROC separating eval-expected from deploy-expected cues (anchors excluded), per placement.
For one variant (--variant): mean score per category and placement, then per cue.

Usage:
    python scripts/analyze_validation.py output/qwen3.5-0.8b/validation.jsonl
    python scripts/analyze_validation.py runpod-artifacts/qwen3.8-27b/validation.jsonl --variant minimal --layer 40
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PLACEMENTS = ["system", "prefix", "user"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--variant", default=None, help="System-prompt variant (default: the first in the file)")
    parser.add_argument("--layer", type=int, default=None, help="Layer (default: best anchor separation)")
    args = parser.parse_args()

    df = pd.read_json(args.path, lines=True)
    variants = list(df["projections"].iloc[0])

    def proj(v):
        return np.stack(df["projections"].map(lambda p: p[v]).to_numpy())  # [n_rows, n_layers]

    def best_layer(p):
        system = df["placement"] == "system"
        gap = (p[(system & (df["category"] == "anchor_eval")).to_numpy()].mean(0)
               - p[(system & (df["category"] == "anchor_deploy")).to_numpy()].mean(0))
        spread = p[(system & (df["category"] == "neutral_word")).to_numpy()].std(0)
        return int(np.nanargmax(gap / np.where(spread > 0, spread, np.nan)))

    def normalized(p, layer):
        """Per placement: 0 = deploy anchors, 1 = eval anchors."""
        out = pd.Series(np.nan, index=df.index)
        for placement, g in df.groupby("placement"):
            x = p[g.index, layer]
            lo = x[(g["category"] == "anchor_deploy").to_numpy()].mean()
            hi = x[(g["category"] == "anchor_eval").to_numpy()].mean()
            out[g.index] = (x - lo) / (hi - lo)
        return out

    test = df["expected"].isin(["eval", "deploy"]) & ~df["category"].str.startswith("anchor")
    summary = {}
    for v in variants:
        p = proj(v)
        layer = args.layer if args.layer is not None else best_layer(p)
        summary[v] = {"layer": layer}
        for placement in PLACEMENTS:
            m = (test & (df["placement"] == placement)).to_numpy()
            summary[v][f"auroc_{placement}"] = roc_auc_score(df["expected"][m] == "eval", p[m, layer])
    print("AUROC, eval- vs deploy-expected cues (anchors excluded):\n")
    print(pd.DataFrame(summary).T.round(3).to_string())

    variant = args.variant or variants[0]
    layer = summary[variant]["layer"]
    df["score"] = normalized(proj(variant), layer)
    print(f"\nVariant '{variant}', layer {layer}. Score: 0 = deploy anchors, 1 = eval anchors.\n")

    by_category = df.pivot_table(index=["category", "expected"], columns="placement", values="score", sort=False)
    print(by_category[[c for c in PLACEMENTS if c in by_category]].round(2).to_string())

    print("\nPer cue:\n")
    df["cue"] = df["text"].str.slice(0, 60)
    per_cue = df.pivot_table(index=["category", "cue"], columns="placement", values="score", sort=False)
    print(per_cue[[c for c in PLACEMENTS if c in per_cue]].round(2).to_string())


if __name__ == "__main__":
    main()
