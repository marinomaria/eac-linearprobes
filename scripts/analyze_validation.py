#!/usr/bin/env python
"""
Summarize validation.jsonl produced by validate.py.

Scores are normalized per placement and layer against the training system prompts (anchors):
0 = mean of anchor_deploy, 1 = mean of anchor_eval. A probe that tracks evaluation awareness should put
eval cues near 1, deploy cues near 0 and neutral cues ("chicken") near the no-cue baseline ("none"),
and should follow negations rather than keywords.

Layer (default): per variant, the layer where the anchors are furthest apart in the system placement,
relative to the spread of neutral words. Anchors are training prompts, so this does not peek at the test cues.

Per variant: AUROC separating eval-expected from deploy-expected cues (anchors excluded), per placement,
over all cue sets and per set (validation_prompts, validation_prompts_zt). --set keeps one set (plus anchors).
For one variant (--variant): mean score per category and placement, then per cue.

Files written by validate.py --trained hold one logit per probe (a one-element list), so "layer" is always 0
and variants are the probes (linear_ensemble, mlp_ensemble, linear_L<l>, ...).

Usage:
    python scripts/analyze_validation.py output/qwen3.5-0.8b/validation.jsonl
    python scripts/analyze_validation.py runpod-artifacts/qwen3.8-27b/validation.jsonl --variant minimal --layer 40
    python scripts/analyze_validation.py output/qwen3.8-27b/trained/all/validation.jsonl --variant mlp_ensemble \
        --set validation_prompts_zt
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
    parser.add_argument("--set", default=None, help="Only this cue set (file stem), e.g. validation_prompts_zt")
    args = parser.parse_args()

    df = pd.read_json(args.path, lines=True)
    if "set" not in df:
        df["set"] = "validation_prompts"
    df.loc[df["category"].str.startswith("anchor"), "set"] = "anchor"
    if args.set:
        df = df[df["set"].isin([args.set, "anchor"]) | (df["category"] == "none")].reset_index(drop=True)
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
    sets = sorted(set(df.loc[test, "set"]))
    summary = {}
    for v in variants:
        p = proj(v)
        layer = args.layer if args.layer is not None else best_layer(p)
        summary[v] = {"layer": layer}
        for subset in ["all"] + (sets if len(sets) > 1 else []):
            in_subset = test if subset == "all" else test & (df["set"] == subset)
            for placement in PLACEMENTS:
                m = (in_subset & (df["placement"] == placement)).to_numpy()
                key = f"auroc_{placement}" if subset == "all" else f"{subset}:{placement}"
                y = df["expected"][m] == "eval"
                summary[v][key] = roc_auc_score(y, p[m, layer]) if y.nunique() == 2 else np.nan
    print(f"AUROC, eval- vs deploy-expected cues (anchors excluded); sets: {sets}\n")
    with pd.option_context("display.width", 250):
        print(pd.DataFrame(summary).T.round(3).to_string())

    variant = args.variant or variants[0]
    layer = summary[variant]["layer"]
    df["score"] = normalized(proj(variant), layer)
    print(f"\nVariant '{variant}', layer {layer}. Score: 0 = deploy anchors, 1 = eval anchors.\n")

    by_category = df.pivot_table(index=["set", "category", "expected"], columns="placement", values="score", sort=False)
    print(by_category[[c for c in PLACEMENTS if c in by_category]].round(2).to_string())

    print("\nPer cue:\n")
    df["cue"] = df["text"].str.slice(0, 60)
    per_cue = df.pivot_table(index=["set", "category", "cue"], columns="placement", values="score", sort=False)
    print(per_cue[[c for c in PLACEMENTS if c in per_cue]].round(2).to_string())


if __name__ == "__main__":
    main()
