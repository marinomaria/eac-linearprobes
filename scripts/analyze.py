#!/usr/bin/env python
"""
Summarize a projections_*.jsonl file produced by run.py.

Per system-prompt variant: best layer and its Spearman correlation (see below).

For one variant (--variant, default: the first), per layer:
  - AUROC separating all-factors-exaggerated prompts (8) from baselines (0)
  - Spearman correlation between projection and number of exaggerated factors

At the best layer (by Spearman, which uses every prompt; 0/8-factor prompts are only
2/256 of the data, so the AUROC is noisy on small --limit runs), per factor and valence:
  - Main effect: mean projection with the factor exaggerated minus naturalistic,
    in units of the layer's projection std

Usage:
    python scripts/analyze.py output/qwen3.5-0.8b/projections_prompts.jsonl
    python scripts/analyze.py output/qwen3.5-0.8b/projections_prompts.jsonl --variant minimal
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FACTORS = [f"F{i}" for i in range(1, 9)]
NATURALISTIC = {"realistic", "casual", "prose"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--variant", default=None, help="System-prompt variant (default: the first in the file)")
    parser.add_argument("--layer", type=int, default=None, help="Layer for factor effects (default: best Spearman)")
    args = parser.parse_args()

    df = pd.read_json(args.path, lines=True)
    n_factors = df["num_factors_varied"].to_numpy()
    variants = list(df["projections"].iloc[0])

    def spearman_by_layer(proj):
        return pd.Series([pd.Series(proj[:, layer]).corr(pd.Series(n_factors), method="spearman")
                          for layer in range(proj.shape[1])])

    summary = {}
    for v in variants:
        rho = spearman_by_layer(np.stack(df["projections"].map(lambda p, v=v: p[v]).to_numpy()))
        summary[v] = {"best_layer": int(rho.idxmax()), "spearman": rho.max()}
    print("Variants (best layer by Spearman with number of exaggerated factors):\n")
    print(pd.DataFrame(summary).T.round(3).to_string())

    variant = args.variant or variants[0]
    proj = np.stack(df["projections"].map(lambda p: p[variant]).to_numpy())  # [n_prompts, n_layers]
    extremes = (n_factors == 0) | (n_factors == 8)

    per_layer = pd.DataFrame({
        "auroc_0_vs_8": [roc_auc_score(n_factors[extremes] == 8, proj[extremes, layer])
                         if len(set(n_factors[extremes])) == 2 else np.nan
                         for layer in range(proj.shape[1])],
        "spearman_n_factors": spearman_by_layer(proj),
    })
    per_layer.index.name = "layer"
    print(f"\nVariant '{variant}': {len(df)} prompts, {proj.shape[1]} layers, "
          f"{extremes.sum()} with 0 or 8 factors\n")
    print(per_layer.round(3).to_string())

    layer = args.layer if args.layer is not None else int(per_layer["spearman_n_factors"].idxmax())
    z = (proj[:, layer] - proj[:, layer].mean()) / proj[:, layer].std()
    effects = {}
    for valence, group in df.groupby("valence"):
        zs = z[group.index]
        effects[valence] = {}
        for f in FACTORS:
            on = ~group["config"].map(lambda c, f=f: c[f] in NATURALISTIC).to_numpy()
            effects[valence][f] = zs[on].mean() - zs[~on].mean()
    print(f"\nFactor main effects at layer {layer} (std units, exaggerated - naturalistic):\n")
    print(pd.DataFrame(effects).round(3).to_string())


if __name__ == "__main__":
    main()
