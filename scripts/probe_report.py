#!/usr/bin/env python
"""
One table saying which learned probes are good.

A probe is good when its AUROC (eval- vs deploy-expected cues) is >= --bar on the validation set (all cue sets
pooled, ZT included) AND on the ZT set alone, in every placement (system, prefix, user). These come from
<main>/validation.jsonl, the run whose probes would score the mix.

Held-out columns: the same probe recipe (name) trained in each --holdout run, scored on the rows that run left
out of training (report.csv, split "heldout", subset "all"). These are reported, not gated on.

Usage:
    python scripts/probe_report.py runpod-artifacts/learned-20261118-full/qwen3.8-27b/trained/all \
        --holdout runpod-artifacts/learned-20261118-full/qwen3.8-27b/trained/hold-*
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PLACEMENTS = ["system", "prefix", "user"]
ZT = "validation_prompts_zt"


def cue_aurocs(path):
    """{probe: {"val:<placement>": auroc, "zt:<placement>": auroc}} from a validate.py --trained output."""
    df = pd.read_json(path, lines=True)
    test = df["expected"].isin(["eval", "deploy"]) & ~df["category"].str.startswith("anchor")
    out = {}
    for probe in df["projections"].iloc[0]:
        s = df["projections"].map(lambda p, k=probe: p[k][0])
        out[probe] = {}
        for name, subset in (("val", test), ("zt", test & (df["set"] == ZT))):
            for placement in PLACEMENTS:
                m = subset & (df["placement"] == placement)
                y = df["expected"][m] == "eval"
                out[probe][f"{name}:{placement}"] = roc_auc_score(y, s[m]) if y.nunique() == 2 else np.nan
    return out


def heldout_auroc(run_dir):
    r = pd.read_csv(os.path.join(run_dir, "report.csv"))
    r = r[(r["split"] == "heldout") & (r["subset"] == "all")]
    return dict(zip(r["probe"], r["auroc"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("main", help="train_probes.py run (with validation.jsonl) whose probes would score the mix")
    parser.add_argument("--holdout", nargs="*", default=[], help="train_probes.py runs with held-out rows")
    parser.add_argument("--bar", type=float, default=0.75)
    parser.add_argument("--out", default=None, help="CSV output (default: <main>/probe_report.csv)")
    args = parser.parse_args()

    table = pd.DataFrame(cue_aurocs(os.path.join(args.main, "validation.jsonl"))).T
    val = [f"val:{p}" for p in PLACEMENTS]
    zt = [f"zt:{p}" for p in PLACEMENTS]
    table["val_min"] = table[val].min(axis=1)
    table["zt_min"] = table[zt].min(axis=1)
    table["good"] = (table["val_min"] >= args.bar) & (table["zt_min"] >= args.bar)
    for run in sorted(args.holdout):
        if os.path.exists(os.path.join(run, "report.csv")):
            table[os.path.basename(os.path.normpath(run))] = pd.Series(heldout_auroc(run))
    table = table.dropna(axis=1, how="all")  # e.g. a deploy-only held-out source has no AUROC
    hold = [c for c in table if c.startswith("hold")]
    if hold:
        table["hold_min"] = table[hold].min(axis=1)
    table = table.sort_values(["good", "zt_min"], ascending=False)

    out = args.out or os.path.join(args.main, "probe_report.csv")
    table.to_csv(out)
    with pd.option_context("display.width", 300, "display.max_columns", None):
        print(table.round(3).to_string())
    good = table.index[table["good"]].tolist()
    print(f"\nGood (val and ZT AUROC >= {args.bar} in every placement): {good or 'none'}\nSaved {out}")


if __name__ == "__main__":
    main()
