#!/usr/bin/env python
"""
Which probes are good, and the data-mix scores of the good ones.

A probe is good when its AUROC (eval- vs deploy-expected cues, src/cues.py) is >= --bar on the validation set (val)
AND on the zero-tolerance set (zt), in every placement (system, prefix, user). ZT is a subset of val (25 of its 51
eval/deploy cues), so the two gates are nested, not independent: zt is the stricter check on the core cues, val
adds the other 26. Cue scores come from <main>/scores_cues.jsonl (score.py <main> <acts>/cues), averaged over the
stems of each cue x placement.

Held-out columns: the same probe (name) trained in each --holdout run, scored on the rows that run left out of
training (report.csv, split "heldout", subset "all"). Reported, not gated on.

Outputs, in <main>/:
  probe_report.csv     one row per probe: cue AUROCs, good, held-out AUROCs
  <mix>_good.jsonl     if <main>/scores_<mix>.jsonl exists (score.py <main> <acts>/<mix>): one line per mix row with
                       id, source (the mix's dataset), meta and "scores": {good probe: logit}

Usage:
    python report.py output/qwen3.8-27b/trained/all --holdout output/qwen3.8-27b/trained/hold-*
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.cues import PLACEMENTS, VAL, ZT


def cue_aurocs(path):
    """{probe: {"val:<placement>": auroc, "zt:<placement>": auroc}} from score.py's output on cue activations."""
    rows = pd.read_json(path, lines=True)
    meta = pd.json_normalize(rows["meta"].tolist())
    scores = pd.json_normalize(rows["scores"].tolist())
    cue = meta.groupby("g").first()
    per_cue = scores.groupby(meta["g"]).mean()  # one score per cue x placement: mean over its stems
    test = cue["expected"].isin(["eval", "deploy"])
    out = {}
    for probe in per_cue:
        out[probe] = {}
        for name, subset in (("val", test & (cue["set"] == VAL)), ("zt", test & (cue["set"] == ZT))):
            for placement in PLACEMENTS:
                m = subset & (cue["placement"] == placement)
                y = cue["expected"][m] == "eval"
                out[probe][f"{name}:{placement}"] = roc_auc_score(y, per_cue[probe][m]) if y.nunique() == 2 else np.nan
    return out


def heldout_auroc(run_dir):
    r = pd.read_csv(os.path.join(run_dir, "report.csv"))
    r = r[(r["split"] == "heldout") & (r["subset"] == "all")]
    return dict(zip(r["probe"], r["auroc"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("main", help="train_probes.py run whose probes score the mix (trained on everything)")
    parser.add_argument("--holdout", nargs="*", default=[], help="train_probes.py runs with held-out families")
    parser.add_argument("--bar", type=float, default=0.75)
    parser.add_argument("--cues", default="cues", help="Name of the cue activation directory scored by score.py")
    parser.add_argument("--mix", default="mix-main", help="Name of the mix activation directory scored by score.py")
    args = parser.parse_args()

    cue_scores = os.path.join(args.main, f"scores_{args.cues}.jsonl")
    if not os.path.exists(cue_scores):
        parser.error(f"{cue_scores} not found: run score.py {args.main} <acts>/{args.cues}")
    table = pd.DataFrame(cue_aurocs(cue_scores)).T
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

    out = os.path.join(args.main, "probe_report.csv")
    table.to_csv(out)
    with pd.option_context("display.width", 300, "display.max_columns", None):
        print(table.round(3).to_string())
    good = table.index[table["good"]].tolist()
    print(f"\nGood (val and ZT AUROC >= {args.bar} in every placement; ZT is a subset of val): {good or 'none'}"
          f"\nSaved {out}")

    mix_scores = os.path.join(args.main, f"scores_{args.mix}.jsonl")
    if not os.path.exists(mix_scores):
        print(f"No {mix_scores} yet: run score.py {args.main} <acts>/{args.mix} for the mix scores")
        return
    mix_out = os.path.join(args.main, f"{args.mix}_good.jsonl")
    n = 0
    with open(mix_scores) as f, open(mix_out, "w") as g:
        for line in f:
            r = json.loads(line)
            g.write(json.dumps({"id": r["id"], "source": r["source"], "meta": r["meta"],
                                "scores": {p: r["scores"][p] for p in good}}) + "\n")
            n += 1
    print(f"Saved {n} {args.mix} rows x {len(good)} good probes to {mix_out}")


if __name__ == "__main__":
    main()
