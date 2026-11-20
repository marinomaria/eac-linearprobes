#!/usr/bin/env python
"""
Which probes are good, and the data-mix scores of the good ones, across one or more probe runs (e.g. "all", trained
on the main training set, and "verbalized", trained on mix-discovery rows labeled by CoT voicing alone).

A probe is good when its AUROC (eval- vs deploy-expected cues, src/cues.py) is >= --bar on the validation set (val)
AND on the zero-tolerance set (zt), in every placement (system, prefix, user). ZT is a subset of val (25 of its 51
eval/deploy cues), so the two gates are nested, not independent: zt is the stricter check on the core cues, val
adds the other 26. Cue scores come from <run>/scores_cues.jsonl (score.py <run> <acts>/cues), averaged over the
stems of each cue x placement.

Held-out columns, one per training source (held:<source>; reported, not gated on): the probe's AUROC on that
source's rows when they were not used to train it.
  - Sources the run trained on: from the --holdout runs of the same recipe (same activation directories) that held
    the source out (report.csv, split "heldout", subset "source:<s>"); the minimum if several did.
  - Sources the run never saw: computed from its labeled scores files, <run>/scores_<acts>.jsonl
    (score.py <run> <labeled acts>), e.g. the verbalized run scored on the main training activations.
hold_min is the minimum over sources.

Outputs, in the first run's directory:
  probe_report.csv     one row per probe (<run>/<probe> with several runs): cue AUROCs, good, held-out AUROCs
  <mix>_good.jsonl     one line per mix row with id, source (the mix's dataset), meta and "scores": {good probe:
                       logit}, from each run's scores_<mix>.jsonl (score.py <run> <acts>/<mix>); written when every
                       run with a good probe has one

Usage:
    python report.py output/qwen3.8-27b/trained/all output/qwen3.8-27b/trained/verbalized \
        --holdout output/qwen3.8-27b/trained/hold-*
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.cues import PLACEMENTS, VAL, ZT
from src.rows import LABELS


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


def run_args(run):
    saved = torch.load(os.path.join(run, "probes.pt"), weights_only=False)
    acts = saved["args"]["acts"]
    acts = [acts] if isinstance(acts, str) else acts
    return {"acts": [os.path.normpath(os.path.abspath(a)) for a in acts], "train_sources": saved.get("train_sources")}


def heldout_by_source(run, holdouts):
    """{probe: {source: auroc}} on sources the probe's training did not include (see module docstring)."""
    info = run_args(run)
    out = {}

    def put(probe, source, auroc):
        if not np.isnan(auroc):
            cell = out.setdefault(probe, {})
            cell[source] = min(cell.get(source, np.inf), auroc)

    for h in holdouts:
        if os.path.samefile(h, run) or run_args(h)["acts"] != info["acts"]:
            continue
        r = pd.read_csv(os.path.join(h, "report.csv"))
        r = r[(r["split"] == "heldout") & r["subset"].str.startswith("source:")]
        for t in r.itertuples():
            put(t.probe, t.subset.removeprefix("source:"), t.auroc)

    for path in glob.glob(os.path.join(run, "scores_*.jsonl")):
        if info["train_sources"] is None:
            break
        rows = pd.read_json(path, lines=True)
        rows = rows[rows["label"].notna() & ~rows["source"].isin(info["train_sources"])]
        for source, g in rows.groupby("source"):
            y = g["label"].map(LABELS)
            if y.nunique() < 2:
                continue
            scores = pd.json_normalize(g["scores"].tolist())
            for probe in scores:
                put(probe, source, roc_auc_score(y, scores[probe]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="train_probes.py runs whose probes may score the mix")
    parser.add_argument("--holdout", nargs="*", default=[], help="train_probes.py runs with held-out families")
    parser.add_argument("--bar", type=float, default=0.75)
    parser.add_argument("--cues", default="cues", help="Name of the cue activation directory scored by score.py")
    parser.add_argument("--mix", default="mix-main", help="Name of the mix activation directory scored by score.py")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument(
        "--probes",
        nargs="+",
        default=None,
        help="Write the mix scores of these probes (names as in probe_report.csv, e.g. all/linear_L42) to "
        "<mix>_selected.jsonl instead of the good ones",
    )
    choice.add_argument(
        "--top",
        type=int,
        default=None,
        help="Write the mix scores of the N best probes (report order: good, then zt_min) to <mix>_top<N>.jsonl "
        "instead of the good ones",
    )
    args = parser.parse_args()
    holdouts = [h for h in args.holdout if os.path.exists(os.path.join(h, "report.csv"))]

    tables, owner = [], {}  # owner: row name -> (run, probe)
    for run in args.runs:
        cue_scores = os.path.join(run, f"scores_{args.cues}.jsonl")
        if not os.path.exists(cue_scores):
            parser.error(f"{cue_scores} not found: run score.py {run} <acts>/{args.cues}")
        t = pd.DataFrame(cue_aurocs(cue_scores)).T
        held = pd.DataFrame(heldout_by_source(run, holdouts)).T.add_prefix("held:")
        t = t.join(held)
        name = os.path.basename(os.path.normpath(run))
        if len(args.runs) > 1:
            t.index = [f"{name}/{p}" for p in t.index]
        owner.update({i: (run, i.split("/")[-1]) for i in t.index})
        tables.append(t)
    table = pd.concat(tables)
    table["val_min"] = table[[f"val:{p}" for p in PLACEMENTS]].min(axis=1)
    table["zt_min"] = table[[f"zt:{p}" for p in PLACEMENTS]].min(axis=1)
    table["good"] = (table["val_min"] >= args.bar) & (table["zt_min"] >= args.bar)
    held = sorted(c for c in table if c.startswith("held:"))
    if held:
        table["hold_min"] = table[held].min(axis=1)
    cues = [f"{s}:{p}" for s in ("val", "zt") for p in PLACEMENTS]
    table = table[cues + ["val_min", "zt_min", "good"] + held + (["hold_min"] if held else [])]
    table = table.sort_values(["good", "zt_min"], ascending=False)

    out = os.path.join(args.runs[0], "probe_report.csv")
    table.to_csv(out)
    with pd.option_context("display.width", 300, "display.max_columns", None):
        print(table.round(3).to_string())
    good = table.index[table["good"]].tolist()
    print(
        f"\nGood (val and ZT AUROC >= {args.bar} in every placement; ZT is a subset of val): {good or 'none'}"
        f"\nSaved {out}"
    )

    if args.probes:
        unknown = [p for p in args.probes if p not in table.index]
        if unknown:
            parser.error(f"--probes not in the report: {unknown}")
        chosen, kind = args.probes, "selected"
    elif args.top:
        chosen, kind = table.index[: args.top].tolist(), f"top{args.top}"
    else:
        chosen, kind = good, "good"
    by_run = {}
    for g in chosen:
        run, probe = owner[g]
        by_run.setdefault(run, []).append((g, probe))
    missing = [
        os.path.join(r, f"scores_{args.mix}.jsonl")
        for r in by_run
        if not os.path.exists(os.path.join(r, f"scores_{args.mix}.jsonl"))
    ]
    if not chosen or missing:
        print(f"No {args.mix} scores written" + (f": run score.py for {missing}" if missing else ""))
        return
    merged = {}
    for run, probes in by_run.items():
        with open(os.path.join(run, f"scores_{args.mix}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                row = merged.setdefault(
                    r["id"], {"id": r["id"], "source": r["source"], "meta": r["meta"], "scores": {}}
                )
                row["scores"].update({g: r["scores"][p] for g, p in probes})
    mix_out = os.path.join(args.runs[0], f"{args.mix}_{kind}.jsonl")
    with open(mix_out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in merged.values())
    print(f"Saved {len(merged)} {args.mix} rows x {len(chosen)} {kind} probes to {mix_out}: {chosen}")


if __name__ == "__main__":
    main()
