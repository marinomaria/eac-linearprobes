#!/usr/bin/env python
"""
Train learned probes on activations stored by extract.py, after the FIBS paper's LinearEnsemble and
MLPEnsemble (see src/classifiers.py):
  - per stored layer: a logistic probe and an MLP probe (BCE, AdamW lr 5e-3, batch 32)
  - linear_ensemble / mlp_ensemble: mean logit over the layers whose train loss is below 0.6

Splits are by group (src/rows.py), so the eval and deploy versions of a stem never straddle train and val.
--holdout-family removes whole families from training and reports on them separately (leave-family-out);
shell-style patterns work, e.g. 'needham:agent:*'.
--balance (default "source") weights the loss so every (source, label) cell counts the same, e.g. Dolly's 1224
short paired rows vs Needham's chat and agentic transcripts.
length_baseline: logistic regression on log(prompt tokens) alone, trained on the same rows and weights. A probe that
does not clearly beat it on a split may be reading prompt length rather than evaluation cues.
Reported overall, per source and per family (families need both labels).
Metrics: AUROC and recall at 1% FPR (threshold set on the same split's negatives), with 95% CIs from a
bootstrap over groups.

Outputs, in output/<model>/trained/<name>/:
  report.csv     every probe x split (x family)
  probes.pt      all trained probes, for score.py
  export.pt      {"logreg": [n_layers, d_model]} raw-activation directions of the logistic probes, zeros
                 for layers not trained: a drop-in for --load-probes (run.py, run_mix.py) and --probes
                 (validate.py), so the existing analysis and ZT checks work on them
  ensemble.json  layers and offsets that turn those projections into linear_ensemble scores
                 (scripts/add_ensemble.py)

Usage:
    python train_probes.py output/qwen3.5-0.8b/acts/dolly_n40
    python train_probes.py output/qwen3.8-27b/acts/train_rows --holdout-family real_chat --name v1
"""
import argparse
import fnmatch
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from src.activations import layer_tensor, load
from src.classifiers import balance_weights, kept_layers, metrics, train_layer_probe
from src.model import get_device
from src.rows import LABELS, in_val


@torch.no_grad()
def logits(probe, x, device, chunk=4096):
    probe = probe.to(device)
    out = torch.cat([probe(x[i:i + chunk].to(device)).cpu() for i in range(0, len(x), chunk)])
    probe.cpu()
    return out.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("acts", help="Activation directory written by extract.py")
    parser.add_argument("--name", default=None, help="Output name (default: the activation directory's name)")
    parser.add_argument("--holdout-family", nargs="*", default=[],
                        help="Families excluded from training (shell-style patterns allowed)")
    parser.add_argument("--balance", choices=["source", "label", "none"], default="source",
                        help="Loss weights: equal total per (source, label), per label, or none")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Default: at least 3 epochs and 2000 optimizer steps")
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    acts, rows, info = load(args.acts)
    df = pd.DataFrame(rows)
    labeled = df["label"].notna().to_numpy()
    if "source" not in df:
        df["source"] = df["family"].str.split(":").str[0]
    families = sorted(df["family"].unique())
    unknown = [p for p in args.holdout_family if not fnmatch.filter(families, p)]
    if unknown:
        parser.error(f"--holdout-family {unknown} match no family; available: {families}")
    held_fams = {f for p in args.holdout_family for f in fnmatch.filter(families, p)}
    heldout = labeled & df["family"].isin(held_fams).to_numpy()
    val = labeled & ~heldout & df["group"].map(lambda g: in_val(g, args.val_frac, args.seed)).to_numpy()
    train = labeled & ~heldout & ~val
    y = df["label"].map(LABELS).to_numpy(dtype=float)
    print(f"train {train.sum()} (eval {int(y[train].sum())}), val {val.sum()}, heldout {heldout.sum()}; "
          f"layers {info['layers']}")
    print(df[labeled].assign(split=np.select([train[labeled], val[labeled]], ["train", "val"], "heldout"))
          .groupby(["split", "source", "label"]).size().unstack(fill_value=0).to_string())
    for split, m in (("train", train), ("val", val)):
        if len(set(y[m])) < 2:
            parser.error(f"{split} split has a single class; check labels / --val-frac")

    device = get_device()
    train_idx = np.flatnonzero(train)
    y_train = torch.tensor(y[train_idx], dtype=torch.float32)
    w_train = balance_weights(y[train_idx], df["source"].to_numpy()[train_idx], args.balance)
    scores = {}  # probe name -> logits for every row
    if "n_tokens" in df:
        log_len = np.log(df["n_tokens"].to_numpy(dtype=float))[:, None]
        baseline = LogisticRegression().fit(log_len[train_idx], y[train_idx], sample_weight=w_train)
        scores["length_baseline"] = baseline.decision_function(log_len)
    probes = {"linear": [], "mlp": []}
    losses = {"linear": [], "mlp": []}
    for j, layer in enumerate(info["layers"]):
        x_all = layer_tensor(acts, j)
        for kind, hidden in (("linear", 0), ("mlp", args.mlp_hidden)):
            probe, loss = train_layer_probe(x_all[train_idx], y_train, hidden=hidden, epochs=args.epochs,
                                            seed=args.seed, device=device,
                                            weights=torch.tensor(w_train, dtype=torch.float32))
            probes[kind].append(probe)
            losses[kind].append(loss)
            scores[f"{kind}_L{layer}"] = logits(probe, x_all, device)
            print(f"  layer {layer:3d} {kind:6s} train BCE {loss:.4f}")

    kept = {}
    for kind in probes:
        kept[kind] = kept_layers(losses[kind])
        scores[f"{kind}_ensemble"] = np.mean([scores[f"{kind}_L{info['layers'][j]}"] for j in kept[kind]], axis=0)
        print(f"{kind}_ensemble: layers {[info['layers'][j] for j in kept[kind]]}")

    report = []
    for split, m in (("val", val), ("heldout", heldout)):
        if not m.any():
            continue
        subsets = [("all", m)]
        for col in ("source", "family"):
            if df[col][m].nunique() > 1:
                subsets += [(f"{col}:{v}", m & (df[col] == v).to_numpy()) for v in sorted(df[col][m].unique())]
        for subset, mm in subsets:
            if len(set(y[mm])) < 2:
                continue
            for name, s in scores.items():
                report.append({"split": split, "subset": subset, "probe": name,
                               **metrics(y[mm], s[mm], df["group"][mm], args.n_boot, args.seed)})
    report = pd.DataFrame(report)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(args.acts))), "trained",
                           args.name or os.path.basename(os.path.normpath(args.acts)))
    os.makedirs(out_dir, exist_ok=True)
    report.to_csv(os.path.join(out_dir, "report.csv"), index=False)

    torch.save({"info": info, "kept": kept, "losses": losses, "mlp_hidden": args.mlp_hidden, "args": vars(args),
                "state": {k: [p.state_dict() for p in v] for k, v in probes.items()}},
               os.path.join(out_dir, "probes.pt"))
    export = torch.zeros(info["n_layers"], info["d_model"])
    offsets = {}
    for j, layer in enumerate(info["layers"]):
        w, c = probes["linear"][j].direction()
        if layer < info["n_layers"]:
            export[layer] = w.detach()
        offsets[layer] = c.item()
    torch.save({"logreg": export}, os.path.join(out_dir, "export.pt"))
    with open(os.path.join(out_dir, "ensemble.json"), "w") as f:
        json.dump({"variant": "logreg", "layers": [info["layers"][j] for j in kept["linear"]],
                   "offsets": offsets}, f, indent=2)

    cols = ["n", "auroc", "auroc_lo", "auroc_hi", "recall@1%fpr", "recall@1%fpr_lo", "recall@1%fpr_hi"]
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        for (split, subset), g in report.groupby(["split", "subset"], sort=False):
            print(f"\n{split} / {subset}:")
            print(g.set_index("probe")[cols].round(3).to_string())
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
