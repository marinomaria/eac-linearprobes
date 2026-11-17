#!/usr/bin/env python
"""
Score activations stored by extract.py with probes trained by train_probes.py. Needed for the MLP probes;
the logistic probes can also be used through export.pt with the existing scripts.

Writes <trained dir>/scores_<acts name>.jsonl: one line per row with id, label, group, family, meta and
"scores": {probe name: logit}. If the rows are labeled, prints AUROC / recall@1%FPR per probe.

Usage:
    python score.py output/qwen3.8-27b/trained/v1 output/qwen3.8-27b/acts/mix-main
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from src.activations import layer_tensor, load
from src.classifiers import LayerProbe, metrics
from src.model import get_device
from src.rows import LABELS
from train_probes import logits


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trained", help="Directory written by train_probes.py")
    parser.add_argument("acts", help="Activation directory written by extract.py")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    saved = torch.load(os.path.join(args.trained, "probes.pt"))
    acts, rows, info = load(args.acts)
    layers = saved["info"]["layers"]
    if info["model"] != saved["info"]["model"] or not set(layers) <= set(info["layers"]):
        parser.error(f"activations ({info['model']}, layers {info['layers']}) do not match probes "
                     f"({saved['info']['model']}, layers {layers})")

    device = get_device()
    scores = {}
    for j, layer in enumerate(layers):
        x = layer_tensor(acts, info["layers"].index(layer))
        for kind, hidden in (("linear", 0), ("mlp", saved["mlp_hidden"])):
            state = saved["state"][kind][j]
            probe = LayerProbe(x.shape[1], state["mean"], state["std"], hidden)
            probe.load_state_dict(state)
            scores[f"{kind}_L{layer}"] = logits(probe.eval(), x, device)
    for kind, kept in saved["kept"].items():
        scores[f"{kind}_ensemble"] = np.mean([scores[f"{kind}_L{layers[j]}"] for j in kept], axis=0)

    name = os.path.basename(os.path.normpath(args.acts))
    out_path = os.path.join(args.trained, f"scores_{name}.jsonl")
    with open(out_path, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({**r, "scores": {k: float(v[i]) for k, v in scores.items()}}) + "\n")

    df = pd.DataFrame(rows)
    m = df["label"].notna().to_numpy()
    y = df["label"][m].map(LABELS).to_numpy()
    if m.any() and len(set(y)) == 2:
        report = pd.DataFrame({k: metrics(y, s[m], df["group"][m], args.n_boot) for k, s in scores.items()}).T
        print(report[["n", "auroc", "auroc_lo", "auroc_hi", "recall@1%fpr"]].round(3).to_string())
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
