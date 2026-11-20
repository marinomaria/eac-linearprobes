#!/usr/bin/env python
"""
Score activations stored by extract.py with every probe of a train_probes.py run (meandiff_L*, linear_L*, mlp_L*,
linear_ensemble, mlp_ensemble). No model or GPU needed.

Writes <trained dir>/scores_<acts name>.jsonl: one line per row with id, label, group, family, source, meta and
"scores": {probe: logit}. If the rows are labeled, prints AUROC / recall@1%FPR per probe.

Usage:
    python score.py output/qwen3.8-27b/trained/all output/qwen3.8-27b/acts/cues
    python score.py output/qwen3.8-27b/trained/all output/qwen3.8-27b/acts/mix-main
"""

import argparse
import json
import os

import pandas as pd

from src.activations import layer_tensor, load
from src.classifiers import load_trained, metrics, score_all
from src.model import get_device
from src.rows import LABELS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trained", help="Directory written by train_probes.py")
    parser.add_argument("acts", help="Activation directory written by extract.py")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    saved, probes = load_trained(args.trained)
    acts, rows, info = load(args.acts)
    layers = saved["info"]["layers"]
    if info["model"] != saved["info"]["model"] or not set(layers) <= set(info["layers"]):
        parser.error(
            f"activations ({info['model']}, layers {info['layers']}) do not match probes "
            f"({saved['info']['model']}, layers {layers})"
        )

    scores = score_all(saved, probes, lambda layer: layer_tensor(acts, info["layers"].index(layer)), get_device())

    name = os.path.basename(os.path.normpath(args.acts))
    out_path = os.path.join(args.trained, f"scores_{name}.jsonl")
    with open(out_path, "w") as f:
        f.writelines(
            json.dumps({**r, "scores": {k: float(v[i]) for k, v in scores.items()}}) + "\n" for i, r in enumerate(rows)
        )

    df = pd.DataFrame(rows)
    m = df["label"].notna().to_numpy()
    y = df["label"][m].map(LABELS).to_numpy()
    if m.any() and len(set(y)) == 2:
        report = pd.DataFrame({k: metrics(y, s[m], df["group"][m], args.n_boot) for k, s in scores.items()}).T
        print(report[["n", "auroc", "auroc_lo", "auroc_hi", "recall@1%fpr"]].round(3).to_string())
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
