#!/usr/bin/env python
"""
Extract last-token activations of conversation rows (src/rows.py) and store them on disk (src/activations.py),
for train_probes.py (training) and score.py (scoring). This is the only step that needs the model / a GPU.

--data accepts one or more of (several are concatenated into one activation directory):
  - a rows .jsonl: training rows from scripts/prepare_*.py, or the data mix (data/mix-main.jsonl; unlabeled)
  - "dolly[:n_stems]": the paired system-prompt design, every Dolly stem under every variant's eval/deploy prompt
  - "cues[:n_stems]": the validation cues (incl. ZT) x placements x stems (default 20; src/cues.py), unlabeled

Layers default to 6 evenly spaced between 30% and 90% of depth; --layers overrides
(indices into hidden_states, i.e. the residual stream entering that block).

Usage:
    python extract.py --model qwen3.5-0.8b --data dolly:40
    python extract.py --model qwen3.8-27b --data dolly data/needham_rows.jsonl data/awarebench_rows.jsonl \
        data/wildchat_rows.jsonl data/verbalized_rows.jsonl --out output/qwen3.8-27b/acts/train
    python extract.py --model qwen3.8-27b --data cues
    python extract.py --model qwen3.8-27b --data data/mix-main.jsonl
"""
import argparse
import os

from src.activations import default_layers, extract
from src.cues import cue_rows
from src.model import load_model
from src.rows import dolly_rows, read_rows

OUTPUT_DIR = "output"


def build_rows(data):
    name, _, n = data.partition(":")
    if name == "dolly":
        return dolly_rows(int(n) if n else None)
    if name == "cues":
        return cue_rows(int(n or 20))
    return read_rows(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Alias from src/model.py:MODELS or HF repo id")
    parser.add_argument("--data", required=True, nargs="+", help='Rows .jsonl, "dolly[:n]" or "cues[:n]"')
    parser.add_argument("--out", default=None, help="Output directory (default: output/<model>/acts/<data name>)")
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=16384,
                        help="Cap on a batch's padded tokens (long agentic transcripts and mix prompts)")
    args = parser.parse_args()

    rows = [r for data in args.data for r in build_rows(data)]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        parser.error("duplicate row ids across --data sources")
    name = "+".join(os.path.splitext(os.path.basename(d))[0].replace(":", "_") for d in args.data)
    out = args.out or os.path.join(OUTPUT_DIR, args.model.replace("/", "_"), "acts", name)

    model, tokenizer, n_layers = load_model(args.model)
    layers = args.layers or default_layers(n_layers)
    print(f"{len(rows)} rows, layers {layers} -> {out}")
    extract(model, tokenizer, rows, layers, n_layers, out, args.model, args.batch_size, args.max_batch_tokens)
    print("Done.")


if __name__ == "__main__":
    main()
