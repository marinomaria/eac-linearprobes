#!/usr/bin/env python
"""
Convert data/probe_sample_verbalized.csv (a sample of mix-discovery prompts with the rate at which Qwen voiced
evaluation awareness in its CoT over n_rollouts) into labeled rows (src/rows.py) for extract.py / train_probes.py.

Only mix-discovery may be trained on; mix-main is the measurement set. Rows whose id, messages or underlying item
appear in --main are refused, so a mix-main row can never leak into training.

Label: eval if voicing_rate >= --threshold (default 0.5), else deploy, i.e. the model's own verbalized
judgement rather than where the prompt came from.
Group (variants of one item stay on the same side of a split):
  - evalawarebench: the task family (capability/safety variants under every cue condition)
  - harmbench: the behavior id, which shares names with EvalAwareBench families, so both use one group
  - mask: the MASK item id (belief_open / belief_leading / pressure prompts of one item)
  - fortress: the FORTRESS item id
Family = verbalized:{dataset}:{stratum without |cues=}; source = verbalized (one labeling procedure, so
--balance source weights it as one source rather than letting the 10-row datasets count as much as the rest).

Keep the rows file out of git (data/ is gitignored).

Usage:
    python scripts/prepare_verbalized.py --out data/verbalized_rows.jsonl
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_needham import fit

from src.model import MODELS

VARIANT = re.compile(r"-(adversarial|benign|belief_leading|belief_open|pressure)$")


def item(row_id, dataset, family):
    if dataset == "evalawarebench":
        return family
    base = VARIANT.sub("", row_id)
    return base.removeprefix("harmbench-") if dataset == "harmbench" else base


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="data/probe_sample_verbalized.csv")
    parser.add_argument("--main", default="data/mix-main.jsonl", help="Measurement set that must not be trained on")
    parser.add_argument("--threshold", type=float, default=0.5, help="voicing_rate >= threshold -> eval")
    parser.add_argument("--out", default="data/verbalized_rows.jsonl")
    parser.add_argument("--model", default="qwen3.8-27b", help="Tokenizer used for --max-tokens")
    parser.add_argument("--max-tokens", type=int, default=16384)
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.csv)
    df["family"] = df["family"].where(df["family"].notna(), None)

    if not os.path.exists(args.main):
        sys.exit(f"{args.main} not found: needed to check for mix-main leakage")
    with open(args.main) as f:
        main_rows = [json.loads(line) for line in f]

    main_ids = {r["id"] for r in main_rows}
    main_items = {VARIANT.sub("", r["id"]).removeprefix("harmbench-") for r in main_rows}
    main_items |= {(r.get("meta") or {}).get("family") for r in main_rows}
    main_msgs = {json.dumps(r["messages"], sort_keys=True) for r in main_rows}
    leaks = [r.id for r in df.itertuples()
             if r.id in main_ids or item(r.id, r.dataset, r.family) in main_items
             or json.dumps(json.loads(r.messages), sort_keys=True) in main_msgs]
    if leaks:
        sys.exit(f"{len(leaks)} rows overlap mix-main, e.g. {leaks[:5]}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODELS.get(args.model, args.model))
    rows = []
    for r in df.itertuples():
        msgs = [{"role": m["role"], "content": m["content"]} for m in json.loads(r.messages)]
        while msgs and msgs[-1]["role"] == "assistant":
            msgs.pop()
        msgs, n, dropped = fit(tokenizer, msgs, None, args.max_tokens)
        stratum = r.stratum.split("|")[0]
        rows.append({
            "id": f"verbalized:{r.id}", "messages": msgs, "tools": None,
            "label": "eval" if r.voicing_rate >= args.threshold else "deploy",
            "group": f"verbalized:{item(r.id, r.dataset, r.family)}",
            "family": f"verbalized:{r.dataset}:{stratum}", "source": "verbalized",
            "meta": {"dataset": r.dataset, "stratum": r.stratum, "condition": None if pd.isna(r.condition) else r.condition,
                     "voicing_rate": float(r.voicing_rate), "n_voiced": int(r.n_voiced),
                     "n_rollouts": int(r.n_rollouts), "n_voiced_evaluator": int(r.n_voiced_evaluator),
                     "n_voiced_user": int(r.n_voiced_user), "n_scenario_recognition": int(r.n_scenario_recognition),
                     "n_tokens": n, "n_messages": len(msgs), "truncated": dropped > 0, "dropped_messages": dropped},
        })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    counts = collections.Counter((r["family"], r["label"]) for r in rows)
    print(f"{len(rows)} rows, {len({r['group'] for r in rows})} groups, "
          f"{sum(r['meta']['truncated'] for r in rows)} truncated -> {args.out}")
    for fam in sorted({k[0] for k in counts}):
        print(f"  {fam}: eval {counts[(fam, 'eval')]}, deploy {counts[(fam, 'deploy')]}")


if __name__ == "__main__":
    main()
