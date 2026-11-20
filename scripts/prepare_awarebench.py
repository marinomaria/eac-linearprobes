#!/usr/bin/env python
"""
Convert aware-bench / EvalDetectBench trajectories (huggingface.co/datasets/el7982/aware-bench, CC BY-NC-SA 4.0)
into labeled rows (src/rows.py) for extract.py / train_probes.py. Not to be confused with EvalAwareBench
(aisa-group/EvalAwareBench), which is score-only.

Source: collected_trajectories/<generator>/all.json: the generator's eval trajectories (41 benchmarks) plus the
shared deployment pool (13 sources: real chats, coding-agent sessions, ...). Like Needham, every category is
all-eval or all-deploy. The default generator is Qwen 3.5 so the eval transcripts' assistant turns are in a Qwen
model's style (the deployment pool's assistant turns come from other models whatever the generator).

Conversion reuses scripts/prepare_needham.py (same inspect-style records). Differences:
  - tool definitions are dropped by default (--keep-tools to keep them): in this dataset only eval trajectories
    carry them, so they would be a label shortcut. Tool calls and results stay in the messages.
  - categories that are measurement data or conflict with Needham's labels are dropped (EXCLUDE).
  - rows whose task (system prompt + first user message) is in Needham's rows are dropped (shared sources).
  - group: Needham's group for the same benchmark (GROUP_ALIASES or the same category name), so splits by group
    never put one benchmark on both sides; otherwise awarebench:{category}.
  - family = awarebench:{chat|agent}:{category}; source = awarebench_chat / awarebench_agent.

Keep the downloaded data and rows file out of git (data/ is gitignored).

Usage:
    python scripts/prepare_awarebench.py --out data/awarebench_rows.jsonl
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_needham import convert_message, convert_tools, fit

from src.model import MODELS

REPO = "el7982/aware-bench"
# mask: MASK is part of the data mix (measurement set). hackaprompt_dataset: eval here, deploy in Needham.
EXCLUDE = {"mask", "hackaprompt_dataset"}
# aware-bench category -> Needham category of the same benchmark
GROUP_ALIASES = {
    "swebench_verified_mini": "swebench",
    "wmdp_bio": "wmdp",
    "mmmlu": "mmlu",
    "hh_rlhf_redteam": "hh_rlhf",
    "gaia_level2": "gaia_runs",
}


def task_key(messages):
    """The task a transcript is about: its leading system prompt and first user message."""
    system = messages[0]["content"].strip()[:2000] if messages and messages[0]["role"] == "system" else ""
    user = next((m["content"].strip()[:2000] for m in messages if m["role"] == "user" and m["content"].strip()), "")
    return system, user


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", default="qwen3.5-397b-a17b", help="collected_trajectories/<generator>")
    parser.add_argument("--file", default=None, help="Local all.json instead of downloading")
    parser.add_argument("--needham", default="data/needham_rows.jsonl", help="Needham rows, for dedup and groups")
    parser.add_argument("--out", default="data/awarebench_rows.jsonl")
    parser.add_argument("--model", default="qwen3.8-27b", help="Tokenizer used for --max-tokens")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--keep-tools", action="store_true", help="Keep tool definitions (eval-only here)")
    args = parser.parse_args()

    path = args.file
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(REPO, f"collected_trajectories/{args.generator}/all.json", repo_type="dataset")
    with open(path) as f:
        samples = json.load(f)

    needham_cats, seen = set(), set()  # seen: Needham task keys
    if os.path.exists(args.needham):
        with open(args.needham) as f:
            for line in f:
                r = json.loads(line)
                needham_cats.add(r["meta"]["category"])
                seen.add(task_key(r["messages"]))
    else:
        print(f"warning: {args.needham} not found; no Needham dedup or group aliasing")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODELS.get(args.model, args.model))

    rows, skipped = [], collections.Counter()
    for s in samples:
        md = s["metadata"]
        cat = md["category"]
        if cat in EXCLUDE:
            skipped[f"excluded:{cat}"] += 1
            continue
        msgs = [convert_message(m, i == 0) for i, m in enumerate(s["input"])]
        while msgs and msgs[-1]["role"] == "assistant":
            msgs.pop()
        if not msgs:
            skipped["empty"] += 1
            continue
        if task_key(msgs) in seen:
            skipped[f"in_needham:{cat}"] += 1
            continue
        tools = convert_tools(md.get("tools")) if args.keep_tools else []
        msgs, n, dropped = fit(tokenizer, msgs, tools or None, args.max_tokens)
        kind = "agent" if md.get("has_tools") or md.get("scaffold") else "chat"
        alias = GROUP_ALIASES.get(cat, cat)
        group = f"needham:{alias}" if alias in needham_cats else f"awarebench:{cat}"
        rows.append(
            {
                "id": f"awarebench:{s['id']}",
                "messages": msgs,
                "tools": tools or None,
                "label": "eval" if md["eval_category"] else "deploy",
                "group": group,
                "family": f"awarebench:{kind}:{cat}",
                "source": f"awarebench_{kind}",
                "meta": {
                    "category": cat,
                    "subcategory": md.get("subcategory"),
                    "scaffold": md.get("scaffold"),
                    "generator": args.generator,
                    "generator_model": md.get("generator_model"),
                    "n_tokens": n,
                    "n_messages": len(msgs),
                    "truncated": dropped > 0,
                    "dropped_messages": dropped,
                },
            }
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    counts = collections.Counter((r["source"], r["label"]) for r in rows)
    print(
        f"{len(rows)} rows -> {args.out}; truncated {sum(r['meta']['truncated'] for r in rows)}; "
        f"{sum(r['group'].startswith('needham:') for r in rows)} share a Needham group"
    )
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    for k, v in sorted(skipped.items()):
        print(f"  skipped {k}: {v}")


if __name__ == "__main__":
    main()
