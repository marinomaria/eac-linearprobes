#!/usr/bin/env python
"""
Sample real user chats from WildChat (huggingface.co/datasets/allenai/WildChat, ODC-BY) as deploy rows
(src/rows.py) for extract.py / train_probes.py.

WildChat is huge (529k conversations), so it is heavily filtered and only --n rows are kept: the default (100)
brings chat deploy level with chat eval in Needham + aware-bench (437 eval vs 337 deploy before WildChat).
Filters, applied to the loaded shards:
  - English, not toxic, not redacted, no OpenAI-moderation flag on any turn
  - at most --max-turns user turns; first user message 20-4000 characters
  - no templates or bots: any conversation whose first user message shares its first 60 characters with another
    conversation's is dropped (AutoGPT prompts, "Ignore previous instructions. I want you to act as ...",
    Midjourney prompt generators, "hi", ...)
  - no jailbreak / test-talk (JAILBREAK): the deploy class should not carry eval-like cues
  - not already in Needham or aware-bench rows (WildBench, in aware-bench, is drawn from WildChat)
Conversion: trailing assistant turn dropped, so the model is read where it would answer the last user message.
group = "wildchat" (one source dataset, as Needham groups by dataset); family = wildchat:chat; source = wildchat.

Usage:
    python scripts/prepare_wildchat.py --out data/wildchat_rows.jsonl
"""

import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_awarebench import task_key
from prepare_needham import fit

from src.model import MODELS

REPO = "allenai/WildChat"
JAILBREAK = re.compile(
    r"jailbr|developer mode|\bDAN\b|do anything now|ignore (all |the |any )?(previous|prior|above) instructions|"
    r"(being|are you|you are) (tested|evaluated|benchmarked)|\bbenchmark|\bevaluation\b|\btest(ing)? you\b",
    re.IGNORECASE,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=100, help="Rows to sample after filtering")
    parser.add_argument("--shards", type=int, nargs="+", default=[0], help="Parquet shards 0-5 (~88k each)")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["data/needham_rows.jsonl", "data/awarebench_rows.jsonl"],
        help="Rows files whose tasks must not reappear",
    )
    parser.add_argument("--out", default="data/wildchat_rows.jsonl")
    parser.add_argument("--model", default="qwen3.8-27b", help="Tokenizer used for --max-tokens")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download

    df = pd.concat(
        [
            pd.read_parquet(hf_hub_download(REPO, f"data/train-{i:05d}-of-00006.parquet", repo_type="dataset"))
            for i in args.shards
        ],
        ignore_index=True,
    )
    n_total = len(df)

    seen = set()
    for path in args.exclude:
        if os.path.exists(path):
            with open(path) as f:
                seen.update(task_key(json.loads(line)["messages"])[1] for line in f)
        else:
            print(f"warning: --exclude {path} not found")

    first = df.conversation.map(lambda c: c[0]["content"].strip())
    prefix_counts = collections.Counter(first.str[:60].str.lower())
    steps = collections.OrderedDict()

    def keep(name, mask):
        nonlocal df, first
        df, first = df[mask], first[mask]
        steps[name] = len(df)

    keep("english, not toxic/redacted", (df.language == "English") & ~df.toxic & ~df.redacted)
    keep("no moderation flag", df.openai_moderation.map(lambda ms: not any(m["flagged"] for m in ms)))
    keep(f"<= {args.max_turns} turns", df.turn <= args.max_turns)
    keep("first message 20-4000 chars", first.str.len().between(20, 4000))
    keep("unique 60-char prefix", first.str[:60].str.lower().map(prefix_counts) == 1)
    keep(
        "no jailbreak/test talk",
        ~df.conversation.map(lambda c: any(JAILBREAK.search(m["content"]) for m in c if m["role"] == "user")),
    )
    keep("not in --exclude rows", ~first.str[:2000].isin(seen))
    if len(df) < args.n:
        sys.exit(f"only {len(df)} conversations pass the filters; load more --shards")
    picked = random.Random(args.seed).sample(list(df.index), args.n)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODELS.get(args.model, args.model))
    rows = []
    for i in picked:
        r = df.loc[i]
        msgs = [{"role": m["role"], "content": m["content"]} for m in r.conversation]
        while msgs and msgs[-1]["role"] == "assistant":
            msgs.pop()
        msgs, n, dropped = fit(tokenizer, msgs, None, args.max_tokens)
        rows.append(
            {
                "id": f"wildchat:{r.conversation_id}",
                "messages": msgs,
                "tools": None,
                "label": "deploy",
                "group": "wildchat",
                "family": "wildchat:chat",
                "source": "wildchat",
                "meta": {
                    "category": "wildchat",
                    "model": r.model,
                    "timestamp": str(r.timestamp),
                    "turn": int(r.turn),
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
    print(f"{n_total} conversations in shards {args.shards}; after each filter:")
    for name, n in steps.items():
        print(f"  {name}: {n}")
    print(
        f"{len(rows)} sampled -> {args.out}; turns {dict(sorted(collections.Counter(r['meta']['turn'] for r in rows).items()))}"
    )


if __name__ == "__main__":
    main()
