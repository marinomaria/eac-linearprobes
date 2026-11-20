#!/usr/bin/env python
"""
Convert the Needham et al. evaluation-awareness dataset (arXiv 2505.23836, huggingface.co/datasets/jjpn2/eval_awareness)
into labeled rows (src/rows.py) for extract.py / train_probes.py.

The dataset is gated, CC BY-NC-SA, and contains GAIA: keep it and the rows file out of git (data/ is gitignored).
Get it with:
    huggingface-cli download jjpn2/eval_awareness dataset.zip --repo-type dataset --local-dir <dir>
    unzip -P isthisreallythepassword <dir>/dataset.zip -d <dir>     # the password is public, in scripts/decrypt.sh

976 transcripts from 61 source datasets (categories); every category is all-eval or all-deploy, so the label is
a function of the category. Hence:
  - group  = the category: splits and bootstrap CIs are by category, so the val split measures transfer to
             unseen source datasets rather than to unseen transcripts of a seen template.
  - family = needham:{chat|agent}:{category}, so e.g. --holdout-family 'needham:agent:*' holds out all agentic data.
  - source = needham_chat / needham_agent (for --balance in train_probes.py).

Conversion, so the model is read where it would start its next turn, as with the Dolly prompts:
  - trailing assistant turns are dropped (the transcript ends at a user / tool / system message)
  - system messages after the first become user messages (Qwen's template accepts only a leading system message)
  - list contents are joined; tool calls are put in the chat-template format and tool definitions kept in "tools"
  - transcripts over --max-tokens drop whole messages from the middle (the leading system + user messages and
    the most recent messages are kept); meta.n_tokens and meta.truncated record the result

Usage:
    python scripts/prepare_needham.py <dir>/dataset.json --out data/needham_rows.jsonl
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import MODELS, format_chat


def text(content):
    if isinstance(content, str):
        return content
    return "\n\n".join(p["text"] for p in content if p.get("type") == "text")


def convert_message(m, first):
    role = m["role"]
    if role == "system" and not first:
        role = "user"
    out = {"role": role, "content": text(m["content"])}
    if m.get("tool_calls"):
        out["tool_calls"] = [
            {
                "type": "function",
                "id": tc.get("id"),
                "function": {"name": tc["function"], "arguments": tc.get("arguments") or {}},
            }
            for tc in m["tool_calls"]
        ]
    if role == "tool":
        out["tool_call_id"] = m.get("tool_call_id")
        if m.get("function"):
            out["name"] = m["function"]
    return out


def convert_tools(tools):
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {}),
            },
        }
        for t in tools or []
    ]


def fit(tokenizer, messages, tools, max_tokens):
    """Drop whole messages after the leading system/user ones, oldest first, until the prompt fits."""

    def n_tokens(msgs):
        return len(tokenizer(format_chat(tokenizer, msgs, tools), add_special_tokens=False)["input_ids"])

    head = 1 + (len(messages) > 1 and messages[1]["role"] == "user")
    n = n_tokens(messages)
    dropped = 0
    while n > max_tokens and len(messages) > head + 1:
        rest = messages[head + 1 :]
        # Never start the kept tail with a tool result whose call was dropped.
        while rest and rest[0]["role"] == "tool":
            rest = rest[1:]
        dropped += len(messages) - head - len(rest)
        messages = messages[:head] + rest
        n = n_tokens(messages)
    return messages, n, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="dataset.json from the decrypted dataset.zip")
    parser.add_argument("--out", default="data/needham_rows.jsonl")
    parser.add_argument("--model", default="qwen3.8-27b", help="Tokenizer used for --max-tokens")
    parser.add_argument("--max-tokens", type=int, default=16384)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODELS.get(args.model, args.model))

    with open(args.dataset) as f:
        samples = json.load(f)
    rows = []
    for s in samples:
        md = s["metadata"]
        msgs = [convert_message(m, i == 0) for i, m in enumerate(s["input"])]
        while msgs and msgs[-1]["role"] == "assistant":
            msgs.pop()
        tools = convert_tools(md.get("tools"))
        msgs, n, dropped = fit(tokenizer, msgs, tools or None, args.max_tokens)
        kind = "agent" if md.get("scaffold") else "chat"
        rows.append(
            {
                "id": f"needham:{s['id']}",
                "messages": msgs,
                "tools": tools or None,
                "label": "eval" if md["eval_category"] else "deploy",
                "group": f"needham:{md['category']}",
                "family": f"needham:{kind}:{md['category']}",
                "source": f"needham_{kind}",
                "meta": {
                    "category": md["category"],
                    "scaffold": md.get("scaffold"),
                    "eval_type": md.get("eval_type"),
                    "safety_category": md.get("safety_category"),
                    "human": (md["eval_author1"] + md["eval_author2"]) / 2,
                    "n_tokens": n,
                    "n_messages": len(msgs),
                    "truncated": dropped > 0,
                    "dropped_messages": dropped,
                },
            }
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    counts = collections.Counter((r["source"], r["label"]) for r in rows)
    print(f"{len(rows)} rows -> {args.out}; truncated {sum(r['meta']['truncated'] for r in rows)}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
