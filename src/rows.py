"""
Conversation rows: the format of every dataset that is extracted, trained on or scored.

One JSON object per line:
    {"id": str, "messages": [{"role", "content"}, ...],
     "label": "eval" | "deploy" | null, "group": str, "family": str, "source": str, "meta": {...},
     "tools": [...] | null}

- label: null for rows that are only scored (the data mix, the validation cues).
- group: rows that must stay on the same side of a split, e.g. the eval and deploy versions of one stem.
  Splits and bootstrap CIs are by group.
- family: the kind of data (e.g. "sysprompt:default", "needham:agent:swebench"). Whole families can be held out.
- source: the dataset a row comes from (default: the mix's "dataset", else the family's first ":" component).
  train_probes.py --balance weights sources equally, so a large or long-prompt dataset does not dominate the probe.
- tools: optional OpenAI-style tool definitions, rendered by the chat template (agentic transcripts).
"""

import hashlib
import json
import random

LABELS = {"deploy": 0, "eval": 1}
DOLLY_STEMS = "datasets/contrastive_dataset_dolly.json"
SYSTEM_PROMPTS = "datasets/system_prompts.json"


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for i, r in enumerate(rows):
        r.setdefault("id", str(i))
        r.setdefault("label", None)
        r.setdefault("group", r["id"])
        r.setdefault("family", "default")
        r.setdefault("source", r.get("dataset") or r["family"].split(":")[0])
        r.setdefault("meta", {})
        r.setdefault("tools", None)
    return rows


def dolly_stems(n=None):
    """Contrastive prompt stems (Dolly); n: a seeded random sample."""
    with open(DOLLY_STEMS) as f:
        stems = [row["stem"].strip() for row in json.load(f)]
    return stems if n is None else random.Random(0).sample(stems, min(n, len(stems)))


def dolly_rows(n_stems=None):
    """The paired system-prompt design: every stem under every variant's eval and deploy system prompt
    (datasets/system_prompts.json). A stem is one group across all variants, so no stem is in both train and val."""
    with open(SYSTEM_PROMPTS) as f:
        variants = json.load(f)
    rows = []
    for v in variants:
        for i, stem in enumerate(dolly_stems(n_stems)):
            for label in ("eval", "deploy"):
                rows.append(
                    {
                        "id": f"dolly:{i}:{v['name']}:{label}",
                        "label": label,
                        "group": f"dolly:{i}",
                        "family": f"sysprompt:{v['name']}",
                        "source": "dolly",
                        "meta": {},
                        "tools": None,
                        "messages": [{"role": "system", "content": v[label]}, {"role": "user", "content": stem}],
                    }
                )
    return rows


def in_val(group, val_frac, seed=0):
    """Deterministic group-level split: the same group always lands on the same side."""
    h = int(hashlib.sha1(f"{seed}:{group}".encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < val_frac
