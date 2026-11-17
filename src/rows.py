"""
Labeled conversation rows: the training-data format for learned probes.

One JSON object per line:
    {"id": str, "messages": [{"role", "content"}, ...],
     "label": "eval" | "deploy" | null, "group": str, "family": str, "meta": {...}}

- label: null for rows that are only scored (e.g. EvalAwareBench, the data mix).
- group: rows that must stay on the same side of a split, e.g. the eval and deploy versions of one stem.
  Splits and bootstrap CIs are by group.
- family: the kind of data (e.g. "sysprompt:default", "real_chat"). Whole families can be held out.
"""
import hashlib
import json

LABELS = {"deploy": 0, "eval": 1}


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for i, r in enumerate(rows):
        r.setdefault("id", str(i))
        r.setdefault("label", None)
        r.setdefault("group", r["id"])
        r.setdefault("family", "default")
        r.setdefault("meta", {})
    return rows


def dolly_rows(stems, variants):
    """The original paired design as labeled rows: every stem under every variant's eval and deploy
    system prompt. A stem is one group across all variants, so no stem appears in both train and val."""
    rows = []
    for v in variants:
        for i, stem in enumerate(stems):
            for label in ("eval", "deploy"):
                rows.append({"id": f"dolly:{i}:{v['name']}:{label}", "label": label, "group": f"dolly:{i}",
                             "family": f"sysprompt:{v['name']}", "meta": {},
                             "messages": [{"role": "system", "content": v[label]},
                                          {"role": "user", "content": stem}]})
    return rows


def in_val(group, val_frac, seed=0):
    """Deterministic group-level split: the same group always lands on the same side."""
    h = int(hashlib.sha1(f"{seed}:{group}".encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < val_frac
