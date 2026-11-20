"""
Validation cues: short hand-written texts with an expected reading (eval / deploy / neutral), from
datasets/validation_prompts.json (the validation set, VAL) and the zero-tolerance set
datasets/validation_prompts_zt.csv (ZT). Each row records its file stem as "set".

ZT is a strict subset of VAL: every ZT cue is also in validation_prompts.json, with the same expected label. Its
rows are extracted twice (once per set); report.py scores VAL from validation_prompts.json alone, so no cue is
counted twice, and ZT on its own.

Each cue ("evaluation", "chicken", "This is not a test.", ...) is shown in three placements:
  - system: the cue is the system prompt, followed by a Dolly stem as the user turn (one row per stem)
  - prefix: the cue is prepended to a Dolly stem in a single user turn (one row per stem)
  - user:   the cue alone is the user turn
An empty cue (category "none") is the no-cue baseline. The rows are unlabeled; meta carries the cue, its
placement and g, the index of its (cue, placement) group, over whose stems report.py averages the scores.
"""
import csv
import json
import os

from src.rows import dolly_stems

CUE_FILES = ["datasets/validation_prompts.json", "datasets/validation_prompts_zt.csv"]
PLACEMENTS = ["system", "prefix", "user"]
VAL = "validation_prompts"
ZT = "validation_prompts_zt"  # subset of VAL


def read_cues(path):
    """[{category, expected, text, set}] from a .json list or a .csv with those columns."""
    with open(path) as f:
        items = list(csv.DictReader(f)) if path.endswith(".csv") else json.load(f)
    name = os.path.splitext(os.path.basename(path))[0]
    return [{"category": i["category"], "expected": i["expected"], "text": i["text"], "set": name} for i in items]


def conversations(text, placement, stems):
    if placement == "system":
        system = [{"role": "system", "content": text}] if text else []
        return [system + [{"role": "user", "content": s}] for s in stems]
    if placement == "prefix":
        return [[{"role": "user", "content": f"{text}\n\n{s}" if text else s}] for s in stems]
    return [[{"role": "user", "content": text}]] if text else []


def cue_rows(n_stems=20, files=CUE_FILES):
    stems = dolly_stems(n_stems)
    rows, g = [], 0
    for item in (i for path in files for i in read_cues(path)):
        for placement in PLACEMENTS:
            convs = conversations(item["text"], placement, stems)
            for k, messages in enumerate(convs):
                rows.append({"id": f"cue:{g}:{k}", "messages": messages, "label": None, "group": f"cue:{g}",
                             "family": f"cues:{item['set']}", "source": "cues", "tools": None,
                             "meta": {**item, "placement": placement, "g": g}})
            g += bool(convs)
    return rows
