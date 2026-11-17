"""
Extract last-token activations for a set of layers once, store them on disk, train probes many times.

Layout of an activation directory:
    acts.npy    [n_rows, n_layers_kept, d_model], bf16 stored as uint16 bits (numpy has no bf16); memory-mapped
    rows.jsonl  one line per row: id, label, group, family, meta (no messages)
    info.json   model, layers (indices into hidden_states), n_layers (of the model), d_model

bf16 rather than fp16: residual streams of large models carry activations beyond fp16's range.
Size: n_rows x n_layers_kept x d_model x 2 bytes (150k rows, 6 layers, d=5120: ~9 GB).
"""
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from src.model import format_chat
from src.probe import token_budget_batches


def default_layers(n_layers, n=6, lo=0.3, hi=0.9):
    """n layers evenly spaced between lo and hi fractions of depth (the paper's 30-90% band)."""
    return sorted({round(f * n_layers) for f in np.linspace(lo, hi, n)})


@torch.no_grad()
def _last_token(model, tokenizer, prompts, layers):
    enc = tokenizer(prompts, return_tensors="pt", padding=True, pad_to_multiple_of=64, add_special_tokens=False)
    out = model(**enc.to(model.device), output_hidden_states=True, use_cache=False, logits_to_keep=1)
    return torch.stack([out.hidden_states[layer][:, -1, :] for layer in layers], dim=1).to(torch.bfloat16).cpu()


def extract(model, tokenizer, rows, layers, n_layers, out_dir, model_name, batch_size=8, max_batch_tokens=16384):
    os.makedirs(out_dir, exist_ok=True)
    formatted = [format_chat(tokenizer, r["messages"]) for r in rows]
    lengths = [len(ids) for ids in tokenizer(formatted, add_special_tokens=False)["input_ids"]]
    order = sorted(range(len(rows)), key=lambda i: lengths[i])
    d_model = model.config.get_text_config().hidden_size

    acts = np.lib.format.open_memmap(os.path.join(out_dir, "acts.npy"), mode="w+", dtype=np.uint16,
                                     shape=(len(rows), len(layers), d_model))
    for idx in tqdm(list(token_budget_batches(order, lengths, batch_size, max_batch_tokens)), desc="Extracting"):
        a = _last_token(model, tokenizer, [formatted[i] for i in idx], layers)
        acts[idx] = a.view(torch.uint16).numpy()
    acts.flush()

    with open(os.path.join(out_dir, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("id", "label", "group", "family", "meta")}) + "\n")
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump({"model": model_name, "layers": layers, "n_layers": n_layers, "d_model": d_model}, f, indent=2)


def load(act_dir):
    """Returns (acts memmap [n, k, d] uint16, rows, info). Use `layer_tensor` to read one layer as float32."""
    with open(os.path.join(act_dir, "info.json")) as f:
        info = json.load(f)
    with open(os.path.join(act_dir, "rows.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    return np.load(os.path.join(act_dir, "acts.npy"), mmap_mode="r"), rows, info


def layer_tensor(acts, j, idx=None):
    """float32 tensor [n, d] for the j-th stored layer (optionally only rows idx)."""
    a = acts[:, j, :] if idx is None else acts[idx, j, :]
    return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16).float()
