"""
Extract last-token activations for a set of layers once, store them on disk, train probes many times.

Layout of an activation directory:
    acts.npy    [n_rows, n_layers_kept, d_model], bf16 stored as uint16 bits (numpy has no bf16); memory-mapped
    rows.jsonl  one line per row: id, label, group, family, source, meta, n_tokens (no messages)
    info.json   model, layers (indices into hidden_states), n_layers (of the model), d_model

Only the last token of the stored layers is kept during the forward pass (pre-hooks on the decoder blocks, and
the pass stops after the deepest one), so memory does not grow with layers x positions as with
output_hidden_states: long agentic transcripts fit.

bf16 rather than fp16: residual streams of large models carry activations beyond fp16's range.
Size: n_rows x n_layers_kept x d_model x 2 bytes (150k rows, 6 layers, d=5120: ~9 GB).
"""

import json
import os

import numpy as np
import torch
from tqdm import tqdm

from src.model import format_chat


def default_layers(n_layers, n=6, lo=0.3, hi=0.9):
    """n layers evenly spaced between lo and hi fractions of depth (the paper's 30-90% band)."""
    return sorted({round(f * n_layers) for f in np.linspace(lo, hi, n)})


def token_budget_batches(order, lengths, batch_size, max_batch_tokens):
    """Split `order` (sorted by ascending length) into batches of at most `batch_size` items whose padded size
    (items x longest) stays within `max_batch_tokens`. A prompt longer than the budget gets a batch of its own."""
    batch = []
    for i in order:
        if batch and (len(batch) == batch_size or (len(batch) + 1) * lengths[i] > max_batch_tokens):
            yield batch
            batch = []
        batch.append(i)
    if batch:
        yield batch


class _Stop(Exception):
    pass


def _decoder_layers(model, n_layers):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if isinstance(layers, torch.nn.ModuleList) and len(layers) == n_layers:
        return layers
    for module in model.modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) == n_layers:
            return module
    raise ValueError("could not find the decoder layers")


@torch.no_grad()
def _last_token(model, tokenizer, prompts, layers, blocks):
    """Residual stream entering block l (= hidden_states[l]) at the last position, for each l in layers."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, pad_to_multiple_of=64, add_special_tokens=False)
    got = {}

    def hook(layer):
        def fn(module, args, kwargs):
            h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            got[layer] = h[:, -1, :].to(torch.bfloat16).cpu()
            if len(got) == len(layers):
                raise _Stop

        return fn

    handles = [blocks[layer].register_forward_pre_hook(hook(layer), with_kwargs=True) for layer in layers]
    try:
        model(**enc.to(model.device), use_cache=False, logits_to_keep=1)
    except _Stop:
        pass
    finally:
        for h in handles:
            h.remove()
    return torch.stack([got[layer] for layer in layers], dim=1)


def extract(model, tokenizer, rows, layers, n_layers, out_dir, model_name, batch_size=8, max_batch_tokens=16384):
    if max(layers) >= n_layers:
        raise ValueError(f"layers must be < {n_layers} (residual stream entering a block)")
    os.makedirs(out_dir, exist_ok=True)
    blocks = _decoder_layers(model, n_layers)
    formatted = [format_chat(tokenizer, r["messages"], r.get("tools")) for r in rows]
    lengths = [len(ids) for ids in tokenizer(formatted, add_special_tokens=False)["input_ids"]]
    order = sorted(range(len(rows)), key=lambda i: lengths[i])
    d_model = model.config.get_text_config().hidden_size

    acts = np.lib.format.open_memmap(
        os.path.join(out_dir, "acts.npy"), mode="w+", dtype=np.uint16, shape=(len(rows), len(layers), d_model)
    )
    for idx in tqdm(list(token_budget_batches(order, lengths, batch_size, max_batch_tokens)), desc="Extracting"):
        a = _last_token(model, tokenizer, [formatted[i] for i in idx], layers, blocks)
        acts[idx] = a.view(torch.uint16).numpy()
    acts.flush()

    with open(os.path.join(out_dir, "rows.jsonl"), "w") as f:
        for r, n in zip(rows, lengths, strict=True):
            keys = ("id", "label", "group", "family", "meta")
            f.write(json.dumps({**{k: r[k] for k in keys}, "source": r["source"], "n_tokens": n}) + "\n")
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump({"model": model_name, "layers": layers, "n_layers": n_layers, "d_model": d_model}, f, indent=2)


def load(act_dirs):
    """One directory or several with the same model and layers (rows concatenated, e.g. a training set extracted
    in parts). Returns (acts, rows, info); acts is a list of memmaps [n_i, k, d] uint16, read with layer_tensor."""
    act_dirs = [act_dirs] if isinstance(act_dirs, str) else list(act_dirs)
    acts, rows, info = [], [], None
    for d in act_dirs:
        with open(os.path.join(d, "info.json")) as f:
            this = json.load(f)
        if info is not None and this != info:
            raise ValueError(f"{d} ({this}) does not match {act_dirs[0]} ({info})")
        info = this
        with open(os.path.join(d, "rows.jsonl")) as f:
            rows += [json.loads(line) for line in f]
        acts.append(np.load(os.path.join(d, "acts.npy"), mmap_mode="r"))
    return acts, rows, info


def layer_tensor(acts, j):
    """float32 tensor [n, d] for the j-th stored layer, all rows."""
    a = np.concatenate([np.ascontiguousarray(x[:, j, :]) for x in acts])
    return torch.from_numpy(a.view(np.int16)).view(torch.bfloat16).float()
