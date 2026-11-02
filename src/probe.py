"""
Linear probes via normalized mean difference of activations.
Mirrors the approach in https://github.com/Jordine/evaluation-awareness-probing
"""
import torch
from tqdm import tqdm

from src.model import format_chat, last_token_activations


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def train_probes(model, tokenizer, stems, eval_system, deploy_system, n_layers, batch_size=8):
    """
    One probe per layer: unit vector along mean(eval) - mean(deploy) last-token activations.

    Each stem is posed twice, under the evaluation and the deployment system prompt.

    Returns:
        float32 tensor [n_layers, d_model]
    """
    pos_sum, neg_sum = 0, 0
    for chunk in tqdm(list(batches(stems, batch_size)), desc="Training probes"):
        pos = [format_chat(tokenizer, [{"role": "system", "content": eval_system},
                                       {"role": "user", "content": s}]) for s in chunk]
        neg = [format_chat(tokenizer, [{"role": "system", "content": deploy_system},
                                       {"role": "user", "content": s}]) for s in chunk]
        pos_sum = pos_sum + last_token_activations(model, tokenizer, pos, n_layers).sum(dim=0)
        neg_sum = neg_sum + last_token_activations(model, tokenizer, neg, n_layers).sum(dim=0)

    diff = (pos_sum - neg_sum) / len(stems)
    # Layer 0 is the token embedding of the same last token in both prompts, so its
    # diff is exactly zero; clamping yields a zero probe there instead of NaN.
    return diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def project(model, tokenizer, probes, prompts, n_layers, batch_size=8):
    """
    Project the last-token activation of each user prompt onto every layer's probe,
    for every probe set (system-prompt variant) at once: one forward pass serves all.

    Args:
        probes: float32 tensor [n_variants, n_layers, d_model]

    Returns:
        float32 tensor [len(prompts), n_variants, n_layers]
    """
    return project_conversations(model, tokenizer, probes, [[{"role": "user", "content": p}] for p in prompts],
                                 n_layers, batch_size)


def token_budget_batches(order, lengths, batch_size, max_batch_tokens):
    """Split `order` (sorted by ascending length) into batches of at most `batch_size` items whose
    padded size (items x longest) stays within `max_batch_tokens`. A prompt longer than the budget
    gets a batch of its own."""
    batch = []
    for i in order:
        if batch and (len(batch) == batch_size or (len(batch) + 1) * lengths[i] > max_batch_tokens):
            yield batch
            batch = []
        batch.append(i)
    if batch:
        yield batch


def project_conversations(model, tokenizer, probes, conversations, n_layers, batch_size=8, max_batch_tokens=None):
    """Like `project`, but each item is a full message list (e.g. with a system prompt).

    max_batch_tokens caps a batch's padded size: hidden states for every layer and position are
    kept during the forward pass, so long prompts need smaller batches.
    """
    formatted = [format_chat(tokenizer, c) for c in conversations]
    # Batch prompts of similar length together to minimize padding, then restore order.
    if max_batch_tokens is None:
        order = sorted(range(len(formatted)), key=lambda i: len(formatted[i]))
        chunks = list(batches(order, batch_size))
    else:
        lengths = [len(ids) for ids in tokenizer(formatted, add_special_tokens=False)["input_ids"]]
        order = sorted(range(len(formatted)), key=lambda i: lengths[i])
        chunks = list(token_budget_batches(order, lengths, batch_size, max_batch_tokens))
    out = torch.empty(len(formatted), probes.shape[0], n_layers)
    for idx in tqdm(chunks, desc="Projecting"):
        acts = last_token_activations(model, tokenizer, [formatted[i] for i in idx], n_layers)
        out[idx] = torch.einsum("bld,kld->bkl", acts, probes)
    return out
