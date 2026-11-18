"""
Learned probes on stored last-token activations, after the FIBS paper's LinearEnsemble / MLPEnsemble:
one probe per layer trained with BCE (AdamW, lr 5e-3, batch 32), then an ensemble over the layers whose
final train loss is below 0.6.

Choices not specified by the paper (adjust freely): features are standardized per layer with train-set
mean/std; the ensemble averages logits; the MLP has one hidden layer of 256 units; cosine learning-rate
decay over a fixed number of epochs.
"""
import collections
import math
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

KEEP_LOSS = 0.6
MIN_EPOCHS, MIN_STEPS = 3, 2000


class LayerProbe(nn.Module):
    """Standardize, then linear (hidden=0) or one-hidden-layer MLP. Outputs a logit."""

    def __init__(self, d, mean, std, hidden=0):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.net = nn.Linear(d, 1) if hidden == 0 else nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                                                     nn.Linear(hidden, 1))
        self.hidden = hidden

    def forward(self, x):
        return self.net((x - self.mean) / self.std).squeeze(-1)

    def direction(self):
        """For a linear probe: (w, c) such that logit = w . x + c on raw activations."""
        assert self.hidden == 0
        w = self.net.weight[0] / self.std
        return w, self.net.bias[0] - (w * self.mean).sum()


def train_layer_probe(x, y, hidden=0, epochs=None, lr=5e-3, batch_size=32, weight_decay=0.01, seed=0, device="cpu",
                      weights=None):
    """x: float32 [n, d], y: float32 [n] in {0, 1}. Returns (probe on CPU, final train BCE).

    weights: optional per-row loss weights [n] (see balance_weights); the reported BCE is weighted too.
    epochs=None: at least MIN_EPOCHS and at least MIN_STEPS optimizer steps (small datasets underfit otherwise).
    """
    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    probe = LayerProbe(x.shape[1], mean, std, hidden).to(device)
    w = torch.ones_like(y) if weights is None else weights / weights.mean()
    x, y, w = x.to(device), y.to(device), w.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    per_epoch = math.ceil(len(x) / batch_size)
    if epochs is None:
        epochs = max(MIN_EPOCHS, math.ceil(MIN_STEPS / per_epoch))
    steps = epochs * per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def loss_fn(idx):
        return (bce(probe(x[idx]), y[idx]) * w[idx]).sum() / w[idx].sum()

    probe.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(x), device=device).split(batch_size):
            opt.zero_grad()
            loss_fn(idx).backward()
            opt.step()
            sched.step()
    probe.eval()
    with torch.no_grad():
        train_loss = loss_fn(torch.arange(len(x), device=device)).item()
    return probe.cpu(), train_loss


def load_trained(trained_dir):
    """probes.pt written by train_probes.py -> (saved dict, {kind: [LayerProbe per stored layer]})."""
    saved = torch.load(os.path.join(trained_dir, "probes.pt"))
    probes = {}
    for kind, hidden in (("linear", 0), ("mlp", saved["mlp_hidden"])):
        probes[kind] = []
        for state in saved["state"][kind]:
            probe = LayerProbe(state["mean"].shape[0], state["mean"], state["std"], hidden)
            probe.load_state_dict(state)
            probes[kind].append(probe.eval())
    return saved, probes


@torch.no_grad()
def score_all(saved, probes, layer_x, device="cpu", chunk=4096):
    """layer_x(layer) -> float32 [n, d]. Returns {probe name: logits [n]} for every per-layer probe and the
    linear / mlp ensembles (mean logit over the kept layers), as reported by train_probes.py."""
    layers = saved["info"]["layers"]
    scores = {}
    for j, layer in enumerate(layers):
        x = layer_x(layer)
        for kind, ps in probes.items():
            p = ps[j].to(device)
            scores[f"{kind}_L{layer}"] = torch.cat([p(x[i:i + chunk].to(device)).cpu()
                                                    for i in range(0, len(x), chunk)]).numpy()
            ps[j] = p.cpu()
    for kind, kept in saved["kept"].items():
        scores[f"{kind}_ensemble"] = np.mean([scores[f"{kind}_L{layers[j]}"] for j in kept], axis=0)
    return scores


def balance_weights(y, sources, mode="source"):
    """Per-row weights. "label": every label gets the same total weight. "source": every label gets the same total
    weight, split equally over the sources that have that label, so a one-label source (e.g. WildChat, deploy only)
    does not tip the label balance; with every source having both labels this is equal weight per (source, label)
    cell. "none": all ones. Mean 1."""
    if mode == "none":
        return np.ones(len(y))
    keys = list(zip(sources, y, strict=True)) if mode == "source" else list(y)
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    if mode == "source":
        n_sources = collections.Counter(label for _, label in counts)
        w = np.array([1 / (counts[k] * n_sources[k[1]]) for k in keys])
    else:
        w = np.array([1 / counts[k] for k in keys])
    return w / w.mean()


def kept_layers(train_losses):
    """Indices of layers with train loss below KEEP_LOSS; all layers if none qualifies."""
    keep = [j for j, loss in enumerate(train_losses) if loss < KEEP_LOSS]
    return keep or list(range(len(train_losses)))


def recall_at_fpr(y, s, fpr=0.01):
    """Recall on positives at the threshold giving `fpr` on these same negatives (in-sample threshold)."""
    neg, pos = s[y == 0], s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return np.nan
    return float((pos > np.quantile(neg, 1 - fpr)).mean())


def metrics(y, s, groups, n_boot=1000, seed=0):
    """AUROC and recall@1%FPR with 95% CIs from a bootstrap over groups."""
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    out = {"n": len(y), "auroc": roc_auc_score(y, s) if len(set(y)) == 2 else np.nan,
           "recall@1%fpr": recall_at_fpr(y, s)}
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    boots = {"auroc": [], "recall@1%fpr": []}
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        if len(set(y[idx])) < 2:
            continue
        boots["auroc"].append(roc_auc_score(y[idx], s[idx]))
        boots["recall@1%fpr"].append(recall_at_fpr(y[idx], s[idx]))
    for k, v in boots.items():
        lo, hi = np.nanpercentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)
        out[f"{k}_lo"], out[f"{k}_hi"] = lo, hi
    return out
