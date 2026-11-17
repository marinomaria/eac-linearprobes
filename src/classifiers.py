"""
Learned probes on stored last-token activations, after the FIBS paper's LinearEnsemble / MLPEnsemble:
one probe per layer trained with BCE (AdamW, lr 5e-3, batch 32), then an ensemble over the layers whose
final train loss is below 0.6.

Choices not specified by the paper (adjust freely): features are standardized per layer with train-set
mean/std; the ensemble averages logits; the MLP has one hidden layer of 256 units; cosine learning-rate
decay over a fixed number of epochs.
"""
import math

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


def train_layer_probe(x, y, hidden=0, epochs=None, lr=5e-3, batch_size=32, weight_decay=0.01, seed=0, device="cpu"):
    """x: float32 [n, d], y: float32 [n] in {0, 1}. Returns (probe on CPU, final train BCE).

    epochs=None: at least MIN_EPOCHS and at least MIN_STEPS optimizer steps (small datasets underfit otherwise).
    """
    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    probe = LayerProbe(x.shape[1], mean, std, hidden).to(device)
    x, y = x.to(device), y.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    per_epoch = math.ceil(len(x) / batch_size)
    if epochs is None:
        epochs = max(MIN_EPOCHS, math.ceil(MIN_STEPS / per_epoch))
    steps = epochs * per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    loss_fn = nn.BCEWithLogitsLoss()
    probe.train()
    for _ in range(epochs):
        for idx in torch.randperm(len(x), device=device).split(batch_size):
            opt.zero_grad()
            loss_fn(probe(x[idx]), y[idx]).backward()
            opt.step()
            sched.step()
    probe.eval()
    with torch.no_grad():
        train_loss = loss_fn(probe(x), y).item()
    return probe.cpu(), train_loss


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
