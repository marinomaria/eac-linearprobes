# Evaluation Awareness Probes on EvalAwareBench (PoC)

A minimal proof of concept that runs evaluation-awareness linear probes over
[EvalAwareBench](https://huggingface.co/datasets/aisa-group/EvalAwareBench). The benchmark has 51,200 prompts:
200 task variants × 2⁸ combinations of eight evaluation-cue factors (F1–F8). The question is which of those
cues move a model's internal "am I being evaluated?" direction, and by how much.

The probing setup is a simplified version of
[sensitivity-analysis-evaluation-awareness](https://github.com/marinomaria/sensitivity-analysis-evaluation-awareness),
adapted to this benchmark and to newer Qwen models.

## What it does

1. **Train probes.** For each prompt stem in `datasets/contrastive_dataset_dolly.json`, the model sees it twice:
   once under an evaluation system prompt and once under a deployment system prompt. For every layer, the
   probe is the normalized mean difference between the two sets of last-token residual-stream activations.
   This is repeated for every system-prompt variant in `datasets/system_prompts.json`.
2. **Score the benchmark.** Every EvalAwareBench prompt is fed as a single user turn and its last-token
   activation is projected onto **every** variant's and layer's probe. This is a forward pass only, with no
   text generation.
3. **Save.** The per-prompt projections are written to `output/<model>/projections_<subset>.jsonl`.
4. **Analyze.** `scripts/analyze.py` picks a layer and reports how much each factor shifts the projection.

## Models

| Alias | HF repo | Role |
|---|---|---|
| `qwen3.8-27b` | `Qwen/Qwen3.8-27B` | Main model (RunPod, 1× 80GB GPU) |
| `qwen3.5-0.8b` | `Qwen/Qwen3.5-0.8B` | Control: same `qwen3_5` architecture and chat template as Qwen3.8. There is no Qwen3.8 small model |

Any other HF repo id can be passed to `--model` as well.

## Design choices

- **Plain HF transformers, not TransformerLens.** Qwen3.8 uses the `qwen3_5` hybrid linear-attention
  architecture, which TransformerLens does not support. Activations come from `output_hidden_states`:
  `hidden_states[l]` is the residual stream entering block `l`. Prompts are batched with left padding;
  on this architecture, padded and unpadded runs differ by ~1e-6 (relative).
- **All layers are kept, and the layer is chosen at analysis time.** EvalAwareBench has no binary eval/deploy
  label to select a layer against; it has factor configurations instead. One forward pass yields every layer,
  so saving all of them costs nothing. For the same reason there are no classification thresholds or verdicts,
  only raw projections.
- **Layer selection.** `analyze.py` picks the layer whose projection has the highest Spearman correlation with
  the number of exaggerated factors (it can be overridden with `--layer`). It also reports AUROC between
  prompts with all 8 factors exaggerated and prompts with none, but those are only 2/256 of the data, so that
  number is noisy on small samples.
- **System-prompt variants share one benchmark pass.** Nearly all the compute is the forward pass over the
  51,200 benchmark prompts, which does not depend on the system prompt. Training probes per variant is cheap
  (204 stems × 2), and the benchmark activations are projected onto all variants at once. Adding a variant
  means adding an entry to `datasets/system_prompts.json` and rerunning. `run.py` prints the cosine similarity
  between variants' probe directions after training: variants with near-identical directions will score
  the benchmark near-identically.
- **Layer 0** is the token embedding. The last token is the same under both system prompts, so its probe is
  all zeros and its projections are 0.
- **Thinking mode.** The Qwen3.5/3.8 chat template defaults to thinking mode, so formatted prompts end in
  `<think>\n`. This is the same for probe training and benchmark scoring, so both read the same position.

## Local setup

Requires Python ≥ 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run.py --model qwen3.5-0.8b --limit 200       # quick test, ~4 min on an M-series Mac
python scripts/analyze.py output/qwen3.5-0.8b/projections_prompts_n200.jsonl
python scripts/analyze.py output/qwen3.5-0.8b/projections_prompts_n200.jsonl --variant minimal
```

`analyze.py` first prints each variant's best layer and Spearman correlation, then the per-layer and per-factor
tables for one variant (`--variant`, default: the first in `datasets/system_prompts.json`).

Without CUDA, the linear-attention layers fall back to a slow PyTorch implementation. A full local run of all
51,200 prompts would take on the order of 11h on an M-series Mac, so run full jobs on a GPU (see below).

Options:
- `--subset prompts_safety|prompts_capability`: score only one valence (default: all prompts).
- `--limit N`: score a seeded random sample of N prompts.
- `--batch-size N` (default 8).
- `--load-probes output/<model>/probes.pt`: reuse trained probes instead of training again.

## RunPod

Use a pod with one 80GB GPU (H100 or A100). Qwen3.8-27B's bf16 weights are about 54GB.

```bash
cd /workspace
git clone <repo_url> eval-awareness
bash eval-awareness/scripts/runpod_setup.sh
cd eval-awareness
bash scripts/runpod_run.sh qwen3.8-27b --limit 500                  # smoke test first
bash scripts/runpod_run.sh qwen3.8-27b --batch-size 16 --load-probes output/qwen3.8-27b/probes.pt
bash scripts/runpod_run.sh qwen3.5-0.8b --batch-size 32             # full control run
```

The setup script also tries to install `flash-linear-attention` and `causal-conv1d`, the fast kernels for
the linear-attention layers. If either fails to build, runs still work, just more slowly.

Then, on your machine:

```bash
bash scripts/download_artifacts.sh <IP> <PORT>
python scripts/analyze.py runpod-artifacts/qwen3.8-27b/projections_prompts.jsonl
```

## Output format

Each line of `projections_*.jsonl` holds the prompt's `task_id`, `task_name`, `valence`, `factors_varied`,
`num_factors_varied`, `config` (F1–F8 levels) and `projections`: `{variant_name: [one float per layer]}`.
`probes.pt` is a dict `{variant_name: tensor[n_layers, d_model]}`. The prompt text is left
out; rejoin it on (`task_id`, `config`) if needed.

## Credits

- Probe methodology and the contrastive system prompts:
  [evaluation-awareness-probing](https://github.com/Jordine/evaluation-awareness-probing).
- Contrastive prompt stems: taken from
  [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) (CC BY-SA 3.0),
  as built in [sensitivity-analysis-evaluation-awareness](https://github.com/marinomaria/sensitivity-analysis-evaluation-awareness).
- EvalAwareBench: Li et al., [*Decomposing and Measuring Evaluation Awareness*](https://arxiv.org/abs/2605.23055)
  (CC BY-NC 4.0; not to be used as training data).

## Layout

```
run.py                     # Train probes, score EvalAwareBench, save output
src/model.py               # Model loading + last-token residual stream for all layers
src/probe.py               # Probe training (mean difference) and projection
scripts/analyze.py         # Per-layer summary + per-factor effects
scripts/runpod_*.sh        # RunPod setup / run
scripts/download_artifacts.sh
datasets/contrastive_dataset_dolly.json
datasets/system_prompts.json   # Eval/deploy system-prompt variants: [{name, eval, deploy}]
```
