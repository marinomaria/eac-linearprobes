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
| `qwen3.5-27b` | `Qwen/Qwen3.5-27B` | Model of the shared jlens_evalaware measurements (data mix) |
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
- **Thinking mode.** `format_chat` passes `enable_thinking=True`, so formatted prompts end in `<think>\n` for
  every model, for both probe training and benchmark scoring. The flag matters: Qwen3.8's template defaults to
  thinking mode but Qwen3.5-0.8B's does not, and appends an empty `<think>\n\n</think>\n\n` block instead.
  Runs made before this flag was added read the 0.8B at a different token than the 27B.

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
git clone https://github.com/marinomaria/eac-linearprobes.git eval-awareness
bash eval-awareness/scripts/runpod_setup.sh
cd eval-awareness
bash scripts/runpod_run.sh qwen3.8-27b --limit 500                  # smoke test first
bash scripts/runpod_run.sh qwen3.8-27b --batch-size 16 --load-probes output/qwen3.8-27b/probes.pt
bash scripts/runpod_run.sh qwen3.5-0.8b --batch-size 32             # full control run
python validate.py --model qwen3.8-27b --batch-size 16              # probe sanity checks (see below)
```

The setup script also tries to install `flash-linear-attention` and `causal-conv1d`, the fast kernels for
the linear-attention layers. If either fails to build, runs still work, just more slowly.

Then, on your machine:

```bash
bash scripts/download_artifacts.sh <IP> <PORT>
SSH_KEY=~/.ssh/id_ed25519_team bash scripts/download_artifacts.sh <IP> <PORT>   # key other than ~/.ssh/id_ed25519
python scripts/analyze.py runpod-artifacts/qwen3.8-27b/projections_prompts.jsonl
```

## Data mix

`run_mix.py` scores the shared data mix of [jlens_evalaware](https://github.com/Spar2026EvalAware/jlens_evalaware)
(`src/evalaware/mix.py`): EvalAwareBench, bare HarmBench, FORTRESS and MASK prompts, split into a main
(1,890 prompts) and a discovery set (1,510). Probes are trained (or loaded) exactly as in `run.py`. Each prompt is
fed as its full message list, so MASK prompts keep their own system prompt (and doubling_down its scripted first
exchange). Build the mix in jlens_evalaware and copy it into `data/` (gitignored):

```bash
# in jlens_evalaware
uv run evalaware mix --out-dir ../evaluation-awareness-convergence/data   # mix-main.jsonl, mix-discovery.jsonl

# here
python run_mix.py --model qwen3.5-0.8b --limit 50                         # quick test, both sets
SCRIPT=run_mix.py bash scripts/runpod_run.sh qwen3.8-27b --batch-size 16 \
    --load-probes output/qwen3.8-27b/probes.pt                             # on RunPod
```

On a pod without the mix, copy it over with
`scp -P <PORT> data/mix-*.jsonl root@<IP>:/workspace/eval-awareness/data/` (or point `--data` at the files on
the shared volume, `/workspace/code/<name>/data/mix-*.jsonl`).

Mix prompts reach ~17k tokens, and the forward pass keeps every layer's hidden states, so batches are also capped
at `--max-batch-tokens` padded tokens (default 16384: about 11 GB of hidden states for the 27B). Output goes to
`output/<model>/projections_mix-{main,discovery}.jsonl`, one line per prompt with `id`, `dataset`, `meta` (as in
the mix: `set`, `unit`, and the source's fields such as `factors`, `condition`, `archetype`) and `projections`.

## Probe sanity checks

`validate.py` reuses `output/<model>/probes.pt` and projects short hand-written cues from
`datasets/validation_prompts.json`: single words ("evaluation", "real user", "chicken", "wardrobe"), short
sentences, negations ("This is not a test."), paraphrases unlike the training prompts, eval-style formatting
without eval keywords, and real-user requests that contain eval keywords ("I have my driving test tomorrow").
Each cue is shown as the system prompt before a contrastive stem (the training shape), prepended to a stem in
the user turn (the benchmark shape), and alone as the user turn. The system/prefix placements average over
`--n-stems` stems (default 20; these are training stems). The variants' training prompts are added as anchors.

```bash
python validate.py --model qwen3.5-0.8b --n-stems 4                 # ~1.5 min on an M-series Mac
python scripts/analyze_validation.py output/qwen3.5-0.8b/validation.jsonl
```

`analyze_validation.py` scores every cue on a 0 (deploy anchors) to 1 (eval anchors) scale per placement and
reports AUROC between eval- and deploy-expected cues. A probe that tracks evaluation awareness rather than
keywords should leave neutral cues near the no-cue baseline (`none`) and flip on negations.

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
run_mix.py                 # Score the jlens_evalaware data mix (main + discovery sets)
validate.py                # Project hand-written sanity-check cues onto saved probes
src/model.py               # Model loading + last-token residual stream for all layers
src/probe.py               # Probe training (mean difference) and projection
scripts/analyze.py         # Per-layer summary + per-factor effects
scripts/analyze_validation.py  # Sanity-check scores per cue category and placement
scripts/runpod_*.sh        # RunPod setup / run
scripts/download_artifacts.sh
datasets/contrastive_dataset_dolly.json
datasets/system_prompts.json   # Eval/deploy system-prompt variants: [{name, eval, deploy}]
datasets/validation_prompts.json  # Sanity-check cues: [{category, expected, text}]
```
