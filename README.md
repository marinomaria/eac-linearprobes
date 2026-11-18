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

The cues come from `datasets/validation_prompts.json` and the zero-tolerance set
`datasets/validation_prompts_zt.csv` (`--prompts` to change; each output row records its file as `set`).

```bash
python validate.py --model qwen3.5-0.8b --n-stems 4                 # ~1.5 min on an M-series Mac
python scripts/analyze_validation.py output/qwen3.5-0.8b/validation.jsonl
python scripts/analyze_validation.py output/qwen3.5-0.8b/validation.jsonl --set validation_prompts_zt
```

`--trained <dir> [<dir> ...]` scores the cues with every probe in `train_probes.py` output directories
(`linear_L*`, `mlp_L*`, `linear_ensemble`, `mlp_ensemble`), from one extraction, and writes
`<dir>/validation.jsonl` for each; with `--probes` too, the direction probes run in the same model load:

```bash
python validate.py --model qwen3.8-27b --batch-size 16 --probes output/qwen3.8-27b/probes.pt \
    --trained output/qwen3.8-27b/trained/all output/qwen3.8-27b/trained/hold-agent
python scripts/analyze_validation.py output/qwen3.8-27b/trained/all/validation.jsonl --variant mlp_ensemble
```

To score new probes on the cues without a GPU, store the cue activations once (`--data cues`: the same cues,
placements and 20 stems, as unlabeled rows), then score any trained directory from them; this gives the same
`validation.jsonl` as running the model:

```bash
python extract.py --model qwen3.8-27b --data cues                                   # on a GPU, once
python validate.py --acts output/qwen3.8-27b/acts/cues --trained output/qwen3.8-27b/trained/<name>
```

`analyze_validation.py` scores every cue on a 0 (deploy anchors) to 1 (eval anchors) scale per placement and
reports AUROC between eval- and deploy-expected cues. A probe that tracks evaluation awareness rather than
keywords should leave neutral cues near the no-cue baseline (`none`) and flip on negations.

## Learned probes (logistic / MLP ensembles)

An alternative to the mean-difference probes, after the FIBS paper's LinearEnsemble and MLPEnsemble
(Hollinsworth, Spies et al., arXiv 2610.12445). Training data is a set of labeled conversation rows, so it is
not limited to the paired stem design. Activations are extracted once and stored, then probes are trained any
number of times.

1. **Rows** (`src/rows.py`): one JSON object per line with `id`, `messages`, `label` (`eval`/`deploy`, or null
   for score-only rows), `group` (rows that must stay on the same side of a split, e.g. both versions of a
   stem), `family` (the kind of data, for leave-family-out evaluation), `source` (the dataset, for loss
   balancing) and optional `tools` (agentic transcripts). `--data dolly` rebuilds the current paired design in
   this format (family `sysprompt:<variant>`, group = stem, source `dolly`).
2. **Extract** (`extract.py`): last-token activations for 6 layers between 30% and 90% of depth, stored as
   bf16 in `output/<model>/acts/<name>/` (rows x 6 x d_model x 2 bytes; ~9 GB for 150k rows on the 27B).
   Several `--data` sources go into one directory. Only the stored layers' last token is kept during the
   forward pass, so long transcripts fit in memory.
3. **Train** (`train_probes.py`): per layer, a logistic and an MLP probe (BCE, AdamW lr 5e-3, batch 32), plus
   `linear_ensemble` / `mlp_ensemble` (mean logit over layers with train loss < 0.6). Validation split by
   group; `--holdout-family` holds out whole families (patterns such as `'needham:agent:*'` work).
   `--balance source` (default) gives every (source, label) cell the same total loss weight. Reports AUROC and
   recall at 1% FPR with 95% CIs from a bootstrap over groups, overall, per source and per family
   (`report.csv`), next to `length_baseline`, a logistic regression on log prompt length alone.
4. **Use**: `export.pt` holds the logistic probes as `{"logreg": [n_layers, d_model]}`, so it works with
   `run.py`/`run_mix.py --load-probes` and `validate.py --probes`; `scripts/add_ensemble.py` then adds the
   `logreg_ensemble` variant to their output. MLP probes need activations: `extract.py` the set, then `score.py`.

```bash
python extract.py --model qwen3.5-0.8b --data dolly --limit 40                    # ~1 min on an M-series Mac
python train_probes.py output/qwen3.5-0.8b/acts/dolly_n40 --holdout-family sysprompt:minimal
python validate.py --model qwen3.5-0.8b --probes output/qwen3.5-0.8b/trained/dolly_n40/export.pt --n-stems 4
python scripts/add_ensemble.py output/qwen3.5-0.8b/validation.jsonl output/qwen3.5-0.8b/trained/dolly_n40/ensemble.json
python scripts/analyze_validation.py output/qwen3.5-0.8b/validation_ens.jsonl --variant logreg_ensemble

# Dolly + Needham et al. (see below)
python scripts/prepare_needham.py <dir>/dataset.json                               # -> data/needham_rows.jsonl
python extract.py --model qwen3.8-27b --data dolly data/needham_rows.jsonl --out output/qwen3.8-27b/acts/dolly+needham
python train_probes.py output/qwen3.8-27b/acts/dolly+needham --holdout-family 'needham:agent:*' --name hold-agent

# + aware-bench and WildChat (see below; run after prepare_needham, which both dedup against)
python scripts/prepare_awarebench.py                                              # -> data/awarebench_rows.jsonl
python scripts/prepare_wildchat.py                                                # -> data/wildchat_rows.jsonl
python extract.py --model qwen3.8-27b --data dolly data/needham_rows.jsonl data/awarebench_rows.jsonl \
    data/wildchat_rows.jsonl --out output/qwen3.8-27b/acts/dolly+needham+awarebench+wildchat

# + mix-discovery rows labeled by the model's own CoT voicing (see below)
python scripts/prepare_verbalized.py                                              # -> data/verbalized_rows.jsonl
python extract.py --model qwen3.8-27b --data dolly data/needham_rows.jsonl data/awarebench_rows.jsonl \
    data/wildchat_rows.jsonl data/verbalized_rows.jsonl --out output/qwen3.8-27b/acts/full

python extract.py --model qwen3.8-27b --data data/mix-main.jsonl                  # score-only rows
python score.py output/qwen3.8-27b/trained/<name> output/qwen3.8-27b/acts/mix-main
```

Never train on mix-main: it is the measurement set, score it only. mix-discovery may be trained on (as in
`prepare_verbalized.py`), and so may EvalAwareBench for probes (its license restricts training LLMs, not classifiers).
With small training sets the MLP probes overfit (near-zero train loss, so the 0.6 filter keeps every layer);
compare them with the linear probes on held-out families before relying on them.

### Needham et al. data

[jjpn2/eval_awareness](https://huggingface.co/datasets/jjpn2/eval_awareness) (Needham et al.,
[arXiv 2505.23836](https://arxiv.org/abs/2505.23836); gated, CC BY-NC-SA, contains GAIA, so keep it and
`data/needham_rows.jsonl` out of git; `data/` is gitignored): 976 real or evaluation transcripts from 61 source
datasets, half chat and half agentic. `scripts/prepare_needham.py` converts the decrypted `dataset.json`
(download and password: see its docstring). Choices worth knowing:

- Every source dataset is all-eval or all-deploy, so **group = source dataset**: the val split and the CIs measure
  transfer to unseen datasets; a per-transcript split would let the probe recognise a dataset's template.
- Family `needham:{chat|agent}:<dataset>`, source `needham_chat` / `needham_agent`.
- Trailing assistant turns are dropped (read where the model would reply, like the Dolly prompts); later system
  messages become user messages (Qwen's template only takes a leading one); transcripts over 16k tokens drop
  middle messages (127 agentic ones).

Lengths (Qwen tokens; medians): Dolly ~80 with no eval/deploy difference; Needham chat 93 eval / 148 deploy;
agentic 5.7k eval / 11k deploy (up to 85k before truncation). Within Needham, length alone separates the labels
(AUROC ~0.6, deploy longer), and the sources differ by orders of magnitude, so:

- `--balance source` weights Dolly, Needham chat and Needham agentic equally, with labels balanced within each,
  so neither the 1224 Dolly rows nor the long transcripts dominate.
- Compare every probe with `length_baseline` per source; a probe close to it is likely reading length.
- Hold out a source (`'needham:agent:*'`, `'needham:chat:*'`, `'sysprompt:*'`) to see whether one kind of data
  transfers to the other.
- Extraction cost is set by the agentic transcripts (~4M tokens after truncation, vs ~0.1M for Dolly); with
  `--max-batch-tokens 16384` they run one per batch.

### aware-bench and WildChat data

[el7982/aware-bench](https://huggingface.co/datasets/el7982/aware-bench) (EvalDetectBench; CC BY-NC-SA, contains
GAIA level 2, keep rows out of git; not the score-only EvalAwareBench): `scripts/prepare_awarebench.py` takes the
Qwen 3.5 generator's `all.json` (41 eval benchmarks + 13 deployment sources) and converts it like Needham's.
- Tool definitions are dropped by default: only eval trajectories carry them, so they would give the label away.
- Dropped: `mask` (in the data mix), `hackaprompt_dataset` (eval here, deploy in Needham), and any transcript whose
  system prompt + first user message is already in Needham's rows (~260, many sources are shared).
- Benchmarks also in Needham get Needham's group (`GROUP_ALIASES` for renamed ones), so a group split never puts
  one benchmark on both sides. Family `awarebench:{chat|agent}:<category>`, source `awarebench_chat` / `_agent`.

[allenai/WildChat](https://huggingface.co/datasets/allenai/WildChat) (ODC-BY): real ChatGPT conversations, deploy
only. `scripts/prepare_wildchat.py` filters one shard heavily (English, not toxic/redacted/flagged, at most 4 turns,
no repeated templates or bots, no jailbreak or test talk, nothing already in Needham/aware-bench) and samples 100
rows: enough to bring chat deploy level with chat eval. Group `wildchat` (one dataset), source `wildchat`.

| source | eval | deploy | median tokens eval / deploy |
|---|---|---|---|
| Needham chat | 287 | 213 | 93 / 148 |
| aware-bench chat | 150 | 124 | 204 / 680 |
| WildChat | – | 100 | – / 203 |
| Needham agentic | 250 | 226 | 5.4k / 7.9k |
| aware-bench agentic | 77 | 136 | 13k / 2.8k |

aware-bench's agentic eval transcripts are longer than its deploy ones, the opposite of Needham, so length alone
no longer points one way. `--balance source` gives both labels equal weight and splits each label's weight equally
over its sources, so deploy-only WildChat does not tip the balance. Together they add ~2M tokens to extraction.

### Verbalized mix-discovery rows

`data/probe_sample_verbalized.csv`: 293 mix-discovery prompts (EvalAwareBench 213, MASK 60, FORTRESS 10, HarmBench 10)
with how often Qwen voiced evaluation awareness in its CoT over 4 rollouts. `scripts/prepare_verbalized.py` labels
`voicing_rate >= 0.5` as eval (152) and the rest deploy (141): the label is the model's own judgement, not where the
prompt came from. It exits if any row's id, messages or item (EvalAwareBench family, MASK/FORTRESS/HarmBench item)
is in mix-main. Group = the item, so cue/stratum variants of one task stay together (117 groups); family
`verbalized:<dataset>:<stratum>`, source `verbalized` (one source, so the 10-row datasets are not upweighted).
Median tokens eval / deploy 261 / 240.

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
- Needham et al., [*Large Language Models Often Know When They Are Being Evaluated*](https://arxiv.org/abs/2505.23836):
  [jjpn2/eval_awareness](https://huggingface.co/datasets/jjpn2/eval_awareness) (CC BY-NC-SA 4.0, gated).
- EvalAwareBench: Li et al., [*Decomposing and Measuring Evaluation Awareness*](https://arxiv.org/abs/2605.23055)
  (CC BY-NC 4.0; not to be used as training data).

## Layout

```
run.py                     # Train probes, score EvalAwareBench, save output
run_mix.py                 # Score the jlens_evalaware data mix (main + discovery sets)
validate.py                # Project hand-written sanity-check cues onto saved probes
extract.py                 # Store last-token activations of labeled/unlabeled rows for learned probes
train_probes.py            # Train logistic/MLP probes + ensembles, leave-family-out metrics
score.py                   # Score stored activations with trained probes
src/model.py               # Model loading + last-token residual stream for all layers
src/probe.py               # Probe training (mean difference) and projection
src/rows.py                # Labeled-rows format, Dolly pairs as rows, group split
src/activations.py         # Activation extraction and on-disk storage
src/classifiers.py         # Per-layer logistic/MLP probes, ensembles, metrics
scripts/analyze.py         # Per-layer summary + per-factor effects
scripts/analyze_validation.py  # Sanity-check scores per cue category and placement
scripts/add_ensemble.py   # Add linear-ensemble scores to projections from an export.pt
scripts/prepare_needham.py # Needham et al. transcripts -> labeled rows
scripts/prepare_awarebench.py  # aware-bench (EvalDetectBench) trajectories -> labeled rows
scripts/prepare_wildchat.py    # Filtered WildChat sample -> deploy rows
scripts/prepare_verbalized.py  # mix-discovery prompts labeled by CoT voicing rate -> labeled rows
scripts/runpod_*.sh        # RunPod setup / run
scripts/download_artifacts.sh
datasets/contrastive_dataset_dolly.json
datasets/system_prompts.json   # Eval/deploy system-prompt variants: [{name, eval, deploy}]
datasets/validation_prompts.json  # Sanity-check cues: [{category, expected, text}]
```
