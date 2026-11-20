# Evaluation-awareness probes for the data mix

Goal: score every row of the main data mix (`mix-main`) with every **good** evaluation-awareness probe.

The probes are trained on activations of labeled eval/deploy conversations. A probe is good when it generalizes,
judged on three things:
- the validation cues;
- the zero-tolerance (ZT) cues, a subset of the validation cues;
- sources held out of training.

There are two probe runs. Every probe in them is scored and compared the same way:
- **`all`**: trained on the main training set: Dolly, Needham, aware-bench and WildChat.
- **`verbalized`**: trained only on mix-discovery rows labeled by whether Qwen voiced evaluation awareness in its
  chain of thought. Its label is the model's own judgement, not where a prompt came from, so it is kept apart
  rather than mixed in.

Each run trains every probe kind on its own rows:
- **meandiff**: the normalized mean-difference direction;
- **linear**: logistic regression;
- **mlp**: a small MLP;
- **linear / mlp ensembles**: average over layers.

## Pipeline

```
scripts/prepare_*.py  ->  data/*_rows.jsonl                    training rows (once, CPU)
extract.py            ->  output/<model>/acts/<name>/          last-token activations (GPU; the only model step)
train_probes.py       ->  output/<model>/trained/<run>/        every probe kind, report.csv, probes.pt (CPU)
score.py              ->  <run>/scores_<acts>.jsonl            every probe of a run on stored activations (CPU)
report.py             ->  <run>/probe_report.csv               good = val and ZT AUROC >= 0.75 in every placement
                          <run>/mix-main_good.jsonl            mix-main scores of the good probes (the deliverable)
```

```bash
# GPU, once per dataset (see RunPod)
python extract.py --model qwen3.8-27b --data dolly data/needham_rows.jsonl data/awarebench_rows.jsonl \
    data/wildchat_rows.jsonl --out output/qwen3.8-27b/acts/train
python extract.py --model qwen3.8-27b --data data/verbalized_rows.jsonl
python extract.py --model qwen3.8-27b --data cues
python extract.py --model qwen3.8-27b --data data/mix-main.jsonl

# anywhere, no GPU
A=output/qwen3.8-27b/acts T=output/qwen3.8-27b/trained
bash scripts/train_all.sh $A/train                                  # $T/all + $T/hold-*
python train_probes.py $A/verbalized_rows --name verbalized         # $T/verbalized
for run in all verbalized; do
    python score.py $T/$run $A/cues; python score.py $T/$run $A/mix-main
done
python score.py $T/all $A/verbalized_rows                           # held-out: all on verbalized
python score.py $T/verbalized $A/train                              # held-out: verbalized on everything else
python report.py $T/all $T/verbalized --holdout $T/hold-*
```

Quick local test on the 0.8B model (a few minutes on an M-series Mac):

```bash
python extract.py --model qwen3.5-0.8b --data dolly:40 --out output/qwen3.5-0.8b/acts/dolly_40
python extract.py --model qwen3.5-0.8b --data cues:2
python train_probes.py output/qwen3.5-0.8b/acts/dolly_40 --name all
python score.py output/qwen3.5-0.8b/trained/all output/qwen3.5-0.8b/acts/cues_2
python report.py output/qwen3.5-0.8b/trained/all --cues cues_2
```

`train_probes.py` accepts several activation directories and concatenates their rows. You can therefore add a
newly extracted training source to a run without re-extracting the others.

## Probes

`src/classifiers.py`, all trained by `train_probes.py`:

- **Training rows.** Within a run, every kind sees the same rows with the same weights.
- **Layers.** One probe per stored layer: 6 layers, evenly spaced between 30% and 90% of depth.
- **Probe names:**
  - `meandiff_L<l>`: unit direction mean(eval) − mean(deploy) of the raw activations, using weighted means. The
    logit is the projection minus the projection of the midpoint between the two means. This is the
    original method from [evaluation-awareness-probing](https://github.com/Jordine/evaluation-awareness-probing),
    trained on the same rows as the other probes.
  - `linear_L<l>`: logistic regression on standardized features.
  - `mlp_L<l>`: an MLP with one hidden layer of 256 units.
  - `linear_ensemble`, `mlp_ensemble`: the mean logit over the layers whose train loss is below 0.6. These follow
    the LinearEnsemble and MLPEnsemble of the FIBS paper (Hollinsworth, Spies et al., arXiv 2610.12445).
- **Training settings.** The linear and MLP probes use BCE loss, AdamW with lr 5e-3, batch size 32, and at least
  3 epochs and 2000 steps.
- **Weights.** `--balance source` (the default) gives both labels the same total weight. Within a label, every
  source gets the same weight. This stops a source from dominating because it is large, has long prompts, or has
  only one label.
- **Splits.** Rows are split by group. `--holdout-family` removes whole families from training and reports on
  them separately. It accepts shell patterns, e.g. `'needham:agent:*'`.
- **Report.** `report.csv` gives AUROC and recall at 1% FPR for the val and heldout splits, overall, per source
  and per family, with 95% CIs from a bootstrap over groups.
- **Length baseline.** `length_baseline` is a logistic regression on log prompt length alone. A probe close to it
  is probably reading prompt length.

## Is a probe good?

**Validation cues** (`src/cues.py`) are short hand-written texts with an expected reading. They come from
`datasets/validation_prompts.json` (the validation set) and from the ZT set `datasets/validation_prompts_zt.csv`.

**ZT is a subset of the validation set.** All 25 ZT eval/deploy cues are also among the validation set's 51, with
the same labels. The two gates are therefore nested, not independent: ZT is the stricter check on the core cues,
and validation adds 26 more cues. `report.py` scores validation from `validation_prompts.json` alone, so no cue
is counted twice.

Each cue is shown in three placements:
- **system:** the cue is the system prompt, followed by a Dolly stem.
- **prefix:** the cue is prepended to a stem in the user turn.
- **user:** the cue alone is the user turn.

The system and prefix placements are averaged over 20 stems.

`extract.py --data cues` stores their activations once. After that, any probe run is scored on them offline.

**Held-out sources.** Each probe gets one `held:<source>` column per training source: its AUROC on that source's
rows when they were not used to train it.
- **Sources the run trained on.** `scripts/train_all.sh` trains `all` on the whole training set, then retrains
  the same recipe once per held-out group: `hold-dolly`, `hold-needham`, `hold-awarebench`, `hold-chat` and
  `hold-agent`. WildChat is not held out on its own, because it is deploy-only and has no AUROC there. The value
  comes from the hold-out runs that left the source out, taking the minimum if several did. The `all` probes are
  trained on everything, so this is evidence about the recipe, not about the exact probe.
- **Sources the run never saw.** The value is computed directly from the run's scores on those rows: `all` on
  verbalized, and `verbalized` on everything else.

`report.py` takes several runs (rows `<run>/<probe>`) and writes one row per probe, containing:
- the cue AUROCs per placement: eval- vs deploy-expected cues, for the validation set and for ZT;
- `good`;
- the held-out AUROCs, with their minimum.

A probe is `good` when the validation and ZT AUROCs are ≥ `--bar` (0.75) in every placement. The held-out AUROCs
are reported but not used to decide `good`. `report.py` writes `mix-main_good.jsonl`, merging every run's good probes, with these fields: `id`,
`source` (the mix dataset), `meta` (from the mix) and `scores` (one logit per good probe).

## Data

| rows | script | license / note |
|---|---|---|
| Dolly paired system prompts | built in (`--data dolly`) | stems from databricks-dolly-15k (CC BY-SA 3.0); prompts in `datasets/system_prompts.json` |
| Needham et al. chat + agentic | `scripts/prepare_needham.py` | gated, CC BY-NC-SA, contains GAIA: never in git |
| aware-bench (EvalDetectBench) | `scripts/prepare_awarebench.py` | CC BY-NC-SA, contains GAIA: never in git |
| WildChat (deploy only) | `scripts/prepare_wildchat.py` | ODC-BY |
| mix-discovery, labeled by CoT voicing (own run) | `scripts/prepare_verbalized.py` | from `data/probe_sample_verbalized.csv`; not in the main training set |
| **mix-main** (score only) | built in jlens_evalaware | measurement set: never train on it |

`data/` is gitignored.

The data mix comes from [jlens_evalaware](https://github.com/Spar2026EvalAware/jlens_evalaware):

```bash
uv run evalaware mix --out-dir ../evaluation-awareness-convergence/data
```

The mix contains EvalAwareBench, HarmBench, FORTRESS and MASK prompts. Each row is fed as its full message list.

Only mix-main is the measurement set. mix-discovery may be trained on.
`prepare_verbalized.py` refuses to run if any of its rows overlaps mix-main.

Choices worth knowing:

- **Row format.** `src/rows.py` defines it: `id`, `messages`, `label` (eval/deploy/null), `group`, `family`,
  `source`, `meta`, `tools`.
- **Groups.** Every Needham and aware-bench source dataset is all-eval or all-deploy, so group = source dataset.
  Splits and CIs therefore measure transfer to unseen datasets. aware-bench benchmarks that are also in Needham
  share Needham's group.
- **Where the model is read.** Transcripts end where the model would reply: trailing assistant turns are dropped.
- **System messages.** Later system messages become user messages, because Qwen's template takes only a leading
  one.
- **Long transcripts.** Transcripts over 16k tokens drop messages from the middle.
- **aware-bench tool definitions are dropped.** Only its eval trajectories carry them, so they would give the
  label away.
- **aware-bench exclusions.** `mask` (it is in the mix), `hackaprompt_dataset` (eval in aware-bench, deploy in
  Needham) and tasks already in Needham are excluded.
- **WildChat filtering.** WildChat is filtered heavily and sampled down to 100 rows: English, no toxic content, no
  templates, no jailbreak or test talk.
- **Verbalized labels.** A mix-discovery prompt is `eval` when Qwen voiced evaluation awareness in at least half of
  its chain-of-thought rollouts. The group is the underlying item.

| source | eval | deploy | median tokens eval / deploy |
|---|---|---|---|
| Needham chat | 287 | 213 | 93 / 148 |
| aware-bench chat | 150 | 124 | 204 / 680 |
| WildChat | – | 100 | – / 203 |
| Needham agentic | 250 | 226 | 5.4k / 7.9k |
| aware-bench agentic | 77 | 136 | 13k / 2.8k |
| verbalized (mix-discovery) | 152 | 141 | 261 / 240 |

## Models

| Alias | HF repo | Role |
|---|---|---|
| `qwen3.8-27b` | `Qwen/Qwen3.8-27B` | Main model (RunPod, 1× 80GB GPU) |
| `qwen3.5-27b` | `Qwen/Qwen3.5-27B` | Model of the shared jlens_evalaware measurements |
| `qwen3.5-0.8b` | `Qwen/Qwen3.5-0.8B` | Local tests: same `qwen3_5` architecture and chat template |

- **Library.** Models run with plain HF transformers, because TransformerLens does not support the `qwen3_5`
  hybrid linear-attention architecture.
- **What is stored.** Activations are the residual stream entering a block, at the last token. Pre-hooks capture
  them, and the forward pass stops after the deepest stored layer.
- **Format.** They are stored as bf16 (`src/activations.py`).
- **Thinking mode.** `format_chat` sets `enable_thinking=True`, so every prompt ends in `<think>\n` on both models.

## RunPod

Use one 80GB GPU (H100/A100):

```bash
cd /workspace && git clone https://github.com/marinomaria/eac-linearprobes.git eval-awareness
bash eval-awareness/scripts/runpod_setup.sh && cd eval-awareness
bash scripts/runpod_run.sh extract.py --model qwen3.8-27b --data cues --batch-size 16
```

Copy `data/*.jsonl` to the pod with `scp -P <PORT> data/*.jsonl root@<IP>:/workspace/eval-awareness/data/`. Then,
locally, download the output:

```bash
SSH_KEY=~/.ssh/id_ed25519_team bash scripts/download_artifacts.sh <IP> <PORT>
```

## Layout

```
extract.py                 # rows -> stored activations (model step)
train_probes.py            # meandiff / linear / mlp probes + ensembles, val + held-out metrics
score.py                   # stored activations -> scores from every probe of a run
report.py                  # good probes (val + ZT bar, held-out columns) and their mix-main scores
src/model.py               # model loading, chat formatting
src/activations.py         # extraction and on-disk storage
src/classifiers.py         # probe kinds, ensembles, weights, metrics
src/rows.py                # row format, Dolly paired rows, group split
src/cues.py                # validation / ZT cues as rows
scripts/prepare_*.py       # training sources -> rows
scripts/train_all.sh       # all + hold-out runs (main training set)
scripts/runpod_*.sh, scripts/download_artifacts.sh
datasets/                  # Dolly stems, system prompts, validation and ZT cues
```

## Credits

- Mean-difference probe method and contrastive system prompts:
  [evaluation-awareness-probing](https://github.com/Jordine/evaluation-awareness-probing).
- Contrastive stems: [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
  (CC BY-SA 3.0), as built in
  [sensitivity-analysis-evaluation-awareness](https://github.com/marinomaria/sensitivity-analysis-evaluation-awareness).
- Needham et al., [*Large Language Models Often Know When They Are Being Evaluated*](https://arxiv.org/abs/2505.23836):
  [jjpn2/eval_awareness](https://huggingface.co/datasets/jjpn2/eval_awareness) (CC BY-NC-SA 4.0, gated).
- EvalAwareBench (in the mix): Li et al., [*Decomposing and Measuring Evaluation Awareness*](https://arxiv.org/abs/2605.23055)
  (CC BY-NC 4.0).
