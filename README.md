# MultimodalMU4Sum

Framework for **multimodal aspect-based summarization** (chest X-ray reports) and **machine unlearning**, extending the MU4Sum paper.
Everything that varies between experiments (model, dataset, training method, evaluation) is a YAML preset; code changes are only needed for genuinely new behaviour.

## Install

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -e ".[all]"                              # extras: download, train, eval, dev
pytest                                               # light tests, CPU only, no model downloads
```

## Pipeline

The pipeline has exactly three steps: **preprocessing -> training -> evaluation**. There is no separate fine-tuning and unlearning stage: every model variant is a *training method*.
All scripts take the minimum (model / dataset / method) plus optional `key.path=value` overrides.
Overrides are strict: a typo fails; use `+key=value` to add a brand-new key.

```bash
# 1. preprocessing: download -> process (group by study) -> teacher augmentation (Qwen2-VL-7B-Instruct)
#    skipped automatically when the data is already there
python scripts/run_preprocess.py --dataset iu_xray --model qwen2_vl_7b
python scripts/run_preprocess.py --dataset iu_xray            # download + process only
python scripts/run_preprocess.py --dataset iu_xray --model qwen2_vl_7b --stages augment --force

# 2. training: one method per run (forget aspects are defined by the dataset preset, e.g. forget_aspects=[impression])
python scripts/run_train.py --model qwen2_vl_7b --dataset iu_xray --method original training.lr=1e-4
python scripts/run_train.py --model qwen2_vl_7b --dataset iu_xray --method ga_gd --tag a0.5 training.alpha=0.5
# student different from the teacher: --model is the student, --teacher the preset that ran preprocessing
python scripts/run_train.py --model qwen2_5_vl_3b --teacher qwen2_vl_7b --dataset iu_xray --method original

# 3. evaluation on both train and test in one run: forget / retain views, full / aspect level, per aspect
python scripts/run_evaluate.py --model qwen2_vl_7b --dataset iu_xray --method original
python scripts/run_evaluate.py --model qwen2_vl_7b --dataset iu_xray --method ga_gd --tag a0.5

# paired t-test of a method against Original (also written automatically by `run_evaluate` when Original exists)
python scripts/run_compare.py --model qwen2_vl_7b --dataset iu_xray --method ga_gd --tag a0.5

# copy light artifacts (configs, metrics, predictions) from Drive into the repo
python scripts/sync_results.py --src "<Drive>/MU4Sum/outputs"
```

### Training methods (`--method`, `configs/training/<method>.yaml`)

| Method | Group | Trains on | Starts from |
|---|---|---|---|
| `original` | standard | all aspects (forget + retain) of D | pretrained model, new LoRA |
| `retrain` | standard | retain aspects only (Dr) | pretrained model, new LoRA |
| `finetune` | standard | retain aspects only (Dr) | Original |
| `default` | standard | forget aspects only (Df), target summaries replaced by `training.refusal_text` ("I can't assist you on this") | Original |
| `ga` | unlearning | forget aspects, loss `-NLL_f` | Original |
| `ga_gd` | unlearning | mixed forget + retain batches, `(-sum NLL_f + alpha * sum NLL_r) / batch` | Original |
| `ga_kd` | unlearning | forget + retain, `-NLL_f + alpha * KL(original \|\| current)` on retain answers | Original (also frozen reference) |
| `npo` | unlearning | forget + retain, `alpha * NLL_r + gamma * NPO_f` (beta 0.1, Original as frozen reference) | Original |
| `tw_ga_gd` | unlearning (token-wise) | all aspects in one answer; ascent (`-beta`) on forget-aspect tokens, descent (`alpha`) on the others | Original |
| `tw_ga_kd` | unlearning (token-wise) | as `tw_ga_gd` but KL to Original on the retain tokens | Original (also frozen reference) |

Hyper-parameters follow MU4Sum: original/retrain/finetune 5 epochs at lr 7e-5; `default` 100 steps; `ga` 300 steps at lr 7e-6; the other unlearning methods 1 epoch at lr 7e-5; AdamW, linear schedule with 10% warm-up, batch size 2, LoRA r8/alpha16/dropout 0.05 on q,k,v,o,gate,up,down + lm_head. The checkpoint with the lowest validation loss is kept (`best.json`).

Run `original` first: every other method except `retrain` starts from it (`--original-tag` selects which Original run). `evaluate --method base` scores the pretrained model without adapter.

### Alignment with MU4Sum

- **One multi-aspect answer per study**: the model is asked for a set of aspects in one prompt and answers `"<aspect name>\n<summary>"` blocks joined by blank lines. Sample-wise methods use two views of the same study, forget aspects and retain aspects; the token-wise methods use the full answer with per-token forget/retain labels.
- **Answer-only loss** (deliberate request for this project): every loss, including the KL and NPO terms, is computed only on the answer tokens, per sample as a token mean; prompt, image and padding tokens are masked. Labels are removed before the forward pass, as in the reference code.
- **NPO** follows the reference code: the log-ratio is the difference of per-sample *mean* token log-probabilities (not sums).
- **Evaluation**: greedy decoding, 256 new tokens; ROUGE-1/2/L (no stemmer) and BERTScore F1 (roberta-large, English) at full-answer level and at aspect level (each aspect read back from the answer; a missing aspect scores 0), per view (forget / retain), per aspect, on train and test. Scores are stored per sample (`scores_<split>.json`); `metrics.json` holds the x100 means, `unlearning_summary` (retention vs. Original) and a paired t-test vs. Original.
- **Aspect parsing** matches block names by substring / edit distance <= 3, then falls back to string similarity (`difflib`) instead of the FastText embeddings of the reference code.
- **Not implemented**: LLM-as-judge evaluation and the random / adversarial / DPO / RMU / task-vector baselines.

Stages skip themselves when their artifacts exist and are fresh. If the config that produced them changed, the run raises `StaleArtifactError` instead of silently mixing data; pass `--force` to rebuild.

## Config system (`configs/`)

```
base.yaml            seed, determinism, runtime (environment / data_root / output_root)
models/*.yaml        HF model id, auto_class, dtype, quantization, processor kwargs, LoRA (r, alpha, target_modules), teacher generation + retry params
datasets/*.yaml      source, processor, columns, aspects (+ descriptions), forget_aspects, prompt, splits, augmentation
training/*.yaml      one preset per method: hyper-parameters and method-specific knobs (alpha, beta, refusal_text)
evaluation/*.yaml    metrics, generation settings (splits evaluated: train + test, fixed)
```

`load_config(operation, ...)` knows which preset groups each operation needs (`preprocess`, `train`, `evaluate`), merges `base` + presets + CLI overrides, resolves the machine-specific `runtime` section and freezes the result.
A copy is written, read-only, to the run folder (`config_<operation>.yaml` + `provenance_<operation>.json` with git commit and package versions). Re-running with a *different* effective config in the same folder fails: use `--tag` for a new run. The `runtime` section is excluded from the comparison, so a run started locally can be resumed on Colab.

Outputs are organised hierarchically: `outputs/<model>/<dataset>/<method>[/<tag>]`, with evaluations in `<run>/evaluation/<eval_tag>`.

**Teacher and student are independent.** The hybrid data lives in `processed/<dataset>/hybrid__<teacher>`, where the teacher is the `--model` given to `run_preprocess.py`. `run_train.py` / `run_evaluate.py` / `run_compare.py` read it from the teacher named by `--teacher` (default: the student `--model` itself, so using one model for both needs no flag). When the two differ, outputs go to `outputs/<student>/<dataset>__teacher-<teacher>/...` so runs on different teachers' data never mix. LoRA settings are per model (`lora:` in `configs/models/<model>.yaml`, override with `lora.r=4`), and sit outside the teacher-data fingerprint, so changing them never invalidates the generated data.

### Aspect prompts

Aspects, forget set and prompts live in the dataset YAML; no code refers to a specific dataset or aspect:

```yaml
aspects:
  names: [heart_and_mediastinum, lungs_and_pleura, bones_soft_tissue_and_devices, impression]
  descriptions: {impression: "..."}      # also used as the placeholder of each aspect in the teacher JSON
forget_aspects: [impression]             # unlearning forgets these, retains every other aspect
prompt:
  mode: incremental                      # show aspect descriptions; `names_only` gives names only
  teacher_system: "..."
  teacher_intro: |-                      # role, task, dataset-specific guidance
    ... {image_info} ... '{source_text}' ...
  expansion: {enabled: true, description: "..."}
  student_task: "..."
```

The teacher prompt has the same shape for every dataset: `teacher_intro` + generic output rules + a JSON template.
`{image_info}` is built per study from its images ("Image 1 (Frontal) and Image 2 (Lateral)"); `{source_text}` is the dataset text of the study and is only shown if the placeholder is present. The teacher answers with

```json
{"expanded_text": "...", "aspect_summaries": {"<aspect>": "...", "...": "..."}}
```

`expanded_text` is requested (and stored in the hybrid records, not used for training) only if `prompt.expansion.enabled`. Invalid output triggers up to `generation.max_retries` retries with temperature `min(base_temperature + k * temperature_step, max_temperature)`. Studies that never validate are written to `failures.jsonl` and excluded.

`forget_aspects` sits outside `aspects` on purpose: it is not part of the artifact fingerprint, so changing it never invalidates the teacher data (it does change the config snapshot of training runs, so use `--tag` for a different forget set).

## Extending without touching the code

- **New VLM** (as teacher, student or both): copy `configs/models/qwen2_vl_7b.yaml`, change `model.name` (and `auto_class`, `lora.target_modules` if the layer names differ).
- **New dataset with the same shape** (CSV grouped by study): copy `configs/datasets/iu_xray.yaml`, set `source`, `columns`, `aspects`, `forget_aspects` and the `prompt` texts.
- **New source / format / training method / metric**: add a function decorated with `@DOWNLOADERS.register("x")`, `@PROCESSORS.register("x")`, `@TRAINING_METHODS.register("x")` or `@METRICS.register("x")`, then reference its name from YAML.

## Reproducibility

- `set_seed` seeds python/numpy/torch/CUDA and enables deterministic algorithms (`deterministic: true`).
- Batches are a pure function of `(seed, epoch)`; checkpoints store weights (LoRA adapter only), optimizer, scheduler, progress counters and RNG states. An interrupted run resumed from a checkpoint gives bit-identical weights to an uninterrupted one (verified on a toy model in `tests/test_training.py`).
- Splits are made **per study**, never per image, so frontal/lateral views of an exam cannot leak across splits.

## Google Colab workflow

1. Develop and debug locally (Visual Studio) with `pytest` and small `--stages`/`augmentation.limit=N` runs.
2. Open `notebooks/colab_runner.ipynb` on Colab: it clones the repo, installs it, points `MU4SUM_DATA_ROOT` / `MU4SUM_OUTPUT_ROOT` to Drive and runs the scripts. If the session dies, re-run the same cell: it resumes.
3. Locally run `scripts/sync_results.py` and commit. Checkpoints and weights are never synced or tracked; existing local `config_*` / `provenance_*` snapshots are never overwritten.

## Status / caveats

- Tests cover config, prompts, teacher retry logic, the preprocessing pipeline (with a fake teacher), answer formatting/parsing, the trainer (resume equivalence, validation, best checkpoint), every training method's objective on toy models, and per-sample scoring.
- The code paths that need real models (Qwen2-VL / Qwen2.5-VL-3B loading, LoRA training, generation, BERTScore, token offsets of the real tokenizer) have **not** been executed in CI/local tests: they need a GPU and model downloads. LoRA target names may also match modules of the vision tower.
- The CSV column names in `configs/datasets/iu_xray.yaml` (`dataset.columns`) are a best guess for the Kaggle `masrursabab/iu-chest-x-rays-cleaned` file. If they are wrong, `process` fails and lists the actual columns; fix them in the YAML.
