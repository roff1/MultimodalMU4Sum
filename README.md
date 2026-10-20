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
# 1. preprocessing: download -> process (raw data -> records) -> teacher augmentation (Qwen2-VL-7B-Instruct)
#    skipped automatically when the data is already there; augment runs only if `augmentation.enabled` (dataset preset)
python scripts/run_preprocess.py --dataset iu_xray --model qwen2_vl_7b
python scripts/run_preprocess.py --dataset iu_xray --stages download process   # no teacher needed
python scripts/run_preprocess.py --dataset iu_xray --model qwen2_vl_7b --stages augment --force

# 2. training: one method per run (forget aspects are defined by the dataset preset, e.g. forget_aspects=[impression])
python scripts/run_train.py --model qwen2_vl_7b --dataset iu_xray --method original training.lr=1e-4
python scripts/run_train.py --model qwen2_vl_7b --dataset iu_xray --method ga_gd --tag a0.5 training.alpha=0.5
# student different from the teacher: just use another --model, the processed data does not depend on it
python scripts/run_train.py --model qwen2_5_vl_3b --dataset iu_xray --method original

# 3. evaluation on both train and test in one run: forget / retain views, full / aspect level, per aspect
python scripts/run_evaluate.py --model qwen2_vl_7b --dataset iu_xray --method original
python scripts/run_evaluate.py --model qwen2_vl_7b --dataset iu_xray --method ga_gd --tag a0.5

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

- **One multi-aspect answer per record**: the model is asked for a set of aspects in one prompt and answers with the JSON `{"aspect_summaries": {"<aspect name>": "<summary>", ...}}`, the structure the teacher produces. Sample-wise methods use two views of the same record, forget aspects and retain aspects; the token-wise methods use the full answer with per-token forget/retain labels.
- **Answer-only loss** (deliberate request for this project): every loss, including the KL and NPO terms, is computed only on the answer tokens, per sample as a token mean; prompt, image and padding tokens are masked. Labels are removed before the forward pass, as in the reference code.
- **NPO** follows the reference code: the log-ratio is the difference of per-sample *mean* token log-probabilities (not sums).
- **Evaluation**: greedy decoding, 256 new tokens; ROUGE-1/2/L (no stemmer) and BERTScore F1 (roberta-large, English) at full-answer level and at aspect level (each aspect read back from the answer; a missing aspect scores 0), per view (forget / retain), per aspect, on train and test. Scores are stored per sample (`scores_<split>.json`); `metrics.json` holds the x100 means, `unlearning_summary` (retention vs. Original) and a paired t-test vs. Original.
- **Aspect parsing** reads the `"name": "summary"` entries of the generated text (it also works on truncated or slightly malformed JSON) and matches names by substring / edit distance <= 3, then falls back to string similarity (`difflib`) instead of the FastText embeddings of the reference code.
- **Not implemented**: LLM-as-judge evaluation and the random / adversarial / DPO / RMU / task-vector baselines.

Stages skip themselves when their artifacts exist and are fresh. If the config that produced them changed, the run raises `StaleArtifactError` instead of silently mixing data; pass `--force` to rebuild.

## Config system (`configs/`)

```
base.yaml            seed, determinism, runtime (environment / data_root / output_root)
models/*.yaml        HF model id, auto_class, dtype, quantization, processor kwargs, LoRA (r, alpha, target_modules), teacher generation + retry params, student_prompt
datasets/*.yaml      source, processor, csv/images locations, columns, filtering, splits, aspects (+ descriptions), forget_aspects, teacher prompt, augmentation
training/*.yaml      one preset per method: hyper-parameters and method-specific knobs (alpha, beta, refusal_text)
evaluation/*.yaml    metrics, generation settings (splits evaluated: train + test, fixed)
```

`load_config(operation, ...)` knows which preset groups each operation needs (`preprocess`, `train`, `evaluate`), merges `base` + presets + CLI overrides, resolves the machine-specific `runtime` section and freezes the result.
A copy is written, read-only, to the run folder (`config_<operation>.yaml` + `provenance_<operation>.json` with git commit and package versions). Re-running with a *different* effective config in the same folder fails: use `--tag` for a new run. The `runtime` section is excluded from the comparison, so a run started locally can be resumed on Colab.

Outputs are organised hierarchically: `outputs/<model>/<dataset>/<method>[/<tag>]`, with evaluations in `<run>/evaluation/<eval_tag>`.

**Processed data and teacher.** `data/raw/<dataset>` is the downloaded dataset, untouched. Every later stage writes into one folder, `data/processed/<dataset>`, and the *last* stage of the dataset's pipeline (`augment`, or `process` when `augmentation.enabled: false`) writes the final data under the standard names that training and evaluation read; earlier stages prefix their files with the stage name:

```
data/processed/<dataset>/
  process_records.jsonl   process_manifest.json     intermediate (only when augment follows)
  records.jsonl  manifest.json  progress.json  failures.jsonl     final data
```

Training and evaluation only read `records.jsonl` (grouped by its `split` field) and `manifest.json`, so they do not know (or care) which stages produced them. Only one final dataset exists per dataset name: re-running with another teacher (or any config change that affects the data) raises `StaleArtifactError` until `--force`, which rebuilds the final files and keeps the intermediate ones. The teacher is recorded in `manifest.json`. The student is just `--model`: LoRA settings are per model (`lora:` in `configs/models/<model>.yaml`, override with `lora.r=4`) and sit outside the data fingerprint, so changing them never invalidates the generated data.

### Records and prompts

Whatever the dataset, `process` produces one **record** per sample (plus its `split`) and `augment` turns it into a **hybrid record** by adding the teacher's `aspect_summaries`. A dataset whose processor already returns `aspect_summaries` sets `augmentation.enabled: false` and skips `augment`; the records have the same shape either way:

```json
{"id": "...", "image_types": ["Frontal", "Lateral"], "image_paths": ["resized_images/320/a.png", "..."],
 "text": "<the original record text>",
 "aspect_summaries": {"<aspect 1>": "summary 1", "...": "..."}}
```

`image_types` has one entry per image (`""` when the dataset has no notion of image type) and `image_paths` are relative to the raw data root. Nothing else in the code refers to a specific dataset or aspect.

Both the teacher (augmentation) and the student (training / evaluation) prompts are text templates filled from one record: `prompt(text, image_types, image_paths)`.
The student answers with `{"aspect_summaries": {"<aspect 1>": "summary 1", ...}}` (only the aspects asked in that prompt), the same structure as the teacher.

| Template | Where | Placeholders |
|---|---|---|
| `prompt.teacher_prompt` | dataset preset | `{text}` `{image_types}` `{num_images}` `{json_example}` (required) |
| `student_prompt` | model preset | `{text}` `{image_types}` `{num_images}` `{json_example}` (required) |

- `{image_types}` is the comma-separated list of the image types. It is only available when the dataset sets `prompt.image_types: true`; otherwise every line containing it is dropped, so keep it on a line of its own.
- `{json_example}` (teacher and student) is built from the preset in the same way: `{"aspect_summaries": {aspect name: description}}`, with `aspects.names` / `aspects.descriptions`. For the student only the aspects asked in that prompt are listed. Put it at the end of the prompt.
- A placeholder that does not exist fails loudly instead of being silently left in the text.

The teacher answers with

```json
{"aspect_summaries": {"<aspect>": "...", "...": "..."}}
```

The teacher never rewrites the record text. Invalid output triggers up to `generation.max_retries` retries with temperature `min(base_temperature + k * temperature_step, max_temperature)`. Records that never validate are written to `failures.jsonl` and excluded.

`forget_aspects` sits outside `aspects` on purpose: it is not part of the artifact fingerprint, so changing it never invalidates the teacher data (it does change the config snapshot of training runs, so use `--tag` for a different forget set). The same holds for `student_prompt`, which lives in the model preset.

### Dataset filtering

`dataset.filtering` is applied by the processor (through `common.rejection_reason`):

```yaml
filtering:
  min_text_chars: 10            # drop records with a shorter text
  images:
    mode: "none"                # none | count | count_and_type
    count: 2                    # mode "count": only records with exactly this many images
    types: {Frontal: 1, Lateral: 1}   # mode "count_and_type": only records with exactly these images (type -> how many)
```

`none` keeps records with any number of images of any type; `count` keeps records with a given number of images; `count_and_type` keeps records whose images are exactly the listed types (for IU X-Ray, `Frontal` + `Lateral` pairs).

## Extending the framework

Every extension point is a name registered in a `Registry` (`mu4sum.utils.registry`) and referenced from a YAML preset. Run `pytest` after each of the steps below.

### Add a dataset

1. **Preset** `configs/datasets/<name>.yaml`: copy `iu_xray.yaml` and set
   - `dataset.name` and `dataset.source` (`type: kaggle | huggingface | url | local` plus `handle` / `path`);
   - `dataset.processor: "<name>"` (the name of the processor of step 2);
   - the locations inside the downloaded data (`csv_path`, `images.path`, ...) and whatever the processor needs to read the raw files (`columns`, ...): these keys are free, only your processor reads them;
   - `filtering` and `splits` (fractions summing to 1, applied per record);
   - `aspects.names` / `aspects.descriptions`, `forget_aspects` (a non-empty subset of the names, not all of them);
   - `prompt.image_types` (mention the image types or not) and `prompt.teacher_prompt` (see *Records and prompts*; it must contain `{json_example}`);
   - `augmentation.enabled`: `true` if a teacher must generate the aspect summaries, `false` if the processor returns them itself (then the teacher prompt is unused).
2. **Processor** `src/mu4sum/preprocessing/processors/<name>.py` (modules of the package are imported automatically):

   ```python
   from mu4sum.preprocessing.processors.common import PROCESSORS, rejection_reason

   @PROCESSORS.register("<name>")
   def my_dataset(ds, raw_root):          # ds = cfg.dataset, raw_root = downloaded data folder (Path)
       records = []
       for sample in ...:                 # read the raw format
           record = {"id": ..., "image_types": [...], "image_paths": [...], "text": ...}
           if rejection_reason(record, ds.filtering) is None:
               records.append(record)
       return records
   ```

   The contract: `id` is a unique string, `image_types` and `image_paths` have the same length, `image_paths` are relative to `raw_root` (posix separators), `text` is the textual input. With `augmentation.enabled: false` every record must also carry `aspect_summaries` (one entry per `aspects.names`). The pipeline checks it (`check_records`) and adds the `split`.
   Helpers used by one dataset only stay in its module; helpers useful to several processors go in `processors/common.py`.
3. **Run** `python scripts/run_preprocess.py --dataset <name> --model <teacher>`. A wrong path or column raises an error naming what was looked for. Start with `augmentation.limit=5` to check the teacher prompt and the output.
4. **Test**: add a tiny fake raw folder like `tests/conftest.py` does and check the records.

A new download source type is a function registered with `@DOWNLOADERS.register("<type>")` in `preprocessing/download.py`.

### Add a model

1. Copy `configs/models/qwen2_vl_7b.yaml` to `configs/models/<name>.yaml` and set:
   - `model.name` (HF id), `model.auto_class`, `torch_dtype`, quantization and `processor` options (`nested_images: true` for processors expecting one image list per sample);
   - `generation`: the teacher-side generation and retry parameters (used when the model is the teacher);
   - `lora`: `r`, `alpha`, `dropout` and `target_modules` (module names depend on the architecture; check they do not match the vision tower unintentionally);
   - `student_prompt`: the prompt the model receives in training / evaluation (see *Records and prompts*; it must contain `{json_example}`).
2. Use it with `--model <name>`: as teacher (`run_preprocess.py`), as student (`run_train.py` / `run_evaluate.py`), or both.
3. If the architecture needs special loading code (not covered by `auto_class` + processor kwargs), extend `mu4sum/preprocessing/teacher.py` (teacher backend, `TeacherBackend.generate(images, prompt, *, temperature, top_p, max_new_tokens)`) and the model loading used by training.

### Add an evaluation method (metric)

1. In `src/mu4sum/evaluation/metrics.py` register a function that scores lists of predictions and references and returns one list of per-pair scores per metric name:

   ```python
   @METRICS.register("mymetric")
   def mymetric(preds, refs, cfg):        # cfg = the `evaluation` section of the config
       return {"mymetric": [score(p, r) for p, r in zip(preds, refs)]}   # scores in [0, 1]
   ```

   Pairs with an empty prediction score 0 without calling the function. Results are reported x100, at full-answer and aspect level, per view (forget / retain) and per aspect.
2. Reference it from an evaluation preset: add `"mymetric"` to `evaluation.metrics` in `configs/evaluation/default_eval.yaml`, or copy that file to `configs/evaluation/<name>.yaml` (with the metric's own settings, like `bertscore.model_type`) and select it with `--eval <name>` (`--eval-tag` names the output folder).
3. Add a test in `tests/test_misc.py` with a few hand-made prediction rows.

A new training method works the same way: a factory registered with `@TRAINING_METHODS.register("<name>")` in `training/methods.py` plus a `configs/training/<name>.yaml` preset (`training.method: "<name>"`).

## Reproducibility

- `set_seed` seeds python/numpy/torch/CUDA and enables deterministic algorithms (`deterministic: true`).
- Batches are a pure function of `(seed, epoch)`; checkpoints store weights (LoRA adapter only), optimizer, scheduler, progress counters and RNG states. An interrupted run resumed from a checkpoint gives bit-identical weights to an uninterrupted one (verified on a toy model in `tests/test_training.py`).
- Splits are made **per record**, never per image, so the images of one record cannot leak across splits.

## Google Colab workflow

1. Develop and debug locally (Visual Studio) with `pytest` and small `--stages`/`augmentation.limit=N` runs.
2. Open `notebooks/colab_runner.ipynb` on Colab: it clones the repo, installs it, points `MU4SUM_DATA_ROOT` / `MU4SUM_OUTPUT_ROOT` to Drive and runs the scripts. If the session dies, re-run the same cell: it resumes.
3. Locally run `scripts/sync_results.py` and commit. Checkpoints and weights are never synced or tracked; existing local `config_*` / `provenance_*` snapshots are never overwritten.

## Status / caveats

- Tests cover config, prompts, teacher retry logic, the preprocessing pipeline (with a fake teacher), answer formatting/parsing, the trainer (resume equivalence, validation, best checkpoint), every training method's objective on toy models, and per-sample scoring.
- The code paths that need real models (Qwen2-VL / Qwen2.5-VL-3B loading, LoRA training, generation, BERTScore, token offsets of the real tokenizer) have **not** been executed in CI/local tests: they need a GPU and model downloads. LoRA target names may also match modules of the vision tower.
- The CSV location, image folder and column names in `configs/datasets/iu_xray.yaml` (`dataset.csv_path`, `dataset.images`, `dataset.columns`) match the CSV columns (`image_id`, `projection`, `org_caption`) of the Kaggle `masrursabab/iu-chest-x-rays-cleaned` file; the CSV and image folder paths (`cleaned_dataset.csv`, `resized_images/<resolution>`) are still unverified. If they are wrong, `process` fails naming the missing path or listing the actual columns; fix them in the YAML.
