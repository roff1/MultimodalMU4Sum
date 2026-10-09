# MultimodalMU4Sum

Framework for **multimodal aspect-based summarization** (chest X-ray reports) and **machine unlearning**, extending the MU4Sum paper.
Everything that varies between experiments (model, dataset, unlearning method, evaluation) is a YAML preset; code changes are only needed for genuinely new behaviour.

## Install

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -e ".[all]"                              # extras: download, train, eval, dev
pytest                                               # light tests, CPU only, no model downloads
```

## Pipeline

All scripts take the minimum (model / dataset / method) plus optional `key.path=value` overrides.
Overrides are strict: a typo fails; use `+key=value` to add a brand-new key.

```bash
# 1. download -> process (group by study) -> teacher augmentation (Qwen2-VL-7B-Instruct)
python scripts/run_data_pipeline.py --dataset iu_xray --model qwen2_vl_7b
python scripts/run_data_pipeline.py --dataset iu_xray            # download + process only
python scripts/run_data_pipeline.py --dataset iu_xray --model qwen2_vl_7b --stages augment --force

# 2. LoRA fine-tuning on all aspects
python scripts/run_finetune.py --model qwen2_vl_7b --dataset iu_xray finetune.lr=1e-4

# 3. unlearning (forget aspects are defined by the dataset preset, e.g. forget_aspects=[impression])
python scripts/run_unlearn.py --model qwen2_vl_7b --dataset iu_xray --method grad_diff --tag a0.5 unlearning.alpha=0.5

# 4. evaluation, per aspect (+ overall / forget / retain groups)
python scripts/run_evaluate.py --model qwen2_vl_7b --dataset iu_xray --target finetune --unlearning grad_diff
python scripts/run_evaluate.py --model qwen2_vl_7b --dataset iu_xray --target grad_diff --tag a0.5

# 5. copy light artifacts (configs, metrics, predictions) from Drive into the repo
python scripts/sync_results.py --src "<Drive>/MU4Sum/outputs"
```

Stages skip themselves when their artifacts exist and are fresh. If the config that produced them changed, the run raises `StaleArtifactError` instead of silently mixing data; pass `--force` to rebuild.

## Config system (`configs/`)

```
base.yaml            seed, determinism, runtime (environment / data_root / output_root)
models/*.yaml        HF model id, auto_class, dtype, quantization, processor kwargs, teacher generation + retry params
datasets/*.yaml      source, processor, columns, aspects (+ descriptions), forget_aspects, prompt, splits, augmentation
finetune/*.yaml      training + LoRA hyper-parameters
unlearning/*.yaml    method, alpha, training hyper-parameters (forget/retain aspects come from the dataset)
evaluation/*.yaml    split, metrics, generation settings
```

`load_config(operation, ...)` knows which preset groups each operation needs (`preprocess`, `finetune`, `unlearn`, `evaluate`), merges `base` + presets + CLI overrides, resolves the machine-specific `runtime` section and freezes the result.
A copy is written, read-only, to the run folder (`config_<operation>.yaml` + `provenance_<operation>.json` with git commit and package versions). Re-running with a *different* effective config in the same folder fails: use `--tag` for a new run. The `runtime` section is excluded from the comparison, so a run started locally can be resumed on Colab.

Outputs are organised hierarchically: `outputs/<model>/<dataset>/<method>[/<tag>]`, with evaluations in `<run>/evaluation/<eval_tag>`.

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

`forget_aspects` sits outside `aspects` on purpose: it is not part of the artifact fingerprint, so changing it never invalidates the teacher data (it does change the config snapshot of unlearn runs, so use `--tag` for a different forget set).

## Extending without touching the code

- **New VLM**: copy `configs/models/qwen2_vl_7b.yaml`, change `model.name` (and `auto_class` / LoRA `target_modules` if needed).
- **New dataset with the same shape** (CSV grouped by study): copy `configs/datasets/iu_xray.yaml`, set `source`, `columns`, `aspects`, `forget_aspects` and the `prompt` texts.
- **New source / format / unlearning method / metric**: add a function decorated with `@DOWNLOADERS.register("x")`, `@PROCESSORS.register("x")`, `@UNLEARNING_METHODS.register("x")` or `@METRICS.register("x")`, then reference its name from YAML.

## Reproducibility

- `set_seed` seeds python/numpy/torch/CUDA and enables deterministic algorithms (`deterministic: true`).
- Batches are a pure function of `(seed, epoch)`; checkpoints store weights (LoRA adapter only), optimizer, scheduler, progress counters and RNG states. An interrupted run resumed from a checkpoint gives bit-identical weights to an uninterrupted one (verified on a toy model in `tests/test_training.py`).
- Splits are made **per study**, never per image, so frontal/lateral views of an exam cannot leak across splits.

## Google Colab workflow

1. Develop and debug locally (Visual Studio) with `pytest` and small `--stages`/`augmentation.limit=N` runs.
2. Open `notebooks/colab_runner.ipynb` on Colab: it clones the repo, installs it, points `MU4SUM_DATA_ROOT` / `MU4SUM_OUTPUT_ROOT` to Drive and runs the scripts. If the session dies, re-run the same cell: it resumes.
3. Locally run `scripts/sync_results.py` and commit. Checkpoints and weights are never synced or tracked; existing local `config_*` / `provenance_*` snapshots are never overwritten.

## Status / caveats

- Tests cover config, prompts, teacher retry logic, the data pipeline (with a fake teacher), the trainer (resume equivalence) and unlearning losses on toy models.
- The code paths that need real models (Qwen2-VL loading, LoRA fine-tuning, generation, BERTScore) have **not** been executed in CI/local tests: they need a GPU and model downloads.
- The CSV column names in `configs/datasets/iu_xray.yaml` (`dataset.columns`) are a best guess for the Kaggle `masrursabab/iu-chest-x-rays-cleaned` file. If they are wrong, `process` fails and lists the actual columns; fix them in the YAML.
