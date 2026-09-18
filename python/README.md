# ZeqouXTraining — Python backend

The desktop app is the shell; this package does the real work: hardware
detection, dataset validation, training and inference.

## Two layers, on purpose

| Layer | Needs | Commands |
| --- | --- | --- |
| Diagnostics | Standard library only | `env-check`, `hardware`, `validate-dataset`, `preview-dataset`, `inspect-model`, `auto-config`, `estimate`, `checkpoints`, `export` (copy mode) |
| ML runtime | `torch`, `transformers`, `peft`, … | `train`, `infer`, `export --merge` |

`torch` is never imported at module import time. That is why the app can still
tell you *why* training is unavailable on a bare Python install, instead of
failing with an opaque traceback.

## Wire protocol

stdout carries newline-delimited JSON events, one per line:

```json
{"event": "training-progress", "detail": {"step": 12, "loss": 1.42, "total_steps": 300}}
```

stderr carries human-readable log lines, including everything third-party
libraries print. The event names are listed in `zeqouxtraining/events.py`.

## Layout

```
zeqouxtraining/
  events.py        protocol helpers (emit/log/stage/result/fail)
  deps.py          installed-package inspection + pip install plan
  hardware.py      nvidia-smi + torch.cuda truth, CPU/RAM, CUDA toolkit
  datasets.py      load / auto-map fields / normalise / validate
  models.py        resolve and inspect local or Hugging Face models
  config.py        defaults, validation, automatic parameter selection
  estimator.py     VRAM estimate with an explicit accuracy disclaimer
  checkpoints.py   list / resume / prune checkpoints
  exporter.py      copy an artefact, or merge an adapter into its base model
  errors.py        exception -> readable message + hint
  trainer.py       job runner
  inference.py     long-lived inference runtime
  cli.py           argparse entry points
  backends/
    base.py        framework-agnostic backend interface
    registry.py    backend registration
    hf_peft.py     transformers + peft (LoRA, QLoRA, SFT, full)
```

Adding a training backend (Unsloth, TRL, a remote cluster, an ONNX runtime)
means adding one module that subclasses `TrainingBackend` and registering it in
`backends/registry.py`. The runner, the event protocol and the UI do not change.

## Installing the ML runtime

The app generates the exact command in **Settings → Environment**. In short:

```bash
# NVIDIA GPU: get a CUDA-enabled torch build
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt

# CPU only
python -m pip install -r requirements.txt
```

## Dataset sources

`validate-dataset` and `preview-dataset` accept a local path **or** a Hub id:

```bash
python -m zeqouxtraining.cli validate-dataset --path my_data.jsonl --context-length 512
python -m zeqouxtraining.cli validate-dataset --hf-id tatsu-lab/alpaca --split train
```

Hub ids need the optional `datasets` package; without it the command returns a
`missing_dependency` result naming the package, rather than guessing.

## Exporting

```bash
python -m zeqouxtraining.cli export --source runs/my-run --describe
python -m zeqouxtraining.cli export --source runs/my-run --output D:/exports/my-run
python -m zeqouxtraining.cli export --source runs/my-run --output D:/exports/merged --merge
```

Copy mode writes the adapter plus a generated `README.md` and `export.json`.
`--merge` folds the adapter into its base model and therefore genuinely needs
torch + peft; without them the command fails with `missing_dependency` instead of
writing a folder that would not load.

## Running without the app

```bash
python -m zeqouxtraining.cli env-check
python tests/test_backend.py
```
