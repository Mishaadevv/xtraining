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

`validate-dataset`, `preview-dataset` and `export-dataset` accept a local file, a
folder of shards, or a Hub id:

```bash
python -m zeqouxtraining.cli dataset-formats          # every type this install can read
python -m zeqouxtraining.cli validate-dataset --path my_data.jsonl --context-length 512
python -m zeqouxtraining.cli validate-dataset --path ./shards --context-length 512
python -m zeqouxtraining.cli validate-dataset --hf-id tatsu-lab/alpaca --split train
```

Readable types: JSON, JSONL/NDJSON, CSV, PSV, TSV/TAB, TXT/Markdown, Parquet,
Arrow/Feather, ORC, SQLite databases (the largest table is read), Excel workbooks
and YAML — plus `gzip`, `bz2` and `xz` compressed variants of the text formats
(`shard-01.jsonl.gz`). Format detection is by extension and never guessed from
content, so a file that cannot be read says exactly why. Readers that need a
package (pyarrow, openpyxl, pyyaml, the Hub `datasets`) report
`missing_dependency` with the `pip install` line instead of failing obscurely.

## Exporting a dataset as one file

```bash
python -m zeqouxtraining.cli export-dataset --path ./shards --output all.jsonl
python -m zeqouxtraining.cli export-dataset --path ./data.db --output all.csv --raw
python -m zeqouxtraining.cli export-dataset --hf-id tatsu-lab/alpaca --output alpaca.jsonl
```

Exactly one file is written — a shard folder, a database or a Hub dataset all
become a single file (JSONL, JSON, CSV, TSV, TXT or Parquet). By default the rows
are the *normalised* samples the trainer would see; `--raw` keeps the original
records instead. An existing file is never overwritten unless `--overwrite` is
given.

## Exporting a trained artefact

```bash
python -m zeqouxtraining.cli export --source runs/my-run --describe
python -m zeqouxtraining.cli export --source runs/my-run --output D:/exports/my-run
python -m zeqouxtraining.cli export --source runs/my-run --output D:/exports/my-run.zip --pack
python -m zeqouxtraining.cli export --source runs/my-run --output D:/exports/merged --merge
```

Copy mode writes the adapter plus a generated `README.md` and `export.json`;
`--pack` writes that same export into one single `.zip` file. `--merge` folds the
adapter into its base model and therefore genuinely needs torch + peft; without
them the command fails with `missing_dependency` instead of writing a folder that
would not load.

## Running without the app

```bash
python -m zeqouxtraining.cli env-check
python tests/test_backend.py
```
