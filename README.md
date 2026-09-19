# ZeqouXTraining

Desktop application for training and fine-tuning AI models. Pick a model, point at a dataset, choose a method, press start — no YAML, no shell scripts, no guessing whether the run will fit in memory.

Part of the [Zeqou ecosystem](https://mishaadevv.github.io/zeqou/).

## Features

- Import models from Hugging Face or a local folder, and inspect them before committing
- Fine-tuning, LoRA, QLoRA and SFT through `transformers` + `peft`
- **Train from scratch**: build a small transformer (micro/tiny/small presets or a custom architecture) and train it on your dataset with no base model and no downloads
- Datasets from JSON, JSONL, CSV, TXT, Parquet, a folder of shards, or the Hugging Face Hub
- Dataset validation before a run: field mapping, duplicates, empty rows, over-length samples
- Real GPU detection (`nvidia-smi` + `torch.cuda`), VRAM estimation and pre-flight warnings
- Automatic parameter selection from the detected hardware, with the reason behind every value
- Start, stop, pause, resume and continue from a checkpoint
- Live loss and learning-rate charts, GPU/VRAM trace, logs and checkpoints
- Export a trained adapter, or merge it into its base model
- Test the finished model in the built-in playground
- Simple mode for a first run, Advanced mode for full control
- One-click ML runtime installation from Settings → Environment, with live pip output
- Background refresh: runs, libraries and the environment snapshot stay current on their own; GPU telemetry is sampled even when nothing is training

## Honest by design

There are no fake progress bars, no invented GPU numbers and no buttons that do nothing. If something needs a package this machine does not have, the app says so in plain language and tells you how to fix it:

```
$ python -m zeqouxtraining.cli env-check
torch         not installed
transformers  not installed
peft          not installed
GPU           no NVIDIA device found (nvidia-smi is not on PATH)
training      unavailable — install the ML runtime in Settings → Environment
```

The training runtime (`torch`, `transformers`, `peft`, `accelerate`) is not bundled: it is machine-specific and several gigabytes. Until it is installed, the backend refuses to start a run with `backend_unavailable` and lists what is missing, rather than producing a simulated result. Everything else — detection, dataset validation, model inspection, VRAM estimation, export — works without it.

## Develop

```bash
npm install
npm run dev      # Vite dev server + Electron, both watched
```

On first launch the app looks for a Python interpreter (`python`, `python3`, `py -3.x`), verifies it can import the backend package, and reports what is missing. Install the ML runtime from **Settings → Environment** using the generated command.

```bash
npm run test:py    # 199 backend tests, standard library only
npm run smoke      # 54 end-to-end checks inside a real Electron main process
npm run verify     # typecheck + both of the above
```

`npm run smoke` is not a mock harness: it boots the actual Electron main process, spawns the actual Python backend, imports real dataset fixtures, validates a malformed file, registers a Hub dataset, exercises the job manager and exports a real artefact folder. It points the app at a throwaway temp directory, so your real library is never touched.

## Build

```bash
npm run pack     # unpacked build in release/
npm run dist     # Windows installer (NSIS) + portable
```

Tagging `v*` runs the cross-platform workflow, which builds Windows, Linux and macOS in parallel and attaches them to the release.

## Architecture

```
Electron main process ──┐
  lib/paths.js          │ userData layout
  lib/store.js          │ atomic JSON stores, OS-encrypted secrets
  lib/hardware.js       │ nvidia-smi sampling, CPU/RAM snapshot
  lib/python.js         │ interpreter discovery + JSON protocol client
  lib/jobs.js           │ run lifecycle, logs, GPU series, checkpoints
  lib/registry.js       │ datasets, models, projects
  lib/exporter.js       │ export orchestration + provenance
  lib/inference.js      │ long-lived inference runtime
  ipc.js                │ one error shape across the bridge
                        │
                        ▼
                    Python backend (python/zeqouxtraining/)
                      events.py      newline-delimited JSON protocol
                      deps.py        installed packages + install plan
                      hardware.py    nvidia-smi and torch.cuda truth
                      datasets.py    load, auto-map, normalise, validate
                      models.py      resolve + inspect local/Hub models
                      config.py      defaults, validation, auto-tuning
                      estimator.py   VRAM estimate
                      checkpoints.py list / resume / prune
                      exporter.py    copy or merge an artefact
                      errors.py      exception → message + hint
                      trainer.py     job runner
                      inference.py   streaming generation
                      backends/      pluggable training backends
```

The Python side is a child process, not a library, so a crash in training can never take the UI down. It speaks one event per line on **stdout**:

```json
{"event": "training-progress", "detail": {"step": 42, "loss": 1.284, "total_steps": 300}}
```

Everything else — including anything a third-party library prints — goes to **stderr** and is shown verbatim in the technical log panel. Errors cross the boundary as data: `{code, message, hint, traceback}`. The UI shows the sentence and the hint; the traceback is one click away.

A training backend subclasses `TrainingBackend` (`backends/base.py`), declares the packages it requires, and registers itself in `backends/registry.py`. The runner, the protocol and the whole UI stay unchanged — that is how Unsloth, TRL, a remote cluster or an ONNX runtime would be added.

## Using it

```
Model → Dataset → Method → Settings → Check → Train → Result
```

**Simple** shows the four settings that decide whether a run succeeds — epochs, batch size, learning rate and context length. **Advanced** adds gradient accumulation, warmup, weight decay, the checkpoint interval and retention, evaluation split, seed, LoRA rank/alpha/dropout/target modules, quantization, precision, scheduler and optimizer.

Turn **Configure parameters automatically** off in Settings → Advanced and the wizard hands you the documented defaults instead of values derived from your machine — and says so, rather than dressing them up as hardware-tuned.

The **Check** step aggregates every blocker — missing runtime, dataset errors, an estimate that exceeds VRAM, a method the hardware cannot run — before you press start.

Stop and pause are cooperative: the app writes a flag file, the trainer notices at the next step boundary, checkpoints, and exits cleanly. That is why a paused run can be resumed later, from its last checkpoint.

## Data and privacy

Everything user-generated lives in Electron's `userData` directory:

```
settings.json  secrets.json  datasets.json  models.json  projects.json  runs.json
runs/<runId>/     job.json, training.log, stderr.log, checkpoints, weights
models/           models trained by this app
cache/huggingface downloaded base models
```

Imported datasets are **referenced, never copied**. The Hugging Face token is encrypted through the OS keychain when available, and stored with an explicit `insecure: true` marker when it is not.

Context isolation on, node integration off, sandbox on, permissions denied by default, navigation refused, new windows blocked. The renderer can only reach the main process through the named preload bridge.

## License

[PolyForm Strict 1.0.0](LICENSE) — viewing and personal use are allowed; copying, modification, redistribution and derivative works are not permitted without the author's permission.
