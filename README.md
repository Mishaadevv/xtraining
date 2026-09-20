# ZeqouXTraining

A desktop studio for fine-tuning small language models — locally, with a real GPU/CPU
pipeline and no cloud dependency. Electron shell, React interface, Python engine.

```
┌────────────────────────────────────────────────────────────┐
│  React renderer (Vite + TypeScript + Tailwind)             │
│  pages · wizard · live charts · playground w/ Thinking     │
├────────────────────────────────────────────────────────────┤
│  Electron main (Node)                                      │
│  preload bridge · IPC · job manager · registry · storage   │
├────────────────────────────────────────────────────────────┤
│  Python engine (zeqouxtraining)                            │
│  CLI protocol · datasets · backends · trainer · inference  │
└────────────────────────────────────────────────────────────┘
```

## What it does

- **Projects** — every training run belongs to a project with its history, best loss
  and recorded configuration.
- **New training wizard** — pick a base model and a dataset, and the app derives a
  complete configuration (batch size, precision, context length, optimizer, rank)
  from the detected hardware, explaining every choice. Simple and Advanced modes.
- **Datasets** — JSON, JSONL, CSV, TXT, Parquet, a folder of shards, or a Hugging Face
  Hub id. Referenced in place, never copied. A built-in Zeqou set (default v2 with
  thinking, plus dialogue, code, science and tech parts and language packs) ships
  inside the app and stands selected until you pick your own — a first run needs no
  import at all. Validation reports exactly what the model would learn from: field
  mapping, duplicates, over-length records, role statistics and normalised previews.
- **Training** — live loss/LR curves, GPU utilisation and VRAM telemetry sampled from
  nvidia-smi, step metrics, ETA, event log, stdout/stderr, checkpoints with pause,
  resume and retention. A LoRA adapter (or 4-bit QLoRA, full fine-tune, SFT, or a
  from-scratch small GPT) is produced into the app model library.
- **Models** — the library of base and trained models, with export to any folder
  (copy, or a real merge of the adapter into the base model) and one-click
  "test in playground".
- **Playground** — load a trained or local model and talk to it with streaming output.
  **Thinking mode** streams the model's private chain of thought into its own panel
  before the answer, with adjustable sampling (temperature, top-p, repetition penalty).
- **Hardware** — what the machine can actually do: NVIDIA GPUs from nvidia-smi,
  the installed PyTorch build, its CUDA support, CPU and RAM.
- **Settings** — interpreter selection, one-click ML runtime installation into an
  isolated venv, Hugging Face token in the OS keychain, storage locations, theme.

## Repository layout

```
electron/          main process: main.js, ipc.js, preload.js, lib/*
src/               renderer: features/* pages, state, components, lib
python/            the zeqouxtraining engine and its test suite
scripts/           dev.mjs (vite+electron) and smoke.js (end-to-end)
```

The Python package speaks a line-delimited JSON protocol on stdout, so the Node side
stays a thin supervisor and the ML logic stays testable in plain Python.

## Development

```bash
npm install
npm run dev            # vite dev server + electron shell
npm run test:py        # python engine tests (206)
npm run smoke          # end-to-end electron smoke suite (54)
npm run verify         # typecheck + python tests + smoke
npm run build          # typecheck + production renderer build
```

The renderer also runs standalone (`npm run dev:web`) for interface work: every
privileged call then resolves with an explicit `desktop_only` result instead of
pretending to work.

## Installers

Windows builds ship an assisted NSIS installer and a portable executable. The
installer shows the PolyForm Strict 1.0.0 license before copying anything and
lets you choose the installation folder; the portable build needs no install.

## Thinking mode

In the Playground, enable **Thinking mode**. The generation request carries
`thinking: true`; the engine keeps everything between `<think>` and `</think>`
out of the answer, streams it on a separate event channel, and reports it in the
result as `thinking` alongside token statistics. The interface shows it in a
collapsible panel above the answer and keeps it in session history. Models that
were never trained to reason simply return an empty thinking block, so the toggle
is always safe.
