# ZeqouXTraining

**Train. Tune. Test. Build.** — a local-first AI model training studio for the Zeqou ecosystem.

ZeqouXTraining is a desktop application that trains, continues, evaluates, tests, converts and serves AI models on
your own machine. Models, datasets, checkpoints and logs are ordinary files in a workspace folder you choose.
Nothing is uploaded anywhere, and nothing is simulated:

* training runs are real processes with real gradients, checkpoints and loss curves;
* hardware numbers come from the CPU, the driver, `nvidia-smi` and the OS;
* every memory/size figure that is a prediction is labelled **Estimated**;
* when a feature cannot work here (no CUDA, no PyTorch, no bitsandbytes, no llama.cpp converter), the interface says
  so instead of pretending.

## Architecture

```
┌──────────────────────────── Electron (Node) ────────────────────────────┐
│  main.js            window, settings, workspace, IPC                    │
│  lib/python.js      interpreter discovery, engine calls, process spawn  │
│  lib/jobs.js        job directories, status/metrics/log tailing         │
│  lib/sidecar.js     resident inference process for the Playground       │
│  lib/store.js       atomic JSON stores (settings, registry)             │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │  stdin/stdout JSON + @@event stream
┌───────────────────────────────▼─────────────────────────────────────────┐
│  Python engine — python/zxtrain                                         │
│   cli.py         one entry point for the GUI and the CLI                │
│   trainer.py     job contract, pre-flight, run loop, lineage, recovery  │
│   backends/      tiny (pure Python) · hf (Transformers + PEFT)          │
│   models.py      safetensors/GGUF/config/tokenizer inspection           │
│   datasets.py    readers, stats, cleaning, splits, export               │
│   bpe.py         BPE tokenizer training + statistics                    │
│   precision.py   dtype conversion, quantization planning                │
│   adapters.py    LoRA/PEFT discovery and merge planning                 │
│   server.py      local OpenAI-compatible inference server               │
│   hardware.py    real CPU/GPU/RAM/disk/precision detection              │
│   storage.py     workspace accounting, orphan detection, guarded delete │
│   environment.py interpreter inventory and runtime installation         │
└─────────────────────────────────────────────────────────────────────────┘
```

Two training backends ship with the app:

| Backend | Needs | Trains |
| --- | --- | --- |
| `tiny` | nothing — pure Python standard library | small transformers from scratch, continued pretraining, full fine-tune, SFT, plus its own BPE tokenizer |
| `hf` | `torch`, `transformers`, `peft` (+ `bitsandbytes` for QLoRA) | real Hugging Face checkpoints with LoRA/QLoRA/SFT/full fine-tune, evaluation and streaming generation |

Availability is *detected* at runtime. The Environment page shows exactly which packages are present in the
interpreter the engine is using, and can create a private environment inside the workspace — never in the system
Python.

## The workflow

```
Import model → Inspect → Import/build dataset → Validate & clean → Split
→ Configure training → Preview (estimates) → Start → Monitor live → Checkpoints
→ Resume / continue / new dataset / multi-stage → Evaluate → Playground → Compare
→ Quantize / convert / merge adapters → Export → Serve locally → Continue from the result
```

Every stage is a page in the sidebar: Dashboard, Hardware, Environment, Models, Datasets, Files, Projects, Training,
Experiments, Evaluation, Playground, Compare, Adapters, Quantization, Conversion, Deploy, Jobs, Settings,
Documentation.

## Continuation training

This is the part the app is built around. Four distinct operations, never conflated:

* **Resume** — the exact interrupted run. Weights, optimizer, scheduler, gradient scaler, RNG and global step are
  restored when they exist on disk; the engine reports per item what was *really* restored. Configuration changes that
  would make an exact resume impossible are refused with an explanation.
* **Continue training** — a new child run from a checkpoint or model with a fresh optimizer (new dataset, new learning
  rate, new method). The parent is recorded in `lineage.json`.
* **Continued pretraining** — the same, on additional raw/domain text.
* **Adapter continuation** — resume LoRA/QLoRA training when the adapter is compatible with its base model.

Every produced model and checkpoint keeps a `parent` link, so the Experiments page can draw the real lineage tree.

## Requirements

* Windows, macOS or Linux, Node 20+ to run the app, and **Python 3.10 – 3.13**.
* **Node 22.4 or newer** to run the tests that boot the packed build: they drive the installed app through its
  DevTools socket using the WebSocket client built into Node, which was experimental before 22.4. That is the version
  CI and the release workflow use; the app itself does not need it.
* Python 3.12/3.13 is recommended: PyTorch publishes no wheels for 3.14 yet, and the app tells you this instead of
  failing later. The `tiny` backend works on any version.
* An NVIDIA GPU is optional. Without one, the app runs the tiny backend on the CPU and reports that CUDA-only
  features are unavailable.

## Getting started

```bash
npm install
npm run dev          # Vite + Electron with hot reload
```

Run the packaged path:

```bash
npm run build        # typecheck + production renderer build
npm start            # build, then electron .
npm run pack         # unpacked build in release/win-unpacked
npm run dist         # installers via electron-builder (release/ )
npm run dist:signed  # the same, signed with a local certificate
```

Verification:

```bash
npm run typecheck    # tsc --noEmit
npm run check:lock   # the lock file installs on every platform, not only this one
npm test             # Python engine test suite (real training, resume, conversion, adapters)
npm run smoke        # Electron smoke test in a throwaway workspace
npm run launch       # starts the real application and walks every page
npm run verify       # typecheck + test + smoke
npm run verify:packed  # starts the packed executable and tests it
npm run verify:update  # proves an update is detected, downloaded, and that an untrusted publisher is refused
```

The smoke test drives the real modules the app uses — interpreter discovery, the job manager, the engine CLI — and
asserts that training produced metrics, weights, checkpoints, a lineage record, real generated tokens and a persisted
registry. `npm test` runs the engine suite: dataset inspection and cleaning, BPE training, model inspection, hardware
detection, checkpoint/resume, generation, evaluation, precision conversion, adapter discovery, storage protection and
the CLI contract.

The launch test goes one step further and boots the actual desktop application — a window opens briefly. It checks
that the preload bridge is exposed while Node stays out of the renderer, that the interface reaches the Python engine
through IPC (real CPU, real backends, real job list), that all 19 sidebar pages render, and that the command palette
opens, filters and closes. It also fails on any renderer console error, failed load or crashed process.

The packed-build test answers a question `electron-builder` cannot answer about itself: that the installed app works.
It starts `release/win-unpacked/ZeqouXTraining.exe` with a throwaway profile and workspace, attaches to it over the
DevTools protocol and asserts that the interface loads from inside `app.asar`, that the engine directory resolves to
`resources/python` rather than to the sources, that the bundled engine answers with real hardware data, and that all
19 pages render. Run it after `npm run pack` or `npm run dist`.

The update test stands up a release channel of its own — a `latest.yml` and an installer, served over loopback — points
the packed app at it with `ZEQOUX_UPDATE_URL` and drives the update the way the interface does: check, try to install
before anything was downloaded (refused), download, and compare the downloaded file's SHA-512 with the one in the
manifest. It then turns signature verification back on and asserts the opposite outcome: an update whose publisher
cannot be verified is refused with the publisher's name in the message and is not kept as an installed update.

Set `ZEQOUX_USER_DATA` to keep the app's settings, caches and logs somewhere other than the default profile — useful
for portable installs and required by nothing else; the tests set it themselves so they never touch your real profile.

CI runs all of it on Windows, macOS and Linux (see `.github/workflows/ci.yml`), packages the app and runs the
packed-build and update tests on Windows, and runs the engine suite on Python 3.11 – 3.13. The release workflow
verifies the installers it is about to publish the same way.

Before installing anything, CI checks that the lock file can be installed on every platform. A `package-lock.json`
is written by whichever machine ran `npm install`, and npm seeds the tree from the local `node_modules` — so a lock
created on Windows can describe only Windows and leave rollup or esbuild without a native binary everywhere else.
That is exactly how this repository once shipped a Windows-only lock and a red ubuntu build, so `npm run check:lock`
now asserts that every optional (that is, per-platform) dependency is really in the lock.

## Distribution and updates

The app updates itself from a release channel, and both halves of that are checked
by the build rather than assumed.

**Signing.** `npm run cert` creates a self-signed code-signing certificate in
`certs/` (git-ignored), and `npm run dist:signed` uses it. `electron-builder` also
signs when `CSC_LINK` and `CSC_KEY_PASSWORD` are set, which is how CI does it once
those secrets exist. The publisher name in the build configuration has to be the
certificate's subject — `electron-updater` compares the two and refuses an update
that does not match — so `scripts/dist.mjs` passes it from the certificate instead
of hard-coding it. A self-signed certificate is not a substitute for a real one:
Windows still warns until the certificate is trusted, and only machines that have
been told to trust it will treat the signature as valid (`node scripts/make-cert.mjs --trust`
does that for the current user, and asks the operating system for permission).

**The channel.** `package.json` names it: `build.publish` points at
`github.com/Mishaadevv/xtraining` releases. `electron-builder` writes that into
`app-update.yml` next to the engine, so the installed app knows where to look. An
unpacked build gets no such file from `electron-builder`, so `scripts/dist.mjs`
writes the same configuration for it — otherwise `release/win-unpacked` could not
test the update path at all.

**In the app.** Settings → Updates shows the installed version, the channel, the
last check and the updater's log; it can check, download and install. Updates are
downloaded only when asked for and installed only on an explicit restart. A check
that runs on its own at launch (on by default, 15 seconds in) reports a failure to
the log and as a quiet note, never as an error to dismiss; a check you ask for
reports what happened.

Some settings exist for hosts and tests, and none of them are in the interface:

| Variable | Effect |
|---|---|
| `ZEQOUX_UPDATE_URL` | Use a generic channel at this URL instead of the release channel |
| `ZEQOUX_UPDATE_SKIP_SIGNATURE=1` | Do not verify the publisher of a downloaded update — for self-signed test builds only, and the interface says so while it is on |
| `ZEQOUX_USER_DATA` | Keep settings, caches and logs in this directory (a portable profile; tests use it to stay out of your real one) |
| `ZEQOUX_PYTHON` | Use this interpreter instead of discovering one |

**Hosting a channel anywhere else.** A generic channel is a `latest.yml` and the
installer it names, at the same base URL:

```bash
node scripts/make-update-manifest.mjs --installer "release/ZeqouXTraining Setup 2.0.0.exe" --version 2.0.1 --out release/channel
npm run update-server -- release/channel        # serves it, including byte ranges
```

`npm run verify:update` uses exactly that: it publishes the current installer as
the next version on a throwaway local channel, starts the packed application
against it, and asserts that the channel is read, the newer version is found,
installing a version that was never downloaded is refused, the payload is
downloaded and its SHA-512 matches the manifest, and the progress the interface
shows is real. The one thing it cannot show is the installer replacing the running
build — the payload is the current build wearing the next version number, because
that needs a real release.

## CLI companion

The GUI and the CLI drive the same engine:

```bash
python -m zxtrain.cli --help                       # every command
echo '{}' | python -m zxtrain.cli hardware.detect  # one command
python -m zxtrain.cli run <job>/spec.json          # run a job exactly as the app does
python -m zxtrain.cli serve                        # inference sidecar used by the Playground
```

To point the app at a specific interpreter, set `ZEQOUX_PYTHON` (or `pythonPath` in Settings).

## Workspace layout

```
<workspace>/
  models/      imported and trained models          datasets/   imported and built datasets
  jobs/        one folder per run: spec, status,     exports/    exported models and cards
               events, metrics, checkpoints,         plugins/    optional engine plugins
               lineage, console.log                 runtime/    the app's own venv, if created
  registry.json   library, projects, servers, evaluations
```

Deletion always requires an explicit confirmation, protected artifacts and running jobs are refused by the engine, and
the workspace can be moved or copied with any file manager.

## Security and privacy

The app is local-only by default. It never downloads a model, dataset or package without asking; it never binds the
inference server beyond loopback unless you change the host; and models that need custom Python code are refused
unless you explicitly enable that in Settings.

The one thing that leaves the machine on its own is the update check at launch, which is a single request to the
release channel. It can be turned off in Settings → Updates, and a downloaded update is never installed without an
explicit restart. When it is installed, the download is checked against its SHA-512 and — unless verification was
deliberately disabled — against the publisher named in the build.

## License

PolyForm Strict License 1.0.0 — see [LICENSE](LICENSE). Copyright 2026 Misha (misakolot6@gmail.com).
