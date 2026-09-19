/**
 * IPC surface.
 *
 * Every handler returns a plain, serialisable object with an `ok` flag rather
 * than throwing across the bridge — the renderer then has one error shape to
 * render, with the backend's human-readable message and hint intact.
 */
const { ipcMain, dialog, shell, app, BrowserWindow } = require("electron");
const fs = require("node:fs");
const path = require("node:path");

const { dirs, files, pythonPackageDir } = require("./lib/paths");
const store = require("./lib/store");
const python = require("./lib/python");
const hardware = require("./lib/hardware");
const jobs = require("./lib/jobs");
const registry = require("./lib/registry");
const exporter = require("./lib/exporter");
const inference = require("./lib/inference");

function sender() {
  return (channel, payload) => {
    for (const window of BrowserWindow.getAllWindows()) {
      if (!window.isDestroyed()) window.webContents.send(channel, payload);
    }
  };
}

function ok(payload = {}) {
  return { ok: true, ...payload };
}

/** Wrap a handler so a thrown error still crosses the bridge as data. */
function wrap(handler) {
  return async (event, ...args) => {
    try {
      return await handler(...args);
    } catch (error) {
      return {
        ok: false,
        error: {
          code: error.code || "internal",
          message: error.message || String(error),
          hint: error.hint || "",
        },
      };
    }
  };
}

function register() {
  const emit = sender();
  jobs.attach(emit);
  inference.attach(emit);

  /* ---------------------------------------------------------------- app */
  ipcMain.handle("zeqou:app:info", wrap(() => ok({
    name: app.getName(),
    version: app.getVersion(),
    platform: process.platform,
    electron: process.versions.electron,
    node: process.versions.node,
    chrome: process.versions.chrome,
    userData: dirs.root(),
    runsDir: dirs.runs(),
    modelsDir: dirs.models(),
    hfCacheDir: dirs.hfCache(),
    backendDir: pythonPackageDir(),
    encryptionAvailable: store.encryptionAvailable(),
  })));

  ipcMain.handle("zeqou:shell:openPath", wrap(async (target) => {
    if (!target) return { ok: false, error: { message: "No path given." } };
    const error = await shell.openPath(target);
    return error ? { ok: false, error: { message: error } } : ok();
  }));

  ipcMain.handle("zeqou:shell:openExternal", wrap(async (url) => {
    await shell.openExternal(url);
    return ok();
  }));

  ipcMain.handle("zeqou:shell:showItem", wrap((target) => {
    if (target && fs.existsSync(target)) shell.showItemInFolder(target);
    return ok();
  }));

  /* ------------------------------------------------------------- dialogs */
  ipcMain.handle("zeqou:dialog:dataset", wrap(async () => {
    const result = await dialog.showOpenDialog({
      title: "Import dataset",
      properties: ["openFile", "openDirectory", "multiSelections"],
      filters: [
        { name: "Datasets", extensions: ["json", "jsonl", "ndjson", "csv", "tsv", "txt", "parquet"] },
        { name: "All files", extensions: ["*"] },
      ],
    });
    if (result.canceled || !result.filePaths.length) return { ok: true, paths: [] };
    return ok({ paths: result.filePaths });
  }));

  ipcMain.handle("zeqou:dialog:modelFolder", wrap(async () => {
    const result = await dialog.showOpenDialog({
      title: "Select a model folder",
      properties: ["openDirectory"],
    });
    if (result.canceled || !result.filePaths.length) return { ok: true, paths: [] };
    return ok({ paths: result.filePaths });
  }));

  ipcMain.handle("zeqou:dialog:python", wrap(async () => {
    const result = await dialog.showOpenDialog({
      title: "Select the Python interpreter",
      properties: ["openFile"],
      filters: process.platform === "win32"
        ? [{ name: "Python", extensions: ["exe"] }, { name: "All files", extensions: ["*"] }]
        : [{ name: "All files", extensions: ["*"] }],
    });
    if (result.canceled || !result.filePaths.length) return { ok: true, paths: [] };
    return ok({ paths: result.filePaths });
  }));

  ipcMain.handle("zeqou:dialog:directory", wrap(async (_title) => {
    const result = await dialog.showOpenDialog({
      title: _title || "Select a folder",
      properties: ["openDirectory", "createDirectory"],
    });
    if (result.canceled || !result.filePaths.length) return { ok: true, paths: [] };
    return ok({ paths: result.filePaths });
  }));

  /* ----------------------------------------------------------- settings */
  ipcMain.handle("zeqou:settings:get", wrap(() => ok({ settings: store.settings().get() })));

  ipcMain.handle("zeqou:settings:set", wrap((patch) => {
    const next = store.settings().merge(patch || {});
    emit("zeqou:settings:changed", { settings: next });
    // A different interpreter invalidates the resolved one.
    if (patch && Object.prototype.hasOwnProperty.call(patch, "interpreterPath")) {
      python.setInterpreter(patch.interpreterPath);
    }
    return ok({ settings: next });
  }));

  ipcMain.handle("zeqou:settings:token", wrap((payload) => {
    const { token } = payload || {};
    store.setSecret("hfToken", token || null);
    const info = store.secretStorageInfo("hfToken");
    return ok({ storage: info });
  }));

  ipcMain.handle("zeqou:settings:tokenStatus", wrap(() => ok({
    storage: store.secretStorageInfo("hfToken"),
    encryptionAvailable: store.encryptionAvailable(),
  })));

  ipcMain.handle("zeqou:settings:reset", wrap(() => {
    store.settings().replace({ ...store.SETTINGS_DEFAULTS });
    python.setInterpreter(null);
    return ok({ settings: store.settings().get() });
  }));

  /* ---------------------------------------------------------- environment */
  ipcMain.handle("zeqou:env:detect", wrap(async (options = {}) => {
    if (options.force) {
      await python.resolve(true);
      await python.discover(true);
    }
    // Node answers immediately for CPU/RAM; Python adds the CUDA truth.
    const system = hardware.systemSnapshot();
    const smi = await hardware.queryGpus();
    const health = await python.health();

    let backend = null;
    let backendError = null;
    if (health.ready) {
      const response = await python.run(["env-check"]);
      if (response.result) backend = response.result;
      else backendError = response.error || { message: response.stderr.split("\n").slice(-3).join(" ") };
    }

    return ok({
      system,
      smi,
      python: health,
      dependencies: backend ? backend.dependencies : null,
      hardware: backend ? backend.hardware : null,
      backends: backend ? backend.backends : null,
      installPlan: backend ? backend.install_plan : null,
      backendError,
      capabilities: backend ? backend.dependencies.capabilities : null,
      refreshedAt: Date.now(),
      force: Boolean(options.force),
    });
  }));

  ipcMain.handle("zeqou:env:interpreters", wrap(async (options = {}) => {
    const list = await python.discover(Boolean(options.force));
    const health = await python.health();
    return ok({ interpreters: list, active: health.python, packageDir: health.packageDir });
  }));

  ipcMain.handle("zeqou:env:setInterpreter", wrap((executablePath) => {
    python.setInterpreter(executablePath);
    return ok({ interpreterPath: executablePath, settings: store.settings().get() });
  }));

  ipcMain.handle("zeqou:env:installPlan", wrap(async (cudaTag) => {
    const response = await python.run(["install-plan", ...(cudaTag ? ["--cuda-tag", cudaTag] : [])]);
    if (!response.result) return { ok: false, error: response.error };
    return ok({ plan: response.result });
  }));

  ipcMain.handle("zeqou:env:backends", wrap(async () => {
    const response = await python.run(["backends"]);
    if (!response.result) return { ok: false, error: response.error };
    return ok({ backends: response.result.backends });
  }));

  /*
   * ML runtime installation, redone from scratch.
   *
   * The old approach pip-installed into whatever interpreter was selected —
   * which fails quietly on Python 3.14 (no torch wheels exist for it yet) and
   * duplicated the --index-url flag. The new approach is honest and isolated:
   *
   *   1. A base interpreter is chosen (3.10–3.13 preferred, 3.14 refused with
   *      a readable reason instead of a pip wall of red).
   *   2. The app creates its OWN virtualenv under userData/venv, so the
   *      user's system Python is never touched.
   *   3. pip runs inside that venv; torch comes from the CUDA wheel index
   *      exactly once when a tag is configured.
   *   4. The venv is saved in settings, and resolveInterpreterOrder puts it
   *      first so every backend call uses it.
   */
  const { spawn: spawnInstall, execFile: execFileInstall } = require("node:child_process");
  let installChild = null;

  const VENV_MIN = [3, 10];
  const VENV_MAX = [3, 13]; // torch ships no wheels for 3.14 yet — say so instead of failing obscurely

  function parsePyVersion(versionString) {
    const match = String(versionString || "").match(/(\d+)\.(\d+)(?:\.(\d+))?/);
    if (!match) return null;
    return { major: Number(match[1]), minor: Number(match[2]) };
  }

  function runInstallStep(command, args, onLine, timeoutMs = 300000) {
    return new Promise((resolve) => {
      const child = execFileInstall(command, args, { windowsHide: true, maxBuffer: 32 * 1024 * 1024, timeout: timeoutMs }, (error, stdout, stderr) => {
        resolve({ code: error && typeof error.code === "number" ? error.code : error ? -1 : 0, stdout: String(stdout || ""), stderr: String(stderr || "") });
      });
      child.stdout?.on("data", (chunk) => String(chunk).split(/\r?\n/).forEach(onLine));
      child.stderr?.on("data", (chunk) => String(chunk).split(/\r?\n/).forEach(onLine));
    });
  }

  async function pickBaseInterpreter() {
    // The configured interpreter first, then every discoverable one; the
    // newest one inside the supported range wins, because newer point
    // releases fix packaging bugs without changing the ABI.
    const discovered = await python.discover(false);
    const usable = discovered.filter((entry) => entry.available && entry.info);
    const scored = usable.map((entry) => {
      const version = parsePyVersion(entry.info.version);
      const ok = version && version.major === 3 && version.minor >= VENV_MIN[1] && version.minor <= VENV_MAX[1];
      return { entry, version, ok };
    });
    const good = scored.filter((item) => item.ok).sort((a, b) => b.version.minor - a.version.minor);
    return good.length ? good[0].entry : null;
  }

  function venvPythonPath(venvDir) {
    return process.platform === "win32"
      ? path.join(venvDir, "Scripts", "python.exe")
      : path.join(venvDir, "bin", "python");
  }

  ipcMain.handle("zeqou:env:installRuntime", wrap(async () => {
    if (installing || installChild) {
      return { ok: false, error: { code: "install_busy", message: "An installation is already running." } };
    }
    installing = true;
    try {
      return await runRuntimeInstall();
    } finally {
      installing = false;
    }
  }));

  let installing = false;

  async function runRuntimeInstall() {

    const send = (payload) => emit("zeqou:runtime:install", payload);
    const base = await pickBaseInterpreter();
    if (!base) {
      send({ phase: "failed", message: "No suitable Python interpreter found." });
      return {
        ok: false,
        error: {
          code: "no_python",
          message: "No suitable Python interpreter was found (3.10–3.13 needed).",
          hint: "Python 3.14 is installed on this machine, but PyTorch does not ship wheels for it yet. Install Python 3.12 or 3.13 from python.org, then press Install again — the app will find it automatically.",
        },
      };
    }
    send({ phase: "output", line: `Base interpreter: ${base.info.executable} (Python ${base.info.version})` });

    const venvDir = path.join(dirs.root(), "venv");
    const venvPython = venvPythonPath(venvDir);
    const isNew = !fs.existsSync(venvPython);

    if (isNew) {
      send({ phase: "output", line: "Creating an isolated environment for the app (your system Python is not touched)…" });
      const created = await runInstallStep(base.command, [...(base.args || []), "-m", "venv", venvDir], (line) => send({ phase: "output", line }));
      if (created.code !== 0 || !fs.existsSync(venvPython)) {
        send({ phase: "failed", message: "The virtual environment could not be created." });
        return { ok: false, error: { code: "venv_failed", message: "Could not create the app environment (venv).", hint: (created.stderr || created.stdout).split("\n").slice(-4).join(" ") } };
      }
    } else {
      send({ phase: "output", line: "Reusing the existing app environment." });
    }

    // Make sure pip itself is current inside the venv, then install the stack.
    const settings = store.settings().get();
    const cudaTag = settings.cudaWheelTag || null;
    send({ phase: "output", line: cudaTag ? `Installing the ML runtime (CUDA ${cudaTag} wheel index for torch)…` : "Installing the ML runtime (CPU build of torch)…" });

    const packages = ["torch", "transformers", "peft", "accelerate", "safetensors", "numpy", "huggingface_hub", "datasets", "pyarrow", "sentencepiece"];
    if (cudaTag) packages.push("bitsandbytes"); // QLoRA needs it, and it only works on CUDA
    const pipArgs = ["-m", "pip", "install", "--upgrade", ...packages];
    if (cudaTag) pipArgs.push("--index-url", `https://download.pytorch.org/whl/${cudaTag}`, "--extra-index-url", "https://pypi.org/simple");

    installChild = spawnInstall(venvPython, pipArgs, { windowsHide: true });
    const exitCode = await new Promise((resolve) => {
      const timer = setTimeout(() => {
        try { installChild.kill(); } catch { /* already gone */ }
      }, 60 * 60 * 1000); // torch + CUDA wheels can be a multi-gigabyte download
      timer.unref?.();
      installChild.on("error", (error) => {
        clearTimeout(timer);
        resolve(-1);
        send({ phase: "failed", message: error.message });
      });
      installChild.stdout.on("data", (chunk) => String(chunk).split(/\r?\n/).forEach((line) => line.trim() && send({ phase: "output", line: line.trim() })));
      installChild.stderr.on("data", (chunk) => String(chunk).split(/\r?\n/).forEach((line) => line.trim() && send({ phase: "output", line: line.trim() })));
      installChild.on("close", (code) => {
        clearTimeout(timer);
        resolve(code ?? -1);
      });
    });
    installChild = null;

    if (exitCode !== 0) {
      send({ phase: "failed", message: `pip exited with code ${exitCode}.` });
      return { ok: false, error: { code: "pip_failed", message: `pip exited with code ${exitCode}.`, hint: "The full pip log is shown above. The most common cause is a proxy/firewall blocking the download." } };
    }

    // Verify inside the venv before declaring success: pip exit 0 with a
    // half-broken environment used to report "done" and training failed later
    // with missing_dependency. Never let that happen again.
    send({ phase: "output", line: "Verifying the installed packages…" });
    python.setInterpreter(venvPython);
    const check = await python.run(["env-check"]);
    const deps = check.result && check.result.dependencies;
    const missing = deps ? deps.missing_core || [] : null;
    if (!check.result || (missing && missing.length)) {
      const names = missing && missing.length ? missing.join(", ") : (check.error && check.error.message) || "unknown";
      send({ phase: "failed", message: `Installed, but still missing: ${names}.` });
      return { ok: false, error: { code: "install_incomplete", message: `Installed, but still missing: ${names}.`, hint: "Press Install again to retry the missing packages, or check the pip log above for blocked downloads." } };
    }

    // Point the app at the venv so every backend call uses it from now on.
    store.settings().merge({ interpreterPath: venvPython });
    send({ phase: "output", line: `Done. The app environment is at ${venvDir}` });
    send({ phase: "done", exitCode: 0, venv: venvDir });
    return ok({ exitCode: 0, venv: venvDir, interpreter: venvPython });
  }

  ipcMain.handle("zeqou:env:installStatus", wrap(async () => {
    const venvPython = venvPythonPath(path.join(dirs.root(), "venv"));
    return ok({ running: Boolean(installChild), venvExists: fs.existsSync(venvPython), venvPython });
  }));

  /* ------------------------------------------------------------ hardware */
  ipcMain.handle("zeqou:hardware:sample", wrap(async () => ok({ sample: await hardware.sample() })));

  // Always-on GPU telemetry: one nvidia-smi call every few seconds, pushed to
  // every window. The renderer keeps a flat series for the whole session, and
  // jobs.js adds its denser per-run sampler on top while a run is live.
  const GPU_SAMPLE_INTERVAL_MS = 3000;
  let gpuTimer = null;
  const startGpuSampler = () => {
    if (gpuTimer) return;
    gpuTimer = setInterval(async () => {
      try {
        const sample = await hardware.sample();
        emit("zeqou:gpu", { runId: null, sample });
      } catch {
        /* the window may be gone; sampling resumes on the next tick */
      }
    }, GPU_SAMPLE_INTERVAL_MS);
    if (gpuTimer.unref) gpuTimer.unref();
  };
  startGpuSampler();

  ipcMain.handle("zeqou:hardware:detect", wrap(async () => {
    const response = await python.run(["hardware"]);
    if (!response.result) {
      return ok({ hardware: null, error: response.error || { message: "Python backend unavailable." } });
    }
    return ok({ hardware: response.result });
  }));

  /* ------------------------------------------------------------ datasets */
  ipcMain.handle("zeqou:datasets:list", wrap(() => ok({ datasets: registry.listDatasets() })));

  ipcMain.handle("zeqou:datasets:import", wrap(async (paths) => {
    const list = Array.isArray(paths) ? paths : [paths];
    const imported = [];
    const failed = [];
    for (const target of list) {
      // eslint-disable-next-line no-await-in-loop - order matters for the UI list
      const result = await registry.importDataset(target);
      if (result.ok) imported.push(result.dataset);
      else failed.push({ path: target, error: result.error });
    }
    emit("zeqou:datasets:changed", { imported });
    return ok({ imported, failed });
  }));

  // Hub datasets are referenced by id and downloaded by the backend on demand.
  ipcMain.handle("zeqou:datasets:addHf", wrap((payload) => {
    const result = registry.addHfDataset(payload.id, payload.split);
    if (result.ok) emit("zeqou:datasets:changed", { dataset: result.dataset });
    return result;
  }));

  ipcMain.handle("zeqou:datasets:validate", wrap((payload) => registry.validateDataset(
    payload.id || payload.path,
    { contextLength: payload.contextLength, mapping: payload.mapping, format: payload.format },
  )));

  ipcMain.handle("zeqou:datasets:preview", wrap((payload) => registry.previewDataset(
    payload.id || payload.path, payload.limit || 5,
  )));

  ipcMain.handle("zeqou:datasets:remove", wrap((id) => {
    const result = registry.removeDataset(id);
    emit("zeqou:datasets:changed", { removed: id });
    return result;
  }));

  /* --------------------------------------------------------------- models */
  ipcMain.handle("zeqou:models:list", wrap(() => ok({ models: registry.listModels() })));

  ipcMain.handle("zeqou:models:add", wrap(async (payload) => {
    const result = await registry.addModel(payload.source, { hfCache: store.settings().get().hfCacheDir });
    if (result.ok) emit("zeqou:models:changed", { model: result.model });
    return result;
  }));

  ipcMain.handle("zeqou:models:remove", wrap((id) => {
    const result = registry.removeModel(id);
    emit("zeqou:models:changed", { removed: id });
    return result;
  }));

  ipcMain.handle("zeqou:models:exportInfo", wrap((payload) => exporter.describe(payload.modelId)));

  ipcMain.handle("zeqou:models:export", wrap(async (payload) => {
    const result = await exporter.exportModel(payload);
    if (result.ok) emit("zeqou:models:changed", { exported: result.modelId });
    return result;
  }));

  ipcMain.handle("zeqou:models:inspect", wrap(async (payload) => {
    const settings = store.settings().get();
    const response = await python.run([
      "inspect-model", "--source", payload.source,
      ...(settings.hfCacheDir ? ["--hf-cache", settings.hfCacheDir] : []),
    ]);
    if (!response.result) return { ok: false, error: response.error };
    return ok({ info: response.result });
  }));

  /* ------------------------------------------------------------- projects */
  ipcMain.handle("zeqou:projects:list", wrap(() => ok({ projects: registry.listProjects() })));
  ipcMain.handle("zeqou:projects:get", wrap((id) => ok({ project: registry.getProject(id) })));
  ipcMain.handle("zeqou:projects:create", wrap((patch) => {
    const project = registry.createProject(patch);
    emit("zeqou:projects:changed", { project });
    return ok({ project });
  }));
  ipcMain.handle("zeqou:projects:update", wrap((payload) => {
    const result = registry.updateProject(payload.id, payload.patch);
    if (result.ok) emit("zeqou:projects:changed", { project: result.project });
    return result;
  }));
  ipcMain.handle("zeqou:projects:remove", wrap((id) => {
    const result = registry.removeProject(id);
    emit("zeqou:projects:changed", { removed: id });
    return result;
  }));

  /* ------------------------------------------------------------ training */
  ipcMain.handle("zeqou:training:autoConfig", wrap(async (payload) => {
    const settings = store.settings().get();
    const args = ["auto-config"];
    if (payload.baseModel) args.push("--base-model", payload.baseModel);
    if (payload.datasetPath) args.push("--dataset-path", payload.datasetPath);
    if (payload.contextLength) args.push("--context-length", String(payload.contextLength));
    if (payload.method) args.push("--method", payload.method);
    if (settings.hfCacheDir) args.push("--hf-cache", settings.hfCacheDir);
    // With automatic configuration off the backend returns the documented
    // defaults and says so, instead of values tuned to this machine.
    if (payload.baseline) args.push("--baseline");

    const response = await python.run(args);
    if (!response.result) {
      return {
        ok: false,
        error: response.error || {
          code: "backend_unavailable",
          message: "Automatic configuration needs the Python backend.",
          hint: response.stderr ? response.stderr.split("\n").slice(-4).join(" ") : "",
        },
      };
    }
    return ok(response.result);
  }));

  ipcMain.handle("zeqou:training:estimate", wrap(async (payload) => {
    const args = ["estimate"];
    if (payload.config) args.push("--config", JSON.stringify(payload.config));
    if (payload.modelInfo) args.push("--model-info", JSON.stringify(payload.modelInfo));
    if (payload.availableVramMb) args.push("--available-vram", String(payload.availableVramMb));
    const response = await python.run(args);
    if (!response.result) return { ok: false, error: response.error };
    return ok({ estimate: response.result });
  }));

  ipcMain.handle("zeqou:training:check", wrap(async (payload) => {
    // Pre-flight without touching the GPU: validate the dataset, inspect the
    // model, then let the backend report blocking issues.
    const results = { issues: [], dataset: null, model: null, estimate: null };
    if (payload.datasetPath) {
      const validation = await registry.validateDataset(payload.datasetPath, {
        contextLength: payload.contextLength,
        mapping: payload.mapping,
      });
      results.dataset = validation.report || null;
      if (!validation.ok) results.issues.push(validation.error);
    }
    if (payload.baseModel) {
      const response = await python.run(["inspect-model", "--source", payload.baseModel]);
      results.model = response.result || null;
      if (!response.result) results.issues.push(response.error);
    }
    return ok(results);
  }));

  ipcMain.handle("zeqou:training:start", wrap(async (payload) => {
    const settings = store.settings().get();
    let hardwareSnapshot = payload.hardware;
    if (!hardwareSnapshot) {
      const response = await python.run(["hardware"]);
      hardwareSnapshot = response.result || null;
    }
    return jobs.startRun({ ...payload, hardware: hardwareSnapshot, settings });
  }));

  ipcMain.handle("zeqou:training:progress", wrap((runId) => {
    const snapshot = jobs.getLiveSnapshot(runId);
    const record = jobs.getRun(runId);
    return ok({ record, live: snapshot });
  }));

  ipcMain.handle("zeqou:training:stop", wrap((runId) => jobs.requestStop(runId)));
  ipcMain.handle("zeqou:training:pause", wrap((runId) => jobs.requestPause(runId)));
  ipcMain.handle("zeqou:training:resume", wrap((runId) => jobs.resumeRun(runId)));

  ipcMain.handle("zeqou:training:runs", wrap((limit) => ok({ runs: jobs.listRuns(limit) })));
  ipcMain.handle("zeqou:training:run", wrap((runId) => ok({ run: jobs.getRun(runId) })));
  ipcMain.handle("zeqou:training:deleteRun", wrap((runId) => jobs.deleteRun(runId)));
  ipcMain.handle("zeqou:training:log", wrap((payload) => ok(
    jobs.readLog(payload.runId, payload.lines || 800),
  )));
  ipcMain.handle("zeqou:training:checkpoints", wrap((runDir) => jobs.listCheckpoints(runDir)));

  /* ----------------------------------------------------------- inference */
  ipcMain.handle("zeqou:inference:load", wrap((payload) => inference.load(payload.modelDir)));
  ipcMain.handle("zeqou:inference:generate", wrap((payload) => inference.generate(payload)));
  ipcMain.handle("zeqou:inference:unload", wrap(() => inference.unload()));
  ipcMain.handle("zeqou:inference:status", wrap(() => ok({ status: inference.status() })));
}

module.exports = { register };
