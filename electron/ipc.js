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
    return ok({ settings: store.settings().get() });
  }));

  /* ---------------------------------------------------------- environment */
  ipcMain.handle("zeqou:env:detect", wrap(async (options = {}) => {
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

  // Install the ML runtime with the selected interpreter, streaming pip's
  // output to the renderer as it goes. One install at a time.
  let installChild = null;
  ipcMain.handle("zeqou:env:installRuntime", wrap(async () => {
    if (installChild) {
      return { ok: false, error: { code: "install_busy", message: "An installation is already running." } };
    }
    const interpreter = await python.resolve();
    if (!interpreter) {
      return {
        ok: false,
        error: {
          code: "no_python",
          message: "No Python interpreter is available.",
          hint: "Install Python 3.10–3.13 and select it in Settings → Environment.",
        },
      };
    }
    const settings = store.settings().get();
    const planResponse = await python.run(["install-plan", ...(settings.cudaWheelTag ? ["--cuda-tag", settings.cudaWheelTag] : [])]);
    const plan = planResponse.result;
    if (!plan || !Array.isArray(plan.argv) || plan.argv.length < 3) {
      return { ok: false, error: planResponse.error || { code: "no_plan", message: "The install plan could not be built." } };
    }

    // plan.argv[0] is the interpreter the *backend* resolved; run pip with the
    // exact interpreter so packages land where the app will look for them.
    const pipArgv = ["-m", "pip", "install", "--upgrade", ...plan.argv.slice(3)];
    // A CUDA tag means torch must come from the pytorch wheel index first.
    if (settings.cudaWheelTag && plan.argv.includes("--index-url")) {
      pipArgv.push("--index-url", `https://download.pytorch.org/whl/${settings.cudaWheelTag}`);
    }

    const { spawn } = require("node:child_process");
    installChild = spawn(interpreter.command, [...(interpreter.args || []), ...pipArgv], {
      env: python.buildEnv(),
      windowsHide: true,
    });
    emit("zeqou:runtime:install", { phase: "started", command: [interpreter.command, ...pipArgv].join(" ") });

    const onLine = (line) => {
      if (line && line.trim()) emit("zeqou:runtime:install", { phase: "output", line: line.trim() });
    };
    installChild.stdout.on("data", (chunk) => String(chunk).split(/\r?\n/).forEach(onLine));
    installChild.stderr.on("data", (chunk) => String(chunk).split(/\r?\n/).forEach(onLine));

    return await new Promise((resolve) => {
      const timer = setTimeout(() => {
        try { installChild.kill(); } catch { /* already gone */ }
      }, 45 * 60 * 1000); // torch downloads are huge; be patient
      timer.unref?.();
      installChild.on("error", (error) => {
        installChild = null;
        clearTimeout(timer);
        emit("zeqou:runtime:install", { phase: "failed", message: error.message });
        resolve({ ok: false, error: { code: "spawn_failed", message: error.message } });
      });
      installChild.on("close", (code) => {
        installChild = null;
        clearTimeout(timer);
        emit("zeqou:runtime:install", { phase: code === 0 ? "done" : "failed", exitCode: code });
        resolve(code === 0
          ? ok({ exitCode })
          : { ok: false, error: { code: "pip_failed", message: `pip exited with code ${code}. See the log for the reason.` } });
      });
    });
  }));

  ipcMain.handle("zeqou:env:installStatus", wrap(() => ok({ running: Boolean(installChild) })));

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
