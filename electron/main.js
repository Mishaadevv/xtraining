/**
 * ZeqouXTraining — Electron main process.
 *
 * Owns the window, the user's settings, the workspace, the job manager and the
 * inference sidecar. All heavy lifting happens in the Python engine; this file
 * only orchestrates and never invents data.
 */
import { BrowserWindow, Notification, app, dialog, ipcMain, nativeTheme, shell } from "electron";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { JobManager } from "./lib/jobs.js";
import { appVersion, ensureWorkspace, engineDir, projectRoot, rendererIndex, userDataDir } from "./lib/paths.js";
import {
  cachedInterpreters,
  discoverInterpreters,
  engineCall,
  pythonSummary,
  resolveInterpreter,
} from "./lib/python.js";
import { Sidecar } from "./lib/sidecar.js";
import { JsonStore } from "./lib/store.js";
import { Updates } from "./lib/updater.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const devServerUrl = process.env.ZEQOU_DEV_SERVER_URL || null;

// Settings, caches and logs live in the app's profile. Pointing that somewhere
// else keeps automated runs out of the real profile and lets an install be
// carried on a USB stick. This has to happen before the app is ready.
if (process.env.ZEQOUX_USER_DATA) {
  const profile = path.resolve(process.env.ZEQOUX_USER_DATA);
  app.setPath("userData", profile);
  app.setPath("cache", path.join(profile, "cache"));
  // electron-updater resolves its download cache from the OS cache directory
  // (`LOCALAPPDATA` on Windows, `XDG_CACHE_HOME` on Linux) rather than from
  // Electron, so a self-contained profile has to claim that as well — otherwise a
  // test run leaves an installer behind in the real user cache. macOS reads
  // `~/Library/Caches` from the home directory, which this deliberately does not
  // move.
  if (process.platform === "win32") process.env.LOCALAPPDATA = path.join(profile, "cache");
  else if (process.platform === "linux") process.env.XDG_CACHE_HOME = path.join(profile, "cache");
}

let mainWindow = null;
let interpreter = null;
let jobs = null;
let sidecar = null;
let updates = null;
let registryStore = null;
let settingsStore = null;
let workspace = null;

function settingsFile() {
  return path.join(userDataDir(), "settings.json");
}

function registryFile() {
  return path.join(workspace, "registry.json");
}

function createStore() {
  settingsStore = new JsonStore(settingsFile(), {
    version: 2,
    workspace: null,
    pythonPath: null,
    theme: "system",
    defaultBackend: null,
    defaultMethod: "lora",
    defaultDevice: "auto",
    notifications: { jobFinished: true, jobFailed: true, serverStarted: false },
    updateChannel: null,
    checkForUpdatesOnStart: true,
    offlineMode: false,
    logLevel: "INFO",
    checkpointPolicy: { keepLast: 3, protectBest: true },
    window: { width: 1440, height: 920 },
  });
  const settings = settingsStore.read();
  workspace = ensureWorkspace(settings.workspace);
  if (settings.workspace !== workspace) settingsStore.update({ workspace });
  registryStore = new JsonStore(registryFile(), {
    version: 1,
    projects: [],
    models: [],
    datasets: [],
    presets: [],
    evaluations: [],
    servers: [],
    conversations: [],
    workflows: [],
    notes: [],
  });
  return settings;
}

async function bootInterpreter(force = false) {
  const settings = settingsStore.read();
  interpreter = await resolveInterpreter({ ...settings, pythonPath: force ? null : settings.pythonPath });
  jobs?.setExecutable(interpreter.executable);
  sidecar?.setExecutable(interpreter.executable);
  return interpreter;
}

function notify(title, body) {
  try {
    if (Notification.isSupported()) new Notification({ title, body }).show();
  } catch {
    /* notifications are a nicety, never a failure */
  }
}

function send(channel, payload) {
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send(channel, payload);
}

function createWindow() {
  const settings = settingsStore.read();
  mainWindow = new BrowserWindow({
    width: settings.window?.width ?? 1440,
    height: settings.window?.height ?? 920,
    minWidth: 1080,
    minHeight: 680,
    backgroundColor: nativeTheme.shouldUseDarkColors ? "#0b0b0c" : "#f6f6f7",
    show: false,
    autoHideMenuBar: true,
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "default",
    icon: path.join(projectRoot, "ico.png"),
    webPreferences: {
      preload: path.join(here, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      spellcheck: false,
    },
  });

  mainWindow.once("ready-to-show", () => mainWindow?.show());
  mainWindow.on("close", () => {
    try {
      const bounds = mainWindow.getBounds();
      settingsStore.update({ window: { width: bounds.width, height: bounds.height } });
    } catch {
      /* ignore */
    }
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });

  if (devServerUrl) {
    mainWindow.loadURL(devServerUrl);
  } else {
    mainWindow.loadFile(rendererIndex());
  }
}

async function bootstrap() {
  const settings = createStore();
  nativeTheme.themeSource = settings.theme ?? "system";

  jobs = new JobManager({
    workspace,
    onEvent: ({ jobId, event }) => {
      send("jobs:event", { jobId, event });
      if (event?.type === "done") {
        const result = event.result ?? {};
        if (settingsStore.read().notifications?.jobFinished !== false) {
          notify("ZeqouXTraining", `Job ${jobId} finished${result.final_loss ? ` (loss ${result.final_loss})` : ""}.`);
        }
      }
      if (event?.type === "error") {
        if (settingsStore.read().notifications?.jobFailed !== false) {
          notify("ZeqouXTraining", `Job ${jobId} failed: ${event.message}`);
        }
      }
    },
  });
  sidecar = new Sidecar({
    onEvent: (event) => send("sidecar:event", event),
  });
  updates = new Updates({
    workspace,
    onState: (state) => send("updates:state", state),
  });
  updates.configure({
    channelUrl: settings.updateChannel ?? null,
    checkOnStart: settings.checkForUpdatesOnStart !== false,
  });

  await bootInterpreter(false);
  createWindow();

  // Recover jobs that survived the previous session.
  try {
    if (interpreter?.executable) {
      const response = await engineCall(interpreter.executable, "jobs.reconcile", { workspace }, { timeout: 120_000 });
      if (response.ok) {
        await jobs.reattach(response.data.findings);
        send("jobs:reconciled", response.data.findings);
      }
    }
  } catch {
    /* recovery is best effort and must never block startup */
  }
}

/* ------------------------------------------------------------------ IPC --- */

function handle(channel, handler) {
  ipcMain.handle(channel, async (_event, ...args) => {
    try {
      return await handler(...args);
    } catch (error) {
      return {
        ok: false,
        error: {
          code: error?.code ?? "ipc_error",
          message: String(error?.message ?? error),
          hint: error?.hint ?? "",
          detail: error?.stack ?? "",
        },
      };
    }
  });
}

handle("app:info", async () => ({
  ok: true,
  data: {
    version: appVersion(),
    packaged: app.isPackaged,
    platform: process.platform,
    electron: process.versions.electron,
    node: process.versions.node,
    engineDir: engineDir(),
    userData: userDataDir(),
    workspace,
    devServerUrl,
  },
}));

handle("settings:get", async () => ({ ok: true, data: settingsStore.read() }));
handle("settings:set", async (patch) => {
  const updated = settingsStore.update(patch ?? {});
  if (patch?.theme) nativeTheme.themeSource = patch.theme;
  if (patch?.workspace) {
    workspace = ensureWorkspace(patch.workspace);
    jobs.setWorkspace(workspace);
    updates.setWorkspace(workspace);
    registryStore = new JsonStore(registryFile(), registryStore.read());
    settingsStore.update({ workspace });
  }
  if ("updateChannel" in (patch ?? {}) || "checkForUpdatesOnStart" in (patch ?? {})) {
    updates.configure({
      channelUrl: updated.updateChannel ?? null,
      checkOnStart: updated.checkForUpdatesOnStart !== false,
    });
  }
  if (patch?.pythonPath) {
    interpreter = await resolveInterpreter(settingsStore.read());
    jobs.setExecutable(interpreter.executable);
    sidecar.setExecutable(interpreter.executable);
  }
  return { ok: true, data: updated };
});

handle("registry:get", async () => ({ ok: true, data: registryStore.read() }));
handle("registry:set", async (patch) => ({ ok: true, data: registryStore.update(patch ?? {}) }));
handle("registry:mutate", async (collection, mutatorName, payload) => {
  const allowed = ["projects", "models", "datasets", "presets", "evaluations", "servers", "conversations", "workflows", "notes"];
  if (!allowed.includes(collection)) {
    return { ok: false, error: { code: "bad_collection", message: `Unknown collection: ${collection}` } };
  }
  const data = registryStore.mutate((state) => {
    const list = Array.isArray(state[collection]) ? state[collection] : [];
    if (mutatorName === "add") {
      state[collection] = [payload, ...list.filter((item) => item.id !== payload.id)];
    } else if (mutatorName === "update") {
      state[collection] = list.map((item) => (item.id === payload.id ? { ...item, ...payload } : item));
    } else if (mutatorName === "remove") {
      state[collection] = list.filter((item) => item.id !== payload.id);
    }
    return state;
  });
  return { ok: true, data: data[collection] };
});

handle("interpreters:list", async (force) => {
  const interpreters = await discoverInterpreters({ force: Boolean(force) });
  return { ok: true, data: interpreters };
});

handle("engine:call", async (method, payload, options) => {
  if (!interpreter?.executable) await bootInterpreter(true);
  const response = await engineCall(interpreter.executable, method, { workspace, ...(payload ?? {}) }, options ?? {});
  return response;
});

handle("engine:interpreter", async () => ({
  ok: true,
  data: interpreter ? { ...interpreter, cached: cachedInterpreters().length } : null,
}));

handle("engine:probe", async (executable) => ({ ok: true, data: await pythonSummary(executable) }));

handle("dialog:pick-folder", async (options) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openDirectory", ...(options?.properties ?? [])],
    title: options?.title,
    defaultPath: options?.defaultPath ?? workspace,
  });
  return { ok: true, data: result.canceled ? null : result.filePaths[0] };
});

handle("dialog:pick-files", async (options) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ["openFile", "multiSelections"],
    title: options?.title,
    filters: options?.filters,
    defaultPath: options?.defaultPath ?? workspace,
  });
  return { ok: true, data: result.canceled ? [] : result.filePaths };
});

handle("dialog:save-file", async (options) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    title: options?.title,
    defaultPath: options?.defaultPath,
    filters: options?.filters,
  });
  return { ok: true, data: result.canceled ? null : result.filePath };
});

handle("shell:reveal", async (target) => {
  if (!target) return { ok: false, error: { code: "no_path", message: "Nothing to reveal." } };
  shell.showItemInFolder(target);
  return { ok: true, data: target };
});

handle("shell:open-external", async (url) => {
  if (typeof url === "string" && /^https?:\/\//.test(url)) await shell.openExternal(url);
  return { ok: true };
});

handle("fs:read-text", async (file, limit = 200_000) => {
  try {
    const content = fs.readFileSync(file, "utf8").slice(0, limit);
    return { ok: true, data: content };
  } catch (error) {
    return { ok: false, error: { code: "read_failed", message: String(error.message) } };
  }
});

handle("jobs:list", async () => ({ ok: true, data: jobs.list() }));
handle("jobs:start", async (spec) => ({ ok: true, data: jobs.start(spec ?? {}) }));
handle("jobs:control", async (jobId, patch) => ({ ok: true, data: jobs.control(jobId, patch) }));
handle("jobs:terminate", async (jobId, force) => ({ ok: true, data: jobs.terminate(jobId, force) }));
handle("jobs:status", async (jobId) => ({ ok: true, data: jobs.status(jobId) }));
handle("jobs:logs", async (jobId, lines) => ({ ok: true, data: jobs.logs(jobId, lines) }));
handle("jobs:events", async (jobId, limit) => ({ ok: true, data: jobs.events(jobId, limit) }));
handle("jobs:metrics", async (jobId, limit) => ({ ok: true, data: jobs.metrics(jobId, limit) }));

handle("sidecar:load", async (model, backend) => {
  try {
    const result = await sidecar.load(model, backend);
    return { ok: true, data: result };
  } catch (error) {
    return { ok: false, error: error.structured ?? { code: "load_failed", message: String(error.message) } };
  }
});
handle("sidecar:unload", async () => ({ ok: true, data: await sidecar.unload() }));
handle("sidecar:status", async () => ({ ok: true, data: sidecar.info() }));
handle("sidecar:stop", async () => {
  sidecar.stop();
  return { ok: true };
});
handle("sidecar:generate", async (payload) => {
  try {
    const result = await sidecar.generate(payload, (event) => send("sidecar:stream", event));
    return { ok: true, data: result };
  } catch (error) {
    return { ok: false, error: error.structured ?? { code: "generate_failed", message: String(error.message) } };
  }
});
handle("sidecar:evaluate", async (payload) => {
  try {
    const result = await sidecar.evaluate(payload, (event) => send("sidecar:stream", event));
    return { ok: true, data: result };
  } catch (error) {
    return { ok: false, error: error.structured ?? { code: "evaluate_failed", message: String(error.message) } };
  }
});

handle("updates:state", async () => ({ ok: true, data: updates.state() }));
handle("updates:check", async () => ({ ok: true, data: await updates.check({ manual: true }) }));
handle("updates:download", async () => ({ ok: true, data: await updates.download() }));
handle("updates:install", async () => updates.install());

handle("notify", async (title, body) => {
  notify(title, body);
  return { ok: true };
});

/* ------------------------------------------------------------- lifecycle --- */

const singleInstance = app.requestSingleInstanceLock();
if (!singleInstance) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(bootstrap).catch((error) => {
    dialog.showErrorBox("ZeqouXTraining could not start", String(error?.message ?? error));
    app.quit();
  });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });

  app.on("window-all-closed", () => {
    if (process.platform !== "darwin") app.quit();
  });

  app.on("before-quit", () => {
    // Training jobs deliberately keep running: they are engine processes writing
    // to real log files, and the next launch re-attaches to them.
    sidecar?.stop();
  });
}
