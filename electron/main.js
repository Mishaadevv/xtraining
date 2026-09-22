/**
 * ZeqouXTraining — Electron main process.
 *
 * Security posture: context isolation on, node integration off, sandbox on, and
 * the renderer only ever talks through the named preload bridge. Navigation away
 * from the app and new windows are both refused; external links open in the
 * user's browser instead.
 */
const { app, BrowserWindow, Menu, shell, dialog } = require("electron");
const path = require("node:path");
const fs = require("node:fs");

const os = require("node:os");

const { ensureDirs, pythonPackageDir } = require("./lib/paths");
const ipc = require("./ipc");
const jobs = require("./lib/jobs");
const inference = require("./lib/inference");

const DEV_SERVER_URL = process.env.ZEQOUX_DEV_SERVER_URL;
const IS_DEV = Boolean(process.env.ZEQOUX_DEV) || Boolean(DEV_SERVER_URL);

let mainWindow = null;

/**
 * `--self-test`: check the installed copy without opening a window.
 *
 * A packaged GUI app has no console, so the report is written to a file in the
 * temp directory (and to stdout, when there is one). It answers the question a
 * screenshot cannot: can this build actually start its Python backend?
 *
 *   ZeqouXTraining.exe --self-test
 */
async function selfTest() {
  const python = require("./lib/python");
  const registry = require("./lib/registry");
  const target = path.join(os.tmpdir(), "zeqouxtraining-self-test.json");
  const report = {
    app: { name: app.getName(), version: app.getVersion(), packaged: app.isPackaged },
    platform: process.platform,
    backendDir: pythonPackageDir(),
    backendPresent: false,
    interpreter: null,
    spawnOk: false,
    datasetFormats: null,
    datasetFolder: null,
    scanned: null,
    error: null,
    // The report is written after every step, so a hang still shows how far the
    // copy under test got.
    steps: [],
  };

  const flush = () => {
    try {
      fs.writeFileSync(target, JSON.stringify(report, null, 2), "utf8");
    } catch {
      /* the exit code still reports the result */
    }
  };
  const step = (name, data) => {
    report.steps.push({ name, at: Date.now(), data: data === undefined ? null : data });
    flush();
  };
  const withTimeout = (promise, ms, label) => Promise.race([
    promise,
    new Promise((resolve) => {
      const timer = setTimeout(() => resolve({ timedOut: label }), ms);
      timer.unref?.();
    }),
  ]);

  flush();
  try {
    step("paths", { backendDir: report.backendDir, userData: app.getPath("userData") });

    const health = await withTimeout(python.health(), 45000, "health");
    if (health.timedOut) throw new Error("Discovering an interpreter timed out.");
    report.backendPresent = health.backendPresent;
    report.interpreter = health.python;
    step("health", { available: health.python.available, spawnedWith: health.python.spawnedWith || null });

    const formats = await withTimeout(python.run(["dataset-formats"]), 60000, "dataset-formats");
    if (formats.timedOut) throw new Error("Starting the Python backend timed out.");
    report.spawnOk = Boolean(formats.result);
    report.datasetFormats = formats.result
      ? {
          ok: true,
          types: formats.result.formats.map((format) => format.id),
          extensions: formats.result.extensions.length,
          exportFormats: formats.result.export_formats,
        }
      : null;
    if (!formats.result) report.error = formats.error;
    step("dataset-formats", { ok: report.spawnOk, error: formats.error || null });

    report.datasetFolder = registry.datasetsFolder();
    const scan = registry.scanDatasetsFolder();
    report.scanned = { ok: scan.ok, found: scan.found, added: scan.added };
    step("scan", report.scanned);
  } catch (error) {
    report.error = { code: error.code || "internal", message: error.message, hint: error.hint || "" };
    step("failed", report.error);
  }

  flush();
  process.stdout.write(`\n${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`\nreport: ${target}\n`);
  return report.spawnOk && report.backendPresent ? 0 : 1;
}

const SELF_TEST = process.argv.includes("--self-test");

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1080,
    minHeight: 700,
    show: false,
    backgroundColor: "#0a0a0c",
    title: "ZeqouXTraining",
    icon: path.join(__dirname, "..", "ico.png"),
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      spellcheck: false,
      webSecurity: true,
    },
  });

  mainWindow.once("ready-to-show", () => {
    mainWindow.show();
  });

  // Refuse to navigate the app window anywhere, and send real links to the OS.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/.test(url)) shell.openExternal(url);
    return { action: "deny" };
  });

  mainWindow.webContents.on("will-navigate", (event, url) => {
    const isDevUrl = DEV_SERVER_URL && url.startsWith(DEV_SERVER_URL);
    if (isDevUrl) return;
    if (url.startsWith("file://")) return;
    event.preventDefault();
    if (/^https?:/.test(url)) shell.openExternal(url);
  });

  // Block every permission request: this app needs no camera, mic or location.
  mainWindow.webContents.session.setPermissionRequestHandler((_wc, _permission, callback) => {
    callback(false);
  });

  if (DEV_SERVER_URL) {
    mainWindow.loadURL(DEV_SERVER_URL).catch(showLoadFailure);
    if (IS_DEV) mainWindow.webContents.openDevTools({ mode: "detach" });
  } else {
    const indexPath = path.join(__dirname, "..", "dist", "index.html");
    if (!fs.existsSync(indexPath)) {
      dialog.showErrorBox(
        "ZeqouXTraining",
        "The interface bundle was not found.\n\nRun `npm run build` (or `npm run dev`) first.",
      );
      app.quit();
      return;
    }
    mainWindow.loadFile(indexPath).catch(showLoadFailure);
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function showLoadFailure(error) {
  dialog.showErrorBox("ZeqouXTraining", `The interface could not be loaded.\n\n${error.message}`);
}

function buildMenu() {
  const isMac = process.platform === "darwin";
  const template = [
    ...(isMac ? [{ role: "appMenu" }] : []),
    {
      label: "File",
      submenu: [
        { label: "Open app data folder", click: () => shell.openPath(app.getPath("userData")) },
        { type: "separator" },
        isMac ? { role: "close" } : { role: "quit" },
      ],
    },
    {
      label: "View",
      submenu: [
        { role: "reload" },
        { role: "forceReload" },
        { type: "separator" },
        { role: "resetZoom" },
        { role: "zoomIn" },
        { role: "zoomOut" },
        { type: "separator" },
        { role: "togglefullscreen" },
        ...(IS_DEV ? [{ type: "separator" }, { role: "toggleDevTools" }] : []),
      ],
    },
    {
      label: "Help",
      submenu: [
        {
          label: "Zeqou ecosystem",
          click: () => shell.openExternal("https://mishaadevv.github.io/zeqou/"),
        },
        { role: "about" },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// A second instance would fight over the run directories; focus the first one.
// The self-test deliberately skips the lock so it still answers while the app
// is open.
const gotLock = SELF_TEST || app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(async () => {
    ensureDirs();

    if (SELF_TEST) {
      // No window, no menu: just tell the caller whether this copy works.
      try {
        app.exit(await selfTest());
      } catch (error) {
        process.stdout.write(`self-test crashed: ${error && error.message}\n`);
        app.exit(1);
      }
      return;
    }

    ipc.register();
    buildMenu();
    createWindow();

    app.on("activate", () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });

  app.on("window-all-closed", () => {
    if (process.platform !== "darwin") app.quit();
  });

  app.on("before-quit", () => {
    // Ask live runs to checkpoint, then release the inference runtime.
    try {
      jobs.stopAll();
    } catch {
      /* best effort */
    }
    try {
      inference.dispose();
    } catch {
      /* best effort */
    }
  });
}
