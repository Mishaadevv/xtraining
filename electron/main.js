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

const { ensureDirs } = require("./lib/paths");
const ipc = require("./ipc");
const jobs = require("./lib/jobs");
const inference = require("./lib/inference");

const DEV_SERVER_URL = process.env.ZEQOUX_DEV_SERVER_URL;
const IS_DEV = Boolean(process.env.ZEQOUX_DEV) || Boolean(DEV_SERVER_URL);

let mainWindow = null;

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
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });

  app.whenReady().then(() => {
    ensureDirs();
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
