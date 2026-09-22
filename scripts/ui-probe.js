/**
 * Temporary layout probe (not shipped).
 *
 *   npx electron scripts/_ui-probe.js
 *
 * Loads the built renderer with the real preload bridge, walks to the screens
 * that were reported broken, and prints the measured geometry of the
 * Hugging Face Hub rows. A screenshot would only say "looks wrong"; numbers say
 * whether a field overflows its row, which is the bug being fixed.
 */
const { app, BrowserWindow } = require("electron");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const SANDBOX = fs.mkdtempSync(path.join(os.tmpdir(), "zeqoux-probe-"));
app.setPath("userData", SANDBOX);

const { ensureDirs } = require("../electron/lib/paths");
const ipc = require("../electron/ipc");

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function evaluate(win, expression) {
  return win.webContents.executeJavaScript(expression, true);
}

async function waitFor(win, expression, timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    try {
      if (await evaluate(win, expression)) return true;
    } catch {
      /* the page may still be loading */
    }
    if (Date.now() > deadline) return false;
    await sleep(250);
  }
}

/** Click the first element whose trimmed text matches, from the document root. */
const clickByText = (selector, text) => `
  (() => {
    const target = [...document.querySelectorAll(${JSON.stringify(selector)})]
      .find((node) => node.textContent.trim().startsWith(${JSON.stringify(text)}));
    if (!target) return false;
    target.click();
    return true;
  })()
`;

const measureRow = (label) => `
  (() => {
    const field = document.querySelector('input[aria-label="${label}"]');
    if (!field) return null;
    const row = field.closest("div.flex");
    const split = document.querySelector('input[aria-label="Hub split"]');
    const button = [...document.querySelectorAll("button")].find((node) => node.textContent.trim() === "Add");
    const box = (node) => {
      if (!node) return null;
      const rect = node.getBoundingClientRect();
      return { left: Math.round(rect.left), right: Math.round(rect.right), width: Math.round(rect.width) };
    };
    return {
      field: box(field),
      split: box(split),
      button: box(button),
      row: box(row),
      rowOverflow: row ? row.scrollWidth - row.clientWidth : null,
    };
  })()
`;

app.whenReady().then(async () => {
  ensureDirs();
  ipc.register();

  const win = new BrowserWindow({
    width: 1440,
    height: 920,
    show: false,
    backgroundColor: "#0a0a0c",
    webPreferences: {
      preload: path.join(__dirname, "..", "electron", "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  await win.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  const ready = await waitFor(win, "document.querySelectorAll('button').length > 3");
  await sleep(1200);
  const out = { ready };
  out.dom = await evaluate(win, `
    (() => ({
      title: document.title,
      rootChildren: document.querySelector("#root")?.children.length ?? -1,
      buttons: [...document.querySelectorAll("button")].slice(0, 24).map((n) => n.textContent.trim().slice(0, 40)),
      links: [...document.querySelectorAll("a")].slice(0, 12).map((n) => n.textContent.trim().slice(0, 40)),
    }))()
  `);

  // ---- New training → Dataset step -------------------------------------
  out.clickedNav = await evaluate(win, clickByText("button", "New training"));
  await sleep(600);
  // Exact match: the sidebar's "Datasets" item must not be picked up here.
  out.clickedStep = await evaluate(win, `
    (() => {
      const target = [...document.querySelectorAll("button")].find((node) => node.textContent.trim() === "Dataset");
      if (!target) return false;
      target.click();
      return true;
    })()
  `);
  await sleep(600);
  out.wizardRow = await evaluate(win, measureRow("Hub dataset id"));
  out.wizardBody = await evaluate(win, `
    (() => {
      const el = document.querySelector("input[aria-label='Hub dataset id']");
      const panel = el && el.closest(".zq-panel");
      return panel ? { panelWidth: Math.round(panel.getBoundingClientRect().width), panelOverflow: panel.scrollWidth - panel.clientWidth } : null;
    })()
  `);

  // ---- Datasets → Add from Hub modal -----------------------------------
  out.clickedDatasets = await evaluate(win, clickByText("button", "Datasets"));
  await sleep(700);
  out.clickedHub = await evaluate(win, clickByText("button", "Add from Hub"));
  await sleep(700);
  out.modalRow = await evaluate(win, measureRow("Hub dataset id"));

  process.stdout.write(`\n${JSON.stringify(out, null, 2)}\n`);
  app.exit(0);
});
