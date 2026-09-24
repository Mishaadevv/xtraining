/**
 * Launch test — runs as the Electron main process:
 *
 *     npm run build        # the renderer must exist first
 *     electron scripts/launch.mjs
 *
 * `smoke.mjs` drives the main-process modules directly; this test goes one level
 * further and starts the *actual application*: the real `electron/main.js`, the
 * real preload bridge, the real window and the real Python engine behind it. It
 * then asserts that the interface came up, that `window.zx` answers with real
 * engine data, that navigating the sidebar renders every declared page, and that
 * the command palette opens, filters and closes.
 *
 * A window will briefly appear on screen. Everything runs in a throwaway
 * workspace under the system temp directory.
 */
import { app, BrowserWindow } from "electron";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
// Static import on purpose: the app must register its own `ready` handler while
// this module is still evaluating. The variables set below are only read later,
// inside `bootstrap()`, so they are in place in time.
import "../electron/main.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const expectedVersion = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8")).version;

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "zxtrain-launch-"));
const workspace = path.join(tmpRoot, "workspace");
const userData = path.join(tmpRoot, "userdata");
fs.mkdirSync(userData, { recursive: true });
fs.mkdirSync(workspace, { recursive: true });
app.setPath("userData", userData);
process.env.ZEQOUX_WORKSPACE = workspace;

const logFile = process.env.ZEQOUX_LAUNCH_LOG || path.join(tmpRoot, "launch.log");

/** Electron's Windows build is a GUI binary: its stdout never reaches the
 *  terminal that launched it, so every line is also written to a log file. */
function stage(line) {
  try {
    fs.appendFileSync(path.join(tmpRoot, "stage.log"), `${new Date().toISOString()} ${line}\n`, "utf8");
  } catch {
    /* nothing else we can do */
  }
}

function log(line) {
  const text = `${line}\n`;
  try {
    fs.appendFileSync(logFile, text, "utf8");
  } catch {
    /* the log is a convenience, never a failure */
  }
  try {
    process.stdout.write(text);
  } catch {
    /* stdout may not exist */
  }
}

stage(`module evaluating; tmp ${tmpRoot}`);

/**
 * Leave nothing behind and never let a stuck child keep the terminal hostage:
 * windows are destroyed, the app is asked to exit, and a hard exit is scheduled
 * in case anything else is still holding the process open.
 */
function finish(code) {
  stage(`finishing with code ${code}`);
  try {
    for (const win of BrowserWindow.getAllWindows()) win.destroy();
  } catch {
    /* the window may already be gone */
  }
  setTimeout(() => {
    stage("hard exit");
    process.exit(code);
  }, 2_000).unref?.();
  app.exit(code);
}

const checks = [];
function record(name, ok, detail = "") {
  checks.push({ name, ok, detail });
  log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
}

/* ------------------------------------------------------------ renderer IO --- */

let window = null;
const consoleErrors = [];
const consoleWarnings = [];

function attachDiagnostics(win) {
  const wc = win.webContents;
  wc.on("console-message", (...args) => {
    // Electron <= 34 passes (event, level, message, line, source); 35+ passes
    // a details object instead. Accept both so the version can be upgraded.
    const [second, third] = [args[1], args[2]];
    const structured = second && typeof second === "object";
    const level = structured ? String(second.level) : second;
    const message = structured ? second.message : third;
    const source = structured ? second.sourceId : args[4];
    const line = structured ? second.lineNumber : args[3];
    const text = `${String(message)} (${source ?? "?"}:${line ?? "?"})`;
    if (level === 3 || level === "error") consoleErrors.push(text);
    else if (level === 2 || level === "warning") consoleWarnings.push(text);
  });
  wc.on("render-process-gone", (_event, details) => {
    consoleErrors.push(`render-process-gone: ${JSON.stringify(details)}`);
  });
  wc.on("did-fail-load", (_event, code, description, url) => {
    consoleErrors.push(`did-fail-load ${code} ${description} ${url}`);
  });
  wc.on("unresponsive", () => consoleErrors.push("the renderer became unresponsive"));
}

/** Evaluate an expression in the page. Promises are awaited by Electron. */
function js(code) {
  return window.webContents.executeJavaScript(code, true);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function waitForWindow(timeoutMs = 300_000) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const timer = setInterval(() => {
      const [first] = BrowserWindow.getAllWindows();
      if (first) {
        clearInterval(timer);
        resolve(first);
        return;
      }
      if (Date.now() - started > timeoutMs) {
        clearInterval(timer);
        reject(new Error(`no window appeared within ${Math.round(timeoutMs / 1000)} s`));
      }
    }, 100);
  });
}

/** Poll a boolean expression in the page until it is true. */
async function waitFor(expression, timeoutMs, label) {
  const started = Date.now();
  for (;;) {
    let value = false;
    try {
      value = Boolean(await js(`(() => (${expression}))()`));
    } catch {
      value = false;
    }
    if (value) return true;
    if (Date.now() - started > timeoutMs) {
      log(`  timed out after ${Math.round(timeoutMs / 1000)} s waiting for ${label}`);
      return false;
    }
    await sleep(200);
  }
}

/* ------------------------------------------------------------------ tests --- */

async function main() {
  window = await waitForWindow();
  record("the application opened a window", Boolean(window));
  attachDiagnostics(window);

  const booted = await waitFor(
    `document.querySelectorAll("aside nav button").length > 0 || document.body.innerText.includes("could not reach its Python engine")`,
    300_000,
    "the interface to boot",
  );
  record("the interface finished booting", booted);
  if (!booted) throw new Error("the renderer never booted");

  const onScreen = await js("document.body.innerText");
  record(
    "no engine boot error is on screen",
    !String(onScreen).includes("could not reach its Python engine") &&
      !String(onScreen).includes("The desktop bridge is not available"),
    String(onScreen).slice(0, 120).replace(/\s+/g, " "),
  );

  // The bridge: context-isolated, namespaced, and answering for real.
  const bridge = await js(`(() => {
    const zx = window.zx;
    return {
      present: Boolean(zx) && typeof zx.engine?.call === "function" && typeof zx.jobs?.start === "function",
      nodeRequireExposed: typeof window.require === "function" || typeof window.process?.versions?.node === "string",
      channels: zx ? Object.keys(zx).sort() : [],
    };
  })()`);
  record(
    "the preload bridge is exposed and Node is not",
    bridge.present === true && bridge.nodeRequireExposed !== true,
    `namespaces: ${(bridge.channels ?? []).join(", ")}`,
  );

  const info = await js("window.zx.app.info()");
  record(
    "the bridge reports the real app info",
    info?.ok === true && info.data?.workspace === workspace && info.data?.version === expectedVersion,
    `version ${info?.data?.version}, electron ${info?.data?.electron}, workspace ${info?.data?.workspace}`,
  );

  // A full round trip: renderer -> IPC -> engine subprocess -> back.
  const capabilities = await js(
    `window.zx.engine.call("engine.capabilities", {}, { timeout: 240000 })`,
  );
  record(
    "the engine answers through the renderer",
    capabilities?.ok === true && Boolean(capabilities.data?.hardware?.cpu?.model),
    capabilities?.ok
      ? `cpu ${capabilities.data.hardware.cpu.model}, ${capabilities.data.backends?.length ?? 0} backends, python ${capabilities.data.hardware.os?.python ?? "?"}`
      : JSON.stringify(capabilities?.error),
  );

  const jobs = await js("window.zx.jobs.list()");
  record(
    "the job list answers in a fresh workspace",
    jobs?.ok === true && Array.isArray(jobs.data) && jobs.data.length === 0,
    `${jobs?.data?.length ?? "?"} job(s)`,
  );

  /* ----------------------------------------------------------- every page --- */

  const navCount = await js(`document.querySelectorAll("aside nav button").length`);
  record("the sidebar lists every page", navCount >= 19, `${navCount} nav items`);

  let previousHeading = null;
  for (let index = 0; index < navCount; index += 1) {
    const label = String(
      await js(`(document.querySelectorAll("aside nav button")[${index}].innerText || "").trim()`),
    );
    await js(`document.querySelectorAll("aside nav button")[${index}].click(); true`);

    const rendered = await waitFor(
      `(() => {
         const main = document.querySelector("main");
         const heading = main?.querySelector("h1");
         const text = (main?.innerText || "").trim();
         return Boolean(heading) && heading.textContent.trim().length > 0 && text.length > 120;
       })()`,
      30_000,
      `page “${label}” to render`,
    );
    const heading = String(await js(`(document.querySelector("main h1")?.textContent || "").trim()`));
    const length = Number(await js(`(document.querySelector("main")?.innerText || "").trim().length`));
    record(
      `page “${label}” renders`,
      rendered && heading !== previousHeading,
      rendered ? `h1 “${heading}”, ${length} characters` : "no heading or empty content",
    );
    previousHeading = heading;
    await sleep(120);
  }

  /* ------------------------------------------------------ command palette --- */

  await js(`window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", ctrlKey: true, bubbles: true })); true`);
  const opened = await waitFor(`Boolean(document.querySelector('input[placeholder^="Search pages"]'))`, 10_000, "the palette to open");
  const filtered = await js(`(() => {
    const input = document.querySelector('input[placeholder^="Search pages"]');
    if (!input) return { ok: false };
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
    setter.call(input, "hardware");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    return { ok: true };
  })()`);
  await sleep(300);
  const paletteText = String(await js(`(document.body.innerText || "")`));
  await js(`window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })); true`);
  await sleep(300);
  const closed = await js(`!document.querySelector('input[placeholder^="Search pages"]')`);
  record(
    "the command palette opens, filters and closes",
    opened && filtered?.ok === true && paletteText.includes("Hardware") && closed === true,
    opened ? "Ctrl+K opened it, “hardware” matched, Escape closed it" : "it never opened",
  );

  /* ------------------------------------------------------------ diagnostics --- */

  record(
    "the renderer reported no console errors",
    consoleErrors.length === 0,
    consoleErrors.length ? consoleErrors.slice(0, 5).join(" | ") : `${consoleWarnings.length} warning(s)`,
  );
  if (consoleWarnings.length) log(`  renderer warnings: ${consoleWarnings.slice(0, 8).join(" | ")}`);

  const alive = !window.isDestroyed() && !window.webContents.isCrashed();
  record("the window is still alive at the end", alive, alive ? "" : "it crashed during the run");
}

/* --------------------------------------------------------------- runner --- */

const watchdog = setTimeout(() => {
  log("FAIL  the launch test did not finish within 12 minutes");
  for (const check of checks) log(`  so far: ${check.ok ? "PASS" : "FAIL"} ${check.name}`);
  finish(2);
}, 720_000);
watchdog.unref?.();

async function run() {
  stage("app ready; running the launch test");
  let failure = null;
  try {
    await main();
    stage("checks finished");
  } catch (error) {
    failure = error;
    stage(`throw: ${error?.stack ?? error}`);
    record("the launch test finished without throwing", false, String(error?.stack ?? error));
  }

  clearTimeout(watchdog);
  const failed = checks.filter((check) => !check.ok);
  log(`\nlaunch: ${checks.length - failed.length}/${checks.length} checks passed`);
  for (const check of failed) log(`  failed: ${check.name} — ${check.detail}`);
  if (!failed.length && !failure) {
    log(`launch: log was ${logFile}`);
    // The window has to go before the workspace can be removed: what Electron
    // keeps open in `userData/Cache` cannot be unlinked while it runs. A
    // leftover directory must never turn a passing run into a hang either.
    try {
      for (const win of BrowserWindow.getAllWindows()) win.destroy();
    } catch {
      /* already gone */
    }
    await sleep(700);
    let lastError = null;
    for (let attempt = 0; attempt < 3; attempt += 1) {
      try {
        fs.rmSync(tmpRoot, { recursive: true, force: true });
        lastError = null;
        break;
      } catch (error) {
        lastError = error;
        await sleep(500);
      }
    }
    if (lastError) log(`launch: left ${tmpRoot} behind — ${lastError?.message}`);
  } else {
    log(`launch: workspace left in ${tmpRoot}`);
    log(`launch: log kept at ${logFile}`);
  }
  finish(failed.length || failure ? 1 : 0);
}

// Another instance would steal the single-instance lock and this run would hang
// waiting for a window that the other process owns.
if (!app.requestSingleInstanceLock()) {
  log("FAIL  ZeqouXTraining is already running — close it and try again.");
  finish(3);
} else {
  process.on("uncaughtException", (error) => {
    stage(`uncaughtException: ${error?.stack ?? error}`);
    log(`FAIL  the launch test threw: ${error?.message ?? error}`);
    finish(1);
  });
  // NOTE: `ready` is emitted only after this module finishes evaluating, so it
  // must not be awaited at the top level here.
  stage("registering ready handler");
  app.whenReady().then(run, (error) => {
    stage(`ready failed: ${error?.stack ?? error}`);
    finish(3);
  });
}
