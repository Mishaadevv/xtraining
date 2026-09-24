/**
 * Shared plumbing for the tests that drive the *packed* application.
 *
 * Both `verify-packed.mjs` and `verify-update.mjs` need the same things: start
 * `release/win-unpacked` with a throwaway profile, attach over the Chrome DevTools
 * protocol, evaluate expressions in the page and clean up without ever deciding
 * the exit code by accident. That lives here so the tests themselves stay about
 * what they assert.
 */
import { execFileSync, spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

export const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

export const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export function packedDir() {
  return path.join(root, "release", process.platform === "win32" ? "win-unpacked" : `${process.platform}-unpacked`);
}

export function packedExecutable() {
  return path.join(packedDir(), process.platform === "win32" ? "ZeqouXTraining.exe" : "ZeqouXTraining");
}

/** Collects pass/fail lines and turns them into an exit code. */
export class Reporter {
  constructor(title) {
    this.title = title;
    this.checks = [];
  }

  record(name, ok, detail = "") {
    this.checks.push({ name, ok, detail });
    console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
    return ok;
  }

  note(line) {
    console.log(`      ${line}`);
  }

  finish() {
    const failed = this.checks.filter((check) => !check.ok);
    console.log(`\n${this.title}: ${this.checks.length - failed.length}/${this.checks.length} checks passed`);
    for (const check of failed) console.log(`  failed: ${check.name} — ${check.detail}`);
    return failed.length;
  }
}

export function freePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}

/**
 * A running packed application, controlled through the DevTools protocol.
 *
 * The profile and workspace are throwaway directories under the system temp
 * folder: the test must never touch the real settings, the real workspace or a
 * real updater cache.
 */
export class PackedApp {
  constructor({ env = {}, label = "app" } = {}) {
    this.label = label;
    this.tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), "zxtrain-packed-"));
    this.workspace = path.join(this.tmpRoot, "workspace");
    this.profile = path.join(this.tmpRoot, "profile");
    this.channel = path.join(this.tmpRoot, "channel");
    for (const dir of [this.workspace, this.profile]) fs.mkdirSync(dir, { recursive: true });
    this.env = { ...env, ZEQOUX_WORKSPACE: this.workspace, ZEQOUX_USER_DATA: this.profile };
    this.child = null;
    this.socket = null;
    this.nextId = 1;
    this.pending = new Map();
    this.pageErrors = [];
  }

  get executable() {
    return packedExecutable();
  }

  /** Settings the app reads at boot, written before it starts. */
  seedSettings(patch) {
    fs.writeFileSync(path.join(this.profile, "settings.json"), `${JSON.stringify(patch, null, 2)}\n`, "utf8");
  }

  async start(extraArgs = []) {
    // The app is driven through its DevTools socket using the WebSocket client
    // that is built into Node — stable since 22.4, and behind
    // `--experimental-websocket` before that. Saying which Node is needed beats
    // a bare "WebSocket is not defined" from somewhere inside the attach code.
    if (typeof WebSocket === "undefined") {
      throw new Error(
        `driving the packed app needs a built-in WebSocket: Node 22.4 or newer (running ${process.version})`,
      );
    }
    const port = await freePort();
    this.debugPort = port;
    console.log(`\nstarting ${path.relative(root, this.executable)}  (debug port ${port})`);
    this.child = spawn(this.executable, [`--remote-debugging-port=${port}`, "--no-sandbox", ...extraArgs], {
      env: { ...process.env, ...this.env },
      stdio: "ignore",
      windowsHide: false,
    });
    this.child.on("error", (error) => console.log(`  spawn error: ${error.message}`));

    const target = await this.#attach(port);
    this.socket = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((resolve, reject) => {
      this.socket.addEventListener("open", resolve, { once: true });
      this.socket.addEventListener("error", () => reject(new Error("the DevTools socket refused to open")), { once: true });
    });
    this.socket.addEventListener("message", (event) => this.#onMessage(JSON.parse(event.data)));
    await this.send("Runtime.enable");
    await this.send("Log.enable");
    console.log(`attached to ${target.url}`);
    return target;
  }

  async #attach(port) {
    const started = Date.now();
    for (;;) {
      try {
        const response = await fetch(`http://127.0.0.1:${port}/json/list`);
        const targets = await response.json();
        const page = targets.find((target) => target.type === "page" && target.webSocketDebuggerUrl);
        if (page) return page;
      } catch {
        /* the app has not opened its debug port yet */
      }
      if (Date.now() - started > 120_000) throw new Error("the packed app never exposed a page over the DevTools protocol");
      await sleep(500);
    }
  }

  #onMessage(message) {
    if (message.id !== undefined) {
      const entry = this.pending.get(message.id);
      if (!entry) return;
      this.pending.delete(message.id);
      if (message.error) entry.reject(new Error(message.error.message));
      else entry.resolve(message.result ?? {});
      return;
    }
    if (message.method === "Runtime.exceptionThrown") {
      this.pageErrors.push(message.params.exceptionDetails.exception?.description ?? message.params.exceptionDetails.text);
    }
    if (message.method === "Runtime.consoleAPICalled" && message.params.type === "error") {
      this.pageErrors.push(message.params.args.map((arg) => arg.value ?? arg.description ?? arg.type).join(" "));
    }
    if (message.method === "Log.entryAdded" && message.params.entry.level === "error") {
      this.pageErrors.push(message.params.entry.text);
    }
    this.onEvent?.(message);
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.pending.set(id, { resolve, reject });
      this.socket.send(JSON.stringify({ id, method, params }));
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`${method} did not answer within 60 s`));
        }
      }, 60_000).unref?.();
    });
  }

  /** Evaluate an expression in the page. Expressions only — the payload is
   *  wrapped in parentheses, so a stray `;` is a syntax error. */
  async js(expression) {
    const response = await this.send("Runtime.evaluate", {
      expression: `(async () => (${expression}))()`,
      awaitPromise: true,
      returnByValue: true,
    });
    if (response.exceptionDetails) {
      throw new Error(response.exceptionDetails.exception?.description ?? response.exceptionDetails.text);
    }
    return response.result?.value;
  }

  async waitFor(expression, timeoutMs, label, { quiet = true } = {}) {
    const started = Date.now();
    for (;;) {
      try {
        if (await this.js(expression)) return true;
      } catch {
        /* the page may still be loading; try again */
      }
      if (Date.now() - started > timeoutMs) {
        if (!quiet) console.log(`  timed out after ${Math.round(timeoutMs / 1000)} s waiting for ${label}`);
        return false;
      }
      await sleep(250);
    }
  }

  /** Wait for the interface itself, which is what every later check depends on. */
  async waitForInterface(timeoutMs = 300_000) {
    return this.waitFor(
      `document.querySelectorAll("aside nav button").length > 0 || document.body.innerText.includes("could not reach its Python engine")`,
      timeoutMs,
      "the interface to boot",
      { quiet: false },
    );
  }

  kill() {
    if (!this.child?.pid) return;
    try {
      if (process.platform === "win32") execFileSync("taskkill", ["/PID", String(this.child.pid), "/T", "/F"], { stdio: "ignore" });
      else process.kill(-this.child.pid, "SIGKILL");
    } catch {
      /* already gone */
    }
    this.child = null;
  }

  /** Reap the app and the throwaway profile. Cleanup never decides the exit
   *  code: a killed process can hold a handle for a moment on Windows, so the
   *  removal is retried and a leftover directory is reported, not thrown. */
  async cleanup() {
    try {
      this.kill();
    } catch {
      /* nothing left to kill */
    }
    try {
      this.socket?.close();
    } catch {
      /* the socket may never have opened */
    }
    for (let attempt = 0; attempt < 5; attempt += 1) {
      try {
        fs.rmSync(this.tmpRoot, { recursive: true, force: true });
        return;
      } catch {
        await sleep(800);
      }
    }
    console.log(`  (left ${this.tmpRoot} behind)`);
  }
}

/** Run a body with a packed app, always cleaning up and always exiting. */
export async function withPackedApp(options, body) {
  const app = new PackedApp(options);
  const watchdog = setTimeout(() => {
    console.log("FAIL  the test did not finish within 10 minutes");
    app.cleanup().finally(() => process.exit(2));
  }, 600_000);
  watchdog.unref?.();

  let failure = null;
  try {
    await body(app);
  } catch (error) {
    failure = error;
  }
  clearTimeout(watchdog);
  await app.cleanup();
  return failure;
}
