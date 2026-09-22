/**
 * Python runtime bridge.
 *
 * Two execution modes are offered, both speaking the backend's newline-delimited
 * JSON protocol on stdout:
 *
 *   run()    — one-shot command, resolves with the terminal `result`/`error`
 *   stream() — long-lived command (training, inference), emits events as they
 *              arrive and exposes stop/pause control
 *
 * Interpreter resolution is venv-first: the configured interpreter wins (after
 * a successful "Install now" that is the app's own venv under userData/venv),
 * then the system candidates are probed in order. The app never writes into
 * the user's system Python.
 *
 * stderr is *never* parsed as protocol; it is forwarded as log lines. That is
 * what keeps third-party library output from corrupting the channel.
 */
const { spawn } = require("node:child_process");
const readline = require("node:readline");
const fs = require("node:fs");
const path = require("node:path");

const { pythonPackageDir, isInsideAsar, dirs } = require("./paths");
const { settings, getSecret } = require("./store");

// Probed in order; the first interpreter that answers wins. The commands that
// are normally on PATH come first, because that is what a user expects the app
// to use; absolute install locations are appended afterwards so a machine whose
// PATH never mentions Python still works.
const CANDIDATES = [
  { command: "python", args: [], label: "python" },
  { command: "python3", args: [], label: "python3" },
  { command: "py", args: ["-3.13"], label: "py -3.13" },
  { command: "py", args: ["-3.12"], label: "py -3.12" },
  { command: "py", args: ["-3.11"], label: "py -3.11" },
  { command: "py", args: ["-3.10"], label: "py -3.10" },
  { command: "py", args: ["-3"], label: "py -3" },
  { command: "py", args: [], label: "py" },
];

/**
 * Python installations are not always on PATH — a per-user install on Windows
 * frequently is not. These directories are read directly so the app can find an
 * interpreter the shell would not.
 */
function knownInstallDirs() {
  if (process.platform === "win32") {
    return [
      process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, "Programs", "Python"),
      process.env.ProgramFiles && path.join(process.env.ProgramFiles, "Python"),
      process.env.ProgramFiles,
      process.env["ProgramFiles(x86)"],
      // WindowsApps is deliberately absent: the install-or-Store aliases there
      // are 0-byte stubs, and probing one can pop the Store instead of answering.
    ].filter(Boolean);
  }
  return ["/usr/bin", "/usr/local/bin", "/opt/homebrew/bin", "/usr/local/opt/python/bin"];
}

/** Absolute interpreter paths found in the well-known install directories. */
function knownInstallPaths() {
  const found = [];
  for (const dir of knownInstallDirs()) {
    let entries;
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue; // the directory does not exist on this machine
    }
    for (const entry of entries) {
      const binary = process.platform === "win32" ? "python.exe" : null;
      if (entry.isFile()) {
        if (/^python(3(\.\d+)?)?(\.exe)?$/i.test(entry.name)) found.push(path.join(dir, entry.name));
      } else if (entry.isDirectory() && /^[Pp]ython3(\d+|\.\d+)?$/.test(entry.name)) {
        found.push(binary
          ? path.join(dir, entry.name, binary)
          : path.join(dir, entry.name, "bin", "python3"));
      }
    }
  }
  return found.filter((candidate) => {
    try {
      return fs.statSync(candidate).isFile();
    } catch {
      return false;
    }
  });
}

/**
 * `py -0p` lists every registered installation. It is the most reliable source
 * on Windows and it costs one short process.
 */
async function launcherInstallPaths() {
  if (process.platform !== "win32") return [];
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn("py", ["-0p"], { windowsHide: true, timeout: 10000 });
    } catch {
      resolve([]);
      return;
    }
    let stdout = "";
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      const paths = [];
      for (const line of stdout.split(/\r?\n/)) {
        const match = line.match(/[A-Za-z]:\\[^\r\n]*python\.exe/i);
        if (match) paths.push(match[0].trim());
      }
      resolve(paths);
    };
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.on("error", finish);
    child.on("close", finish);
    setTimeout(finish, 10000).unref?.();
  });
}

// Joined with newlines: the interpreter is fed this as a single `-c` program, so
// omitting the separator would glue the imports onto the print call.
const PROBE = [
  "import json, sys, platform",
  "print(json.dumps({",
  "  'executable': sys.executable,",
  "  'version': platform.python_version(),",
  "  'prefix': sys.prefix,",
  "  'implementation': platform.python_implementation(),",
  "  'debug': hasattr(sys, 'gettotalrefcount'),",
  "}))",
].join("\n");

let resolvedInterpreter = null; // { command, args, label, info }
let probeCache = new Map();
let lastInterpreter = null; // the one the most recent spawn used, for diagnostics
let lastStart = null; // { command, cwd } of the most recent spawn attempt

/**
 * How to actually start an interpreter that already answered the probe.
 *
 * The probe reports `sys.executable`, and that absolute path is what every
 * later call uses. This is what makes the app immune to the classic
 * "spawn python ENOENT": the bare name only has to work once, during probing.
 */
function invocation(interpreter) {
  if (!interpreter) return null;
  const discovered = interpreter.info && interpreter.info.executable;
  if (typeof discovered === "string" && discovered.trim()) {
    try {
      if (fs.existsSync(discovered)) return { command: discovered, args: [] };
    } catch {
      /* fall through to the probed command */
    }
  }
  return { command: interpreter.command, args: interpreter.args || [] };
}

/** Drop every cached result so the next call re-probes from scratch. */
function invalidate() {
  resolvedInterpreter = null;
  probeCache = new Map();
}

function baseArgs(start) {
  // -X utf8 keeps non-ASCII dataset content and paths intact on Windows.
  return [...(start.args || []), "-X", "utf8", "-m", "zeqouxtraining.cli"];
}

function buildEnv(extra = {}) {
  const config = settings().get();
  const env = {
    ...process.env,
    PYTHONIOENCODING: "utf-8",
    PYTHONUNBUFFERED: "1",
    PYTHONUTF8: "1",
    // Keep Hugging Face from writing progress bars into the protocol.
    HF_HUB_DISABLE_PROGRESS_BARS: "1",
    HF_HUB_DISABLE_TELEMETRY: "1",
    TRANSFORMERS_NO_ADVISORY_WARNINGS: "1",
    TOKENIZERS_PARALLELISM: "false",
    TQDM_DISABLE: "1",
    ...extra,
  };

  const hfHome = config.hfCacheDir || dirs.hfCache();
  env.HF_HOME = hfHome;
  env.HUGGINGFACE_HUB_CACHE = path.join(hfHome, "hub");
  env.TRANSFORMERS_CACHE = path.join(hfHome, "transformers");
  if (config.advanced && config.advanced.trustRemoteCode) env.ZEQOUX_TRUST_REMOTE_CODE = "1";

  const token = getSecret("hfToken");
  if (token) env.HF_TOKEN = token;
  return env;
}

function isEmptyDirOrMissing(target) {
  try {
    return !fs.existsSync(target) || fs.readdirSync(target).length === 0;
  } catch {
    return true;
  }
}

async function probe(interpreter) {
  const key = `${interpreter.command} ${(interpreter.args || []).join(" ")}`;
  if (probeCache.has(key)) return probeCache.get(key);

  const result = await new Promise((resolve) => {
    let settled = false;
    let child;
    try {
      child = spawn(interpreter.command, [...(interpreter.args || []), "-c", PROBE], {
        windowsHide: true,
        timeout: 20000,
      });
    } catch (error) {
      resolve({ available: false, reason: error.message });
      return;
    }

    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });

    const finish = (payload) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(payload);
    };

    const timer = setTimeout(() => {
      try { child.kill(); } catch { /* already gone */ }
      finish({ available: false, reason: "Timed out while probing the interpreter." });
    }, 20000);

    child.on("error", (error) => {
      const missing = error.code === "ENOENT";
      finish({
        available: false,
        reason: missing ? `'${interpreter.command}' was not found on PATH.` : error.message,
      });
    });
    child.on("close", (code) => {
      if (code !== 0) {
        finish({
          available: false,
          reason: (stderr || stdout).trim().split("\n").slice(-3).join(" ") || `exit code ${code}`,
        });
        return;
      }
      try {
        const info = JSON.parse(stdout.trim().split("\n").pop());
        finish({ available: true, info });
      } catch {
        finish({ available: false, reason: "Unexpected probe output." });
      }
    });
  });

  const payload = { ...interpreter, label: interpreter.label || key, ...result };
  probeCache.set(key, payload);
  return payload;
}

/**
 * Every interpreter worth probing, in priority order: the one the user chose,
 * the commands on PATH, then absolute paths found on disk.
 */
async function candidates() {
  const configured = settings().get().interpreterPath;
  const list = [];
  const seen = new Set();
  const push = (candidate) => {
    const key = `${candidate.command} ${(candidate.args || []).join(" ")}`;
    if (seen.has(key)) return;
    seen.add(key);
    list.push(candidate);
  };

  if (configured) push({ command: configured, args: [], label: configured, configured: true });
  for (const candidate of CANDIDATES) push(candidate);
  for (const executable of [...knownInstallPaths(), ...(await launcherInstallPaths())]) {
    push({ command: executable, args: [], label: executable });
  }
  return list;
}

/** Probe every candidate (including a configured one) and report each result. */
async function discover(force = false) {
  if (force) invalidate();
  const results = [];
  for (const candidate of await candidates()) {
    // eslint-disable-next-line no-await-in-loop - sequential on purpose, keeps output ordered
    results.push(await probe(candidate));
  }
  return results;
}

/** The interpreter the app will actually use, or null with a reason. */
async function resolve(force = false) {
  if (resolvedInterpreter && !force) return resolvedInterpreter;
  if (force) invalidate();

  for (const candidate of await candidates()) {
    // eslint-disable-next-line no-await-in-loop - sequential on purpose
    const result = await probe(candidate);
    if (result.available) {
      resolvedInterpreter = result;
      return result;
    }
  }
  return null;
}

/** Why nothing worked, spelled out for a human. */
async function diagnosis() {
  const tried = await candidates();
  const reasons = [];
  for (const candidate of tried) {
    // eslint-disable-next-line no-await-in-loop - only reached when nothing works
    const result = await probe(candidate);
    if (result.reason) reasons.push(`${result.label}: ${result.reason}`);
  }
  return {
    message: "No working Python interpreter was found.",
    hint: process.platform === "win32"
      ? "Install Python 3.10–3.13 from python.org (tick 'Add python.exe to PATH'), then press "
        + "Install in Settings → Environment. Already installed? Choose the interpreter by hand "
        + "under Settings → Environment."
      : "Install Python 3.10–3.13 with your package manager, then press Install in "
        + "Settings → Environment, or pick the interpreter by hand.",
    tried: reasons.slice(0, 12),
  };
}

function setInterpreter(executablePath) {
  invalidate();
  settings().merge({ interpreterPath: executablePath || null });
}

async function health() {
  const interpreter = await resolve();
  const packageDir = pythonPackageDir();
  const hasPackage = fs.existsSync(path.join(packageDir, "zeqouxtraining", "cli.py"));
  return {
    python: interpreter
      ? {
          available: true,
          ...interpreter.info,
          label: interpreter.label,
          command: interpreter.command,
          args: interpreter.args || [],
          executable: (interpreter.info && interpreter.info.executable) || interpreter.command,
          spawnedWith: invocation(interpreter).command,
        }
      : { available: false, ...(await diagnosis()) },
    packageDir,
    backendPresent: hasPackage,
    cacheDir: dirs.hfCache(),
    ready: Boolean(interpreter && hasPackage),
  };
}

async function spawnBackend(commandArgs, options = {}) {
  const interpreter = await resolve();
  if (!interpreter) {
    const info = await diagnosis();
    const error = new Error(info.message);
    error.hint = info.hint;
    error.code = "no_python";
    error.tried = info.tried;
    throw error;
  }

  const packageDir = pythonPackageDir();
  if (!fs.existsSync(path.join(packageDir, "zeqouxtraining", "cli.py"))) {
    throw new Error(`The Python backend package was not found at ${packageDir}.`);
  }
  if (isInsideAsar(packageDir)) {
    // The OS cannot chdir into an archive; spawning from there fails with a
    // misleading ENOENT. Say what is actually wrong instead.
    const error = new Error(`The Python backend is inside app.asar (${packageDir}).`);
    error.code = "backend_unavailable";
    error.hint = "Reinstall the app: a packaged build must keep the backend in resources/python.";
    throw error;
  }

  lastInterpreter = interpreter;
  const start = invocation(interpreter);
  lastStart = { command: start.command, cwd: packageDir };
  return spawn(
    start.command,
    [...baseArgs(start), ...commandArgs],
    {
      cwd: packageDir,
      env: buildEnv(options.env),
      windowsHide: true,
    },
  );
}

/**
 * One-shot command: collects protocol events and resolves with the final
 * `result` or `error` payload.
 */
async function run(commandArgs, options = {}) {
  const started = Date.now();
  let child;
  try {
    child = await spawnBackend(commandArgs, options);
  } catch (error) {
    return {
      ok: false,
      events: [],
      result: null,
      error: {
        code: error.code || "no_python",
        message: error.message,
        hint: error.hint || "",
        tried: error.tried || [],
        traceback: "",
      },
      stderr: (error.tried || []).join("\n"),
      code: -1,
      durationMs: Date.now() - started,
    };
  }

  const events = [];
  const stderrLines = [];
  let result = null;
  let errorPayload = null;

  const rl = readline.createInterface({ input: child.stdout });
  rl.on("line", (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    let parsed;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      // Not protocol output; keep it out of the event stream.
      stderrLines.push(trimmed);
      if (options.onLogLine) options.onLogLine(trimmed, "warn");
      return;
    }
    events.push(parsed);
    if (parsed.event === "result") result = parsed.detail;
    else if (parsed.event === "error") errorPayload = parsed.detail;
    if (options.onEvent) options.onEvent(parsed);
  });

  const rlErr = readline.createInterface({ input: child.stderr });
  rlErr.on("line", (line) => {
    stderrLines.push(line);
    if (options.onLogLine) options.onLogLine(line, "info");
  });

  const code = await new Promise((resolve) => {
    child.on("error", (error) => {
      stderrLines.push(String(error.message));
      resolve(-1);
    });
    child.on("close", (exitCode) => resolve(exitCode ?? -1));
  });

  // Nothing was spoken on the protocol: the backend never got as far as
  // answering. Say what actually happened instead of letting the caller invent
  // a reason, and drop the cached interpreter so the next call re-discovers it.
  if (!result && !errorPayload) {
    const detail = stderrLines.join(" ").trim();
    const notFound = /ENOENT|cannot find the file|not recognized|No such file/i.test(detail);
    const used = (lastStart && lastStart.command)
      || (lastInterpreter && invocation(lastInterpreter).command)
      || "python";
    const cwd = (lastStart && lastStart.cwd) || "";
    invalidate();
    return {
      ok: false,
      events,
      result: null,
      error: {
        code: notFound ? "no_python" : "backend_unavailable",
        message: notFound
          ? `The Python backend could not be started (${used}).`
          : "The Python backend stopped before it answered.",
        hint: notFound
          ? `Started as '${used}'${cwd ? ` from '${cwd}'` : ""}. Check that both the interpreter and the backend folder still exist — reinstalling the app repairs a damaged install, and Settings → Environment lets you pick the interpreter by hand.`
          : (detail.slice(-500) || "No output was produced. Try again with the environment panel open."),
        traceback: "",
      },
      stderr: stderrLines.join("\n"),
      code,
      durationMs: Date.now() - started,
    };
  }

  return {
    ok: code === 0 && !errorPayload,
    events,
    result,
    error: errorPayload,
    stderr: stderrLines.join("\n"),
    code,
    durationMs: Date.now() - started,
  };
}

/**
 * Long-lived command. Returns a handle with the child process so the caller can
 * stop it, plus a promise that resolves when it exits.
 */
function stream(commandArgs, handlers = {}) {
  const started = Date.now();
  let child = null;
  let settled = false;

  const ready = spawnBackend(commandArgs, handlers).then((spawned) => {
    child = spawned;

    const rl = readline.createInterface({ input: spawned.stdout });
    rl.on("line", (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      let parsed;
      try {
        parsed = JSON.parse(trimmed);
      } catch {
        if (handlers.onStderr) handlers.onStderr(trimmed);
        return;
      }
      if (handlers.onEvent) handlers.onEvent(parsed);
    });

    const rlErr = readline.createInterface({ input: spawned.stderr });
    rlErr.on("line", (line) => {
      if (line.trim() && handlers.onStderr) handlers.onStderr(line);
    });

    return spawned;
  });

  const done = ready.then(
    (spawned) =>
      new Promise((resolve) => {
        spawned.on("error", (error) => {
          if (settled) return;
          settled = true;
          resolve({ code: -1, error: error.message, durationMs: Date.now() - started });
        });
        spawned.on("close", (code, signal) => {
          if (settled) return;
          settled = true;
          resolve({ code: code ?? -1, signal, durationMs: Date.now() - started });
        });
      }),
  );

  return {
    ready,
    done,
    get pid() {
      return child ? child.pid : null;
    },
    /** Best-effort termination; the backend also watches its stop file. */
    kill(signal = "SIGTERM") {
      try {
        if (child && !child.killed) child.kill(signal);
      } catch {
        /* already gone */
      }
    },
    write(payload) {
      try {
        if (child && child.stdin.writable) child.stdin.write(`${JSON.stringify(payload)}\n`);
        return true;
      } catch {
        return false;
      }
    },
    closeStdin() {
      try {
        if (child) child.stdin.end();
      } catch {
        /* already closed */
      }
    },
  };
}

module.exports = {
  discover,
  resolve,
  setInterpreter,
  health,
  diagnosis,
  invocation,
  invalidate,
  run,
  stream,
  buildEnv,
  isEmptyDirOrMissing,
};
