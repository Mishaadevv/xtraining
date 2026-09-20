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

const { pythonPackageDir, dirs } = require("./paths");
const { settings, getSecret } = require("./store");

// Probed in order; the first interpreter that answers wins.
const CANDIDATES = [
  { command: "python", args: [], label: "python" },
  { command: "python3", args: [], label: "python3" },
  { command: "py", args: ["-3.13"], label: "py -3.13" },
  { command: "py", args: ["-3.12"], label: "py -3.12" },
  { command: "py", args: ["-3.11"], label: "py -3.11" },
  { command: "py", args: ["-3.10"], label: "py -3.10" },
  { command: "py", args: [], label: "py" },
];

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

function baseArgs(interpreter) {
  // -X utf8 keeps non-ASCII dataset content and paths intact on Windows.
  return [...(interpreter.args || []), "-X", "utf8", "-m", "zeqouxtraining.cli"];
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

/** Probe every candidate (including a configured one) and report each result. */
async function discover(force = false) {
  if (force) probeCache = new Map();
  const configured = settings().get().interpreterPath;
  const list = [];
  if (configured) list.push({ command: configured, args: [], label: configured, configured: true });
  for (const candidate of CANDIDATES) {
    if (configured && candidate.command === configured) continue;
    list.push(candidate);
  }

  const results = [];
  for (const candidate of list) {
    // eslint-disable-next-line no-await-in-loop - sequential on purpose, keeps output ordered
    results.push(await probe(candidate));
  }
  return results;
}

/** The interpreter the app will actually use, or null with a reason. */
async function resolve(force = false) {
  if (resolvedInterpreter && !force) return resolvedInterpreter;

  const configured = settings().get().interpreterPath;
  if (configured) {
    const result = await probe({ command: configured, args: [], label: configured });
    if (result.available) {
      resolvedInterpreter = result;
      return result;
    }
  }

  for (const candidate of CANDIDATES) {
    // eslint-disable-next-line no-await-in-loop - sequential on purpose
    const result = await probe(candidate);
    if (result.available) {
      resolvedInterpreter = result;
      return result;
    }
  }
  return null;
}

function setInterpreter(executablePath) {
  resolvedInterpreter = null;
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
        }
      : { available: false, reason: "No working Python interpreter was found." },
    packageDir,
    backendPresent: hasPackage,
    cacheDir: dirs.hfCache(),
    ready: Boolean(interpreter && hasPackage),
  };
}

async function spawnBackend(commandArgs, options = {}) {
  const interpreter = await resolve();
  if (!interpreter) {
    throw new Error(
      "No Python interpreter was found. Install Python 3.10+ and select it in Settings → Environment.",
    );
  }

  const packageDir = pythonPackageDir();
  if (!fs.existsSync(path.join(packageDir, "zeqouxtraining", "cli.py"))) {
    throw new Error(`The Python backend package was not found at ${packageDir}.`);
  }

  return spawn(
    interpreter.command,
    [...baseArgs(interpreter), ...commandArgs],
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
      error: { code: "no_python", message: error.message, hint: "", traceback: "" },
      stderr: "",
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
  run,
  stream,
  buildEnv,
  isEmptyDirOrMissing,
};
