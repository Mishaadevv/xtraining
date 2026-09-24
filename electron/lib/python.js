/**
 * The bridge to the Python engine.
 *
 * Responsibilities:
 *  - find the Python interpreters this machine actually has (never assume one),
 *  - pick the best one for the job (an interpreter with PyTorch beats a newer one without),
 *  - run engine commands and jobs, streaming their output,
 *  - turn every failure into a structured error the UI can explain.
 */
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { engineDir } from "./paths.js";

// Each line matters: this is a real script, and a missing newline here once
// made every probe fail, which looked like "no Python installed".
const PROBE = [
  "import json, sys, importlib.util",
  "payload = {",
  "  'version': '.'.join(map(str, sys.version_info[:3])),",
  "  'major': sys.version_info[0],",
  "  'minor': sys.version_info[1],",
  "  'executable': sys.executable,",
  "  'inVenv': sys.prefix != getattr(sys, 'base_prefix', sys.prefix),",
  "  'torch': importlib.util.find_spec('torch') is not None,",
  "  'transformers': importlib.util.find_spec('transformers') is not None,",
  "  'peft': importlib.util.find_spec('peft') is not None,",
  "  'zxtrain': importlib.util.find_spec('zxtrain') is not None,",
  "}",
  "print(json.dumps(payload))",
].join("\n");

export function engineEnv(extra = {}) {
  const dir = engineDir();
  const existing = process.env.PYTHONPATH ? path.delimiter + process.env.PYTHONPATH : "";
  return {
    ...process.env,
    PYTHONPATH: dir + existing,
    PYTHONUNBUFFERED: "1",
    PYTHONIOENCODING: "utf-8",
    PYTHONDONTWRITEBYTECODE: "1",
    ...extra,
  };
}

let cache = { at: 0, interpreters: [] };

export function cachedInterpreters() {
  return cache.interpreters;
}

export function rememberInterpreter(info) {
  cache = { at: Date.now(), interpreters: [info] };
}

function run(executable, args, options = {}) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn(executable, args, {
        env: engineEnv(options.env),
        cwd: options.cwd ?? engineDir(),
        windowsHide: true,
      });
    } catch (error) {
      resolve({ code: -1, stdout: "", stderr: String(error?.message ?? error) });
      return;
    }
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill();
      resolve({ code: -2, stdout, stderr: `${stderr}\ntimed out after ${options.timeout ?? 30000} ms` });
    }, options.timeout ?? 30000);
    child.stdout?.on("data", (chunk) => (stdout += chunk.toString()));
    child.stderr?.on("data", (chunk) => (stderr += chunk.toString()));
    child.on("error", (error) => {
      clearTimeout(timer);
      resolve({ code: -1, stdout, stderr: String(error?.message ?? error) });
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      resolve({ code: code ?? 0, stdout, stderr });
    });
    if (options.stdin !== undefined) {
      child.stdin?.end(options.stdin);
    }
  });
}

async function probe(executable, label) {
  const result = await run(executable, ["-c", PROBE], { timeout: 25000 });
  if (result.code !== 0) return null;
  const line = result.stdout.trim().split("\n").pop();
  let info;
  try {
    info = JSON.parse(line);
  } catch {
    return null;
  }
  if (!fs.existsSync(info.executable)) return null;
  // The engine itself is pure Python, so any interpreter can run inference-less
  // work; PyTorch only ships wheels up to 3.13 today. Both facts are scored, not
  // assumed: an interpreter with torch always outranks a newer one without it.
  const torchCompatible = info.minor <= 13;
  return {
    ...info,
    label,
    torchCompatible,
    note: torchCompatible ? "" : "PyTorch publishes no wheels for this Python version yet.",
    score: (info.zxtrain ? 4 : 0) + (info.torch ? 8 : 0) + (info.transformers ? 2 : 0) +
      (info.peft ? 1 : 0) + (info.inVenv ? 1 : 0) + (torchCompatible ? 4 : -8) +
      Math.min(info.minor, 13) * 0.1,
  };
}

function candidatePaths() {
  const seen = new Set();
  const candidates = [];
  const add = (executable, label) => {
    if (!executable) return;
    const key = path.resolve(executable);
    if (seen.has(key)) return;
    seen.add(key);
    candidates.push({ executable, label });
  };

  add(process.env.ZEQOUX_PYTHON, "ZEQOUX_PYTHON");
  add(process.env.ZX_PYTHON, "ZX_PYTHON");
  add(process.env.PYTHON, "PYTHON");
  add(process.env.PYTHON3, "PYTHON3");

  const names = process.platform === "win32"
    ? ["python.exe", "python3.exe", "py.exe"]
    : ["python3.13", "python3.12", "python3.11", "python3.10", "python3", "python"];
  for (const name of names) {
    const found = which(name);
    if (found) add(found, name);
  }
  if (process.platform === "win32") {
    const roots = [
      "C:\\",
      process.env.LOCALAPPDATA ? path.join(process.env.LOCALAPPDATA, "Programs", "Python") : null,
      process.env.PROGRAMFILES ? path.join(process.env.PROGRAMFILES, "Python") : null,
    ].filter(Boolean);
    for (const root of roots) {
      for (const version of ["313", "312", "311", "310", "314"]) {
        const candidate = path.join(root, `Python${version}`, "python.exe");
        if (fs.existsSync(candidate)) add(candidate, `Python ${version}`);
      }
    }
  } else {
    for (const candidate of ["/usr/bin/python3", "/usr/local/bin/python3", "/opt/homebrew/bin/python3", "/usr/bin/python"]) {
      if (fs.existsSync(candidate)) add(candidate, path.basename(candidate));
    }
  }
  return candidates;
}

function which(name) {
  const paths = (process.env.PATH ?? "").split(path.delimiter).filter(Boolean);
  const extensions = process.platform === "win32" ? [""] : [""];
  for (const dir of paths) {
    for (const extension of extensions) {
      const candidate = path.join(dir, name + extension);
      try {
        if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return candidate;
      } catch {
        /* keep looking */
      }
    }
  }
  return null;
}

/** Probe every interpreter, best first. Results are cached for a minute. */
export async function discoverInterpreters({ force = false, preferred = null } = {}) {
  if (!force && Date.now() - cache.at < 60_000 && cache.interpreters.length) return cache.interpreters;
  const found = [];
  const pyLauncher = process.platform === "win32" ? which("py.exe") : null;
  if (pyLauncher) {
    const listed = await run(pyLauncher, ["-0p"], { timeout: 15000 });
    for (const line of listed.stdout.split("\n")) {
      const match = line.match(/-V:([\d.]+)\s+\*?\s*(.+?)\s*$/);
      if (match) found.push({ executable: match[2].trim(), label: `python ${match[1]}` });
    }
  }
  found.push(...candidatePaths());

  const seen = new Set();
  const probes = [];
  const ordered = preferred ? [preferred, ...found.map((item) => item.executable)] : found.map((item) => item.executable);
  for (const executable of ordered) {
    if (!executable) continue;
    let key;
    try {
      key = path.resolve(executable);
    } catch {
      continue;
    }
    if (seen.has(key) || !fs.existsSync(key)) continue;
    seen.add(key);
    probes.push(probe(key, preferred && key === path.resolve(preferred) ? "selected" : path.basename(key)));
  }
  const results = (await Promise.all(probes)).filter(Boolean);
  results.sort((a, b) => b.score - a.score);
  cache = { at: Date.now(), interpreters: results };
  return results;
}

export async function resolveInterpreter(settings) {
  const interpreters = await discoverInterpreters({
    force: false,
    preferred: settings?.pythonPath ?? null,
  });
  if (!interpreters.length) {
    const error = new Error(
      "No Python interpreter was found. Install Python 3.12 or 3.13, or point the app at an interpreter in Settings.",
    );
    error.code = "no_python";
    throw error;
  }
  if (settings?.pythonPath) {
    const exact = interpreters.find((item) => path.resolve(item.executable) === path.resolve(settings.pythonPath));
    if (exact) return exact;
    const fallback = await probe(settings.pythonPath, "selected");
    if (fallback) {
      cache = { at: Date.now(), interpreters: [fallback, ...interpreters] };
      return fallback;
    }
  }
  const best = interpreters.find((item) => item.torch) ?? interpreters[0];
  rememberInterpreter(best);
  return best;
}

/** Run one engine command and return its parsed JSON response. */
export async function engineCall(executable, method, payload = {}, options = {}) {
  const result = await run(executable, ["-m", "zxtrain.cli", method], {
    timeout: options.timeout ?? 180000,
    stdin: JSON.stringify(payload ?? {}),
  });
  const line = result.stdout.trim().split("\n").filter(Boolean).pop() ?? "";
  let parsed;
  try {
    parsed = JSON.parse(line);
  } catch {
    parsed = null;
  }
  if (!parsed) {
    return {
      ok: false,
      error: {
        code: result.code === -2 ? "engine_timeout" : "engine_unavailable",
        message:
          result.code === -2
            ? `The engine did not answer within ${Math.round((options.timeout ?? 180000) / 1000)} seconds.`
            : "The engine produced no usable response.",
        hint:
          "Open Environment to check the Python interpreter, then retry. The raw output is below.",
        detail: `${result.stdout}\n${result.stderr}`.trim().slice(-4000),
      },
    };
  }
  return parsed;
}

/** Spawn a long running engine process (jobs, installs, the inference sidecar). */
export function engineSpawn(executable, args, options = {}) {
  const child = spawn(executable, args, {
    env: engineEnv(options.env),
    cwd: options.cwd ?? engineDir(),
    windowsHide: true,
  });
  return child;
}

export async function pythonSummary(executable) {
  return probe(executable, "active");
}

export function engineDirectory() {
  return engineDir();
}

export function platformNote() {
  return `${os.platform()} ${os.release()} · node ${process.versions.node}`;
}
