/**
 * Filesystem layout for ZeqouXTraining.
 *
 * Everything user-generated lives under Electron's userData directory:
 *   settings.json, secrets.json, datasets.json, models.json, runs.json
 *   datasets/              files the user drops in, scanned into the library
 *   runs/<runId>/          checkpoints, trainer state, logs, metadata
 *   models/                trained models produced by this app
 */
const { app } = require("electron");
const path = require("node:path");
const fs = require("node:fs");

function root() {
  return app.getPath("userData");
}

const dirs = {
  root,
  runs: () => path.join(root(), "runs"),
  models: () => path.join(root(), "models"),
  cache: () => path.join(root(), "cache"),
  hfCache: () => path.join(root(), "cache", "huggingface"),
  logs: () => path.join(root(), "logs"),
  datasets: datasetsDir,
};

const files = {
  settings: () => path.join(root(), "settings.json"),
  secrets: () => path.join(root(), "secrets.json"),
  datasetsIndex: () => path.join(root(), "datasets.json"),
  modelsIndex: () => path.join(root(), "models.json"),
  projectsIndex: () => path.join(root(), "projects.json"),
  runsIndex: () => path.join(root(), "runs.json"),
};

/**
 * True for a path inside an asar archive.
 *
 * `fs` sees through an asar, but the operating system does not: a child process
 * cannot be started with a working directory inside one. Node reports that as
 * "spawn python ENOENT", which reads like a missing interpreter and is not —
 * so an asar path is never used as the backend directory.
 */
function isInsideAsar(target) {
  const value = String(target).replace(/\\/g, "/");
  return value.includes("/app.asar/") && !value.includes("/app.asar.unpacked/");
}

/**
 * The Python backend package — always a real directory on disk.
 *
 * A packaged build copies it next to the app through electron-builder's
 * extraResources (a real folder); development uses the repo folder next to
 * `electron/`. The unpacked-archive location is kept as a last resort.
 */
function pythonPackageDir() {
  const candidates = [
    path.join(process.resourcesPath || "", "python"),
    path.join(process.resourcesPath || "", "app.asar.unpacked", "python"),
    path.join(__dirname, "..", "..", "python"),
  ];
  for (const candidate of candidates) {
    if (isInsideAsar(candidate)) continue;
    try {
      if (fs.existsSync(path.join(candidate, "zeqouxtraining", "cli.py"))) return candidate;
    } catch {
      /* keep looking */
    }
  }
  return candidates[candidates.length - 1];
}

/**
 * The dataset folder the app scans.
 *
 * Deliberately inside userData: the app ships no datasets at all, so the files
 * that appear under Datasets are exactly the ones the user put there. The
 * folder can be pointed somewhere else in Settings → Datasets.
 */
function datasetsDir() {
  return path.join(root(), "datasets");
}

function ensureDirs() {
  for (const dir of [dirs.root(), dirs.runs(), dirs.models(), dirs.cache(), dirs.logs(), dirs.datasets()]) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

/** Turn an arbitrary label into a filesystem-safe folder name. */
function slugify(text, fallback = "run") {
  const slug = String(text || "")
    .replace(/[^A-Za-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64);
  return slug || fallback;
}

/**
 * True when `target` lives inside `parent`.
 *
 * Recorded paths use forward slashes for portability while `path.join` produces
 * backslashes on Windows, so a plain `startsWith` comparison silently fails
 * there. Both sides are normalised before comparing.
 */
function isInside(parent, target) {
  if (!parent || !target) return false;
  const normalise = (value) => {
    const resolved = path.resolve(String(value).replace(/[\\/]+/g, path.sep));
    return process.platform === "win32" ? resolved.toLowerCase() : resolved;
  };
  const base = normalise(parent);
  const candidate = normalise(target);
  if (candidate === base) return true;
  return candidate.startsWith(base.endsWith(path.sep) ? base : base + path.sep);
}

function uniqueDir(parent, base) {
  const slug = slugify(base);
  let candidate = path.join(parent, slug);
  let counter = 2;
  while (fs.existsSync(candidate)) {
    candidate = path.join(parent, `${slug}-${counter}`);
    counter += 1;
  }
  return candidate;
}

module.exports = {
  dirs,
  files,
  pythonPackageDir,
  datasetsDir,
  ensureDirs,
  slugify,
  uniqueDir,
  isInside,
  isInsideAsar,
  root,
};
