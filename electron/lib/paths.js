/**
 * Filesystem layout for ZeqouXTraining.
 *
 * Everything user-generated lives under Electron's userData directory:
 *   settings.json, secrets.json, datasets.json, models.json, runs.json
 *   runs/<runId>/          checkpoints, trainer state, logs, metadata
 *   datasets/              copies of imported dataset files
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
  datasets: () => path.join(root(), "datasets"),
  models: () => path.join(root(), "models"),
  cache: () => path.join(root(), "cache"),
  hfCache: () => path.join(root(), "cache", "huggingface"),
  logs: () => path.join(root(), "logs"),
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
 * The Python backend package. In development it sits next to the repo; in a
 * packaged build electron-builder copies it into the app's resources.
 */
function pythonPackageDir() {
  const candidates = [
    path.join(__dirname, "..", "..", "python"),
    path.join(process.resourcesPath || "", "python"),
    path.join(process.resourcesPath || "", "app.asar.unpacked", "python"),
  ];
  for (const candidate of candidates) {
    try {
      if (fs.existsSync(path.join(candidate, "zeqouxtraining", "cli.py"))) return candidate;
    } catch {
      /* keep looking */
    }
  }
  return candidates[0];
}

function ensureDirs() {
  for (const dir of [dirs.root(), dirs.runs(), dirs.datasets(), dirs.models(), dirs.cache(), dirs.logs()]) {
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

module.exports = { dirs, files, pythonPackageDir, ensureDirs, slugify, uniqueDir, isInside, root };
