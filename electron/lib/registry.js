/**
 * Registries for datasets, models and projects.
 *
 * Datasets are referenced in place rather than copied: a fine-tuning dataset can
 * easily be gigabytes, and silently duplicating it into an app folder would be
 * a surprising use of the user's disk. The registry stores the path plus the
 * last validation report so the list can show real counts without re-reading.
 *
 * Validation and model inspection are delegated to the Python backend, which is
 * the single source of truth for anything that depends on the ML stack.
 */
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");

const {
  settings: settingsStore,
  datasets: datasetsStore,
  models: modelsStore,
  projects: projectsStore,
} = require("./store");
const python = require("./python");
const { dirs, isInside } = require("./paths");

function uid(prefix) {
  return `${prefix}_${crypto.randomBytes(6).toString("hex")}`;
}

function fileSize(target) {
  try {
    const stat = fs.statSync(target);
    if (stat.isFile()) return stat.size;
    let total = 0;
    const walk = (dir) => {
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const next = path.join(dir, entry.name);
        if (entry.isDirectory()) walk(next);
        else if (entry.isFile()) total += fs.statSync(next).size;
      }
    };
    walk(target);
    return total;
  } catch {
    return null;
  }
}

/* --------------------------------------------------------------- datasets */

/**
 * Dataset library.
 *
 * Nothing is bundled: the app ships without a single dataset, and the library
 * is exactly what the user put there. Two ways in, one list out:
 *
 *   · **scan** the dataset folder (Settings → Datasets). Every supported file
 *     in it is listed, and every subfolder that contains dataset files is
 *     listed as one shard set. This is what makes "I dropped my files in a
 *     folder" work without importing anything by hand.
 *   · **import** an individual file, folder or Hub id from anywhere on disk.
 *
 * Datasets are still referenced in place — a scan records paths and never
 * copies data. Validation reports are stored with the entry, because a report
 * is the expensive part and the file is not app-managed.
 */

/**
 * The extensions the scanner watches. `python/zeqouxtraining/datasets.py`
 * (`FORMAT_BY_EXTENSION`) is the single source of truth for what can actually
 * be *read*; the smoke test compares the two lists so they cannot drift.
 */
const SCAN_EXTENSIONS = [
  ".json", ".jsonl", ".ndjson", ".jsonlines",
  ".csv", ".psv", ".tsv", ".tab",
  ".txt", ".text", ".md", ".markdown",
  ".parquet", ".pq", ".arrow", ".feather", ".orc",
  ".sqlite", ".sqlite3", ".db",
  ".xlsx", ".xlsm", ".xls",
  ".yaml", ".yml",
];
const COMPRESSION_SUFFIXES = [".gz", ".bz2", ".xz", ".lzma"];
const FOLDER_ORIGIN = "folder";
const MAX_SCAN_DEPTH = 6;

/** A case- and separator-insensitive key, so ids survive a rescan. */
function pathKey(target) {
  const normalised = path.resolve(String(target).replace(/[\\/]+/g, path.sep));
  return process.platform === "win32" ? normalised.toLowerCase() : normalised;
}

/** True for a file name the backend can open, compressed or not. */
function isSupportedFile(name) {
  let lowered = String(name).toLowerCase();
  for (const suffix of COMPRESSION_SUFFIXES) {
    if (lowered.endsWith(suffix)) {
      lowered = lowered.slice(0, -suffix.length);
      break;
    }
  }
  const dot = lowered.lastIndexOf(".");
  return dot > 0 && SCAN_EXTENSIONS.includes(lowered.slice(dot));
}

/** The folder that is scanned. Configurable; defaults to the app's own folder. */
function datasetsFolder() {
  const configured = settingsStore().get().datasetsDir;
  return configured && String(configured).trim() ? String(configured) : dirs.datasets();
}

function containsSupportedFiles(dir, depth = 0) {
  if (depth > MAX_SCAN_DEPTH) return false;
  let items;
  try {
    items = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return false;
  }
  for (const item of items) {
    if (item.name.startsWith(".")) continue;
    if (item.isFile() && isSupportedFile(item.name)) return true;
    if (item.isDirectory() && containsSupportedFiles(path.join(dir, item.name), depth + 1)) return true;
  }
  return false;
}

/**
 * What the folder holds: loose dataset files directly in it, plus one entry per
 * subfolder that contains dataset files (a shard set). Nested files are covered
 * by their shard-set entry, so the list never shows the same data twice.
 */
function discoverInFolder(root) {
  const found = [];
  let items;
  try {
    items = fs.readdirSync(root, { withFileTypes: true });
  } catch (error) {
    return {
      found,
      error: {
        code: "folder_unreadable",
        message: `The dataset folder '${root}' could not be read.`,
        hint: process.env.ZEQOUX_DEV
          ? "Check the path in Settings → Datasets."
          : `Create the folder ('${root}') or choose another one in Settings → Datasets.`,
      },
    };
  }

  for (const item of items) {
    if (item.name.startsWith(".")) continue;
    const full = path.join(root, item.name);
    if (item.isDirectory()) {
      if (containsSupportedFiles(full)) found.push({ path: full, isDirectory: true, name: item.name });
      continue;
    }
    if (item.isFile() && isSupportedFile(item.name)) {
      found.push({ path: full, isDirectory: false, name: item.name });
    }
  }
  return { found, error: null };
}

function datasetFormatOf(name, isDirectory) {
  if (isDirectory) return "folder";
  let lowered = String(name).toLowerCase();
  let compression = null;
  for (const suffix of COMPRESSION_SUFFIXES) {
    if (lowered.endsWith(suffix)) {
      compression = suffix;
      lowered = lowered.slice(0, -suffix.length);
      break;
    }
  }
  const extension = lowered.slice(lowered.lastIndexOf(".") + 1);
  return compression ? `${extension}${compression}` : extension;
}

/**
 * Scan the dataset folder into the library.
 *
 * The scan never deletes data and never touches an existing report: entries it
 * finds again keep their validation history, and entries whose file has gone
 * are dropped from the list (removed by the user, they stay hidden).
 */
function scanDatasetsFolder(options = {}) {
  const root = datasetsFolder();
  const discovery = discoverInFolder(root);
  const base = {
    folder: root,
    found: discovery.found.length,
    scannedAt: Date.now(),
  };

  if (discovery.error) return { ok: false, ...base, error: discovery.error, datasets: listDatasets() };
  if (options.register === false) return { ok: true, ...base, datasets: listDatasets() };

  const store = datasetsStore();
  const list = store.get();
  const index = new Map(list.map((item) => [pathKey(item.path), item]));
  const seen = new Set();
  const added = [];

  for (const item of discovery.found) {
    const key = pathKey(item.path);
    seen.add(key);
    const existing = index.get(key);
    if (existing) {
      // Only what the filesystem owns is refreshed; the report stays put.
      existing.origin = existing.origin || FOLDER_ORIGIN;
      existing.sizeBytes = fileSize(item.path);
      existing.isDirectory = item.isDirectory;
      existing.format = datasetFormatOf(item.name, item.isDirectory);
      continue;
    }

    added.push({
      id: `ds_fs_${crypto.createHash("sha1").update(key).digest("hex").slice(0, 12)}`,
      name: item.name,
      path: item.path.replace(/\\/g, "/"),
      folder: root.replace(/\\/g, "/"),
      origin: FOLDER_ORIGIN,
      format: datasetFormatOf(item.name, item.isDirectory),
      isDirectory: item.isDirectory,
      sizeBytes: fileSize(item.path),
      addedAt: Date.now(),
      records: 0,
      usable: 0,
      status: "unvalidated",
      mapping: null,
      issues: [],
      report: null,
    });
  }

  added.sort((a, b) => a.name.localeCompare(b.name));
  const kept = list.filter((item) => item.origin !== FOLDER_ORIGIN
    || item.hidden
    || seen.has(pathKey(item.path)));
  store.replace([...added, ...kept]);

  return {
    ok: true,
    ...base,
    added: added.length,
    removed: list.length - kept.length,
    hidden: kept.filter((item) => item.hidden).length,
    datasets: listDatasets(),
  };
}

/** Which dataset types this install can actually read, straight from the backend. */
async function datasetFormats() {
  const response = await python.run(["dataset-formats"]);
  if (response.result) return { ok: true, ...response.result };
  return {
    ok: false,
    error: response.error || {
      code: "backend_unavailable",
      message: "The list of supported dataset types needs the Python backend.",
    },
    extensions: [...SCAN_EXTENSIONS],
    compression: [...COMPRESSION_SUFFIXES],
    export_formats: ["jsonl", "json", "csv", "tsv", "txt", "parquet"],
    formats: [],
  };
}

/** Every dataset in the library that the user has not hidden. */
function listDatasets() {
  return datasetsStore().get().filter((item) => !item.hidden);
}

/** Datasets the user hid — shown separately so nothing disappears silently. */
function hiddenDatasets() {
  return datasetsStore().get().filter((item) => item.hidden);
}

/** A stored entry or null — an id or a path both resolve. */
function findDataset(idOrPath) {
  if (!idOrPath) return null;
  const key = pathKey(idOrPath);
  return datasetsStore().get().find(
    (item) => item.id === idOrPath || pathKey(item.path) === key,
  ) || null;
}

function importDataset(targetPath) {
  const exists = fs.existsSync(targetPath);
  if (!exists) {
    return { ok: false, error: { code: "not_found", message: `'${targetPath}' does not exist.` } };
  }

  const name = path.basename(targetPath);
  const isDir = fs.statSync(targetPath).isDirectory();
  const fromFolder = isInside(datasetsFolder(), targetPath);

  const entry = {
    id: uid("ds"),
    name,
    path: path.resolve(targetPath).replace(/\\/g, "/"),
    folder: fromFolder ? datasetsFolder().replace(/\\/g, "/") : null,
    origin: fromFolder ? FOLDER_ORIGIN : "import",
    format: datasetFormatOf(name, isDir),
    isDirectory: isDir,
    sizeBytes: fileSize(targetPath),
    addedAt: Date.now(),
    records: 0,
    usable: 0,
    status: "unvalidated",
    mapping: null,
    issues: [],
    report: null,
  };

  const store = datasetsStore();
  const list = store.get();
  const duplicate = list.find((item) => pathKey(item.path) === pathKey(entry.path));
  if (duplicate) {
    // Re-importing the same path refreshes it instead of duplicating the row.
    Object.assign(duplicate, {
      ...entry,
      id: duplicate.id,
      addedAt: duplicate.addedAt,
      hidden: false,
      // A returning dataset keeps the report it already had.
      records: duplicate.records,
      usable: duplicate.usable,
      status: duplicate.status,
      report: duplicate.report,
      issues: duplicate.issues,
      mapping: duplicate.mapping,
    });
    store.replace(list);
    return { ok: true, dataset: duplicate, refreshed: true };
  }
  list.unshift(entry);
  store.replace(list);
  return { ok: true, dataset: entry };
}

/**
 * Register a Hugging Face dataset by id. Nothing is downloaded here — the
 * backend fetches the rows during validation or at training start.
 */
function addHfDataset(datasetId, split = "train") {
  const trimmed = String(datasetId || "").trim().replace(/^https?:\/\/huggingface\.co\/datasets\//, "");
  if (!trimmed) {
    return {
      ok: false,
      error: {
        code: "empty",
        message: "Enter a Hugging Face dataset id.",
        hint: "For example: tatsu-lab/alpaca",
      },
    };
  }
  if (!/^[\w.-]+\/[\w.-]+$/.test(trimmed)) {
    return {
      ok: false,
      error: {
        code: "bad_id",
        message: `'${trimmed}' is not a dataset id.`,
        hint: "A Hub dataset id looks like owner/name, e.g. tatsu-lab/alpaca.",
      },
    };
  }

  const cleanSplit = String(split || "train").trim() || "train";
  const store = datasetsStore();
  const list = store.get();
  const duplicate = list.find(
    (item) => item.format === "hf" && item.path === trimmed && (item.split || "train") === cleanSplit,
  );
  if (duplicate) {
    return { ok: true, dataset: duplicate, refreshed: true };
  }

  const entry = {
    id: uid("ds"),
    name: `${trimmed} (${cleanSplit})`,
    path: trimmed,
    hfId: trimmed,
    split: cleanSplit,
    format: "hf",
    isDirectory: false,
    sizeBytes: null,
    addedAt: Date.now(),
    records: 0,
    usable: 0,
    status: "unvalidated",
    mapping: null,
    issues: [],
    report: null,
    origin: "hub",
  };
  list.unshift(entry);
  store.replace(list);
  return { ok: true, dataset: entry };
}

/** True when the entry (or raw id) refers to a Hugging Face dataset. */
function isHubDataset(entry) {
  return Boolean(entry && (entry.format === "hf" || entry.hfId));
}

/** Arguments shared by the dataset commands, honouring the source type. */
function datasetSourceArgs(entryOrPath, options = {}) {
  const entry = typeof entryOrPath === "string" ? null : entryOrPath;
  const raw = typeof entryOrPath === "string" ? entryOrPath : entryOrPath.path;
  if (isHubDataset(entry)) {
    return ["--hf-id", entry.hfId || entry.path, "--split", entry.split || "train"];
  }
  // A bare id typed into the Hub field still reaches the backend as a Hub id.
  if (!entry && /^[\w.-]+\/[\w.-]+$/.test(String(raw)) && !fs.existsSync(String(raw))) {
    return ["--hf-id", String(raw), "--split", String(options.split || "train")];
  }
  return ["--path", raw, "--format", options.format || "auto"];
}

async function validateDataset(idOrPath, options = {}) {
  const entry = findDataset(idOrPath);
  const target = entry ? entry.path : idOrPath;
  if (!target) {
    return { ok: false, error: { code: "not_found", message: "Dataset not found." } };
  }

  const args = ["validate-dataset", ...datasetSourceArgs(entry || target, options),
    "--context-length", String(options.contextLength || 512)];
  if (options.mapping) args.push("--mapping", JSON.stringify(options.mapping));
  if (options.maxRecords) args.push("--max-records", String(options.maxRecords));

  const response = await python.run(args);
  const report = response.result;
  if (!report) {
    const fallback = await python.diagnosis();
    return {
      ok: false,
      error: response.error || {
        code: "validate_failed",
        message: "The dataset could not be validated: the backend produced no report.",
        hint: response.stderr
          ? response.stderr.split("\n").slice(-4).join(" ")
          : fallback.hint,
      },
    };
  }

  if (entry) {
    // Keep what was already known when the report has nothing to say: a failed
    // validation must not wipe the stored format or size.
    const store = datasetsStore();
    const list = store.get();
    const stored = list.find((item) => item.id === entry.id) || entry;
    const reported = report.dataset || {};
    Object.assign(stored, {
      records: reported.records ?? 0,
      usable: report.stats ? report.stats.usable : 0,
      status: report.status,
      mapping: report.mapping,
      issues: report.issues || [],
      report,
      format: reported.format || stored.format,
      sizeBytes: typeof reported.bytes === "number" ? reported.bytes : stored.sizeBytes,
      validatedAt: Date.now(),
    });
    store.replace(list);
    return { ok: true, report, dataset: stored };
  }

  return { ok: true, report, dataset: null };
}

function removeDataset(id) {
  const store = datasetsStore();
  const list = store.get();
  const target = list.find((item) => item.id === id);
  if (!target) {
    return { ok: false, error: { code: "not_found", message: "That dataset is not in the library." } };
  }
  if (target.origin === FOLDER_ORIGIN) {
    // A scanned dataset still exists on disk, so removal hides the row rather
    // than pretending the file was deleted — and a rescan will not bring it back.
    target.hidden = true;
    store.replace(list);
    return { ok: true, hidden: true, path: target.path };
  }
  store.replace(list.filter((item) => item.id !== id));
  return { ok: true };
}

/** Bring a hidden (scanned) dataset back into the list. */
function restoreDataset(id) {
  const store = datasetsStore();
  const list = store.get();
  const target = list.find((item) => item.id === id);
  if (!target) {
    return { ok: false, error: { code: "not_found", message: "That dataset is not in the library." } };
  }
  target.hidden = false;
  store.replace(list);
  return { ok: true };
}

/** Point the scanner at another folder and index it straight away. */
function setDatasetsFolder(folder) {
  const clean = folder && String(folder).trim() ? String(folder).trim() : null;
  settingsStore().merge({ datasetsDir: clean });
  const scan = scanDatasetsFolder();
  return { ...scan, folder: datasetsFolder(), settings: settingsStore().get() };
}

/**
 * Write any dataset — file, folder of shards or Hub id — as ONE file.
 *
 * ``outputPath`` is a file name; the format comes from its extension unless
 * ``outputFormat`` overrides it. Nothing else is written: no sidecar manifests,
 * no folder of parts.
 */
async function exportDataset(idOrPath, options = {}) {
  const entry = findDataset(idOrPath);
  const target = entry ? entry.path : idOrPath;
  if (!target) {
    return { ok: false, error: { code: "not_found", message: "Dataset not found." } };
  }
  const outputPath = String(options.outputPath || "").trim();
  if (!outputPath) {
    return {
      ok: false,
      error: {
        code: "no_output",
        message: "Choose where the exported file should be written.",
        hint: "The export is a single file — give it a name such as dataset.jsonl.",
      },
    };
  }

  // --overwrite is always passed because the destination came from a native
  // save dialog, which already asked the user about replacing an existing file.
  const args = [
    "export-dataset", ...datasetSourceArgs(entry || target, {}),
    "--output", outputPath, "--overwrite",
  ];
  if (options.outputFormat) args.push("--out-format", String(options.outputFormat));
  if (options.mapping) args.push("--mapping", JSON.stringify(options.mapping));
  if (options.raw) args.push("--raw");
  if (options.maxRecords) args.push("--limit", String(options.maxRecords));

  const response = await python.run(args);
  if (!response.result) {
    return {
      ok: false,
      error: response.error || {
        code: "export_failed",
        message: "The dataset could not be exported.",
        hint: response.stderr ? response.stderr.split("\n").slice(-4).join(" ") : "",
      },
    };
  }
  return { ok: true, ...response.result, datasetId: entry ? entry.id : null };
}

async function previewDataset(idOrPath, limit = 5) {
  const entry = findDataset(idOrPath);
  const target = entry ? entry.path : idOrPath;
  const response = await python.run([
    "preview-dataset", ...datasetSourceArgs(entry || target, {}),
    "--limit", String(limit), "--max-records", "500",
  ]);
  if (!response.result) {
    return { ok: false, error: response.error || { message: "Preview failed." } };
  }
  return { ok: true, ...response.result };
}

/* ----------------------------------------------------------------- models */

function listModels() {
  return modelsStore().get();
}

async function addModel(source, options = {}) {
  const trimmed = String(source || "").trim();
  if (!trimmed) {
    return { ok: false, error: { code: "empty", message: "Provide a model id or a local folder." } };
  }

  const response = await python.run(
    ["inspect-model", "--source", trimmed, ...(options.hfCache ? ["--hf-cache", options.hfCache] : [])],
  );

  if (!response.result) {
    return {
      ok: false,
      error: response.error || {
        code: "inspect_failed",
        message: "The model could not be inspected.",
        hint: "Check the Python interpreter in Settings → Environment.",
      },
    };
  }

  const info = response.result;
  const entry = {
    id: uid("md"),
    name: info.name,
    source: trimmed,
    kind: info.kind, // "local" | "huggingface"
    path: info.kind === "local" ? info.source : info.cached_path || null,
    cached: Boolean(info.cached),
    params: info.params,
    paramsExact: Boolean(info.params_exact),
    sizeBytes: info.size_bytes,
    architecture: (info.fields && info.fields.architectures && info.fields.architectures[0]) || info.fields?.model_type || null,
    maxPositionEmbeddings: info.max_position_embeddings || null,
    torchDtype: info.fields ? info.fields.torch_dtype : null,
    trainable: Boolean(info.trainable),
    issues: info.issues || [],
    addedAt: Date.now(),
    trained: false,
    adapter: false,
    // Set when the folder is a PEFT adapter (a trained model that can itself
    // be fine-tuned further): training on it continues from that adapter.
    isAdapterFolder: Boolean(info.adapter),
    adapterBase: info.adapter_base || null,
  };

  const store = modelsStore();
  const list = store.get();
  const duplicate = list.find((item) => item.source === entry.source && item.kind === entry.kind);
  if (duplicate) {
    Object.assign(duplicate, { ...entry, id: duplicate.id, addedAt: duplicate.addedAt });
    store.replace(list);
    return { ok: true, model: duplicate, refreshed: true, info };
  }
  list.unshift(entry);
  store.replace(list);
  return { ok: true, model: entry, info };
}

function registerTrainedModel(run) {
  if (!run || !run.outputDir) return null;
  const store = modelsStore();
  const list = store.get();
  const existing = list.find((item) => item.path === run.outputDir);
  const entry = existing || {
    id: uid("md"),
    addedAt: Date.now(),
  };

  Object.assign(entry, {
    name: run.name,
    source: run.outputDir,
    kind: "trained",
    path: run.outputDir,
    trained: true,
    adapter: run.method !== "full",
    method: run.method,
    baseModel: run.baseModel,
    runId: run.id,
    params: run.trainableParams || null,
    totalParams: run.totalParams || null,
    finalLoss: run.finalLoss ?? null,
    steps: run.totalSteps || 0,
    status: run.status,
    createdAt: run.finishedAt || Date.now(),
    issues: [],
  });

  if (existing) {
    store.replace(list);
  } else {
    list.unshift(entry);
    store.replace(list);
  }
  return entry;
}

function removeModel(id) {
  const store = modelsStore();
  const list = store.get();
  const target = list.find((item) => item.id === id);
  // Trained models live in the app's own models folder; remove the files too.
  if (target && target.trained && target.path && isInside(dirs.models(), target.path)) {
    try {
      fs.rmSync(target.path, { recursive: true, force: true });
    } catch (error) {
      return { ok: false, error: error.message };
    }
  }
  store.replace(list.filter((item) => item.id !== id));
  return { ok: true };
}

/* --------------------------------------------------------------- projects */

function listProjects() {
  return projectsStore().get();
}

function getProject(id) {
  return projectsStore().get().find((item) => item.id === id) || null;
}

function createProject(patch = {}) {
  const store = projectsStore();
  const list = store.get();
  const project = {
    id: uid("prj"),
    name: patch.name || "Untitled project",
    description: patch.description || "",
    createdAt: Date.now(),
    updatedAt: Date.now(),
    method: patch.method || null,
    baseModel: patch.baseModel || null,
    datasetName: patch.datasetName || null,
    datasetPath: patch.datasetPath || null,
    config: patch.config || null,
    runIds: [],
    runCount: 0,
    lastStatus: null,
    lastRunAt: null,
    bestLoss: null,
  };
  list.unshift(project);
  store.replace(list);
  return project;
}

function updateProject(id, patch = {}) {
  const store = projectsStore();
  const list = store.get();
  const project = list.find((item) => item.id === id);
  if (!project) return { ok: false, error: "Project not found." };
  Object.assign(project, patch, { updatedAt: Date.now() });
  store.replace(list);
  return { ok: true, project };
}

function removeProject(id) {
  const store = projectsStore();
  const list = store.get();
  const project = list.find((item) => item.id === id);
  if (!project) return { ok: false, error: "Project not found." };

  // Project runs are kept in history; only the grouping is removed.
  const runsStore = require("./store").runs();
  const allRuns = runsStore.get();
  for (const run of allRuns) {
    if (run.projectId === id) run.projectId = null;
  }
  runsStore.replace(allRuns);

  store.replace(list.filter((item) => item.id !== id));
  return { ok: true, detachedRuns: (project.runIds || []).length };
}

module.exports = {
  listDatasets,
  hiddenDatasets,
  findDataset,
  importDataset,
  addHfDataset,
  validateDataset,
  removeDataset,
  restoreDataset,
  previewDataset,
  scanDatasetsFolder,
  datasetsFolder,
  supportedExtensions: () => [...SCAN_EXTENSIONS],
  compressionSuffixes: () => [...COMPRESSION_SUFFIXES],
  setDatasetsFolder,
  datasetFormats,
  exportDataset,
  isSupportedFile,
  listModels,
  addModel,
  registerTrainedModel,
  removeModel,
  listProjects,
  getProject,
  createProject,
  updateProject,
  removeProject,
};
