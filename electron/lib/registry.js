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

const { datasets: datasetsStore, models: modelsStore, projects: projectsStore } = require("./store");
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
 * Datasets that ship with the application.
 *
 * The `datasets/` folder next to the app (repo folder in development,
 * `resources/datasets` when packaged) is scanned on demand, so a built-in
 * dataset is always in sync with what was actually installed. Names, record
 * counts and the default flag come from `datasets/manifest.json`; anything not
 * listed there still appears, with a name derived from its file name.
 *
 * Validation reports for built-ins are kept in memory only: the files
 * themselves are app assets and may be replaced by an update, so a stale
 * stored report would lie about the current file.
 */
const BUILTIN_PREFIX = "builtin:";
const builtinReports = new Map(); // id -> { report, validatedAt }

function builtinManifest() {
  try {
    const file = path.join(dirs.datasets(), "manifest.json");
    const parsed = JSON.parse(fs.readFileSync(file, "utf8"));
    const map = new Map();
    for (const item of parsed.datasets || []) {
      if (item && item.file) map.set(String(item.file).replace(/\\/g, "/"), item);
    }
    return map;
  } catch {
    return new Map();
  }
}

/** Fallback label when the manifest has nothing to say: "code_python_5" -> "Code Python 5". */
function prettyBuiltinName(relative) {
  const base = path.basename(relative, path.extname(relative));
  return base
    .split(/[_\-.]+/)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function listBuiltinDatasets() {
  const root = dirs.datasets();
  const manifest = builtinManifest();
  const entries = [];

  const walk = (dir) => {
    let items = [];
    try {
      items = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const item of items) {
      const full = path.join(dir, item.name);
      if (item.isDirectory()) {
        walk(full);
        continue;
      }
      if (item.name === "manifest.json" || !/\.(json|jsonl)$/i.test(item.name)) continue;

      const relative = path.relative(root, full).replace(/\\/g, "/");
      const meta = manifest.get(relative) || {};
      const id = BUILTIN_PREFIX + relative;
      const cached = builtinReports.get(id);

      entries.push({
        id,
        name: meta.name || prettyBuiltinName(relative),
        path: full.replace(/\\/g, "/"),
        format: path.extname(item.name).replace(".", "").toLowerCase(),
        isDirectory: false,
        sizeBytes: fileSize(full),
        addedAt: 0,
        records: meta.records ?? cached?.report?.dataset?.records ?? 0,
        usable: cached?.report?.stats?.usable ?? 0,
        status: cached?.report?.status ?? "unvalidated",
        mapping: cached?.report?.mapping ?? null,
        issues: cached?.report?.issues ?? [],
        report: cached?.report ?? null,
        builtin: true,
        default: Boolean(meta.default),
        lang: meta.lang || null,
        thinking: Boolean(meta.thinking),
        validatedAt: cached?.validatedAt ?? null,
      });
    }
  };

  walk(root);
  // The default stands first, everything else alphabetically.
  entries.sort((a, b) => Number(b.default) - Number(a.default) || a.name.localeCompare(b.name));
  return entries;
}

/** A stored entry, a built-in entry or null — ids and paths both resolve. */
function findDataset(idOrPath) {
  const stored = datasetsStore().get().find((item) => item.id === idOrPath || item.path === idOrPath);
  if (stored) return stored;
  return listBuiltinDatasets().find((item) => item.id === idOrPath || item.path === idOrPath) || null;
}

/** User imports first, built-ins after — the library always has something usable. */
function listDatasets() {
  return [...datasetsStore().get(), ...listBuiltinDatasets()];
}

function importDataset(targetPath) {
  const exists = fs.existsSync(targetPath);
  if (!exists) {
    return { ok: false, error: { code: "not_found", message: `'${targetPath}' does not exist.` } };
  }

  const name = path.basename(targetPath);
  const isDir = fs.statSync(targetPath).isDirectory();
  const format = isDir ? "folder" : path.extname(targetPath).replace(".", "").toLowerCase();

  const entry = {
    id: uid("ds"),
    name,
    path: targetPath.replace(/\\/g, "/"),
    format,
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
  const duplicate = list.find((item) => item.path === entry.path);
  if (duplicate) {
    // Re-importing the same path refreshes it instead of duplicating the row.
    Object.assign(duplicate, { ...entry, id: duplicate.id, addedAt: duplicate.addedAt });
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
    return {
      ok: false,
      error: response.error || {
        code: "validate_failed",
        message: "The dataset could not be validated. Is the Python backend available?",
        hint: response.stderr ? response.stderr.split("\n").slice(-4).join(" ") : "",
      },
    };
  }

  if (entry && entry.builtin) {
    // Built-in files are app assets that an update may replace, so their
    // reports live in memory for this session instead of on disk.
    builtinReports.set(entry.id, { report, validatedAt: Date.now() });
    return {
      ok: true,
      report,
      dataset: {
        ...entry,
        records: report.dataset?.records ?? entry.records,
        usable: report.stats ? report.stats.usable : 0,
        status: report.status,
        mapping: report.mapping,
        issues: report.issues || [],
        report,
        validatedAt: Date.now(),
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
  if (String(id).startsWith(BUILTIN_PREFIX)) {
    return {
      ok: false,
      error: {
        code: "builtin",
        message: "Built-in datasets ship with the app and cannot be removed.",
      },
    };
  }
  const store = datasetsStore();
  store.replace(store.get().filter((item) => item.id !== id));
  return { ok: true };
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
  listBuiltinDatasets,
  findDataset,
  importDataset,
  addHfDataset,
  validateDataset,
  removeDataset,
  previewDataset,
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
