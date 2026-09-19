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

function listDatasets() {
  return datasetsStore().get();
}

async function importDataset(targetPath) {
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
  const store = datasetsStore();
  const list = store.get();
  const entry = list.find((item) => item.id === idOrPath || item.path === idOrPath) || null;
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

  if (entry) {
    // Keep what was already known when the report has nothing to say: a failed
    // validation must not wipe the stored format or size.
    const reported = report.dataset || {};
    Object.assign(entry, {
      records: reported.records ?? 0,
      usable: report.stats ? report.stats.usable : 0,
      status: report.status,
      mapping: report.mapping,
      issues: report.issues || [],
      report,
      format: reported.format || entry.format,
      sizeBytes: typeof reported.bytes === "number" ? reported.bytes : entry.sizeBytes,
      validatedAt: Date.now(),
    });
    store.replace(list);
  }

  return { ok: true, report, dataset: entry || null };
}

function removeDataset(id) {
  const store = datasetsStore();
  store.replace(store.get().filter((item) => item.id !== id));
  return { ok: true };
}

async function previewDataset(idOrPath, limit = 5) {
  const entry = datasetsStore().get().find((item) => item.id === idOrPath) || null;
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
