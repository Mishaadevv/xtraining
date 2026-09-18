/**
 * Model export.
 *
 * The Python backend does the actual work: it copies the artefact together with
 * a generated README and a machine-readable manifest, and — when the ML runtime
 * is installed — can merge the adapter into its base model.
 *
 * This module's job is to resolve which run a model came from, so the exported
 * folder carries real provenance (method, dataset, loss, configuration) instead
 * of an anonymous pile of weights.
 */
const fs = require("node:fs");
const path = require("node:path");

const python = require("./python");
const { models: modelsStore, runs: runsStore } = require("./store");

/** The run that produced this model, if it is still in history. */
function findRun(model) {
  if (!model) return null;
  const runs = runsStore().get();
  if (model.runId) {
    const byId = runs.find((run) => run.id === model.runId);
    if (byId) return byId;
  }
  return runs.find((run) => run.outputDir && model.path && run.outputDir === model.path) || null;
}

function findModel(modelId) {
  return modelsStore().get().find((model) => model.id === modelId) || null;
}

/** Provenance written into export.json and the README. */
function buildMetadata(model, run) {
  const metadata = {
    modelId: model.id,
    name: model.name,
    method: model.method || (run && run.method) || null,
    baseModel: model.baseModel || (run && run.baseModel) || null,
    finalLoss: model.finalLoss ?? (run ? run.finalLoss : null) ?? null,
    steps: model.steps ?? (run ? run.totalSteps : null) ?? null,
    createdAt: model.createdAt || (run ? run.finishedAt : null) || null,
    app: "ZeqouXTraining",
  };
  if (run) {
    metadata.runId = run.id;
    metadata.projectId = run.projectId || null;
    metadata.datasetName = run.datasetName || null;
    metadata.datasetPath = run.datasetPath || null;
    metadata.datasetRecords = run.datasetReport ? run.datasetReport.records : null;
    metadata.trainableParams = run.trainableParams ?? null;
    metadata.totalParams = run.totalParams ?? null;
    metadata.device = run.device ?? null;
    metadata.precision = run.precision ?? null;
    metadata.targetModules = run.targetModules ?? null;
    metadata.config = run.config || null;
    metadata.startedAt = run.startedAt ?? null;
    metadata.finishedAt = run.finishedAt ?? null;
    metadata.gpuSummary = run.gpuSummary ?? null;
  }
  return metadata;
}

/** Describe an artefact folder: what it holds and which modes apply. */
async function describe(modelId) {
  const model = findModel(modelId);
  if (!model) {
    return { ok: false, error: { code: "not_found", message: "That model is not in the library." } };
  }
  if (!model.path) {
    return {
      ok: false,
      error: {
        code: "no_path",
        message: "This model has no local folder to export.",
        hint: "Hub models that have never been downloaded are fetched at training time.",
      },
    };
  }
  if (!fs.existsSync(model.path)) {
    return {
      ok: false,
      error: { code: "not_found", message: `'${model.path}' no longer exists on disk.` },
    };
  }

  const response = await python.run(["export", "--source", model.path, "--describe"]);
  if (!response.result) return { ok: false, error: response.error };
  return { ok: true, info: response.result.info, model, run: findRun(model) };
}

/** Export a model folder to a destination the user chose. */
async function exportModel(payload = {}) {
  const model = findModel(payload.modelId);
  if (!model) {
    return { ok: false, error: { code: "not_found", message: "That model is not in the library." } };
  }
  if (!model.path) {
    return {
      ok: false,
      error: { code: "no_path", message: "This model has no local folder to export." },
    };
  }
  const outputDir = String(payload.outputDir || "").trim();
  if (!outputDir) {
    return {
      ok: false,
      error: {
        code: "no_output",
        message: "Choose a destination folder first.",
        hint: "Exports are written into an empty folder — nothing is overwritten.",
      },
    };
  }

  const run = findRun(model);
  const metadata = buildMetadata(model, run);
  const args = [
    "export",
    "--source", model.path,
    "--output", outputDir,
    "--metadata", JSON.stringify(metadata),
  ];
  const baseModel = model.baseModel || (run && run.baseModel);
  if (baseModel) args.push("--base-model", baseModel);
  if (payload.merge) args.push("--merge");

  const response = await python.run(args);
  if (!response.result) {
    return { ok: false, error: response.error };
  }
  return {
    ok: true,
    ...response.result,
    modelId: model.id,
    provenance: metadata,
  };
}

module.exports = { describe, exportModel, findRun, buildMetadata };
