/**
 * Training job manager.
 *
 * Owns the full lifecycle of a run:
 *
 *   1. allocate a run directory under userData/runs/<slug>
 *   2. write the job file the Python backend consumes
 *   3. spawn `train --job <file>` and forward every event to the renderer
 *   4. sample nvidia-smi once a second while the run is alive
 *   5. persist the run record, history, GPU series and logs to disk
 *
 * Stop and pause are cooperative and cross-platform: the app touches a flag file
 * that the trainer checks at every step boundary, then the run checkpoints
 * before exiting. Nothing is force-killed unless the process is unresponsive.
 */
const fs = require("node:fs");
const path = require("node:path");

const { dirs, files, slugify, uniqueDir, isInside } = require("./paths");
const { runs: runsStore, projects: projectsStore } = require("./store");
const python = require("./python");
const hardware = require("./hardware");

const GPU_SAMPLE_INTERVAL_MS = 1000;
const MAX_GPU_SAMPLES = 4000;

/** A status after which the record will never change again. */
const TERMINAL_STATUSES = new Set(["completed", "stopped", "paused", "failed"]);

/** runId -> live state */
const active = new Map();
let emitToRenderer = () => {};

function attach(sender) {
  emitToRenderer = sender;
}

function send(channel, payload) {
  try {
    emitToRenderer(channel, payload);
  } catch {
    /* the window may be gone; runs continue regardless */
  }
}

/* ------------------------------------------------------------------ helpers */

function findProject(projectId) {
  if (!projectId) return null;
  return projectsStore().get().find((project) => project.id === projectId) || null;
}

function persistentRun(runId) {
  return runsStore().get().find((run) => run.id === runId) || null;
}

/** Drop the large in-memory-only fields before writing the index. */
function slimRun(run) {
  const { gpuSamples, ...rest } = run;
  return rest;
}

/**
 * Wait until the run leaves `active` so the caller never races the async
 * `finalise()` (it writes artefacts, scans checkpoints and only then removes
 * the entry). Without this, the caller can see a terminal status while the
 * record is still being written — and a racing delete gets undone when the
 * final state is persisted afterwards.
 */
async function waitForFinalize(runId, { timeoutMs = 60000, intervalMs = 100 } = {}) {
  const deadline = Date.now() + timeoutMs;
  while (active.has(runId) && Date.now() <= deadline) {
    // eslint-disable-next-line no-await-in-loop - polling with a backoff is the point
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

function upsertRun(run) {
  const store = runsStore();
  const list = store.get();
  const index = list.findIndex((entry) => entry.id === run.id);
  if (index >= 0) list[index] = slimRun(run);
  else list.unshift(slimRun(run));
  store.replace(list);
}

/* ------------------------------------------------------------------- start */

async function startRun(request) {
  const config = request.config || {};
  // Training from scratch is the one method without a base model.
  if (!config.base_model && config.method !== "scratch") {
    return { ok: false, error: { code: "no_base_model", message: "Select a base model first." } };
  }
  const datasetPath = config.dataset && config.dataset.path;
  const hfId = config.dataset && config.dataset.hf_id;
  if (!datasetPath && !hfId) {
    return { ok: false, error: { code: "no_dataset", message: "Select a dataset first." } };
  }

  const name = request.name || config.output_name || `${slugify(config.base_model)}-${config.method || "lora"}`;
  const runDir = uniqueDir(dirs.runs(), name);
  const runId = path.basename(runDir);
  fs.mkdirSync(runDir, { recursive: true });

  const stopFile = path.join(runDir, ".stop");
  const pauseFile = path.join(runDir, ".pause");
  const jobPath = path.join(runDir, "job.json");

  const software = python.buildEnv();
  // The scratch method is implemented by its own backend; anything else
  // defaults to the HF+PEFT engine.
  const backendName = request.backend || (config.method === "scratch" ? "scratch" : "hf-peft");
  const job = {
    job_id: runId,
    project_id: request.projectId || null,
    run_dir: runDir.replace(/\\/g, "/"),
    stop_file: stopFile.replace(/\\/g, "/"),
    pause_file: pauseFile.replace(/\\/g, "/"),
    backend: backendName,
    hardware: request.hardware || {},
    model_info: request.modelInfo || {},
    config: {
      ...config,
      output_name: name,
      resume_from_checkpoint: request.resumeFromCheckpoint || null,
    },
  };
  fs.writeFileSync(jobPath, JSON.stringify(job, null, 2), "utf8");

  const record = {
    id: runId,
    projectId: request.projectId || null,
    name,
    method: config.method || "lora",
    baseModel: config.base_model || (config.method === "scratch" ? "(from scratch)" : ""),
    datasetName: request.datasetName || (config.dataset && (config.dataset.name || config.dataset.path)) || "",
    datasetPath: datasetPath || hfId || "",
    status: "starting",
    phase: "prepare",
    progress: 0,
    step: 0,
    totalSteps: 0,
    loss: null,
    evalLoss: null,
    learningRate: null,
    epoch: 0,
    elapsedSeconds: 0,
    etaSeconds: null,
    samplesPerSecond: null,
    secondsPerStep: null,
    gpuMemoryAllocatedMb: null,
    gpuMemoryReservedMb: null,
    startedAt: Date.now(),
    finishedAt: null,
    runDir: runDir.replace(/\\/g, "/"),
    outputDir: null,
    lastCheckpoint: null,
    finalLoss: null,
    config: job.config,
    history: null,
    datasetReport: request.datasetReport || null,
    error: null,
    checkpoints: [],
    resumedFrom: request.resumeFromCheckpoint || null,
    environment: {
      hfHome: software.HF_HOME,
      tokenPresent: Boolean(software.HF_TOKEN),
    },
  };

  const logStream = fs.createWriteStream(path.join(runDir, "training.log"), { flags: "a" });
  const stderrStream = fs.createWriteStream(path.join(runDir, "stderr.log"), { flags: "a" });

  const state = {
    record,
    runDir,
    stopFile,
    pauseFile,
    jobPath,
    gpuSamples: [],
    gpuTimer: null,
    handle: null,
    logStream,
    stderrStream,
    finishing: false,
  };
  active.set(runId, state);
  upsertRun(record);
  send("zeqou:training:state", { runId, state: "starting", record: slimRun(record) });

  state.gpuTimer = setInterval(async () => {
    const sample = await hardware.sample();
    state.gpuSamples.push(sample);
    if (state.gpuSamples.length > MAX_GPU_SAMPLES) state.gpuSamples.shift();
    send("zeqou:gpu", { runId, sample });
  }, GPU_SAMPLE_INTERVAL_MS);
  // Sampling should never hold the event loop open on shutdown.
  if (state.gpuTimer.unref) state.gpuTimer.unref();

  let handle;
  try {
    handle = python.stream(["train", "--job", jobPath], {
      onEvent: (parsed) => handleEvent(state, parsed),
      onStderr: (line) => {
        stderrStream.write(`${line}\n`);
        state.record.stderrLines = (state.record.stderrLines || 0) + 1;
        send("zeqou:training:log", { runId, line, stream: "stderr" });
      },
    });
  } catch (error) {
    await finalize(state, {
      status: "failed",
      error: { code: "spawn_failed", message: error.message },
    });
    return { ok: false, runId, error: { code: "spawn_failed", message: error.message } };
  }
  state.handle = handle;

  handle.done
    .then((exit) => {
      const failed = exit.code !== 0 && state.record.status === "running";
      const status = failed
        ? "failed"
        : state.record.status === "running"
          ? "completed"
          : state.record.status;
      return finalize(state, {
        status,
        exitCode: exit.code,
        error:
          failed
            ? state.record.error || {
                code: "process_exit",
                message: `The training process exited with code ${exit.code}. See the log for details.`,
                hint: "The technical log at the bottom of the Training screen has the full output.",
              }
            : state.record.error,
      });
    })
    .catch((error) => finalize(state, {
      status: "failed",
      error: { code: "process_error", message: error.message },
    }));

  return { ok: true, runId, record: slimRun(record) };
}

/* ------------------------------------------------------------------ events */

function handleEvent(state, parsed) {
  const { event, detail = {} } = parsed;
  const record = state.record;
  state.logStream.write(`${JSON.stringify(parsed)}\n`);

  switch (event) {
    case "ready":
      record.status = "running";
      break;

    case "stage":
      record.phase = detail.stage || record.phase;
      break;

    case "dataset-report":
      record.datasetReport = {
        source: detail.source,
        mapping: detail.mapping,
        records: detail.records,
        samples: detail.samples,
        loadSeconds: detail.load_seconds,
      };
      break;

    case "model-info":
      record.modelParams = detail.params;
      break;

    case "training-status":
      record.phase = detail.phase || record.phase;
      record.message = detail.message;
      if (detail.total_steps) record.totalSteps = detail.total_steps;
      if (detail.target_modules) record.targetModules = detail.target_modules;
      if (detail.device) record.device = detail.device;
      if (detail.precision) record.precision = detail.precision;
      if (detail.optimizer) record.optimizer = detail.optimizer;
      if (detail.trainable_params) record.trainableParams = detail.trainable_params;
      if (detail.total_params) record.totalParams = detail.total_params;
      break;

    case "training-progress":
      record.status = "running";
      record.progress = detail.progress ?? record.progress;
      record.step = detail.step ?? record.step;
      record.totalSteps = detail.total_steps ?? record.totalSteps;
      record.loss = detail.loss ?? record.loss;
      record.evalLoss = detail.eval_loss ?? record.evalLoss;
      record.learningRate = detail.learning_rate ?? record.learningRate;
      record.gradNorm = detail.grad_norm ?? record.gradNorm;
      record.epoch = detail.epoch ?? record.epoch;
      record.elapsedSeconds = detail.elapsed_seconds ?? record.elapsedSeconds;
      record.etaSeconds = detail.eta_seconds ?? record.etaSeconds;
      record.samplesPerSecond = detail.samples_per_second ?? record.samplesPerSecond;
      record.secondsPerStep = detail.seconds_per_step ?? record.secondsPerStep;
      record.gpuMemoryAllocatedMb = detail.gpu_memory_allocated_mb ?? record.gpuMemoryAllocatedMb;
      record.gpuMemoryReservedMb = detail.gpu_memory_reserved_mb ?? record.gpuMemoryReservedMb;
      break;

    case "training-checkpoint":
      record.lastCheckpoint = detail.path;
      record.checkpoints = [
        {
          step: detail.step,
          path: detail.path,
          sizeBytes: detail.size_bytes,
          loss: detail.loss,
          epoch: detail.epoch,
          createdAt: Date.now(),
        },
        ...(record.checkpoints || []).filter((entry) => entry.step !== detail.step),
      ].slice(0, 50);
      break;

    case "training-complete":
      record.status = detail.paused ? "paused" : detail.stopped ? "stopped" : "completed";
      record.progress = 100;
      record.finishedAt = Date.now();
      record.outputDir = detail.output_dir;
      record.finalLoss = detail.final_loss ?? record.finalLoss;
      record.totalSteps = detail.total_steps || record.totalSteps;
      record.lastCheckpoint = detail.last_checkpoint || record.lastCheckpoint;
      record.history = detail.history || record.history;
      record.message = detail.message;
      break;

    case "error":
      record.status = "failed";
      record.error = {
        code: detail.code || "error",
        message: detail.message || "The run failed.",
        hint: detail.hint || "",
      };
      record.finishedAt = Date.now();
      break;

    case "log":
    default:
      break;
  }

  send("zeqou:training:event", { runId: record.id, event, detail });
  send("zeqou:training:state", { runId: record.id, state: record.status, record: slimRun(record) });
}

/* ---------------------------------------------------------------- finalize */

async function finalize(state, outcome) {
  if (state.finishing) return;
  state.finishing = true;

  if (state.gpuTimer) clearInterval(state.gpuTimer);
  const record = state.record;
  record.status = outcome.status;
  if (outcome.error) record.error = outcome.error;
  if (!record.finishedAt) record.finishedAt = Date.now();
  if (record.status === "completed" && record.progress < 100) record.progress = 100;

  try {
    // Persist real artefacts: history, GPU series, checkpoints, final state.
    if (record.history) {
      fs.writeFileSync(
        path.join(state.runDir, "history.json"),
        JSON.stringify(record.history, null, 2),
        "utf8",
      );
    }
    fs.writeFileSync(
      path.join(state.runDir, "gpu-samples.json"),
      JSON.stringify(state.gpuSamples, null, 2),
      "utf8",
    );

    const checkpoints = await python.run(["checkpoints", "--run-dir", state.runDir]);
    if (checkpoints.result && Array.isArray(checkpoints.result.checkpoints)) {
      record.checkpoints = checkpoints.result.checkpoints.map((entry) => ({
        name: entry.name,
        path: entry.path,
        step: entry.step,
        sizeBytes: entry.size_bytes,
        loss: entry.loss,
        epoch: entry.epoch,
        hasOptimizer: entry.has_optimizer,
      }));
      record.runSizeBytes = checkpoints.result.size_bytes;
      if (!record.lastCheckpoint && record.checkpoints.length) {
        record.lastCheckpoint = record.checkpoints[0].path;
      }
    }
  } catch (error) {
    record.warnings = [...(record.warnings || []), `Could not collect run artefacts: ${error.message}`];
  }

  record.gpuSummary = summarizeGpu(state.gpuSamples);
  state.logStream.end();
  state.stderrStream.end();

  upsertRun(record);
  updateProject(record);

  // Anything that produced weights becomes a usable model in the Models view.
  if (record.outputDir && ["completed", "paused", "stopped"].includes(record.status)) {
    try {
      // Required lazily to keep the module graph acyclic at load time.
      const registry = require("./registry");
      const model = registry.registerTrainedModel(record);
      if (model) send("zeqou:models:changed", { model });
    } catch (error) {
      record.warnings = [...(record.warnings || []), `Could not register the trained model: ${error.message}`];
    }
  }

  active.delete(record.id);

  send("zeqou:training:state", { runId: record.id, state: record.status, record: slimRun(record), final: true });
  send("zeqou:training:finished", { runId: record.id, record: slimRun(record) });
  return slimRun(record);
}

function summarizeGpu(samples) {
  const usable = samples.filter((entry) => entry && entry.available && entry.gpu);
  if (!usable.length) return null;

  let peakUtil = 0;
  let peakVram = 0;
  let peakTemp = 0;
  let sumUtil = 0;
  for (const entry of usable) {
    peakUtil = Math.max(peakUtil, entry.gpu.utilization_gpu || 0);
    peakVram = Math.max(peakVram, entry.gpu.memory_used_mb || 0);
    peakTemp = Math.max(peakTemp, entry.gpu.temperature_c || 0);
    sumUtil += entry.gpu.utilization_gpu || 0;
  }
  return {
    samples: usable.length,
    peakUtilization: Math.round(peakUtil),
    averageUtilization: Math.round(sumUtil / usable.length),
    peakVramMb: Math.round(peakVram),
    peakTemperatureC: Math.round(peakTemp),
    device: usable[0].gpu.name,
  };
}

function updateProject(record) {
  const store = projectsStore();
  const list = store.get();
  const project = list.find((entry) => entry.id === record.projectId);
  if (!project) return;

  project.updatedAt = Date.now();
  project.lastStatus = record.status;
  project.lastRunAt = record.finishedAt || Date.now();
  project.runIds = [record.id, ...(project.runIds || []).filter((id) => id !== record.id)].slice(0, 100);
  project.runCount = (project.runCount || 0) + 1;
  if (record.finalLoss != null && (project.bestLoss == null || record.finalLoss < project.bestLoss)) {
    project.bestLoss = record.finalLoss;
  }
  store.replace(list);
}

/* -------------------------------------------------------------- control ops */

function requestStop(runId) {
  const state = active.get(runId);
  if (!state) return { ok: false, error: "This run is not active." };
  try {
    fs.writeFileSync(state.stopFile, String(Date.now()), "utf8");
  } catch (error) {
    return { ok: false, error: error.message };
  }
  state.record.pendingAction = "stopping";
  send("zeqou:training:state", { runId, state: state.record.status, record: slimRun(state.record) });
  return { ok: true, action: "stop" };
}

function requestPause(runId) {
  const state = active.get(runId);
  if (!state) return { ok: false, error: "This run is not active." };
  try {
    fs.writeFileSync(state.pauseFile, String(Date.now()), "utf8");
  } catch (error) {
    return { ok: false, error: error.message };
  }
  state.record.pendingAction = "pausing";
  send("zeqou:training:state", { runId, state: state.record.status, record: slimRun(state.record) });
  return { ok: true, action: "pause" };
}

async function resumeRun(runId) {
  const previous = persistentRun(runId) || (active.get(runId) && active.get(runId).record);
  if (!previous) return { ok: false, error: { code: "not_found", message: "That run was not found." } };

  const checkpoint = previous.lastCheckpoint;
  if (!checkpoint) {
    return {
      ok: false,
      error: {
        code: "no_checkpoint",
        message: "This run has no checkpoint to resume from.",
        hint: "Start a new run instead.",
      },
    };
  }
  if (!fs.existsSync(checkpoint)) {
    return {
      ok: false,
      error: {
        code: "checkpoint_missing",
        message: "The checkpoint this run would resume from no longer exists on disk.",
        hint: "Checkpoints are deleted by the retention limit; start a new run.",
      },
    };
  }

  return startRun({
    projectId: previous.projectId,
    name: `${previous.name}-resume`,
    config: {
      ...previous.config,
      output_name: `${previous.name}-resume`,
    },
    modelInfo: previous.modelInfo || null,
    hardware: previous.hardware || null,
    datasetName: previous.datasetName,
    datasetReport: previous.datasetReport,
    resumeFromCheckpoint: checkpoint,
  });
}

/* ---------------------------------------------------------------- read side */

function listRuns(limit = 200) {
  return runsStore().get().slice(0, limit);
}

function getRun(runId) {
  const state = active.get(runId);
  if (state) return slimRun(state.record);
  return persistentRun(runId);
}

function getLiveSnapshot(runId) {
  const state = active.get(runId);
  if (!state) return null;
  return {
    record: slimRun(state.record),
    gpuSamples: state.gpuSamples.slice(-600),
  };
}

function deleteRun(runId) {
  return waitForFinalize(runId).then(() => {
    const state = active.get(runId);
    // A live process must stay protected: deleting it mid-flight would orphan
    // the trainer. Anything finalising has been waited out above.
    if (state && TERMINAL_STATUSES.has(state.record.status) === false) {
      return { ok: false, error: "Stop the run before deleting it." };
    }
    const store = runsStore();
    const target = persistentRun(runId);
    store.replace(store.get().filter((run) => run.id !== runId));
    active.delete(runId);
    if (target && target.runDir && isInside(dirs.runs(), target.runDir)) {
      try {
        fs.rmSync(target.runDir, { recursive: true, force: true });
      } catch (error) {
        return { ok: false, error: error.message, recordRemoved: true };
      }
    }
    return { ok: true };
  });
}

function readLog(runId, tailLines = 800) {
  const record = persistentRun(runId) || (active.get(runId) && active.get(runId).record);
  // Same shape as the success path: callers can always read `.events`.
  if (!record || !record.runDir) return { events: [], stderr: [] };

  const readTail = (fileName, limit) => {
    const filePath = path.join(record.runDir, fileName);
    if (!fs.existsSync(filePath)) return [];
    const content = fs.readFileSync(filePath, "utf8");
    return content.split(/\r?\n/).filter(Boolean).slice(-limit);
  };

  const raw = readTail("training.log", tailLines);
  const events = [];
  for (const line of raw) {
    try {
      events.push(JSON.parse(line));
    } catch {
      events.push({ event: "log", detail: { message: line, level: "info" } });
    }
  }
  return { events, stderr: readTail("stderr.log", Math.floor(tailLines / 2)) };
}

async function listCheckpoints(runDir) {
  const response = await python.run(["checkpoints", "--run-dir", runDir]);
  if (!response.ok) {
    return { ok: false, error: response.error, checkpoints: [] };
  }
  return { ok: true, ...response.result };
}

function stopAll() {
  for (const [runId, state] of active.entries()) {
    try {
      fs.writeFileSync(state.stopFile, String(Date.now()), "utf8");
    } catch {
      /* ignore */
    }
    // Give the trainer a moment to checkpoint, then make sure it is gone.
    setTimeout(() => {
      if (state.handle) state.handle.kill();
    }, 2500).unref?.();
    active.delete(runId);
  }
}

module.exports = {
  attach,
  startRun,
  requestStop,
  requestPause,
  resumeRun,
  listRuns,
  getRun,
  getLiveSnapshot,
  deleteRun,
  readLog,
  listCheckpoints,
  stopAll,
  files,
};
